"""Verify rollback at every possible accepted prefix, including zero drafts."""
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nemotron_h import NemotronHBlock
from mlx_lm.models.cache import ArraysCache, KVCache
from cactus.mlx.models.nemotron_verify import mamba_verify, attention_verify, VerifyGraph


def test_recurrent_prefixes_match_sequential_updates():
    mx.random.seed(7)
    args = SimpleNamespace(hidden_size=64, mamba_num_heads=4, mamba_head_dim=8,
        ssm_state_size=32, conv_kernel=4, n_groups=2, time_step_limit=(0.,100.),
        use_conv_bias=True, mamba_proj_bias=False, layer_norm_epsilon=1e-5)
    layer = NemotronHBlock(args, 'M')
    mixer = layer.mixer
    # Isolate the recurrent scan from shape-dependent GEMM rounding.
    class Projection(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = width
        def __call__(self, x):
            return mx.take(x, mx.arange(self.width) % x.shape[-1], axis=-1) * 0.25
    mixer.in_proj = Projection(mixer.intermediate_size + mixer.conv_dim + mixer.num_heads)
    mixer.out_proj = Projection(args.hidden_size)
    x = mx.random.normal((1,4,64))
    conv = mx.random.normal((1,3,mixer.conv_dim))
    state = mx.random.normal((1,4,8,32))
    actual, history = mamba_verify(layer, x, conv, state)
    compact, final = mamba_verify(layer, x, conv, state, snapshots=False)
    np.testing.assert_array_equal(compact, actual)
    np.testing.assert_array_equal(final[0], history[0][-1])
    np.testing.assert_array_equal(final[1], history[1][-1])
    cache = ArraysCache(2)
    cache[0], cache[1] = conv, state
    expected = []
    for i in range(4):
        expected.append(layer(x[:,i:i+1],cache=cache))
        np.testing.assert_allclose(history[0][i],cache[0],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(history[1][i],cache[1],rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(actual,mx.concatenate(expected,1),rtol=2e-5,atol=2e-5)


def test_commit_restores_each_accepted_prefix():
    conv = mx.arange(12).reshape(3,1,2,2)
    recurrent = mx.arange(24).reshape(3,1,2,2,2)
    for keep in (1,2,3):
        state = ArraysCache(2)
        kv = KVCache()
        kv.update_and_fetch(mx.ones((1,1,10,2)),mx.ones((1,1,10,2)))
        VerifyGraph.commit([state,kv],{0:(conv,recurrent)},keep,3)
        assert kv.offset == 7 + keep
        np.testing.assert_array_equal(state[0],conv[keep-1])
        np.testing.assert_array_equal(state[1],recurrent[keep-1])


def test_generator_closes_with_only_emitted_tokens_committed(monkeypatch):
    from cactus.mlx.generation import mtp_generate_step
    from cactus.mlx.models import nemotron_speculative
    closed = []
    class Decoder:
        def __init__(self, *args, **kwargs):
            self.stats = {}
        def prefill(self, *args):
            return 3, mx.zeros(8)
        def step(self):
            return [4,5,6], mx.zeros((3,8))
        def finish_prefix(self, n):
            closed.append(n)
    monkeypatch.setattr(nemotron_speculative, 'MTPDecoder', Decoder)
    model = SimpleNamespace()
    tokens = mtp_generate_step([1,2],model,max_tokens=9)
    assert next(tokens)[0] == 3
    assert next(tokens)[0] == 4
    tokens.close()
    assert closed == [1]
    closed.clear()
    assert [t[0] for t in mtp_generate_step([1,2],model,max_tokens=3)] == [3,4,5]
    assert closed == [2]


def test_sampling_uses_native_generator(monkeypatch):
    from cactus.mlx import generation
    calls = []
    def native(*args, **kwargs):
        calls.append(kwargs)
        yield 'native'
    monkeypatch.setattr(generation, 'native_stream_generate', native)
    sample = lambda logits: mx.random.categorical(logits)
    result = list(generation.stream_generate(None,None,[1],mtp=True,sampler=sample))
    assert result == ['native']
    assert calls[0]['sampler'] is sample


def test_verification_attention_uses_native_causal_prefixes():
    from mlx_lm.models.base import scaled_dot_product_attention
    mx.random.seed(19)
    class Projection(nn.Module):
        def __call__(self, x):
            return x * 0.25
    mixer = SimpleNamespace(q_proj=Projection(), k_proj=Projection(),
        v_proj=Projection(), o_proj=Projection(), num_heads=4,
        num_key_value_heads=4, scale=8**-0.5)
    layer = SimpleNamespace(mixer=mixer, norm=nn.RMSNorm(32))
    x = mx.random.normal((1,7,32)).astype(mx.bfloat16)
    native, verified = KVCache(), KVCache()
    prefix = mx.random.normal((1,4,13,8)).astype(mx.bfloat16)
    native.update_and_fetch(prefix,prefix)
    verified.update_and_fetch(prefix,prefix)
    expected = []
    for i in range(7):
        h = layer.norm(x[:,i:i+1])
        q = mixer.q_proj(h).reshape(1,1,4,8).transpose(0,2,1,3)
        k,v = native.update_and_fetch(q,q)
        y = scaled_dot_product_attention(q,k,v,cache=native,scale=mixer.scale,mask=None)
        expected.append(x[:,i:i+1]+mixer.o_proj(y.transpose(0,2,1,3).reshape(1,1,32)))
    actual = attention_verify(layer,x,verified)
    np.testing.assert_array_equal(actual,mx.concatenate(expected,axis=1))
    assert verified.offset == native.offset == 20


def test_external_cache_includes_last_emitted_token(monkeypatch):
    from cactus.mlx.generation import mtp_generate_step
    from cactus.mlx.models import nemotron_speculative
    commits, forwarded = [], []
    class Decoder:
        def __init__(self,*args,**kwargs): self.stats = {}
        def prefill(self,*args): return 3,mx.zeros(8)
        def step(self): return [4,5,6],mx.zeros((3,8))
        def finish_prefix(self,n): commits.append(n)
    monkeypatch.setattr(nemotron_speculative,'MTPDecoder',Decoder)
    model = SimpleNamespace(forward_hidden=lambda tokens,cache:forwarded.append(tokens.tolist()))
    cache = [SimpleNamespace(state=[])]
    iterator = mtp_generate_step([1,2],model,max_tokens=8,cache=cache)
    next(iterator)
    next(iterator)
    iterator.close()
    assert commits == [1]
    assert forwarded == [[[4]]]


def test_reused_prompt_cache_uses_native_generator(monkeypatch):
    from cactus.mlx import generation
    cache = ArraysCache(2)
    cache[0] = mx.zeros((1,3,4))
    monkeypatch.setattr(generation,'native_stream_generate',lambda *a,**kw:iter(['native']))
    assert list(generation.stream_generate(None,None,[1],mtp=True,prompt_cache=[cache])) == ['native']


def test_partial_block_restores_seed_hidden_and_both_caches():
    from cactus.mlx.models.nemotron_speculative import MTPDecoder
    decoder = MTPDecoder.__new__(MTPDecoder)
    state, kv, draft = ArraysCache(2), KVCache(), KVCache()
    kv.update_and_fetch(mx.ones((1,1,13,2)),mx.ones((1,1,13,2)))
    draft.update_and_fetch(mx.ones((1,1,7,2)),mx.ones((1,1,7,2)))
    history = {0:(mx.arange(12).reshape(3,1,2,2),mx.arange(24).reshape(3,1,2,2,2))}
    hidden, logits = mx.arange(12).reshape(1,3,4), mx.arange(24).reshape(3,8)
    decoder.verify, decoder.cache, decoder.draft_cache = VerifyGraph,[state,kv],draft
    decoder._pending = history,3,4,hidden,[4,5,6],logits
    decoder.finish_prefix(1)
    assert decoder.seed == 4 and kv.offset == 11 and draft.offset == 5
    np.testing.assert_array_equal(decoder.hidden,hidden[:,:1])
    np.testing.assert_array_equal(decoder.logits,logits[0])
    np.testing.assert_array_equal(state[1],history[0][1][0])


def test_compact_verification_replays_from_original_cache():
    from cactus.mlx.models.nemotron_verify import VerificationState
    state, kv = ArraysCache(2), KVCache()
    state[0], state[1] = mx.array([2.]), mx.array([3.])
    kv.update_and_fetch(mx.ones((1,1,5,2)), mx.ones((1,1,5,2)))
    cache = [state, kv]
    calls = []
    def graph(tokens, cache):
        calls.append((tokens.tolist(), cache[1].offset, cache[0][1].item()))
        result = VerificationState(graph, tokens, cache)
        n = tokens.shape[1]
        cache[1].update_and_fetch(mx.ones((1,1,n,2)), mx.ones((1,1,n,2)))
        result[0] = (cache[0][0] + n, cache[0][1] + mx.sum(tokens))
        return None, None, result
    _, _, history = graph(mx.array([[7,11,13]]), cache)
    VerifyGraph.commit(cache, history, 3, 3)
    assert len(calls) == 1 and kv.offset == 8 and state[1].item() == 34
    # Closing a stream inside an already accepted block restores its prefix.
    VerifyGraph.commit(cache, history, 2, 3)
    assert calls[-1] == ([[7,11]],5,3.)
    assert kv.offset == 7 and state[1].item() == 21
    VerifyGraph.commit(cache, history, 1, 2)
    assert calls[-1] == ([[7]],5,3.)
    assert kv.offset == 6 and state[1].item() == 10


def test_mtp_cache_projection_matches_full_attention_path():
    from cactus.mlx.models.nemotron_mtp import MTPHead
    args = SimpleNamespace(hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, attention_bias=False, num_experts_per_tok=2, moe_latent_size=16,
        moe_intermediate_size=32, n_routed_experts=4, n_shared_experts=None,
        norm_topk_prob=True, routed_scaling_factor=1., n_group=1, topk_group=1,
        mlp_bias=False, layer_norm_epsilon=1e-5)
    mx.random.seed(32)
    head = MTPHead(args)
    for dtype in (mx.float32, mx.bfloat16):
        head.set_dtype(dtype)
        for n in (1,2,3,7):
            e, h = [mx.random.normal((1,n,32)).astype(dtype) for _ in range(2)]
            full, projected = KVCache(), KVCache()
            prefix = mx.random.normal((1,2,5,8)).astype(dtype)
            full.update_and_fetch(prefix, prefix)
            projected.update_and_fetch(prefix, prefix)
            head._attention(e, h, full)
            head.cache_inputs(e, h, projected)
            assert full.offset == projected.offset == 5 + n
            for expected, actual in zip(full.state, projected.state):
                if isinstance(expected, mx.array):
                    np.testing.assert_array_equal(actual.astype(mx.float32), expected.astype(mx.float32))
                else:
                    assert actual == expected
