"""The shared attention/MoE MTP head packed alongside the target checkpoint."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nemotron_h import NemotronHBlock
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from ..checkpoint import Checkpoint, books
from ..ops import CQLinear, CQExperts
from .nemotron import locate


class MTPHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = [NemotronHBlock(args, '*'), NemotronHBlock(args, 'E')]
        self.layers[0].enorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.layers[0].hnorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.layers[0].eh_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.layers[1].final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self._prepare = None
        self._proposal = None
        self._cache_projection = None

    def cache_inputs(self, embeddings, previous_hidden, cache):
        """Update draft keys/values without constructing unused attention or MoE outputs."""
        if self._cache_projection is None:
            def project(e, h):
                layer = self.layers[0]
                x = layer.eh_proj(mx.concatenate([layer.enorm(e), layer.hnorm(h)], axis=-1))
                x = layer.norm(x)
                return layer.mixer.k_proj(x), layer.mixer.v_proj(x)
            self._cache_projection = mx.compile(project)
        k, v = self._cache_projection(embeddings, previous_hidden)
        batch, length, _ = k.shape
        heads = self.layers[0].mixer.num_key_value_heads
        cache.update_and_fetch(k.reshape(batch, length, heads, -1).transpose(0, 2, 1, 3),
                               v.reshape(batch, length, heads, -1).transpose(0, 2, 1, 3))

    def _attention(self, embeddings, previous_hidden, cache):
        if self._prepare is None:
            self._prepare = mx.compile(lambda e, h: self.layers[0].eh_proj(mx.concatenate([
                self.layers[0].enorm(e), self.layers[0].hnorm(h)], axis=-1)))
        x = self._prepare(embeddings, previous_hidden)
        return self.layers[0](x, mask=create_attention_mask(x, cache), cache=cache)

    def propose(self, embeddings, previous_hidden, cache, lm_head):
        if self._proposal is None:
            def finish(x):
                hidden = self.layers[1].final_layernorm(self.layers[1](x))
                logits = lm_head(hidden)[0, -1].astype(mx.float32)
                token = mx.argmax(logits)
                probability = mx.exp(logits[token] - mx.logsumexp(logits))
                return hidden, mx.stack([token.astype(mx.float32), probability])
            self._proposal = mx.compile(finish)
        return self._proposal(self._attention(embeddings, previous_hidden, cache))

    def make_cache(self):
        return KVCache()


def load_mtp(model, path):
    cp = Checkpoint(path, 'mtp')
    try:
        head = MTPHead(model.args)
        bs = books(cp)
        experts = {}
        for name, leaf in cp.leaves.items():
            op = CQLinear(cp, name, bs[leaf['rate']])
            path = name.removesuffix('.weight')
            if '.experts.' in path:
                prefix, which = path.rsplit('.experts.', 1)
                experts.setdefault(prefix, {})[which] = op
            elif path.endswith('.gate'):
                target, key = locate(head, path + '.weight')
                setattr(target, key, op.gather(mx.arange(op._n)))
            else:
                target, key = locate(head, path)
                setattr(target, key, op)
        for prefix, pair in experts.items():
            target, key = locate(head, prefix + '.switch_mlp')
            setattr(target, key, CQExperts(pair['up'], pair['down'], model.args.n_routed_experts))
        for name in cp.manifest['rest']:
            target, key = locate(head, name.removeprefix('language_model.mtp.'))
            setattr(target, key, cp.array(name))
        head.eval()
        mx.eval(head.parameters())
        return head
    finally:
        cp.close()
