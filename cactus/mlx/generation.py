"""MLX-LM-compatible text streaming with optional verified greedy MTP."""
import time
import mlx.core as mx
from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
from mlx_lm.generate import stream_generate as native_stream_generate
from mlx_lm.sample_utils import greedy_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.models.cache import ArraysCache


def mtp_generate_step(prompt, model, *, max_tokens, draft_tokens=3, min_draft_prob=0.8,
                      cache=None, prefill_step_size=4, progress=None, input_embeddings=None):
    from .models.nemotron_speculative import MTPDecoder
    decoder = MTPDecoder(model, cache=cache, draft_tokens=draft_tokens, min_draft_prob=min_draft_prob)
    model._last_mtp_stats = decoder.stats
    last = None
    try:
        if input_embeddings is None:
            token, logits = decoder.prefill(prompt, prefill_step_size, progress)
        else:
            token, logits = decoder.prefill(prompt, prefill_step_size, progress, input_embeddings=input_embeddings)
        last = token
        logits = logits.astype(mx.float32)
        yield token, logits - mx.logsumexp(logits), False
        count = 1
        while max_tokens < 0 or count < max_tokens:
            block, logits = decoder.step()
            logits = logits.astype(mx.float32)
            emitted = 0
            try:
                for i, token in enumerate(block):
                    emitted += 1
                    count += 1
                    last = token
                    yield token, logits[i] - mx.logsumexp(logits[i]), i < len(block) - 1
                    if max_tokens >= 0 and count >= max_tokens:
                        return
            finally:
                decoder.finish_prefix(emitted)
    finally:
        if cache is not None and last is not None:
            # MLX-LM's server caches every emitted token. Include the final one,
            # which ordinary speculative stepping leaves as the next input.
            model.forward_hidden(mx.array([[last]]), cache)
            mx.eval([c.state for c in cache])


def stream_generate(model, tokenizer, prompt, max_tokens=256, *, mtp=True,
                    num_draft_tokens=3, min_draft_prob=0.8, stream=generation_stream, images=None, **kwargs):
    if images is not None:
        if not isinstance(prompt, str) or kwargs.get('input_embeddings') is not None:
            raise ValueError('Images require a string prompt with image placeholders and no precomputed embeddings')
        prompt, pixels = model.image_processor(prompt, images=images)
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
        kwargs['input_embeddings'] = model.embed_inputs(mx.array(prompt), pixel_values=pixels)
        mx.eval(kwargs['input_embeddings'])
    # Native sampling and logit processors retain the stock MLX-LM behavior.
    unsupported = any(kwargs.get(k) is not None for k in
                      ('draft_model', 'kv_bits', 'max_kv_size'))
    reused_cache = any(isinstance(c, ArraysCache) and c[0] is not None
                       for c in (kwargs.get('prompt_cache') or []))
    known = {'sampler', 'logits_processors', 'prompt_cache', 'prefill_step_size',
             'prompt_progress_callback', 'draft_model', 'input_embeddings', 'kv_bits',
             'max_kv_size', 'kv_group_size', 'quantized_kv_start'}
    if (not mtp or unsupported or reused_cache or kwargs.keys() - known or kwargs.get('logits_processors')
            or kwargs.get('sampler') not in (None, greedy_sampler)):
        yield from native_stream_generate(model, tokenizer, prompt, max_tokens=max_tokens,
                                          num_draft_tokens=num_draft_tokens, stream=stream, **kwargs)
        return
    if not hasattr(model, 'load_mtp'):
        raise ValueError('This model adapter does not implement MTP')
    if max_tokens == 0:
        raise ValueError('max_tokens must be positive, or -1 for no limit')
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if isinstance(prompt, str):
        add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        prompt = tokenizer.encode(prompt, add_special_tokens=add_special)
    elif isinstance(prompt, mx.array):
        prompt = prompt.tolist()
    prompt = list(prompt)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tokens = mtp_generate_step(prompt, model, max_tokens=max_tokens,
        draft_tokens=num_draft_tokens, min_draft_prob=min_draft_prob, cache=kwargs.get('prompt_cache'),
        prefill_step_size=kwargs.get('prefill_step_size', 4),
        progress=kwargs.get('prompt_progress_callback'), input_embeddings=kwargs.get('input_embeddings'))
    try:
        with mx.stream(stream), wired_limit(model, [stream]):
            started = time.perf_counter()
            count = 0
            for token, logprobs, from_draft in tokens:
                if count == 0:
                    prompt_tps = len(prompt) / (time.perf_counter() - started)
                    decode_start = time.perf_counter()
                count += 1
                eos = token in tokenizer.eos_token_ids
                if not eos:
                    detokenizer.add_token(token)
                done = eos or (max_tokens > 0 and count >= max_tokens)
                if done:
                    detokenizer.finalize()
                yield GenerationResponse(text=detokenizer.last_segment, token=token,
                    logprobs=logprobs, from_draft=from_draft, prompt_tokens=len(prompt),
                    prompt_tps=prompt_tps, generation_tokens=count,
                    generation_tps=count / max(time.perf_counter() - decode_start, 1e-9),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=('stop' if eos else 'length') if done else None)
                if done:
                    return
    finally:
        with mx.stream(stream):
            tokens.close()


def generate(model, tokenizer, prompt, **kwargs):
    return ''.join(response.text for response in stream_generate(model, tokenizer, prompt, **kwargs))
