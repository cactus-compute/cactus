"""Greedy MTP proposals with target verification and recoverable Mamba state."""
import time
import mlx.core as mx
from .nemotron_verify import VerifyGraph


class MTPDecoder:
    def __init__(self, model, cache=None, draft_tokens=3, min_draft_prob=0.):
        if not 1 <= draft_tokens <= 7:
            raise ValueError('MTP draft length must be between 1 and 7')
        if not 0 <= min_draft_prob <= 1:
            raise ValueError('MTP confidence must be between 0 and 1')
        self.model = model
        self.head = model.load_mtp()
        if not hasattr(model, '_verify_graph'):
            model._verify_graph = VerifyGraph(model)
        self.verify = model._verify_graph
        self.cache = cache if cache is not None else model.make_cache()
        self.draft_cache = self.head.make_cache()
        self.draft_tokens = draft_tokens
        self._draft_budget = min(2, draft_tokens)
        self._accepted_blocks = 0
        self.min_draft_prob = min_draft_prob
        self.stats = dict(proposed=0, accepted=0, blocks=0, draft_s=0., verify_s=0., repair_s=0.)
        self.stats['skipped'] = 0
        self._pending = None

    def prefill(self, tokens, step=4, progress=None, input_embeddings=None):
        if not len(tokens) or step < 1:
            raise ValueError('MTP requires a nonempty prompt and positive prefill step')
        previous = None
        if input_embeddings is not None and input_embeddings.shape != (len(tokens), self.model.args.hidden_size):
            raise ValueError('Input embeddings must match prompt length and model width')
        for start in range(0, len(tokens), step):
            chunk = tokens[start:start + step]
            embeds = None if input_embeddings is None else input_embeddings[start:start + step][None]
            if embeds is None:
                logits, hidden = self.model.forward_hidden(mx.array([chunk]), self.cache)
            else:
                logits, hidden = self.model.forward_hidden(mx.array([chunk]), self.cache, input_embeddings=embeds)
            if previous is None:
                shifted_tokens, shifted_hidden = chunk[1:], hidden[:, :-1]
            else:
                shifted_tokens = chunk
                shifted_hidden = mx.concatenate([previous, hidden[:, :-1]], axis=1)
            if shifted_tokens:
                shifted_embeds = (self.model.backbone.embeddings(mx.array([shifted_tokens])) if embeds is None
                                  else embeds[:, 1:] if previous is None else embeds)
                self.head.cache_inputs(shifted_embeds, shifted_hidden, self.draft_cache)
            mx.eval(logits, hidden, [c.state for c in self.cache], self.draft_cache.state)
            previous = hidden[:, -1:]
            if progress:
                progress(min(start + step, len(tokens)), len(tokens))
        self.hidden = previous
        self.logits = logits[0, -1]
        self.seed = int(mx.argmax(self.logits).item())
        return self.seed, self.logits

    def step(self):
        self._pending = None
        started = time.perf_counter()
        offset = self.draft_cache.offset
        seed = self.seed
        hidden = self.hidden
        proposals = []
        token = seed
        for _ in range(self._draft_budget):
            hidden, choice = self.head.propose(self.model.backbone.embeddings(mx.array([[token]])),
                                               hidden, self.draft_cache, self.model.lm_head)
            candidate, probability = choice.tolist()
            if probability < self.min_draft_prob:
                break
            token = int(candidate)
            proposals.append(token)
        drafted = time.perf_counter()
        if not proposals:
            self._accepted_blocks = 0
            logits, self.hidden = self.model.forward_hidden(mx.array([[seed]]), self.cache)
            self.logits = logits[0, -1]
            self.seed = int(mx.argmax(self.logits).item())
            mx.eval(self.draft_cache.state)
            self.stats['skipped'] += 1
            self.stats['draft_s'] += drafted - started
            self.stats['verify_s'] += time.perf_counter() - drafted
            return [self.seed], self.logits[None]
        inputs = [seed, *proposals]
        logits, hidden, history = self.verify(mx.array([inputs]), self.cache)
        predicted = mx.argmax(logits[0], axis=-1).tolist()
        accepted = 0
        while accepted < len(proposals) and proposals[accepted] == predicted[accepted]:
            accepted += 1
        keep = accepted + 1
        self.verify.commit(self.cache, history, keep, len(inputs))
        verified = time.perf_counter()
        # The first proposal used the verified target state already. Retain that
        # KV entry and repair only subsequent positions with verified states.
        self.draft_cache.trim(self.draft_cache.offset - offset - 1)
        if accepted:
            self.head.cache_inputs(self.model.backbone.embeddings(mx.array([inputs[1:keep]])),
                                   hidden[:, :accepted], self.draft_cache)
        mx.eval([c.state for c in self.cache], self.draft_cache.state)
        self.hidden = hidden[:, accepted:accepted + 1]
        self.seed = predicted[accepted]
        self.logits = logits[0, accepted]
        self._pending = (history, keep, offset, hidden, predicted[:keep], logits[0, :keep])
        if accepted < len(proposals):
            self._draft_budget = 1
            self._accepted_blocks = 0
        elif len(proposals) == self._draft_budget:
            self._accepted_blocks += 1
            if self._accepted_blocks >= 4:
                self._draft_budget = min(self.draft_tokens, self._draft_budget + 1)
                self._accepted_blocks = 0
        else:
            self._accepted_blocks = 0
        self.stats['proposed'] += len(proposals)
        self.stats['accepted'] += accepted
        self.stats['blocks'] += 1
        self.stats['draft_s'] += drafted - started
        self.stats['verify_s'] += verified - drafted
        self.stats['repair_s'] += time.perf_counter() - verified
        return predicted[:keep], logits[0, :keep]

    def finish_prefix(self, emitted):
        """Leave the target cache one token behind the last emitted token."""
        if self._pending is not None:
            history, keep, offset, hidden, predicted, logits = self._pending
            if not 1 <= emitted <= keep:
                raise ValueError('Invalid emitted prefix')
            if emitted != keep:
                self.verify.commit(self.cache, history, emitted, total=keep)
                self.draft_cache.trim(self.draft_cache.offset - offset - emitted)
                self.hidden = hidden[:, emitted - 1:emitted]
                self.seed = predicted[emitted - 1]
                self.logits = logits[emitted - 1]
                mx.eval([c.state for c in self.cache])
            self._pending = None
