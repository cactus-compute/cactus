import json
import os
import re
import unicodedata
from dataclasses import replace

import numpy as np

from .checkpoints import is_safetensors, read_adapter, write_adapter
from .finetune import _training_rng, init_lora, lora_target_paths, merge_lora
from .whistle import LANGUAGES, Whistle, load_whistle

DEFAULT_BASE = "checkpoints/whistle.safetensors"
SAMPLE_RATE = 16000
SECONDS = 30
SPACE = "▁"


class WhistleTokenizer:
    def __init__(self, pieces=(), languages=LANGUAGES):
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2
        from .tokenizer import get_tokenizer

        self.base = get_tokenizer().sp
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(self.base.serialized_model_proto())
        low = min(p.score for p in proto.pieces)
        for k, piece in enumerate(pieces):
            proto.pieces.add(piece=piece, score=low - 1 - k)
        self.proto, self.pieces, self.languages = proto, tuple(pieces), tuple(languages)
        self.sp = spm.SentencePieceProcessor(model_proto=proto.SerializeToString())

    @property
    def text(self):
        return len(self.proto.pieces)

    def encode(self, text):
        return self.sp.encode(text)

    def blob(self):
        from .export import whistle_tokenizer_blob
        return whistle_tokenizer_blob(self.proto, self.languages)


def _symbols(sp, text):
    out, raw = [], bytearray()
    for piece in sp.encode(text, out_type=str) + [""]:
        if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
            raw.append(int(piece[3:5], 16))
            continue
        if raw:
            out += list(raw.decode("utf-8", "ignore"))
            raw.clear()
        if piece:
            out.append(piece)
    return out


def read_tokens(path):
    if not os.path.isfile(path):
        raise SystemExit(f"--tokens takes a file with one token per line; {path} is not a file")
    with open(path, encoding="utf-8-sig") as handle:
        tokens = [line.strip() for line in handle if line.strip()]
    for token in tokens:
        if re.fullmatch(r"<\|.*\|>", token):
            raise SystemExit(f"{path}: {token} looks like a language tag; a new language gets its tag from the clips' "
                             "\"language\" code, so leave it out of --tokens")
    return tokens


def token_pieces(sp, tokens):
    known = {sp.id_to_piece(i) for i in range(sp.get_piece_size())}
    pieces = []
    for token in tokens:
        for text in (" " + token, token):
            symbols = _symbols(sp, text)
            for end in range(1, len(symbols) + 1):
                piece = "".join(symbols[:end])
                if piece not in known:
                    known.add(piece)
                    pieces.append(piece)
    return pieces


def read_audio(path):
    import wave
    from ..agent.whistle import _read_wav
    try:
        return np.asarray(_read_wav(path), np.float32)
    except wave.Error:
        try:
            import soundfile
        except ImportError:
            raise RuntimeError(f"{path} is not a PCM or float WAV file; reading it needs soundfile: "
                               "pip install soundfile") from None
        data, rate = soundfile.read(path, dtype="float32", always_2d=True)
        samples = data.mean(1)
    if rate != SAMPLE_RATE:
        try:
            import soxr
        except ImportError:
            raise RuntimeError(f"{path} is {rate} Hz; resampling it to {SAMPLE_RATE} Hz needs soxr: "
                               'pip install "cactus-compute[train]"') from None
        samples = soxr.resample(samples, rate, SAMPLE_RATE, quality="HQ")
    return samples.astype(np.float32)


def read_clips(path, language=None):
    root = os.path.dirname(os.path.abspath(path))
    clips, skipped = [], 0
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: not JSON ({error})") from None
            lang = row.get("language") or language
            if not lang or not re.fullmatch(r"[A-Za-z0-9_-]+", lang):
                raise SystemExit(f"{path}:{number}: give the line a \"language\" code such as \"pt\", "
                                 "or pass --language")
            if not row.get("audio"):
                raise SystemExit(f"{path}:{number}: no \"audio\" path")
            audio = os.path.join(root, row["audio"])
            if not os.path.isfile(audio):
                raise SystemExit(f"{path}:{number}: {audio} does not exist")
            samples, text = read_audio(audio), (row.get("text") or "").strip()
            if not text or not len(samples) or len(samples) > SECONDS * SAMPLE_RATE:
                skipped += 1
                continue
            clips.append((samples, text, lang))
    return clips, skipped


def load_base(path):
    if not os.path.exists(path):
        from ..agent import fetch
        print(f"  {'fetch':<9} {path}  downloading from Hugging Face", flush=True)
        path = fetch.fetch_checkpoint(os.path.basename(path), os.path.dirname(path) or fetch.CHECKPOINT_PREFIX,
                                      generation=fetch.WHISTLE)
    params, cfg = load_whistle(path)
    if "encoder" not in params or "stem" not in params:
        raise SystemExit(f"{path} is not a Whistle checkpoint")
    return params, cfg


def initial_rows(embedding, cfg, tokenizer):
    embedding = np.asarray(embedding)
    rows = [embedding[tokenizer.base.encode(piece.replace(SPACE, " "))].mean(0) for piece in tokenizer.pieces]
    tag = embedding[cfg.text:].mean(0)
    rows += [tag] * (len(tokenizer.languages) - len(cfg.languages))
    return np.asarray(rows, np.float32).reshape(-1, embedding.shape[1])


def tuned_params(params, cfg, tokenizer, lora, scale, rows):
    import jax.numpy as jnp
    params = merge_lora(params, lora, scale) if lora else dict(params)
    embedding = jnp.asarray(params["embedding"]["embedding"])
    if rows is not None and len(rows):
        rows, pieces = jnp.asarray(rows), len(tokenizer.pieces)
        embedding = jnp.concatenate([embedding[:cfg.text], rows[:pieces], embedding[cfg.text:], rows[pieces:]])
    return {**params, "embedding": {"embedding": embedding}}


def _arrays(items, idx, cfg):
    wav = np.zeros((len(idx), SECONDS * SAMPLE_RATE), np.float32)
    lengths = np.zeros(len(idx), np.int32)
    tokens = np.full((len(idx), cfg.max_tokens + 1), cfg.pad, np.int32)
    for row, i in enumerate(idx):
        samples, ids = items[i][:2]
        wav[row, :len(samples)], lengths[row], tokens[row, :len(ids)] = samples, len(samples), ids
    return wav, lengths, tokens


def _normalizer(language):
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer
    return EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()


def by_character(text):
    return any(unicodedata.name(c, "").startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI")) for c in text)


def errors(reference, hypothesis, language):
    normalize = _normalizer(language)
    ref, hyp = normalize(reference), normalize(hypothesis)
    if by_character(reference):
        ref, hyp = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    else:
        ref, hyp = ref.split(), hyp.split()
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1], len(ref)


def _score_quantised(model, params, cfg, tokenizer, lora, scale, rows, clips, batch):
    import jax
    from .quantize import cq_ste_params, WEIGHT_BITS

    merged = cq_ste_params(tuned_params(params, cfg, tokenizer, lora, scale, rows), WEIGHT_BITS)
    tuned = model.cfg
    encode = jax.jit(lambda p, wav, lengths: model.apply({"params": p}, wav, lengths, True, method=Whistle.encode))
    step = jax.jit(lambda p, tokens, enc, frames: model.apply({"params": p}, tokens, enc, frames, True,
                                                              method=Whistle.logits))
    tags = np.arange(tuned.vocab) >= tuned.text
    edits = units = 0
    for start in range(0, len(clips), batch):
        part = clips[start:start + batch]
        enc, frames = encode(merged, *_arrays(part, range(len(part)), tuned)[:2])
        tokens = np.full((len(part), tuned.max_tokens), tuned.pad, np.int32)
        tokens[:, 0] = tuned.bos
        done = np.zeros(len(part), bool)
        for pos in range(tuned.max_tokens - 1):
            size = min(tuned.max_tokens, (pos // 64 + 1) * 64)
            logits = np.array(step(merged, tokens[:, :size], enc, frames)[:, pos])
            logits[:, tags != (pos == 0)] = -np.inf
            pick = logits.argmax(-1)
            tokens[:, pos + 1] = np.where(done, tuned.pad, pick)
            done |= pick == tuned.eos
            if done.all():
                break
        for (_, wanted, text), ids in zip(part, tokens.tolist()):
            ids = ids[2:ids.index(tuned.eos)] if tuned.eos in ids else ids[2:]
            language = tuned.languages[wanted[1] - tuned.text]
            e, n = errors(text, tokenizer.sp.decode([i for i in ids if i != tuned.pad]), language)
            edits, units = edits + e, units + n
    return edits, units


def rung(params, cfg, layers):
    from .architecture import ladder_config, ladder_slice

    if not layers or layers == cfg.dec_layers:
        return params, cfg
    if not 2 <= layers < cfg.dec_layers:
        raise ValueError(f"--layers must be in [2, {cfg.dec_layers}], got {layers}")
    config = cfg.cactus()
    sites = ladder_config(config, layers).engram_layers
    return ladder_slice(params, config, layers), replace(cfg, dec_layers=layers, sites=sites)


def build(params, cfg, tokenizer, out):
    from .export import write_export
    from .quantize import WEIGHT_BITS
    cfg = replace(cfg, text=tokenizer.text, languages=tokenizer.languages)
    return write_export(params, cfg.cactus(), out, bits=WEIGHT_BITS, tokenizer=tokenizer.blob(), audio=cfg)


def finetune_main(args):
    import jax
    import jax.numpy as jnp
    import optax
    from .quantize import WEIGHT_BITS, configure_deploy, cq_ste_params

    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be at least 1")
    out = args.out or os.path.join(args.checkpoint_dir, "whistle_lora.safetensors")
    if not is_safetensors(out):
        raise SystemExit(f"--out must be a .safetensors file, got {out}")
    base_path = args.checkpoint or DEFAULT_BASE
    params, cfg = load_base(base_path)
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    backend = jax.default_backend().lower()
    if backend == "metal":
        cfg = replace(cfg, flash=False, remat=False, scan_unroll=max(cfg.enc_layers, cfg.dec_layers))
    params = jax.device_put(params)
    print(f"  {'backend':<9} {backend}  float32")

    splits, skipped = [], 0
    for path in (args.train, args.validation, args.test):
        clips, dropped = read_clips(path, args.language)
        if not clips:
            raise SystemExit("no usable clips in " + path)
        splits.append(clips)
        skipped += dropped
    codes = {lang for *_, lang in splits[0]}
    languages = tuple(cfg.languages) + tuple(sorted(codes - set(cfg.languages)))
    untrained = {lang for clips in splits[1:] for *_, lang in clips} - set(languages)
    if untrained:
        raise SystemExit(f"no training clips for {', '.join(sorted(untrained))}")
    tokens = read_tokens(args.tokens) if args.tokens else []
    pieces = token_pieces(WhistleTokenizer().sp, tokens)
    tokenizer = WhistleTokenizer(pieces, languages)
    tuned = replace(cfg, text=tokenizer.text, languages=languages)
    note = f"  {skipped} skipped (empty, or over {SECONDS} s)" if skipped else ""
    minutes = sum(len(c[0]) for c in splits[0]) / SAMPLE_RATE / 60
    print(f"  {'data':<9} {len(splits[0])} train, {len(splits[1])} validation, {len(splits[2])} test clips"
          f"  {minutes:.1f} min to train on{note}")
    added = [lang for lang in languages if lang not in cfg.languages]
    print(f"  {'vocab':<9} {len(tokens)} tokens  {len(pieces)} new pieces"
          + (f"  new languages {', '.join(added)}" if added else ""))
    split = [t for t in tokens if any(len(tokenizer.encode(x)) != 1 for x in (t, " " + t))]
    if split:
        print(f"  {'warning':<9} not one piece: {', '.join(split)}")

    def encode(clips):
        return [(samples, [cfg.bos, tuned.tag(lang), *tokenizer.encode(text), cfg.eos], text)
                for samples, text, lang in clips]

    encoded = [encode(clips) for clips in splits]
    long = sum(len(ids) > cfg.max_tokens + 1 for clips in encoded for _, ids, _ in clips)
    items, validation, test = ([item for item in clips if len(item[1]) <= cfg.max_tokens + 1] for clips in encoded)
    if long:
        print(f"  {'skipped':<9} {long} clips with transcripts over {cfg.max_tokens} tokens")
    for path, clips in zip((args.train, args.validation, args.test), (items, validation, test)):
        if not clips:
            raise SystemExit("no usable clips in " + path)
    if tokens:
        found = sum(any(token in text for *_, text in items) for token in tokens)
        print(f"  {'tokens':<9} {found}/{len(tokens)} appear in the transcripts")

    configure_deploy(act_bits=8, kv_bits=8)
    model = Whistle(tuned)
    print(f"  {'numerics':<9} CQ W{WEIGHT_BITS} STE + A8 (matches export)")
    paths = lora_target_paths(params)
    scale = args.lora_alpha / args.lora_rank
    seed = int(getattr(args, "seed", 0))
    rng = _training_rng(seed)
    lora = init_lora(params, paths, args.lora_rank, jax.random.PRNGKey(seed))
    train = {"lora": lora, "rows": jnp.asarray(initial_rows(params["embedding"]["embedding"], cfg, tokenizer))}
    print(f"  {'lora':<9} rank {args.lora_rank}  alpha {args.lora_alpha:g}  {len(paths)} weight groups"
          f"  +{len(train['rows'])} vocabulary rows")

    def loss_fn(train, wav, lengths, tokens):
        merged = cq_ste_params(tuned_params(params, cfg, tokenizer, train["lora"], scale, train["rows"]), WEIGHT_BITS)
        logits = model.apply({"params": merged}, wav, lengths, tokens[:, :-1], quant=True)
        targets = tokens[:, 1:]
        mask = targets != cfg.pad
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    batch, count = args.batch_size, len(items)
    steps_per_epoch = -(-count // batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(max(1, total_steps // 20), total_steps - 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=warmup, decay_steps=total_steps)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = optimizer.init(train)
    print(f"  {'schedule':<9} {total_steps} steps  warmup {warmup}  cosine decay  clip 1.0  (compiling...)")

    @jax.jit
    def train_step(train, opt_state, wav, lengths, tokens):
        loss, grads = jax.value_and_grad(loss_fn)(train, wav, lengths, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, train)
        return optax.apply_updates(train, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)

    every = max(1, total_steps // 50)
    step_i = 0
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        last = 0.0
        for start in range(0, count, batch):
            train, opt_state, loss = train_step(train, opt_state, *_arrays(items, order[start:start + batch], cfg))
            last = float(loss)
            step_i += 1
            if step_i % every == 0:
                print(f"  {'step':<9} {step_i}/{total_steps}  loss {last:.4f}", flush=True)
        val = np.mean([float(eval_step(train, *_arrays(validation, range(i, min(i + batch, len(validation))), cfg)))
                       for i in range(0, len(validation), batch)])
        print(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {last:.4f}  val {val:.4f}", flush=True)

    edits, units = _score_quantised(model, params, cfg, tokenizer, train["lora"], scale, train["rows"], test, batch)
    kind = "cer" if all(by_character(text) for *_, text in test) else "wer"
    if units:
        print(f"  {kind:<9} {edits / units:.1%} on {len(test)} test clips, "
              f"greedy decoding on the W{WEIGHT_BITS} weights")
        print(f"  {'note':<9} for real-world numbers, build the .cact and transcribe the test set with it")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_adapter(out, {
        "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])} for p, v in train["lora"].items()},
        "rows": np.asarray(train["rows"]),
        "scale": float(scale),
        "base": base_path,
        "rank": args.lora_rank,
        "seed": seed,
        "pieces": list(pieces),
        "languages": list(languages),
    })
    print(f"  {'adapter':<9} {out}")
    print(f"  {'next':<9} cactus whistle build --lora {out}")


def build_main(args):
    import shutil
    import jax.numpy as jnp
    from ..agent import fetch
    from .export import read_layers
    from .build import upload_archive

    base_archive = fetch.fetch_weights(fetch.WHISTLE, force=True)
    print(f"  {'base':<9} {base_archive}  {os.path.getsize(base_archive) / 1e6:.2f} MB")
    platform = getattr(args, "platform", None)
    layers = getattr(args, "layers", None)
    folder = None
    if platform:
        folder = os.path.abspath(args.out or platform)
        for path in fetch.download_platform(platform, os.path.dirname(folder), generation=3, dest=folder):
            print(f"  {'engine':<9} {path}  {os.path.getsize(path) / 1e6:.2f} MB")
        out = os.path.join(folder, fetch.base_weights(fetch.WHISTLE))
    else:
        out = args.out
    if not args.lora and (not layers or layers == read_layers(base_archive)):
        if not out:
            raise SystemExit("pass --out <archive.cact>, or --platform to build a runnable folder")
        shutil.copyfile(base_archive, out)
        print(f"  {'wrote':<9} {out}  {os.path.getsize(out) / 1e6:.2f} MB  the published base archive")
    else:
        adapter = read_adapter(args.lora) if args.lora else None
        if adapter and adapter.get("languages") is None:
            raise SystemExit(f"{args.lora} is a Needle adapter; build it with cactus build")
        checkpoint = args.checkpoint
        if not checkpoint and adapter and adapter.get("base") and os.path.exists(adapter["base"]):
            checkpoint = adapter["base"]
        checkpoint = checkpoint or DEFAULT_BASE
        params, cfg = load_base(checkpoint)
        tokenizer = WhistleTokenizer(languages=cfg.languages)
        if args.lora:
            tokenizer = WhistleTokenizer(adapter.get("pieces") or (), adapter.get("languages") or cfg.languages)
            lora = {tuple(key.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                    for key, v in adapter["lora"].items()}
            params = tuned_params(params, cfg, tokenizer, lora, adapter["scale"], adapter.get("rows"))
            print(f"  {'merged':<9} {len(lora)} weight groups  {len(tokenizer.pieces)} new pieces  {args.lora}")
        params, cfg = rung(params, cfg, layers)
        print(f"  {'depth':<9} {cfg.dec_layers} decoder layers")
        out = out or (os.path.splitext(os.path.basename(checkpoint))[0] + ".cact")
        info = build(params, cfg, tokenizer, out)
        print(f"  {'wrote':<9} {info['path']}  {info['bytes'] / 1e6:.2f} MB  {info['tensors']} tensors  W4A8"
              f"  languages {', '.join(tokenizer.languages)}")
    if folder:
        runner = next((n for n in ("cactus", "cactus.exe") if os.path.exists(os.path.join(folder, n))), None)
        if runner:
            print(f"  {'next':<9} {os.path.join(folder, runner)} --model {os.path.basename(out)} --audio clip.wav")
    else:
        print(f"  {'next':<9} cactus.transcribe('clip.wav', weights={out!r})")

    if args.upload:
        upload_archive(out)
