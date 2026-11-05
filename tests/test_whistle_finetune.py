from cactus.cli import speech
import json
import types
import wave

import pytest

from conftest import _engine_available

TINY = dict(dim=64, kv_heads=2, qk_dim=16, v_dim=16, enc_layers=2, dec_layers=3, conv_kernel=3, lanes=2,
            enc_lanes=2, sites=(2,), slots=64, hashes=1, stem=8, stem_stages=2, max_tokens=64)


@pytest.fixture(scope="module")
def tiny_whistle(tmp_path_factory):
    import numpy as np
    import jax
    from cactus.model.checkpoints import write_checkpoint
    from cactus.model.whistle import Whistle, WhistleConfig

    cfg = WhistleConfig(**TINY, flash=False)
    params = jax.tree.map(np.asarray, Whistle(cfg).init(
        jax.random.PRNGKey(0), np.zeros((1, 3200), np.float32), np.array([3200]), np.zeros((1, 4), np.int32))["params"])
    path = tmp_path_factory.mktemp("whistle") / "tiny.safetensors"
    write_checkpoint(str(path), {"params": params, "config": cfg.__dict__})
    return str(path)


@pytest.fixture
def published_whistle(tiny_whistle, tmp_path, monkeypatch):
    from cactus.agent import fetch
    from cactus.model.whistle_finetune import WhistleTokenizer, build, load_base

    archive = str(tmp_path / "whistle.cact")
    params, cfg = load_base(tiny_whistle)
    build(params, cfg, WhistleTokenizer(languages=cfg.languages), archive)
    calls = []
    monkeypatch.setattr(fetch, "fetch_weights", lambda generation=2, dest_dir=None, force=False:
                        calls.append((generation, force)) or archive)
    return archive, calls


def _build_args(**kw):
    return types.SimpleNamespace(**{"checkpoint": None, "lora": None, "out": None, "upload": False, "layers": None,
                                    "platform": None, **kw})


def _clips(folder, lines):
    import numpy as np
    rng = np.random.default_rng(0)
    with open(folder / "clips.jsonl", "w", encoding="utf-8") as out:
        for i, text in enumerate(lines):
            with wave.open(str(folder / f"{i}.wav"), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
                w.writeframes((rng.standard_normal(16000) * 3000).astype("<i2").tobytes())
            out.write(json.dumps({"audio": f"{i}.wav", "text": text}, ensure_ascii=False) + "\n")
    return str(folder / "clips.jsonl")


def test_declared_tokens_become_one_piece_anywhere_and_match_the_engine_encoder(tmp_path):
    from cactus.model.export import RefTokenizer, parse_tokenizer_blob
    from cactus.model.whistle_finetune import WhistleTokenizer, read_tokens, token_pieces

    listed = tmp_path / "tokens.txt"
    listed.write_text("ação\nSão Paulo, SP\n🙂\n\nПривет\nASP.NET\n你好\n", encoding="utf-8")
    tokens = read_tokens(str(listed))
    assert tokens == ["ação", "São Paulo, SP", "🙂", "Привет", "ASP.NET", "你好"]
    base = WhistleTokenizer()
    tokenizer = WhistleTokenizer(token_pieces(base.sp, tokens))
    meta = parse_tokenizer_blob(tokenizer.blob())
    assert set(meta["types"][base.text:tokenizer.text]) == {0}
    engine = RefTokenizer(meta)
    for token in tokens:
        for text in (token, " " + token, f"disse {token} hoje"):
            ids = tokenizer.encode(text)
            assert any(tokenizer.sp.id_to_piece(i).replace("\u2581", " ").strip() == token for i in ids)
            assert engine.encode(text) == ids and tokenizer.sp.decode(ids) == text
    assert token_pieces(base.sp, ["the"]) == []


def test_extended_tokenizer_appends_pieces_and_tags():
    from cactus.model.export import parse_tokenizer_blob
    from cactus.model.whistle_finetune import WhistleTokenizer

    plain = WhistleTokenizer((), ("en",))
    tokenizer = WhistleTokenizer(("ã", "ão"), ("en", "pt"))
    assert tokenizer.text == plain.text + 2
    assert tokenizer.encode("não") == [plain.sp.piece_to_id("n"), plain.text + 1]
    meta = parse_tokenizer_blob(tokenizer.blob())
    assert meta["pieces"][plain.text:] == ["ã", "ão", "<|en|>", "<|pt|>"]
    assert meta["types"][-2:] == [3, 3] and meta["byte_fallback"]


@pytest.mark.slow
def test_finetune_adds_a_language_and_builds_an_archive_with_it(tiny_whistle, published_whistle, tmp_path, capsys):
    from cactus.model.checkpoints import read_adapter
    from cactus.model.export import parse_tokenizer_blob, read_export
    from cactus.model.whistle_finetune import WhistleTokenizer, build_main, finetune_main, token_pieces

    files = {}
    for name, lines in (("train", ["não é coração", "ação não", "não não", "coração"] * 2),
                        ("validation", ["não é coração", "ação"]), ("test", ["coração não", "não"])):
        (tmp_path / name).mkdir()
        files[name] = _clips(tmp_path / name, lines)
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("ação\ncoração\n", encoding="utf-8")
    adapter = str(tmp_path / "adapter.safetensors")
    finetune_main(types.SimpleNamespace(
        **files, language="pt", tokens=str(tokens), checkpoint=tiny_whistle, epochs=1, batch_size=2, lr=1e-3,
        lora_rank=4, lora_alpha=8.0, seed=0, checkpoint_dir=str(tmp_path), out=adapter))
    printed = capsys.readouterr().out
    assert "8 train, 2 validation, 2 test clips" in printed and "val " in printed
    assert "CQ W4 STE + A8 (matches export)" in printed and "on 2 test clips, greedy decoding on the W4 weights" in printed
    saved = read_adapter(adapter)
    pieces = token_pieces(WhistleTokenizer().sp, ["ação", "coração"])
    assert saved["languages"][-1] == "pt" and saved["pieces"] == pieces
    assert saved["rows"].shape == (len(pieces) + 1, TINY["dim"]) and len(saved["lora"]) == 15

    out = str(tmp_path / "tuned.cact")
    build_main(_build_args(lora=adapter, out=out))
    header, tensors = read_export(out)
    tokenizer = parse_tokenizer_blob(tensors[-1])
    text = 8192 + len(pieces)
    assert header["vocab_size"] == text + 8 and tensors[0].shape == (text + 8, TINY["dim"])
    assert tokenizer["pieces"][-1] == "<|pt|>" and tokenizer["pieces"][8192:text] == saved["pieces"]
    manifest = next(t for t in tensors if getattr(t, "shape", None) == (13,) and t[0] == 3246)
    assert manifest[2] == text
    from cactus.agent.whistle import LANGUAGES, languages
    assert languages(out) == LANGUAGES + ("pt",)


@pytest.mark.slow
def test_build_without_an_adapter_copies_the_published_archive(published_whistle, tmp_path):
    from cactus.model.whistle_finetune import build_main

    archive, calls = published_whistle
    out = tmp_path / "copy.cact"
    build_main(_build_args(out=str(out)))
    assert out.read_bytes() == open(archive, "rb").read() and calls == [("whistle", True)]
    with pytest.raises(SystemExit, match="--out"):
        build_main(_build_args())


@pytest.mark.slow
def test_build_layers_exports_a_decoder_rung(tiny_whistle, published_whistle, tmp_path):
    from cactus.model.export import read_export
    from cactus.model.whistle_finetune import build_main

    out = str(tmp_path / "rung.cact")
    build_main(_build_args(checkpoint=tiny_whistle, layers=2, out=out))
    header, _ = read_export(out)
    assert header["num_layers"] == 2 and header["engram_layers"] == (1,)
    with pytest.raises(ValueError, match="--layers"):
        build_main(_build_args(checkpoint=tiny_whistle, layers=5, out=out))


def test_lines_need_a_language(tmp_path):
    from cactus.model.whistle_finetune import read_clips

    data = _clips(tmp_path, ["hello"])
    with pytest.raises(SystemExit, match="language"):
        read_clips(data)
    clips, skipped = read_clips(data, "en")
    assert len(clips) == 1 and clips[0][1:] == ("hello", "en") and not skipped


def test_errors_use_the_whisper_normalizers_and_characters_for_unspaced_scripts():
    from cactus.model.whistle_finetune import errors

    assert errors("Olá, mundo!", "olá mundo", "pt") == (0, 2)
    assert errors("a b c", "a x c d", "pt") == (2, 3)
    assert errors("Mr. Smith paid ten dollars.", "mister smith paid $10", "en") == (0, 4)
    assert errors("今天天气很好。", "今天天汽很好", "zh") == (1, 6)
    assert errors("こんにちは 世界", "こんにちわ世界", "ja") == (1, 7)
    assert errors("Привет, мир", "привет мир", "be") == (0, 2)


def test_bad_lines_name_the_line(tmp_path):
    from cactus.model.whistle_finetune import read_clips

    data = _clips(tmp_path, ["olá"])
    for line, message in ([("not json", "2: not JSON"), ('{"text": "a"}', "2: no \"audio\""),
                           ('{"audio": "gone.wav", "text": "a"}', "gone.wav does not exist"),
                           ('{"audio": "0.wav", "text": "a", "language": "p t"}', "2: give the line")]):
        with open(data, "a", encoding="utf-8") as out:
            out.write(line + "\n")
        with pytest.raises(SystemExit, match=message):
            read_clips(data, "pt")
        lines = open(data, encoding="utf-8").read().splitlines()[:-1]
        open(data, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def test_tokens_come_from_a_file(tmp_path):
    from cactus.model.whistle_finetune import read_tokens

    with pytest.raises(SystemExit, match="one token per line"):
        read_tokens("ação,coração")
    listed = tmp_path / "tokens.txt"
    listed.write_bytes("﻿ação \r\n\n  coração\n".encode("utf-8"))
    assert read_tokens(str(listed)) == ["ação", "coração"]
    listed.write_text("ação\n<|pt|>\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="looks like a language tag"):
        read_tokens(str(listed))


@pytest.mark.slow
def test_wrong_kinds_of_files_are_refused(tiny_whistle, tiny_checkpoint, published_whistle, tmp_path):
    from cactus.model.checkpoints import write_adapter
    from cactus.model.build import build_main as needle_build
    from cactus.model.whistle_finetune import build_main, load_base

    with pytest.raises(SystemExit, match="not a Whistle checkpoint"):
        load_base(tiny_checkpoint)
    needle_adapter = str(tmp_path / "cactus.safetensors")
    write_adapter(needle_adapter, {"lora": {}, "scale": 2.0, "base": tiny_checkpoint, "rank": 4, "seed": 0})
    with pytest.raises(SystemExit, match="Needle adapter"):
        build_main(_build_args(lora=needle_adapter, out=str(tmp_path / "x.cact")))
    whistle_adapter = str(tmp_path / "whistle.safetensors")
    write_adapter(whistle_adapter, {"lora": {}, "scale": 2.0, "base": tiny_whistle, "rank": 4, "seed": 0,
                                    "pieces": [], "languages": ["en"]})
    with pytest.raises(SystemExit, match="Whistle adapter"):
        needle_build(types.SimpleNamespace(checkpoint=tiny_checkpoint, lora=whistle_adapter,
                                           out=str(tmp_path / "y.cact"), upload=False))


def test_finetune_needs_an_epoch_and_a_batch():
    from cactus.model.whistle_finetune import finetune_main

    for epochs, batch in ((0, 8), (1, 0)):
        with pytest.raises(SystemExit, match="at least 1"):
            finetune_main(types.SimpleNamespace(epochs=epochs, batch_size=batch))


def _splits(tmp_path, **lines):
    files = {}
    for name in ("train", "validation", "test"):
        (tmp_path / name).mkdir()
        files[name] = _clips(tmp_path / name, lines.get(name, ["não é coração", "ação"]))
    return files


def _finetune(tiny_whistle, tmp_path, files, **kw):
    from cactus.model.whistle_finetune import finetune_main

    finetune_main(types.SimpleNamespace(**{
        **files, "language": "pt", "tokens": None, "checkpoint": tiny_whistle, "epochs": 1, "batch_size": 2,
        "lr": 1e-3, "lora_rank": 4, "lora_alpha": 8.0, "seed": 0, "checkpoint_dir": str(tmp_path),
        "out": str(tmp_path / "adapter.safetensors"), **kw}))


@pytest.mark.slow
def test_every_split_needs_clips_within_the_token_limit(tiny_whistle, tmp_path):
    files = _splits(tmp_path, validation=["não é coração " * 30])
    with pytest.raises(SystemExit, match="validation"):
        _finetune(tiny_whistle, tmp_path, files)


@pytest.mark.slow
def test_a_new_language_needs_training_clips(tiny_whistle, tmp_path):
    files = _splits(tmp_path)
    rows = [json.loads(line) for line in open(files["test"], encoding="utf-8")]
    with open(files["test"], "w", encoding="utf-8") as out:
        out.writelines(json.dumps({**row, "language": "xx"}) + "\n" for row in rows)
    with pytest.raises(SystemExit, match="xx"):
        _finetune(tiny_whistle, tmp_path, files)


@pytest.mark.slow
def test_the_adapter_path_is_checked_before_training(tiny_whistle, tmp_path, capsys):
    files = _splits(tmp_path)
    with pytest.raises(SystemExit, match=r"\.safetensors"):
        _finetune(tiny_whistle, tmp_path, files, out=str(tmp_path / "adapter.bin"))
    assert "epoch" not in capsys.readouterr().out
    out = tmp_path / "new" / "adapter.safetensors"
    _finetune(tiny_whistle, tmp_path, files, out=str(out))
    assert out.exists()


def test_cli_routes_finetune_and_build_and_takes_a_tuned_language(monkeypatch):
    import sys
    import cactus._telemetry
    import cactus.agent.whistle
    import cactus.cli
    import cactus.model.whistle_finetune as tune

    seen = []
    monkeypatch.setattr(cactus._telemetry, "track", lambda *a, **k: None)
    monkeypatch.setattr(tune, "finetune_main", lambda args: seen.append(("finetune", args)))
    monkeypatch.setattr(tune, "build_main", lambda args: seen.append(("build", args)))
    monkeypatch.setattr(speech, "playground", lambda args: seen.append(("playground", args)))
    for argv in (["finetune", "train.jsonl", "dev.jsonl", "test.jsonl", "--language", "pt", "--tokens", "tokens.txt"],
                 ["build", "--lora", "a.safetensors", "--layers", "4", "--out", "t.cact"],
                 ["playground", "--language", "pt", "--weights", "t.cact"]):
        monkeypatch.setattr(sys, "argv", ["cactus", "whistle", *argv])
        cactus.cli.main()
    (_, tuned), (_, built), (_, played) = seen
    assert (tuned.train, tuned.validation, tuned.test, tuned.language, tuned.tokens, tuned.batch_size, tuned.lr) == (
        "train.jsonl", "dev.jsonl", "test.jsonl", "pt", "tokens.txt", 16, 1e-4)
    assert (built.lora, built.layers, built.out, built.platform, built.upload) == ("a.safetensors", 4, "t.cact", None, False)
    assert played.language == "pt"


@pytest.mark.slow
@pytest.mark.skipif(not _engine_available(3), reason="Whistle engine not installed")
def test_the_python_encoder_matches_the_engine_embedding(tiny_whistle, published_whistle):
    import numpy as np
    from cactus.agent.whistle import Whistle as Engine
    from cactus.model.quantize import WEIGHT_BITS, cq_ste_params
    from cactus.model.whistle import Whistle
    from cactus.model.whistle_finetune import load_base

    params, cfg = load_base(tiny_whistle)
    quantised = cq_ste_params(params, WEIGHT_BITS)
    engine = Engine(weights=published_whistle[0])
    t = np.arange(32000)
    sound = 0.2 * np.sin(0.07 * t) * (1 + np.sin(0.002 * t)) + 0.01 * np.random.default_rng(0).standard_normal(32000)
    for level in (1.0, 0.01):
        audio = (sound * level).astype(np.float32)
        theirs = np.array(engine.embed(audio)).reshape(-1, cfg.dim)
        spans = np.sqrt((audio[:len(audio) // 320 * 320].reshape(-1, 320) ** 2).mean(1))
        levelled = audio * (0.1 / np.sort(spans)[int(0.99 * (len(spans) - 1))])
        ours, frames = Whistle(cfg).apply({"params": quantised}, levelled[None], np.array([len(audio)]), True, method=Whistle.encode)
        ours = np.asarray(ours[0, :int(frames[0])])
        assert ours.shape == theirs.shape
        assert np.linalg.norm(ours - theirs) < 0.05 * np.linalg.norm(theirs)
