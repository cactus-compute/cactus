"""Comprehensive tests for cactus Python bindings.

Tests the helper functions, auto-serialization, edge cases, and package
structure. The actual C FFI calls require a compiled libcactus — those
are integration tests run via `cactus test`.
"""
import json
import ctypes
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Import and structure tests ──────────────────────────────────────


class TestPackageStructure:
    """Verify the package is importable and has the right public API."""

    def test_version_exists(self):
        from cactus import __version__
        assert __version__ and __version__[0].isdigit()

    def test_cli_module_importable(self):
        from cactus.cli import main, create_parser
        assert callable(main)
        assert callable(create_parser)

    def test_model_module_importable(self):
        from cactus.cli.model import ensure_weights, ensure_bundle, TranspileOptions
        assert callable(ensure_weights)
        assert callable(ensure_bundle)

    def test_public_api_reexports(self):
        from cactus import ensure_model, get_weights_dir, get_model_dir_name
        assert callable(ensure_model)
        assert callable(get_weights_dir)
        assert callable(get_model_dir_name)

    def test_py_typed_marker_exists(self):
        marker = Path(__file__).parent.parent / "cactus" / "py.typed"
        assert marker.exists(), "py.typed PEP 561 marker missing"


# ── Helper function tests ───────────────────────────────────────────


class TestToJson:
    """Test the _to_json auto-serialization helper."""

    def setup_method(self):
        from cactus.bindings.cactus import _to_json
        self.to_json = _to_json

    def test_none_passthrough(self):
        assert self.to_json(None) is None

    def test_dict_serialized(self):
        result = self.to_json({"temperature": 0.7, "max_tokens": 100})
        assert isinstance(result, bytes)
        parsed = json.loads(result)
        assert parsed == {"temperature": 0.7, "max_tokens": 100}

    def test_list_serialized(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = self.to_json(messages)
        assert isinstance(result, bytes)
        parsed = json.loads(result)
        assert parsed == messages

    def test_string_encoded(self):
        result = self.to_json('{"already": "json"}')
        assert isinstance(result, bytes)
        assert result == b'{"already": "json"}'

    def test_bytes_passthrough(self):
        raw = b'{"raw": "bytes"}'
        result = self.to_json(raw)
        assert result is raw

    def test_empty_dict(self):
        result = self.to_json({})
        assert json.loads(result) == {}

    def test_empty_list(self):
        result = self.to_json([])
        assert json.loads(result) == []

    def test_nested_structure(self):
        data = {"messages": [{"role": "user", "content": "test"}], "options": {"temp": 0.5}}
        result = self.to_json(data)
        assert json.loads(result) == data

    def test_unicode_content(self):
        data = [{"role": "user", "content": "Привет мир 你好世界"}]
        result = self.to_json(data)
        parsed = json.loads(result)
        assert parsed[0]["content"] == "Привет мир 你好世界"


class TestFromJson:
    """Test the _from_json buffer decoder helper."""

    def setup_method(self):
        from cactus.bindings.cactus import _from_json
        self.from_json = _from_json

    def _make_buf(self, text):
        encoded = text.encode() if isinstance(text, str) else text
        buf = ctypes.create_string_buffer(len(encoded) + 1)
        buf.value = encoded
        return buf

    def test_valid_json(self):
        result = self.from_json(self._make_buf('{"key": "value"}'))
        assert result == {"key": "value"}

    def test_empty_buffer_returns_empty_dict(self):
        buf = ctypes.create_string_buffer(64)
        result = self.from_json(buf)
        assert result == {}

    def test_array_json(self):
        result = self.from_json(self._make_buf('[1, 2, 3]'))
        assert result == [1, 2, 3]

    def test_numeric_values(self):
        result = self.from_json(self._make_buf('{"score": 0.95, "count": 42}'))
        assert result["score"] == 0.95
        assert result["count"] == 42

    def test_nested_json(self):
        text = '{"response": "Hi", "metrics": {"tokens": 5, "latency_ms": 123.4}}'
        result = self.from_json(self._make_buf(text))
        assert result["response"] == "Hi"
        assert result["metrics"]["tokens"] == 5

    def test_unicode_response(self):
        result = self.from_json(self._make_buf('{"text": "日本語テスト"}'))
        assert result["text"] == "日本語テスト"


class TestPreparePcm:
    """Test the _prepare_pcm audio marshaling helper."""

    def setup_method(self):
        from cactus.bindings.cactus import _prepare_pcm
        self.prepare_pcm = _prepare_pcm

    def test_none_returns_null(self):
        ptr, size = self.prepare_pcm(None)
        assert ptr is None
        assert size == 0

    def test_bytes_marshaled(self):
        data = b"\x00\x01\x02\x03\xff"
        ptr, size = self.prepare_pcm(data)
        assert ptr is not None
        assert size == 5

    def test_empty_bytes(self):
        ptr, size = self.prepare_pcm(b"")
        assert size == 0

    def test_large_buffer(self):
        data = bytes(range(256)) * 100  # 25600 bytes
        ptr, size = self.prepare_pcm(data)
        assert size == 25600


class TestEnc:
    """Test the _enc string encoding helper."""

    def setup_method(self):
        from cactus.bindings.cactus import _enc
        self.enc = _enc

    def test_none_passthrough(self):
        assert self.enc(None) is None

    def test_string_encoded(self):
        result = self.enc("hello")
        assert result == b"hello"

    def test_bytes_passthrough(self):
        raw = b"hello"
        result = self.enc(raw)
        assert result is raw

    def test_unicode_encoded(self):
        result = self.enc("日本語")
        assert result == "日本語".encode()

    def test_empty_string(self):
        result = self.enc("")
        assert result == b""


# ── Model ID resolution tests ──────────────────────────────────────


# ── TranspileOptions tests ──────────────────────────────────────────


class TestTranspileOptions:
    """Test the TranspileOptions dataclass."""

    def test_defaults(self):
        from cactus.cli.model import TranspileOptions
        opts = TranspileOptions()
        assert opts.task == "auto"
        assert opts.prompt is None
        assert opts.image_files is None
        assert opts.audio_file is None
        assert opts.max_new_tokens is None
        assert opts.component_pipeline == "auto"
        assert opts.components is None
        assert opts.system_prompt is None
        assert opts.trust_remote_code is False
        assert opts.local_files_only is False
        assert opts.cache_context_length is None

    def test_custom_values(self):
        from cactus.cli.model import TranspileOptions
        opts = TranspileOptions(
            task="causal_lm_logits",
            prompt="Hello",
            max_new_tokens=256,
            trust_remote_code=True,
            cache_context_length="131072",
        )
        assert opts.task == "causal_lm_logits"
        assert opts.prompt == "Hello"
        assert opts.max_new_tokens == 256
        assert opts.trust_remote_code is True
        assert opts.cache_context_length == "131072"

    def test_frozen(self):
        from cactus.cli.model import TranspileOptions
        opts = TranspileOptions()
        with pytest.raises(AttributeError):
            opts.task = "something"


# ── CLI parser tests ────────────────────────────────────────────────


class TestCliParser:
    """Test CLI argument parsing."""

    def setup_method(self):
        from cactus.cli import create_parser
        self.parser = create_parser()

    def test_download_command(self):
        args = self.parser.parse_args(["download", "google/gemma-4-E2B-it"])
        assert args.command == "download"
        assert args.model_id == "google/gemma-4-E2B-it"

    def test_download_defaults(self):
        args = self.parser.parse_args(["download"])
        assert args.token is None
        assert args.bits == 4

    def test_run_command(self):
        args = self.parser.parse_args(["run", "google/gemma-4-E2B-it", "--prompt", "hi"])
        assert args.command == "run"
        assert args.model_id == "google/gemma-4-E2B-it"
        assert args.prompt == "hi"

    def test_run_command_chunked_bundle_flags(self):
        args = self.parser.parse_args([
            "run", "Foo/Bar",
            "--audio", "audio.wav",
            "--image", "image.png",
            "--input-ids", "1,2,3",
            "--input-ids-file", "tokens.txt",
            "--max-new-tokens", "4",
            "--result-json", "result.json",
        ])
        assert args.command == "run"
        assert args.audio == "audio.wav"
        assert args.image == "image.png"
        assert args.input_ids == "1,2,3"
        assert args.input_ids_file == "tokens.txt"
        assert args.max_new_tokens == 4
        assert args.result_json == "result.json"

    def test_convert_command(self):
        args = self.parser.parse_args(["convert", "Qwen/Qwen3-0.6B", "--bits", "2"])
        assert args.command == "convert"
        assert args.model_id == "Qwen/Qwen3-0.6B"
        assert args.bits == 2

    def test_build_command(self):
        args = self.parser.parse_args(["build", "--apple"])
        assert args.command == "build"
        assert args.apple is True

    def test_test_command_suite(self):
        args = self.parser.parse_args(["test", "--suite", "llm"])
        assert args.command == "test"
        assert args.suite == "llm"
        assert args.component == "all"

    def test_test_command_component(self):
        args = self.parser.parse_args(["test", "--component", "kernels"])
        assert args.command == "test"
        assert args.component == "kernels"
        assert args.suite is None

    def test_test_command_defaults(self):
        args = self.parser.parse_args(["test"])
        assert args.command == "test"
        assert args.component == "all"
        assert args.suite is None

    def test_auth_command(self):
        args = self.parser.parse_args(["auth", "--status"])
        assert args.command == "auth"
        assert args.status is True

    def test_clean_command(self):
        args = self.parser.parse_args(["clean"])
        assert args.command == "clean"

    def test_no_command_prints_help(self):
        args = self.parser.parse_args([])
        assert args.command is None

    def test_run_rejects_bare_name(self):
        import pytest
        with pytest.raises(SystemExit):
            self.parser.parse_args(["run", "whisper-base", "--prompt", "hi"])

    def test_download_bits_flag(self):
        args = self.parser.parse_args(["download", "Foo/Bar", "--bits", "2"])
        assert args.bits == 2



# ── Index API tests ────────────────────────────────────────────────


class TestIndexApi:
    """Test the vector index API (no model weights needed)."""

    def setup_method(self):
        import tempfile
        from cactus.bindings.cactus import (
            cactus_index_init, cactus_index_add, cactus_index_query,
            cactus_index_get, cactus_index_delete, cactus_index_compact,
            cactus_index_destroy,
        )
        self.tmpdir = tempfile.mkdtemp()
        self.init = cactus_index_init
        self.add = cactus_index_add
        self.query = cactus_index_query
        self.get = cactus_index_get
        self.delete = cactus_index_delete
        self.compact = cactus_index_compact
        self.destroy = cactus_index_destroy
        self._handles = []

    def _create_index(self, dim):
        idx = self.init(self.tmpdir, dim)
        self._handles.append(idx)
        return idx

    def teardown_method(self):
        import shutil
        for h in self._handles:
            try:
                self.destroy(h)
            except Exception:
                pass
        self._handles.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_destroy(self):
        idx = self._create_index(4)
        assert idx is not None

    def test_add_and_query(self):
        idx = self._create_index(4)
        self.add(idx, [0, 1, 2], ["doc a", "doc b", "doc c"],
                 embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        result = self.query(idx, [1, 0, 0, 0], {"n_results": 2})
        ids = [r["id"] for r in result["results"]]
        assert 0 in ids

    def test_get_by_id(self):
        idx = self._create_index(3)
        self.add(idx, [42], ["hello world"], embeddings=[[1, 0, 0]])
        result = self.get(idx, [42])
        assert result["results"][0]["document"] == "hello world"

    def test_delete(self):
        idx = self._create_index(3)
        self.add(idx, [0, 1], ["a", "b"], embeddings=[[1, 0, 0], [0, 1, 0]])
        self.delete(idx, [0])
        result = self.query(idx, [1, 0, 0], {"n_results": 10})
        ids = [r["id"] for r in result["results"]]
        assert 0 not in ids

    def test_compact(self):
        idx = self._create_index(3)
        self.add(idx, [0], ["x"], embeddings=[[1, 0, 0]])
        self.compact(idx)
        result = self.query(idx, [1, 0, 0], {"n_results": 1})
        assert len(result["results"]) >= 1

    def test_persistence(self):
        idx = self._create_index(3)
        self.add(idx, [0], ["persistent"], embeddings=[[1, 0, 0]])
        self.destroy(idx)
        self._handles.remove(idx)
        idx2 = self._create_index(3)
        result = self.get(idx2, [0])
        assert result["results"][0]["document"] == "persistent"


# ── Streaming callback tests ──────────────────────────────────────


class TestStreamingCallbacks:
    """Test the token callback wrapper (no model weights needed)."""

    def setup_method(self):
        from cactus.bindings.cactus import _make_token_callback, TokenCallback
        self.make_cb = _make_token_callback
        self.TokenCallback = TokenCallback

    def test_none_returns_valid_callback(self):
        cb = self.make_cb(None)
        assert cb is not None

    def test_false_returns_valid_callback(self):
        cb = self.make_cb(False)
        assert cb is not None

    def test_callable_returns_valid_callback(self):
        cb = self.make_cb(lambda text, tid: None)
        assert cb is not None

    def test_callback_is_correct_ctypes_type(self):
        cb = self.make_cb(lambda text, tid: None)
        assert isinstance(cb, self.TokenCallback)

    def test_multiple_callbacks_independent(self):
        results_a = []
        results_b = []
        cb_a = self.make_cb(lambda t, i: results_a.append(t))
        cb_b = self.make_cb(lambda t, i: results_b.append(t))
        assert cb_a is not cb_b


# ── Error path tests ──────────────────────────────────────────────


class TestErrorPaths:
    """Test error handling (no model weights needed)."""

    def test_init_bad_path_raises(self):
        from cactus.bindings.cactus import cactus_init
        with pytest.raises(RuntimeError):
            cactus_init("/nonexistent/path/to/model")

    def test_init_empty_path_raises(self):
        from cactus.bindings.cactus import cactus_init
        with pytest.raises(RuntimeError):
            cactus_init("")

    def test_get_last_error_returns_string(self):
        from cactus.bindings.cactus import cactus_init, cactus_get_last_error
        try:
            cactus_init("/nonexistent")
        except RuntimeError:
            pass
        err = cactus_get_last_error()
        assert isinstance(err, str)
        assert len(err) > 0

    def test_init_bad_path_error_message(self):
        from cactus.bindings.cactus import cactus_init, cactus_get_last_error
        with pytest.raises(RuntimeError, match="config"):
            cactus_init("/nonexistent/model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
