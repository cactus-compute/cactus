from __future__ import annotations

from dataclasses import dataclass
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from cactus.transpile.capture_jax import JaxGraphSpec
from cactus.transpile.jax_user_graph_bundle import build_jax_user_graph_bundle
from cactus.transpile.jax_user_graph_bundle import flatten_jax_params


DEFAULT_OUT_DIR = Path("artifacts/needle_jax_user_graph_bundle")
DEFAULT_CHECKPOINT = Path("checkpoints/needle.pkl")
DEFAULT_TOKENIZER_MODEL = Path("checkpoints/tokenizer/needle.model")
DEFAULT_NEEDLE_ROOT = Path("/private/tmp/cactus-needle")
DEFAULT_TOOLS = (
    '[{"name":"get_weather","description":"Get current weather for a city.",'
    '"parameters":{"location":{"type":"string","description":"City name.","required":true}}}]'
)


def _time_ms(fn, *, warmup: int = 2, iterations: int = 6) -> float:
    for _ in range(warmup):
        fn()
    start = perf_counter()
    for _ in range(iterations):
        fn()
    return (perf_counter() - start) * 1000.0 / iterations


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_tool_call(text: str) -> str:
    return text[len("<tool_call>") :] if text.startswith("<tool_call>") else text


def _decode_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@dataclass
class GenerationResult:
    query: str
    text: str
    parsed: Any | None
    generated_tokens: list[int]
    cactus_prefill_ms: float
    cactus_prefill_tps: float
    cactus_decode_ms: float
    cactus_decode_tps: float


class NeedleJaxUserGraphSession:
    """Needle demo that consumes the generic JAX user-graph bundler."""

    def __init__(
        self,
        *,
        out_dir: str | Path = DEFAULT_OUT_DIR,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        tokenizer_model: str | Path = DEFAULT_TOKENIZER_MODEL,
        needle_root: str | Path = DEFAULT_NEEDLE_ROOT,
        max_enc_len: int = 256,
        max_gen_len: int = 48,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.checkpoint = Path(checkpoint)
        self.tokenizer_model = Path(tokenizer_model)
        self.needle_root = Path(needle_root)
        self.max_enc_len = int(max_enc_len)
        self.max_gen_len = int(max_gen_len)
        self.bundle = None
        self.params = None
        self.model = None
        self.tokenizer = None
        self.needle_arch = None
        self.needle_tok = None

    def build(self, *, example_query: str = "What's the weather in San Francisco?", tools: str = DEFAULT_TOOLS) -> None:
        self._load_needle()
        flat_params = flatten_jax_params(self.params)

        enc_input, src_mask = self._encode_source(example_query, tools)
        encoder_out, enc_mask = self._encoder_graph(self.params, enc_input, src_mask)
        encoder_out.block_until_ready()
        tgt_mask = self._decoder_self_mask()
        seed_buffer = self._seed_decoder_tokens()

        result = build_jax_user_graph_bundle(
            params=self.params,
            output_dir=self.out_dir,
            model_id="needle_real_jax_user_graph",
            task="text-generation",
            family="needle",
            inputs_metadata={
                "max_enc_len": self.max_enc_len,
                "max_gen_len": self.max_gen_len,
                "tokenizer_model": str(self.tokenizer_model),
            },
            graph_meta={"model": "needle_real"},
            weight_arrays={name: value.astype(np.float16) for name, value in flat_params.items()},
            exclude_weights={"log_temp"},
            specs=(
                JaxGraphSpec(
                    name="encoder",
                    role="encoder",
                    fn=self._encoder_graph,
                    example_args=(enc_input, src_mask),
                    input_names=("source_tokens", "src_mask"),
                    output_names=("encoder_out", "enc_mask"),
                ),
                JaxGraphSpec(
                    name="decoder_prefill",
                    role="decoder_prefill",
                    fn=self._decoder_prefill_graph,
                    example_args=(seed_buffer, encoder_out, tgt_mask, enc_mask),
                    input_names=("target_tokens", "encoder_out", "self_mask", "cross_mask"),
                    output_names=("logits",),
                ),
            ),
        )
        self.bundle = result.bundle
        self._write_compat_manifest(result.components_manifest_path, result.weights_dir)

    def generate(self, query: str, tools: str = DEFAULT_TOOLS, *, max_new_tokens: int | None = None) -> GenerationResult:
        if self.bundle is None:
            self.build(example_query=query, tools=tools)
        assert self.bundle is not None

        enc_input, src_mask = self._encode_source(query, tools)
        tgt_mask = self._decoder_self_mask()

        encoder_graph = self.bundle.graphs["encoder"].graph
        decoder_graph = self.bundle.graphs["decoder_prefill"].graph
        encoder_graph.set_inputs([np.asarray(enc_input), np.asarray(src_mask)])
        encoder_cactus, mask_cactus = [output.numpy() for output in encoder_graph.execute()]

        cactus_buffer = np.asarray(self._seed_decoder_tokens()).copy()
        decoder_inputs = [cactus_buffer, encoder_cactus, np.asarray(tgt_mask), mask_cactus]
        decoder_graph.set_inputs(decoder_inputs)
        cactus_decoder_ms = _time_ms(lambda: decoder_graph.execute())
        cactus_encoder_ms = _time_ms(lambda: encoder_graph.execute())
        cactus_prefill_ms = cactus_encoder_ms + cactus_decoder_ms

        limit = min(max_new_tokens or self.max_gen_len - 1, self.max_gen_len - 1)
        generated: list[int] = []
        start = perf_counter()
        for index in range(limit):
            logits = self.bundle.execute(
                "decoder_prefill",
                cactus_buffer,
                encoder_cactus,
                tgt_mask,
                mask_cactus,
            )[0].numpy().astype(np.float32)
            next_token = int(np.nanargmax(logits[0, index]))
            if next_token == self.tokenizer.eos_token_id:
                break
            generated.append(next_token)
            cactus_buffer[0, index + 1] = next_token
        cactus_decode_ms = (perf_counter() - start) * 1000.0

        text = _strip_tool_call(self.tokenizer.decode(generated))
        generated_count = max(len(generated), 1)
        source_count = max(int(np.count_nonzero(np.asarray(enc_input) != self.tokenizer.pad_token_id)), 1)
        return GenerationResult(
            query=query,
            text=text,
            parsed=_decode_json(text),
            generated_tokens=generated,
            cactus_prefill_ms=cactus_prefill_ms,
            cactus_prefill_tps=source_count / (cactus_prefill_ms / 1000.0),
            cactus_decode_ms=cactus_decode_ms,
            cactus_decode_tps=generated_count / (cactus_decode_ms / 1000.0),
        )

    def _load_needle(self) -> None:
        arch_path = self.needle_root / "needle/model/architecture.py"
        tokenizer_path = self.needle_root / "needle/dataset/tokenizer.py"
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint: {self.checkpoint}")
        if not self.tokenizer_model.exists():
            raise FileNotFoundError(f"missing tokenizer model: {self.tokenizer_model}")
        if not arch_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(f"missing Needle checkout at {self.needle_root}")

        self.needle_arch = _load_module("needle_architecture_bundle", arch_path)
        self.needle_tok = _load_module("needle_tokenizer_bundle", tokenizer_path)
        with self.checkpoint.open("rb") as f:
            data = pickle.load(f)
        self.params = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.bfloat16), data["params"])
        config = self.needle_arch.TransformerConfig(**data["config"])
        self.model = self.needle_arch.SimpleAttentionNetwork(config)
        self.tokenizer = self.needle_tok.NeedleTokenizer(str(self.tokenizer_model))

    def _encode_source(self, query: str, tools: str) -> tuple[Any, Any]:
        query_tokens = self.tokenizer.encode(query)[: self.max_enc_len - 2]
        tool_tokens = self.tokenizer.encode(tools)
        remaining = self.max_enc_len - len(query_tokens) - 1
        enc_tokens = query_tokens + [self.tokenizer.tools_token_id] + tool_tokens[:remaining]
        padded = enc_tokens + [self.tokenizer.pad_token_id] * (self.max_enc_len - len(enc_tokens))
        enc_input = jnp.asarray([padded], dtype=jnp.int32)
        return enc_input, self.needle_arch.make_padding_mask(enc_input, self.tokenizer.pad_token_id)

    def _seed_decoder_tokens(self) -> Any:
        seed_buffer = jnp.full((1, self.max_gen_len), self.tokenizer.pad_token_id, dtype=jnp.int32)
        return seed_buffer.at[0, 0].set(self.tokenizer.eos_token_id)

    def _decoder_self_mask(self) -> Any:
        return self.needle_arch.make_causal_mask(self.max_gen_len)

    def _encoder_graph(self, model_params, source_tokens, src_mask_arg):
        return self.model.apply({"params": model_params}, source_tokens, src_mask=src_mask_arg, method="encode")

    def _decoder_prefill_graph(self, model_params, target_tokens, encoder_state, self_mask, cross_mask):
        return self.model.apply(
            {"params": model_params},
            target_tokens,
            encoder_state,
            self_mask=self_mask,
            cross_mask=cross_mask,
            method="decode",
        )

    def _write_compat_manifest(self, components_manifest_path: Path, weights_dir: Path) -> None:
        assert self.bundle is not None
        components_manifest = json.loads(components_manifest_path.read_text())
        legacy_graphs = {
            component["component"]: {
                "path": component["graph"],
                "role": component["component"],
                "input_names": component["logical_inputs"],
                "output_names": component["logical_outputs"],
                "runtime_input_node_ids": component["runtime_input_node_ids"],
                "output_node_ids": component["output_node_ids"],
                "bound_constant_bindings": component["bound_constant_bindings"],
            }
            for component in components_manifest["components"]
        }
        legacy_manifest = {
            "format": "jax_user_graph_bundle.v0",
            "model": "needle_real",
            "max_enc_len": self.max_enc_len,
            "max_gen_len": self.max_gen_len,
            "checkpoint": str(self.checkpoint),
            "tokenizer_model": str(self.tokenizer_model),
            "weights_dir": str(weights_dir),
            "weights_manifest": str(weights_dir / "weights_manifest.json"),
            "components_manifest": str(components_manifest_path),
            "graphs": legacy_graphs,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(legacy_manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run a generic JAX user-graph Needle bundle.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-model", type=Path, default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--needle-root", type=Path, default=DEFAULT_NEEDLE_ROOT)
    parser.add_argument("--query", default="What's the weather in San Francisco?")
    parser.add_argument("--tools", default=DEFAULT_TOOLS)
    parser.add_argument("--max-enc-len", type=int, default=256)
    parser.add_argument("--max-gen-len", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    session = NeedleJaxUserGraphSession(
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        tokenizer_model=args.tokenizer_model,
        needle_root=args.needle_root,
        max_enc_len=args.max_enc_len,
        max_gen_len=args.max_gen_len,
    )
    session.build(example_query=args.query, tools=args.tools)
    result = session.generate(args.query, args.tools, max_new_tokens=args.max_new_tokens)

    print(f"bundle_dir={session.out_dir}")
    print(f"manifest={session.out_dir / 'manifest.json'}")
    print(f"query={result.query!r}")
    print(f"text={result.text}")
    print(f"parsed={json.dumps(result.parsed, sort_keys=True) if result.parsed is not None else None}")
    print(
        "timings="
        f"prefill_ms={result.cactus_prefill_ms:.2f}, "
        f"prefill_tps={result.cactus_prefill_tps:.2f}, "
        f"decode_ms={result.cactus_decode_ms:.2f}, "
        f"decode_tps={result.cactus_decode_tps:.2f}"
    )


if __name__ == "__main__":
    main()
