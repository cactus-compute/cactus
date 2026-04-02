#!/usr/bin/env python3
"""Export Needle metadata in the same layout used by other Cactus LLMs."""

from __future__ import annotations

import argparse
import struct
import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

MODEL_REPO = "Cactus-Compute/checkpoints"
TOKENIZER_REPO = "Cactus-Compute/needle-tokenizer"
DEFAULT_CHECKPOINT_FILE = "needle_16_640_best.pkl"
DEFAULT_TOKENIZER_REVISION = "5a50f268260b546cbcff02a2b5d4e1a51ac03ef1"
CHECKPOINT_REVISIONS = {
    "needle_12_512_best.pkl": "c3abe44ccce513833aa1a671a508433a6c4224eb",
    "needle_8_512_best.pkl": "bb785f03ebaa395bc7eb4cd4b18c713f3a05a58d",
    "needle_12_768_best.pkl": None,
    "needle_16_640_best.pkl": "9447786ba0d421434c35bf7b77239e83540a641f",
    "needle_16_768_best.pkl": "bb785f03ebaa395bc7eb4cd4b18c713f3a05a58d",
}
CHECKPOINT_TOKENIZER_REVISIONS = {
    # Older 12x512 family
    "needle_12_512_best.pkl": "662ab737ec41afed5acd215271d6d0e26690dd8b",
    "needle_12_768_best.pkl": "662ab737ec41afed5acd215271d6d0e26690dd8b",
    # Newer multilingual tokenizer family
    "needle_8_512_best.pkl": "f1fd238b770af3175898a5541de69e1e26ef6b20",
    "needle_16_640_best.pkl": "5a50f268260b546cbcff02a2b5d4e1a51ac03ef1",
    "needle_16_768_best.pkl": "f1fd238b770af3175898a5541de69e1e26ef6b20",
}
MODEL_ID_TO_CHECKPOINT_FILE = {
    "needle": DEFAULT_CHECKPOINT_FILE,
    "needle-12-512": "needle_12_512_best.pkl",
    "needle-8-512": "needle_8_512_best.pkl",
    "needle-12-768": "needle_12_768_best.pkl",
    "needle-16-640": "needle_16_640_best.pkl",
    "needle-16-768": "needle_16_768_best.pkl",
}
DEFAULT_MODEL_FILE = "needle.model"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PYTHON_ROOT = SCRIPT_DIR.parent

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from src.tensor_io import (
    create_quantization_stats,
    print_quantization_summary,
    save_tensor_with_header,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Needle config/tokenizer metadata using Cactus naming."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory (default: derived under weights/ from the selected Needle checkpoint)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Local Needle checkpoint .pkl path",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=DEFAULT_CHECKPOINT_FILE,
        help="Checkpoint filename in the Needle Hugging Face repo",
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=None,
        help="Optional checkpoint revision in the Needle Hugging Face repo (default: automatic per checkpoint)",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Local tokenizer .model path or directory",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Optional tokenizer revision in the Needle Hugging Face repo",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token; also reads HF_TOKEN/HUGGINGFACE_HUB_TOKEN",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--precision",
        choices=["INT4", "INT8", "FP16"],
        default="FP16",
        help="Requested export precision for config metadata (default: FP16)",
    )
    return parser.parse_args()


def normalize_needle_model_id(model_id: str | None) -> str:
    normalized = (model_id or "").strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    return normalized


def resolve_needle_checkpoint_file(
    model_id: str | None,
    checkpoint_file: str | None,
) -> str:
    if checkpoint_file:
        return checkpoint_file
    normalized = normalize_needle_model_id(model_id)
    if normalized in MODEL_ID_TO_CHECKPOINT_FILE:
        return MODEL_ID_TO_CHECKPOINT_FILE[normalized]
    if normalized.startswith("needle_") and normalized.endswith(".pkl"):
        return normalized
    return DEFAULT_CHECKPOINT_FILE


def resolve_checkpoint_revision(
    checkpoint_file: str,
    checkpoint_revision: str | None,
):
    if checkpoint_revision:
        return checkpoint_revision
    return CHECKPOINT_REVISIONS.get(checkpoint_file, None)


def checkpoint_file_to_weights_dir_name(checkpoint_file: str) -> str:
    stem = Path(checkpoint_file).stem.lower()
    match = re.fullmatch(r"needle_(\d+)_(\d+)_best", stem)
    if match:
        return f"needle-{match.group(1)}-{match.group(2)}"
    return stem.replace("_", "-")


def resolve_needle_output_dir(
    model_id: str | None = None,
    checkpoint_file: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    normalized = normalize_needle_model_id(model_id)
    effective_checkpoint_file = resolve_needle_checkpoint_file(model_id, checkpoint_file)
    if normalized == "needle" or (not normalized and effective_checkpoint_file == DEFAULT_CHECKPOINT_FILE):
        return (PROJECT_ROOT / "weights" / "needle").resolve()
    return (PROJECT_ROOT / "weights" / checkpoint_file_to_weights_dir_name(effective_checkpoint_file)).resolve()


def format_config_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def get_hf_token(explicit: str | None):
    if explicit:
        return explicit
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def require_hf_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required when local Needle paths are not provided"
        ) from exc
    return hf_hub_download, snapshot_download


def find_model_file(path: str | Path):
    path = Path(path).expanduser()
    if path.is_file() and path.suffix == ".model":
        return path
    if path.is_dir():
        exact = path / DEFAULT_MODEL_FILE
        if exact.exists():
            return exact
        models = sorted(path.rglob("*.model"))
        if models:
            return models[0]
    return None


def get_checkpoint_path(
    local_path: str | None,
    checkpoint_file: str,
    checkpoint_revision: str | None,
    token: str | None,
    cache_dir: str | None,
):
    if local_path:
        path = Path(local_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    hf_hub_download, _ = require_hf_hub()
    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=checkpoint_file,
            repo_type="model",
            cache_dir=cache_dir,
            token=token,
            revision=checkpoint_revision,
        )
    )


def resolve_tokenizer_revision(
    checkpoint_file: str,
    tokenizer_revision: str | None,
):
    if tokenizer_revision:
        return tokenizer_revision
    return CHECKPOINT_TOKENIZER_REVISIONS.get(
        checkpoint_file,
        DEFAULT_TOKENIZER_REVISION,
    )


def get_tokenizer_path(
    local_path: str | None,
    tokenizer_revision: str | None,
    token: str | None,
    cache_dir: str | None,
):
    if local_path:
        path = find_model_file(local_path)
        if path:
            return path
        raise FileNotFoundError(f"Tokenizer model not found under: {local_path}")

    _, snapshot_download = require_hf_hub()
    snapshot = Path(
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            repo_type="dataset",
            cache_dir=cache_dir,
            allow_patterns=["*.model"],
            token=token,
            revision=tokenizer_revision,
        )
    )
    path = find_model_file(snapshot)
    if path:
        return path
    raise FileNotFoundError(f"No .model file found in tokenizer snapshot: {snapshot}")


def load_checkpoint(path: str | Path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Needle checkpoint payload type: {type(payload)!r}")
    config = payload.get("config")
    params = payload.get("params")
    if not isinstance(config, dict) or params is None:
        raise ValueError("Needle checkpoint is missing config or params")
    return params, dict(config)


def param_count(tree):
    if isinstance(tree, dict):
        return sum(param_count(value) for value in tree.values())
    if hasattr(tree, "numel"):
        return int(tree.numel())
    shape = getattr(tree, "shape", None)
    if shape is None:
        return 0
    count = 1
    for dim in shape:
        count *= int(dim)
    return int(count)


def take_layer(tree, index: int):
    if isinstance(tree, dict):
        return {key: take_layer(value, index) for key, value in tree.items()}
    return tree[index]


def read_varint(data: bytes, pos: int):
    shift = 0
    value = 0
    while True:
        if pos >= len(data):
            raise ValueError("Unexpected end of protobuf varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7


def skip_protobuf_value(data: bytes, pos: int, wire_type: int):
    if wire_type == 0:
        _, pos = read_varint(data, pos)
        return pos
    if wire_type == 1:
        return pos + 8
    if wire_type == 2:
        length, pos = read_varint(data, pos)
        return pos + length
    if wire_type == 5:
        return pos + 4
    raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def parse_sentencepiece_pieces(path: str | Path):
    data = Path(path).read_bytes()
    pos = 0
    pieces = []

    while pos < len(data):
        tag, pos = read_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 7

        if field_number != 1 or wire_type != 2:
            pos = skip_protobuf_value(data, pos, wire_type)
            continue

        message_len, pos = read_varint(data, pos)
        end = pos + message_len
        piece_message = data[pos:end]
        pos = end

        inner_pos = 0
        piece = None
        score = None
        while inner_pos < len(piece_message):
            inner_tag, inner_pos = read_varint(piece_message, inner_pos)
            inner_field = inner_tag >> 3
            inner_wire = inner_tag & 7
            if inner_field == 1 and inner_wire == 2:
                value_len, inner_pos = read_varint(piece_message, inner_pos)
                value_end = inner_pos + value_len
                piece = piece_message[inner_pos:value_end].decode(
                    "utf-8", errors="replace"
                )
                inner_pos = value_end
            elif inner_field == 2 and inner_wire == 5:
                value_end = inner_pos + 4
                if value_end > len(piece_message):
                    raise ValueError("Unexpected end of protobuf float field")
                score = struct.unpack("<f", piece_message[inner_pos:value_end])[0]
                inner_pos = value_end
            else:
                inner_pos = skip_protobuf_value(piece_message, inner_pos, inner_wire)

        if piece is None:
            raise ValueError("Failed to parse SentencePiece vocabulary entry")
        pieces.append({
            "piece": piece,
            "score": float(score) if score is not None else 0.0,
        })

    if not pieces:
        raise ValueError(f"No SentencePiece pieces found in tokenizer: {path}")
    return pieces


def build_tokenizer_metadata(pieces, model_max_length: int):
    piece_texts = [piece["piece"] for piece in pieces]
    piece_to_id = {piece: idx for idx, piece in enumerate(piece_texts)}
    pad_id = piece_to_id.get("<pad>", 0)
    eos_id = piece_to_id.get("</s>", 1)
    bos_id = piece_to_id.get("<s>", 2)
    unk_id = piece_to_id.get("<unk>", 3)

    special_tokens = {
        int(pad_id): "<pad>",
        int(eos_id): "</s>",
        int(bos_id): "<s>",
        int(unk_id): "<unk>",
    }

    additional_special_tokens = []
    for token in ("<tool_call>", "<tools>"):
        token_id = piece_to_id.get(token)
        if token_id is None:
            continue
        special_tokens[int(token_id)] = token
        additional_special_tokens.append({"token": token, "id": int(token_id)})

    return {
        "vocab_size": len(piece_texts),
        "pad_token_id": int(pad_id),
        "eos_token_id": int(eos_id),
        "bos_token_id": int(bos_id),
        "unk_token_id": int(unk_id),
        "model_max_length": int(model_max_length),
        "sp_model_type": "bpe",
        "sp_add_dummy_prefix": True,
        "sp_remove_extra_whitespaces": True,
        "sp_escape_whitespaces": True,
        "sp_byte_fallback": True,
        "special_tokens": special_tokens,
        "additional_special_tokens": additional_special_tokens,
    }


def build_model_config(model_cfg: dict, total_params: int, requested_precision: str):
    hidden_dim = int(model_cfg["d_model"])
    heads = int(model_cfg["num_heads"])
    decoder_layers = int(model_cfg["num_decoder_layers"])
    requested_precision = (requested_precision or "FP16").upper()
    compute_precision = "FP16" if requested_precision in ("INT4", "INT8") else requested_precision
    config = {
        "model_type": "needle",
        "model_variant": "default",
        "precision": compute_precision,
        "quantization": requested_precision,
        "vocab_size": int(model_cfg["vocab_size"]),
        "hidden_dim": hidden_dim,
        "num_layers": decoder_layers,
        "num_encoder_layers": int(model_cfg["num_encoder_layers"]),
        "num_decoder_layers": decoder_layers,
        "attention_heads": heads,
        "attention_kv_heads": int(model_cfg["num_kv_heads"]),
        "attention_head_dim": hidden_dim // max(1, heads),
        "ffn_intermediate_dim": int(model_cfg["d_ff"]),
        "context_length": int(model_cfg["max_seq_len"]),
        "rope_theta": float(model_cfg.get("rope_theta", 10000.0)),
        "layer_norm_eps": 1e-6,
        "pad_token_id": int(model_cfg.get("pad_token_id", 0)),
        "tie_word_embeddings": True,
        "dtype": str(model_cfg.get("dtype", "bfloat16")),
        "activation": str(model_cfg.get("activation", "swiglu")),
        "enc_hidden_act": str(model_cfg.get("activation", "swiglu")),
        "dec_hidden_act": str(model_cfg.get("activation", "swiglu")),
        "num_memory_slots": int(model_cfg.get("num_memory_slots", 0)),
        "n_mels": int(model_cfg.get("n_mels", 0)),
        "dropout_rate": float(model_cfg.get("dropout_rate", 0.0)),
        "contrastive_dim": int(model_cfg.get("contrastive_dim", 0)),
        "conv_kernel_size": int(model_cfg.get("conv_kernel_size", 0)),
        "enable_speech": bool(model_cfg.get("enable_speech", False)),
        "no_feedforward": bool(model_cfg.get("no_feedforward", False)),
        "architecture": "text-only encoder-decoder transformer",
        "uses_tied_embeddings_for_logits": True,
        "inference_input_format": "[query_tokens] + <tools> + [tools_json_tokens]",
        "decoder_starts_with": "</s>",
        "decoder_predicts_first": "<tool_call>",
        "param_count": int(total_params),
    }
    return config


def write_config_txt(output_dir: Path, config: dict):
    config_path = output_dir / "config.txt"
    with open(config_path, "w", encoding="utf-8") as handle:
        for key, value in config.items():
            handle.write(f"{key}={format_config_value(value)}\n")
    return config_path


def write_tokenizer_files(output_dir: Path, pieces, tokenizer_meta: dict):
    vocab_path = output_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as handle:
        for token_id, piece in enumerate(pieces):
            handle.write(f"{token_id}\t{piece['piece']}\t{piece['score']}\n")

    merges_path = output_dir / "merges.txt"
    with open(merges_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("#version: 0.2\n")

    special_tokens_path = output_dir / "special_tokens.json"
    with open(special_tokens_path, "w", encoding="utf-8") as handle:
        json.dump(tokenizer_meta, handle, indent=2, ensure_ascii=False)

    tokenizer_config_path = output_dir / "tokenizer_config.txt"
    with open(tokenizer_config_path, "w", encoding="utf-8") as handle:
        handle.write(f"vocab_size={tokenizer_meta['vocab_size']}\n")
        handle.write(f"eos_token_id={tokenizer_meta['eos_token_id']}\n")
        handle.write(f"pad_token_id={tokenizer_meta['pad_token_id']}\n")
        handle.write(f"bos_token_id={tokenizer_meta['bos_token_id']}\n")
        handle.write(f"unk_token_id={tokenizer_meta['unk_token_id']}\n")
        handle.write(f"model_max_length={tokenizer_meta['model_max_length']}\n")
        handle.write("tokenizer_type=sentencepiece\n")
        handle.write(f"sp_model_type={tokenizer_meta['sp_model_type']}\n")
        handle.write(
            f"sp_add_dummy_prefix={format_config_value(tokenizer_meta['sp_add_dummy_prefix'])}\n"
        )
        handle.write(
            "sp_remove_extra_whitespaces="
            f"{format_config_value(tokenizer_meta['sp_remove_extra_whitespaces'])}\n"
        )
        handle.write(
            f"sp_escape_whitespaces={format_config_value(tokenizer_meta['sp_escape_whitespaces'])}\n"
        )
        handle.write(
            f"sp_byte_fallback={format_config_value(tokenizer_meta['sp_byte_fallback'])}\n"
        )
        handle.write("has_chat_template=false\n")
        if tokenizer_meta["additional_special_tokens"]:
            handle.write("has_tool_support=true\n")
            handle.write(
                f"tool_token_count={len(tokenizer_meta['additional_special_tokens'])}\n"
            )

    return {
        "vocab_path": vocab_path,
        "merges_path": merges_path,
        "special_tokens_path": special_tokens_path,
        "tokenizer_config_path": tokenizer_config_path,
    }


def export_model_weights(output_dir: Path, params: dict, model_cfg: dict, requested_precision: str):
    precision = (requested_precision or "FP16").upper()
    stats = create_quantization_stats()
    saved_paths = []

    def save(filename: str, tensor, transpose: bool = False, tensor_precision: str | None = None):
        output_path = output_dir / filename
        save_tensor_with_header(
            tensor,
            output_path,
            precision=tensor_precision or precision,
            transpose=transpose,
            stats_tracker=stats,
            model_type="needle",
        )
        saved_paths.append(output_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        embedding = params["embedding"]["embedding"]
        save("token_embeddings.weights", embedding)
        # Needle ties embeddings for logits; export the explicit output matrix as well.
        save("output_weight.weights", embedding)

        save("encoder_layer_norm_weight.weights", params["encoder"]["final_norm"]["scale"])
        save("output_norm.weights", params["decoder"]["ZCRMSNorm_0"]["scale"])

        contrastive_proj = params.get("contrastive_proj", {}).get("kernel")
        if contrastive_proj is not None:
            save("contrastive_proj.weights", contrastive_proj, transpose=True)

        if "log_temp" in params:
            save("log_temp.weights", params["log_temp"], tensor_precision="FP16")

        encoder_stack = params["encoder"]["layers"]["EncoderBlock_0"]
        for layer_idx in range(int(model_cfg["num_encoder_layers"])):
            block = take_layer(encoder_stack, layer_idx)
            attn = block["self_attn"]
            prefix = f"encoder_layer_{layer_idx}_"

            save(prefix + "input_norm.weights", block["ZCRMSNorm_0"]["scale"])
            if "attn_gate" in block:
                save(prefix + "attn_gate.weights", block["attn_gate"])
            if "ZCRMSNorm_1" in block:
                save(prefix + "post_attn_norm.weights", block["ZCRMSNorm_1"]["scale"])
            save(prefix + "attn_q.weights", attn["q_proj"]["kernel"], transpose=True)
            save(prefix + "attn_k.weights", attn["k_proj"]["kernel"], transpose=True)
            save(prefix + "attn_v.weights", attn["v_proj"]["kernel"], transpose=True)
            save(prefix + "attn_output.weights", attn["out_proj"]["kernel"], transpose=True)
            save(prefix + "attn_q_norm.weights", attn["q_norm"]["scale"])
            save(prefix + "attn_k_norm.weights", attn["k_norm"]["scale"])
            ffn = block.get("FeedForward_0")
            if ffn is not None:
                save(prefix + "ffn_gate.weights", ffn["gate_proj"]["kernel"], transpose=True)
                save(prefix + "ffn_up.weights", ffn["up_proj"]["kernel"], transpose=True)
                save(prefix + "mlp_fc2.weights", ffn["down_proj"]["kernel"], transpose=True)

        decoder_stack = params["decoder"]["layers"]["DecoderBlock_0"]
        for layer_idx in range(int(model_cfg["num_decoder_layers"])):
            block = take_layer(decoder_stack, layer_idx)
            self_attn = block["self_attn"]
            cross_attn = block["cross_attn"]
            prefix = f"layer_{layer_idx}_"

            save(prefix + "input_norm.weights", block["ZCRMSNorm_0"]["scale"])
            save(prefix + "post_attn_norm.weights", block["ZCRMSNorm_1"]["scale"])
            if "self_attn_gate" in block:
                save(prefix + "self_attn_gate.weights", block["self_attn_gate"])
            if "cross_attn_gate" in block:
                save(prefix + "cross_attn_gate.weights", block["cross_attn_gate"])
            if "ZCRMSNorm_2" in block:
                save(prefix + "final_norm.weights", block["ZCRMSNorm_2"]["scale"])

            save(prefix + "attn_q.weights", self_attn["q_proj"]["kernel"], transpose=True)
            save(prefix + "attn_k.weights", self_attn["k_proj"]["kernel"], transpose=True)
            save(prefix + "attn_v.weights", self_attn["v_proj"]["kernel"], transpose=True)
            save(prefix + "attn_output.weights", self_attn["out_proj"]["kernel"], transpose=True)
            save(prefix + "attn_q_norm.weights", self_attn["q_norm"]["scale"])
            save(prefix + "attn_k_norm.weights", self_attn["k_norm"]["scale"])

            save(prefix + "encoder_attn_q.weights", cross_attn["q_proj"]["kernel"], transpose=True)
            save(prefix + "encoder_attn_k.weights", cross_attn["k_proj"]["kernel"], transpose=True)
            save(prefix + "encoder_attn_v.weights", cross_attn["v_proj"]["kernel"], transpose=True)
            save(prefix + "encoder_attn_output.weights", cross_attn["out_proj"]["kernel"], transpose=True)
            save(prefix + "encoder_attn_q_norm.weights", cross_attn["q_norm"]["scale"])
            save(prefix + "encoder_attn_k_norm.weights", cross_attn["k_norm"]["scale"])

            ffn = block.get("FeedForward_0")
            if ffn is not None:
                save(prefix + "ffn_gate.weights", ffn["gate_proj"]["kernel"], transpose=True)
                save(prefix + "ffn_up.weights", ffn["up_proj"]["kernel"], transpose=True)
                save(prefix + "mlp_fc2.weights", ffn["down_proj"]["kernel"], transpose=True)

    return {"saved_weight_paths": saved_paths, "stats": stats}


def export_needle_metadata(
    output_dir=None,
    model_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
    checkpoint_revision: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_revision: str | None = None,
    hf_token: str | None = None,
    cache_dir: str | None = None,
    requested_precision: str = "FP16",
):
    checkpoint_file = resolve_needle_checkpoint_file(model_id, checkpoint_file)
    checkpoint_revision = resolve_checkpoint_revision(checkpoint_file, checkpoint_revision)
    output_dir = resolve_needle_output_dir(
        model_id=model_id,
        checkpoint_file=checkpoint_file,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    token = get_hf_token(hf_token)
    tokenizer_revision = resolve_tokenizer_revision(checkpoint_file, tokenizer_revision)
    checkpoint_path = get_checkpoint_path(
        str(checkpoint_path) if checkpoint_path is not None else None,
        checkpoint_file,
        checkpoint_revision,
        token,
        cache_dir,
    )
    tokenizer_path = get_tokenizer_path(
        str(tokenizer_path) if tokenizer_path is not None else None,
        tokenizer_revision,
        token,
        cache_dir,
    )

    params, model_cfg = load_checkpoint(checkpoint_path)
    pieces = parse_sentencepiece_pieces(tokenizer_path)

    model_config = build_model_config(
        model_cfg,
        param_count(params),
        requested_precision=requested_precision,
    )
    tokenizer_meta = build_tokenizer_metadata(
        pieces,
        model_max_length=int(model_cfg.get("max_seq_len", 131072)),
    )
    model_weight_outputs = export_model_weights(
        output_dir,
        params,
        model_cfg,
        requested_precision=requested_precision,
    )

    config_path = write_config_txt(output_dir, model_config)
    tokenizer_outputs = write_tokenizer_files(output_dir, pieces, tokenizer_meta)

    return {
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "config_path": config_path,
        **model_weight_outputs,
        **tokenizer_outputs,
    }


def main():
    args = parse_args()
    result = export_needle_metadata(
        output_dir=args.output_dir,
        model_id=None,
        checkpoint_path=args.checkpoint_path,
        checkpoint_file=args.checkpoint_file,
        checkpoint_revision=args.checkpoint_revision,
        tokenizer_path=args.tokenizer_path,
        tokenizer_revision=args.tokenizer_revision,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        requested_precision=args.precision,
    )

    print(f"Saved Needle weights and metadata to {result['output_dir']}")
    print(f"  checkpoint: {result['checkpoint_path']}")
    print(f"  tokenizer:  {result['tokenizer_path']}")
    print(f"  config:     {result['config_path']}")
    print(f"  vocab:      {result['vocab_path']}")
    print(f"  tok_cfg:    {result['tokenizer_config_path']}")
    print(f"  weights:    {len(result['saved_weight_paths'])} files")
    print_quantization_summary(result["stats"])


if __name__ == "__main__":
    main()
