import json
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

SENTENCEPIECE_MODEL_TYPES = {
    'gemma', 'gemma3n', 'llama', 'smol', 'bert', 't5',
}

SENTENCEPIECE_MODEL_TYPES = {
    'gemma', 'gemma3n', 'llama', 'gemma4', 'smol', 'bert', 't5',
}

BPE_MODEL_TYPES = {
    'qwen', 'qwen3_5', 'lfm2',
    'whisper', 'moonshine',
    'parakeet', 'parakeet_tdt',
}


def _is_metaspace_normalizer(normalizer):
    return (
        isinstance(normalizer, dict)
        and normalizer.get("type") == "Replace"
        and normalizer.get("pattern", {}).get("String") == " "
        and normalizer.get("content") == "▁"
    )


def _decoder_has_type(decoder, decoder_type):
    if not isinstance(decoder, dict):
        return False
    if decoder.get("type") == decoder_type:
        return True
    if decoder.get("type") == "Sequence":
        return any(_decoder_has_type(item, decoder_type) for item in decoder.get("decoders", []))
    return False


def _is_replace_metaspace_decoder(decoder):
    if not isinstance(decoder, dict):
        return False
    if decoder.get("type") == "Replace":
        return (
            decoder.get("pattern", {}).get("String") == "▁"
            and decoder.get("content") == " "
        )
    if decoder.get("type") == "Sequence":
        return any(_is_replace_metaspace_decoder(item) for item in decoder.get("decoders", []))
    return False


def convert_hf_tokenizer(tokenizer, output_dir, token=None, model_id=None, labels=None, model_type=None):
    """Convert a HuggingFace tokenizer to Cactus format."""
    model_name_l = (model_id or getattr(tokenizer, 'name_or_path', '') or '').lower()

    # Parakeet-TDT exports labels directly in config.json (8192 token classes).
    # HF tokenizer object for this repo exposes only a byte-level 256 vocab,
    # which breaks runtime decode. Prefer labels when provided.
    if labels and isinstance(labels, (list, tuple)) and 'parakeet-tdt' in model_name_l:
        id_to_token = [str(tok) for tok in labels]
        vocab_size = len(id_to_token)

        vocab_output = output_dir / "vocab.txt"
        with open(vocab_output, 'w', encoding='utf-8') as f:
            for token_id, token_str in enumerate(id_to_token):
                f.write(f"{token_id}\t{token_str}\n")
        print(f"  Saved Parakeet-TDT vocabulary from labels (ID\\ttoken format, {vocab_size} tokens)")

        merges_output = output_dir / "merges.txt"
        with open(merges_output, 'w', encoding='utf-8', newline='') as f:
            f.write("#version: 0.2\n")

        token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
        unk_id = token_to_id.get('<unk>', 0)
        pad_id = token_to_id.get('<pad>', getattr(tokenizer, 'pad_token_id', 0) or 0)
        eos_id = token_to_id.get('<|endoftext|>', getattr(tokenizer, 'eos_token_id', None))
        if eos_id is None:
            eos_id = pad_id
        bos_id = token_to_id.get('<|startoftranscript|>', None)

        special_token_ids = {
            'pad_token_id': int(pad_id),
            'unk_token_id': int(unk_id),
            'eos_token_id': int(eos_id),
        }
        if bos_id is not None:
            special_token_ids['bos_token_id'] = int(bos_id)

        # Keep special token map dense for known control tags in labels.
        special_tokens = {}
        for token_id, tok in enumerate(id_to_token):
            if tok.startswith('<') and tok.endswith('>'):
                special_tokens[token_id] = tok
        # Ensure core IDs are present even if token strings differ.
        special_tokens.setdefault(int(pad_id), id_to_token[int(pad_id)] if int(pad_id) < vocab_size else "<pad>")
        special_tokens.setdefault(int(unk_id), id_to_token[int(unk_id)] if int(unk_id) < vocab_size else "<unk>")
        special_tokens.setdefault(int(eos_id), id_to_token[int(eos_id)] if int(eos_id) < vocab_size else "<|endoftext|>")

        special_tokens_output = output_dir / "special_tokens.json"
        with open(special_tokens_output, 'w', encoding='utf-8') as f:
            json.dump({
                **special_token_ids,
                "vocab_size": vocab_size,
                "model_max_length": getattr(tokenizer, 'model_max_length', 131072),
                "special_tokens": special_tokens,
                "additional_special_tokens": [],
            }, f, indent=2, ensure_ascii=False)

        tokenizer_config_output = output_dir / "tokenizer_config.txt"
        with open(tokenizer_config_output, 'w') as f:
            f.write(f"vocab_size={vocab_size}\n")
            for key, value in special_token_ids.items():
                f.write(f"{key}={value}\n")
            f.write(f"model_max_length={getattr(tokenizer, 'model_max_length', 131072)}\n")
            f.write("tokenizer_type=sentencepiece\n")
            f.write("has_chat_template=false\n")
        return

    tokenizer_json_data = {}
    tokenizer_json_path = output_dir / "tokenizer.json"
    try:
        tokenizer.save_pretrained(output_dir)
        if tokenizer_json_path.exists():
            with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
                tokenizer_json_data = json.load(f)

        unused_files = [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
        ]
        for filename in unused_files:
            filepath = output_dir / filename
            if filepath.exists():
                filepath.unlink()
    except Exception as e:
        print(f"  Warning: Could not save tokenizer JSON: {e}")

    tokenizer_model = tokenizer_json_data.get("model", {}) if tokenizer_json_data else {}
    tokenizer_model_type = str(tokenizer_model.get("type", "")).upper()
    is_sentencepiece = tokenizer_model_type != "BPE" and model_type in SENTENCEPIECE_MODEL_TYPES

    if tokenizer_model_type == "BPE" and tokenizer_model.get("vocab"):
        vocab = tokenizer_model["vocab"]
        vocab_size = max(vocab.values()) + 1
        id_to_token = [""] * vocab_size
        for token_str, token_id in vocab.items():
            if token_id < vocab_size:
                id_to_token[token_id] = token_str
    else:
        vocab = tokenizer.get_vocab()
        id_to_token = [""] * len(vocab)
        for token_str, token_id in vocab.items():
            if token_id < len(id_to_token):
                id_to_token[token_id] = token_str

    # vocab.txt is written later, after special tokens are collected


    merges_output = output_dir / "merges.txt"

    def write_merges_file(merges_list):
        with open(merges_output, 'w', encoding='utf-8', newline='') as f:
            f.write("#version: 0.2\n")
            for merge in merges_list:
                f.write(f"{' '.join(merge)}\n")

    merges_written = False

    if not is_sentencepiece and tokenizer_json_data:
        merges_from_json = tokenizer_json_data.get("model", {}).get("merges", []) or []
        write_merges_file(merges_from_json)
        merges_written = True

    if not merges_written and hf_hub_download:
        try:
            import shutil
            merges_file = hf_hub_download(repo_id=tokenizer.name_or_path, filename="merges.txt", token=token)
            shutil.copy2(merges_file, merges_output)
            merges_written = True
        except Exception:
            pass

    if not merges_written and hasattr(tokenizer, 'backend_tokenizer') and tokenizer.backend_tokenizer:
        backend = tokenizer.backend_tokenizer
        merges = []

        if hasattr(backend, 'model'):
            model = backend.model
            if hasattr(model, 'merges'):
                merges = model.merges

        write_merges_file(merges)
        merges_written = True

    if not merges_written:
        write_merges_file([])


    special_tokens = {}
    special_token_ids = {}

    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_token_ids['eos_token_id'] = tokenizer.eos_token_id
        special_tokens[tokenizer.eos_token_id] = tokenizer.eos_token or "<|endoftext|>"

    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        special_token_ids['pad_token_id'] = tokenizer.pad_token_id
        special_tokens[tokenizer.pad_token_id] = tokenizer.pad_token or "<|endoftext|>"

    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_token_ids['bos_token_id'] = tokenizer.bos_token_id
        special_tokens[tokenizer.bos_token_id] = tokenizer.bos_token or "<|startoftext|>"

    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        special_token_ids['unk_token_id'] = tokenizer.unk_token_id
        special_tokens[tokenizer.unk_token_id] = tokenizer.unk_token or "<|unknown|>"

    core_token_fallbacks = {
        'pad_token_id': '<pad>',
        'eos_token_id': '<eos>',
        'bos_token_id': '<bos>',
        'unk_token_id': '<unk>',
    }
    vocab_lookup = {token: token_id for token_id, token in enumerate(id_to_token) if token}
    for key, token_str in core_token_fallbacks.items():
        if key not in special_token_ids and token_str in vocab_lookup:
            token_id = vocab_lookup[token_str]
            special_token_ids[key] = token_id
            special_tokens[token_id] = token_str

    if 'eos_token_id' not in special_token_ids and 'parakeet' in model_name_l:
        pad_id = special_token_ids.get('pad_token_id')
        if pad_id is not None:
            special_token_ids['eos_token_id'] = pad_id

    additional_special_tokens = []
    if hasattr(tokenizer, 'additional_special_tokens'):
        for token_str in tokenizer.additional_special_tokens or []:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id != tokenizer.unk_token_id:
                special_tokens[token_id] = token_str
                additional_special_tokens.append({"token": token_str, "id": token_id})

    for token_info in tokenizer_json_data.get("added_tokens", []) or []:
        token_str = token_info.get("content")
        token_id = token_info.get("id")
        if token_str is None or token_id is None:
            continue
        special_tokens[int(token_id)] = token_str
        if not any(item["token"] == token_str and item["id"] == int(token_id) for item in additional_special_tokens):
            additional_special_tokens.append({"token": token_str, "id": int(token_id)})

    model_type = model_name_l or getattr(tokenizer, 'name_or_path', '').lower()
    if 'gemma' in model_type:
        gemma_special_tokens = {
            '<start_of_turn>': None,
            '<end_of_turn>': None,
            '<start_of_image>': None,
            '<end_of_image>': None,
            # Gemma 3 function calling tokens
            '<start_function_declaration>': None,
            '<end_function_declaration>': None,
            '<start_function_call>': None,
            '<end_function_call>': None,
            '<start_function_response>': None,
            '<end_function_response>': None,
            '<escape>': None
        }

        vocab = tokenizer.get_vocab()
        for token_str in gemma_special_tokens.keys():
            if token_str in vocab:
                token_id = vocab[token_str]
                gemma_special_tokens[token_str] = token_id
                special_tokens[token_id] = token_str
                print(f"    Found Gemma special token: {token_str} (ID: {token_id})")

        missing_tokens = [k for k, v in gemma_special_tokens.items() if v is None]
        if missing_tokens:
            unk_id = getattr(tokenizer, 'unk_token_id', None)
            for token_str in missing_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                if token_id != unk_id and token_id is not None:
                    gemma_special_tokens[token_str] = token_id
                    special_tokens[token_id] = token_str
                    print(f"    Found Gemma special token: {token_str} (ID: {token_id})")

        if gemma_special_tokens['<start_of_turn>'] is None:
            hardcoded_ids = {
                '<start_of_turn>': 105,
                '<end_of_turn>': 106
            }
            for token_str, token_id in hardcoded_ids.items():
                if token_str in gemma_special_tokens and gemma_special_tokens[token_str] is None:
                    if token_id not in special_tokens:
                        gemma_special_tokens[token_str] = token_id
                        special_tokens[token_id] = token_str
                        print(f"    Using hardcoded Gemma special token: {token_str} (ID: {token_id})")

    chat_template_data = {}
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        chat_template_output = output_dir / "chat_template.jinja2"
        with open(chat_template_output, 'w', encoding='utf-8') as f:
            f.write(tokenizer.chat_template)
        chat_template_data["chat_template"] = tokenizer.chat_template
    elif (output_dir / "chat_template.jinja2").exists():
        chat_template_data["chat_template"] = (output_dir / "chat_template.jinja2").read_text(encoding='utf-8')

    tokenizer_full_config = {}
    added_tokens_decoder = {}
    tool_tokens = {}

    try:
        config_path = None
        if hasattr(tokenizer, 'name_or_path') and hf_hub_download:
            try:
                local_candidate = Path(tokenizer.name_or_path) / "tokenizer_config.json"
                if Path(tokenizer.name_or_path).is_dir() and local_candidate.exists():
                    config_path = str(local_candidate)
                else:
                    config_path = hf_hub_download(repo_id=tokenizer.name_or_path, filename="tokenizer_config.json", token=token)
                with open(config_path, 'r') as f:
                    tokenizer_full_config = json.load(f)

                    if 'chat_template' in tokenizer_full_config and not chat_template_data:
                        chat_template_output = output_dir / "chat_template.jinja2"
                        with open(chat_template_output, 'w', encoding='utf-8') as f:
                            f.write(tokenizer_full_config['chat_template'])
                        chat_template_data["chat_template"] = tokenizer_full_config['chat_template']

                    if 'added_tokens_decoder' in tokenizer_full_config:
                        added_tokens_decoder = tokenizer_full_config['added_tokens_decoder']

                        print("  Extracting special tokens from tokenizer_config.json...")
                        for token_id_str, token_info in added_tokens_decoder.items():
                            content = token_info.get('content', '')
                            token_id = int(token_id_str)

                            tool_related = ['<tool_call>', '</tool_call>',
                                          '<tool_response>', '</tool_response>',
                                          '<tools>', '</tools>',
                                          '<think>', '</think>',
                                          # Gemma 3 function calling tokens
                                          '<start_function_declaration>', '<end_function_declaration>',
                                          '<start_function_call>', '<end_function_call>',
                                          '<start_function_response>', '<end_function_response>',
                                          '<escape>']

                            if any(x == content for x in tool_related):
                                tool_tokens[token_id] = token_info
                                print(f"    Found tool token: {content} (ID: {token_id})")
                                special_tokens[token_id] = content

            except Exception as e:
                print(f"  Note: Could not load full tokenizer config: {e}")
                pass
    except Exception:
        pass


    # Extend id_to_token to include special/added tokens so vocab.txt is self-contained
    if special_tokens:
        max_special_id = max(special_tokens.keys())
        if max_special_id >= len(id_to_token):
            id_to_token.extend([""] * (max_special_id - len(id_to_token) + 1))
        for token_id, token_str in special_tokens.items():
            id_to_token[token_id] = token_str

    vocab_output = output_dir / "vocab.txt"
    with open(vocab_output, 'w', encoding='utf-8') as f:
        for token_id, token_str in enumerate(id_to_token):
            if token_str:
                f.write(f"{token_id}\t{token_str}\n")
    print(f"  Saved tokenizer vocabulary (ID\\ttoken format)")

    special_tokens_output = output_dir / "special_tokens.json"
    with open(special_tokens_output, 'w', encoding='utf-8') as f:
        json.dump({
            **special_token_ids,
            "vocab_size": len(id_to_token),
            "model_max_length": getattr(tokenizer, 'model_max_length', 131072),
            "special_tokens": special_tokens,
            "additional_special_tokens": additional_special_tokens,
            **chat_template_data
        }, f, indent=2, ensure_ascii=False)

    normalizer = "none"
    decoder = "none"
    byte_fallback = False
    vocab_format = "id_tab_token"
    tokenizer_type = "sentencepiece" if is_sentencepiece else "bpe"

    if tokenizer_model_type == "BPE":
        tokenizer_type = "bpe"
        byte_fallback = bool(tokenizer_model.get("byte_fallback", False))
        if _is_metaspace_normalizer(tokenizer_json_data.get("normalizer")):
            normalizer = "metaspace"
        elif _decoder_has_type(tokenizer_json_data.get("decoder"), "ByteFallback"):
            normalizer = "byte_level"

        if _is_replace_metaspace_decoder(tokenizer_json_data.get("decoder")):
            decoder = "replace_metaspace"
        elif _decoder_has_type(tokenizer_json_data.get("decoder"), "ByteFallback"):
            decoder = "byte_level"
    elif is_sentencepiece:
        tokenizer_type = "sentencepiece"

    tokenizer_config_output = output_dir / "tokenizer_config.txt"
    with open(tokenizer_config_output, 'w') as f:
        f.write(f"vocab_size={len(id_to_token)}\n")
        for key, value in special_token_ids.items():
            f.write(f"{key}={value}\n")
        f.write(f"model_max_length={getattr(tokenizer, 'model_max_length', 131072)}\n")
        f.write(f"tokenizer_type={tokenizer_type}\n")
        f.write(f"vocab_format={vocab_format}\n")
        f.write(f"normalizer={normalizer}\n")
        f.write(f"decoder={decoder}\n")
        f.write(f"byte_fallback={'true' if byte_fallback else 'false'}\n")

        if chat_template_data:
            f.write("has_chat_template=true\n")
        else:
            f.write("has_chat_template=false\n")
        if len(tool_tokens) > 0:
            f.write(f"has_tool_support=true\n")
            f.write(f"tool_token_count={len(tool_tokens)}\n")


def _read_varint(data, pos):
    shift, value = 0, 0
    while True:
        byte = data[pos]; pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80): return value, pos
        shift += 7


def _skip_proto(data, pos, wire_type):
    if wire_type == 0: _, pos = _read_varint(data, pos); return pos
    if wire_type == 1: return pos + 8
    if wire_type == 2: length, pos = _read_varint(data, pos); return pos + length
    if wire_type == 5: return pos + 4
    raise ValueError(f"Unsupported wire type: {wire_type}")


def parse_sentencepiece_pieces(path):
    import struct
    data = Path(path).read_bytes()
    pos, pieces = 0, []
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        if (tag >> 3) != 1 or (tag & 7) != 2:
            pos = _skip_proto(data, pos, tag & 7); continue
        msg_len, pos = _read_varint(data, pos)
        end = pos + msg_len; msg = data[pos:end]; pos = end
        inner_pos, piece, score = 0, None, 0.0
        while inner_pos < len(msg):
            itag, inner_pos = _read_varint(msg, inner_pos)
            ifield, iwire = itag >> 3, itag & 7
            if ifield == 1 and iwire == 2:
                vlen, inner_pos = _read_varint(msg, inner_pos)
                piece = msg[inner_pos:inner_pos + vlen].decode("utf-8", errors="replace")
                inner_pos += vlen
            elif ifield == 2 and iwire == 5:
                score = struct.unpack("<f", msg[inner_pos:inner_pos + 4])[0]
                inner_pos += 4
            else:
                inner_pos = _skip_proto(msg, inner_pos, iwire)
        if piece is not None:
            pieces.append({"piece": piece, "score": float(score)})
    return pieces


def _build_sentencepiece_metadata(pieces, model_max_length):
    piece_to_id = {p["piece"]: i for i, p in enumerate(pieces)}
    pad_id = piece_to_id.get("<pad>", 0)
    eos_id = piece_to_id.get("</s>", 1)
    bos_id = piece_to_id.get("<s>", 2)
    unk_id = piece_to_id.get("<unk>", 3)

    special = {pad_id: "<pad>", eos_id: "</s>", bos_id: "<s>", unk_id: "<unk>"}
    tool_tokens = []
    for tok in ("<tool_call>", "<tools>"):
        tid = piece_to_id.get(tok)
        if tid is not None:
            special[tid] = tok
            tool_tokens.append({"token": tok, "id": tid})

    return {
        "vocab_size": len(pieces), "pad_token_id": pad_id, "eos_token_id": eos_id,
        "bos_token_id": bos_id, "unk_token_id": unk_id, "model_max_length": model_max_length,
        "sp_model_type": "bpe", "sp_add_dummy_prefix": True,
        "sp_remove_extra_whitespaces": True, "sp_escape_whitespaces": True,
        "sp_byte_fallback": True, "special_tokens": {str(k): v for k, v in special.items()},
        "additional_special_tokens": tool_tokens,
    }


def _write_sentencepiece_files(output_dir, pieces, meta):
    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(pieces):
            f.write(f"{i}\t{p['piece']}\t{p['score']}\n")

    with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")

    with open(output_dir / "special_tokens.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    tool_tokens = meta.get("additional_special_tokens", [])
    with open(output_dir / "tokenizer_config.txt", "w", encoding="utf-8") as f:
        f.write(f"vocab_size={meta['vocab_size']}\n")
        f.write(f"eos_token_id={meta['eos_token_id']}\npad_token_id={meta['pad_token_id']}\n")
        f.write(f"bos_token_id={meta['bos_token_id']}\nunk_token_id={meta['unk_token_id']}\n")
        f.write(f"model_max_length={meta['model_max_length']}\n")
        f.write("tokenizer_type=sentencepiece\nsp_model_type=bpe\n")
        f.write("sp_add_dummy_prefix=true\nsp_remove_extra_whitespaces=true\n")
        f.write("sp_escape_whitespaces=true\nsp_byte_fallback=true\n")
        f.write("has_chat_template=false\n")
        if tool_tokens:
            f.write(f"has_tool_support=true\ntool_token_count={len(tool_tokens)}\n")


def convert_sentencepiece_tokenizer(tokenizer_path, output_dir, model_max_length=131072):
    output_dir = Path(output_dir)
    pieces = parse_sentencepiece_pieces(tokenizer_path)
    meta = _build_sentencepiece_metadata(pieces, model_max_length)
    _write_sentencepiece_files(output_dir, pieces, meta)
