import json
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files, scan_cache_dir
except ImportError:
    hf_hub_download = None
    list_repo_files = None
    scan_cache_dir = None


def _find_sentencepiece_model(repo_id, token=None):
    local_path = Path(repo_id)
    if local_path.is_dir():
        if sp := next((f for f in local_path.iterdir() if f.suffix == '.model' and f.is_file()), None):
            return str(sp)
        return None

    sentencepiece_file = None
    if scan_cache_dir:
        try:
            if repo := next((r for r in scan_cache_dir().repos if r.repo_id == repo_id), None):
                latest = max(repo.revisions, key=lambda r: r.last_modified)
                sentencepiece_file = next((f.file_name for f in latest.files if f.file_name.endswith('.model') and '/' not in f.file_name), None)
        except Exception:
            pass
    if not sentencepiece_file and list_repo_files:
        try:
            repo_files = list_repo_files(repo_id, token=token)
            sentencepiece_file = next((f for f in repo_files if f.endswith('.model') and '/' not in f), None)
        except Exception:
            pass

    if sentencepiece_file and hf_hub_download:
        return hf_hub_download(repo_id=repo_id, filename=sentencepiece_file, token=token)

    return None


def convert_hf_tokenizer(tokenizer, output_dir, token=None):
    """Convert a HuggingFace tokenizer to Cactus format."""
    sentencepiece_tokenizer_model_path = _find_sentencepiece_model(tokenizer.name_or_path, token=token)

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

    vocab = tokenizer.get_vocab()

    id_to_token = [""] * len(vocab)
    for token_str, token_id in vocab.items():
        if token_id < len(id_to_token):
            id_to_token[token_id] = token_str

    vocab_output = output_dir / "vocab.txt"

    if sentencepiece_tokenizer_model_path:
        with open(vocab_output, 'w', encoding='utf-8') as f:
            for token_id, token_str in enumerate(id_to_token):
                if token_str:
                    f.write(f"{token_id}\t{token_str}\n")
        print(f"  Saved SentencePiece vocabulary (ID\\ttoken format)")
    else:
        with open(vocab_output, 'w', encoding='utf-8') as f:
            for token_str in id_to_token:
                f.write(token_str + '\n')
        print(f"  Saved BPE vocabulary (line-by-line format)")


    merges_output = output_dir / "merges.txt"

    def write_merges_file(merges_list):
        with open(merges_output, 'w', encoding='utf-8', newline='') as f:
            f.write("#version: 0.2\n")
            for merge in merges_list:
                f.write(f"{' '.join(merge)}\n")

    merges_written = False

    if not sentencepiece_tokenizer_model_path and tokenizer_json_data:
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

    model_name_l = getattr(tokenizer, 'name_or_path', '').lower()
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

    model_type = getattr(tokenizer, 'name_or_path', '').lower()
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
        if missing_tokens and sentencepiece_tokenizer_model_path:
            try:
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=sentencepiece_tokenizer_model_path)
                for token_str in missing_tokens:
                    token_id = sp.piece_to_id(token_str)
                    if token_id != sp.unk_id():
                        gemma_special_tokens[token_str] = token_id
                        special_tokens[token_id] = token_str
                        print(f"    Found Gemma special token via SentencePiece: {token_str} (ID: {token_id})")
            except Exception as e:
                print(f"    Warning: Could not check SentencePiece for Gemma tokens: {e}")

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

    tokenizer_full_config = {}
    added_tokens_decoder = {}
    tool_tokens = {}

    try:
        config_path = None
        if hasattr(tokenizer, 'name_or_path') and hf_hub_download:
            try:
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


    special_tokens_output = output_dir / "special_tokens.json"
    with open(special_tokens_output, 'w', encoding='utf-8') as f:
        json.dump({
            **special_token_ids,
            "vocab_size": len(vocab),
            "model_max_length": getattr(tokenizer, 'model_max_length', 131072),
            "special_tokens": special_tokens,
            "additional_special_tokens": additional_special_tokens,
            **chat_template_data
        }, f, indent=2, ensure_ascii=False)


    tokenizer_config_output = output_dir / "tokenizer_config.txt"
    with open(tokenizer_config_output, 'w') as f:
        f.write(f"vocab_size={len(vocab)}\n")
        for key, value in special_token_ids.items():
            f.write(f"{key}={value}\n")
        f.write(f"model_max_length={getattr(tokenizer, 'model_max_length', 131072)}\n")

        if sentencepiece_tokenizer_model_path:
            f.write("tokenizer_type=sentencepiece\n")
        else:
            f.write("tokenizer_type=bpe\n")

        if chat_template_data:
            f.write("has_chat_template=true\n")
        else:
            f.write("has_chat_template=false\n")
        if len(tool_tokens) > 0:
            f.write(f"has_tool_support=true\n")
            f.write(f"tool_token_count={len(tool_tokens)}\n")