#!/usr/bin/env python3
import argparse
import copy
import inspect
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.generation.candidate_generator import (
    _prepare_attention_mask,
    _prepare_position_ids,
    _prepare_token_type_ids,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList


TARGET_MODEL_ID = "google/gemma-4-E2B-it"
ASSISTANT_MODEL_ID = "google/gemma-4-E2B-it-assistant"
PROMPT_SUITE = [
    ("short_chat", "who are you"),
    ("desert_sentence", "Write one short sentence about desert rain."),
    ("json_object", "Return a JSON object with keys name, role, and status."),
    ("python_function", "Write a short Python function that adds two numbers."),
    ("count_1_to_100", "Count from 1 to 100, separated by commas."),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare HF Gemma 4 MTP with a local manual MTP loop.")
    parser.add_argument("prompt", nargs="?", default="Write one short sentence about speculative decoding.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--trace", action="store_true", help="Print round-by-round manual MTP trace rows.")
    parser.add_argument("--num-assistant-tokens", type=int, default=None)
    parser.add_argument("--acceptance-suite", action="store_true", help="Print greedy acceptance totals for the prompt suite.")
    return parser.parse_args()


def clone_inputs(inputs):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in inputs.items()}


def prepare_generation(model, inputs, assistant_model, max_new_tokens, do_sample, temperature, top_p=None, top_k=None):
    kwargs = clone_inputs(inputs)
    kwargs.update(
        {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k

    generation_config, model_kwargs = model._prepare_generation_config(None, **kwargs)
    logits_processor = LogitsProcessorList()
    stopping_criteria = StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        None, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]
    device = inputs_tensor.device
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    if not kwargs_has_attention_mask and not model.config.is_encoder_decoder and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("attention_mask passed to generate must be 2D.")

    kwargs_has_position_ids = model_kwargs.get("position_ids", None) is not None
    accepts_position_ids = "position_ids" in set(inspect.signature(model.forward).parameters.keys())
    if not kwargs_has_position_ids and accepts_position_ids and not model.config.is_encoder_decoder:
        model_kwargs["position_ids"] = model._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )

    input_ids_length = input_ids.shape[1]
    generation_config = model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=False,
        has_default_min_length=True,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    if model._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        model_kwargs["logits_to_keep"] = 1
    model_kwargs["use_cache"] = generation_config.use_cache

    generation_mode = generation_config.get_generation_mode(assistant_model)
    max_cache_length = generation_config.max_length - 1
    model._prepare_cache_for_generation(
        generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
    )

    return generation_config, input_ids, model_kwargs, logits_processor, stopping_criteria


def eos_token_ids(generation_config, assistant_generation_config, device):
    ids = set()
    for value in (generation_config.eos_token_id, assistant_generation_config.eos_token_id):
        if value is None:
            continue
        if isinstance(value, int):
            ids.add(value)
        else:
            ids.update(value)
    return torch.tensor(list(ids), dtype=torch.long, device=device) if ids else None


def manual_draft(target_model, assistant_model, input_ids, model_kwargs, outputs, n_matches, generation_config):
    max_new_tokens = min(
        int(assistant_model.generation_config.num_assistant_tokens),
        generation_config.max_length - input_ids.shape[1] - 1,
    )
    if max_new_tokens <= 0:
        return input_ids, None

    current_length = input_ids.shape[1]
    last_hidden_state = outputs.hidden_states[-1][:, n_matches : n_matches + 1]
    shared_kv_states = {
        key: (value[0][:, :, :current_length, :], value[1][:, :, :current_length, :])
        for key, value in outputs.shared_kv_states.items()
    }
    last_token_id = input_ids[:, -1:]
    position_ids = torch.tensor([[input_ids.shape[1] - 1]], dtype=torch.long, device=assistant_model.device)
    stopped = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    eos_ids = eos_token_ids(generation_config, assistant_model.generation_config, last_token_id.device)

    drafted_logits = []
    drafted_tokens = []
    for _ in range(max_new_tokens):
        last_token_embedding = target_model.get_input_embeddings()(last_token_id)
        inputs_embeds = torch.cat([last_token_embedding, last_hidden_state], dim=-1)
        with torch.no_grad():
            assistant_outputs = assistant_model(
                inputs_embeds=inputs_embeds,
                attention_mask=model_kwargs.get("attention_mask"),
                position_ids=position_ids,
                shared_kv_states=shared_kv_states,
                use_cache=False,
            )

        last_token_id = assistant_outputs.logits.argmax(dim=-1)
        last_hidden_state = assistant_outputs.last_hidden_state
        if stopped.any():
            stopped_expanded = stopped.unsqueeze(1)
            last_token_id = torch.where(stopped_expanded, generation_config.pad_token_id, last_token_id)
            drafted_logits.append(
                torch.where(stopped_expanded.unsqueeze(-1), torch.zeros_like(assistant_outputs.logits), assistant_outputs.logits)
            )
        else:
            drafted_logits.append(assistant_outputs.logits)

        drafted_tokens.append(last_token_id)
        if eos_ids is not None:
            stopped = torch.logical_or(stopped, torch.isin(last_token_id.squeeze(1), eos_ids))
            if stopped.all():
                break

    return torch.cat([input_ids, torch.cat(drafted_tokens, dim=1)], dim=1), torch.cat(drafted_logits, dim=1)


def speculative_sample(candidate_input_ids, candidate_logits, candidate_length, new_logits, is_done_candidate):
    candidate_new_ids = candidate_input_ids[:, -candidate_length:]
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), candidate_new_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), candidate_new_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i
    is_accepted = torch.rand_like(probability_ratio) <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()

    if is_done_candidate and n_matches == candidate_length:
        n_matches -= 1
        valid_tokens = candidate_new_ids[:, : n_matches + 1]
    else:
        gamma = candidate_logits.shape[1]
        p_next = p[:, n_matches, :]
        if n_matches < gamma:
            q_next = q[:, n_matches, :]
            p_prime = torch.clamp((p_next - q_next), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_next
        next_token = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
        valid_tokens = torch.cat((candidate_new_ids[:, :n_matches], next_token), dim=-1) if n_matches > 0 else next_token

    return valid_tokens, n_matches


def ids_for_trace(tensor):
    if tensor is None:
        return []
    if tensor.ndim == 2:
        tensor = tensor[0]
    return [int(x) for x in tensor.detach().cpu().tolist()]


def manual_generate(model, assistant_model, inputs, max_new_tokens, do_sample, temperature,
                    top_p=None, top_k=None, trace=False, stats=None):
    generation_config, input_ids, model_kwargs, logits_processor, stopping_criteria = prepare_generation(
        model, inputs, assistant_model, max_new_tokens, do_sample, temperature, top_p, top_k
    )
    if not model_kwargs["use_cache"]:
        raise ValueError("Manual assisted generation requires use_cache=True.")

    outputs = None
    n_matches = 0
    is_first_iteration = True

    round_index = 0
    while input_ids.shape[1] < generation_config.max_length:
        cur_len = input_ids.shape[1]
        previous_n_matches = int(n_matches)
        if is_first_iteration:
            candidate_input_ids, candidate_logits = input_ids, None
        else:
            candidate_input_ids, candidate_logits = manual_draft(
                model, assistant_model, input_ids, model_kwargs, outputs, n_matches, generation_config
            )
        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        is_done_candidate = stopping_criteria(candidate_input_ids, None)
        eos_ids = generation_config._eos_token_tensor
        if eos_ids is not None and candidate_length > 0:
            is_done_candidate = is_done_candidate or torch.isin(
                candidate_input_ids[:, -1], eos_ids.to(candidate_input_ids.device)
            ).all()

        candidate_kwargs = copy.copy(model_kwargs)
        candidate_kwargs = _prepare_attention_mask(
            candidate_kwargs, candidate_input_ids.shape[1], model.config.is_encoder_decoder
        )
        candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
        if (position_ids := candidate_kwargs.get("position_ids")) is not None and candidate_length > 0:
            new_length = candidate_length + position_ids.shape[-1]
            candidate_kwargs = _prepare_position_ids(candidate_kwargs, new_length, model.config.is_encoder_decoder)

        next_sequence_length = candidate_length + 1 if not is_first_iteration else None
        model_inputs = model.prepare_inputs_for_generation(
            candidate_input_ids,
            next_sequence_length=next_sequence_length,
            is_first_iteration=is_first_iteration,
            **candidate_kwargs,
        )
        if "logits_to_keep" in model_inputs:
            model_inputs["logits_to_keep"] = candidate_length + 1
        model_inputs["output_hidden_states"] = True
        model_inputs["return_shared_kv_states"] = True

        with torch.no_grad():
            outputs = model(**model_inputs)

        new_logits = outputs.logits[:, -candidate_length - 1 :].to(dtype=torch.float32, device=input_ids.device)
        for i in range(candidate_length + 1):
            new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

        if do_sample and candidate_logits is not None:
            valid_tokens, n_matches = speculative_sample(
                candidate_input_ids, candidate_logits.to(input_ids.device), candidate_length, new_logits, is_done_candidate
            )
        else:
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
            else:
                selected_tokens = new_logits.argmax(dim=-1)

            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
            if is_done_candidate and n_matches == candidate_length:
                n_matches -= 1
            valid_tokens = selected_tokens[:, : n_matches + 1]

        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        if not is_first_iteration:
            round_index += 1
            if stats is not None:
                stats.append({
                    "drafted": int(candidate_length),
                    "accepted": int(n_matches),
                    "rejected": int(n_matches) < int(candidate_length),
                })
        if trace and not is_first_iteration:
            print(
                "[hf_trace] "
                f"round={round_index} "
                f"input_len={cur_len} "
                f"previous_n_matches={previous_n_matches} "
                f"assistant_position_id={cur_len - 1} "
                f"assistant_input_token={int(candidate_input_ids[0, cur_len - 1])} "
                f"assistant_draft_tokens={ids_for_trace(candidate_input_ids[:, cur_len:])} "
                f"target_verify_input_tokens={ids_for_trace(candidate_input_ids[:, cur_len - 1:])} "
                f"target_argmax_tokens={ids_for_trace(new_logits.argmax(dim=-1))} "
                f"accepted_count={int(n_matches)} "
                f"valid_committed_tokens={ids_for_trace(valid_tokens)} "
                f"target_cache_crop_length={input_ids.shape[1] - 1}"
            )
        outputs.past_key_values.crop(input_ids.shape[1] - 1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
            num_new_tokens=n_matches + 1,
        )
        is_first_iteration = False

        eos_ids = generation_config._eos_token_tensor
        if eos_ids is not None and torch.isin(input_ids[:, -1], eos_ids.to(input_ids.device)).all():
            break

    return input_ids


def builtin_generate(model, assistant_model, inputs, max_new_tokens, do_sample, temperature, top_p=None, top_k=None):
    kwargs = {
        **clone_inputs(inputs),
        "assistant_model": assistant_model,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    return model.generate(**kwargs)


def run_case(label, model, assistant_model, processor, inputs, max_new_tokens, seed,
             do_sample, temperature, top_p, top_k, trace):
    torch.manual_seed(seed)
    builtin_ids = builtin_generate(model, assistant_model, inputs, max_new_tokens, do_sample, temperature, top_p, top_k)
    torch.manual_seed(seed)
    manual_ids = manual_generate(
        model, assistant_model, inputs, max_new_tokens, do_sample, temperature, top_p, top_k, trace)

    input_len = inputs["input_ids"].shape[-1]
    builtin_new = builtin_ids[0, input_len:].tolist()
    manual_new = manual_ids[0, input_len:].tolist()
    print(f"\n{label}")
    print(f"match: {builtin_new == manual_new}")
    print(f"builtin ids: {builtin_new}")
    print(f"manual  ids: {manual_new}")
    print(f"builtin: {processor.decode(builtin_ids[0, input_len:], skip_special_tokens=True)!r}")
    print(f"manual:  {processor.decode(manual_ids[0, input_len:], skip_special_tokens=True)!r}")
    if builtin_new != manual_new:
        raise AssertionError(f"{label} manual MTP output did not match built-in HF assisted generation")


def prompt_inputs(processor, model, prompt):
    messages = [{"role": "user", "content": prompt}]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def run_acceptance_suite(model, assistant_model, processor, max_new_tokens, seed):
    print("source,prompt,draft,max_new_tokens,drafted,accepted,rejected,rounds,acceptance_rate")
    for name, prompt in PROMPT_SUITE:
        inputs = prompt_inputs(processor, model, prompt)
        stats = []
        torch.manual_seed(seed)
        manual_generate(model, assistant_model, inputs, max_new_tokens, False, None, None, None, False, stats)
        drafted = sum(row["drafted"] for row in stats)
        accepted = sum(row["accepted"] for row in stats)
        rejected = sum(1 for row in stats if row["rejected"])
        rounds = len(stats)
        rate = accepted / drafted if drafted else 0.0
        print(
            f"hf,{name},{int(assistant_model.generation_config.num_assistant_tokens)},"
            f"{max_new_tokens},{drafted},{accepted},{rejected},{rounds},{rate:.4f}"
        )


def main():
    args = parse_args()
    print(f"target_model_id={TARGET_MODEL_ID}", file=sys.stderr)
    print(f"assistant_model_id={ASSISTANT_MODEL_ID}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    assistant_model = AutoModelForCausalLM.from_pretrained(ASSISTANT_MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    if args.num_assistant_tokens is not None:
        assistant_model.generation_config.num_assistant_tokens = args.num_assistant_tokens
    if args.acceptance_suite:
        run_acceptance_suite(model, assistant_model, processor, args.max_new_tokens, args.seed)
        return
    messages = [{"role": "user", "content": args.prompt}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    run_case("temperature=0", model, assistant_model, processor, inputs, args.max_new_tokens,
             args.seed, False, None, None, None, args.trace)
    run_case("temperature=0.7", model, assistant_model, processor, inputs, args.max_new_tokens,
             args.seed, True, 0.7, args.top_p, args.top_k, args.trace)


if __name__ == "__main__":
    main()
