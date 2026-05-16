from __future__ import annotations

import json
from argparse import Namespace

from cactus.cli import assistant_bundle


def test_package_assistant_reuses_main_prompt_token_ids(tmp_path):
    main_dir = tmp_path / "main"
    assistant_dir = main_dir / "components" / "assistant"
    (main_dir / "components" / "decoder").mkdir(parents=True, exist_ok=True)
    (main_dir / "components" / "target_embedding").mkdir(parents=True, exist_ok=True)
    (main_dir / "components" / "decoder" / "graph.cactus").write_bytes(b"graph")
    (main_dir / "components" / "target_embedding" / "graph.cactus").write_bytes(b"graph")
    (main_dir / "components" / "manifest.json").write_text(
        json.dumps(
            {
                "task": "causal_lm_logits",
                "inputs": {"prompt_input_ids": [[2, 105, 2364, 107]]},
                "component_order": ["decoder", "target_embedding"],
                "components": [
                    {"component": "decoder", "graph": "components/decoder/graph.cactus", "bound_constant_bindings": []},
                    {"component": "target_embedding", "graph": "components/target_embedding/graph.cactus", "bound_constant_bindings": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    captured_extra_args: list[str] = []

    def _fake_transpile(namespace: Namespace) -> int:
        captured_extra_args.extend(namespace.extra_args)
        (assistant_dir / "components" / "assistant").mkdir(parents=True, exist_ok=True)
        (assistant_dir / "components" / "assistant" / "graph.cactus").write_bytes(b"graph")
        (assistant_dir / "components" / "manifest.json").write_text(
            json.dumps(
                {
                    "task": "causal_lm_logits",
                    "component_order": ["assistant"],
                    "components": [
                        {
                            "component": "assistant",
                            "directory": "components/assistant",
                            "graph": "components/assistant/graph.cactus",
                            "bound_constant_bindings": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return 0

    assistant_bundle.package_assistant_for_convert(
        cq_main=lambda _command: None,
        cmd_transpile=_fake_transpile,
        main_output_dir=main_dir,
        assistant_model_id="assistant/model",
        assistant_bits=4,
        device="cpu",
        prompt="ignored when ids exist",
        max_new_tokens=4,
        torch_dtype="bfloat16",
        token=None,
        cache_dir=None,
        trust_remote_code=False,
        local_files_only=False,
    )

    input_ids_index = captured_extra_args.index("--input-ids")
    assert captured_extra_args[input_ids_index + 1] == "2,105,2364,107"
    assert "--prompt" not in captured_extra_args
