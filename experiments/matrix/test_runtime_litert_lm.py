from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import runtime_litert_lm as litert_lm_module  # noqa: E402
from runtime_litert_lm import (  # noqa: E402
    EXPECTED_QUANTIZATION,
    _litert_lm_benchmark_command,
    _parse_litert_lm_metrics,
    android_litert_lm_unsupported_reason,
    litert_lm_artifact_status,
    litert_lm_thread_note,
    litert_lm_unsupported_reason,
    paired_litert_lm_requests,
    run_litert_lm_llm_operations,
)


def llm_operations() -> list[dict[str, object]]:
    return [
        {"operation": "prefill", "seqlen": 256, "decode_tokens": 0, "input_path": "ids_256.csv"},
        {"operation": "prefill", "seqlen": 4096, "decode_tokens": 0, "input_path": "ids_4096.csv"},
        {"operation": "decode", "seqlen": 256, "decode_tokens": 100, "input_path": "ids_256.csv"},
    ]


def matrix_config(artifact_path: str | None) -> dict[str, object]:
    artifact: dict[str, object] = {"path": artifact_path}
    if artifact_path is None:
        artifact["unsupported_reason"] = "LiteRT-LM Gemma artifact is not configured yet"
    return {
        "devices": {"mac_m4pro": {"kind": "mac"}},
        "models": {
            "gemma": {
                "type": "llm",
                "artifacts": {"litert_lm": artifact},
            }
        },
        "operations": {"llm": {"warmup_runs": 0, "measurement_runs": 1}},
    }


def android_matrix_config(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "devices": {"pixel_10a": {"kind": "android"}},
        "models": {
            "gemma": {
                "type": "llm",
                "artifacts": {"litert_lm": artifact},
            }
        },
        "operations": {"llm": {"warmup_runs": 0, "measurement_runs": 1}},
    }


def asr_matrix_config(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "devices": {"mac_m4pro": {"kind": "mac"}},
        "models": {
            "parakeet": {
                "type": "asr",
                "artifacts": {"litert_lm": artifact},
            }
        },
        "operations": {"llm": {"warmup_runs": 0, "measurement_runs": 1}},
    }


def android_asr_matrix_config(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "devices": {"pixel_10a": {"kind": "android"}},
        "models": {
            "parakeet": {
                "type": "asr",
                "artifacts": {"litert_lm": artifact},
            }
        },
        "operations": {"llm": {"warmup_runs": 0, "measurement_runs": 1}},
    }


def write_valid_litertlm_manifest(model_path: Path) -> None:
    model_path.write_bytes(b"LITERTLM")
    model_path.with_suffix(model_path.suffix + ".json").write_text(
        json.dumps(
            {
                "artifact_type": "litert_lm_e2e",
                "quantization": EXPECTED_QUANTIZATION,
                "runner_contract": "paired_prefill_decode",
                "entry_points": {"prefill": "prefill.tflite", "decode": "decode.tflite"},
            }
        ),
        encoding="utf-8",
    )


def write_valid_binary_litertlm(model_path: Path) -> None:
    from ai_edge_litert.internal import litertlm_builder

    root = model_path.parent
    llm_metadata = root / "llm_metadata.pb"
    tflite_model = root / "prefill_decode.tflite"
    tokenizer = root / "tokenizer.model"
    llm_metadata.write_bytes(b"")
    tflite_model.write_bytes(b"TFL3")
    tokenizer.write_bytes(b"sentencepiece")

    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors",
            value="cactus-test",
            dtype=litertlm_builder.DType.STRING,
        )
    )
    builder.add_llm_metadata(str(llm_metadata))
    builder.add_tflite_model(
        str(tflite_model),
        litertlm_builder.TfLiteModelType.PREFILL_DECODE,
    )
    builder.add_sentencepiece_tokenizer(str(tokenizer))
    with model_path.open("wb") as handle:
        builder.build(handle)


class LiteRTLMRuntimeTest(unittest.TestCase):
    def test_paired_request_contract_includes_prefill_only_tail(self) -> None:
        requests = paired_litert_lm_requests(llm_operations())

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].context_tokens, 256)
        self.assertEqual(requests[0].generated_tokens, 100)
        self.assertIsNotNone(requests[0].decode)
        self.assertEqual(requests[1].context_tokens, 4096)
        self.assertEqual(requests[1].generated_tokens, 0)
        self.assertIsNone(requests[1].decode)

    def test_paired_request_contract_includes_decode_only_filter(self) -> None:
        requests = paired_litert_lm_requests([llm_operations()[2]])

        self.assertEqual(len(requests), 1)
        self.assertIsNone(requests[0].prefill)
        self.assertEqual(requests[0].decode.operation, "decode")
        self.assertEqual(requests[0].context_tokens, 256)
        self.assertEqual(requests[0].generated_tokens, 100)

    def test_unconfigured_artifact_keeps_all_rows_unsupported(self) -> None:
        rows = run_litert_lm_llm_operations(
            matrix_config(None),
            "mac_m4pro",
            "litert_lm",
            "gemma",
            llm_operations(),
        )

        self.assertEqual([row["status"] for row in rows], ["unsupported", "unsupported", "unsupported"])
        self.assertTrue(all("not configured" in str(row["notes"]) for row in rows))

    def test_decode_only_filter_emits_decode_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.litertlm"
            write_valid_litertlm_manifest(model_path)

            with mock.patch.object(litert_lm_module, "litert_lm_unsupported_reason", return_value=None):
                with mock.patch.object(
                    litert_lm_module,
                    "_measure_request",
                    return_value=[{"prefill_tps": 10.0, "decode_tps": 3.5}],
                ) as measure_request:
                    rows = run_litert_lm_llm_operations(
                        matrix_config(str(model_path)),
                        "mac_m4pro",
                        "litert_lm",
                        "gemma",
                        [llm_operations()[2]],
                    )

        self.assertEqual([row["operation"] for row in rows], ["decode"])
        self.assertEqual(rows[0]["throughput_tok_per_s"], "3.500000")
        self.assertNotIn("paired_prefill_decode=true", rows[0]["notes"])
        self.assertIsNone(measure_request.call_args.args[4].prefill)
        self.assertEqual(measure_request.call_args.args[4].decode.operation, "decode")

    def test_missing_positive_throughput_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.litertlm"
            write_valid_litertlm_manifest(model_path)

            with mock.patch.object(litert_lm_module, "litert_lm_unsupported_reason", return_value=None):
                with mock.patch.object(
                    litert_lm_module,
                    "_measure_request",
                    return_value=[{"peak_ram_usage_mb": 12.0}],
                ):
                    rows = run_litert_lm_llm_operations(
                        matrix_config(str(model_path)),
                        "mac_m4pro",
                        "litert_lm",
                        "gemma",
                        [llm_operations()[1]],
                    )

        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["throughput_tok_per_s"], "")
        self.assertIn("did not report positive prefill throughput", rows[0]["notes"])
        self.assertIn("throughputs=0.000000", rows[0]["notes"])

    def test_kernel_only_tflite_inventory_is_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "decoder_prefill.tflite").write_bytes(b"TFL3")
            (root / "decoder_step.tflite").write_bytes(b"TFL3")

            status = litert_lm_artifact_status(root)

        self.assertFalse(status.supported)
        self.assertIn("kernel-only LiteRT graph inventory is unsupported", status.reason)
        self.assertIn("paired prefill+decode contract", status.reason)

    def test_manifest_requires_int4_per_output_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "litert_lm_manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "litert_lm_e2e",
                        "quantization": "int8",
                        "entry_points": {"prefill": "prefill.tflite", "decode": "decode.tflite"},
                    }
                ),
                encoding="utf-8",
            )

            status = litert_lm_artifact_status(root)

        self.assertFalse(status.supported)
        self.assertIn("expects int4 per-output-channel", status.reason)

    def test_valid_manifest_marks_artifact_supported_before_runner_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "litert_lm_manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "litert_lm_e2e",
                        "quantization": EXPECTED_QUANTIZATION,
                        "runner_contract": "paired_prefill_decode",
                        "entry_points": {"prefill": "prefill.tflite", "decode": "decode.tflite"},
                    }
                ),
                encoding="utf-8",
            )

            status = litert_lm_artifact_status(root)

        self.assertTrue(status.supported, status.reason)
        self.assertEqual(status.quantization, EXPECTED_QUANTIZATION)

    def test_binary_litertlm_artifact_is_supported_with_quantization_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.litertlm"
            write_valid_binary_litertlm(model_path)

            status = litert_lm_artifact_status(
                model_path,
                {"quantization": EXPECTED_QUANTIZATION},
            )

        self.assertTrue(status.supported, status.reason)
        self.assertEqual(status.artifact_kind, "litertlm")

    def test_binary_litertlm_requires_quantization_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.litertlm"
            write_valid_binary_litertlm(model_path)

            status = litert_lm_artifact_status(model_path)

        self.assertFalse(status.supported)
        self.assertIn("requires matrix quantization metadata", status.reason)

    def test_litertlm_metadata_file_names_missing_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "Qwen3-0.6B.litertlm.metadata"
            metadata_path.write_text("etag", encoding="utf-8")

            status = litert_lm_artifact_status(metadata_path)

        self.assertFalse(status.supported)
        self.assertIn("download metadata is present", status.reason)
        self.assertIn("Qwen3-0.6B.litertlm", status.reason)

    def test_mac_asr_configured_unsupported_reason_is_preserved(self) -> None:
        self.assertEqual(
            litert_lm_unsupported_reason(
                asr_matrix_config(
                    {
                        "path": None,
                        "unsupported_reason": "LiteRT-LM Parakeet requires an ASR .litertlm artifact",
                    }
                ),
                "mac_m4pro",
                "parakeet",
            ),
            "LiteRT-LM Parakeet requires an ASR .litertlm artifact",
        )

    def test_benchmark_command_uses_litert_lm_flags(self) -> None:
        request = paired_litert_lm_requests(llm_operations())[0]

        command = _litert_lm_benchmark_command(
            ["/tmp/litert_lm_advanced_main"],
            Path("/tmp/model.litertlm"),
            {"backend": "cpu"},
            request,
        )

        self.assertIn("--model_path=/tmp/model.litertlm", command)
        self.assertIn("--benchmark=true", command)
        self.assertIn("--benchmark_prefill_tokens=256", command)
        self.assertIn("--benchmark_decode_tokens=100", command)
        self.assertIn("--max_num_tokens=356", command)
        self.assertIn("--prefill_batch_sizes=256", command)
        self.assertIn("--max_output_tokens=100", command)
        self.assertIn("--num_cpu_threads=1", command)
        self.assertNotIn("--artifact", command)

    def test_benchmark_command_honors_native_max_num_tokens_config(self) -> None:
        request = paired_litert_lm_requests(llm_operations())[0]

        command = _litert_lm_benchmark_command(
            ["/tmp/litert_lm_advanced_main"],
            Path("/tmp/model.litertlm"),
            {"backend": "cpu", "max_num_tokens": 4096},
            request,
        )

        self.assertIn("--max_num_tokens=4096", command)

    def test_benchmark_command_reserves_one_token_at_context_limit_for_prefill_only(self) -> None:
        request = paired_litert_lm_requests([llm_operations()[1]])[0]

        command = _litert_lm_benchmark_command(
            ["/tmp/litert_lm_advanced_main"],
            Path("/tmp/model.litertlm"),
            {"backend": "cpu", "max_num_tokens": 4096},
            request,
        )

        self.assertIn("--benchmark_prefill_tokens=4095", command)
        self.assertIn("--prefill_batch_sizes=4095", command)
        self.assertIn("--max_num_tokens=4096", command)

    def test_benchmark_command_reserves_decode_token_limit(self) -> None:
        request = paired_litert_lm_requests(
            [
                {"operation": "prefill", "seqlen": 4096, "decode_tokens": 0, "input_path": "ids_4096.csv"},
                {"operation": "decode", "seqlen": 4096, "decode_tokens": 100, "input_path": "ids_4096.csv"},
            ]
        )[0]

        command = _litert_lm_benchmark_command(
            ["/tmp/litert_lm_advanced_main"],
            Path("/tmp/model.litertlm"),
            {"backend": "cpu", "max_num_tokens": 4096},
            request,
        )

        self.assertIn("--max_num_tokens=4196", command)

    def test_android_measure_request_runs_warmups_before_measurements(self) -> None:
        request = paired_litert_lm_requests([llm_operations()[0]])[0]
        config = android_matrix_config({"path": "/tmp/model.litertlm"})
        config["operations"]["llm"]["warmup_runs"] = 2
        config["operations"]["llm"]["measurement_runs"] = 3

        with mock.patch.object(
            litert_lm_module,
            "_run_android_litert_lm_once",
            side_effect=[{"run": index} for index in range(5)],
        ) as run_once:
            measurements = litert_lm_module._measure_request(
                config,
                "pixel_10a",
                Path("/tmp/model.litertlm"),
                {},
                request,
                Path("/tmp/repo"),
            )

        self.assertEqual(measurements, [{"run": 2}, {"run": 3}, {"run": 4}])
        self.assertEqual(run_once.call_count, 5)

    def test_native_failure_message_keeps_litert_error_context(self) -> None:
        message = litert_lm_module._litert_lm_failure_message(
            """
ERROR: external/litert/tflite/kernels/dynamic_update_slice.cc:68 SizeOfDimension(update, i) <= SizeOfDimension(operand, i) was not true.
ERROR: Node number 2122 (DYNAMIC_UPDATE_SLICE) failed to prepare.
F0000 litert_lm_advanced_main.cc:246] Check failed: MainHelper(argc, argv) is OK
*** Check failure stack trace: ***
    @        0x180b09d54  start
""",
            134,
        )

        self.assertIn("DYNAMIC_UPDATE_SLICE", message)
        self.assertIn("Check failed", message)
        self.assertNotEqual(message, "@        0x180b09d54  start")

    def test_thread_note_marks_cli_as_unverified(self) -> None:
        config = matrix_config("/tmp/model.litertlm")

        with mock.patch.object(litert_lm_module, "litert_lm_runner_command", return_value=["litert-lm"]):
            note = litert_lm_thread_note(config, "mac_m4pro", {})

        self.assertEqual(note, "threads=unverified_litert_lm_cli_no_thread_flag")

    def test_thread_note_marks_native_runner_as_threads_one(self) -> None:
        config = matrix_config("/tmp/model.litertlm")

        note = litert_lm_thread_note(config, "mac_m4pro", {"runner": ["/tmp/litert_lm_advanced_main"]})

        self.assertEqual(note, "threads=1")

    def test_benchmark_command_uses_installed_litert_lm_cli(self) -> None:
        request = paired_litert_lm_requests(llm_operations())[0]

        command = _litert_lm_benchmark_command(
            ["/tmp/litert-lm"],
            Path("/tmp/model.litertlm"),
            {"backend": "cpu", "max_num_tokens": 2048, "enable_speculative_decoding": "auto"},
            request,
        )

        self.assertEqual(command[:3], ["/tmp/litert-lm", "benchmark", "/tmp/model.litertlm"])
        self.assertIn("--prefill-tokens", command)
        self.assertIn("256", command)
        self.assertIn("--decode-tokens", command)
        self.assertIn("100", command)
        self.assertIn("--max-num-tokens", command)
        self.assertIn("2048", command)
        self.assertIn("--enable-speculative-decoding", command)

    def test_parse_text_benchmark_info(self) -> None:
        metrics = _parse_litert_lm_metrics(
            """
BenchmarkInfo:
  Time to first token: 0.25 s
  Prefill Turns (Total 1 turns):
    Prefill Turn 1: Processed 256 tokens in 123 ms duration.
      Prefill Speed: 2048.50 tokens/sec.
  Decode Turns (Total 1 turns):
    Decode Turn 1: Processed 100 tokens in 400 ms duration.
      Decode Speed: 250.00 tokens/sec.
"""
        )

        self.assertEqual(metrics["prefill_tps"], 2048.5)
        self.assertEqual(metrics["decode_tps"], 250.0)
        self.assertEqual(metrics["time_to_first_token_ms"], 250.0)

    def test_parse_cli_benchmark_results(self) -> None:
        metrics = _parse_litert_lm_metrics(
            """
----- Results -----
Prefill speed:        220.50 tokens/s
Decode speed:         0.69 tokens/s
Init time:            7.6006 s
Time to first token:  2.0292 s
"""
        )

        self.assertEqual(metrics["prefill_tps"], 220.5)
        self.assertEqual(metrics["decode_tps"], 0.69)
        self.assertAlmostEqual(metrics["init_time_ms"], 7600.6)
        self.assertAlmostEqual(metrics["time_to_first_token_ms"], 2029.2)

    def test_android_missing_artifact_names_expected_litertlm(self) -> None:
        reason = android_litert_lm_unsupported_reason(
            android_matrix_config(
                {
                    "path": None,
                    "expected_android_artifact": "weights/litert_lm/gemma-4-e2b-it-int4.litertlm",
                }
            ),
            "gemma",
        )

        self.assertEqual(
            reason,
            "Android LiteRT-LM artifact does not exist: weights/litert_lm/gemma-4-e2b-it-int4.litertlm",
        )

    def test_android_valid_artifact_rejects_missing_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.litertlm"
            write_valid_litertlm_manifest(model_path)
            runner_path = Path(tmp) / "missing_litert_lm_advanced_main"

            reason = android_litert_lm_unsupported_reason(
                android_matrix_config({"path": str(model_path), "android_runner": str(runner_path)}),
                "gemma",
            )

        self.assertIn("Android LiteRT-LM runner binary does not exist", str(reason))
        self.assertIn("missing_litert_lm_advanced_main", str(reason))

    def test_android_gpu_backend_names_missing_shared_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.litertlm"
            write_valid_litertlm_manifest(model_path)

            reason = android_litert_lm_unsupported_reason(
                android_matrix_config(
                    {
                        "path": str(model_path),
                        "android_backend": "gpu",
                        "android_gpu_lib_dir": str(root / "missing_libs"),
                    }
                ),
                "gemma",
            )

        self.assertIn("prebuilt/android_arm64 shared libraries", str(reason))
        self.assertIn("libLiteRtGpuAccelerator.so", str(reason))

    def test_android_npu_backend_points_to_ai_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.litertlm"
            write_valid_litertlm_manifest(model_path)

            reason = android_litert_lm_unsupported_reason(
                android_matrix_config({"path": str(model_path), "android_backend": "google_tensor_npu"}),
                "gemma",
            )

        self.assertIn("LiteRT AOT AI Pack app/runtime deployment", str(reason))

    def test_android_parakeet_litert_asr_names_missing_runner_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_root = root / "parakeet"
            artifact_root.mkdir()
            encoder = artifact_root / "parakeet-encoder.tflite"
            decoder_joint = artifact_root / "parakeet-decoder-joint.tflite"
            vocab = artifact_root / "vocab.json"
            config = artifact_root / "config.json"
            encoder.write_bytes(b"TFL3")
            decoder_joint.write_bytes(b"TFL3")
            vocab.write_text("{}", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")

            reason = android_litert_lm_unsupported_reason(
                android_asr_matrix_config(
                    {
                        "path": str(artifact_root),
                        "artifact_format": "litert_tflite_asr",
                        "encoder_path": str(encoder),
                        "decoder_joint_path": str(decoder_joint),
                        "vocab_path": str(vocab),
                        "config_path": str(config),
                        "android_asr_runner_path": str(root / "missing_runner"),
                    }
                ),
                "parakeet",
            )

        self.assertIn("Android LiteRT Parakeet ASR runner does not exist", str(reason))
        self.assertIn("missing_runner", str(reason))

    def test_android_parakeet_litert_asr_supported_with_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_root = root / "parakeet"
            artifact_root.mkdir()
            encoder = artifact_root / "parakeet-encoder.tflite"
            decoder_joint = artifact_root / "parakeet-decoder-joint.tflite"
            vocab = artifact_root / "vocab.json"
            config = artifact_root / "config.json"
            runner = root / "litert_parakeet_asr"
            encoder.write_bytes(b"TFL3")
            decoder_joint.write_bytes(b"TFL3")
            vocab.write_text("{}", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")
            runner.write_text("#!/bin/sh\n", encoding="utf-8")

            reason = android_litert_lm_unsupported_reason(
                android_asr_matrix_config(
                    {
                        "path": str(artifact_root),
                        "artifact_format": "litert_tflite_asr",
                        "encoder_path": str(encoder),
                        "decoder_joint_path": str(decoder_joint),
                        "vocab_path": str(vocab),
                        "config_path": str(config),
                        "android_asr_runner_path": str(runner),
                    }
                ),
                "parakeet",
            )

        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
