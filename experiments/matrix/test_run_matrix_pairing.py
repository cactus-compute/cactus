from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_matrix


def prefill(seqlen: int) -> dict[str, object]:
    return {
        "operation": "prefill",
        "seqlen": seqlen,
        "decode_tokens": 0,
        "input_path": f"ids_{seqlen}.csv",
    }


def decode(seqlen: int) -> dict[str, object]:
    return {
        "operation": "decode",
        "seqlen": seqlen,
        "decode_tokens": 100,
        "input_path": f"ids_{seqlen}.csv",
    }


def config(kind: str = "mac") -> dict[str, object]:
    return {
        "devices": {
            "mac_m4pro": {"kind": kind},
            "pixel_10a": {"kind": "android"},
        },
        "runtimes": {
            "cactus": {"version": "repo test"},
            "llama_cpp": {"version": "test"},
        },
        "models": {
            "lfm_2_5_vl_1_6b": {
                "type": "llm",
                "artifacts": {
                    "cactus": {"path": "."},
                    "llama_cpp": {"path": "."},
                },
            },
        },
        "operations": {
            "llm": {
                "warmup_runs": 0,
                "measurement_runs": 1,
            },
        },
    }


class RunMatrixPairingTest(unittest.TestCase):
    def test_llm_operations_default_to_generated_token_ids(self) -> None:
        cfg = config()
        cfg["operations"]["llm"]["prefill"] = {"seqlens": [3], "decode_tokens": 0}
        cfg["operations"]["llm"]["decode"] = {"contexts": [3], "decode_tokens": 2}

        operations = run_matrix.operations_for_model(cfg, "lfm_2_5_vl_1_6b")

        self.assertEqual([operation["input_path"] for operation in operations], ["", ""])
        self.assertEqual(run_matrix.input_ids_for_operation(operations[0]), "2,2,2")

    def test_pair_helper_keeps_prefill_tail_and_decode_only_filter(self) -> None:
        pairs = run_matrix.paired_llm_operations([prefill(256), decode(256), prefill(4096)])

        self.assertEqual(
            [
                (
                    pair[0]["operation"] if pair[0] is not None else None,
                    pair[1]["operation"] if pair[1] is not None else None,
                    pair[0]["seqlen"] if pair[0] is not None else pair[1]["seqlen"],
                )
                for pair in pairs
            ],
            [("prefill", "decode", 256), ("prefill", None, 4096)],
        )

        decode_only = run_matrix.paired_llm_operations([decode(256)])
        self.assertEqual(len(decode_only), 1)
        self.assertIsNone(decode_only[0][0])
        self.assertEqual(decode_only[0][1]["operation"], "decode")

    def test_prefill_throughput_prefers_cache_prime_timing(self) -> None:
        operation = prefill(512)
        result = {
            "prefill_tps": 34.0,
            "prefill_compute_tps": 52.0,
            "cache_prime_tokens": 511,
            "cache_prime_compute_ms": 9830.0,
            "cache_prime_ms": 10000.0,
            "time_to_first_token_ms": 15000.0,
        }

        self.assertAlmostEqual(run_matrix.measured_throughput(result, operation=operation), 52.0)
        self.assertIn(
            "cache-primed 511 tokens",
            run_matrix.actual_token_note([result], operation),
        )

    def test_base_row_accepts_asr_operation_without_llm_fields(self) -> None:
        row = run_matrix.base_row(
            "mac_m4pro",
            "executorch",
            "parakeet_tdt_v3",
            {"operation": "parakeet_transcribe"},
        )

        self.assertEqual(row["seqlen"], "")
        self.assertEqual(row["decode_tokens"], "")

    def test_mac_cactus_parakeet_uses_native_json_runner_and_configured_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "parakeet-tdt-0.6b-v3-transpiled"
            artifact.mkdir()
            audio = root / "audio.wav"
            audio.write_bytes(b"RIFF")
            reference = root / "reference.txt"
            reference.write_text("hello world", encoding="utf-8")
            cfg = {
                "devices": {"mac_m4pro": {"kind": "mac"}},
                "runtimes": {"cactus": {"version": "repo test"}},
                "models": {
                    "parakeet_tdt_v3": {
                        "type": "asr",
                        "artifacts": {"cactus": {"path": str(artifact)}},
                    }
                },
                "operations": {
                    "parakeet": {
                        "warmup_runs": 0,
                        "measurement_runs": 1,
                        "audio_seconds": 1,
                    }
                },
            }
            operation = {
                "operation": "parakeet_transcribe",
                "input_path": str(audio),
                "reference_path": str(reference),
            }

            with mock.patch.object(run_matrix, "native_transcribe_runner", return_value=Path("/tmp/native_transcribe_json")):
                with mock.patch.object(
                    run_matrix,
                    "run_transcribe_once",
                    return_value={
                        "response": "hello world",
                        "elapsed_seconds": 0.5,
                        "peak_process_memory_mb": 123.0,
                    },
                ) as run_once:
                    row = run_matrix.run_cactus_parakeet(
                        cfg,
                        "mac_m4pro",
                        "cactus",
                        "parakeet_tdt_v3",
                        operation,
                        None,
                    )

        command = run_once.call_args.args[0]
        self.assertEqual(command[0], "/tmp/native_transcribe_json")
        self.assertEqual(command[1], str(artifact))
        self.assertEqual(row["status"], "ok")
        self.assertIn("runner=native_transcribe_json", row["notes"])
        self.assertIn("timing_source=wrapper_elapsed_seconds", row["notes"])

    def test_android_cactus_parakeet_uses_native_json_engine_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "parakeet-tdt-0.6b-v3-transpiled"
            artifact.mkdir()
            audio = root / "audio.wav"
            audio.write_bytes(b"RIFF")
            reference = root / "reference.txt"
            reference.write_text("hello world", encoding="utf-8")
            cfg = {
                "devices": {"pixel_10a": {"kind": "android"}},
                "runtimes": {"cactus": {"version": "repo test"}},
                "models": {
                    "parakeet_tdt_v3": {
                        "type": "asr",
                        "artifacts": {"cactus": {"path": str(artifact)}},
                    }
                },
                "operations": {
                    "parakeet": {
                        "warmup_runs": 0,
                        "measurement_runs": 1,
                        "audio_seconds": 1,
                    }
                },
            }
            operation = {
                "operation": "parakeet_transcribe",
                "input_path": str(audio),
                "reference_path": str(reference),
            }

            with mock.patch.object(run_matrix, "run_android_transpiled_parakeet") as transpiled:
                with mock.patch.object(run_matrix, "run_android_native_cactus_parakeet", return_value={"status": "ok"}) as native:
                    row = run_matrix.run_cactus_parakeet(
                        cfg,
                        "pixel_10a",
                        "cactus",
                        "parakeet_tdt_v3",
                        operation,
                        None,
                    )

        self.assertEqual(row, {"status": "ok"})
        native.assert_called_once()
        transpiled.assert_not_called()

    def test_asr_rtf_uses_elapsed_seconds_not_runner_total_ms(self) -> None:
        rtfs = run_matrix.asr_rtfs_from_elapsed(
            [{"elapsed_seconds": 2.0, "total_ms": 100.0}],
            audio_seconds=4.0,
        )

        self.assertEqual(rtfs, [0.5])

    def test_mac_cactus_pairs_non_gemma_llms_and_keeps_prefill_tail(self) -> None:
        operations = [prefill(256), decode(256), prefill(4096)]
        with mock.patch.object(run_matrix, "run_cactus_llm_pair", return_value=[{"row": "pair"}]) as pair:
            with mock.patch.object(run_matrix, "run_cactus_gemma", return_value={"row": "standalone"}) as standalone:
                rows = run_matrix.run_cactus_llm_operations(
                    config(),
                    "mac_m4pro",
                    "cactus",
                    "lfm_2_5_vl_1_6b",
                    operations,
                    Path("out.csv"),
                    None,
                )

        self.assertEqual(rows, [{"row": "pair"}, {"row": "standalone"}])
        pair.assert_called_once()
        self.assertEqual(pair.call_args.args[3], "lfm_2_5_vl_1_6b")
        self.assertEqual(pair.call_args.args[4]["operation"], "prefill")
        self.assertEqual(pair.call_args.args[5]["operation"], "decode")
        standalone.assert_called_once()
        self.assertEqual(standalone.call_args.args[4]["seqlen"], 4096)

    def test_decode_only_filter_runs_direct_backend_decode_rows(self) -> None:
        operation = decode(256)

        with mock.patch.object(run_matrix, "run_cactus_gemma", return_value={"operation": "decode"}) as cactus:
            rows = run_matrix.run_cactus_llm_operations(
                config(),
                "mac_m4pro",
                "cactus",
                "lfm_2_5_vl_1_6b",
                [operation],
                Path("out.csv"),
                None,
            )
        self.assertEqual(rows, [{"operation": "decode"}])
        self.assertEqual(cactus.call_args.args[4]["operation"], "decode")

        with mock.patch.object(run_matrix, "llama_cpp_measurements", return_value=[{"decode_tps": 12.0}]) as llama:
            rows = run_matrix.run_llama_cpp_llm_operations(
                config(),
                "mac_m4pro",
                "llama_cpp",
                "lfm_2_5_vl_1_6b",
                [operation],
                None,
            )
        self.assertEqual([row["operation"] for row in rows], ["decode"])
        self.assertEqual(llama.call_args.args[3]["operation"], "decode")

        metadata = {"serial": "SERIAL", "android_release": "15", "thermal_status": "none"}
        with mock.patch.object(run_matrix, "android_cactus_llm_operation_unsupported_reason", return_value=None):
            with mock.patch.object(run_matrix, "android_cactus_llm_measurements", return_value=([{"decode_tps": 13.0}], metadata)) as cactus_android:
                rows = run_matrix.run_android_cactus_llm_operations(
                    config(),
                    "pixel_10a",
                    "cactus",
                    "lfm_2_5_vl_1_6b",
                    [operation],
                    None,
                )
        self.assertEqual([row["operation"] for row in rows], ["decode"])
        self.assertEqual(cactus_android.call_args.args[4]["operation"], "decode")

        with mock.patch.object(run_matrix, "android_llama_cpp_measurements", return_value=([{"decode_tps": 14.0}], metadata)) as llama_android:
            rows = run_matrix.run_android_llama_cpp_llm_operations(
                config(),
                "pixel_10a",
                "llama_cpp",
                "lfm_2_5_vl_1_6b",
                [operation],
                None,
            )
        self.assertEqual([row["operation"] for row in rows], ["decode"])
        self.assertEqual(llama_android.call_args.args[4]["operation"], "decode")

    def test_full_core_prefill_mode_filters_decode_and_uses_direct_prefill(self) -> None:
        cfg = config()
        cfg["_benchmark_mode"] = run_matrix.FULL_CORE_PREFILL_BENCHMARK_MODE
        operations = [prefill(256), decode(256)]

        with mock.patch.object(run_matrix, "run_cactus_llm_pair") as pair:
            with mock.patch.object(run_matrix, "run_cactus_gemma", return_value={"operation": "prefill"}) as standalone:
                rows = run_matrix.run_cactus_llm_operations(
                    cfg,
                    "mac_m4pro",
                    "cactus",
                    "lfm_2_5_vl_1_6b",
                    operations,
                    Path("out.csv"),
                    None,
                )

        self.assertEqual(rows, [{"operation": "prefill"}])
        pair.assert_not_called()
        self.assertEqual(standalone.call_args.args[4]["operation"], "prefill")

    def test_full_core_prefill_cactus_command_uses_zero_decode_tokens(self) -> None:
        cfg = config()
        cfg["_benchmark_mode"] = run_matrix.FULL_CORE_PREFILL_BENCHMARK_MODE

        with mock.patch.object(run_matrix, "input_ids_for_operation", return_value="1,2,3"):
            command = run_matrix.cactus_command(
                cfg,
                "lfm_2_5_vl_1_6b",
                "artifact",
                prefill(3),
                Path("result.json"),
            )

        self.assertEqual(command[command.index("--max-new-tokens") + 1], "0")

    def test_full_core_prefill_android_invocation_omits_taskset(self) -> None:
        prepared = {
            "runner": "/data/local/tmp/cactus_matrix/bin/cactus_llm_bench",
            "model": "/data/local/tmp/cactus_matrix/models/model",
            "input": "/data/local/tmp/cactus_matrix/inputs/ids.csv",
            "logs": "/data/local/tmp/cactus_matrix/logs",
            "full_core_threads": "8",
        }

        with mock.patch.object(run_matrix, "run_android_logged_json_once", return_value={}) as logged:
            run_matrix.run_android_cactus_llm_once("SERIAL", prepared, prefill(256), "measure", None)

        invocation = logged.call_args.args[1]
        self.assertNotIn("taskset", invocation)
        self.assertIn("OMP_NUM_THREADS=8", invocation)

    def test_full_core_prefill_android_llama_cpp_invocation_omits_taskset_and_uses_threads(self) -> None:
        prepared = {
            "runner": "/data/local/tmp/cactus_matrix/bin/llama_cpp_bench",
            "model": "/data/local/tmp/cactus_matrix/models/model.gguf",
            "logs": "/data/local/tmp/cactus_matrix/logs",
        }

        with mock.patch.object(run_matrix, "run_android_logged_json_once", return_value={}) as logged:
            run_matrix.run_android_llama_cpp_once("SERIAL", prepared, prefill(256), "measure", None, 8)

        invocation = logged.call_args.args[1]
        self.assertNotIn("taskset", invocation)
        self.assertTrue(invocation.endswith(" 256 0 8"))

    def test_full_core_prefill_mode_android_disables_taskset_notes(self) -> None:
        cfg = config()
        cfg["_benchmark_mode"] = run_matrix.FULL_CORE_PREFILL_BENCHMARK_MODE
        cfg["_full_core_threads"] = 8
        operation = prefill(256)
        metadata = {"serial": "SERIAL", "android_release": "15", "thermal_status": "none"}

        rows = run_matrix.android_cactus_llm_row_from_measurements(
            cfg,
            "pixel_10a",
            "cactus",
            "lfm_2_5_vl_1_6b",
            operation,
            [{"prefill_tps": 12.0}],
            metadata,
            None,
        )

        self.assertEqual(rows["status"], "ok")
        self.assertIn("benchmark_mode=full_core_prefill", rows["notes"])
        self.assertIn("thread_count=8", rows["notes"])
        self.assertIn("taskset_mask=none", rows["notes"])
        self.assertIn("provider=cpu", rows["notes"])
        self.assertIn("gpu=disabled", rows["notes"])
        self.assertIn("cactus_chunk_size=c128", rows["notes"])


if __name__ == "__main__":
    unittest.main()
