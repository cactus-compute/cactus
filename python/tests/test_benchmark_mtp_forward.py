import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "cactus-engine" / "tests" / "benchmark_mtp_forward.py"
    spec = importlib.util.spec_from_file_location("benchmark_mtp_forward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_sections_and_matmul_rows_parse_target_shapes():
    bench = _load_benchmark_module()
    profile = """noise
=== Graph Execution Profile ===
Operation               Time (ms)   Output Shape        Backend
------------------------------------------------------------------------
MATMUL                  0.100       [2,4096]
Total execution time: 1.000 ms
=== Graph Execution Profile ===
Operation               Time (ms)   Output Shape        Backend
------------------------------------------------------------------------
MATMUL                  0.200       [150,1536]
MATMUL                  0.300       [150,262144]
Total execution time: 2.000 ms
"""

    sections = bench._profile_sections(profile)

    assert len(sections) == 2
    assert bench._matmul_row_dims(sections[0]) == [2]
    assert bench._matmul_row_dims(sections[1]) == [150, 150]


def test_small_m_gate_is_strict_about_expected_width():
    bench = _load_benchmark_module()

    assert 3 in bench._matmul_row_dims(["MATMUL                  0.100       [3,4096]"])
    assert 3 not in bench._matmul_row_dims(["MATMUL                  0.100       [150,4096]"])


def test_require_small_m_exits_nonzero_when_profile_gate_fails(monkeypatch, tmp_path):
    bench = _load_benchmark_module()
    chat = tmp_path / "chat"
    chat.write_text("#!/bin/sh\n", encoding="utf-8")

    result = {
        "label": "baseline",
        "draft_tokens": None,
        "target": {"median": 1.0, "min": 1.0, "max": 1.0, "stddev": 0.0},
        "target_context": {"median": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0},
        "assistant": {"median": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0},
        "misc": {"median": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0},
        "generated_median": 1.0,
        "prompt_tokens_median": 1.0,
        "accepted_median": 0.0,
        "rejected_median": 0.0,
        "acceptance_median": 0.0,
        "verifier_width_median": 1.0,
        "ram_peak": 0.0,
    }

    def fake_run_mode(args, label, draft_tokens):
        row = dict(result)
        row["label"] = label
        row["draft_tokens"] = draft_tokens
        row["verifier_width_median"] = 1.0 if draft_tokens is None else float(draft_tokens + 1)
        return row

    def fake_profile_small_m(args, draft_tokens):
        return {
            "M": draft_tokens + 1,
            "reported_verifier_width": draft_tokens + 1,
            "graph_executions": 1,
            "verifier_matmul_rows": [150],
            "small_m_verified": False,
        }

    monkeypatch.setattr(bench, "run_mode", fake_run_mode)
    monkeypatch.setattr(bench, "profile_small_m", fake_profile_small_m)
    monkeypatch.setattr(sys, "argv", [
        "benchmark_mtp_forward.py",
        str(tmp_path / "bundle"),
        "--chat",
        str(chat),
        "--prompt",
        "test prompt",
        "--profile-check",
        "--require-small-m",
    ])

    assert bench.main() == 2
