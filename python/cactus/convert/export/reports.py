from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_reports(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversion_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    weights = []
    for row in rows:
        output_file = str(row.get("output_file") or "")
        if output_file.endswith((".weights", ".bias")):
            weights.append({
                "source_name": row.get("source_name"),
                "hf_name": row.get("hf_name") or row.get("source_name"),
                "adapter_name": row.get("adapter_name") or row.get("source_name"),
                "output_name": row.get("output_file"),
                "shape": row.get("shape"),
                "precision": row.get("precision"),
                "status": row.get("status"),
                "component": row.get("component"),
                "scale_factor": row.get("scale_factor", 1.0),
                "adapter_family": row.get("adapter_family"),
                "source_names": row.get("source_names") or [row.get("source_name")],
                "transform": row.get("transform", "none"),
                "qdq_restore": row.get("qdq_restore", "hf_key"),
            })
    (out_dir / "weights_manifest.json").write_text(json.dumps({"weights": weights}, indent=2, sort_keys=True), encoding="utf-8")
    by_status = Counter(r["status"] for r in rows)
    by_component = Counter(r["component"] for r in rows)
    by_precision = Counter(r["precision"] for r in rows)
    fallback = Counter(r.get("fallback_reason") or "" for r in rows if r["status"] in {"fallback", "unrecognized"})
    gptq_expected = [r for r in rows if r.get("gptq_expected")]
    gptq_used = [r for r in gptq_expected if r.get("gptq_used")]
    gptq_uncalibrated = [r for r in gptq_expected if not r.get("gptq_used")]
    gptq_zero_samples = [
        r for r in gptq_uncalibrated
        if int(r.get("hessian_samples", 0) or 0) <= 0
    ]
    gptq_unusable_hessian = [
        r for r in gptq_uncalibrated
        if int(r.get("hessian_samples", 0) or 0) > 0
    ]
    total_bytes = sum(int(r.get("bytes", 0) or 0) for r in rows)
    summary = {
        "total_tensors": len(rows),
        "counts_by_status": dict(by_status),
        "counts_by_component": dict(by_component),
        "counts_by_precision": dict(by_precision),
        "fallback_reasons": dict(fallback),
        "gptq": {
            "expected_tensors": len(gptq_expected),
            "calibrated_tensors": len(gptq_used),
            "uncalibrated_tensors": len(gptq_uncalibrated),
            "zero_sample_tensors": len(gptq_zero_samples),
            "unusable_hessian_tensors": len(gptq_unusable_hessian),
        },
        "total_bytes": total_bytes,
    }
    (out_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\nConversion summary")
    print("------------------")
    for key in ["converted", "fallback", "ignored", "unrecognized"]:
        print(f"{key:13s} {summary.get('counts_by_status', {}).get(key, 0)}")
    print(f"{'total bytes':13s} {summary.get('total_bytes', 0)}")
    gptq = summary.get("gptq", {})
    expected = int(gptq.get("expected_tensors", 0) or 0)
    calibrated = int(gptq.get("calibrated_tensors", 0) or 0)
    uncalibrated = int(gptq.get("uncalibrated_tensors", 0) or 0)
    zero_samples = int(gptq.get("zero_sample_tensors", 0) or 0)
    unusable_hessian = int(gptq.get("unusable_hessian_tensors", 0) or 0)
    if expected:
        print(f"{'GPTQ calibrated':13s} {calibrated}/{expected}")
    if uncalibrated:
        print(
            f"\nWARNING: {uncalibrated} GPTQ-eligible CQ tensors used RTN fallback "
            f"({zero_samples} zero-sample, {unusable_hessian} unusable Hessian). "
            "Provide --calibration-manifest with representative data; see "
            "conversion_manifest.json for affected tensors."
        )
