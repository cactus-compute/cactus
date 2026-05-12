from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def write_reports(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conversion_manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    weights = []
    for row in rows:
        if row.get("output_file") and str(row.get("output_file")).endswith(".weights"):
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
    total_bytes = sum(int(r.get("bytes", 0) or 0) for r in rows)
    summary = {
        "total_tensors": len(rows),
        "counts_by_status": dict(by_status),
        "counts_by_component": dict(by_component),
        "counts_by_precision": dict(by_precision),
        "fallback_reasons": dict(fallback),
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
