#!/usr/bin/env python3

"""Compare Block3 Q MatMul timing between two TensorRT layer profiles."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


Q_MATMUL_PATTERN = re.compile(r"/model/block3\.(\d+)/attn/q/MatMul(?:\]|\")")
Q_ACTIVATION_PATTERN = re.compile(
    r"/model/block3\.(\d+)/attn/q/MatMul_V17_Activation_QuantizeLinear"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-block Block3 Q MatMul latency from trtexec JSON."
    )
    parser.add_argument("baseline_profile", type=Path)
    parser.add_argument("baseline_layer_info", type=Path)
    parser.add_argument("candidate_profile", type=Path)
    parser.add_argument("candidate_layer_info", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rt", encoding="utf-8") as source:
        return json.load(source)


def load_profile(path: Path) -> tuple[int, dict[str, float]]:
    document = load_json(path)
    if not isinstance(document, list) or not document or "count" not in document[0]:
        raise ValueError(f"invalid trtexec profile: {path}")
    timings = {
        str(row["name"]): float(row["averageMs"])
        for row in document[1:]
    }
    return int(document[0]["count"]), timings


def find_layers(
    path: Path, pattern: re.Pattern[str]
) -> dict[int, dict[str, Any]]:
    document = load_json(path)
    layers = document.get("Layers") if isinstance(document, dict) else None
    if not isinstance(layers, list):
        raise ValueError(f"invalid TensorRT layer-info JSON: {path}")

    matches: dict[int, dict[str, Any]] = {}
    for layer in layers:
        match = pattern.search(str(layer.get("Metadata", "")))
        if match is None:
            continue
        block = int(match.group(1))
        if block in matches:
            raise ValueError(f"duplicate Block3.{block} match in {path}")
        matches[block] = layer
    return matches


def datatype(layer: dict[str, Any], field: str) -> str:
    tensors = layer.get(field, [])
    return ";".join(str(tensor.get("Format/Datatype", "")) for tensor in tensors)


def main() -> None:
    args = parse_args()
    baseline_iterations, baseline_timings = load_profile(args.baseline_profile)
    candidate_iterations, candidate_timings = load_profile(args.candidate_profile)
    baseline_q = find_layers(args.baseline_layer_info, Q_MATMUL_PATTERN)
    candidate_q = find_layers(args.candidate_layer_info, Q_MATMUL_PATTERN)
    candidate_quantize = find_layers(args.candidate_layer_info, Q_ACTIVATION_PATTERN)

    blocks = sorted(baseline_q)
    if blocks != list(range(18)):
        raise ValueError(f"expected baseline Block3.0..17, got {blocks}")
    if sorted(candidate_q) != blocks or sorted(candidate_quantize) != blocks:
        raise ValueError("candidate Q MatMul/activation quantize layers are incomplete")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = {"baseline": 0.0, "candidate_q": 0.0, "quantize": 0.0}
    with args.output.open("wt", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "block",
                "v16_q_ms",
                "v17_activation_quantize_ms",
                "v17_q_ms",
                "v17_q_subchain_ms",
                "net_saved_ms",
                "net_speedup_percent",
                "v16_q_layer",
                "v16_input_dtype",
                "v16_output_dtype",
                "v17_quantize_layer",
                "v17_q_layer",
                "v17_input_dtype",
                "v17_output_dtype",
                "v16_tactic",
                "v17_tactic",
            ]
        )
        for block in blocks:
            old_layer = baseline_q[block]
            new_layer = candidate_q[block]
            quantize_layer = candidate_quantize[block]
            old_ms = baseline_timings[str(old_layer["Name"])]
            new_ms = candidate_timings[str(new_layer["Name"])]
            quantize_ms = candidate_timings[str(quantize_layer["Name"])]
            subchain_ms = new_ms + quantize_ms
            saved_ms = old_ms - subchain_ms
            speedup_percent = saved_ms / old_ms * 100.0
            totals["baseline"] += old_ms
            totals["candidate_q"] += new_ms
            totals["quantize"] += quantize_ms
            writer.writerow(
                [
                    f"block3.{block}",
                    f"{old_ms:.9f}",
                    f"{quantize_ms:.9f}",
                    f"{new_ms:.9f}",
                    f"{subchain_ms:.9f}",
                    f"{saved_ms:.9f}",
                    f"{speedup_percent:.6f}",
                    old_layer["Name"],
                    datatype(old_layer, "Inputs"),
                    datatype(old_layer, "Outputs"),
                    quantize_layer["Name"],
                    new_layer["Name"],
                    datatype(new_layer, "Inputs"),
                    datatype(new_layer, "Outputs"),
                    old_layer.get("TacticName", ""),
                    new_layer.get("TacticName", ""),
                ]
            )

    candidate_subchain = totals["candidate_q"] + totals["quantize"]
    saved = totals["baseline"] - candidate_subchain
    speedup_percent = saved / totals["baseline"] * 100.0
    print(f"[PASS] baseline iterations: {baseline_iterations}")
    print(f"[PASS] candidate iterations: {candidate_iterations}")
    print(f"[PASS] baseline Q GEMM: {totals['baseline']:.6f} ms")
    print(f"[PASS] candidate activation quantize: {totals['quantize']:.6f} ms")
    print(f"[PASS] candidate Q GEMM: {totals['candidate_q']:.6f} ms")
    print(f"[PASS] net saved: {saved:.6f} ms ({speedup_percent:.3f}%)")
    print(f"[PASS] output: {args.output}")


if __name__ == "__main__":
    main()
