#!/usr/bin/env python3

"""Merge trtexec profile/layer-info JSON and sort layers by average time."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a descending CSV with average time, layer name/type, "
            "and input/output tensor names."
        )
    )
    parser.add_argument("profile", type=Path, help="trtexec --exportProfile JSON")
    parser.add_argument(
        "layer_info", type=Path, help="trtexec --exportLayerInfo JSON"
    )
    parser.add_argument("output", type=Path, help="destination CSV")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rt", encoding="utf-8") as source:
        return json.load(source)


def tensor_names(layer: dict[str, Any], field: str) -> str:
    tensors = layer.get(field, [])
    return ";".join(str(tensor.get("Name", "")) for tensor in tensors)


def main() -> None:
    args = parse_args()
    profile = load_json(args.profile)
    layer_info_document = load_json(args.layer_info)

    if not isinstance(profile, list) or not profile:
        raise ValueError("profile JSON must be a non-empty list")
    if "count" not in profile[0]:
        raise ValueError("profile JSON is missing its iteration-count header")
    if not isinstance(layer_info_document, dict):
        raise ValueError("layer-info JSON must be an object")

    layers = layer_info_document.get("Layers")
    if not isinstance(layers, list):
        raise ValueError("layer-info JSON does not contain a Layers list")

    layer_by_name: dict[str, dict[str, Any]] = {}
    for layer in layers:
        name = layer.get("Name")
        if not isinstance(name, str) or not name:
            raise ValueError("layer-info entry has no valid Name")
        if name in layer_by_name:
            raise ValueError(f"duplicate layer-info name: {name}")
        layer_by_name[name] = layer

    timing_rows = profile[1:]
    profile_names = [row.get("name") for row in timing_rows]
    if len(profile_names) != len(set(profile_names)):
        raise ValueError("profile JSON contains duplicate layer names")

    missing = sorted(set(profile_names) - set(layer_by_name))
    extra = sorted(set(layer_by_name) - set(profile_names))
    if missing or extra:
        raise ValueError(
            "profile/layer-info mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    timing_rows.sort(key=lambda row: float(row["averageMs"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wt", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            ["average_ms", "layer_name", "layer_type", "input_names", "output_names"]
        )
        for timing in timing_rows:
            layer = layer_by_name[timing["name"]]
            writer.writerow(
                [
                    f'{float(timing["averageMs"]):.9f}',
                    timing["name"],
                    layer.get("LayerType", ""),
                    tensor_names(layer, "Inputs"),
                    tensor_names(layer, "Outputs"),
                ]
            )

    print(f"[PASS] iterations: {profile[0]['count']}")
    print(f"[PASS] layers: {len(timing_rows)}")
    print(f"[PASS] output: {args.output}")


if __name__ == "__main__":
    main()
