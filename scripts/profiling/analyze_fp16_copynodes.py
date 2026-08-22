#!/usr/bin/env python3
"""Extract TensorRT Reformatting CopyNode connections from layer-info JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


COPY_NODE_PREFIX = "Reformatting CopyNode"


def tensor_dimensions(tensor: dict[str, Any] | None) -> str:
    if not tensor or "Dimensions" not in tensor:
        return ""
    return "x".join(str(value) for value in tensor["Dimensions"])


def tensor_format(tensor: dict[str, Any] | None) -> str:
    if not tensor:
        return ""
    return str(tensor.get("Format/Datatype", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record the producer and consumer around each TensorRT CopyNode."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("results/profile/fp16_layers.json"),
        help="TensorRT layer-info JSON",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results/profile/fp16_copynode_analysis.csv"),
        help="Output CSV",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional TensorRT per-layer profile JSON used to attach timing data",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as stream:
        layers = json.load(stream)["Layers"]

    profile_by_name: dict[str, dict[str, Any]] = {}
    if args.profile:
        with args.profile.open(encoding="utf-8") as stream:
            profile_rows = json.load(stream)
        profile_by_name = {
            str(row["name"]): row for row in profile_rows if "name" in row
        }

    producers: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    consumers: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)

    for layer in layers:
        layer_name = str(layer.get("Name", ""))
        for output_index, tensor in enumerate(layer.get("Outputs", [])):
            producers[str(tensor.get("Name", ""))].append(
                (layer_name, output_index, tensor)
            )
        for input_index, tensor in enumerate(layer.get("Inputs", [])):
            consumers[str(tensor.get("Name", ""))].append(
                (layer_name, input_index, tensor)
            )

    fieldnames = [
        "copy_index",
        "upstream_layer",
        "upstream_output_index",
        "upstream_tensor",
        "upstream_output_format_datatype",
        "copy_node",
        "copy_input_format_datatype",
        "copy_output_format_datatype",
        "average_ms",
        "median_ms",
        "percentage",
        "downstream_layer",
        "downstream_input_index",
        "downstream_tensor",
        "downstream_input_format_datatype",
        "tensor_dimensions",
    ]

    rows: list[dict[str, str | int]] = []
    copy_index = 0
    for layer in layers:
        copy_name = str(layer.get("Name", ""))
        if not copy_name.startswith(COPY_NODE_PREFIX):
            continue

        copy_index += 1
        copy_input = layer["Inputs"][0]
        copy_output = layer["Outputs"][0]
        profile = profile_by_name.get(copy_name, {})
        input_name = str(copy_input.get("Name", ""))
        output_name = str(copy_output.get("Name", ""))

        upstream_matches = [
            match for match in producers.get(input_name, []) if match[0] != copy_name
        ]
        downstream_matches = [
            match for match in consumers.get(output_name, []) if match[0] != copy_name
        ]

        if not upstream_matches:
            upstream_matches = [("<Network Input>", -1, copy_input)]
        if not downstream_matches:
            downstream_matches = [("<Network Output>", -1, copy_output)]

        for upstream, output_index, upstream_tensor in upstream_matches:
            for downstream, input_index, downstream_tensor in downstream_matches:
                input_dims = tensor_dimensions(copy_input)
                output_dims = tensor_dimensions(copy_output)
                dimensions = (
                    input_dims
                    if input_dims == output_dims
                    else f"{input_dims} -> {output_dims}"
                )
                rows.append(
                    {
                        "copy_index": copy_index,
                        "upstream_layer": upstream,
                        "upstream_output_index": (
                            "" if output_index < 0 else output_index
                        ),
                        "upstream_tensor": input_name,
                        "upstream_output_format_datatype": tensor_format(
                            upstream_tensor
                        ),
                        "copy_node": copy_name,
                        "copy_input_format_datatype": tensor_format(copy_input),
                        "copy_output_format_datatype": tensor_format(copy_output),
                        "average_ms": profile.get("averageMs", ""),
                        "median_ms": profile.get("medianMs", ""),
                        "percentage": profile.get("percentage", ""),
                        "downstream_layer": downstream,
                        "downstream_input_index": (
                            "" if input_index < 0 else input_index
                        ),
                        "downstream_tensor": output_name,
                        "downstream_input_format_datatype": tensor_format(
                            downstream_tensor
                        ),
                        "tensor_dimensions": dimensions,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CopyNodes: {copy_index}")
    print(f"CSV rows: {len(rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
