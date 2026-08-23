#!/usr/bin/env python3

"""Insert one shared Q/DQ pair before every Block3 attention module."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


BLOCK_COUNT = 18
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / "models/egcienet_352_multiclass_int8.cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert a shared activation Q/DQ pair after each Block3 norm1 "
            "and feed both the attention q and sr/kv branches from its DQ output."
        )
    )
    parser.add_argument("input", type=Path, help="source ONNX model")
    parser.add_argument("output", type=Path, help="destination ONNX model")
    parser.add_argument(
        "--calibration-cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="TensorRT calibration cache containing Block3 norm1 output scales",
    )
    return parser.parse_args()


def read_calibration_scales(cache_path: Path) -> dict[str, float]:
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)

    scales: dict[str, float] = {}
    with cache_path.open("rt", encoding="utf-8") as cache:
        header = cache.readline().strip()
        if not header.startswith("TRT-"):
            raise ValueError(f"invalid TensorRT calibration cache: {cache_path}")

        for line_number, line in enumerate(cache, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                tensor_name, hexadecimal = line.rsplit(":", maxsplit=1)
                encoded = bytes.fromhex(hexadecimal.strip())
                if len(encoded) != 4:
                    raise ValueError("scale must contain four bytes")
                scale = struct.unpack(">f", encoded)[0]
            except ValueError as error:
                raise ValueError(
                    f"invalid calibration entry at line {line_number}: {line}"
                ) from error

            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"invalid calibration scale for {tensor_name}: {scale}"
                )
            scales[tensor_name] = scale
    return scales


def require_node(nodes: dict[str, gs.Node], name: str) -> gs.Node:
    try:
        return nodes[name]
    except KeyError as error:
        raise ValueError(f"required ONNX node not found: {name}") from error


def replace_tensor_input(
    node: gs.Node,
    old_tensor: gs.Tensor,
    new_tensor: gs.Tensor,
) -> None:
    replacement_count = sum(tensor is old_tensor for tensor in node.inputs)
    if replacement_count != 1:
        raise ValueError(
            f"{node.name} must consume {old_tensor.name} exactly once, got "
            f"{replacement_count}"
        )
    node.inputs = [
        new_tensor if tensor is old_tensor else tensor for tensor in node.inputs
    ]


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output ONNX paths must differ")

    calibration_scales = read_calibration_scales(args.calibration_cache)
    graph = gs.import_onnx(onnx.load(args.input, load_external_data=True))
    nodes = {node.name: node for node in graph.nodes}
    inserted_nodes: list[gs.Node] = []

    for block_index in range(BLOCK_COUNT):
        prefix = f"/model/block3.{block_index}"
        norm = require_node(nodes, f"{prefix}/norm1/Add_1")
        q_matmul = require_node(nodes, f"{prefix}/attn/q/MatMul")
        sr_transpose = require_node(nodes, f"{prefix}/attn/Transpose_1")
        residual_add = require_node(nodes, f"{prefix}/Add")

        if len(norm.outputs) != 1:
            raise ValueError(f"{norm.name} must have exactly one output")
        source = norm.outputs[0]
        if source.dtype != np.float16:
            raise ValueError(
                f"{source.name} must be FP16, got {source.dtype}"
            )

        expected_consumers = {q_matmul.name, sr_transpose.name}
        actual_consumers = {node.name for node in source.outputs}
        if actual_consumers != expected_consumers:
            raise ValueError(
                f"unexpected consumers of {source.name}: "
                f"{sorted(actual_consumers)}"
            )
        if source in residual_add.inputs:
            raise ValueError(
                f"residual path unexpectedly consumes attention input: "
                f"{residual_add.name}"
            )

        try:
            cache_scale = calibration_scales[source.name]
        except KeyError as error:
            raise ValueError(
                f"calibration scale not found for {source.name}"
            ) from error

        # 现有显式 Q/DQ 图使用 FP16 activation scale。Q 和 DQ 各保留一个
        # initializer，并使用相同的 FP16 舍入值，避免两端 scale 不一致。
        scale_value = np.float16(cache_scale)
        quantize_scale = gs.Constant(
            name=f"{source.name}_scale_1",
            values=np.asarray(scale_value, dtype=np.float16),
        )
        quantize_zero_point = gs.Constant(
            name=f"{source.name}_zero_point_1",
            values=np.asarray(0, dtype=np.int8),
        )
        dequantize_scale = gs.Constant(
            name=f"{source.name}_scale_2",
            values=np.asarray(scale_value, dtype=np.float16),
        )
        dequantize_zero_point = gs.Constant(
            name=f"{source.name}_zero_point_2",
            values=np.asarray(0, dtype=np.int8),
        )
        quantized = gs.Variable(
            name=f"{source.name}_QuantizeLinear_Output",
            dtype=np.int8,
            shape=source.shape,
        )
        dequantized = gs.Variable(
            name=f"{source.name}_DequantizeLinear_Output",
            dtype=np.float16,
            shape=source.shape,
        )
        quantize = gs.Node(
            op="QuantizeLinear",
            name=f"{source.name}_QuantizeLinear",
            inputs=[source, quantize_scale, quantize_zero_point],
            outputs=[quantized],
        )
        dequantize = gs.Node(
            op="DequantizeLinear",
            name=f"{source.name}_DequantizeLinear",
            inputs=[quantized, dequantize_scale, dequantize_zero_point],
            outputs=[dequantized],
        )

        # 一组 Q/DQ 同时服务 q 和 sr/kv 两个 attention 分支。这里只改这两个
        # consumer；Block3 的残差 Add 继续读取 norm1 之前的原始 block 输入。
        replace_tensor_input(q_matmul, source, dequantized)
        replace_tensor_input(sr_transpose, source, dequantized)
        inserted_nodes.extend([quantize, dequantize])

        print(
            f"[INFO] Block3.{block_index} attention input scale: "
            f"cache={cache_scale:.9g}, fp16={float(scale_value):.9g}"
        )

    graph.nodes.extend(inserted_nodes)
    graph.cleanup(remove_unused_node_outputs=True).toposort()
    output_model = gs.export_onnx(graph)
    onnx.checker.check_model(output_model)

    output_nodes = {node.name: node for node in output_model.graph.node}
    for block_index in range(BLOCK_COUNT):
        prefix = f"/model/block3.{block_index}"
        source_name = f"{prefix}/norm1/Add_1_output_0"
        quantize_name = f"{source_name}_QuantizeLinear"
        dequantize_name = f"{source_name}_DequantizeLinear"
        q_matmul = output_nodes[f"{prefix}/attn/q/MatMul"]
        sr_transpose = output_nodes[f"{prefix}/attn/Transpose_1"]
        residual_add = output_nodes[f"{prefix}/Add"]

        if quantize_name not in output_nodes or dequantize_name not in output_nodes:
            raise RuntimeError(f"missing exported Q/DQ for Block3.{block_index}")
        dequantized_name = f"{source_name}_DequantizeLinear_Output"
        if q_matmul.input[0] != dequantized_name:
            raise RuntimeError(f"q branch is not rewired in Block3.{block_index}")
        if sr_transpose.input[0] != dequantized_name:
            raise RuntimeError(f"sr/kv branch is not rewired in Block3.{block_index}")
        if dequantized_name in residual_add.input:
            raise RuntimeError(
                f"residual branch was rewired in Block3.{block_index}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output_model, args.output)
    print(f"[PASS] shared Block3 attention Q/DQ pairs: {BLOCK_COUNT}")
    print(f"[PASS] output: {args.output}")


if __name__ == "__main__":
    main()
