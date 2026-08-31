#!/usr/bin/env python3

"""Share one calibrated Block2 activation Q between Q MatMul and fused SR."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct

import numpy as np
import onnx
import onnx_graphsurgeon as gs


BLOCK_COUNT = 4
CHANNELS = 128
PLUGIN_OP = "EGCINET_FusedSpatialReduction"
PLUGIN_DOMAIN = "egcinet"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / "models/egcienet_352_multiclass_int8.cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert Block2 Q-projection QDQ at the norm1/SR split and feed "
            "the same INT8 activation to the fused spatial-reduction plugin."
        )
    )
    parser.add_argument("input", type=Path, help="source V26 ONNX model")
    parser.add_argument("output", type=Path, help="destination V27 ONNX model")
    parser.add_argument(
        "--calibration-cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="TensorRT calibration cache containing Block2 norm1 scales",
    )
    return parser.parse_args()


def read_calibration_scales(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    scales: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as cache:
        header = cache.readline().strip()
        if "EntropyCalibration" not in header:
            raise ValueError(f"unsupported calibration cache header: {header}")
        for line_number, line in enumerate(cache, start=2):
            if ": " not in line:
                continue
            name, encoded = line.rstrip().split(": ", 1)
            try:
                value = struct.unpack(">f", bytes.fromhex(encoded))[0]
            except (ValueError, struct.error) as error:
                raise ValueError(
                    f"invalid calibration entry at line {line_number}"
                ) from error
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"invalid calibration scale for {name}: {value}")
            scales[name] = value
    return scales


def require_node(nodes: dict[str, gs.Node], name: str) -> gs.Node:
    try:
        return nodes[name]
    except KeyError as error:
        raise ValueError(f"required ONNX node not found: {name}") from error


def require_constant(tensor: gs.Tensor, label: str) -> np.ndarray:
    if not isinstance(tensor, gs.Constant):
        raise ValueError(f"{label} is not a constant initializer: {tensor.name}")
    return np.asarray(tensor.values)


def make_qdq(
    prefix: str,
    source: gs.Tensor,
    scale: np.ndarray,
    zero_point: np.ndarray,
    output_dtype: np.dtype,
    axis: int | None = None,
) -> tuple[list[gs.Node], gs.Variable, gs.Variable]:
    quantized = gs.Variable(
        name=f"{prefix}_QuantizeLinear_Output",
        dtype=np.int8,
        shape=source.shape,
    )
    dequantized = gs.Variable(
        name=f"{prefix}_DequantizeLinear_Output",
        dtype=output_dtype,
        shape=source.shape,
    )
    attributes = {} if axis is None else {"axis": axis}
    quantize = gs.Node(
        op="QuantizeLinear",
        name=f"{prefix}_QuantizeLinear",
        attrs=attributes,
        inputs=[
            source,
            gs.Constant(name=f"{prefix}_scale_1", values=scale.copy()),
            gs.Constant(
                name=f"{prefix}_zero_point_1", values=zero_point.copy()
            ),
        ],
        outputs=[quantized],
    )
    dequantize = gs.Node(
        op="DequantizeLinear",
        name=f"{prefix}_DequantizeLinear",
        attrs=attributes,
        inputs=[
            quantized,
            gs.Constant(name=f"{prefix}_scale_2", values=scale.copy()),
            gs.Constant(
                name=f"{prefix}_zero_point_2", values=zero_point.copy()
            ),
        ],
        outputs=[dequantized],
    )
    return [quantize, dequantize], quantized, dequantized


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
    maximum_weight_error = 0.0

    for block_index in range(BLOCK_COUNT):
        prefix = f"/model/block2.{block_index}/attn"
        q_matmul = require_node(nodes, f"{prefix}/q/MatMul")
        plugin = require_node(nodes, f"{prefix}/FusedSpatialReduction")
        if plugin.op != PLUGIN_OP or plugin.domain != PLUGIN_DOMAIN:
            raise ValueError(f"unexpected SR plugin: {plugin.name}")
        if len(q_matmul.inputs) != 2 or len(plugin.inputs) != 4:
            raise ValueError(f"unexpected Block2 interface: {prefix}")

        activation = q_matmul.inputs[0]
        if plugin.inputs[0] is not activation:
            raise ValueError(
                f"Q and SR do not share the same activation in Block2.{block_index}"
            )
        if activation.dtype != np.float16:
            raise ValueError(
                f"unexpected Block2 activation dtype: {activation.dtype}"
            )
        if int(plugin.attrs.get("int8_mode", -1)) != 2:
            raise ValueError(f"expected fused-Q SR mode in {plugin.name}")

        try:
            cached_scale = calibration_scales[activation.name]
        except KeyError as error:
            raise ValueError(
                f"calibration scale not found for {activation.name}"
            ) from error
        activation_scale = np.asarray(np.float16(cached_scale), dtype=np.float16)
        plugin_scale = np.float16(plugin.attrs.get("activation_scale", 0.0))
        if plugin_scale != activation_scale:
            raise ValueError(
                f"plugin/cache scale mismatch in Block2.{block_index}: "
                f"{plugin_scale} != {activation_scale}"
            )

        activation_nodes, quantized_activation, dequantized_activation = make_qdq(
            f"{prefix}/q/MatMul_V27_Shared_Activation",
            activation,
            activation_scale,
            np.asarray(0, dtype=np.int8),
            np.dtype(np.float16),
        )

        weight_tensor = q_matmul.inputs[1]
        weight = require_constant(weight_tensor, "Block2 Q weight")
        if weight.dtype != np.float16 or weight.shape != (CHANNELS, CHANNELS):
            raise ValueError(
                f"unexpected Block2 Q weight: shape={weight.shape}, "
                f"dtype={weight.dtype}"
            )
        weight_fp32 = weight.astype(np.float32)
        absolute_maximum = np.max(np.abs(weight_fp32), axis=0)
        weight_scale = np.where(
            absolute_maximum > 0.0,
            absolute_maximum / np.float32(127.0),
            np.float32(1.0),
        ).astype(np.float16)
        weight_zero_point = np.zeros((CHANNELS,), dtype=np.int8)
        weight_nodes, _, dequantized_weight = make_qdq(
            f"{prefix}/q/MatMul_V27_Shared_Weight",
            weight_tensor,
            weight_scale,
            weight_zero_point,
            np.dtype(np.float16),
            axis=1,
        )
        quantized_weight = np.clip(
            np.rint(weight_fp32 / weight_scale.astype(np.float32)[None, :]),
            -128,
            127,
        ).astype(np.int8)
        reconstructed_weight = (
            quantized_weight.astype(np.float32)
            * weight_scale.astype(np.float32)[None, :]
        )
        maximum_weight_error = max(
            maximum_weight_error,
            float(np.max(np.abs(weight_fp32 - reconstructed_weight))),
        )

        q_matmul.inputs = [dequantized_activation, dequantized_weight]
        plugin.inputs[0] = quantized_activation
        plugin.attrs.update(
            {
                "int8_mode": 1,
                "activation_scale": 0.0,
                "input_layout": "BNC_INT8",
            }
        )
        inserted_nodes.extend(activation_nodes)
        inserted_nodes.extend(weight_nodes)

        print(
            f"[INFO] Block2.{block_index} shared activation scale: "
            f"cache={cached_scale:.9g}, fp16={float(activation_scale):.9g}"
        )

    graph.nodes.extend(inserted_nodes)
    graph.cleanup(remove_unused_node_outputs=True).toposort()
    output_model = gs.export_onnx(graph)
    onnx.checker.check_model(output_model)

    output_nodes = {node.name: node for node in output_model.graph.node}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in output_model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    for block_index in range(BLOCK_COUNT):
        prefix = f"/model/block2.{block_index}/attn"
        activation_q = (
            f"{prefix}/q/MatMul_V27_Shared_Activation_QuantizeLinear_Output"
        )
        activation_dq = (
            f"{prefix}/q/MatMul_V27_Shared_Activation_DequantizeLinear_Output"
        )
        q_matmul = output_nodes[f"{prefix}/q/MatMul"]
        plugin = output_nodes[f"{prefix}/FusedSpatialReduction"]
        if q_matmul.input[0] != activation_dq or plugin.input[0] != activation_q:
            raise RuntimeError(f"Block2.{block_index} shared Q wiring failed")
        consumer_names = {node.name for node in consumers[activation_q]}
        expected_consumers = {
            f"{prefix}/q/MatMul_V27_Shared_Activation_DequantizeLinear",
            plugin.name,
        }
        if consumer_names != expected_consumers:
            raise RuntimeError(
                f"unexpected shared Q consumers in Block2.{block_index}: "
                f"{sorted(consumer_names)}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output_model, args.output)
    print(f"[PASS] shared Block2 Q/SR activation quantizers: {BLOCK_COUNT}")
    print(f"[INFO] maximum Q weight reconstruction error: {maximum_weight_error:.8f}")
    print(f"[PASS] output: {args.output}")


if __name__ == "__main__":
    main()
