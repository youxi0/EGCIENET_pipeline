#!/usr/bin/env python3

"""Fuse calibrated Block1/2 activation quantization into their SR plugins."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np
import onnx
import onnx_graphsurgeon as gs


PLUGIN_OP = "EGCINET_FusedSpatialReduction"
PLUGIN_DOMAIN = "egcinet"
STAGES = {
    1: {"blocks": 3, "packed_k": 4096, "channels": 64},
    2: {"blocks": 4, "packed_k": 2048, "channels": 128},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Block1/2 fused SR weights to per-output-channel INT8 "
            "and quantize the FP activation while packing SR windows."
        )
    )
    parser.add_argument("input", type=Path, help="source V25 ONNX model")
    parser.add_argument("output", type=Path, help="destination V26 ONNX model")
    parser.add_argument(
        "calibration_cache",
        type=Path,
        help="TensorRT entropy calibration cache containing norm1 scales",
    )
    return parser.parse_args()


def load_calibration_scales(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    scales: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as cache:
        header = cache.readline().strip()
        if "EntropyCalibration" not in header:
            raise ValueError(f"unsupported calibration cache header: {header}")
        for line in cache:
            if ": " not in line:
                continue
            name, encoded = line.rstrip().split(": ", 1)
            try:
                value = struct.unpack(">f", bytes.fromhex(encoded))[0]
            except (ValueError, struct.error) as error:
                raise ValueError(f"invalid cache entry for {name}") from error
            if not np.isfinite(value) or value <= 0.0:
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


def quantize_weight(
    weight: np.ndarray, packed_k: int, channels: int
) -> tuple[np.ndarray, np.ndarray, float]:
    if weight.shape != (packed_k, channels):
        raise ValueError(f"unexpected SR weight shape: {weight.shape}")
    weight_fp32 = np.asarray(weight, dtype=np.float32)
    absolute_maximum = np.max(np.abs(weight_fp32), axis=0)
    weight_scale = np.where(
        absolute_maximum > 0.0,
        absolute_maximum / np.float32(127.0),
        np.float32(1.0),
    ).astype(np.float32)
    quantized = np.clip(
        np.rint(weight_fp32 / weight_scale[np.newaxis, :]), -127, 127
    ).astype(np.int8)
    reconstructed = quantized.astype(np.float32) * weight_scale[np.newaxis, :]
    error = float(np.max(np.abs(weight_fp32 - reconstructed)))
    return np.ascontiguousarray(quantized.T), weight_scale, error


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output ONNX paths must differ")

    calibration_scales = load_calibration_scales(args.calibration_cache)
    graph = gs.import_onnx(onnx.load(args.input, load_external_data=True))
    nodes = {node.name: node for node in graph.nodes}
    maximum_weight_error = 0.0
    fused_quantize_plugins = 0

    for stage, configuration in STAGES.items():
        for block_index in range(configuration["blocks"]):
            prefix = f"/model/block{stage}.{block_index}/attn"
            plugin = require_node(nodes, f"{prefix}/FusedSpatialReduction")
            if plugin.op != PLUGIN_OP or plugin.domain != PLUGIN_DOMAIN:
                raise ValueError(f"unexpected SR plugin node: {plugin.name}")
            if len(plugin.inputs) != 3 or len(plugin.outputs) != 1:
                raise ValueError(f"unexpected FP16 SR interface: {plugin.name}")

            activation = plugin.inputs[0]
            try:
                cached_scale = calibration_scales[activation.name]
            except KeyError as error:
                raise ValueError(
                    f"calibration scale not found for {activation.name}"
                ) from error

            # 显式 QDQ 的 scale 是 FP16；插件用同一舍入值量化并计算
            # accumulator scale，保证 fused-Q 与独立 QuantizeLinear 等价。
            activation_scale_value = np.float16(cached_scale)
            fused_quantize_plugins += 1

            weight = require_constant(plugin.inputs[1], "FP16 SR weight")
            bias = require_constant(plugin.inputs[2], "FP16 SR bias")
            channels = configuration["channels"]
            packed_k = configuration["packed_k"]
            if bias.shape != (channels,) or bias.dtype != np.float16:
                raise ValueError(
                    f"unexpected Block{stage}.{block_index} SR bias: "
                    f"shape={bias.shape}, dtype={bias.dtype}"
                )
            quantized_weight, weight_scale, weight_error = quantize_weight(
                weight, packed_k, channels
            )
            maximum_weight_error = max(maximum_weight_error, weight_error)
            accumulator_scale = np.ascontiguousarray(
                weight_scale * np.float32(activation_scale_value),
                dtype=np.float32,
            )

            plugin.inputs = [
                activation,
                gs.Constant(
                    name=(
                        f"model.rgb_encoder.block{stage}.{block_index}.attn."
                        "sr.weight.V26_CO_KHKWCI_INT8"
                    ),
                    values=quantized_weight,
                ),
                plugin.inputs[2],
                gs.Constant(
                    name=(
                        f"model.rgb_encoder.block{stage}.{block_index}.attn."
                        "sr.V26_ACCUM_DEQUANT_SCALE_FP32"
                    ),
                    values=accumulator_scale,
                ),
            ]
            plugin.attrs.update(
                {
                    "int8_mode": 2,
                    "activation_scale": float(
                        np.float32(activation_scale_value)
                    ),
                    "input_layout": "BNC_FP_PACK_INT8",
                    "packed_weight_layout": "CO_KHKW_CI_ROW_MAJOR_INT8",
                }
            )

    graph.cleanup(remove_unused_node_outputs=True).toposort()
    output_model = gs.export_onnx(graph)
    onnx.checker.check_model(output_model)

    int8_plugins = []
    for node in output_model.graph.node:
        if node.domain != PLUGIN_DOMAIN or node.op_type != PLUGIN_OP:
            continue
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if attributes.get("int8_mode") in (1, 2):
            int8_plugins.append(node)
    if fused_quantize_plugins != 7 or len(int8_plugins) != 25:
        raise RuntimeError(
            f"expected 7 fused-Q and 25 total INT8 SR plugins, got "
            f"{fused_quantize_plugins} and {len(int8_plugins)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output_model, args.output)
    print(f"[PASS] fused Block1/2 activation quantizers: {fused_quantize_plugins}")
    print(f"[PASS] total INT8 SR plugins: {len(int8_plugins)}")
    print(f"[INFO] maximum new weight reconstruction error: {maximum_weight_error:.8f}")
    print(f"[PASS] output: {args.output}")


if __name__ == "__main__":
    main()
