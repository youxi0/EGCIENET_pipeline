#!/usr/bin/env python3

"""Convert the 18 Block3 fused spatial-reduction plugins to INT8 GEMM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


PLUGIN_OP = "EGCINET_FusedSpatialReduction"
PLUGIN_DOMAIN = "egcinet"
BLOCK_COUNT = 18
PACKED_K = 1280
CHANNELS = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse every Block3 Q-projection activation QuantizeLinear output "
            "as the INT8 activation of the fused spatial-reduction plugin, and "
            "quantize each packed SR weight per output channel."
        )
    )
    parser.add_argument("input", type=Path, help="source V24 ONNX model")
    parser.add_argument("output", type=Path, help="destination V25 ONNX model")
    return parser.parse_args()


def require_node(nodes: dict[str, gs.Node], name: str) -> gs.Node:
    try:
        return nodes[name]
    except KeyError as error:
        raise ValueError(f"required ONNX node not found: {name}") from error


def require_constant(tensor: gs.Tensor, label: str) -> np.ndarray:
    if not isinstance(tensor, gs.Constant):
        raise ValueError(f"{label} is not a constant initializer: {tensor.name}")
    return np.asarray(tensor.values)


def quantize_weight_per_output_channel(
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if weight.shape != (PACKED_K, CHANNELS):
        raise ValueError(f"unexpected Block3 SR weight shape: {weight.shape}")

    weight_fp32 = np.asarray(weight, dtype=np.float32)
    absolute_maximum = np.max(np.abs(weight_fp32), axis=0)
    weight_scale = absolute_maximum / np.float32(127.0)
    weight_scale = np.where(
        weight_scale > 0.0, weight_scale, np.float32(1.0)
    ).astype(np.float32)
    quantized_weight = np.clip(
        np.rint(weight_fp32 / weight_scale[np.newaxis, :]), -127, 127
    ).astype(np.int8)
    return np.ascontiguousarray(quantized_weight), weight_scale


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output ONNX paths must differ")

    graph = gs.import_onnx(onnx.load(args.input, load_external_data=True))
    nodes = {node.name: node for node in graph.nodes}
    maximum_weight_error = 0.0

    for block_index in range(BLOCK_COUNT):
        prefix = f"/model/block3.{block_index}/attn"
        plugin = require_node(nodes, f"{prefix}/FusedSpatialReduction")
        activation_quantize = require_node(
            nodes, f"{prefix}/q/MatMul_V22_Activation_QuantizeLinear"
        )

        if plugin.op != PLUGIN_OP or plugin.domain != PLUGIN_DOMAIN:
            raise ValueError(f"unexpected SR plugin node: {plugin.name}")
        if len(plugin.inputs) != 3 or len(plugin.outputs) != 1:
            raise ValueError(f"unexpected FP16 SR interface: {plugin.name}")
        if activation_quantize.op != "QuantizeLinear" or (
            len(activation_quantize.inputs) != 3 or
            len(activation_quantize.outputs) != 1
        ):
            raise ValueError(
                f"unexpected activation Q node: {activation_quantize.name}"
            )
        if activation_quantize.inputs[0].name != plugin.inputs[0].name:
            raise ValueError(
                f"Block3.{block_index} Q and SR do not share the same input: "
                f"{activation_quantize.inputs[0].name} != {plugin.inputs[0].name}"
            )

        activation_scale_array = require_constant(
            activation_quantize.inputs[1], "Q activation scale"
        )
        if activation_scale_array.size != 1:
            raise ValueError(
                f"Block3.{block_index} activation scale must be scalar"
            )
        activation_scale = np.float32(activation_scale_array.reshape(-1)[0])
        if not np.isfinite(activation_scale) or activation_scale <= 0.0:
            raise ValueError(
                f"invalid Block3.{block_index} activation scale: "
                f"{activation_scale}"
            )

        weight = require_constant(plugin.inputs[1], "FP16 SR weight")
        bias = require_constant(plugin.inputs[2], "FP16 SR bias")
        if bias.shape != (CHANNELS,) or bias.dtype != np.float16:
            raise ValueError(
                f"unexpected Block3.{block_index} SR bias: "
                f"shape={bias.shape}, dtype={bias.dtype}"
            )
        quantized_weight, weight_scale = quantize_weight_per_output_channel(
            weight
        )
        reconstructed_weight = (
            quantized_weight.astype(np.float32) * weight_scale[np.newaxis, :]
        )
        maximum_weight_error = max(
            maximum_weight_error,
            float(
                np.max(
                    np.abs(np.asarray(weight, dtype=np.float32) -
                           reconstructed_weight)
                )
            ),
        )
        accumulator_scale = np.ascontiguousarray(
            weight_scale * activation_scale, dtype=np.float32
        )

        quantized_activation = activation_quantize.outputs[0]
        quantized_activation.dtype = np.int8
        quantized_activation.shape = plugin.inputs[0].shape
        quantized_weight_tensor = gs.Constant(
            name=(
                f"model.rgb_encoder.block3.{block_index}.attn.sr.weight."
                "V25_KHKWCI_CO_INT8"
            ),
            # cuBLASLt 的普通 INT8 IMMA 路径只支持 TN。转成连续 [N,K]
            # 后可直接按 column-major [K,N] 使用，不需要运行时重排。
            values=np.ascontiguousarray(quantized_weight.T),
        )
        accumulator_scale_tensor = gs.Constant(
            name=(
                f"model.rgb_encoder.block3.{block_index}.attn.sr."
                "V25_ACCUM_DEQUANT_SCALE_FP32"
            ),
            values=accumulator_scale,
        )

        plugin.inputs = [
            quantized_activation,
            quantized_weight_tensor,
            plugin.inputs[2],
            accumulator_scale_tensor,
        ]
        plugin.attrs.update(
            {
                "int8_mode": 1,
                "input_layout": "BNC_INT8",
                "packed_weight_layout": "CO_KHKW_CI_ROW_MAJOR_INT8",
            }
        )

    graph.cleanup(remove_unused_node_outputs=True).toposort()
    output_model = gs.export_onnx(graph)
    onnx.checker.check_model(output_model)

    int8_plugins = [
        node
        for node in output_model.graph.node
        if node.domain == PLUGIN_DOMAIN
        and node.op_type == PLUGIN_OP
        and any(attribute.name == "int8_mode" for attribute in node.attribute)
    ]
    if len(int8_plugins) != BLOCK_COUNT:
        raise RuntimeError(
            f"expected {BLOCK_COUNT} INT8 Block3 SR plugins, "
            f"got {len(int8_plugins)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output_model, args.output)
    print(f"[PASS] quantized Block3 SR plugins: {len(int8_plugins)}")
    print(f"[INFO] maximum SR weight reconstruction error: {maximum_weight_error:.8f}")
    print(f"[PASS] output: {args.output}")


if __name__ == "__main__":
    main()
