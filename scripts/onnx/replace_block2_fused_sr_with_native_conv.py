#!/usr/bin/env python3

"""Replace the four Block2 fused-SR plugins with native QDQ Conv nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


BLOCK_COUNT = 4
CHANNELS = 128
INPUT_HEIGHT = 44
INPUT_WIDTH = 44
OUTPUT_HEIGHT = 11
OUTPUT_WIDTH = 11
KERNEL_HEIGHT = 4
KERNEL_WIDTH = 4
PACKED_K = CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH
PLUGIN_OP = "EGCINET_FusedSpatialReduction"
PLUGIN_DOMAIN = "egcinet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace V27 Block2 fused spatial-reduction plugins with native "
            "ONNX Conv nodes that reuse the shared Q-projection QDQ input."
        )
    )
    parser.add_argument("input", type=Path, help="source V27 ONNX model")
    parser.add_argument("output", type=Path, help="destination V28 ONNX model")
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


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output ONNX paths must differ")

    graph = gs.import_onnx(onnx.load(args.input, load_external_data=True))
    nodes = {node.name: node for node in graph.nodes}
    inserted_nodes: list[gs.Node] = []
    maximum_scale_rounding_error = 0.0

    for block_index in range(BLOCK_COUNT):
        prefix = f"/model/block2.{block_index}/attn"
        plugin = require_node(nodes, f"{prefix}/FusedSpatialReduction")
        activation_quantize = require_node(
            nodes,
            f"{prefix}/q/MatMul_V27_Shared_Activation_QuantizeLinear",
        )
        activation_dequantize = require_node(
            nodes,
            f"{prefix}/q/MatMul_V27_Shared_Activation_DequantizeLinear",
        )
        if plugin.op != PLUGIN_OP or plugin.domain != PLUGIN_DOMAIN:
            raise ValueError(f"unexpected SR plugin: {plugin.name}")
        if len(plugin.inputs) != 4 or len(plugin.outputs) != 1:
            raise ValueError(f"unexpected plugin interface: {plugin.name}")
        if int(plugin.attrs.get("stage", -1)) != 2 or int(
            plugin.attrs.get("int8_mode", -1)
        ) != 1:
            raise ValueError(f"expected Block2 mode-1 plugin: {plugin.name}")
        if activation_quantize.outputs[0] is not plugin.inputs[0]:
            raise ValueError(f"plugin does not use shared Q: {plugin.name}")
        if activation_dequantize.inputs[0] is not plugin.inputs[0]:
            raise ValueError(f"activation DQ does not use shared Q: {prefix}")

        activation_scale_array = require_constant(
            activation_quantize.inputs[1], "Block2 activation scale"
        )
        if activation_scale_array.size != 1:
            raise ValueError(f"activation scale must be scalar: {prefix}")
        activation_scale = np.float32(activation_scale_array.reshape(-1)[0])
        if not np.isfinite(activation_scale) or activation_scale <= 0.0:
            raise ValueError(f"invalid activation scale: {activation_scale}")

        packed_weight = require_constant(plugin.inputs[1], "INT8 SR weight")
        bias = require_constant(plugin.inputs[2], "FP16 SR bias")
        accumulator_scale = require_constant(
            plugin.inputs[3], "FP32 accumulator scale"
        )
        if packed_weight.dtype != np.int8 or packed_weight.shape != (
            CHANNELS,
            PACKED_K,
        ):
            raise ValueError(
                f"unexpected SR weight in Block2.{block_index}: "
                f"shape={packed_weight.shape}, dtype={packed_weight.dtype}"
            )
        if bias.dtype != np.float16 or bias.shape != (CHANNELS,):
            raise ValueError(
                f"unexpected SR bias in Block2.{block_index}: "
                f"shape={bias.shape}, dtype={bias.dtype}"
            )
        if accumulator_scale.dtype != np.float32 or accumulator_scale.shape != (
            CHANNELS,
        ):
            raise ValueError(
                f"unexpected accumulator scale in Block2.{block_index}"
            )

        # Plugin storage is [CO, KH, KW, CI]. Restore ONNX Conv's
        # [CO, CI, KH, KW] order without changing any quantized coefficient.
        conv_weight = np.ascontiguousarray(
            packed_weight.reshape(
                CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH, CHANNELS
            ).transpose(0, 3, 1, 2)
        )
        weight_scale_fp32 = accumulator_scale / activation_scale
        weight_scale = weight_scale_fp32.astype(np.float16)
        maximum_scale_rounding_error = max(
            maximum_scale_rounding_error,
            float(
                np.max(
                    np.abs(
                        weight_scale_fp32 - weight_scale.astype(np.float32)
                    )
                )
            ),
        )
        dequantized_weight = gs.Variable(
            name=f"{prefix}/sr/V28_Weight_DequantizeLinear_Output",
            dtype=np.float16,
            shape=(CHANNELS, CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH),
        )
        weight_dequantize = gs.Node(
            op="DequantizeLinear",
            name=f"{prefix}/sr/V28_Weight_DequantizeLinear",
            attrs={"axis": 0},
            inputs=[
                gs.Constant(
                    name=f"{prefix}/sr/V28_Weight_INT8",
                    values=conv_weight,
                ),
                gs.Constant(
                    name=f"{prefix}/sr/V28_Weight_scale",
                    values=np.ascontiguousarray(weight_scale),
                ),
                gs.Constant(
                    name=f"{prefix}/sr/V28_Weight_zero_point",
                    values=np.zeros((CHANNELS,), dtype=np.int8),
                ),
            ],
            outputs=[dequantized_weight],
        )

        # Keep the shared activation quantized while changing its layout.  The
        # activation DQ must sit directly on the Conv input; moving it ahead of
        # Transpose/Reshape allows TensorRT to discard the INT8 boundary and
        # select an FP32 convolution.
        #
        # BNC -> BCN -> NCHW. Tokens are row-major flattened HxW positions.
        transposed_input = gs.Variable(
            name=f"{prefix}/sr/V28_Input_Transpose_Output",
            dtype=np.int8,
            shape=(1, CHANNELS, INPUT_HEIGHT * INPUT_WIDTH),
        )
        input_transpose = gs.Node(
            op="Transpose",
            name=f"{prefix}/sr/V28_Input_Transpose",
            attrs={"perm": [0, 2, 1]},
            inputs=[activation_quantize.outputs[0]],
            outputs=[transposed_input],
        )
        quantized_nchw_input = gs.Variable(
            name=f"{prefix}/sr/V28_Input_Reshape_Output",
            dtype=np.int8,
            shape=(1, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
        )
        input_reshape = gs.Node(
            op="Reshape",
            name=f"{prefix}/sr/V28_Input_Reshape",
            inputs=[
                transposed_input,
                gs.Constant(
                    name=f"{prefix}/sr/V28_Input_Shape",
                    values=np.asarray(
                        [1, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH],
                        dtype=np.int64,
                    ),
                ),
            ],
            outputs=[quantized_nchw_input],
        )
        nchw_input = gs.Variable(
            name=f"{prefix}/sr/V28_Input_DequantizeLinear_Output",
            dtype=np.float16,
            shape=(1, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
        )
        input_dequantize = gs.Node(
            op="DequantizeLinear",
            name=f"{prefix}/sr/V28_Input_DequantizeLinear",
            inputs=[
                quantized_nchw_input,
                activation_quantize.inputs[1],
                activation_quantize.inputs[2],
            ],
            outputs=[nchw_input],
        )
        conv_output = gs.Variable(
            name=f"{prefix}/sr/V28_Conv_Output",
            dtype=np.float16,
            shape=(1, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH),
        )
        convolution = gs.Node(
            op="Conv",
            name=f"{prefix}/sr/V28_Conv",
            attrs={
                "dilations": [1, 1],
                "group": 1,
                "kernel_shape": [KERNEL_HEIGHT, KERNEL_WIDTH],
                "pads": [0, 0, 0, 0],
                "strides": [KERNEL_HEIGHT, KERNEL_WIDTH],
            },
            inputs=[nchw_input, dequantized_weight, plugin.inputs[2]],
            outputs=[conv_output],
        )

        flattened_output = gs.Variable(
            name=f"{prefix}/sr/V28_Output_Reshape_Output",
            dtype=np.float16,
            shape=(1, CHANNELS, OUTPUT_HEIGHT * OUTPUT_WIDTH),
        )
        output_reshape = gs.Node(
            op="Reshape",
            name=f"{prefix}/sr/V28_Output_Reshape",
            inputs=[
                conv_output,
                gs.Constant(
                    name=f"{prefix}/sr/V28_Output_Shape",
                    values=np.asarray(
                        [1, CHANNELS, OUTPUT_HEIGHT * OUTPUT_WIDTH],
                        dtype=np.int64,
                    ),
                ),
            ],
            outputs=[flattened_output],
        )
        original_output = plugin.outputs[0]
        plugin.outputs.clear()
        output_transpose = gs.Node(
            op="Transpose",
            name=f"{prefix}/sr/V28_Output_Transpose",
            attrs={"perm": [0, 2, 1]},
            inputs=[flattened_output],
            outputs=[original_output],
        )
        inserted_nodes.extend(
            [
                weight_dequantize,
                input_transpose,
                input_reshape,
                input_dequantize,
                convolution,
                output_reshape,
                output_transpose,
            ]
        )

    graph.nodes.extend(inserted_nodes)
    graph.cleanup(remove_unused_node_outputs=True).toposort()
    output_model = gs.export_onnx(graph)
    onnx.checker.check_model(output_model)

    block2_plugins = [
        node
        for node in output_model.graph.node
        if node.domain == PLUGIN_DOMAIN
        and node.op_type == PLUGIN_OP
        and any(
            attribute.name == "stage"
            and onnx.helper.get_attribute_value(attribute) == 2
            for attribute in node.attribute
        )
    ]
    native_convs = [
        node
        for node in output_model.graph.node
        if node.op_type == "Conv" and node.name.endswith("/sr/V28_Conv")
    ]
    if block2_plugins or len(native_convs) != BLOCK_COUNT:
        raise RuntimeError(
            f"expected 0 Block2 plugins and {BLOCK_COUNT} native Conv nodes, "
            f"got {len(block2_plugins)} and {len(native_convs)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output_model, args.output)
    print(f"[PASS] native Block2 SR convolutions: {len(native_convs)}")
    print(
        "[INFO] maximum FP16 weight-scale rounding error: "
        f"{maximum_scale_rounding_error:.10f}"
    )
    print(f"[PASS] output: {args.output}")


if __name__ == "__main__":
    main()
