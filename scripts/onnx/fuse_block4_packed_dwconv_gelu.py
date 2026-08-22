#!/usr/bin/env python3

"""Replace the three Block4 layout/DWConv/GELU chains with packed plugins."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


PLUGIN_OP = "EGCINET_Block1PackedDwconv"
PLUGIN_DOMAIN = "egcinet"
PLUGIN_VERSION = "1"
HEIGHT = 11
WIDTH = 11
CHANNELS = 2048
BLOCK_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse Block4 token-major/NCHW layout conversion, 3x3 depthwise "
            "convolution and GELU into the packed DWConv TensorRT plugin."
        )
    )
    parser.add_argument("input", type=Path, help="source V15 ONNX model")
    parser.add_argument("output", type=Path, help="destination V16 ONNX model")
    return parser.parse_args()


def pack_half2_weights(weight: np.ndarray) -> list[int]:
    if weight.shape != (CHANNELS, 1, 3, 3):
        raise ValueError(f"unexpected Block4 DWConv weight shape: {weight.shape}")
    channel_kernel = np.asarray(weight, dtype=np.float16).reshape(CHANNELS, 9)
    channel_pairs = np.stack(
        (channel_kernel[0::2], channel_kernel[1::2]), axis=-1
    )
    kernel_major = np.ascontiguousarray(channel_pairs.transpose(1, 0, 2))
    return kernel_major.view(np.int32).reshape(-1).tolist()


def pack_half2_bias(bias: np.ndarray) -> list[int]:
    if bias.shape != (CHANNELS,):
        raise ValueError(f"unexpected Block4 DWConv bias shape: {bias.shape}")
    half_bias = np.asarray(bias, dtype=np.float16)
    channel_pairs = np.ascontiguousarray(
        np.stack((half_bias[0::2], half_bias[1::2]), axis=-1)
    )
    return channel_pairs.view(np.int32).reshape(-1).tolist()


def require_node(nodes: dict[str, gs.Node], name: str) -> gs.Node:
    try:
        return nodes[name]
    except KeyError as error:
        raise ValueError(f"required ONNX node not found: {name}") from error


def require_constant(tensor: gs.Tensor, label: str) -> np.ndarray:
    if not isinstance(tensor, gs.Constant):
        raise ValueError(f"{label} is not a constant initializer: {tensor.name}")
    return tensor.values


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output ONNX paths must differ")

    model = onnx.load(args.input, load_external_data=True)
    graph = gs.import_onnx(model)
    nodes = {node.name: node for node in graph.nodes}

    for block_index in range(BLOCK_COUNT):
        prefix = f"/model/block4.{block_index}/mlp"
        fc1 = require_node(nodes, f"{prefix}/fc1/Add")
        conv = require_node(nodes, f"{prefix}/dwconv/dwconv/Conv")
        gelu_output = require_node(nodes, f"{prefix}/act/Mul_1").outputs[0]

        if len(fc1.outputs) != 1 or len(conv.inputs) != 3:
            raise ValueError(f"unexpected Block4.{block_index} graph structure")
        if gelu_output.dtype != np.float16:
            raise ValueError(
                f"Block4.{block_index} plugin output must be FP16, got "
                f"{gelu_output.dtype}"
            )

        packed_weights = pack_half2_weights(
            require_constant(conv.inputs[1], "DWConv weight")
        )
        packed_bias = pack_half2_bias(
            require_constant(conv.inputs[2], "DWConv bias")
        )

        # 复用原 GELU 输出张量名，让 fc2 与其他下游节点无需改动。
        # cleanup() 会删除已断开的前后布局转换、DWConv 和 exact-ERF GELU。
        gelu_output.inputs.clear()
        graph.nodes.append(
            gs.Node(
                op=PLUGIN_OP,
                domain=PLUGIN_DOMAIN,
                name=f"{prefix}/Block4PackedDwconvGelu",
                inputs=[fc1.outputs[0]],
                outputs=[gelu_output],
                attrs={
                    "height": HEIGHT,
                    "width": WIDTH,
                    "channels": CHANNELS,
                    "fuse_gelu": 1,
                    "packed_weights": packed_weights,
                    "packed_bias": packed_bias,
                    "plugin_version": PLUGIN_VERSION,
                    "plugin_namespace": "",
                },
            )
        )

    graph.cleanup(remove_unused_node_outputs=True).toposort()
    output_model = gs.export_onnx(graph)
    onnx.checker.check_model(output_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(output_model, args.output)

    custom_nodes = [
        node for node in output_model.graph.node
        if node.domain == PLUGIN_DOMAIN and node.op_type == PLUGIN_OP
    ]
    block4_nodes = [
        node for node in custom_nodes if "Block4PackedDwconvGelu" in node.name
    ]
    if len(block4_nodes) != BLOCK_COUNT:
        raise RuntimeError(
            f"expected {BLOCK_COUNT} Block4 plugin nodes, got {len(block4_nodes)}"
        )

    print(f"[PASS] fused Block4 nodes: {len(block4_nodes)}")
    print(f"[INFO] total packed plugin nodes: {len(custom_nodes)}")
    print(f"[PASS] output: {args.output}")


if __name__ == "__main__":
    main()
