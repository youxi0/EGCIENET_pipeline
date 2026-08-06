# EGCIENet ONNX ↔ TensorRT 关键层映射（暂不含边缘分支）

## 结论

当前 ONNX 主干/decoder 与 TensorRT 报告可以继续用于精确或融合映射；边缘分支已按要求完全排除。需要注意：当前 ONNX 新增了边缘 token 的顶层 `Reshape/Transpose`，因此 encoder 各 stage 出口在当前 ONNX 中是 `Transpose_1..4`，在报告对应的旧 engine 中语义等价节点是 `Transpose..3`。这些出口使用张量尺寸和上下游关系做“语义匹配”，不是按同名硬匹配。

机器可筛选的主表见 `egcienet_onnx_trt_key_mapping.csv`。它只包含模块边界、热点层、CopyNode 模式、精度边界和拟修改层，没有展开全部 722 个 TensorRT 层。

数据来源：

- 当前 ONNX：`models/egcienet_352.onnx`，SHA-256 `B01BDB6BF431CD36851DC2EDCFDE4BCD40C1A61A11AA85B24880859969DB89B3`，2825 个节点。
- FP16 层结构：`fp16_layers.json`，722 个 TensorRT 层。
- FP16 逐层耗时：`fp16_profile.json`。
- FP16/INT8 CopyNode：`fp16_copynode_nodes.csv`、`int8_copynode_nodes.csv`。
- INT8 精度边界：`int8_copynode_quantization_boundaries.csv`。
- 现有资料没有 INT8 逐层 profile；因此本表中的“热点耗时”均为 FP16 基线，INT8 只用于判断精度边界。

## 1. 模块入口和出口

| 模块 | 当前 ONNX 入口 → 出口 | TensorRT 入口 → 出口 | 出口张量 | 匹配方式 |
|---|---|---|---:|---|
| Encoder stage 1 | `patch_embed1/proj/Conv` → `Transpose_1` | `/model/patch_embed1/proj/Conv` → `__myl_...myl48_3` | `1×64×88×88` | 出口语义匹配 |
| Encoder stage 2 | `patch_embed2/proj/Conv` → `Transpose_2` | `/model/patch_embed2/proj/Conv` → `__myl_...myl88_3` | `1×128×44×44` | 出口语义匹配 |
| Encoder stage 3 | `patch_embed3/proj/Conv` → `Transpose_3` | `/model/patch_embed3/proj/Conv` → `__myl_...myl252_3` | `1×320×22×22` | 出口语义匹配 |
| Encoder stage 4 | `patch_embed4/proj/Conv` → `Transpose_4` | `/model/patch_embed4/proj/Conv` → `__myl_...myl262_3` | `1×512×11×11` | 出口语义匹配 |
| FEM1 | `FEM1/conv1/Conv` → `FEM1/prelu_2/PRelu` | conv1+PReLU 融合层 → 与 `d31/d42_31 Add` 融合的 PWN | `1×64×88×88` | 精确/融合 |
| FEM2 | `FEM2/conv1/Conv` → `FEM2/prelu_2/PRelu` | conv1+PReLU 融合层 → 与 `d42/Add` 融合的 PWN | `1×128×44×44` | 精确/融合 |
| FEM3 | `FEM3/conv1/Conv` → `FEM3/prelu_2/PRelu` | conv1+PReLU 融合层 → `PWN(FEM3/prelu_2)` | `1×320×22×22` | 精确/融合 |
| FEM4 | `FEM4/conv1/Conv` → `FEM4/prelu_2/PRelu` | conv1+PReLU 融合层 → `PWN(FEM4/prelu_2)` | `1×512×11×11` | 精确/融合 |
| d31 | `d31/Resize` → `d31/Add` | Resize → conv_pre+ReLU → 下游融合 PWN | `320@22² → 64@88²` | 精确/融合 |
| d42 | `d42/Resize` → `d42/Add` | Resize → conv_pre+ReLU → `PWN(d42/Add)` | `512@11² → 128@44²` | 精确/融合 |
| d42_31 | `d42_31/Resize` → `d42_31/Add` | Resize → conv_pre+ReLU → 下游融合 PWN | `128@44² → 64@88²` | 精确/融合 |
| 输出头 | `score_1/Conv` → `/Resize` → `/Sigmoid` | 同名 Conv → Resize → `PWN(/Sigmoid)` | `1×1×88×88 → 1×1×352×352` | 精确/融合 |

## 2. 高耗时层（FP16 基线，已聚合重复层）

| 优先级 | ONNX 范围 | TensorRT 层/模式 | 数量 | FP16 合计 |
|---:|---|---|---:|---:|
| 1 | `block3.{0..17}/mlp/fc1/MatMul` | `/model/block3.*/mlp/fc1/MatMul_myl*_7` | 18 | `5.193 ms` |
| 2 | `decoder/d31/conv_pre/.../Conv` | conv_pre Conv+ReLU | 1 | `0.674 ms` |
| 3 | `decoder/d42/conv_pre/.../Conv` | conv_pre Conv+ReLU | 1 | `0.511 ms` |
| 4 | `block1.{0..2}` 的 DWConv 后 GELU 链 | `__myl_ReshTranDivCastErfCastAddMulMul_*` | 3 | `1.012 ms` |
| 5 | `block1.{0..2}/mlp/dwconv/dwconv/Conv` | 同名 CaskConvolution | 3 | `0.858 ms` |
| 6 | `block1.{0..2}/mlp/fc1/Add → Reshape` | `__myl_Add_myl{26,35,44}_*` | 3 | `0.798 ms` |
| 7 | `block4.{0..2}/mlp/fc1/MatMul` | `/model/block4.*/mlp/fc1/MatMul_myl*_7` | 3 | `0.777 ms` |
| 8 | `decoder/d42_31/conv_pre/.../Conv` | conv_pre Conv+ReLU | 1 | `0.301 ms` |
| 9 | `decoder/FEM3/conv1/Conv` | Conv+PReLU 融合 | 1 | `0.280 ms` |
| 10 | `decoder/FEM1/Reshape + Transpose` | Shuffle | 1 | `0.275 ms` |

这里的第一名是 18 个重复的 stage 3 MLP `fc1`，不是单个层异常。第一轮混合精度实验仍建议先改 decoder，因为已有 INT8 边界证据直接指向 decoder 内部的精度来回切换；encoder stage 3 留作第二轮独立实验。

## 3. CopyNode 映射

| CopyNode 模式 | ONNX 前后链路 | 数量 | FP16 合计 | 处理建议 |
|---|---|---:|---:|---|
| MLP → depthwise-conv 布局转换 | `mlp/fc1/Add → dwconv/Reshape → dwconv/dwconv/Conv` | 25 | `3.655 ms` | 调整导出布局或融合；这是最明确的 CopyNode 优化点 |
| Decoder FEM 内部转换 | `decoder/FEM{1..4}/**` | 33 | `0.648 ms` | 用连续的 INT8 Conv 区和完整 FP16 CBAM 区减少来回转换 |
| 其中：名称直接落在 CBAM 内的转换 | `decoder/FEM{1..4}/cbam/**` | 8 | `0.378 ms` | 整块 CBAM 固定 FP16 |
| 全部非边缘分支显式 CopyNode | encoder + decoder | 168 | `8.440 ms` | 个别行查原始 CSV，不在主表逐条展开 |

25 个 MLP→DWConv CopyNode 的 stage 分布：stage 1 为 3 个、`0.827 ms`；stage 2 为 4 个、`0.607 ms`；stage 3 为 18 个、`2.221 ms`。对应 TensorRT 层名规则为：

```text
Reformatting CopyNode for Input Tensor 0 to
/model/block{1..3}.*/mlp/dwconv/dwconv/Conv
```

## 4. FP16/INT8 边界（排除边缘分支后共 26 个）

| 区域 | INT8 CopyNode index | 主要方向 | 张量/说明 | 建议 |
|---|---|---|---|---|
| stage 1 fan-out | `20–21` | FP16→INT8→FP16 | `1×64×88×88` 同时流向 patch_embed2、FEM1 | encoder 路径保持 FP16，FEM1 Conv 分支进入 INT8 |
| stage 3 fan-out | `132–133` | FP16→INT8→FP16 | `1×320×22×22`，随后 patch_embed4 输出 `512×11×11` | encoder 路径保持 FP16，FEM3 Conv 分支进入 INT8 |
| FEM1/CBAM | `134–142` | INT8↔FP16，另有 FP32 scalar→INT8 | 9 个边界，最密集 | CBAM 整体 FP16，避免 CA/SA 内部反复切换 |
| FEM2/CBAM | `147–149` | FP16↔INT8 | 3 个边界 | 同上 |
| FEM3/CBAM | `150,152,154–156` | INT8→FP16、FP16/FP32→INT8 | 5 个边界 | 同上 |
| FEM4/CBAM→d42 | `160–161,165` | FP16↔INT8 | 3 个边界 | CBAM FP16，d42 大卷积路径 INT8 |
| decoder 标量广播 | `166–167` | FP32 scalar→INT8 | PReLU/Add 标量 | 若仍显著，令相邻 pointwise/residual island 保持 FP16 |

方向统计：FP16→INT8 10 个，INT8→FP16 9 个，FP32 scalar→INT8 7 个。原报告 index `1–3` 属于已排除的旧边缘分支/旧 `proj_1` 路径，未计入。

## 5. 准备修改的层

| 动作 | ONNX 节点范围 | TensorRT 表现 | FP16 基线 |
|---|---|---|---:|
| 强制 INT8 | `decoder/FEM{1..4}/{conv1,conv2,conv3}/Conv` | conv1 与 PReLU 融合，conv3 与残差 Add 融合 | 12 层合计 `1.923 ms` |
| 强制 INT8 | `decoder/{d31,d42,d42_31}/conv_pre/conv_pre.0/Conv` | 3 个 Conv+ReLU 融合层 | 合计 `1.486 ms` |
| 整体强制 FP16 | `decoder/FEM{1..4}/cbam/**` | 60 个 compute/reformat profile 项 | 合计 `1.599 ms` |
| 布局优化 | `block{1..3}.*/mlp/{fc1,dwconv}/**` | 25 个 MLP→DWConv CopyNode | CopyNode 合计 `3.655 ms` |
| 暂不修改 | `score_1/Conv → Resize → Sigmoid` | 输出头链路 | 合计 `0.128 ms` |

第一轮 engine 策略应是：decoder 的大卷积形成连续 INT8 区，四个 CBAM 各自形成完整 FP16 岛；不要逐个 CBAM 小算子单独设精度。完成后必须重新导出 INT8 `layerInfo` 和逐层 profile，才能把本表中缺失的 INT8 层耗时补齐并验证 CopyNode 是否真正减少。
