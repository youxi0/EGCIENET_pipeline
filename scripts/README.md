# Scripts

脚本按用途分类，所有命令均从项目根目录执行。

- `onnx/`：生成或改写 ONNX 模型。
- `engine/`：校准并构建 TensorRT engine。
- `pipeline/`：构建、清理和启动完整 pipeline。
- `plugins/`：单独构建 TensorRT plugin。
- `profiling/`：性能测试与 profile 分析。
- `validation/`：单图推理与结果验证。
- `common/`：其他脚本共享的环境和函数；不直接执行。

常用命令：

```bash
bash scripts/pipeline/build.sh
bash scripts/plugins/build_packed_dwconv_plugin.sh
bash scripts/plugins/build_fused_sr_plugin.sh
python3 scripts/onnx/quantize_block3_fused_sr.py INPUT_V24.onnx OUTPUT_V25.onnx
bash scripts/engine/build_v25_block3_int8_sr_engine.sh
bash scripts/profiling/benchmark_v25_block3_int8_sr_engine.sh
python3 scripts/onnx/quantize_block12_fused_sr.py INPUT_V25.onnx OUTPUT_V26.onnx models/egcienet_352_multiclass_int8.cache
bash scripts/engine/build_v26_all_int8_sr_engine.sh
bash scripts/profiling/benchmark_v26_all_int8_sr_engine.sh
python3 scripts/onnx/share_block2_q_with_fused_sr.py INPUT_V26.onnx OUTPUT_V27.onnx
bash scripts/engine/build_v27_block2_shared_q_sr_engine.sh
bash scripts/profiling/benchmark_v27_block2_shared_q_sr_engine.sh
python3 scripts/onnx/replace_block2_fused_sr_with_native_conv.py INPUT_V27.onnx OUTPUT_V28.onnx
bash scripts/engine/build_v28_block2_native_conv_engine.sh
bash scripts/profiling/benchmark_v28_block2_native_conv_engine.sh
bash scripts/engine/build_v17_block3_attn_q_matmul_qdq_engine.sh
bash scripts/profiling/benchmark_v17_block3_attn_q_matmul_qdq_engine.sh
bash scripts/pipeline/run_server.sh
```
