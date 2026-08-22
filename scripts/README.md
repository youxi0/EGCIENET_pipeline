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
bash scripts/engine/build_v15_block2_packed_dwconv_gelu_engine.sh
bash scripts/profiling/benchmark_v15_block2_packed_dwconv_gelu_engine.sh
bash scripts/pipeline/run_server.sh
```
