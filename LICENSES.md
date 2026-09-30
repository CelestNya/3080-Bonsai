# 第三方组件与许可

本仓库**只包含**自己编写的脚本、文档与工具；所有第三方产物通过 `scripts/fetch_runtime.ps1` 从官方发布页获取，许可随原包分发。

| 组件 | 来源 | 许可 | 获取方式 |
|---|---|---|---|
| llama.cpp（上游） | ggml-org/llama.cpp | MIT | — |
| KVMem fork（`llama-kvmem-server.exe`） | [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) `v0.16.0-rc3-prism.3` | 包内 `licenses/llama.cpp-MIT.txt`；KVMem 增补部分见包内 `licenses/KVMem-README.md` | fetch_runtime.ps1（含 SHA256 校验） |
| NVIDIA CUDA 运行时（cublas/cudart DLL） | 随官方包分发 | 包内 `licenses/NVIDIA-CUDA.txt`（NVIDIA CUDA EULA，允许再分发） | 同上 |
| KVMem webui（share/kvmem/ui） | 同上 | 随包分发 | 同上 |
| Bonsai-2-27B 三元模型 | prism-ml/Ternary-Bonsai-2-27B-gguf（HF） | 见 HF 仓库页（Apache-2.0 声明，自行复核） | models/README.md |
| 社区消融/合并版模型 | 各 HF 仓库 | 以各发布页为准 | models/README.md |

模型权重一律不入库；本仓库不重分发任何第三方二进制或权重。
