# 3080-Bonsai

在 RTX 3080 Laptop（10GB 显存）上长期运行 Bonsai-2-27B 三元量化模型的脚本、工具与调优记录。所有内容均在下述环境中实测验证。

## 验证环境

| 项目 | 配置 |
|---|---|
| GPU | RTX 3080 Laptop 10GB（sm_86，951GB/s） |
| 内存 | 16GB |
| 系统 | Windows 11 |
| 运行时 | llama-kvmem-server v0.16.0-rc3-prism.3（Prism 三元内核 + KV 分层换页） |

## 快速开始

1. `powershell -ExecutionPolicy Bypass -File scripts\fetch_runtime.ps1` —— 从官方 release 下载锁定版本的运行时（带 SHA256 校验），解包出 `bin/` 与 `share/`
2. 按 `models/README.md` 获取模型文件到 `models/`
3. 运行 `scripts\start_bonsai_128k.bat`，浏览器开 `http://127.0.0.1:29187`
4. 停止：运行 `scripts\stop_all.ps1`，或直接结束 llama-kvmem-server 进程

> 本仓库不分发任何第三方二进制或模型权重，全部按许可从官方来源获取，见 LICENSES.md。

## 脚本与模式

| 脚本 | 上下文 | 说明 |
|---|---|---|
| `start_bonsai_64k.bat` | 64K | KV 全 GPU，无 KVMem，decode 最快 |
| `start_bonsai_128k.bat` | 128K | KVMem 40960 token GPU 驻留 |
| `start_bonsai_256k.bat` | 256K | 同上，寻址扩展到 256K（超驻留部分换页） |

服务为单发模式：启动前检查端口占用（被占则拒绝启动）；进程退出后记录退出码到 `service.log`，不自动重启。计划任务启动会显示控制台窗口；笔记本注意「电池模式暂停任务」设置。

## 实测性能（本环境）

| 指标 | 数值 |
|---|---|
| decode（64K 模式，短上下文） | 93-98 tok/s |
| decode（128K/256K 模式，40K 驻留内） | 39-73 tok/s |
| prefill（cuBLAS 路径） | 750-1050 tok/s |
| 256K 换页区召回 | 250K token 测试点全对 |

## 关键参数依据

- `nvidia-smi -pl 320`：解除出厂 100W 功耗墙（实测 100W 时 decode 7.4 tok/s，320W 时 110+ tok/s）。该设置重启失效，必须在启动时执行
- `GGML_CUDA_BATCH_INVARIANT=1`：保证投机解码草稿/验证 logits 逐位一致；该开关只覆盖 1-4 列输出，因此 `--spec-draft-n-max` 上限为 3
- `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0`：prefill 强制走 cuBLAS（实测 +51%）
- KV 采用 q8_0 K + q4_0 V 混合：需要 `GGML_CUDA_FA_ALL_QUANTS=ON` 编译的运行时，否则 flash attention 静默回退 CPU（实测 prefill 1082 → 49.6 tok/s）
- `--kvmem-cpu-gb`：CPU 溢出池在启动时整块提交，值越大常驻内存越大
- 运行时需静态编译（`BUILD_SHARED_LIBS=OFF`），DLL 拆分实测 decode 67 → 36 tok/s

参数选择与测量过程的完整记录见 `EXPERIMENTS.md`。

## 许可

本仓库代码与文档以 [MIT](LICENSE) 发布。运行时与模型权重为第三方产物，不在本仓库分发，获取方式与各自许可见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。
