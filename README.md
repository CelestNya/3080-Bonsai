# 3080-Bonsai

在 RTX 3080（10GB 显存级）上长期运行 Bonsai-2-27B 三元量化模型的全部工程产物：可复现的二进制、全部配置脚本、排障工具和调优实录。

## 硬件基线

| 项目 | 验证环境 | 说明 |
|---|---|---|
| GPU | RTX 3080 Laptop 10GB (sm_86) | 桌面 3080 10/12GB 同样适用，budget 按 README 表调 |
| 内存 | 16GB | KVMem 溢出池要求 ≥16GB；8GB 机器把 `--kvmem-cpu-gb` 降到 0-1 |
| 系统 | Windows 11 + 计划任务 | 脚本均为单发模式（无看门狗，用户偏好） |

## 快速开始

1. `powershell -ExecutionPolicy Bypass -File scriptsetch_runtime.ps1` —— 从官方 release 下载锁定版本的运行时（带 SHA256 校验），解包出 `bin/` 与 `share/`
2. 按 `models/README.md` 下载模型到 `models/`
3. 双击 `scripts/start_bonsai_128k.bat`（日常推荐），浏览器开 `http://127.0.0.1:29187`
4. 停止：运行 `scripts/stop_all.ps1`，或直接结束 llama-kvmem-server 进程

> 本仓库不分发任何第三方二进制或模型权重，全部按许可从官方来源获取，见 LICENSES.md。

## 三种上下文模式（bonsai-abliterated-mtp，6GB）

| 脚本 | 上下文 | 适用 |
|---|---|---|
| `start_bonsai_64k.bat` | 64K，KV 全 GPU，无 KVMem | 最快，日常对话 |
| `start_bonsai_128k.bat` | 128K，KVMem 40K 驻留 | 中长文档 |
| `start_bonsai_256k.bat` | 256K 寻址，~6 万驻留 | 超长文档（换页代价见 EXPERIMENTS 曲线） |

## 关键参数说明（为什么是这些值）

- `nvidia-smi -pl 320`：**功耗墙是第一坑**。出厂锁 100W 时 decode 只有 7.4 t/s，解锁后 110+。重启失效，必须在启动脚本里
- `GGML_CUDA_BATCH_INVARIANT=1`：投机解码正确性的前提（草稿/验证 logits 逐位一致），只覆盖 1-4 列 → `--spec-draft-n-max` 最大 3
- `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0`：强制 prefill 走 cuBLAS，+51%
- KV 用 q8_0 K + q4_0 V：混合 KV 需要 **FA_ALL_QUANTS=ON 编译**的运行时（bin/ 里的已带），否则 FA 静默回退 CPU（21.9 倍减速）
- `--kvmem-budget`：GPU 驻留 token 数。**KVMem 池是启动即整块提交的**，不是按需分配——池大小 = 实打实的内存开销，按需选不要贪大
- 静态编译（`BUILD_SHARED_LIBS=OFF`）优于 DLL 拆分

## 显存档位适配

| 卡 | budget 建议 |
|---|---|
| 8GB（3080 Laptop 低配） | 24576，cpu-gb 2 |
| 10GB（本仓库验证环境） | 32768-40960 |
| 12GB（3080 12G） | 49152-61440 |

## 服务行为约定

- 单发模式：脚本占端口即拒绝启动；服务退出记录退出码到 `service.log`，不自动重启
- 停止 = 结束进程（`scripts/stop_all.ps1`）
- 计划任务启动会显示控制台窗口（用户偏好可见窗口）；注意笔记本「电池模式暂停任务」

## tools/

| 工具 | 用途 |
|---|---|
| `mtp_migrate.py` | MTP 头二进制移植手术（合成 bonsai-abliterated-mtp.gguf） |
| `repair5.py` | 修复 aifasthub 下载文件被塞 16 字节垃圾头的 GGUF |
| `hexhead.py` | 查看文件头 |
| `find_all_eog.py` | 扫描模型全部终止符 token |
| `bias_ab.py` | logit_bias A/B 测试 |
| `cntest.py` / `pfcheck.py` | 中文写作质量 / prefill 速度基准 |

## 文档

- `EXPERIMENTS.md`：16+ 轮调优全记录（含失败方案与测量方法学）
- `blog/`：实战博客（可对外发布版）
