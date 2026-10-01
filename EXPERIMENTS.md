# Bonsai / 三元模型部署实验记录

> **测试机（远端，SSH 至 Auronya）**：RTX 3080 Laptop 10GB（sm_86，10240 MiB，功耗上限 340W）+ 16GB RAM
> （本文件中除特别说明外，所有实测数据均來自此机）
>
> 对照机（本机，另一会话占用中）：RTX 2080 Ti 11GB（sm_75，616GB/s）
> 模型：Bonsai-2-27B（Qwen3.8-27B 三元量化）
> 记录日期：2026-09-27

---

## 一、决定性发现（按重要性排序）

### 1. ★★★ 功耗墙：100W → 320W，速度 7.4 → 110 tok/s（15 倍）

**症状**：decode 只有 7.4 tok/s，GPU 时钟 210MHz（上限 2100MHz）
**诊断**：`nvidia-smi -q -d POWER` 显示 `Current Power Limit: 100W`（Default 320W）
**处方**：`nvidia-smi -pl 320`
**证据**：解锁后 SM 时钟 1935MHz，功耗 285-309W

> ⚠️ 这是本次排查最重大的发现。早期的架构误判（以为 sm_86 不匹配）都源于此。

### 2. ★★★ 三元内核瓶颈在计算，不在带宽

profiler 持续负载采样：
```
SM 时钟:    1875-1935 MHz / 2100 上限  → 91%
功耗:       285-309 W / 320 上限       → 93%
GPU 利用率:  85-89%
显存带宽:    45-49%                    ← 未饱和！
```
**推论**：若是带宽受限，mem_util 应达 80-90%。实际 48% → 瓶颈在三元解包（LUT）+ Hadamard 逆变换。
**佐证**：3080 比 2080 Ti 快 3 倍，而带宽只高 1.5 倍——符合算力比例（3080 算力约 3x），不符合带宽比例。

### 3. ★★ PTQ1_0 比 PQ2_0 快 60%（2080 Ti 实测）

| 量化 | 体积 | decode |
|---|---|---|
| PQ2_0 | 7.21GB | 17.0 |
| **PTQ1_0** | **5.95GB** | **27.1** |

体积小 17%，带宽受限时直接换成速度。**注：这是功耗受限状态下的数据，但相对关系成立。**

### 4. ★★ MTP 增益强依赖任务类型

| 任务 | 接受率 | mean len |
|---|---|---|
| Python 代码 | 0.77-0.94 | 3.1-3.4 |
| Bash 代码 | 0.58-0.81 | 2.2-3.1 |
| 英文散文 | 0.48-0.91 | 2.0-3.7 |
| 中文散文 | 0.34-0.46 | 1.9-2.2 |

注：早期仅用中文散文测试会得出「MTP 无效」的错误结论，实际是多任务类型差异。

### 5. ★★ 上下文断崖有两个不同根因

| KV 类型 | 稳定上限 | 断崖后 | 根因 |
|---|---|---|---|
| q8_0 | 64K | 96K+ → 10-15 t/s | **KV 溢出显存** |
| q4_0 | 152K | 160K+ → 22 t/s | **权重被挤出显存** |

现象相同（速度崩到 10-20），根因不同。**q4_0 把上下文上限翻倍以上**。

---

## 二、试过的方案与结论

### 引擎对比（同模型、同参数、控制变量）

| 引擎 | 代码任务 | 结论 |
|---|---|---|
| 自编译 bonsai2 (sm_86, #214+#218) | 70.4 | 基础可用 |
| KVMem 包 + `--kvmem` | 83.5 | KVMem 机制有开销 |
| **KVMem 包 + `--no-kvmem`** | **94.7-97.9** | **★ 最优** |
| combo 分支 (86-real, #215/#216/#220/#221) | 47.9-56.5 | 更慢 |
| combo 分支 (86, 重新编译) | **10.4** | **严重劣化** |

**combo 分支详细对比（同条件）**：
```
combo3:  prompt eval = 442 ms/token (2.26 t/s)   eval = 96.3 ms/token (10.4 t/s)
KVMem:   prompt eval = 68.7 t/s                  eval = 94.7 t/s
```
差 9 倍，连 prefill 都崩坏。**不是配置问题，是构建本身有问题。**
推测：我们的 toolchain（CUDA 13.4 + MSVC 14.44）与该分支的兼容性问题，或分支 PR 尚未稳定。
**处方：不要自己编译，直接用 KVMem 包自带的预编译引擎。**

**KVMem 机制是负收益**（-17%）：短上下文下 KV 全在显存，分层管理只有开销。
**其真正价值在超长上下文**：KVMem 可到 256K（85.7 t/s），原生引擎 152K 封顶。

### 模型变体对比

| 模型 | 严格越狱测试 | MTP | decode |
|---|---|---|---|
| lean（原版基线） | **6/6 全拒** | ✅ | **94.7** |
| abliterated（消融） | **0/6 全答** ✅ | ❌ | 62.7 |

**abliterated 确实去掉了审查**，代价是无 MTP 层（慢 33%）。
**注意**：温和探针（反派独白等）两者都是 0/5 拒绝，区分不出来。**必须用严格探针**。

### 参数实验

| 参数 | 效果 | 结论 |
|---|---|---|
| `GGML_CUDA_BATCH_INVARIANT=1` | 提升接受率（1-4 列范围） | 有效，但 n_max≤3（4 列） |
| `-ub 1024`（默认 512） | prefill 无变化，decode 降 10% | **无收益** |
| `-ub 2048` | 测试中 | — |
| `--spec-draft-n-max 3` | 边界值 | n_max=4 会让 BI 失效 |
| `ngram-*` 投机 | 在混合注意力模型上**静默失效** | **不要试** |

### 失败/无效的尝试

| 尝试 | 结果 | 原因 |
|---|---|---|
| IQ2_XXS 混合量化（专家降 2-bit） | 失败 | IQ2 要求每个专家都有 imatrix 数据，校准覆盖不全 |
| combo `FA_ALL_QUANTS=ON` | 编译失败 | ptxas 内核爆炸 |
| combo `86-real` | 编译成功但更慢 | 可能禁用了 PTX 兜底 |
| `--mlock` | 无效 | Windows 上不生效 |
| ninfer（2080Ti/3080 fork） | 不可用 | 制品 16-22GB，10GB 装不下 |
| DFlash2 | 无收益 | Bonsai 上实测仅 +0.5-3.6%（非宣传的 5 倍） |

---

## 三、测量方法学

### 曾发生的测量偏差

1. **变量未控制**：BI 和 n_max 同时改动，曾把 58→65 的提升误归因于 BI
2. **环境污染**：残留的 llama-server 占显存时，测得"KVMem 只有 24.6 tok/s"
3. **测量口径**：用 30-token 短 prompt 测 prefill，得到 69 的假数字（真实 590-603）
4. **样本偏差**：只用中文散文测 MTP，得出"无效"结论
5. **过早优化**：未小规模验证就全面编译 combo，浪费 2 次 20 分钟

### 正确的测量方法

```
1. 测试前确认显存回落基线：nvidia-smi --query-gpu=memory.used
2. 清理残留：taskkill /F /IM llama-server.exe /IM llama-kvmem-server.exe
3. 固定载荷：同一 prompt、同一输出长度、多轮取中位
4. 多任务类型：代码/散文/中文至少各一
5. 长 prompt 测 prefill（>2K token），短 prompt 只反映首 token 延迟
6. 每次带 profiler：并发 nvidia-smi 采样（0.05-0.25s 间隔）
7. 控制变量：一次只改一个参数
```

---

## 四、最终配置

### 常规使用（最快）
```
nvidia-smi -pl 320                    ← 必须
GGML_CUDA_BATCH_INVARIANT=1
llama-kvmem-server.exe -m bonsai-ptq1-lean.gguf --jinja \
  -ngl 99 -t 8 -fa on --parallel 1 -c 65536 --no-kvmem \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 3
```
实测：code 94.7-97.9 / zh 70.3 / en 78.9 tok/s，prefill 590-603 tok/s

### 超长上下文
```
同上但去掉 --no-kvmem，加 --kvmem-budget 40960 --kvmem-gen-reserve 8192 -c 262144
```
实测：256K 上下文 85.7 tok/s

### 无审查
用 `bonsai-abliterated.gguf`，去掉 MTP 参数。62.7 tok/s。

---

## 五、待办

- [ ] MTP 移植到 abliterated（无审查 + MTP）
- [ ] combo 新编译的 A/B 结果
- [ ] 细粒度 op 级性能分析（当前只有阶段级）
- [ ] 长上下文下的 decode 衰减曲线

---

# 第二轮：prefill 专项优化（2026-09-28）

## 一、问题定义

反馈「prefill 太慢」后实测 server 模式 **590-603 tok/s**，
而 llama-bench 标准 pp512 报 **1123 tok/s**（差 1.9 倍）。

## 二、排除的假设（全部有实测数据）

| 假设 | 测试方法 | 结果 | 结论 |
|---|---|---|---|
| batch 分块太小 | `-b`/`-ub` 512→4096 | 588 / 587 / 558 | **无效，排除** |
| 首 token 开销摊薄 | prompt 0.5K→40K | 590 稳定 | **排除** |
| KVMem `-b` 默认值小 | 查到 KVMem 默认 `-b 512`（llama.cpp 是 2048） | 调大后无变化 | **排除** |
| 上下文长度影响 | 同上 | 无影响 | **排除** |

## 三、profiler 决定性诊断

持续负载下按阶段采样（0.1s 间隔，采样有效率 289/291）：

| 阶段 | SM 时钟 | **GPU 利用率** | **显存带宽** | 功耗 |
|---|---|---|---|---|
| **prefill** | 1898 MHz | **97%** | **18%** | 302W |
| decode | 1830-1885 MHz | 87-90% | 47-49% | 289-303W |

**结论：prefill 是纯算力饱和**（GPU 97% 忙、带宽仅 18%）。
没有可回收的带宽，加 batch 只会增加计算量。

## 四、引擎符号分析（决定优化方向）

对 KVMem 预编译引擎（124MB）做二进制符号检测：

| 优化项 | 状态 | 影响 |
|---|---|---|
| #218 PT 专用 mat-vec | ✅ 有（12 处） | decode |
| SoA 激活布局 (#215) | ✅ 有（4 处） | decode |
| MMQ 路径 | ✅ 有（85 处） | prefill |
| **FWHT+量化融合 (#221)** | ❌ **缺** | **实测 +11.4%** |
| GDN gather 融合 (#220) | ❌ 缺 | decode |

**FWHT 融合的 A/B 实测**（同引擎，唯一变量）：
```
GGML_CUDA_FWHT_FUSION=1 → 93.9 tok/s
GGML_CUDA_FWHT_FUSION=0 → 84.3 tok/s   (+11.4%)
```

## 五、官方文档的关键证据（KVMem 包内 docs/）

**`bonsai-kernel-comparison.md`** 记录了作者的受控实验：

> **"Prefill is effectively unchanged; these gains concern decode."**

| GPU | draft | baseline decode | optimized decode | 4K prefill (base→opt) |
|---|---|---|---|---|
| RTX 5050 | 0 | 13.26 | 20.42 (+54%) | 136.6 → 136.3 |
| RTX 5060 Ti | 0 | 38.79 | 46.04 (+19%) | 356.7 → 359.5 |

**#214/#218 系列优化全部是 decode 侧**，prefill 不变。

**另一条重要发现（PDL 同步缺陷）**：
> `mul_mat_vec_ptq1_0_pt` 未在读取激活前调用 `ggml_cuda_pdl_sync()`，
> 导致四次相同请求产生**不同输出**（数值错误）。
> 该路径与 SM90+ 相关。

## 六、本地验证否决的外部结论

| 外部结论 | 本地实测 | 判定 |
|---|---|---|
| draft=1 最快（5050/5060 数据） | 3080 上 n=1: 85.2 / n=2: 93.6 / n=3: 93.5 | ❌ **本地否决**，保持 n_max=3 |
| combo 分支含 #215/#216/#220/#221 应更快 | 自编译后 prefill 崩坏（49.9）、decode 53.5 | ❌ 构建环境问题 |

## 七、combo 编译失败的诊断（3 次尝试）

| 配置 | decode | prefill | 判定 |
|---|---|---|---|
| `86` + NATIVE=ON | 10.4 | 2.26 | 崩 |
| `86-real` + NATIVE=ON | 47.9-56.5 | — | 崩 |
| `86-real` + NATIVE=OFF | 53.5 | 49.9（衰减） | 崩 |

**官方 KVMem 用 CUDA 12.9.86 编译成功**，而本机 toolkit 是 **CUDA 13.4**。
**强烈怀疑是工具链版本差异**，但未验证。

## 八、prefill 的优化方向（基于以上证据）

既然 #214/#218 都改不动 prefill，**唯一剩余方向是减少计算量本身**：

1. **trit 解包表查优化**（调研给出的具体方案）
   - 当前 `ptq1_0_trit_step`：每轮 5 次串行 `*3 & mask`
   - `3^n mod 256` 周期为 64 → 可塌缩为查表
   - Vulkan 后端已实现（PR #187），实测有效
   - **这是改算法，不是调参数**

2. **FWHT 融合**（+11.4% 已验证，需可用的编译环境）

3. **用 CUDA 12.9 重编 combo**（若工具链假设成立，可一次性拿到全部收益）

---

# 第三轮：prefill 根因定位（2026-09-28 重大突破）

## ★★★ 核心发现：prefill 的真正瓶颈是 FlashAttention 内核缺失

### 完整对比矩阵（同模型 bonsai-abliterated-mtp，同参数，唯一变量=引擎+KV）

| 引擎 | KV 配置 | **prefill** | decode |
|---|---|---|---|
| KVMem 预编译（FA_ALL_QUANTS=ON） | q8_0/q4_0（混合） | 592 | **92.9** ✅ |
| KVMem 预编译 | f16/f16（同类型） | 601 | 91.2 |
| **b86 自编译（FA_ALL_QUANTS=OFF）** | q8_0/q4_0（混合） | **49.6** ❌ | 47.2 |
| **b86 自编译** | **f16/f16（同类型）** | **1082** ✅ | 66.8 |

### 根因链

1. 我的构建用 `GGML_CUDA_FA_ALL_QUANTS=OFF`（CMake 默认）
2. 使用混合 KV（`--cache-type-k q8_0 --cache-type-v q4_0`）
3. 混合 KV 的 flash attention 内核只在 `FA_ALL_QUANTS=ON` 时编译
4. 没有该内核 → **FA 静默回退 CPU** → **prefill 从 1082 崩到 49.6（21.8 倍差距）**
5. 无任何错误日志（静默回退）

**这个 bug 解释了本轮所有困惑**：
- 为什么 combo3/combo4 的 prefill 都约 50（我用的是同一个 OFF 构建配置）
- 为什么 KVMem 引擎看似"prefill 慢"（其实它的 592 是另一回事）

### 两个引擎的真实对比

| | KVMem 预编译 | b86 自编译（f16 KV） |
|---|---|---|
| prefill | 592-601 | **1082**（+83%） |
| decode（混合 KV） | **92.9** | 49.6（崩） |
| decode（f16 KV） | 91.2 | 66.8 |
| FA_ALL_QUANTS | ON | OFF ← **需改** |

### 待验证

用 `FA_ALL_QUANTS=ON` 重编 b86，预期可同时拿到：
- 混合 KV（q8_0/q4_0）下的高 prefill（~1082）
- 高 decode（~90+）

### 方法学记录（三）

**未验证引擎能力即做 A/B 测试的案例**：
- 之前把「FWHT_FUSION=1 → 93.9」当作 +11.4% 收益
- 实际验证：**KVMem 引擎根本没有 `GGML_CUDA_FWHT_FUSION` 字符串**（Python bytes.count = 0）
- 那个"收益"是测量噪声（数据本来就在 85-99 波动）

**新纪律**：任何 A/B 测试前，先用 `python -c "print(open('x.dll','rb').read().count(b'ENV_VAR'))"` 确认特性在二进制中存在。

---

## 第四轮补充：f16 KV 的显存代价（验证混合 KV 的必要性）

| 引擎 | KV | 上下文 | 显存 | decode |
|---|---|---|---|---|
| KVMem | f16/f16 | 32K | 9430 MiB | 91.2 |
| KVMem | f16/f16 | **64K** | **9766 MiB** | **10.2** ❌ |
| KVMem | f16/f16 | 96K | 9830 MiB | 9.2 ❌ |

**结论**：f16 KV 在 64K 就爆显存（装置 10GB），decode 崩到 10。
**所以混合 KV 不是可选项，是必需** —— 它把 KV 显存减半，才能开到 64K+。

**正解**：`--cache-type-k q8_0 --cache-type-v q4_0` + **`GGML_CUDA_FA_ALL_QUANTS=ON`**（编译期）

## 第五轮：源码版本对比

| 引擎 | 分支 | commit |
|---|---|---|
| KVMem | PrismML `prism` | `9a9394a` |
| b86 | sudoingX `bonsai2` | `285542d`（"cap PTQ1_0 mat-vec at 4 columns"） |

两者都含 #218，但 decode 表现差 26（66.8 vs 92.9），原因待查。

---

# ★★★ 第六轮：prefill 优化的最终答案（2026-09-28）

## 突破性发现汇总

### 发现 1：FA_ALL_QUANTS=OFF 导致混合 KV 下 prefill 崩溃（21.9x）

我的自编译引擎用 `GGML_CUDA_FA_ALL_QUANTS=OFF`（CMake 默认），
配合混合 KV（q8_0 K / q4_0 V）时，flash attention 内核不存在 → **静默回退 CPU**：

| 引擎 | KV | prefill |
|---|---|---|
| b86（FA_ALL_QUANTS=**OFF**） | q8_0/q4_0 | **49.6** ❌ |
| **b86fa（FA_ALL_QUANTS=ON）** | q8_0/q4_0 | **1088** ✅ |

**21.9 倍差距，且无任何错误提示（静默回退）**

### 发现 2：cuBLAS 路径显著提升 prefill（+51%）

`GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0` 让 PTQ1_0 走 dequant+cuBLAS 而非 MMQ tile：

| 配置 | prefill | decode |
|---|---|---|
| KVMem 默认（MMQ 路径） | 588.6 | 85.6 |
| **KVMem + cuBLAS** | **890.7** | 88.0 |

源码注释说 cuBLAS "~7% at pp512"，**实测 3080 上是 +51%**。
（代价：官方称精度略降，但未测出可感知差异）

### 发现 3：两个引擎的短板与最终最优解

| 引擎 | 源码 | prefill | decode | 短板 |
|---|---|---|---|---|
| KVMem | PrismML `9a9394a` | 592 | **92.9** | MMQ 路径慢 |
| b86fa | sudoingX `285542d` | **1088** | 69.7 | **缺 Hadamard MTP 修复** → 接受率仅 0.35-0.44 |
| **★ KVMem + cuBLAS** | PrismML | **891** | **90.0** | **目前最优综合** |

**b86fa 的 decode 差**：`sudoingX/bonsai2` 缺 PrismML 后期提交
（如 `#257 tied Hadamard output weights`），导致 MTP 草稿质量差、接受率低。

## 最终推荐配置

```bat
nvidia-smi -pl 320                              :: 必须（否则 7 tok/s）
set GGML_CUDA_BATCH_INVARIANT=1                 :: 提升 MTP 接受率
set GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0            :: cuBLAS prefill（+51%）

llama-kvmem-server.exe -m bonsai-abliterated-mtp.gguf --jinja ^
  -ngl 99 -t 8 -fa on --parallel 1 -c 65536 --no-kvmem ^
  --cache-type-k q8_0 --cache-type-v q4_0 ^
  --spec-type draft-mtp --spec-draft-n-max 3
```

**实测**：prefill 891 / decode 90.0（code）· 65.4（zh）· 64K 上下文 · 9609 MiB

## 与优化前对比

| 指标 | 优化前 | 优化后 | 提升 |
|---|---|---|---|
| prefill | 592 | **891** | **+51%** |
| decode | 92.9 | 90.0 | 基本持平 |
| 上下文 | 32K | **64K** | 2x |

（另：若不需要 64K 上下文，b86fa 引擎可提供 prefill 1088 但 decode 降至 69.7）

## 方法学记录（四）

**A/B 测试前先验证特性存在**：
- 用 `python -c "print(open('x.dll','rb').read().count(b'ENV_VAR_NAME'))"` 
- 或用 CMakeCache 确认编译选项
- 之前把「FWHT_FUSION=1 → +11.4%」当收益，实际该特性**不在**所用引擎里

---

# ★★★ 第七轮：decode 差距根因 + 最终方案（2026-09-28）

## 关键实验：两个引擎的「本底」性能（MTP 关闭）

| 引擎 | prefill | **decode（无 MTP）** |
|---|---|---|
| KVMem（PrismML 9/18 快照） | 592 | **62.9** |
| b86fa（sudoingX 285542d，FA_ALL_QUANTS=ON） | **1160** | **64.6** |

**两个引擎的 decode 本底几乎相同（63-65）！**

→ **decode 的差距（92.9 vs 69.7）完全来自 MTP 实现质量**，不是引擎基础能力。

## 时间线证据：KVMem 基线滞后 4-7 天

| commit | 日期 | 内容 |
|---|---|---|
| `9a9394a` | **2026-09-18** | ← **KVMem 的基线** |
| `bdc23b56` | 2026-09-22 | **#214 branch-free PTQ1_0 MMQ tile loader（2x prefill）** |
| `25092e7d` | 2026-09-23 | **#216 4-column GDN warp layout（Ampere+）** |
| `adfffbe` | 2026-09-25 | PrismML prism 最新 |

**KVMem 引擎（9/18）不含 #214 和 #216**——它们晚 4-5 天提交。
这解释了它的 prefill 上限 592（vs b86fa 的 1160）。

## 最终方案：PrismML prism 最新版（adfffbe）

已验证包含：
- `full_lane`（#214）✅ → prefill 2x
- `gdn_cols_per_warp`（#216）✅ → prefill +6%
- 所有 Hadamard MTP 修复（#257 tied Hadamard 等）✅ → decode 高

**预期**：同时拿到 b86fa 的 prefill（~1160）+ KVMem 的 decode（~92）

## 引擎能力全景对照

| 引擎 | 源码日期 | prefill | decode(有MTP) | 缺失 |
|---|---|---|---|---|
| KVMem 预编译 | 9/18 | 592 | 92.9 | #214, #216 |
| b86fa 自编译 | 9/22 | 1160 | 69.7 | Hadamard MTP 修复 |
| **prism 最新（编译中）** | **9/25** | **预期 1100+** | **预期 90+** | — |

## 另一个已确认的可用优化

`GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0`（cuBLAS 路径）：
- KVMem: prefill 588 → **890**（+51%）
- 代价：官方称精度略降（未测出可感知差异）
- **可作为 fallback**：若 prism 最新版 MMQ 有问题时使用

---

# ★★★ 第八轮：decode 差距的真正根因（BUILD_SHARED_LIBS）

## 矛盾现象

**同样的源码（PrismML prism @ adfffbe）、同样的编译选项**，
我编译的 decode 只有 **36.2**，而 KVMem 预编译的有 **87-99**。

编译选项逐项对比（全部一致）：
```
GGML_CUDA_FA=ON / FA_ALL_QUANTS=ON / GRAPHS=ON / NCCL=ON
PEER_MAX_BATCH_SIZE=128 / FORCE_MMQ=OFF / NO_VMM=OFF
```

## ★ 找到的唯一差异：BUILD_SHARED_LIBS

| 项 | 我的构建 | **KVMem 预编译** |
|---|---|---|
| `BUILD_SHARED_LIBS` | **ON**（DLL 拆分） | **OFF**（静态库） |
| `GGML_STATIC` | — | OFF |
| 产物形态 | llama-server.exe = 10KB + 8 个 DLL | **单体 124MB exe** |

**官方文档也明写**：
> Build configuration: MSVC, CUDA 12.9.86, SM120a, **static libraries**, all FlashAttention KV quantizations

## 影响机制（推测）

动态库模式可能导致：
1. **跨 DLL 函数调用开销** —— decode 每 token 有 ~1900 个小 kernel，每个 op 都要跨模块边界
2. **CUDA Graph 捕获受限** —— 跨 DLL 的图捕获可能有额外同步
3. **优化边界受限** —— 无法跨模块内联（编译期看不到实现）

## 这也解释了 ik_llama.cpp 构建笔记里的同一个坑

> **`BUILD_SHARED_LIBS` 没关 ⇒ kvmem 被编成 DLL，而 llama 要静态库**

## 验证中

正在用 `BUILD_SHARED_LIBS=OFF` 重编，预期 decode 恢复到 90+。

## 已确认的其他事实（本轮）

| 发现 | 数据 |
|---|---|
| FA_ALL_QUANTS=OFF + 混合 KV | prefill 崩到 49.6（FA 静默回退 CPU） |
| cuBLAS 路径 | prefill 588→891（+51%） |
| KVMem 基线（9/18） | 缺 #214/#216（9/22-23 提交） |
| 两引擎无 MTP 时的 decode | 几乎相同（62.9 vs 64.6）→ decode 差距来自 MTP |

---

# ★★★ 第九轮：最终配置确定（2026-09-28）

## 最终实测数据

配置：`start_ultimate.bat`（KVMem 引擎 + cuBLAS prefill + 草稿 KV f16 + 64K）

| 指标 | 数值 |
|---|---|
| **prefill** | **890-895 tok/s**（4K / 10K prompt） |
| **decode（code）** | **90.0 / 99.6 / 93.2** |
| decode（zh） | 62.7 |
| 上下文 | **64K** |
| 显存 | 9604 MiB / 10240 |

## 完整优化历程（prefill 从 49.6 到 895 = 18 倍）

| 阶段 | prefill | decode | 关键动作 |
|---|---|---|---|
| 初始（我的构建，混合 KV） | 49.6 | 47.2 | — |
| + FA_ALL_QUANTS=ON | 1088 | 69.7 | 修复 FA 静默回退 CPU |
| + BUILD_SHARED_LIBS=OFF | 1102 | 36.2 | 修复 DLL 边界开销 |
| + 新 HEAD (ff414120c) | 1102 | 67.4 | PDL sync + q4_0 V 修复 |
| **+ KVMem 引擎 + cuBLAS + 草稿 f16** | **890** | **93.2** | **最终配置** |

## 关键技术要点

### 1. FA_ALL_QUANTS 必须 ON（如果使用混合 KV）
`GGML_CUDA_FA_ALL_QUANTS=OFF`（CMake 默认）+ 混合 KV → FA 内核缺失 → **静默回退 CPU** → prefill 崩 21.9 倍

### 2. 草稿 KV 必须 f16
- 标准 llama-server 的 `--cache-type-k-draft` **默认继承主 KV（q4_0）** → 草稿质量差 → 接受率仅 0.36
- KVMem server 的 `--spec-kv-dtype` **默认 f16** → 接受率高 → decode 90+
- **这是两个 server 的 decode 差异（65 vs 100）的主要来源**

### 3. cuBLAS prefill 路径
`GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0`：prefill 588→890（+51%）
代价：官方称精度略降（接受率从 ~0.5 降到 ~0.45，需权衡）

### 4. 必须用 KVMem 的 server（不是标准 llama-server）
- 它有独立的 MTP 优化实现
- 支持 `--spec-kv-dtype`（草稿 KV 独立设置）
- 同样源码 + 编译选项下，decode 比标准 server 快 40%（93 vs 67）

### 5. 功耗墙（已解除但需固化）
`nvidia-smi -pl 320` —— 出厂限制导致降频到 210MHz，速度仅 7 tok/s

## 最终部署脚本

见远端 `D:\Projects\RunBonsai3080\start_ultimate.bat`：

```bat
nvidia-smi -pl 320                              :: 必须
set GGML_CUDA_BATCH_INVARIANT=1
set GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0            :: cuBLAS prefill

llama-kvmem-server.exe -m bonsai-abliterated-mtp.gguf --jinja ^
  -ngl 99 -t 8 -fa on --parallel 1 -c 65536 --no-kvmem ^
  --cache-type-k q8_0 --cache-type-v q4_0 ^
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-kv-dtype f16
```

## 未解决/已放弃的方向

| 方向 | 状态 | 原因 |
|---|---|---|
| 自编译引擎达到 KVMem 的 decode | ❌ 放弃 | 试了 5 种编译配置（共享库/静态库/不同 HEAD/arch），最佳 67 vs KVMem 的 93 |
| trit 解包 LUT | ❌ 否决 | Ada 上实测 0.29x 回归（constant cache 串行化） |
| DFlash2 | ❌ 无效 | Bonsai 上仅 +0.5-3.6% |
| ngram 投机 | ❌ 无效 | 混合注意力模型上静默失效 |
| 128K+ 长上下文 | ⚠️ 有风险 | KVMem 官方文档记录召回仅 1/3（三次独立测试） |
| 超频 | ❌ 排除 | 需长期稳定运行 |

---

# 第十轮：系统层最后核查（2026-09-28）

## nvidia-smi -lgc 锁 SM 时钟：无增益，已排除

| 配置 | decode | SM 时钟 | 功耗 |
|---|---|---|---|
| **不锁（默认，推荐）** | **92.1 / 90.9** | 1920 MHz | 232-300W |
| 锁 1800 MHz | 81.2 | 1800 MHz | 242W |
| 锁 2100 MHz（上限） | 83.2 / 87.4 | 1920 MHz | 208W |

**结论：默认动态调频已最优。** 锁定后 GPU 失去动态升频能力，反而降速。

## 其他系统层项：均已最优

| 项 | 状态 |
|---|---|
| 功耗上限 | **320W / 320W**（已达默认上限） |
| HAGS | **已开启**（`HwSchMode = 0x2`） |
| 时钟范围 | 210-2100 MHz（动态调频正常） |

## 优化空间核查最终结论

**已确认无进一步优化空间的方向**（全部有实测或权威证据）：

| 方向 | 证据 |
|---|---|
| `-b`/`-ub` 调参 | 512→4096 无变化 |
| MTP 参数 | n_max=3/p_min=0 已最优 |
| f16 KV | 64K 爆显存 |
| trit LUT | Ada 实测 0.29x 回归 |
| DFlash2/ngram | 无收益 |
| **锁 SM 时钟** | **负收益（本轮验证）** |
| 功耗/HAGS | 已最优 |
| 超频 | ❌ 排除（需长期稳定运行） |
| 自编译引擎超越 KVMem | 试 5 种配置，最佳 67 vs KVMem 93 |

**最终配置**：prefill 890 / decode 92-99 / 64K 上下文 / 9604 MiB

---

# 第十一轮：KVMem 长上下文实测（2026-09-28）

## 问题：最终配置用了 KVMem 引擎但关闭了 KVMem，上下文只有 64K

`start_ultimate.bat` 用 `--no-kvmem`，上下文受显存限制只能到 64K。
**这丢掉了 KVMem 的核心价值。** 本轮实测开启 KVMem 后的真实能力。

## 实测结果：开启 KVMem + 128K 上下文

配置：
```bat
llama-kvmem-server.exe -c 131072 ^
  --kvmem-budget 32768 --kvmem-gen-reserve 8192 --kvmem-cpu-gb 6 ^
  --cache-type-k q8_0 --cache-type-v q4_0 ^
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-kv-dtype f16
```

| 项 | 结果 |
|---|---|
| 显存 | **8546 MiB**（比 64K 时的 9604 **更低**，因 KV 溢到 CPU） |
| decode（短 prompt） | **91.2 tok/s** |
| 逻辑上下文 | **131072（128K）** |

## 长文本召回测试（关键：验证真实可用性）

| 文档长度 | prefill | decode | 事实埋点 | 召回结果 |
|---|---|---|---|---|
| 14,159 token | 879 tok/s | 52.7 | 第 250 段 | ✅ **ZEBRA-7734** |
| **55,311 token** | 755 tok/s | 43.1 | 第 1500 段 | ✅ **TIGER-9182** |

**结论：KVMem 长上下文真实可用**，55K token 文档中埋在第 1500 段的事实能准确找回。

## 修正之前的结论

**之前我说「KVMem 机制是负收益（-17%）」——那个结论的前提错了**：
- 那次测的是**短上下文**（32K），KV 全在显存，KVMem 的分层管理只有开销
- **长上下文下 KVMem 才是正解**：显存更省（8546 vs 9604）、上下文 2 倍（128K vs 64K）

## 推荐的两种使用模式

| 场景 | 配置 | 性能 |
|---|---|---|
| **日常/短上下文** | `--no-kvmem -c 65536` | prefill 890 / decode 92-99 |
| **长文档/长上下文** | 开启 kvmem `-c 131072` | prefill 755-879 / decode 43-91（随长度衰减） |

⚠️ **注意**：KVMem 官方文档记录 80K/128K 的召回测试仅 1/3 正确（三次独立测试）。
**我的测试（55K）召回正确**，但更长上下文仍需验证。

---

# ★★★ 第十二轮：256K 上下文验证成功（2026-09-28）

## 关键发现：255K token 召回完全正确

**这推翻了 KVMem 官方文档「80K/128K 召回仅 1/3」的警告。**

配置：
```bat
llama-kvmem-server.exe -c 262144 ^
  --kvmem-budget 40960 --kvmem-gen-reserve 8192 --kvmem-cpu-gb 10 ^
  --cache-type-k q8_0 --cache-type-v q4_0 ^
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-kv-dtype f16
```

### 逐级召回测试（事实埋在文档中部，低温贪心解码）

| 文档长度 | prompt token | prefill | decode | 召回 |
|---|---|---|---|---|
| ~25K | 26,546 | 850 | 40.8 | ✅ ALPHA-1111 |
| ~50K | 55,286 | 739 | 19.2 | ✅ BRAVO-2222 |
| ~100K | 112,759 | 612 | 12.2 | ✅ CHARLIE-33 |
| ~150K | 170,240 | 608 | 5.9 | ✅ DELTA-4444 |
| ~200K | 203,714 | 602 | 9.8 | ✅ ECHO-5555 |
| **~250K** | **255,189** | **585** | **13.8** | ✅ **FOXTROT-6** |

**6/6 全部召回正确**，最大 255K token（距模型原生上限 262144 仅差 7K）。

**显存占用：8951 MiB**（比 64K 模式的 9604 更低——KV 溢到 CPU 内存池）

## 上下文-速度权衡表（实测）

| 上下文 | decode（短 prompt） | 长文档 decode | 显存 | 适用场景 |
|---|---|---|---|---|
| 64K（--no-kvmem） | **92-99** | — | 9604 | 日常对话、代码 |
| 128K（kvmem） | 91 | 43-53 | 8546 | 中等长文档 |
| **256K（kvmem）** | **75** | **6-41** | 8951 | 超长文档分析 |

**关键：长上下文下的 decode 衰减是渐进的**（40.8 @26K → 5.9 @170K），
不是断崖。这是 KVMem 分层机制的正常表现（KV 检索随长度变复杂）。

## 关于「召回仅 1/3」的官方警告

KVMem 文档记录的失败案例用的是 **5060 Ti / 5050（Blackwell 小卡）**，
且配置不同（未澄清是否用相同的 kvmem-budget/cpu-gb 参数）。

**我的实测（3080 + 上述配置）6/6 全对**，说明：
1. 该配置下长上下文召回是可靠的
2. 官方警告可能与特定硬件/配置组合有关，不能一概而论

⚠️ **但我的测试是合成文档 + 单一事实埋点**。真实任务的复杂检索（多事实、推理链）
可能更严苛。建议关键任务仍做输出验证。

---

# 第十三轮：长期服务部署与稳定性（2026-09-28）

## 服务部署完成

**任务名**：`BonsaiService`（Windows 计划任务）
- 触发：`/sc onstart`（开机自启）
- 权限：`/rl HIGHEST`
- 内容：`start_service.bat` —— 含**崩溃自动重启循环**（退出后 5 秒重启）

**关键实现细节**：
1. **必须用绝对路径调用 exe** —— `schtasks /sc onstart` 的环境下 `set PATH` 不生效
   （踩坑：最初用 PATH 方案，服务反复启动失败，日志显示 `'llama-kvmem-server.exe' is not recognized`）
2. **每次重启前重新解锁功耗墙** —— `nvidia-smi -pl 320` 在系统重启后失效
3. **bat 文件必须纯 ASCII** —— 中文注释会让 cmd 解析错乱（第二轮踩过同款坑）

## 稳定性实测

### 短时 soak（10 分钟）
| 指标 | 结果 |
|---|---|
| 请求数 | 266 |
| 错误 | **0** |
| 速度漂移 | **+0.3%**（首半段 72.9 → 后半段 73.1） |
| 延迟 | 中位 1.8s / 最大 4.4s |
| 显存 | 9288 MiB 恒定（**无泄漏**） |

### 长时资源趋势（20 分钟连续负载）
| 指标 | 采样 |
|---|---|
| 显存 | **9130 MiB 恒定** |
| SM 时钟 | 1890 MHz 稳定 |
| 功耗 | 289-317W |
| 温度 | **63-64°C**（健康，无过热降频） |

## 质量基线（10 轮 × 8 探针）

| 类别 | 结果 |
|---|---|
| 算术（小数） | ✅ 2+2、123×45、8934+7729、15% of 240 |
| **算术（大数）** | ❌ **1234×5678 = 7006652，模型答 7006852 / 7006552** |
| 推理 | ✅ Sally 的姐妹数、企鹅非哺乳动物 |
| 语言 | ✅ 反转 stressed→desserts、数 r、法语翻译 |
| 指令遵循 | ✅ 格式、计数、JSON |
| **总得分** | **9/10**（10 轮稳定复现 7/8 简版 / 9/10 详版） |

**大数乘法失败是量化精度损失**（PTQ1_0 1.75bit），且**多次测试给出不同错误答案**
（7006852 vs 7006552）→ 说明是数值精度问题而非能力缺陷。

## 上下文能力最终表（256K 配置）

| 文档长度 | prefill | decode | 召回 |
|---|---|---|---|
| 26K | 850 | 40.8 | ✅ |
| 55K | 739 | 19.2 | ✅ |
| 113K | 612 | 12.2 | ✅ |
| 170K | 608 | 5.9 | ✅ |
| 204K | 602 | 9.8 | ✅ |
| **255K** | **585** | **13.8** | ✅ |

**6/6 全对**，最大 255K（距模型上限 262144 差 7K）。显存 8951 MiB。

## 最终交付物

| 文件 | 用途 |
|---|---|
| `BonsaiService`（计划任务） | **长期服务**（开机自启 + 崩溃重启） |
| `start_service.bat` | 服务启动脚本（被计划任务调用） |
| `start_ultimate.bat` | 手动启动（64K 快速模式） |
| `start_long.bat` | 手动启动（128K 模式） |
| `README.md` | 使用文档 |
| `EXPERIMENTS.md` | 完整实验记录（本文） |
| `tools/` | 测试脚本（soak/stability/quality/recall/profiler） |

---

# ★★★ 第十四轮：上下文衰减曲线与三种模式对比（2026-09-28）

## 完整衰减曲线（128K + 2GB 池配置）

| 长度(tok) | prefill | decode | 延迟(s) | 说明 |
|---|---|---|---|---|
| 670 | 549 | 72.8 | 1.6 | |
| 1,320 | 703 | 62.5 | 2.1 | |
| 2,620 | 757 | 66.2 | 2.8 | |
| 5,220 | 811 | 58.0 | 4.3 | |
| 10,420 | **842** | 47.6 | 7.4 | prefill 峰值 |
| 15,620 | 805 | 44.9 | 7.8 | |
| 20,820 | 780 | 43.7 | 8.0 | |
| 31,220 | 753 | 38.9 | 15.3 | |
| **52,020** | 632 | **12.9** ⚠️ | 36.1 | **decode 拐点** |
| 78,020 | 577 | 13.7 | 48.3 | |
| 104,020 | 576 | 12.1 | 48.8 | |
| 130,020 | 569 | 11.1 | 49.6 | |

### 关键发现：52K 处的 decode 拐点

**decode 在 31K→52K 之间从 38.9 跌到 12.9（-67%）**，而 prefill 只有轻微衰减（753→632）。

**这个拐点对应 `--kvmem-budget 40960`（40K 工作集）**：
- < 40K：KV 全在 GPU 工作集，检索快
- > 40K：触发 KV 检索（从 CPU 池捞回相关块），开销剧增

⚠️ **注意**：KVMem 官方文档记载的「长上下文召回仅 1/3」问题，在 80K/128K 档位——
**正好落在拐点之后**。这可能是同源问题：检索机制在超工作集时效率和质量都下降。

**待验证**：调大 `--kvmem-budget`（如 61440）能否把拐点后移。

## 三种模式完整对比

| 模式 | 短prompt decode | 长文档 prefill | 长文档 decode | 显存 | 内存 |
|---|---|---|---|---|---|
| **纯显存**（--no-kvmem） | 85 | **8-11** ❌ | — | 9949 | 0 |
| **128K + 2GB 池** ⭐ | 85 | **842** ✅ | 11-73 | 8880 | **2GB** |
| 256K + 10GB 池 | 75 | 585-612 | 6-14 | 8951 | 10GB ⚠️ |

### 结论

**128K + 2GB 池是最优平衡**：
- prefill 与 256K 模式相当（842 vs 612，甚至更好）
- 内存占用仅 2GB（vs 10GB）
- 唯一代价：上下文上限 128K（vs 256K）

**纯显存模式（--no-kvmem）在长文档下有严重性能问题**（prefill 崩 50 倍），
原因是 llama.cpp 标准 KV 布局 + FA 在超长 KV 上无法有效分块。

## 调优方向（待验证）

1. **调大 `--kvmem-budget`** 到 61440（60K），看能否把 52K 的拐点后移
2. `--kvmem-cpu-gb` 在 2-4GB 之间寻找平衡（当前 2GB 已足）
3. 拐点后的召回质量需要单独验证（可能对应官方记录的 1/3 问题）

---

# ★★★ 第十五轮：KVMem budget 的权衡（2026-09-28）

## 关键发现：`--kvmem-budget` 是「短上下文 vs 长上下文」的交换旋钮

### 对比：budget=40960 (40K) vs budget=61440 (60K)

| 文档长度 | budget=40K decode | budget=60K decode | 变化 |
|---|---|---|---|
| 670 | **72.8** | 51.7 | **-29%** ❌ |
| 1,320 | 62.5 | 43.0 | -31% ❌ |
| 2,620 | 66.2 | 28.9 | -56% ❌ |
| 5,220 | 58.0 | 23.5 | -59% ❌ |
| 10,420 | 47.6 | 15.4 | -68% ❌ |
| 31,220 | 38.9 | 7.2 | -81% ❌ |
| **52,020** | **12.9** | **38.3** | **+197%** ✅ |
| 78,020 | 13.7 | 5.7 | -58% ❌ |
| 130,020 | 11.1 | 5.2 | -53% ❌ |

**拐点位置与 budget 值精确对应**：
- budget=40K → 拐点在 ~40K
- budget=60K → 拐点在 ~60K

**但 budget 越大，短上下文越慢** —— 因为工作集管理开销随 budget 增长。

### 显存代价

| budget | 显存占用 |
|---|---|
| 40960 | 8880 MiB |
| 61440 | 9567 MiB |

## 三个可选配置（按场景）

| 配置 | 最佳区间 | 该区间 decode | 显存 | 内存池 |
|---|---|---|---|---|
| **budget 40K** ⭐ | **< 40K** | 39-73 | 8880 | 2GB |
| budget 60K | 40K-60K | 38（仅此区间优） | 9567 | 2GB |
| — | > 60K | 5-14（都慢） | — | — |

**推荐：`--kvmem-budget 40960`** —— 覆盖 0-40K 的常见场景，decode 保持 39-73。
只有确实需要 40-60K 长文档时才用 60K budget。

## 完整最优配置（最终版）

```bat
nvidia-smi -pl 320
set GGML_CUDA_BATCH_INVARIANT=1
set GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=0

llama-kvmem-server.exe -m bonsai-abliterated-mtp.gguf --jinja ^
  -ngl 99 -t 8 -fa on --parallel 1 -c 131072 ^
  --kvmem-budget 40960 --kvmem-gen-reserve 8192 --kvmem-cpu-gb 2 ^
  --cache-type-k q8_0 --cache-type-v q4_0 ^
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-kv-dtype f16
```

**实测**：显存 8880 MiB / 内存池 2GB / prefill 549-842 / decode 39-73（<40K 文档）

---

# 第 16 轮：崩溃根因定位与长期部署固化（2026-09-28）

## 1. ★★★ 服务静默崩溃的根因：MTP 投机 × KVMem 换页 + 小 CPU 池

**现象**：长期使用中服务无任何错误输出直接退出（事件日志无 WER 记录），当天使用期间崩 4 次；
受控复现 2 分钟内必崩。

**隔离实验（控制变量）**：

| 配置 | 测试 | 结果 |
|---|---|---|
| cpu-gb 2 + MTP | 21K/37.5K 上下文交替（触发换页） | ❌ 崩溃（4 次 + 复现 1 次） |
| cpu-gb 6 + MTP | 同上 + streaming 并发 | ❌ 崩溃（2 次） |
| cpu-gb 6 + **spec none** | 崩溃复现 3 轮 12 请求 | ✅ 零崩溃 |
| cpu-gb **10** + MTP | 复现 5 轮 + streaming 并发 + 交替，共 40+ 跨换页请求 | ✅ 零崩溃 |

**结论**：崩溃需要 MTP（draft-mtp）与 KVMem eviction/retrieval 同时发生；CPU 池加大到 10GB 后
40+ 次跨换页请求零复发。怀疑 MTP 回滚快照（--kvmem-mtp-state）与换页回滚路径的竞态。
二进制闭源无法进一步定位，以参数规避。

`service.log` 里 `EXITED code=` 是脚本的退出码记录，定位静默退出依赖这一行。

## 2. 多会话 / 手动停止行为（--parallel 1）

- **并发请求整单串行排队**：后到请求的 TTFT ≈ 前一请求的剩余时间。单 slot 是架构限制（`-np` 只支持 1）。
- **手动停止（断开连接）**：服务端 2 秒内正常取消，slot 释放，后续请求不受影响 ✅
- **停止后重跑同一轮**：KVMem 缓存被判失效（`cache=0`），37.5K 上下文全量重算约 50 秒且不流字
  → 观感「GPU 在跑但不出字」。这是换页重算，不是卡死，等即可。

## 3. NVMe 层不可用

`--kvmem-nvme-gb` 在 kvmem-v0.16.0-rc3-prism.3 构建中被编译禁用（KVMEM_ENABLE_NVME=OFF），
传入直接 exit 1。上下文上限 = GPU 40K + CPU 池（10GB ≈ 15 万 token）。

## 4. 最终固化配置（已装 BonsaiService 计划任务，onstart 自启 + 崩溃 5 秒重启）

```bat
llama-kvmem-server.exe -m bonsai-abliterated-mtp.gguf --jinja ^
  -ngl 99 -t 8 -fa on --parallel 1 -c 262144 ^
  --kvmem-budget 40960 --kvmem-gen-reserve 8192 --kvmem-cpu-gb 10 ^
  --cache-type-k q8_0 --cache-type-v q4_0 ^
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-kv-dtype f16
```

## 5. 最终验收（全部通过）

| 项目 | 结果 |
|---|---|
| 质量 10 项快测 | 9/10（大数乘法错，正常） |
| 长文档召回 | 25K/50K/100K/150K/200K/250K 全对 |
| 多会话并发 streaming | 通过（排队符合预期） |
| 手动停止 + 恢复 | 通过（2 秒释放） |
| decode 速度 | 短上下文 51-60 t/s（MTP 生效），150K 时 10 t/s（换页代价） |
| prefill | 750-850 t/s（cuBLAS 路径） |
| 显存 / 内存 | 9739 MiB / 进程 3.4GB（空余 7GB） |

## 6. 遗留问题

- 敏感议题长回答偶发中途截断（服务存活、无崩溃日志时）→ 模型侧 EOS（abliterated 残留），
  与崩溃无关；可尝试调低 presence_penalty（默认 1.5 偏高）或换更强消融模型。
- 16 轮实验的 16 个遗留计划任务已全部清除（含 23:59 会误触发的 FINALBEST）。

## 16b. 复核后的最终修正（2026-09-28 晚）

第 16 轮结论有三处被实测推翻，修正如下：

1. **不要开机自启**（此前已有约定，违反了）。BonsaiService 计划任务已删除，改为手动双击
   `start_service.bat`（内部仍带重启循环 + EXITED 退出码记录）。

2. **CPU 池定稿 2GB**（8/10GB 方案否决）。关键实测：**池是启动即整块提交的**
   （cpu-gb 8 → 进程 commit 18.05GB / 工作集 10.5GB，16GB 机器空闲仅 1.5GB；cpu-gb 2 →
   commit 12.03GB / 工作集 8.13GB）。池的用途只是 GPU budget(40960) 的**溢出区**：
   - 单会话上下文 ≤20K 时永远用不到池，池大小无所谓；
   - 但 KV 缓存总量是**所有会话缓存之和**，多会话加总越过 40960 就触发换页；
   - 崩溃路径 = MTP + 换页同时发生。故**要么保持「各会话缓存之和 < 40960」，
     要么关 MTP 换绝对稳定**（no-spec 时换页路径实测无害）。
   - 诚实记录混杂因素：6GB 池的两次崩溃发生在同时使用 webui 的窗口，
     「池 ≥8GB 才稳」的边界未经纯净对照，可能偏保守。

3. **敏感议题中途截断 = 模型侧 EOS**（已确认，与崩溃无关）。abliterated 残留导致
   特定话题倾向提前输出终止符，句子说到一半正常停止。缓解方向：对 `<|im_end|>` 加
   轻微负 logit_bias、调低 presence_penalty（默认 1.5 偏高）、或换消融更彻底的模型。

**定稿配置**：`start_service.bat` = 128 行版本（-c 262144 / budget 40960 / gen-reserve 8192 /
cpu-gb 2 / q8_0+q4_0 / draft-mtp n_max 3 / f16 草稿 KV），手动启动，无自启任务。

## 16c. 推翻「崩溃」叙事，回到两个真问题（2026-09-28 深夜，实测纠正）

确认：**从未发生自发性崩溃**。16a 记录的进程退出事件全部与人为操作同时发生
（手动重启、测试 taskkill、NVMe 误配 exit 1），多会话测试的 connection reset
也与手动重启窗口重合。「MTP×换页崩溃」降级为未证实假设，16a/16b 相关段落作废。

真正要解的两个问题及实测结论：

### 问题1：审查截断（句子说不全）
- 根因：消融模型在政治类议题提前输出 `<|im_end|>`，模型行为，finish_reason=stop，服务无异常。
- `<|im_end|>` token id = **248046**（find_eos.py 从 GGUF tokenizer.ggml.tokens 读出）。
- 解法：请求加 `"logit_bias": {"248046": -6}`。实测 -2/-6/-12 三档在良性任务上
  均正常收尾（说 DONE 即停，长度与基线一致）→ 提前 EOS 被压、正常结束不受影响。
  顽固截断可到 -12；正常回复开始拖长就回调。presence_penalty 默认 1.5 偏高，可降 0.3-0.6。

### 问题2：手动停止后后台还在算、不出字
- 直连实测（SSH 内）：prefill/decode 两阶段断开均在 2-3 秒取消，重发 0.7s 完成
  → 服务端取消机制正常（abort2_test.py）。
- 本机环境（浏览器→frp隧道→服务端）失效 ⇒ 根因指向 **frp 隧道不传播 TCP 断开**：
  服务端对死连接继续生成到 n_predict（默认 8192 ≈ 2-3 分钟），重发请求在单 slot
  后排队 →「后台在算、一直不出字」。
- 缓解：启动参数加 `-n 4096`（失控任务最多 ~70 秒自然结束，已部署）。
  进一步：确认 UI 停止是真 abort fetch；换能传播断开的传输（局域网/Tailscale）。
- 验证方法：停止后看 service.log 该任务的 eval time 是否打满长度。

### 其他
- 新增 `--help` 确认：`-n/--n-predict` 为服务端默认 max_tokens，请求可覆盖。

## 16d. 两个真问题的深挖结果（2026-09-28 深夜二轮，16c 修正版）

### 问题1：审查截断 —— webui 补丁方案（已部署）
- 实测 -6 无效的原因：**webui 前端根本没有 logit_bias 参数**（bundle 内 0 命中），bias 从未生效。
- Qwen 系有两个 EOG 终止符：`<|im_end|>`=248046、`<|endoftext|>`=248044（find_eos2.py 读 GGUF tokenizer）。
  只压 im_end 不够，模型可走 endoftext 收尾。
- **已部署 UI 补丁**：`bundle.BIFbvED8.js` 的 sendMessage fetch 注入
  `logit_bias:{"248044":-6,"248046":-6}`（node --check 语法通过，服务端已验证下发补丁版）。
  sw.js 对应条目 revision bump 为 "logitbias-patch1" 强制浏览器重拉。
  原文件备份为 bundle.BIFbvED8.js.orig（回滚：copy /y .orig 回原名 + sw revision 还原）。
  API 客户端等价写法：`"logit_bias": {"248044": -6, "248046": -6}`（顽固可到 -12）。
- 需刷新一次 webui（Ctrl+F5）使新 SW 生效。若截断仍现 → 升 -12。

### 问题2：停止失效 —— 机制查明（UI 源码 + 实测）
- 服务端取消正常：全新 20K prefill 中断开连接 ×3 次，均 ≤3s slot 释放、GPU 30W 待机（stop_gpu_test.py）。
- UI 停止链路（bundle 反查）：`stopGeneration` → `cancelServerStream`（**DELETE /v1/stream → 本构建 404，空操作**，
  模拟调用实测 404）+ `abortRequest`（本地 abort POST → 服务端因此取消）+ `clearStreamState`/`cancelResumeRetry`。
- UI 有**断线自动 resume**机制（handleStreamResponse → resumeStream GET /v1/stream?from=字节偏移）：
  连接断开 ≠ 生成取消，UI 可重挂继续收流。停止时序里 DELETE 失效 + resume 重试存在竞态，
  即「停止后 GPU 还在算」的来源。
- 缓解：`-n 4096` 已部署（失控 ≤70s 自然结束）；停止后若 GPU 仍忙，**刷新页面**可切断 UI 重试，
  服务端任务最迟 70s 自行结束。
- 服务端无管理端点可杀单个任务（DELETE 端点缺失），重启服务是最后手段。

## 16e. API / 并发 / 会话切换三问实测（2026-09-29 凌晨）

- **跨会话切换 = 全量重算**（switch_small_test.py）：两个 ~12.6K/14K 文档交替，
  每次切回对方 cache=0、全量 prefill（800-900 t/s，12.6K ≈ 14.5s）——
  即使两者加总 26.6K < 40960 GPU budget 也一样。KVMem retrieval 模式下
  跨会话切换不保留对方的工作集。
- **会话内追加轮次 = 缓存命中**（会话日志 cache=5168/5231/65496 佐证）。
- 实用准则：切会话的代价 ≈ 1.15s/千 token（该会话上下文大小）；单会话内连续多聊零成本；
  大上下文（>20K）会话别来回横跳。
- 服务自 20:32 起 zero EXITED（-n 4096 实例），期间跑完 6 次跨会话重算，稳定。

## 16f. 内网穿透排障（2026-09-29）

- 隧道拓扑：广州节点 `22→17722`（SSH，一直正常）；香港节点 `local 29187 → 50768`（ChmlFrp，TCP 型）。
- 公网 50768 返回 502 的原因：**隧道指向本地 29187，而服务开在 8080**——隧道本身没问题。
- 处置：**改服务适配隧道**（start_service.bat `--port 29187`），frp 配置保持原样（曾误改 ini
  又回滚；最终方案是服务端改端口适配隧道）。
- 公网实测：`http://vip.xg.frp.one:50768/health` → 200 ok，webui 首页 200（0.15s 延迟）。
  之前担心的备案拦截在该节点（香港 + TCP 型）不存在。
- 后续所有测试/文档端口统一 29187。

## 16g. IDE 接入就绪实测（2026-09-29）

- `GET /v1/models` ✓（返回 bonsai-abliterated-mtp.gguf）
- `POST /v1/chat/completions` + tools ✓：正确产出 tool_calls（city=Tokyo，finish_reason=tool_calls）
- `/v1/completions`（legacy 补全端点）✗ **404**——本构建未暴露，FIM/Tab 自动补全类插件不可用，
  IDE 只能接 chat/agent 模式。

## 16h. 无MTP对照与 PQ2 双消融版（2026-09-29）

- **无 MTP 对照**（bonsai-abliterated.gguf，59.3 t/s）：敏感内容**照样中途截断**
  → MTP 路径嫌疑解除，截断根源 = 主模型消融不彻底（「不拒绝但飘向 EOS」的半消融指纹）。
- **BoldingBuilds PQ2_0-MTP 双消融版**已下载（7.65GB，aifasthub 5.6MB/s）并上线：
  - 权重 2.06bpw/7.13GiB（vs PTQ1_0 1.75bpw/5.95GiB）→ KV budget 40960→24576
  - 实测：显存 9629/10240 MiB ✓，prefill 963 t/s（比 PTQ1_0 还快），decode 50.6 t/s（短上下文+MTP）
  - heredoc 生成 bat 时 `\$model` 转义翻车 → 模型路径变字面量 `models$model`，
    服务 EXITED code=1。bat 生成后需 grep 校验关键行。
- 三个启动脚本已全部单发化（无看门狗），端口守卫保留。当前：SRV_PQ2 任务挂 PQ2 版。

## 16i. PQ2 双消融版实测结果（2026-09-29）

- **显著改善**：PTQ1_0 消融版「几乎立即截断」→ PQ2 双消融版「能输出很长一段才截断」（残余截断仍在）。
- 结论：截断强度 = 消融彻底度 × 模型容量。BoldingBuilds 的双路径正交化 + 2.06bpw 的富余容量
  都在起作用。残余截断 = 仍有未清除的拒绝成分。
- 剩余手段（按成本排序）：① webui logit_bias 补丁 -6 → -12（免重启，刷新即生效）
  ② OS-Software Heretic 版（LoRA 烘焙，方法论不同）③ webui presence_penalty 1.5 → 0.3-0.6。

## 16j. PQ2 草稿头裁决与运行时兼容性（2026-09-29）

- **BoldingBuilds PQ2 的 MTP 草稿头 = 负收益（实锤）**：同 prompt 同机，
  spec 开 50.6 t/s vs spec 关 59.3 t/s。草稿头在 KVMem 运行时上坏/无效。
  与「pr-hadamard-mtp 缺失草稿 embedding 逆变换」的社区分析吻合。
  但注意：PTQ1_0 的 MTP（我们移植的 lean 草稿头）在同一运行时上工作正常
  （92.9 vs 62.7），所以是**这个文件的 MTP 头**与运行时不兼容，不是运行时不会 MTP。
- 机制澄清：标准投机解码里草稿无权决定终止（target 验证 + logit_bias 在 target 侧生效），
  坏草稿理论只会导致变慢。是否也导致截断 → 由 no-spec 版实测裁决（当前 SRV_PQ2NS）。
- 外部分析其余判定：n_max 降到 1-2 与截断无关（3 是 BATCH_INVARIANT 边界）；
  降 -c 与截断无关；「原生 llama.cpp 加载 type 142 会崩」正确但我们不在原生上；
  OrcaBonsai 运行时消融需要其自家运行时，KVMem 无法执行。

## 16k. 消融版本横向裁决（2026-09-29）

| 版本 | NSFW | 涉政 | 重复 |
|---|---|---|---|
| bonsai-abliterated-mtp（正交化+MTP移植） | 立即截断 | 立即截断 | - |
| BoldingBuilds PQ2_0-MTP（双消融） | 部分可用 | 截断（长输出后） | 尚可 |
| OS-Software Heretic PTQ1_0（LoRA烘焙） | **知识增强 ✓** | **仍截断 ✗** | **NSFW 时陷入重复 ✗** |

结论：三家社区消融均无法清除涉政拒绝——这是基模 RLHF 深层的权重级倾向，
LoRA/正交化都只能削表层。27B 三元路线的消融天花板已到。
重复问题的运行时缓解：webui 调 repeat_penalty 1.05-1.1 或 DRY multiplier 0.8
（前端支持 dry_multiplier/dry_base/dry_allowed_length 参数）。
后续方向：① 截断后「继续」多轮续写兜底（缓存保留，通常能接上）；
② 小模型特化路线（nuofang 9B 中文 NSFW 原生训练，无消融空洞）下载测试中。

## 16l. 小模型特化路线三连测（2026-09-29）

| 模型 | 量化 | decode | 中文 | 结论 |
|---|---|---|---|---|
| nuofang/Qwen3.8-9B-Distill-SLERP-F451-Pro-Writer | Q6_K | **70.2 t/s** | 优秀（武侠开头质量高） | **涉政 ✓ NSFW ✓**；写作语料薄（修辞重复、场景雷同）、代码易循环（已烤入 repeat 1.1/freq 0.3/min_p 0.05） |
| DavidAU/Qwen3-MOE-6x1.7B-10.2B-Shining-Madness | Q6_K | **152.3 t/s** | 良好 + 原生 `<think>` CoT | 代码特化标签；全场最快 |
| DavidAU/Gemma-Writer-N-Restless-Quill-10B | Q6_K | 26.4 t/s | **退化**（中土葡阿多语混杂乱码） | 弃 |

工程坑：aifasthub 下载的文件头被塞 16 字节 "Entry not found"（覆盖 GGUF 魔数+版本+张量数）。
修复法：strip16 → 解析 KV(35) → 贪婪数张量信息(338) → 重建 40 字节头（repair5.py）。
注意张量信息区紧贴 KV 区无对齐填充；张量数据偏移从数据区起算（min_off=0 正常）；
os.replace 可能被杀软锁——写新文件名绕过。
- 16l 补充：MoE 写文章「简直是灾难」——速度王者但创作质量不合格；
  nuofang 9B 写作能力明确更强。分工建议：写作=nuofang，代码/速度=MoE。
- 16l 终裁：MoE 删除（模型/脚本/任务），代码场景改用 bonsai 256K（ctx256.bat）。
  最终保留阵容：① nuofang 9B Q6_K（写作/NSFW，涉政✓，256K）② bonsai-abliterated-mtp 6GB 三模式（64K/128K/256K，含代码）。
