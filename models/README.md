# models / 模型文件说明

模型文件不入 git（体积大），本目录放下载好的 `.gguf`。

## bonsai-abliterated-mtp.gguf（主力，三模式通用）

- 用途：Bonsai-2-27B 三元量化（PTQ1_0，1.75bpw），跑三个上下文模式（64K/128K/256K）
- SHA256: `aad9e983b639b23a7b935a1d4d62ff53632092759ce6c83f01748798102d367e`（6,006 MB）
- **这个文件是自己合成的**，没有现成下载：
  1. 下载 `bonsai-ptq1-lean.gguf`（官方 MTP 版）与无审查社区版 GGUF（5,946,648,928 B）
  2. 运行 `tools/mtp_migrate.py` 做 二进制层手术：保留无审查版数据区 + 追加 lean 的 15 个 `blk.64.*` MTP 张量 + `block_count` 64→65
  3. 合并后两段数据区 SHA256 必须与源文件逐位一致（脚本自校验）
- lean 源文件下载（hf-mirror）：
  - `https://hf-mirror.com/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf`
- 官方原版（未消融，参考用）：`https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf`
