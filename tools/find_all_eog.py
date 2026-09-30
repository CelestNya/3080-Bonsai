# -*- coding: utf-8 -*-
# 扫描 GGUF 词表 + KV，列出运行时会当成 EOG 的全部 token
import struct
path = r"D:\Projects\RunBonsai3080\models\BoldingBuilds-abliterated-pq2-mtp.gguf"
EOG_NAMES = {"<|eot_id|>","<|im_end|>","<|end|>","<|return|>","<|call|>","<|flush|>","<|calls|>",
             "<end_of_turn>","<|endoftext|>","</s>","<|eom_id|>","<EOT>","_<EOT>","[EOT]","[EOS]",
             "<|end_of_text|>","<end_of_utterance>","<eos>","<turn|>","<|tool_response>",
             "<｜end▁of▁sentence｜>","[e~["}
f = open(path, "rb")
def rd(fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def rstr():
    n = rd("<Q"); return f.read(n).decode("utf-8", "replace")
rd("<4s"); rd("<I"); rd("<Q"); n_kv = rd("<Q")
sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip_val(t):
    if t in sizes: f.read(sizes[t])
    elif t == 8: rstr()
    elif t == 9:
        ety = rd("<I"); n = rd("<Q")
        if ety == 8:
            for _ in range(n): rstr()
        else: f.read(sizes[ety]*n)
kv_hits = {}
tokens_hits = {}
for _ in range(n_kv):
    k = rstr(); t = rd("<I")
    if k == "tokenizer.ggml.tokens":
        rd("<I"); n = rd("<Q")
        for i in range(n):
            s = rstr()
            if s in EOG_NAMES: tokens_hits.setdefault(s, i)
        break
    if k in ("tokenizer.ggml.eos_token_id","tokenizer.ggml.eot_token_id","tokenizer.ggml.eom_token_id",
             "tokenizer.ggml.fim_pad_token_id","tokenizer.ggml.fim_rep_token_id","tokenizer.ggml.fim_sep_token_id"):
        kv_hits[k] = rd("<I" if t == 4 else "<Q")
    else:
        skip_val(t)
print("== KV 声明的终止类 id ==")
for k, v in kv_hits.items(): print(f"  {k} = {v}")
print("== 词表中命中 EOG 名单的 token ==")
for s, i in sorted(tokens_hits.items(), key=lambda x: x[1]): print(f"  {i:>8}  {s}")
