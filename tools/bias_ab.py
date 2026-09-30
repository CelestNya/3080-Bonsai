# -*- coding: utf-8 -*-
"""
logit_bias A/B 工具：把你的测试 prompt 写进同目录 prompt.txt，然后运行：
    python bias_ab.py
会依次用 不压 / -6 / -12 三档跑同一个 prompt，输出各自的结尾和统计。
结果只存在这台机器上，不经过任何网络。
"""
import json, urllib.request, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = "http://127.0.0.1:29187"

try:
    prompt = io.open("prompt.txt", encoding="utf-8").read().strip()
except FileNotFoundError:
    print("请先在本目录创建 prompt.txt，放入你的测试 prompt（完整对话开场白）")
    sys.exit(1)

def run(bias):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500, "temperature": 0.7,
        "presence_penalty": 0.4,
    }
    if bias is not None:
        body["logit_bias"] = {"248044": bias, "248046": bias}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = __import__("time").time()
    d = json.loads(urllib.request.urlopen(r, timeout=900).read().decode())
    txt = d["choices"][0]["message"]["content"]
    tag = "无bias" if bias is None else f"bias {bias}"
    print(f"===== {tag} =====")
    print(f"finish_reason={d['choices'][0].get('finish_reason')}  "
          f"completion={d['usage']['completion_tokens']} tok  用时 {__import__('time').time()-t0:.0f}s")
    print(f"结尾 300 字:\n{txt[-300:]}")
    print()

for b in (None, -6, -12):
    run(b)
print("对比要点：看三档各自的「结尾 300 字」是否完整收束、有没有说到一半断掉。")
