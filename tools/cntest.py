# -*- coding: utf-8 -*-
import json, urllib.request, time
BASE = "http://127.0.0.1:29187"
def gen(prompt, max_tokens=600, tag=""):
    body = json.dumps({"messages":[{"role":"user","content":prompt}],
                       "max_tokens":max_tokens,"temperature":0.7}).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=600).read().decode())
    t = d.get("timings", {})
    txt = d["choices"][0]["message"]["content"]
    print(f"== {tag}: prefill={t.get('prompt_per_second',0):.0f} t/s  decode={t.get('predicted_per_second',0):.1f} t/s  生成={d['usage']['completion_tokens']} tok")
    print(txt[:600])
    print()
gen("写一段500字的武侠小说开头：主角雨夜回到阔别十年的家乡，发现师父的墓被人挖了。要求有画面感和情绪张力。", tag="中文武侠开头")
