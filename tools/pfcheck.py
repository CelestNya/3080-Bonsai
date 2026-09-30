import json, urllib.request, time
chunk = "The transformer architecture uses self-attention mechanisms to process sequential data efficiently. "
doc = chunk * 700   # ~10K tokens
body = json.dumps({"messages":[{"role":"user","content":doc + "\nReply only: ACK"}],
                   "max_tokens":5,"temperature":0.6}).encode()
r = urllib.request.Request("http://127.0.0.1:29187/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
t0=time.time()
d = json.loads(urllib.request.urlopen(r, timeout=600).read().decode())
t = d.get("timings", {})
u = d.get("usage", {})
print(f"prompt={u.get('prompt_tokens')} prefill={t.get('prompt_per_second',0):.0f} t/s  decode={t.get('predicted_per_second',0):.1f} t/s  wall={time.time()-t0:.1f}s")
