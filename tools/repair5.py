# -*- coding: utf-8 -*-
import struct, sys, os, time
p = sys.argv[1]
f = open(p, "rb")
def u32(): return struct.unpack("<I", f.read(4))[0]
def u64(): return struct.unpack("<Q", f.read(8))[0]
def rstr():
    n = u64(); return f.read(n).decode("utf-8", "replace")
sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def rdval(t):
    if t in (0,1,7): f.read(1)
    elif t in (2,3): f.read(2)
    elif t in (4,5,6,10,11,12): f.read(sizes[t])
    elif t == 8: rstr()
    elif t == 9:
        ety = u32(); n = u64()
        if ety == 8:
            for _ in range(n): rstr()
        else: f.read(sizes[ety]*n)
    else: raise ValueError("bad type %d" % t)

kv_count = u64()
alignment = 32
for _ in range(kv_count):
    k = rstr(); t = u32()
    if k == "general.alignment":
        alignment = struct.unpack("<I", f.read(4))[0]; continue
    rdval(t)
kv_end = f.tell()
f.seek(kv_end)
count = 0; min_off = None; names = []
while True:
    pos = f.tell()
    nlen = u64()
    if nlen == 0 or nlen > 4096: break
    name = f.read(nlen).decode("utf-8", "replace")
    ndims = u32()
    if ndims == 0 or ndims > 8: break
    dims = struct.unpack("<%dQ" % ndims, f.read(8*ndims))
    ttype = u32()
    off = u64()
    if off >= os.path.getsize(p): break
    min_off = off if min_off is None else min(min_off, off)
    count += 1
    names.append(name)
aligned = (pos + alignment - 1) // alignment * alignment
print("tensor_count =", count, " first=%s last=%s" % (names[0] if names else "-", names[-1] if names else "-"))
print("info_end=%d aligned_data_start=%d min_off=%d" % (pos, aligned, min_off))
if count < 10: sys.exit("bad count")

out = p + ".fixed"
with open(p, "rb") as src, open(out, "wb") as dst:
    dst.write(b"GGUF"); dst.write(struct.pack("<I", 3))
    dst.write(struct.pack("<Q", count)); dst.write(struct.pack("<Q", kv_count))
    src.seek(8); dst.write(src.read())
for a in range(20):
    try:
        os.replace(out, p); break
    except PermissionError:
        time.sleep(3)
with open(p, "rb") as chk: print("magic:", chk.read(4), "size:", os.path.getsize(p))
print("repaired OK")
