import struct, os, shutil, sys, json

LEAN = r"D:\Projects\RunBonsai3080\models\bonsai-ptq1-lean.gguf"
ABLIT = r"D:\Projects\RunBonsai3080\models\bonsai-abliterated.gguf"
OUT  = r"D:\Projects\RunBonsai3080\models\bonsai-abliterated-mtp.gguf"

SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def parse_header(path):
    """Return dict with version, raw kv entries, raw tensor-info entries, data_off."""
    with open(path, 'rb') as f:
        buf = f.read(16*1024*1024)  # header is ~11MB, enough
    assert buf[:4] == b'GGUF', path
    version = struct.unpack_from('<I', buf, 4)[0]
    n_tensors = struct.unpack_from('<Q', buf, 8)[0]
    n_kv = struct.unpack_from('<Q', buf, 16)[0]
    p = 24
    def skip(p, vt):
        if vt == 8:
            n = struct.unpack_from('<Q', buf, p)[0]; return p+8+n
        if vt == 9:
            et = struct.unpack_from('<I', buf, p)[0]
            cnt = struct.unpack_from('<Q', buf, p+4)[0]; p += 12
            for _ in range(cnt):
                p = skip(p, et)
            return p
        return p + SZ[vt]
    kvs = []  # (key, start, end, vtype, val_start)
    for _ in range(n_kv):
        start = p
        n = struct.unpack_from('<Q', buf, p)[0]
        key = buf[p+8:p+8+n].decode('utf-8'); p += 8+n
        vt = struct.unpack_from('<I', buf, p)[0]; p += 4
        vs = p
        p = skip(p, vt)
        kvs.append({'key':key,'start':start,'end':p,'vtype':vt,'val_start':vs,'val_end':p})
    tis = []
    for _ in range(n_tensors):
        start = p
        n = struct.unpack_from('<Q', buf, p)[0]
        name = buf[p+8:p+8+n].decode('utf-8'); p += 8+n
        nd = struct.unpack_from('<I', buf, p)[0]; p += 4
        dims = [struct.unpack_from('<Q', buf, p+8*i)[0] for i in range(nd)]; p += 8*nd
        tt = struct.unpack_from('<I', buf, p)[0]; p += 4
        off = struct.unpack_from('<Q', buf, p)[0]; p += 8
        tis.append({'name':name,'start':start,'end':p,'dims':dims,'type':tt,'off':off})
    align = 32
    for k in kvs:
        if k['key'] == 'general.alignment':
            align = struct.unpack_from('<I', buf, k['val_start'])[0]
    data_off = p + ((-p) % align)
    with open(path,'rb') as f:
        header = f.read(data_off)
    return {'version':version,'n_tensors':n_tensors,'n_kv':n_kv,'align':align,
            'data_off':data_off,'header':header,'kvs':kvs,'tis':tis,
            'file_size':os.path.getsize(path)}

L = parse_header(LEAN)
A = parse_header(ABLIT)
print("lean  : nt=%d nkv=%d data_off=%d size=%d" % (L['n_tensors'],L['n_kv'],L['data_off'],L['file_size']))
print("ablit : nt=%d nkv=%d data_off=%d size=%d" % (A['n_tensors'],A['n_kv'],A['data_off'],A['file_size']))

mtp_tis = [t for t in L['tis'] if t['name'].startswith('blk.64.')]
mtp_names = set(t['name'] for t in mtp_tis)
assert len(mtp_tis) == 15, len(mtp_tis)
assert not (set(t['name'] for t in A['tis']) & mtp_names)
print("MTP tensors to add:", len(mtp_tis))

# region boundaries
mtp_rel_start = min(t['off'] for t in mtp_tis)   # relative to data section
# ablit regular data length
ablit_data_len = A['file_size'] - A['data_off']
lean_data_len  = L['file_size'] - L['data_off']
print("ablit data_len=%d  lean data_len=%d  lean mtp_rel_start=%d" % (ablit_data_len, lean_data_len, mtp_rel_start))
assert ablit_data_len == mtp_rel_start, (ablit_data_len, mtp_rel_start)

# sanity: lean MTP offsets ascending & contiguous end == lean_data_len
mo = sorted(t['off'] for t in mtp_tis)
assert mo[0] == mtp_rel_start
assert mo[-1] < lean_data_len
# max end
max_end = max(t['off'] for t in mtp_tis)  # last tensor offset; its data ends at region end because contiguous
print("lean MTP region: [%d, %d) len=%d" % (mtp_rel_start, lean_data_len, lean_data_len-mtp_rel_start))

# ---- Build new header ----
buf = bytearray()
buf += L['header'][:4]                       # magic
buf += L['header'][4:8]                      # version (3)
buf += struct.pack('<Q', A['n_tensors'] + len(mtp_tis))
buf += struct.pack('<Q', A['n_kv'] + 1)
for k in A['kvs']:
    raw = bytearray(A['header'][k['start']:k['end']])
    if k['key'] == 'qwen35.block_count':
        assert k['vtype'] == 4
        v = struct.unpack_from('<I', A['header'], k['val_start'])[0]
        assert v == 64, v
        struct.pack_into('<I', raw, len(raw)-4, 65)
    buf += raw
# append nextn_predict_layers KV from lean
nkv = [k for k in L['kvs'] if k['key'] == 'qwen35.nextn_predict_layers']
assert len(nkv) == 1
buf += L['header'][nkv[0]['start']:nkv[0]['end']]
# tensor infos: ablit all, then lean MTP
for t in A['tis']:
    buf += A['header'][t['start']:t['end']]
for t in mtp_tis:
    buf += L['header'][t['start']:t['end']]
# pad
pad = (-len(buf)) % A['align']
buf += b'\x00'*pad
new_data_off = len(buf)
print("new header bytes=%d new_data_off=%d" % (len(buf), new_data_off))

new_size = new_data_off + ablit_data_len + (lean_data_len - mtp_rel_start)
print("expected output size=%d" % new_size)
free = shutil.disk_usage(os.path.dirname(OUT)).free
print("disk free=%d" % free)
assert free > new_size + 1_000_000_000

if os.path.exists(OUT):
    os.remove(OUT)

CH = 64*1024*1024
with open(OUT, 'wb') as out:
    out.write(buf)
    # copy ablit data section
    with open(ABLIT, 'rb') as f:
        f.seek(A['data_off'])
        remaining = ablit_data_len
        while remaining:
            d = f.read(min(CH, remaining)); out.write(d); remaining -= len(d)
    # copy lean MTP blob
    with open(LEAN, 'rb') as f:
        f.seek(L['data_off'] + mtp_rel_start)
        remaining = lean_data_len - mtp_rel_start
        while remaining:
            d = f.read(min(CH, remaining)); out.write(d); remaining -= len(d)
    out.flush()
    os.fsync(out.fileno())

print("written:", OUT, os.path.getsize(OUT))
