import sys
p = sys.argv[1]
with open(p, "rb") as f: b = f.read(300)
print("ASCII:", b.decode("ascii", "replace"))
print("HEX:", b[:32].hex())
