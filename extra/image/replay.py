#!/usr/bin/env python3
import pathlib, pickle, sys, difflib, re
from tinygrad.helpers import ansistrip

def count_gated_reads(src):
  cnt = src.count("?read_image")
  for v in [m.group(1) for m in re.finditer(r'(val\d+)\s*=\s*read_imagef\(', src)]:
    if len(re.findall(fr'[\?\:]{v}\.[xyzw]', src)) > 0: cnt += 1
  return cnt

def diff(a, b): return ''.join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True)))

assert len(sys.argv) == 4, f"usage: {sys.argv[0]} SRCPKL1 SRCPKL2 ASTDIR"
PKL1 = pathlib.Path(sys.argv[1])
PKL2 = pathlib.Path(sys.argv[2])
ASTDIR = pathlib.Path(sys.argv[3])

with open(PKL1, "rb") as f: srcs1 = pickle.load(f)
with open(PKL2, "rb") as f: srcs2 = pickle.load(f)
for (nm1, ast1, src1), (nm2, ast2, src2) in zip(srcs1, srcs2):
  if "__kernel" not in src1: continue
  if (cnt1:=count_gated_reads(src1)) != (cnt2:=count_gated_reads(src2)):
    print(f"{nm1}: {cnt1} -> {cnt2}\n{diff(src1, src2)}")
    if ast1 != ast2:
      with open(ASTDIR / f"1-{ansistrip(nm1)}.py", "w") as f: f.write(ast1)
      with open(ASTDIR / f"2-{ansistrip(nm1)}.py", "w") as f: f.write(ast2)
    else:
      with open(ASTDIR / f"{ansistrip(nm1)}.py", "w") as f: f.write(ast1)
