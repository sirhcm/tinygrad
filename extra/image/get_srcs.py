#!/usr/bin/env python3
import pathlib, pickle, sys
from tinygrad.viz import serve as viz
from tinygrad.uop.ops import RewriteTrace
from tinygrad.codegen import get_program
from tinygrad.renderer.cstyle import QCOMRenderer
from tinygrad.helpers import ansistrip
from extra.viz.cli import optional_eq, print_data

assert len(sys.argv) == 3, f"usage: {sys.argv[0]} PKL OUTPKL"
PKL1 = pathlib.Path(sys.argv[1])
PKL2 = pathlib.Path(sys.argv[2])

def get_srcs(pkl):
  srcs = []
  viz.trace = viz.load_pickle(pkl, default=RewriteTrace([], [], {}))
  viz.ctxs = viz.get_rewrites(viz.trace)

  for k in viz.ctxs:
    ast, src = None, None
    for s in k["steps"]:
      if s['name'] == "View Base AST": ast = next(viz.get_render(s['query'])['value'])['uop']
      if s['name'] == "View Source": src = viz.get_render(s['query'])["src"]
    if ast and src: srcs.append((k['name'], ast, src))
  return srcs

with open(PKL2, "wb") as out: pickle.dump(srcs:=get_srcs(PKL1), out)
print(f"Extracted {len(srcs)} kernels")
