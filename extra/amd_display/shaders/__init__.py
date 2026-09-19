"""Shadertoy-style RGB shaders sharing a framebuffer, pixel coordinates and GPU frame clock."""
import math
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import fromimport
from tinygrad.uop.ops import Ops, UOp, KernelInfo, AxisType
from tinygrad.codegen.opt import Opt, OptOps

def render_kernel(width:int, height:int, device:str, shader:str="fractal") -> UOp:
  pixels = UOp.param(0, dtypes.uint32, width * height, device=device)
  time = UOp.param(1, dtypes.uint64, 1, device=device).index(0).load().cast(dtypes.float32) / 60.0
  pixel = UOp.range(width * height, 0)
  # Shadertoy uses pixel centers and a bottom-left origin; scanout rows start at the top.
  x, y = (pixel % width).cast(dtypes.float32) + 0.5, height - 0.5 - (pixel // width).cast(dtypes.float32)
  color = fromimport(f"extra.amd_display.shaders.{shader}", "shade")((x, y), (width, height), time)
  opts = None
  if any(u.op is Ops.RANGE and (u.dtype is dtypes.void or u.arg[-1] is AxisType.REDUCE) for u in UOp.sink(*color).toposort()):
    # Automatic pixel upcasting cannot handle independent loop exit conditions yet.
    # Use one pixel per thread, grouped into workgroups on GPU backends.
    # Explicit accumulation loops also retain their bounds instead of being unrolled.
    # Coordinate arithmetic splits the flat pixel range into rows, then columns.
    local = math.gcd(width, 64) if Device[device].renderer.has_local else 1
    opts = (Opt(OptOps.SPLIT, int(height > 1), (local, AxisType.LOCAL)),) if local > 1 else ()
  # XRGB8888 is opaque: shaders return RGB; alpha is unused by scanout.
  red, green, blue = [(channel.clip(0.0, 1.0) * 255.0 + 0.5).cast(dtypes.uint32) for channel in color]
  return pixels.index(pixel).store((red << 16) | (green << 8) | blue).end(pixel).sink(arg=KernelInfo(f"hdmi_{shader}", opts_to_apply=opts))
