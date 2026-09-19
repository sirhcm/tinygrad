"""Port of the supplied repeating-octahedra raymarcher, with the default circular camera motion.
Palette and distance function: https://iquilezles.org/articles/palettes/
https://iquilezles.org/articles/distfunctions/
"""
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import UOp

def scene_distance(p:tuple[UOp, UOp, UOp], time:UOp) -> UOp:
  x, y, z = p[0], p[1], p[2] + time * 0.4
  x, y = x - x.floor() - 0.5, y - y.floor() - 0.5
  z = z - (z / 0.25).floor() * 0.25 - 0.125
  return (x.abs() + y.abs() + z.abs() - 0.15) * 0.57735027

def raymarch(frag_coord:tuple[UOp, UOp], resolution:tuple[int, int], time:UOp) -> tuple[UOp, UOp]:
  width, height = resolution
  uv = ((frag_coord[0] * 2.0 - width) / height, (frag_coord[1] * 2.0 - height) / height)
  norm = (uv[0].square() + uv[1].square() + 1.0).sqrt()
  rd = (uv[0] / norm, uv[1] / norm, 1.0 / norm)
  mouse = ((time * 0.2).cos(), (time * 0.2).sin())

  # Private state is initialized for each pixel, then carried through a compiled shader loop.
  scope = tuple(UOp.sink(*frag_coord).ranges)
  distance = UOp.placeholder((1,), dtypes.float32, addrspace=AddrSpace.REG)
  step = UOp.placeholder((1,), dtypes.int32, addrspace=AddrSpace.REG)
  distance = distance.after(distance.after(*scope).index(0).store(0.0))
  step = step.after(step.after(*scope).index(0).store(0))
  loop = UOp.loop(1)
  t, i = distance.after(loop).index(0).load(), step.after(loop).index(0).load()
  x, y, z = rd[0] * t, rd[1] * t, -3.0 + rd[2] * t
  angle = t * 0.15 * mouse[0]
  cs, sn = angle.cos(), angle.sin()
  # GLSL's p.xy *= rot2D(angle) is a row-vector multiply with a column-major mat2.
  x, y = x * cs - y * sn, x * sn + y * cs
  y = y + (t * (mouse[1] + 1.0) * 0.5).sin() * 0.35
  d = scene_distance((x, y, z), time)
  next_t = t + d
  continuing = (d >= 0.001) & (next_t <= 100.0)
  # A break preserves i; exhausting the for-loop leaves i == 80.
  next_i = continuing.where(i + 1, i)
  end = UOp.group(distance.index(0).store(next_t), step.index(0).store(next_i)).end(loop, continuing & (next_i < 80))
  return distance.after(end).index(0).load(), step.after(end).index(0).load()

def shade(frag_coord:tuple[UOp, UOp], resolution:tuple[int, int], time:UOp) -> list[UOp]:
  t, i = raymarch(frag_coord, resolution, time)
  phase = t * 0.04 + i.cast(dtypes.float32) * 0.005
  return [0.5 + 0.5 * ((phase + offset) * 6.28318).cos() for offset in (0.3, 0.416, 0.557)]
