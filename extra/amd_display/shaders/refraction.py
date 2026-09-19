"""Port of the supplied turbulent-cylinder raymarcher.
The original leaves z, i and O uninitialized; this port starts depth and color at zero.
"""
import math
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import UOp, AxisType

def atan2(y:UOp, x:UOp) -> UOp:
  """GLSL atan(y, x), keeping the atan argument within [0, 1] before restoring quadrants."""
  ax, ay = x.abs(), y.abs()
  angle = (ax.minimum(ay) / ax.maximum(ay).maximum(1e-20)).atan()
  angle = (ay > ax).where(math.pi / 2 - angle, angle)
  angle = (x < 0).where(math.pi - angle, angle)
  return (y < 0).where(-angle, angle)

def shade(frag_coord:tuple[UOp, UOp], resolution:tuple[int, int], time:UOp) -> list[UOp]:
  width, height = resolution
  # iResolution.xyx gives a z component of -width, not -height.
  ray = (frag_coord[0] * 2.0 - width, frag_coord[1] * 2.0 - height, UOp.const(-float(width), dtypes.float32))
  norm = sum(v.square() for v in ray).sqrt()
  ray = tuple(v / norm for v in ray)
  scope = tuple(UOp.sink(*frag_coord).ranges)
  depth = UOp.placeholder((1,), dtypes.float32, addrspace=AddrSpace.REG)
  color = UOp.placeholder((3,), dtypes.float32, addrspace=AddrSpace.REG)
  depth = depth.after(depth.after(*scope).index(0).store(0.0))
  color = color.after(*[color.after(*scope).index(c).store(0.0) for c in range(3)])

  march = UOp.range(20, 1, AxisType.REDUCE)
  i, z = (march + 1).cast(dtypes.float32), depth.after(march).index(0).load()
  x, y, pz = (z * v + 0.1 for v in ray)
  polar = (atan2(y / 0.2, x) * 2.0, pz / 3.0, (x.square() + y.square()).sqrt() - 5.0 - z * 0.2)
  point = UOp.placeholder((3,), dtypes.float32, addrspace=AddrSpace.REG)
  point = point.after(*[point.index(c).store(v) for c, v in enumerate(polar)])

  turbulence = UOp.range(7, 2, AxisType.REDUCE)
  d = (turbulence + 1).cast(dtypes.float32)  # GLSL's post-increment tests make the body use d = 1..7
  p = [point.after(turbulence).index(c).load() for c in range(3)]
  warped = [p[c] + (p[(c + 1) % 3] * d + time + 0.3 * i).sin() / d for c in range(3)]
  update = UOp.group(*[point.index(c).store(v) for c, v in enumerate(warped)]).end(turbulence)
  p = [point.after(update).index(c).load() for c in range(3)]
  distance = (sum((0.4 * v.cos() - 0.4).square() for v in p) + p[2].square()).sqrt().maximum(1e-6)
  next_z = z + distance
  rgb = [color.after(march).index(c).load() + (1.0 + (p[0] + i * 0.4 + next_z + phase).cos()) / distance
    for c, phase in enumerate((6.0, 1.0, 2.0))]
  end = UOp.group(depth.index(0).store(next_z), *[color.index(c).store(v) for c, v in enumerate(rgb)]).end(march)
  return [(color.after(end).index(c).load().square() / 400.0).tanh() for c in range(3)]
