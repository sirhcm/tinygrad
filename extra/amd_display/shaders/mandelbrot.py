"""A small, fixed-iteration Mandelbrot shader."""
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, UOp


def shade(frag_coord:tuple[UOp, UOp], resolution:tuple[int, int], time:UOp) -> list[UOp]:
  (px, py), (width, height) = frag_coord, resolution
  frame = (time * 60.0 + 0.5).cast(dtypes.uint64) % 5000
  px, py, time = px.cast(dtypes.float64), py.cast(dtypes.float64), frame.cast(dtypes.float64) / 60.0

  # Start with the full set in frame, then zoom toward a detailed boundary point.
  zoom = (-time * 0.18).exp()
  target_x, target_y = -0.7436438870371587, 0.1318259042053120
  x0 = target_x + ((((px - width * 0.5) / height) * 3.0 - 0.5) - target_x) * zoom
  y0 = target_y + ((((py - height * 0.5) / height) * 2.0) - target_y) * zoom

  # Keep the loop-carried values in private registers. All lanes run the same
  # fixed-size loop; escaped points retain their state through an active mask.
  scope = tuple(UOp.sink(*frag_coord).ranges)
  zr = UOp.placeholder((1,), dtypes.float64, addrspace=AddrSpace.REG)
  zi = UOp.placeholder((1,), dtypes.float64, addrspace=AddrSpace.REG)
  iterations = UOp.placeholder((1,), dtypes.int32, addrspace=AddrSpace.REG)
  active = UOp.placeholder((1,), dtypes.int32, addrspace=AddrSpace.REG)
  zr = zr.after(zr.after(*scope).index(0).store(0.0))
  zi = zi.after(zi.after(*scope).index(0).store(0.0))
  iterations = iterations.after(iterations.after(*scope).index(0).store(0))
  active = active.after(active.after(*scope).index(0).store(1))

  loop = UOp.range(512, 1, AxisType.REDUCE)
  zre = zr.after(loop).index(0).load()
  zim = zi.after(loop).index(0).load()
  count = iterations.after(loop).index(0).load()
  running = active.after(loop).index(0).load() > 0
  candidate_zre = zre.square() - zim.square() + x0
  candidate_zim = 2.0 * zre * zim + y0
  next_zre = running.where(candidate_zre, zre)
  next_zim = running.where(candidate_zim, zim)
  next_count = running.where(count + 1, count)
  next_active = (running & (candidate_zre.square() + candidate_zim.square() <= 4.0)).cast(dtypes.int32)
  end = UOp.group(
    zr.index(0).store(next_zre),
    zi.index(0).store(next_zim),
    iterations.index(0).store(next_count),
    active.index(0).store(next_active),
  ).end(loop)

  count = iterations.after(end).index(0).load().cast(dtypes.float32)
  escaped = 1.0 - active.after(end).index(0).load().cast(dtypes.float32)
  brightness = 0.45 + 0.55 * (count / 512.0).sqrt()
  phase = count * 0.075
  inside = 1.0 - escaped
  return [inside * base + escaped * brightness * (0.55 + 0.45 * (phase + offset).cos())
    for base, offset in zip((0.14, 0.18, 0.26), (0.0, 2.094, 4.189))]
