"""Shader by Danilo Guanabara.
Original: http://www.pouet.net/prod.php?which=57245
"""
from tinygrad.uop.ops import UOp

def shade(frag_coord:tuple[UOp, UOp], resolution:tuple[int, int], time:UOp) -> list[UOp]:
  width, height = resolution
  uv = (frag_coord[0] / width, frag_coord[1] / height)
  p = ((uv[0] - 0.5) * (width / height), uv[1] - 0.5)
  radius = (p[0].square() + p[1].square()).sqrt()
  direction = tuple(v / radius.maximum(1e-6) for v in p)
  color, z = [], time
  for _ in range(3):
    z = z + 0.07
    displacement = (z.sin() + 1.0) * (radius * 9.0 - z - z).sin().abs()
    warped = tuple(v + d * displacement for v, d in zip(uv, direction))
    # GLSL mod(x, 1) is x - floor(x), including for negative coordinates.
    cell = tuple(v - v.floor() - 0.5 for v in warped)
    distance = (cell[0].square() + cell[1].square()).sqrt()
    color.append(0.01 / distance.maximum(1e-6) / radius.maximum(1e-6))
  return color
