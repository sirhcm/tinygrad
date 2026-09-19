"""Port of the supplied Shadertoy https://www.shadertoy.com/view/mtyGWy."""
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp

def shade(frag_coord:tuple[UOp, UOp], resolution:tuple[int, int], time:UOp) -> list[UOp]:
  x, y = frag_coord
  width, height = resolution
  uv = ((x * 2.0 - width) / height, (y * 2.0 - height) / height)
  radius = (uv[0].square() + uv[1].square()).sqrt()
  color = [UOp.const(0.0, dtypes.float32) for _ in range(3)]
  for i in range(4):
    uv = tuple((v * 1.5) - (v * 1.5).floor() - 0.5 for v in uv)
    d = (uv[0].square() + uv[1].square()).sqrt() * (-radius).exp()
    d = ((d * 8.0 + time).sin() / 8.0).abs().maximum(1e-6)  # avoid the singularity at sin(...) == 0
    intensity = (0.01 / d).pow(1.2)
    t = radius + i * 0.4 + time * 0.4
    palette = [0.5 + 0.5 * ((t + phase) * 6.28318).cos() for phase in (0.263, 0.416, 0.557)]
    color = [acc + col * intensity for acc, col in zip(color, palette)]
  return color
