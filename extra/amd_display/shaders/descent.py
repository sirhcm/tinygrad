"""tinygrad showing off: every pixel is a starting point for gradient descent on an animated loss landscape.
The gradient is computed symbolically by tinygrad's own autodiff (UOp.gradient), then compiled into the shader.
Pixels are colored by the basin of attraction they converge to; valley depth shades each basin.
"""
from tinygrad.uop.ops import UOp

def loss(x:UOp, y:UOp, time:UOp) -> UOp:
  # An animated egg-carton loss surface with several local minima in view.
  return (x + 0.3 * time).sin() * (y - 0.2 * time).cos() + 0.5 * (1.7 * x - 1.3 * y + 0.7 * time).sin()

def shade(frag_coord:tuple[UOp, UOp], resolution:tuple[int, int], time:UOp) -> list[UOp]:
  (x, y), (width, height) = frag_coord, resolution
  px, py = (x * 2.0 - width) / height * 2.0, (y * 2.0 - height) / height * 2.0
  # Gradient descent, unrolled. The learning rate breathes slowly.
  lr = 0.25 + 0.1 * (time).sin()
  for _ in range(24):
    gx, gy = loss(px, py, time).gradient(px, py)
    px, py = px - lr * gx, py - lr * gy
  # Color encodes where descent landed (which basin), with topographic bands of the starting height.
  bands = 0.85 + 0.15 * (loss((x * 2.0 - width) / height * 2.0, (y * 2.0 - height) / height * 2.0, time) * 6.0).cos()
  depth = (1.5 - loss(px, py, time)) * 0.3  # deeper minimum -> brighter
  color = [0.5 + 0.5 * c for c in ((px * 1.7).cos(), (py * 1.9 + 2.094).cos(), ((px + py) * 1.3 + 4.189).cos())]
  return [c * depth.clip(0.2, 1.0) * bands for c in color]
