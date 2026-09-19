from tinygrad.uop.ops import UOp

def shade(frag_coord:tuple[UOp, UOp], resolution:tuple[int, int], time:UOp) -> list[UOp]:
  (x, y), (width, height) = frag_coord, resolution
  r = (time  * 60) % height
  return [((x - width/2).square() + (y - height/2).square()) < r.square()] * 3
