"""Compile a double-buffered animation with one HCQ2 submission."""
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, KernelInfo, graph_rewrite
from tinygrad.engine.realize import lower_and_compile, link_linear
from tinygrad.runtime.support.hcq2 import BatchCtx, _finalize_batch, pm_encode
from extra.amd_display.scanout import Scanout
from extra.amd_display.shaders import render_kernel

def advance_kernel(frames:int, device:str) -> UOp:
  """Advance the shader's frame counter and optionally stop at a finite limit."""
  counter = UOp.param(0, dtypes.uint64, 1, device=device)
  stop = UOp.param(1, dtypes.uint32, 1, device=device)
  frame = counter.index(0).load()
  advance = counter.index(0).store(frame + 1)
  # Only write 1 at the limit. Writing 0 on other frames could erase a host stop request.
  if frames: advance = stop.after(advance).index(UOp.const(0).valid(frame.eq(frames - 1))).store(1)
  return advance.sink(arg=KernelInfo("hdmi_advance"))

def animation(display:Scanout, frames:int=600, shader:str="fractal") -> UOp:
  """Return one linked submission. frames=0 runs until display.stop is set to 1."""
  if frames < 0: raise ValueError("Frame count must be nonnegative")
  device, m = display.dev.device, display.mode
  count, stop = (UOp.from_buffer(b) for b in (display.counter, display.stop))
  advance = advance_kernel(frames, device).call(count, stop)
  render = render_kernel(m.width, m.height, device, shader)
  pair = tuple(c for buf in display.buffers for c in (render.call(UOp.from_buffer(buf), count), advance))
  compiled = lower_and_compile(UOp(Ops.LINEAR, src=pair))
  batch = _finalize_batch(BatchCtx([(c, (device,), "COMPUTE:0") for c in compiled.src], profile=False))

  # Keep the batch's prologue/epilogue once. Only the two-frame body is repeated; signals
  # and frame state advance on the GPU. The conditional loop encodes a hardware branch.
  submit = next(u for u in batch.body.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_amd_compute")
  commands = submit.src[0]
  start = next(i for i, u in enumerate(commands.src) if u.op is Ops.CALL)
  end = max(i for i, u in enumerate(commands.src) if u.op is Ops.CALL) + 1
  calls = commands.src[start:end]
  assert len(calls) == 4 and all(call.op is Ops.CALL for call in calls)
  body = UOp(Ops.LINEAR, src=(calls[0], *display.flip_commands(0), calls[1],
    UOp(Ops.INS, arg=("check_stop", dtypes.void)), calls[2], *display.flip_commands(1), calls[3]))
  running = UOp(Ops.CMPEQ, src=(stop.index(0).load(), UOp.const(0, dtypes.uint32)))
  commands = commands.replace(src=commands.src[:start] + (body.end(UOp.loop(0), running),) + commands.src[end:])
  batch = batch.substitute({submit: submit.replace(src=(commands,))}, enter_calls=True)
  encoded = graph_rewrite(UOp(Ops.LINEAR, src=(batch,)), pm_encode, walk=True)
  return link_linear(lower_and_compile(encoded), allow_cache=False)
