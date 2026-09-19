"""Compile a finite animation into one HCQ2 submission, including GPU-side presentation."""
import math, statistics, struct
from tinygrad.device import Buffer, BufferSpec
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, KernelInfo, graph_rewrite
from tinygrad.engine.realize import lower_and_compile, link_linear
from tinygrad.runtime.support.hcq2 import BatchCtx, _finalize_batch, pm_encode
from extra.amd_display.scanout import Scanout
from extra.amd_display.shaders import render_kernel

def advance_kernel(frames:int, device:str) -> UOp:
  """Archive the three timestamp slots, then advance the GPU's frame counter."""
  counter = UOp.param(0, dtypes.uint32, 1, device=device)
  slots = UOp.param(1, dtypes.uint64, 6, device=device)
  history = UOp.param(2, dtypes.uint64, frames * 6, device=device)
  frame = counter.index(0).load()
  copies = [history.index(frame * 6 + i).store(slots.index(i).load()) for i in range(6)]
  return counter.after(*copies).index(0).store(frame + 1).sink(arg=KernelInfo("hdmi_advance"))

def animation(display:Scanout, frames:int=600, shader:str="fractal") -> tuple[UOp, Buffer, Buffer]:
  """Return the linked graph, frame counter and GPU timestamps. The caller submits once."""
  if frames < 2 or frames % 2: raise ValueError("Use a positive, even number of frames")
  device, m = display.dev.device, display.mode
  counter = Buffer(device, 1, dtypes.uint32, options=BufferSpec(cpu_access=True), initial_value=bytes(4))
  # Three [signal, timestamp] slots per frame: render start, render end, presentation complete.
  timestamps = Buffer(device, frames * 6, dtypes.uint64, options=BufferSpec(cpu_access=True, uncached=True), initial_value=bytes(frames * 48))
  slots = Buffer(device, 6, dtypes.uint64, options=BufferSpec(cpu_access=True, uncached=True), initial_value=bytes(48))
  count = UOp.from_buffer(counter)
  advance = advance_kernel(frames, device).call(count, UOp.from_buffer(slots), UOp.from_buffer(timestamps))
  render = render_kernel(m.width, m.height, device, shader)
  pair = []
  for buf in display.buffers:
    fb = UOp.from_buffer(buf)
    pair += [render.call(fb, count), UOp.custom_function("present", UOp.const(0)).call(fb), advance]
  compiled = lower_and_compile(UOp(Ops.LINEAR, src=tuple(pair)))
  batch = _finalize_batch(BatchCtx([(c, (device,), "COMPUTE:0") for c in compiled.src], profile=False))

  # Keep the batch's prologue/epilogue once. Only the two-frame body is repeated; signals
  # and frame state advance on the GPU. DisplayQueue.loop encodes a hardware branch, not an unrolled buffer.
  submit = next(u for u in batch.body.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_amd_compute")
  commands = submit.src[0]
  start = next(i for i, u in enumerate(commands.src) if u.op is Ops.CALL)
  end = max(i for i, u in enumerate(commands.src) if u.op is Ops.CALL) + 1
  repeat = UOp.range(frames // 2, 0) if frames > 2 else UOp.const(0)
  calls, timed = commands.src[start:end], []
  assert len(calls) == 6 and all(call.op is Ops.CALL for call in calls)
  for i in range(2):
    stamps = [UOp(Ops.INS, arg=("timestamp", dtypes.void), src=(
      UOp.from_buffer(slots).getaddr(device) + point * 16,)) for point in range(3)]
    timed += [stamps[0], calls[i * 3], stamps[1], calls[i * 3 + 1], stamps[2], calls[i * 3 + 2]]
  body = UOp(Ops.LINEAR, src=tuple(timed))
  repeated = (body.end(repeat),) if frames > 2 else body.src
  commands = commands.replace(src=commands.src[:start] + repeated + commands.src[end:])
  batch = batch.substitute({submit: submit.replace(src=(commands,))}, enter_calls=True)
  encoded = graph_rewrite(UOp(Ops.LINEAR, src=(batch,)), pm_encode, walk=True)
  return link_linear(lower_and_compile(encoded), allow_cache=False), counter, timestamps

def timing_stats(data:bytes, ticks_per_us:float, refresh_hz:float=60.0) -> dict:
  """Summarize GPU time after completion; rendering excludes cache writeback and presentation waits."""
  samples = [(start, end, presented) for _, start, _, end, _, presented in struct.iter_unpack("<6Q", data)]
  if not samples or any(not 0 < start <= end <= presented for start, end, presented in samples):
    raise ValueError("Missing or unordered GPU timestamps")
  if any(a[2] > b[0] for a, b in zip(samples, samples[1:])): raise ValueError("GPU frames are out of order")
  scale, budget = ticks_per_us * 1000, 1000 / refresh_hz
  render = sorted((end - start) / scale for start, end, _ in samples)
  intervals = [(b[2] - a[2]) / scale for a, b in zip(samples, samples[1:])]
  return {"frame_budget_ms": round(budget, 4), "render_ms": {"mean": round(statistics.mean(render), 4),
    "p95": round(render[math.ceil(len(render) * 0.95) - 1], 4), "max": round(render[-1], 4)},
    "worst_render_budget_pct": round(render[-1] / budget * 100, 2), "worst_render_headroom_ms": round(budget - render[-1], 4),
    "render_over_budget_frames": sum(t > budget for t in render),
    "present_interval_ms": {"mean": round(statistics.mean(intervals), 4), "max": round(max(intervals), 4)} if intervals else {},
    "missed_refreshes": sum(max(0, round(t / budget) - 1) for t in intervals)}
