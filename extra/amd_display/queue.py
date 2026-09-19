"""GPU command loops, cache writeback and register waits for the HDMI animation."""
from contextlib import contextmanager
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.runtime.ops_amd import AMDDevice, AMDComputeQueue, WAIT_REG_MEM_FUNCTION_EQ
from tinygrad.runtime.support.hcq2 import encode_submit, patch, unwrap_view

class DisplayQueue(AMDComputeQueue):
  q_rewrite = PatternMatcher([
    (UPat(Ops.INS, arg=("flush", dtypes.void)), lambda ctx: ctx.acquire_mem(gli=0)),
    (UPat(Ops.INS, arg=("wait_reg", dtypes.void), src=(UPat(name="reg"), UPat(name="value"), UPat(name="mask"))),
     lambda ctx, reg, value, mask: ctx.wait_reg_mem(reg=reg.val, value=value.val, mask=mask.val, op=WAIT_REG_MEM_FUNCTION_EQ)),
    (UPat(Ops.INS, arg=("write_reg", dtypes.void), src=(UPat(name="reg"), UPat(name="value"))),
     lambda ctx, reg, value: ctx.pkt3(ctx.pm4.PACKET3_WRITE_DATA, 0, reg.val, 0, value.val)),
    (UPat(Ops.END, src=(UPat(Ops.LINEAR, name="body"), UPat(Ops.RANGE, name="r"),
      UPat(Ops.CMPEQ, src=(UPat(Ops.LOAD, name="value"), UPat(Ops.CONST).or_casted("ref"))))),
     lambda ctx, body, r, value, ref: ctx.conditional_loop(body, r, value, ref)),
  ]) + AMDComputeQueue.q_rewrite

  def __init__(self, ctx, submit):
    super().__init__(ctx, submit)
    self.branches:list[tuple[int, int, int]] = [] # condition, continue target, loop exit

  def conditional_loop(self, body:UOp, r:UOp, value:UOp, ref:UOp):
    # The command processor compares memory at the end of each iteration. Loop state
    # lives in buffers; command operands cannot depend on the loop's index.
    index = value.src[0]
    if r in body.toposort() or r.src[0].op is not Ops.NOOP or value.dtype is not dtypes.uint32 \
        or index.op is not Ops.INDEX or index.src[1].op is not Ops.CONST:
      raise ValueError("display loops require a uint32 memory comparison and buffer-based iteration state")
    address = index.src[0].getaddr(self.devs) + index.src[1].val * 4
    start, checks = len(self.blob), []
    # An optional mid-loop check lets an odd frame count or cancellation stop after either buffer.
    check = UOp(Ops.INS, arg=("check_stop", dtypes.void))
    for u in (*body.src, check):
      if u is not check:
        self.q_rewrite.rewrite(u, ctx=self)
        continue
      self.acquire_mem(gli=0) # publish the advance kernel's counter/stop writes before the CP reads them
      checks.append(len(self.blob))
      # COND_INDIRECT_BUFFER: if/else (2), memory == reference (3), low 32 bits only.
      # Branches are chains, so repetition does not consume the indirect-buffer stack.
      self.pkt3(self.pm4.PACKET3_COND_INDIRECT_BUFFER, 2 | (3 << 8), address, UOp.const(0xffffffff, dtypes.uint64),
        ref.cast(dtypes.uint64), UOp.const(0, dtypes.uint64), 0, UOp.const(0, dtypes.uint64), 0)
    self.branches += [(c, start if c == checks[-1] else c + 14 * 4, len(self.blob)) for c in checks]

  def submit(self, cmdbuf:UOp) -> UOp:
    base, offset = unwrap_view(cmdbuf)
    address, size, branches = base.getaddr(self.devs) + offset, cmdbuf.max_numel(), []
    for condition, start, end in self.branches:
      # Continue with the next frame (wrapping at the end), or exit to the timeline signal.
      for field, target, length in ((32, start, end - start), (44, end, size - end)):
        if not 0 < length // 4 < (1 << 20): raise ValueError("display loop branch exceeds the AMD indirect-buffer size limit")
        branches += [(condition + field, address + target), (condition + field + 8, UOp.const(length // 4, dtypes.uint32))]
    # Patch the allocation directly so these addresses are resolved at link time, independently
    # of the command buffer's runtime timeline patches.
    return super().submit(cmdbuf.after(patch(base, [(offset + o, value) for o, value in branches])) if branches else cmdbuf)

def encode_display(ctx, submit): return encode_submit(DisplayQueue(ctx, submit))

@contextmanager
def display_queue(dev:AMDDevice):
  """Enable command loops and privileged display-register access on the dedicated GPU."""
  if not dev.is_am() or dev.is_vf or dev.is_aql or dev.xccs != 1: raise RuntimeError("display requires the single-XCC AMD PCI/USB PM4 driver")
  adev = dev.iface.dev_impl
  dev.synchronize()
  dev.compute_queue  # allocate the queue before changing its register-access privilege
  adev.gfx._grbm_select(me=1)
  previous_control = adev.regCP_HQD_PQ_CONTROL.read()
  adev.regCP_HQD_PQ_CONTROL.update(priv_state=1)
  adev.gfx._grbm_select()
  old_encode = dev.pm_encode
  dev.pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_amd_compute", name="submit"), encode_display),
  ]) + old_encode
  try: yield
  finally:
    try:
      dev.synchronize()
      adev.gfx._grbm_select(me=1)
      try: adev.regCP_HQD_PQ_CONTROL.write(previous_control)
      finally: adev.gfx._grbm_select()
    finally:
      dev.pm_encode = old_encode
