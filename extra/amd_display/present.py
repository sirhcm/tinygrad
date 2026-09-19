"""DCN401 custom-function lowering for presentation on the GPU's PM4 compute queue."""
from contextlib import contextmanager
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.runtime.ops_amd import AMDComputeQueue, WAIT_REG_MEM_FUNCTION_EQ
from tinygrad.runtime.support.hcq2 import encode_submit, patch, unwrap_view
from extra.amd_display.scanout import Scanout

displays:dict[str, Scanout] = {}

class DisplayQueue(AMDComputeQueue):
  def __init__(self, ctx, submit):
    super().__init__(ctx, submit)
    self.loops:list[tuple[int, int]] = []

  def loop(self, body:UOp, r:UOp):
    # This backend keeps frame state in buffers. A RANGE value in command operands would
    # require GPU-side operand patching, which this small loop encoder does not implement.
    if r in body.toposort() or r.src[0].op is not Ops.CONST or r.src[0].val <= 0:
      raise ValueError("display ranges require a positive constant bound and buffer-based iteration state")
    count = UOp.placeholder((1,), dtypes.uint64, device=self.devs, volatile=True, tag="display_loop_count").getaddr(self.devs)
    self.pkt3(self.pm4.PACKET3_WRITE_DATA, self.pm4.WRITE_DATA_DST_SEL(5) | self.pm4.WR_CONFIRM, count, 0, 0)
    start = len(self.blob)
    for u in body.src: self.q_rewrite.rewrite(u, ctx=self)
    # GFX12 ATOMIC_MEM: ADD_RTN_64 (47), wait for write confirmation (2).
    self.pkt3(self.pm4.PACKET3_ATOMIC_MEM, 47 | (2 << 8), count, UOp.const(1, dtypes.uint64), UOp.const(0, dtypes.uint64), 0)
    condition = len(self.blob)
    # COND_INDIRECT_BUFFER: if/else (2), count < reference (1). Branches are chains,
    # so repeating does not consume the command processor's indirect-buffer stack.
    self.pkt3(self.pm4.PACKET3_COND_INDIRECT_BUFFER, 2 | (1 << 8), count, UOp.const(0xffffffffffffffff, dtypes.uint64),
      r.src[0].cast(dtypes.uint64), UOp.const(0, dtypes.uint64), 0, UOp.const(0, dtypes.uint64), 0)
    self.loops.append((start, condition))

  def submit(self, cmdbuf:UOp) -> UOp:
    base, offset = unwrap_view(cmdbuf)
    address, size, branches = base.getaddr(self.devs) + offset, cmdbuf.max_numel(), []
    for start, condition in self.loops:
      end = condition + 14 * 4
      # True repeats the body; false proceeds to the remaining commands, including the timeline signal.
      for field, target, length in ((32, start, end - start), (44, end, size - end)):
        if not 0 < length // 4 < (1 << 20): raise ValueError("display loop branch exceeds the AMD indirect-buffer size limit")
        branches += [(condition + field, address + target), (condition + field + 8, UOp.const(length // 4, dtypes.uint32))]
    # Patch the allocation directly so these addresses are resolved at link time, independently
    # of the command buffer's runtime timeline patches.
    return super().submit(cmdbuf.after(patch(base, [(offset + o, value) for o, value in branches])) if branches else cmdbuf)

  def present(self, call:UOp):
    display = displays[self.dev.device]
    if len(call.body.src) != 1 or call.body.src[0].op is not Ops.CONST or type(call.body.src[0].val) is not int or call.body.src[0].val != 0:
      raise ValueError("present requires constant display output 0")
    if call.dtype is not dtypes.void: raise ValueError("present does not return a value")
    args = get_call_arg_uops(call)
    if len(args) != 1 or args[0].base.op is not Ops.BUFFER or args[0].base.is_unbound \
        or not isinstance(buf:=args[0].buffer, Buffer) or buf not in display.buffers:
      raise ValueError("present requires one of this display's complete, allocated framebuffers")
    address = display.addresses[display.buffers.index(buf)]
    reg = display.reg

    # Re-arm a private fence on the GPU, then wait for rendering's writeback to VRAM.
    # A host-side timeline update or an old, already-signalled fence would be wrong inside a range.
    fence = UOp.placeholder((1,), dtypes.uint64, device=self.devs, volatile=True, tag="present_fence")
    addr = fence.getaddr(self.devs)
    self.pkt3(self.pm4.PACKET3_WRITE_DATA, self.pm4.WRITE_DATA_DST_SEL(5) | self.pm4.WR_CONFIRM, addr, 0)
    self.release_mem(addr, 1, self.pm4.data_sel__mec_release_mem__send_32_bit_low,
      self.pm4.int_sel__mec_release_mem__none, cache_flush=True)
    self.wait_reg_mem(mem=addr, value=1, op=WAIT_REG_MEM_FUNCTION_EQ)

    def wait(name:str, value:int, mask:int=0xffffffff):
      self.wait_reg_mem(reg=reg(name).addr[0], value=value, mask=mask, op=WAIT_REG_MEM_FUNCTION_EQ)
    def write(name:str, value:int): self.pkt3(self.pm4.PACKET3_WRITE_DATA, 0, reg(name).addr[0], 0, value)

    # Arm during active scanout and wait through the following blanking interval. This gives
    # each frame its own refresh even when rendering finishes before the previous vblank ends.
    wait("OTG0_OTG_STATUS", 0, reg("OTG0_OTG_STATUS").fields_mask("otg_v_blank"))
    write("HUBPREQ0_DCSURF_PRIMARY_SURFACE_ADDRESS_HIGH", address >> 32)
    write("HUBPREQ0_DCSURF_PRIMARY_SURFACE_ADDRESS", address & 0xffffffff)
    wait("OTG0_OTG_STATUS", 1, reg("OTG0_OTG_STATUS").fields_mask("otg_v_blank"))
    wait("HUBPREQ0_DCSURF_FLIP_CONTROL", 0, reg("HUBPREQ0_DCSURF_FLIP_CONTROL").fields_mask("surface_flip_pending"))
    wait("HUBPREQ0_DCSURF_SURFACE_EARLIEST_INUSE_HIGH", address >> 32, 0xffff)
    wait("HUBPREQ0_DCSURF_SURFACE_EARLIEST_INUSE", address & 0xffffffff)
    wait("OTG0_OTG_STATUS", 0, reg("OTG0_OTG_STATUS").fields_mask("otg_v_blank"))

def encode_display(ctx, submit): return encode_submit(DisplayQueue(ctx, submit))

@contextmanager
def presentation(display:Scanout):
  """Install the experimental backend only for the dedicated userspace-owned GPU."""
  dev, adev = display.dev, display.adev
  if dev.device in displays: raise RuntimeError("A presentation backend is already installed for this GPU")
  if not dev.is_am() or dev.is_vf or dev.is_aql or dev.xccs != 1: raise RuntimeError("present requires the single-XCC AMD PCI/USB PM4 driver")
  dev.synchronize()
  dev.compute_queue  # allocate the queue before changing its register-access privilege
  adev.gfx._grbm_select(me=1)
  previous_control = adev.regCP_HQD_PQ_CONTROL.read()
  adev.regCP_HQD_PQ_CONTROL.update(priv_state=1)
  adev.gfx._grbm_select()
  old_encode = dev.pm_encode
  displays[dev.device] = display
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
      displays.pop(dev.device)
