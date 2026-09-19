"""DCN401 display microcontroller initialization and synchronous commands.

Firmware is authenticated by the GPU's PSP and loaded into RAM.
"""
import hashlib, struct, time
from tinygrad.helpers import round_up
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.am.amdev import AMDev, AMRegister
from extra.amd_display.autogen.dcn_4_1_0 import REGS

FIRMWARE_SHA256 = "99766ee9d5b452bf48f395afb354141a4185d16fba52e3d984aba5070907d01a"
RING_SIZE = 8192

def wait_for(fn, expected, timeout:float=2.0, description:str="display controller"):
  deadline = time.monotonic() + timeout
  while (value := fn()) != expected:
    if time.monotonic() >= deadline: raise TimeoutError(f"{description}: expected {expected!r}, got {value!r}")
    time.sleep(0.001)

class DCN:
  def __init__(self, adev:AMDev):
    if adev.ip_ver.get(am.DCE_HWIP) not in ((4, 1, 0), (4, 0, 1)): raise RuntimeError("Unsupported display IP")
    self.regs = {name: AMRegister(name=name, offset=off, segment=seg, fields=fields,
      bases=adev.regs_offset[am.DCE_HWIP], adev=adev) for name, (off, seg, fields) in REGS.items()}

  def __call__(self, name:str) -> AMRegister: return self.regs[name]

class Firmware:
  def __init__(self, blob:bytes):
    if hashlib.sha256(blob).hexdigest() != FIRMWARE_SHA256: raise ValueError("Unexpected DCN401 firmware SHA256")
    total, header, major, minor, ipmajor, ipminor, _, size, offset, _, inst, bss = struct.unpack_from("<IIHHHHIIIIII", blob)
    if (major, minor, ipmajor, ipminor) != (1, 0, 4, 0) or total != len(blob): raise ValueError("Unexpected firmware header")
    if header < 40 or inst < 512 or inst + bss > size or offset + size > len(blob): raise ValueError("Invalid firmware layout")
    self.signed = memoryview(blob)[offset:offset + inst]
    self.code = blob[offset + 256:offset + inst - 256]
    self.bss = blob[offset + inst:offset + inst + bss]
    meta_blob, candidates = (self.bss, (0,)) if bss else (self.code, range(16))
    for pad in candidates:
      meta = meta_blob[len(meta_blob) - 64 - pad:len(meta_blob) - pad if pad else len(meta_blob)]
      if len(meta) == 64 and meta[:4] == b"BUMD":
        _, self.state_size, self.trace_size, _, dal, self.shared_size = struct.unpack_from("<IIIIB3xI", meta)
        if not dal: raise ValueError("Expected DAL display firmware")
        break
    else: raise ValueError("Display firmware metadata missing")

class DMUB:
  def __init__(self, adev:AMDev, bios:bytes, firmware:Firmware):
    self.adev, self.reg, self.paddr = adev, DCN(adev), None
    self.wptr = 0
    if self.reg("DMCUB_CNTL").read_bitfields()["dmcub_enable"]:
      raise RuntimeError("Display firmware is already running; refusing to replace its memory")
    # DMCUB service layout: instructions, stack, BSS, VBIOS, mailboxes, trace, state, scratch, shared state.
    # PSP installs the executable windows in protected memory; the remaining windows use this VRAM allocation.
    sizes = [len(firmware.code), 640 << 10, len(firmware.bss), len(bios), 2 * RING_SIZE,
             firmware.trace_size, firmware.state_size, 1024, max(1280, firmware.shared_size)]
    offsets, end = [], 0
    for size in sizes:
      offsets.append(round_up(end, 256))
      end = offsets[-1] + round_up(size, 64)
    self.sizes = [round_up(size, 64) for size in sizes]
    self.size = round_up(end, 4096)
    self.paddr = adev.mm.palloc(self.size, align=4096, zero=True)
    self.offsets = offsets
    self.memory = adev.vram.view(self.paddr, self.size)
    try:
      self.memory[offsets[3]:offsets[3] + len(bios)] = bios
      if firmware.bss: self.memory[offsets[2]:offsets[2] + len(firmware.bss)] = firmware.bss
      adev.gmc.flush_hdp()
      adev.psp._load_ip_fw_cmd([am.GFX_FW_TYPE_DMUB], firmware.signed)
      for name in ("DMCUB_INBOX1_RPTR", "DMCUB_INBOX1_WPTR", "DMCUB_OUTBOX1_RPTR", "DMCUB_OUTBOX1_WPTR",
                   "DMCUB_OUTBOX0_RPTR", "DMCUB_OUTBOX0_WPTR", "DMCUB_SCRATCH0", "DMCUB_GPINT_DATAIN1"):
        self.reg(name).write(0)
      for window in (3, 4, 5, 6):
        base = 0x60000000 + (window << 24)
        self.window(f"DMCUB_REGION3_CW{window}", base, base + self.sizes[window], self.address(window))
      for window, index in ((5, 5), (6, 8)):
        prefix = f"DMCUB_REGION{window}"
        self.reg(prefix + "_OFFSET").write(self.address(index) & 0xffffffff)
        self.reg(prefix + "_OFFSET_HIGH").write(self.address(index) >> 32)
        self.reg(prefix + "_TOP_ADDRESS").write((1 << 31) | (self.sizes[index] - 1))
      for mailbox, base, size in (("INBOX1", 0x64000000, RING_SIZE), ("OUTBOX1", 0x64000000 + RING_SIZE, RING_SIZE),
                                  ("OUTBOX0", 0xa0000010, self.sizes[5] - 16)):
        self.reg(f"DMCUB_{mailbox}_BASE_ADDRESS").write(base)
        self.reg(f"DMCUB_{mailbox}_SIZE").write(size)
      self.reg("DMCUB_SCRATCH14").write(0)
      self.reg("MMHUBBUB_SOFT_RESET").update(dmuif_soft_reset=0)
      self.reg("DMCUB_SCRATCH15").write(0)
      self.reg("DMCUB_CNTL").update(dmcub_enable=1, dmcub_traceport_en=1)
      self.reg("DMCUB_CNTL2").update(dmcub_soft_reset=0)
      wait_for(lambda: self.reg("DMCUB_SCRATCH0").read() & 3, 3, description="DMUB boot")
    except BaseException:
      self.close()
      raise

  def address(self, window:int) -> int:
    if self.paddr is None: raise RuntimeError("Display firmware memory has been released")
    return self.adev.paddr2mc(self.paddr + self.offsets[window])

  def window(self, prefix:str, base:int, top:int, address:int):
    self.reg(prefix + "_OFFSET").write(address & 0xffffffff)
    self.reg(prefix + "_OFFSET_HIGH").write(address >> 32)
    self.reg(prefix + "_BASE_ADDRESS").write(base)
    self.reg(prefix + "_TOP_ADDRESS").write((1 << 31) | (top & 0x1fffffff))

  def command(self, kind:int, subtype:int, payload:bytes):
    if self.paddr is None: raise RuntimeError("Display firmware has been stopped")
    if len(payload) > 60: raise ValueError("DMUB command payload exceeds 60 bytes")
    packet = struct.pack("<BBBB", kind, subtype, 0, len(payload)) + payload.ljust(60, b"\x00")
    pos = self.offsets[4] + self.wptr
    self.memory[pos:pos + 64] = packet
    if bytes(self.memory[pos:pos + 64]) != packet: raise RuntimeError("DMUB command upload mismatch")
    self.wptr = (self.wptr + 64) % RING_SIZE
    self.reg("DMCUB_INBOX1_WPTR").write(self.wptr)
    wait_for(lambda: self.reg("DMCUB_INBOX1_RPTR").read(), self.wptr, description="DMUB command")

  def close(self):
    if self.paddr is None: return
    # Stop memory fetches before returning backing storage to AMDev's allocator.
    self.reg("DMCUB_CNTL2").update(dmcub_soft_reset=1)
    self.reg("DMCUB_CNTL").update(dmcub_enable=0)
    wait_for(lambda: self.reg("DMCUB_CNTL").read_bitfields()["dmcub_enable"], 0, description="DMUB stop")
    self.reg("MMHUBBUB_SOFT_RESET").update(dmuif_soft_reset=1)
    self.adev.mm.pfree(self.paddr)
    self.paddr = None
