"""Single uncompressed XRGB8888 plane: HUBP0 -> DPP0 -> MPCC0 -> OPP0.

This deliberately supports only the experiment's 720p mode, without scaling, GPUVM or DCC.
"""
import time
from tinygrad.device import Buffer, BufferSpec
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.runtime.ops_amd import AMDDevice
from extra.amd_display.hdmi import HDMI
from extra.amd_display.dmub import wait_for
from extra.amd_display.bios import AtomBIOS

FRAMEBUFFER_OFFSET = 0x1000

class Scanout:
  def __init__(self, dev:AMDDevice, hdmi:HDMI, bios:AtomBIOS):
    self.dev, self.hdmi, self.reg, self.mode = dev, hdmi, hdmi.reg, hdmi.mode
    if (self.mode.width, self.mode.height) != (1280, 720): raise RuntimeError("Scanout currently requires 720p")
    self.adev = dev.iface.dev_impl
    self.refclock = bios.display_refclock()
    if self.refclock != 100_000_000: raise RuntimeError("Scanout deadlines currently require a 100 MHz reference clock")
    # cpu_access requests physically contiguous VRAM from the PCI allocator. The same allocation
    # has a compute virtual address and a physical memory-controller address for display.
    # One application allocation: stop word at byte 0, uint64 frame count at 8,
    # space for future input/state, then two page-aligned framebuffers.
    pixels = self.mode.width * self.mode.height
    self.memory = Buffer(dev.device, FRAMEBUFFER_OFFSET + pixels * 8, dtypes.uint8,
      options=BufferSpec(cpu_access=True, uncached=True, nolru=True), preallocate=True)
    self.memory.host[:FRAMEBUFFER_OFFSET] = bytes(FRAMEBUFFER_OFFSET)
    self.stop = self.memory.view(1, dtypes.uint32, 0).ensure_allocated()
    self.counter = self.memory.view(1, dtypes.uint64, 8).ensure_allocated()
    self.buffers = [self.memory.view(pixels, dtypes.uint32, FRAMEBUFFER_OFFSET + i * pixels * 4).ensure_allocated() for i in range(2)]
    if len(self.memory.meta.mapping.paddrs) != 1: raise RuntimeError("Scanout requires contiguous VRAM")
    self.addresses = [self.adev.paddr2mc(self.memory.meta.mapping.paddrs[0][0] + b.offset) for b in self.buffers]

  def enable(self):
    m, u = self.mode, self.hdmi.update
    u("DC_IP_REQUEST_CNTL", ip_request_en=1)
    u("DOMAIN0_PG_CONFIG", domain_power_gate=0)
    wait_for(lambda: self.reg("DOMAIN0_PG_STATUS").read_bitfields()["domain_pgfsm_pwr_status"], 0, description="HUBP power")
    u("DPPCLK0_DTO_PARAM", dppclk0_dto_phase=255, dppclk0_dto_modulo=255)
    u("DPPCLK_CTRL", dppclk0_en=1)
    u("HUBP0_HUBP_CLK_CNTL", hubp_clock_enable=1)
    u("DPP_TOP0_DPP_CONTROL", dpp_clock_enable=1)
    u("HUBP0_DCHUBP_CNTL", hubp_blank_en=1, hubp_vtg_sel=0, hubp_vready_at_or_after_vsync=0, hubp_ttu_disable=0)

    # Display's physical aperture is independent of the compute GPUVM mapping.
    for name, field, value in (("FB_LOCATION_BASE", "fb_base", self.adev.gmc.fb_base >> 24),
        ("FB_LOCATION_TOP", "fb_top", self.adev.gmc.fb_end >> 24), ("FB_OFFSET", "fb_offset", 0),
        ("AGP_BOT", "agp_bot", 0xffffff), ("AGP_TOP", "agp_top", 0), ("AGP_BASE", "agp_base", 0)):
      u("DCN_VM_" + name, **{field: value})
    u("HUBPREQ0_DCN_VM_SYSTEM_APERTURE_LOW_ADDR", mc_vm_system_aperture_low_addr=self.adev.gmc.fb_base >> 18)
    u("HUBPREQ0_DCN_VM_SYSTEM_APERTURE_HIGH_ADDR", mc_vm_system_aperture_high_addr=(self.adev.gmc.fb_end | 0xffffff) >> 18)
    u("HUBPREQ0_DCN_VM_MX_L1_TLB_CNTL", enable_l1_tlb=1, system_access_mode=3)
    u("HUBPREQ0_VMID_SETTINGS_0", vmid=0)

    u("DCHUBBUB_COMPBUF_CTRL", compbuf_size=0)
    u("DCHUBBUB_DET0_CTRL", det0_size=4)
    u("HUBPRET0_HUBPRET_CONTROL", det_buf_plane1_base_address=0, crossbar_src_alpha=0,
      crossbar_src_y_g=1, crossbar_src_cb_b=2, crossbar_src_cr_r=3)
    u("HUBP0_DCSURF_SURFACE_CONFIG", surface_pixel_format=8, rotation_angle=0, h_mirror_en=0)
    u("HUBP0_DCSURF_TILING_CONFIG", sw_mode=0, dim_type=1)
    u("HUBPREQ0_DCSURF_SURFACE_CONTROL", primary_surface_dcc_en=0, secondary_surface_dcc_en=0, primary_surface_tmz=0)
    u("HUBPREQ0_DCSURF_SURFACE_PITCH", pitch=m.width - 1)
    for which in ("PRI", "SEC"):
      u(f"HUBP0_DCSURF_{which}_VIEWPORT_START", **{which.lower() + "_viewport_x_start": 0, which.lower() + "_viewport_y_start": 0})
      u(f"HUBP0_DCSURF_{which}_VIEWPORT_DIMENSION", **{which.lower() + "_viewport_width": m.width,
        which.lower() + "_viewport_height": m.height})

    self.program_deadlines()
    # Nonzero update/ready windows allow HUBP to prepare the next frame's fetches.
    u("OTG0_OTG_VUPDATE_PARAM", vupdate_offset=100, vupdate_width=100)
    u("OTG0_OTG_VREADY_PARAM", vready_offset=100)
    u("CNVC_CFG0_CNVC_SURFACE_PIXEL_FORMAT", cnvc_surface_pixel_format=8)
    u("CNVC_CFG0_FORMAT_CONTROL", cnvc_bypass=0, format_expansion_mode=0, alpha_en=0,
      format_crossbar_r=0, format_crossbar_g=1, format_crossbar_b=2)
    u("CM0_CM_CONTROL", cm_bypass=1)
    u("DSCL0_DSCL_AUTOCAL", autocal_mode=0)
    u("DSCL0_RECOUT_START", recout_start_x=0, recout_start_y=0)
    u("DSCL0_RECOUT_SIZE", recout_width=m.width, recout_height=m.height)
    u("DSCL0_MPC_SIZE", mpc_width=m.width, mpc_height=m.height)
    u("DSCL0_SCL_MODE", dscl_mode=0)
    u("MPCC0_MPCC_TOP_SEL", mpcc_top_sel=0)
    u("MPCC0_MPCC_BOT_SEL", mpcc_bot_sel=15)
    u("MPCC0_MPCC_OPP_ID", mpcc_opp_id=0)
    u("MPCC0_MPCC_UPDATE_LOCK_SEL", mpcc_update_lock_sel=0)
    u("MPCC0_MPCC_CONTROL", mpcc_mode=1, mpcc_alpha_blnd_mode=2, mpcc_global_alpha=255, mpcc_global_gain=255)
    u("MPC_OUT0_MUX", mpc_out_mux=0)
    # Start on buffer 1; the animation first renders into buffer 0.
    u("HUBPREQ0_DCSURF_FLIP_CONTROL", surface_flip_type=0)
    u("HUBPREQ0_DCSURF_PRIMARY_SURFACE_ADDRESS_HIGH", primary_surface_address_high=self.addresses[1] >> 32)
    u("HUBPREQ0_DCSURF_PRIMARY_SURFACE_ADDRESS", primary_surface_address=self.addresses[1] & 0xffffffff)
    u("DPG0_DPG_CONTROL", dpg_en=0)
    u("HUBP0_DCHUBP_CNTL", hubp_blank_en=0, hubp_ttu_disable=0, hubp_underflow_clear=1)
    time.sleep(0.05)  # discard startup underflow, then leave detection enabled throughout the animation
    u("HUBP0_DCHUBP_CNTL", hubp_underflow_clear=0)

  def program_deadlines(self):
    m, u = self.mode, self.hdmi.update
    # This board's ATOM DCE table specifies 100 MHz, divided by two at HUBBUB.
    # Fixed-point formats follow DML2 DCN4; delivery covers active pixels, not horizontal blanking.
    ratio = (self.refclock / 2) / m.pixel_clock
    line = int(m.width * ratio)
    u("DCHUBBUB_GLOBAL_TIMER_CNTL", dchubbub_global_timer_enable=1, dchubbub_global_timer_refdiv=2)
    self.hdmi.write("HUBP0_HUBPREQ_DEBUG_DB", 1 << 8)  # DLG mission mode
    u("HUBP0_DCHUBP_REQ_SIZE_CONFIG", chunk_size=3, min_chunk_size=0, swath_height=0, pte_row_height_linear=0)
    u("HUBPREQ0_DCN_EXPANSION_MODE", drq_expansion_mode=2, prq_expansion_mode=1, mrq_expansion_mode=1, crq_expansion_mode=1)
    u("HUBPREQ0_BLANK_OFFSET_0", refcyc_h_blank_end=int((m.htotal - m.hfront - m.width) * ratio),
      dlg_v_blank_end=m.vtotal - m.vfront - m.height)
    u("HUBPREQ0_BLANK_OFFSET_1", min_dst_y_next_start=m.vtotal * 4)
    u("HUBPREQ0_DST_DIMENSIONS", refcyc_per_htotal=int(m.htotal * ratio * 256))
    u("HUBPREQ0_DST_AFTER_SCALER", refcyc_x_after_scaler=100, dst_y_after_scaler=0)
    u("HUBPREQ0_REF_FREQ_TO_PIX_FREQ", ref_freq_to_pix_freq=int(ratio * (1 << 19)))
    u("HUBPREQ0_PREFETCH_SETTINGS", vratio_prefetch=1 << 19, dst_y_prefetch=8 * 4)
    u("HUBPREQ0_PER_LINE_DELIVERY", refcyc_per_line_delivery_l=line)
    u("HUBPREQ0_PER_LINE_DELIVERY_PRE", refcyc_per_line_delivery_pre_l=line)
    u("HUBPREQ0_DCN_TTU_QOS_WM", qos_level_low_wm=0, qos_level_high_wm=0x3fff)
    u("HUBPREQ0_DCN_SURF0_TTU_CNTL0", refcyc_per_req_delivery=int(line * 1024 / (m.width * 4 / 256)),
      qos_level_fixed=15, qos_ramp_disable=1)
    u("HUBPREQ0_DCN_SURF0_TTU_CNTL1", refcyc_per_req_delivery_pre=int(line * 1024 / (m.width * 4 / 256)))
    u("HUBPREQ0_DCN_GLOBAL_TTU_CNTL", min_ttu_vblank=1000, qos_level_flip=15)

  def status(self) -> dict[str, int]:
    hubp = self.reg("HUBP0_DCHUBP_CNTL").read_bitfields()
    return {"underflow": hubp["hubp_underflow_status"], "timeout": hubp["hubp_timeout_status"],
      "segment_error": hubp["hubp_seg_alloc_err_status"], "vm_fault": self.reg("DCN_VM_FAULT_STATUS").read(),
      "fifo_error": self.reg(f"{self.hdmi.fe}_DIG_FIFO_CTRL0").read_bitfields()["dig_fifo_error"]}

  def flip_commands(self, index:int) -> tuple[UOp, ...]:
    """Latch a completed buffer at vblank, then wait until the previous buffer is free."""
    def wait(name:str, value:int, mask:int=0xffffffff):
      return UOp(Ops.INS, arg=("wait_reg", dtypes.void), src=tuple(UOp.const(v, dtypes.uint32) for v in (self.reg(name).addr[0], value, mask)))
    def write(name:str, value:int):
      return UOp(Ops.INS, arg=("write_reg", dtypes.void), src=tuple(UOp.const(v, dtypes.uint32) for v in (self.reg(name).addr[0], value)))
    address = self.addresses[index]
    blank = self.reg("OTG0_OTG_STATUS").fields_mask("otg_v_blank")
    return (UOp(Ops.INS, arg=("flush", dtypes.void)), wait("OTG0_OTG_STATUS", 0, blank),
      write("HUBPREQ0_DCSURF_PRIMARY_SURFACE_ADDRESS_HIGH", address >> 32),
      write("HUBPREQ0_DCSURF_PRIMARY_SURFACE_ADDRESS", address & 0xffffffff),
      wait("OTG0_OTG_STATUS", blank, blank),
      wait("HUBPREQ0_DCSURF_FLIP_CONTROL", 0, self.reg("HUBPREQ0_DCSURF_FLIP_CONTROL").fields_mask("surface_flip_pending")),
      wait("HUBPREQ0_DCSURF_SURFACE_EARLIEST_INUSE_HIGH", address >> 32, 0xffff),
      wait("HUBPREQ0_DCSURF_SURFACE_EARLIEST_INUSE", address & 0xffffffff), wait("OTG0_OTG_STATUS", 0, blank))

  def close(self):
    self.dev.synchronize()  # queued refresh waits need the timing generator running
    self.hdmi.update("HUBP0_DCHUBP_CNTL", hubp_blank_en=1)
    wait_for(lambda: self.reg("HUBP0_DCHUBP_CNTL").read_bitfields()["hubp_no_outstanding_req"], 1, description="scanout drain")
    for buf in (*self.buffers, self.counter, self.stop, self.memory):
      if buf.is_allocated(): buf.deallocate()
