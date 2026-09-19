"""Single-pipe 720p60 RGB8 HDMI bring-up for DCN401."""
import dataclasses, struct
from extra.amd_display.bios import Connector
from extra.amd_display.dmub import DMUB, wait_for

@dataclasses.dataclass(frozen=True)
class Mode:
  width:int = 1280
  height:int = 720
  pixel_clock:int = 74250000
  htotal:int = 1650
  hfront:int = 110
  hsync:int = 40
  vtotal:int = 750
  vfront:int = 5
  vsync:int = 5

class HDMI:
  def __init__(self, dmub:DMUB, connector:Connector):
    self.dmub, self.reg, self.connector, self.mode = dmub, dmub.reg, connector, Mode()
    self.frontend = connector.transmitter
    self.fe = f"DIG{self.frontend}"
    self.programmed:set[str] = set()
    self.phy_enabled = False
    if any(self.reg(f"OTG{i}_OTG_CONTROL").read_bitfields()["otg_master_en"] for i in range(4)):
      raise RuntimeError("A display pipe is already active")

  def update(self, name:str, **fields):
    self.programmed.add(name)
    self.reg(name).update(**fields)

  def write(self, name:str, value:int):
    self.programmed.add(name)
    self.reg(name).write(value)

  def transmitter(self, action:int):
    c = self.connector
    self.dmub.command(128, 1, struct.pack("<BBBBIBBBB48x", c.transmitter, action, 3, 4, self.mode.pixel_clock // 10000,
                                        c.hpd_id, 1 << self.frontend, c.object_id & 0xff, 0))

  def enable(self):
    self.program_clock()
    self.program_timing()
    # Feed the output from DPG until Scanout.enable switches to the initialized framebuffer.
    m, u = self.mode, self.update
    u("OPP_PIPE0_OPP_PIPE_CONTROL", opp_pipe_clock_en=1)
    u("DPG0_DPG_DIMENSIONS", dpg_active_width=m.width, dpg_active_height=m.height)
    u("DPG0_DPG_OFFSET_SEGMENT", dpg_x_offset=0, dpg_segment_width=0)
    u("DPG0_DPG_CONTROL", dpg_en=1, dpg_mode=0, dpg_dynamic_range=0, dpg_bit_depth=1, dpg_vres=6, dpg_hres=6)
    u("FMT0_FMT_CONTROL", fmt_pixel_encoding=0)
    u("VTG0_CONTROL", vtg0_enable=1)
    u("OTG0_OTG_CONTROL", otg_disable_point_cntl=2, otg_master_en=1)
    self.enable_encoder()

  def program_clock(self):
    m, u = self.mode, self.update
    u("ODM0_OPTC_INPUT_CLOCK_CONTROL", optc_input_clk_en=1, optc_input_clk_gate_dis=1)
    u("OTG0_OTG_CLOCK_CONTROL", otg_clock_en=1, otg_clock_gate_dis=1)
    self.set_tmds_divider(1)
    u("OTG0_PIXEL_RATE_CNTL", dp_dto0_enable=0, pipe0_dto_src_sel=0)
    # Each connector uses its own combo-PHY PLL. Force programming while the transmitter is disabled;
    # selecting PLL0 for PHY C lets transmitter enable switch OTG to an unprogrammed clock.
    self.dmub.command(128, 2, struct.pack("<IBBBBBB2xI", m.pixel_clock // 100, 20 + self.connector.transmitter,
                                        self.connector.encoder_id & 0xff, 3, 1, 0, 0, 0))

  def set_tmds_divider(self, value:int):
    if self.reg("OTG_PIXEL_RATE_DIV").read_bitfields()["otg0_tmds_pixel_rate_div"] == value: return
    self.update("OTG_PIXEL_RATE_DIV", otg0_tmds_pixel_rate_div=value)
    # DCN401's divider latch needs a dentist transaction; omitting it can cause DIG FIFO errors.
    self.reg("DENTIST_DISPCLK_CNTL").write(self.reg("DENTIST_DISPCLK_CNTL").read())
    wait_for(lambda: self.reg("DENTIST_DISPCLK_CNTL").read_bitfields()["dentist_dispclk_chg_done"], 1,
             description="pixel divider latch")

  def program_timing(self):
    m, u = self.mode, self.update
    # DCN's counters start at the sync pulse, so blanking ends before active pixels begin.
    u("OTG0_OTG_H_TOTAL", otg_h_total=m.htotal - 1)
    u("OTG0_OTG_H_SYNC_A", otg_h_sync_a_start=0, otg_h_sync_a_end=m.hsync)
    u("OTG0_OTG_H_SYNC_A_CNTL", otg_h_sync_a_pol=0)
    u("OTG0_OTG_H_BLANK_START_END", otg_h_blank_start=m.htotal - m.hfront, otg_h_blank_end=m.htotal - m.hfront - m.width)
    for suffix in ("TOTAL", "TOTAL_MIN", "TOTAL_MAX"):
      u(f"OTG0_OTG_V_{suffix}", **{f"otg_v_{suffix.lower()}": m.vtotal - 1})
    u("OTG0_OTG_V_SYNC_A", otg_v_sync_a_start=0, otg_v_sync_a_end=m.vsync)
    u("OTG0_OTG_V_SYNC_A_CNTL", otg_v_sync_a_pol=0)
    u("OTG0_OTG_V_BLANK_START_END", otg_v_blank_start=m.vtotal - m.vfront, otg_v_blank_end=m.vtotal - m.vfront - m.height)
    u("VTG0_CONTROL", vtg0_enable=0, vtg0_vcount_init=m.vtotal - m.vfront, vtg0_fp2=0)
    u("OTG0_OTG_CONTROL", otg_start_point_cntl=0, otg_field_number_cntl=0, otg_out_mux=0)
    u("OTG0_OTG_VSTARTUP_PARAM", vstartup_start=m.vtotal - m.height - 2)
    u("OTG0_OTG_VUPDATE_PARAM", vupdate_offset=0, vupdate_width=0)
    u("OTG0_OTG_VREADY_PARAM", vready_offset=0)
    u("ODM0_OPTC_DATA_SOURCE_SELECT", optc_num_of_input_segment=0, optc_seg0_src_sel=0,
      optc_seg1_src_sel=15, optc_seg2_src_sel=15, optc_seg3_src_sel=15)
    u("ODM0_OPTC_MEMORY_CONFIG", optc_mem_sel=0)
    u("OTG0_OTG_H_TIMING_CNTL", otg_h_timing_div_mode=0, otg_h_timing_div_mode_manual=1)

  def enable_encoder(self):
    m, u = self.mode, self.update
    be = f"DIG{self.connector.transmitter}"
    u(f"PHY_MUX{self.connector.transmitter}_PHY_MUX_CONTROL", enc_type_sel=0)  # Legacy TMDS, not FRL/HPO.
    u(be + "_DIG_BE_CLK_CNTL", dig_be_mode=3, dig_be_clk_en=1)
    u(be + "_DIG_BE_EN_CNTL", dig_be_enable=1)
    u(be + "_DIG_BE_CNTL", dig_fe_source_select=1 << self.frontend)
    u(f"{self.fe}_DIG_FE_CNTL", dig_source_select=0)
    u(f"{self.fe}_STREAM_MAPPER_CONTROL", dig_stream_link_target=self.connector.transmitter)
    u(f"{self.fe}_DIG_FE_CLK_CNTL", dig_fe_mode=3, dig_fe_clk_en=1)
    u(f"{self.fe}_DIG_FE_EN_CNTL", dig_fe_enable=1)
    symbol = f"symclk{'abcd'[self.frontend]}"
    u(symbol.upper() + "_CLOCK_ENABLE", **{symbol + "_fe_en": 1, symbol + "_fe_src_sel": self.connector.transmitter})
    self.dmub.command(128, 0, struct.pack("<BBBBIBB2x", self.frontend, 0xf, 3, 4, m.pixel_clock // 10000, 2, 0))
    u(f"{self.fe}_DIG_CLOCK_PATTERN", dig_clock_pattern=0x1f)
    u(f"{self.fe}_HDMI_CONTROL", tmds_pixel_encoding=0, tmds_color_format=0, hdmi_packet_gen_version=1, hdmi_keepout_mode=1,
      hdmi_deep_color_enable=0, hdmi_deep_color_depth=0, hdmi_data_scramble_en=0, hdmi_no_extra_null_packet_filled=1,
      hdmi_clock_channel_rate=0)
    u(f"{self.fe}_HDMI_VBI_PACKET_CONTROL", hdmi_gc_cont=1, hdmi_gc_send=1, hdmi_null_send=1, hdmi_acp_send=0)
    u(f"{self.fe}_HDMI_INFOFRAME_CONTROL0", hdmi_audio_info_send=0)
    u(f"{self.fe}_HDMI_GC", hdmi_gc_avmute=0)
    u(be + "_TMDS_CTL_BITS", tmds_ctl0=1)
    self.phy_enabled = True
    self.transmitter(1)
    u(f"{self.fe}_DIG_FIFO_CTRL0", dig_fifo_output_pixel_per_cycle=0, dig_fifo_read_start_level=7, dig_fifo_reset=1)
    wait_for(lambda: self.reg(f"{self.fe}_DIG_FIFO_CTRL0").read_bitfields()["dig_fifo_reset_done"], 1, description="encoder FIFO reset")
    u(f"{self.fe}_DIG_FIFO_CTRL0", dig_fifo_reset=0, dig_fifo_error=3)  # Clear sticky underrun/overrun bits.
    wait_for(lambda: self.reg(f"{self.fe}_DIG_FIFO_CTRL0").read_bitfields()["dig_fifo_reset_done"], 0, description="encoder FIFO release")
    u(f"{self.fe}_DIG_FIFO_CTRL0", dig_fifo_enable=1)

  def close(self):
    # Stop the consumer before removing its clock: disabling the PHY first strands OTG mid-frame
    # and can stall later SMU display-clock commands.
    if f"{self.fe}_DIG_FIFO_CTRL0" in self.programmed: self.reg(f"{self.fe}_DIG_FIFO_CTRL0").update(dig_fifo_enable=0)
    if "OTG0_OTG_CONTROL" in self.programmed:
      self.reg("OTG0_OTG_CONTROL").update(otg_master_en=0)
      wait_for(lambda: self.reg("OTG0_OTG_CONTROL").read_bitfields()["otg_current_master_en_state"], 0,
               description="timing generator stop")
    if self.phy_enabled:
      self.transmitter(0)
      self.phy_enabled = False
    # Keep display clocks and power configured; restoring idle defaults can strand the memory fabric.
    self.programmed.clear()
