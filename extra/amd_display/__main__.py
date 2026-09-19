"""Render a shader to the RX 9060's HDMI port with one HCQ2 submission."""
import argparse, json, os, time
from pathlib import Path
from tinygrad import Device
from tinygrad.helpers import Context, fetch_fw
from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.ops_amd import AMDDevice
from extra.amd_display.bios import AtomBIOS
from extra.amd_display.dmub import DMUB, Firmware, FIRMWARE_SHA256
from extra.amd_display.hdmi import HDMI
from extra.amd_display.shaders import render_kernel

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", nargs="?", choices=["animate"], default="animate")
  parser.add_argument("--device", type=int, default=0, help="USB GPU index")
  parser.add_argument("--frames", type=int, default=600, help="positive even frame count at 60 Hz (default: 600, ten seconds)")
  parser.add_argument("--shader", default="fractal", help="module in shaders/ (default: fractal)")
  parser.add_argument("--firmware", type=Path, help="DCN401 DMCUB firmware file; otherwise download and verify it")
  parser.add_argument("--ftdi", default="ftdi://ftdi:230x/1", help="provisioned FTDI debug adapter URL")
  args = parser.parse_args()
  if args.frames < 2 or args.frames % 2: parser.error("--frames must be positive and even")
  firmware = Firmware(args.firmware.read_bytes() if args.firmware else bytes(fetch_fw("amdgpu", "dcn_4_0_1_dmcub.bin", FIRMWARE_SHA256)))
  from extra.usbgpu.debug import USBGPUDebug
  with USBGPUDebug(args.ftdi) as debug:
    if not debug.provisioned: raise RuntimeError("Display output requires an already provisioned FTDI reset adapter")
    try:
      with Context(DEV="USB+AMD", HCQ2=1):
        dev = Device[f"AMD:{args.device}"]
        assert isinstance(dev, AMDDevice)
        # AMDev.fini's clock reduction hangs after display use; FTDI resets the board instead.
        Device._opened_devices.discard(dev.device)
        dev.can_recover = False
        pci = dev.iface.pci_dev
        try:
          if pci.read_config(0, 4) != 0x75901002: raise RuntimeError("This experiment supports PCI 1002:7590 only")
          run_animation(dev, AtomBIOS.read(pci), firmware, args.frames, args.shader)
        finally:
          libusb.libusb_release_interface(pci.usb.usb.handle, 0)
          libusb.libusb_close(pci.usb.usb.handle)
          os.close(pci.lock_fd)
    finally: debug.reset(wait=True)

def run_animation(dev, bios, firmware, frames:int, shader:str):
  from tinygrad.uop.ops import Ops, UOp
  from tinygrad.engine.realize import run_linear
  from extra.amd_display.animation import animation, timing_stats
  from extra.amd_display.scanout import Scanout
  from extra.amd_display.present import presentation

  # Recovery is disabled for this experiment, so synchronize(timeout=...) is ignored.
  # Cover both the explicit wait and implicit waits in HCQ execution and display cleanup.
  dev.wait_timeout_ms = max(dev.wait_timeout_ms, frames * 1000 // 60 + 10000)
  dmub = DMUB(dev.iface.dev_impl, bios.data, firmware)
  try:
    hdmi = HDMI(dmub, bios.hdmi_connector())
    mode = hdmi.mode
    try:
      if not hdmi.reg(f"HPD{hdmi.connector.hpd_id - 1}_DC_HPD_INT_STATUS").read_bitfields()["dc_hpd_sense"]:
        raise RuntimeError("HDMI hotplug is not asserted; connect and power on the monitor")
      hdmi.enable()
      display = Scanout(dev, hdmi, bios)
      with presentation(display):
        print(f"Compiling and uploading {frames} frames ({frames / 60:g} seconds)...", flush=True)
        graph, counter, timestamps = animation(display, frames, shader)
        # Initialize the front buffer before enabling scanout. The animation starts on the other buffer.
        init = render_kernel(mode.width, mode.height, dev.device, shader).call(UOp.from_buffer(display.buffers[1]), UOp.from_buffer(counter))
        run_linear(UOp(Ops.LINEAR, src=(init,)))
        dev.synchronize()
        try:
          display.enable()
          print("Animation running: one submission, no host work per frame.", flush=True)
          start = time.monotonic()
          run_linear(graph, jit=True)
          dev.synchronize()
          elapsed = time.monotonic() - start
          completed, status = counter.host.view(fmt="I")[0], display.status()
          timing = timing_stats(bytes(timestamps.host[:]), dev.timestamp_divider)
          print(json.dumps({"shader": shader, "frames": completed, "seconds": round(elapsed, 4), "submissions": len(graph.src), "scanout": status,
            "timing": timing}), flush=True)
          if completed != frames: raise RuntimeError(f"Expected {frames} frames, GPU completed {completed}")
          if any(status.values()): raise RuntimeError("Display reported a scanout error")
        finally: display.close()
    finally: hdmi.close()
  finally: dmub.close()

if __name__ == "__main__": main()
