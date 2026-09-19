"""Render a shader to the RX 9060's HDMI port with one HCQ2 submission."""
import argparse, json, signal, threading, time
from pathlib import Path
from tinygrad import Device
from tinygrad.helpers import Context, fetch_fw
from tinygrad.runtime.ops_amd import AMDDevice
from extra.amd_display.bios import AtomBIOS
from extra.amd_display.dmub import DMUB, Firmware, FIRMWARE_SHA256
from extra.amd_display.hdmi import HDMI
from extra.amd_display.shaders import render_kernel

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", nargs="?", choices=["animate"], default="animate")
  parser.add_argument("--device", type=int, default=0, help="USB GPU index")
  parser.add_argument("--frames", type=int, default=600, help="frame count at 60 Hz; 0 runs until Ctrl-C (default: 600)")
  parser.add_argument("--shader", default="fractal", help="module in shaders/ (default: fractal)")
  parser.add_argument("--firmware", type=Path, help="DCN401 DMCUB firmware file; otherwise download and verify it")
  args = parser.parse_args()
  if args.frames < 0: parser.error("--frames must be nonnegative")
  firmware = Firmware(args.firmware.read_bytes() if args.firmware else bytes(fetch_fw("amdgpu", "dcn_4_0_1_dmcub.bin", FIRMWARE_SHA256)))
  with Context(DEV="USB+AMD", HCQ2=1):
    dev = Device[f"AMD:{args.device}"]
    assert isinstance(dev, AMDDevice)
    # Finalize explicitly with clocks high. Keep USB open for buffers released during process exit.
    Device._opened_devices.discard(dev.device)
    dev.can_recover = False
    pci = dev.iface.pci_dev
    try:
      if pci.read_config(0, 4) != 0x75901002: raise RuntimeError("This experiment supports PCI 1002:7590 only")
      run_animation(dev, AtomBIOS.read(pci), firmware, args.frames, args.shader)
    finally:
      dev.synchronize()
      dev.iface.dev_impl.fini(lower_clocks=False)

def run_animation(dev, bios, firmware, frames:int, shader:str):
  from tinygrad.uop.ops import Ops, UOp
  from tinygrad.engine.realize import run_linear
  from extra.amd_display.animation import animation
  from extra.amd_display.scanout import Scanout
  from extra.amd_display.queue import display_queue

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
      with display_queue(dev):
        try:
          duration = f"{frames} frames ({frames / 60:g} seconds)" if frames else "an indefinite animation"
          print(f"Compiling and uploading {duration}...", flush=True)
          graph = animation(display, frames, shader)
          # Initialize the front buffer; the animation starts by rendering into the other one.
          init = render_kernel(mode.width, mode.height, dev.device, shader).call(
            UOp.from_buffer(display.buffers[1]), UOp.from_buffer(display.counter))
          run_linear(UOp(Ops.LINEAR, src=(init,)))
          dev.synchronize()
          display.enable()
          elapsed, interrupted = run_graph(display, graph, frames)
          completed, status = display.counter.host.view(fmt="Q")[0], display.status()
          print(json.dumps({"shader": shader, "frames": completed, "seconds": round(elapsed, 4),
            "submissions": len(graph.src), "scanout": status}), flush=True)
          if frames and not interrupted and completed != frames: raise RuntimeError(f"Expected {frames} frames, GPU completed {completed}")
          if any(status.values()): raise RuntimeError("Display reported a scanout error")
        finally: display.close()
    finally: hdmi.close()
  finally: dmub.close()

def run_graph(display, graph, frames:int) -> tuple[float, bool]:
  from tinygrad.engine.realize import run_linear

  dev, stopped = display.dev, threading.Event()
  # A handler only wakes the owner. It must not interrupt the submitter or re-enter a USB transfer.
  previous = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
  start = time.monotonic()
  try:
    try:
      # DEBUG's automatic timing wait would block forever before we could process a stop request.
      with Context(DEBUG=0, PROFILE=0): run_linear(graph, jit=True)
      timeline = dev.timeline.host.view(fmt="Q")
      target = timeline[1]
      print("Animation running: one submission, GPU frame loop. Ctrl-C stops it.", flush=True)
      # This only watches completion/cancellation; it never submits or advances frames.
      while not stopped.wait(0.25):
        if timeline[0] >= target: break
        if frames and time.monotonic() - start > dev.wait_timeout_ms / 1000:
          raise RuntimeError("Animation exceeded its completion timeout")
    finally:
      display.stop.host.view(fmt="I")[0] = 1
      display.adev.gmc.flush_hdp()
      dev.synchronize() # the loop exits through the normal HCQ2 timeline signal
  finally:
    for sig, handler in previous.items(): signal.signal(sig, handler)
  return time.monotonic() - start, stopped.is_set()

if __name__ == "__main__": main()
