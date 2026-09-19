"""ATOM tables for the board's HDMI transmitter, hotplug pin and display reference clock.

Layouts: Linux v6.16, drivers/gpu/drm/amd/include/atomfirmware.h.
"""
import dataclasses, struct
from tinygrad.runtime.support.system import USBPCIDevice

@dataclasses.dataclass(frozen=True)
class Connector:
  object_id:int
  encoder_id:int
  hpd_id:int

  @property
  def transmitter(self) -> int:
    # INTERNAL_UNIPHY, INTERNAL_UNIPHY1, INTERNAL_UNIPHY2; enum IDs select A/B within each pair.
    pair = {0x1e: 0, 0x20: 2, 0x21: 4}.get(self.encoder_id & 0xff)
    enum_id = (self.encoder_id >> 8) & 0xf
    if pair is None or enum_id not in (1, 2): raise ValueError(f"Unsupported encoder {self.encoder_id:#06x}")
    return pair + enum_id - 1

class AtomBIOS:
  @staticmethod
  def read(pci:USBPCIDevice):
    # Map the option ROM just beyond BAR5 within the bridge's non-prefetchable window.
    address = (pci.bar_info(5)[0] + pci.bar_info(5)[1] + 0x1fffff) & ~0x1fffff
    previous = pci.read_config(0x30, 4)
    try:
      pci.write_config_flush(0x30, address | 1, 4)
      return AtomBIOS(bytes(pci.usb.pcie_mem_read(address, 256 << 10)))
    finally: pci.write_config_flush(0x30, previous, 4)

  def __init__(self, data:bytes):
    self.data = data
    if self.slice(0, 2) != b"\x55\xaa": raise ValueError("Invalid PCI option ROM signature")
    rom = self.u16(0x48)
    if self.slice(rom + 4, 4) != b"ATOM": raise ValueError("Missing ATOM ROM header")
    self.master = self.u16(rom + 32)
    self.table_header(self.master)

  def slice(self, offset:int, size:int) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(self.data): raise ValueError(f"VBIOS range outside image: {offset:#x}+{size:#x}")
    return self.data[offset:offset + size]

  def u16(self, offset:int) -> int: return struct.unpack("<H", self.slice(offset, 2))[0]

  def table_header(self, offset:int) -> tuple[int, int, int]:
    size, major, minor = struct.unpack("<HBB", self.slice(offset, 4))
    if size < 4: raise ValueError(f"Invalid VBIOS table size at {offset:#x}")
    self.slice(offset, size)
    return size, major, minor

  def table(self, index:int) -> tuple[int, int, int, int]:
    master_size, major, minor = self.table_header(self.master)
    if (major, minor) != (2, 1): raise ValueError(f"Unsupported ATOM master table {major}.{minor}")
    if index < 0 or 4 + index * 2 + 2 > master_size: raise ValueError(f"Missing ATOM data table {index}")
    offset = self.u16(self.master + 4 + index * 2)
    if offset == 0: raise ValueError(f"Missing ATOM data table {index}")
    return (offset, *self.table_header(offset))

  def hdmi_connector(self) -> Connector:
    offset, size, major, minor = self.table(22)
    if (major, minor) not in ((1, 4), (1, 5)): raise ValueError(f"Unsupported display object table {major}.{minor}")
    count = self.slice(offset + 6, 1)[0]
    if 8 + count * 16 > size: raise ValueError("Display paths exceed their table")
    result = []
    for i in range(count):
      obj, record, encoder, _, _, _, _, _ = struct.unpack("<8H", self.slice(offset + 8 + i * 16, 16))
      if obj & 0xff != 0x0c: continue
      hpd = None
      if record < 8 + count * 16 or record >= size: raise ValueError("Connector records outside display object table")
      while record < size:
        kind, length = self.slice(offset + record, 2)
        if kind == 0xff: break
        if length < 2 or record + length > size: raise ValueError("Invalid connector record length")
        payload = self.slice(offset + record + 2, length - 2)
        if kind == 2:
          if len(payload) < 2: raise ValueError("Truncated hotplug pin record")
          hpd = payload[0]
        record += length
      else: raise ValueError("Unterminated connector records")
      if hpd not in range(1, 5): raise ValueError("Unsupported HDMI hotplug pin")
      result.append(Connector(obj, encoder, hpd))
    if len(result) != 1: raise ValueError(f"Expected one HDMI connector, found {len(result)}")
    return result[0]

  def display_refclock(self) -> int:
    offset, size, major, minor = self.table(27)
    if (major, minor) != (4, 5) or size < 14: raise ValueError("Expected ATOM display controller table 4.5")
    if not (clock:=self.u16(offset + 12) * 10000): raise ValueError("Missing display reference clock")
    return clock
