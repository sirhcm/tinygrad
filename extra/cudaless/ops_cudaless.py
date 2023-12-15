# pylint: disable-all

from fcntl import ioctl
import mmap
import os
import ctypes, struct

# IOCTLS:
UVM_INITIALIZE = 0x30000001
UVM_REGISTER_GPU = 37
UVM_REGISTER_GPU_VASPACE = 25
UVM_CREATE_EXTERNAL_RANGE = 73
UVM_MAP_EXTERNAL_ALLOCATION = 33
UVM_REGISTER_CHANNEL = 27

# RM (resouce manager ioctls)
NV_ESC_RM_CONTROL = 0x2A
NV_ESC_RM_ALLOC = 0x2B
NV_ESC_RM_MAP_MEMORY = 0x4E
NV_ESC_RM_VID_HEAP_CONTROL = 0x4A
NV_IOCTL_MAGIC = 0x46 # 'F'

NV01_ROOT_CLIENT = 0x00000041
# NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES = 0x00000002

NV01_DEVICE_0 = 0x80
NV20_SUBDEVICE_0 = 0x2080
TURING_USERMODE_A = 0xc461
KEPLER_CHANNEL_GROUP_A = 0x0000A06C
FERMI_CONTEXT_SHARE_A = 0x00009067
FERMI_VASPACE_A = 0x000090f1
AMPERE_CHANNEL_GPFIFO_A = 0x0000C56F
AMPERE_COMPUTE_B = 0xC7C0

NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN = 0xc36f0108
NV2080_CTRL_CMD_GPU_GET_GID_INFO = 0x2080014a
NVA06C_CTRL_CMD_GPFIFO_SCHEDULE = 0xa06c0101
NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER = 0x20801230

def hexdump(bytes: bytearray) -> None:
  for i, byte in enumerate(bytes):
    if i % 0x10 == 0: print(f"\n{i:06X}\t", end="")
    print(f"{byte:02X}", end="" if i % 2 == 0 else " ")
  print()

# There has to be a better way to do this
def addressof(bytes: bytearray) -> int:
  return ctypes.addressof((ctypes.c_char * len(bytes)).from_buffer(bytes))

def ioc_wr(nr, size): return nr | NV_IOCTL_MAGIC << 8 | size << 16

class TcContext():
  mem_error = 0x7ffff7ffb000
  def alloc_object(self, hClass:int, root:int, parent:int, params:int=0) -> int:
    struct.pack_into("2I4xIP4x", p := bytearray(28), 0, root, parent, hClass, params)
    assert ioctl(self.fd_ctl, ioc_wr(NV_ESC_RM_ALLOC, 32), p) == 0 and struct.unpack_from("I", p, 0x18)[0] == 0, "alloc failed"
    return struct.unpack_from("I", p, 8)[0]
  
  def mmap_object(self, memory, length, target, flags):
    fd_dev0 = os.open("/dev/nvidia0", os.O_RDWR | os.O_CLOEXEC)
    struct.pack_into("3I8xQ12xIi", p := bytearray(52), 0, self.root, self.subdevice, memory, length, flags, fd_dev0)
    ret = ioctl(self.fd_ctl, ioc_wr(NV_ESC_RM_MAP_MEMORY, 56), p)
    assert ret == 0 and struct.unpack_from("I", p, 40)[0] == 0, "mmap failed"
    return mmap.mmap(fd_dev0, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE) # MAP_FIXED?
  
  def rm_control(self, cmd, client, object, params, paramsize):
    # NVOS54_PARAMETERS
    struct.pack_into("3I4xPI4x", p := bytearray(32), 0, client, object, cmd, params, paramsize)
    ret = ioctl(self.fd_ctl, ioc_wr(NV_ESC_RM_CONTROL, 32), p)
    assert ret == 0 and struct.unpack_from("I", p, 28)[0] == 0, "rm control"

  def heap_alloc(self, addr, length, flags, mmap_flags, type):
    # NVOS32_PARAMETERS
    struct.pack_into("3I26xI4x2I28xQ", p := bytearray(184), 0, self.root, self.device, 2, self.root, type, flags, length)
    ret = ioctl(self.fd_ctl, ioc_wr(NV_ESC_RM_VID_HEAP_CONTROL, 184), p)
    assert struct.unpack_from("I", p, 18)[0] == 0, "rm heap control"
    mem = struct.unpack_from("I", p, 44)[0]
    local_ptr = self.mmap_object(mem, length, addr, mmap_flags)
    # assert local_ptr == addr, "mmap heap"

    if type == 0:
      # UVM_CREATE_EXTERNAL_RANGE_PARAMS
      struct.pack_into("2Q4x", p := bytearray(20), 0, addr, length)
      ret = ioctl(self.fd_uvm, UVM_CREATE_EXTERNAL_RANGE, p)
      assert ret == 0 and struct.unpack_from("I", p, 16)[0] == 0, "uvm create external range"

      # UVM_MAP_EXTERNAL_ALLOCATION_PARAMS
      struct.pack_into("2Q8x16sI1132xQi2I4x", p := bytearray(1200), 0, addr, length, self.gpu_uuid, 1, 1, self.fd_ctl, self.root, mem)
      ret = ioctl(self.fd_uvm, UVM_MAP_EXTERNAL_ALLOCATION, p)
      assert ret == 0 and struct.unpack_from("I", p, 1196)[0] == 0, "uvm map external alloc"
    
    return mem

  def __init__(self):
    self.init_dev()
    self.init_uvm()
    self.init_mem()
    self.init_fifo()
    print("init done")
    
  def init_dev(self):
    self.fd_ctl = os.open("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    self.fd_uvm = os.open("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
    self.fd_dev0 = os.open("/dev/nvidia0", os.O_RDWR | os.O_CLOEXEC)

    self.root = self.alloc_object(NV01_ROOT_CLIENT, 0,  0)

    # NV_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES
    device_id = 0x0 # why? sometimes 0x1
    struct.pack_into("2I36xI", ap0080 := bytearray(48), 0, device_id, self.root, 0x2)
    
    self.device = self.alloc_object(NV01_DEVICE_0, self.root, self.root, addressof(ap0080))
    self.subdevice = self.alloc_object(NV20_SUBDEVICE_0, self.root, self.device)
    self.usermode = self.alloc_object(TURING_USERMODE_A, self.root, self.subdevice)

    self.gpu_mmio_ptr = self.mmap_object(self.usermode, 0x10000, None, 2)

    # NV_VASPACE_ALLOCATION_PARAMETERS
    struct.pack_into("4xI28xQ", vap := bytearray(48), 0, 0x48, 0x1000)
    self.vaspace = self.alloc_object(FERMI_VASPACE_A, self.root, self.device, addressof(vap))

    # NV2080_CTRL_GPU_GET_GID_INFO_PARAMS
    struct.pack_into("4x2I256x", p := bytearray(268), 0, 2, 16)
    self.rm_control(NV2080_CTRL_CMD_GPU_GET_GID_INFO, self.root, self.subdevice, addressof(p), 268)
    self.gpu_uuid = p[12:28]
  
  def init_uvm(self):
    p = bytearray(12)
    ret = ioctl(self.fd_uvm, UVM_INITIALIZE, p)
    assert ret == 0 and struct.unpack_from("I", p, 8)[0] == 0, "uvm init"

    # UVM_REGISTER_GPU_PARAMS: 40 instead of 37 bc of alignment
    struct.pack_into("16s5xi12x", p := bytearray(40), 0, self.gpu_uuid, -1)
    ret = ioctl(self.fd_uvm, UVM_REGISTER_GPU, p)
    assert ret == 0 and struct.unpack_from("I", p, 36)[0] == 0, "uvm register gpu"

    # UVM_REGISTER_GPU_VASPACE_PARAMS
    struct.pack_into("16si2I4x", p := bytearray(32), 0, self.gpu_uuid, self.fd_ctl, self.root, self.vaspace)
    ret = ioctl(self.fd_uvm, UVM_REGISTER_GPU_VASPACE, p)
    assert ret == 0 and struct.unpack_from("I", p, 28)[0] == 0, "register uvm vaspace"

  def init_mem(self):
    self.mem_handle = self.heap_alloc(0x200400000, 0x200000, 0x0001c101, 0xc0000, 0) # 0 = TYPE_IMAGE (nvos.h)
    self.mem_error_handle = self.heap_alloc(self.mem_error, 0x1000, 0xc001, 0, 13) # 13 = TYPE_NOTIFIER

  def init_fifo(self):
    # NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS
    struct.pack_into("12xIx", cgap := bytearray(17), 0, 0x00000001) # ENGINE_TYPE_GRAPHICS
    self.channel_group = self.alloc_object(KEPLER_CHANNEL_GROUP_A, self.root, self.device, addressof(cgap))

    # NV_CTXSHARE_ALLOCATION_PARAMETERS
    struct.pack_into("2I4x", cap := bytearray(12), 0, self.vaspace, 0x00000001) # SUBCONTEXT_ASYNC
    self.share = self.alloc_object(FERMI_CONTEXT_SHARE_A, self.root, self.channel_group, addressof(cap))

    # NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS
    struct.pack_into("2IQI4xI4xI28xQ72x", fifoap := bytearray(144), 0,
                     self.mem_error_handle, self.mem_handle,0x200400000, 0x400, self.share, self.mem_handle, 0x2000)
    self.gpfifo = self.alloc_object(AMPERE_CHANNEL_GPFIFO_A, self.root, self.channel_group, addressof(fifoap))
    self.compute = self.alloc_object(AMPERE_COMPUTE_B, self.root, self.gpfifo)

    # NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS
    struct.pack_into("I", sp := bytearray(4), 0, 0xFFFFFFFF)
    self.rm_control(NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, self.root, self.gpfifo, addressof(sp), 4)
    self.work_submit_token = struct.unpack("I", sp)[0]
    assert self.work_submit_token != -1, "failed to get work submit token"

    # UVM_REGISTER_CHANNEL_PARAMS
    struct.pack_into("16si2I2Q4x", p := bytearray(52), 0, self.gpu_uuid, self.fd_ctl, self.root, self.gpfifo, 0x203600000, 0xf6e000) # magic values?
    assert ioctl(self.fd_uvm, UVM_REGISTER_CHANNEL, p) == 0, "uvm register fifo"

    # NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS
    struct.pack_into("?x", sp := bytearray(2), 0, True)
    self.rm_control(NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, self.root, self.channel_group, addressof(sp), 2)

    p = bytearray(32)
    self.rm_control(NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER, self.root, self.subdevice, addressof(p), 32)
    imla0, fmla16, dp, fmla32, ffma, imla1, imla2, imla3, imla4 = struct.unpack_from("9B", p, 16)
    print(f"rate modifiers -- imla0:{imla0} fmla16:{fmla16} dp:{dp} fmla32:{fmla32} ffma:{ffma} imla1:{imla1} imla2:{imla2} imla3:{imla3} imla4:{imla4}")

if __name__ == "__main__":
  ctx = TcContext()
