# pylint: disable-all

from fcntl import ioctl
# import mmap
import os, time
import ctypes, struct
from tinygrad.helpers import getenv

libc = ctypes.CDLL(None)
mmap64 = libc.mmap64
mmap64.restype = ctypes.c_void_p
mmap64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]

MAP_SHARED = 0x1
MAP_FIXED = 0x10

PROT_READ = 0x1
PROT_WRITE = 0x2

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

# PUSHBUFS
NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A = 0x02a0
NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A = 0x02e4

NVC6C0_OFFSET_OUT_UPPER = 0x0188
NVC6C0_LINE_LENGTH_IN = 0x0180
NVC6C0_LAUNCH_DMA = 0x01b0
NVC6C0_LOAD_INLINE_DATA = 0x01b4

NVC6C0_SET_INLINE_QMD_ADDRESS_A = 0x0318

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
    return mmap64(target, length, PROT_READ | PROT_WRITE, MAP_SHARED | (MAP_FIXED if target else 0), fd_dev0, 0)
  
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
    mem = struct.unpack_from("I", p, 44)[0] # data.AllocSize.hMemory
    local_ptr = self.mmap_object(mem, length, addr, mmap_flags)
    assert local_ptr == addr, "mmap heap"

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
    self.gpu_mmio_ptr = ctypes.cast(self.gpu_mmio_ptr, ctypes.POINTER(ctypes.c_uint32 * (0x10000 // 4)))

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
    print(f"memhandle: {self.mem_handle:x}")
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

def clear_gpu_control(): ctypes.memset(0x200400000, 0, 0x200600000-0x200400000)
  
trivial = bytearray.fromhex("00005a00ff057624 000fe200078e00ff"  # IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x168]
                            "0000580000027a02 000fe20000000f00"  # MOV R2, c[0x0][0x160]
                            "0000590000037a02 000fca0000000f00"  # MOV R3, c[0x0][0x164]
                            "0000000502007986 000fe2000c101904"  # STG.E [R2.64], R5
                            "000000000000794d 000fea0003800000") # EXIT

class PushBuf():
  def __init__(self):
    self.gpu_base = 0x200500000
    self.cmdq = self.gpu_base + 0x6000
    self.cur = self.cmdq

  def NVC0_FIFO_PKHDR_SQ(self, subc, mthd, size): return 0x20000000 | (size << 16) | (subc << 13) | (mthd >> 2)
  def NVC0_FIFO_PKHDR_1I(self, subc, mthd, size): return 0xa0000000 | (size << 16) | (subc << 13) | (mthd >> 2)
  def BEGIN_NVC0(self, subc, mthd, size): self.PUSH_DATA(self.NVC0_FIFO_PKHDR_SQ(subc, mthd, size))
  def BEGIN_NIC0(self, subc, mthd, size): self.PUSH_DATA(self.NVC0_FIFO_PKHDR_1I(subc, mthd, size))

  def PUSH_DATA(self, data):
    ctypes.cast(ctypes.c_void_p(self.cur), ctypes.POINTER(ctypes.c_uint32)).contents = ctypes.c_uint32(data)
    self.cur += 1
  def PUSH_DATAh(self, data):
    ctypes.cast(ctypes.c_void_p(self.cur), ctypes.POINTER(ctypes.c_uint32)).contents = ctypes.c_uint32(data >> 32)
    self.cur += 1
  def PUSH_DATAl(self, data):
    ctypes.cast(ctypes.c_void_p(self.cur), ctypes.POINTER(ctypes.c_uint32)).contents = ctypes.c_uint32(data >> 0)
    self.cur += 1
  def PUSH_DATAhl(self, data):
    self.PUSH_DATAh(data)
    self.PUSH_DATAl(data)

  def gpu_setup(self):
    self.BEGIN_NVC0(1, NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, 2)
    self.PUSH_DATAhl(0x00007FFFF4000000)

    self.BEGIN_NVC0(1, NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, 2)
    self.PUSH_DATAhl(0x004B0000)
  
  def gpu_memcpy(self, dest, data, length):
    assert length % 4 == 0

    self.BEGIN_NVC0(1, NVC6C0_OFFSET_OUT_UPPER, 2)
    self.PUSH_DATAhl(dest)

    self.BEGIN_NVC0(1, NVC6C0_LINE_LENGTH_IN, 2)
    self.PUSH_DATA(length)
    self.PUSH_DATA(1) # NVC6C0_LINE_COUNT
    self.BEGIN_NVC0(1, NVC6C0_LAUNCH_DMA, 1)
    
    self.PUSH_DATA(0x41)

    words = length // 4
    self.BEGIN_NIC0(1, NVC6C0_LOAD_INLINE_DATA, words)
    for i in range(words): self.PUSH_DATA(data[i])

  def gpu_compute(self, qmd, program_address, constant_address, constant_length):
    self.BEGIN_NVC0(1, NVC6C0_SET_INLINE_QMD_ADDRESS_A, 2)
    self.PUSH_DATAhl(qmd >> 8)

    # TODO: can't actually hardcode this...
    data = bytearray.fromhex("00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
                             "7F 00 00 00 00 00 00 3C 00 00 00 00 00 00 00 00"
                             "10 50 00 02 00 00 00 00 00 00 00 00 00 00 01 44"
                             "01 00 00 00 01 00 00 00 01 00 00 00 00 00 00 00")
    self.BEGIN_NVC0(1, 0x0320, 0x40) # NVC6C0_LOAD_INLINE_QMD_DATA(0)
    for i in range(0x40): self.PUSH_DATA(data[i])

def kick(doorbell, cb_index):
  print(f"doorbell=0x{doorbell:x}")
  db = ctypes.cast(doorbell, ctypes.POINTER(ctypes.c_uint32))
  print(db)
  db[0] = ctypes.c_uint32(cb_index)

if __name__ == "__main__":
  ctx = TcContext()
  print("**************** INIT DONE ****************")
  clear_gpu_control()

  push = PushBuf()

  push.gpu_setup()
  
  push.gpu_memcpy(push.gpu_base + 4, b"\xaa\xbb\xcc\xdd", 4)

  program = trivial

  # why is this 0x10? and not 0x0b?
  struct.pack_into("QI", args := bytearray(0x10), 0, push.gpu_base, 0x1337) # addr, value

  # load program and args
  # NOTE: normal memcpy also works here
  push.gpu_memcpy(push.gpu_base+0x1000, program, min(len(program), 0x180))
  push.gpu_memcpy(push.gpu_base+0x2160, args, 0x10)
  print(f"memcpyed program into gpu memory @ 0x{push.gpu_base:x}")
  
  push.gpu_compute(push.gpu_base + 0x4000, push.gpu_base + 0x1000, push.gpu_base + 0x2000, 0x160 + len(args))

  # skipping dma_copy for now...

  size = push.cur - push.cmdq
  # *((uint64_t*)0x200400000) = cmdq | (sz << 40) | 0x20000000000;
  # *((uint64_t*)0x20040208c) = 1;
  ctypes.cast(ctypes.c_void_p(0x200400000), ctypes.POINTER(ctypes.c_uint64))[0] = ctypes.c_uint64(push.cmdq | (size << 40) | 0x20000000000)
  ctypes.cast(ctypes.c_void_p(0x20040208c), ctypes.POINTER(ctypes.c_uint64))[0] = ctypes.c_uint64(1)

  # print(f"loc: 0x{ctx.gpu_mmio_ptr+0x90:x}")
  # print(f"work token: 0x{ctx.work_submit_token:x}")
  # print(f"0x{ctx.gpu_mmio_ptr:x}:")
  # test = bytearray(ctypes.cast(ctx.gpu_mmio_ptr, ctypes.POINTER(ctypes.c_byte * 0x1000)).contents)
  # print(len(test))
  # hexdump(test)
  print(type(ctx.gpu_mmio_ptr[0][0x90 // 4]))
  # mmio = bytearray(ctx.gpu_mmio_ptr[0])
  # print(mmio)
  # hexdump(mmio)
  ctx.gpu_mmio_ptr[0][0x90 // 4] = ctypes.c_uint32(ctx.work_submit_token)
  # kick(ctx.gpu_mmio_ptr + 0x90, ctx.work_submit_token)

  done = ctypes.cast(ctypes.c_void_p(0x200402088), ctypes.POINTER(ctypes.c_uint32))
  print(f"ran to queue {done.contents}")
  cnt = 0
  while not done.contents and cnt < 1000:
    time.sleep(0.001)
    cnt += 1
  time.sleep(0.01)
  print(f"ran to queue {done.contents}")
