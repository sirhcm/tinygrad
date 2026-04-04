from __future__ import annotations
import ctypes, functools, inspect, os, pathlib, re, struct, sys, sysconfig
from tinygrad.helpers import ceildiv, getenv, mv_address, to_mv, unwrap, DEBUG, OSX, WIN
from _ctypes import _SimpleCData
from typing import get_type_hints, get_args, get_origin, Annotated

def _do_ioctl(__idir, __base, __nr, __struct, __fd, *args, __payload=None, **kwargs):
  assert not WIN, "ioctl not supported"
  import tinygrad.runtime.support.hcq as hcq, fcntl
  ioctl = __fd.ioctl if isinstance(__fd, hcq.FileIOInterface) else functools.partial(fcntl.ioctl, __fd)
  if __struct is None: return ioctl((__base<<8)|__nr, __payload or (args[0] if args else 0))
  if (rc:=ioctl((__idir<<30)|(ctypes.sizeof(out:=(__payload or __struct(*args, **kwargs)))<<16)|(__base<<8)|__nr, out)):
    raise RuntimeError(f"ioctl returned {rc}")
  return out

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, typ): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, del_an(typ))
def _IOR(base, nr, typ): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, del_an(typ))
def _IOWR(base, nr, typ): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, del_an(typ))

class Enum:
  def __init_subclass__(cls): cls._val_to_name_ = {}

  @classmethod
  def get(cls, val, default="unknown"): return cls._val_to_name_.get(val, default)
  @classmethod
  def items(cls): return cls._val_to_name_.items()
  @classmethod
  def define(cls, name:str, val:int) -> int:
    cls._val_to_name_[val] = name
    return val

def del_an(ty):
  if isinstance(ty, type) and issubclass(ty, Enum): return del_an(ty.__orig_bases__[0]) # type: ignore
  return ty.__metadata__[0] if get_origin(ty) is Annotated else (None if ty is type(None) else ty)

_pending_records = []

# the backing type for "compound" C types, ie. struct, union, array
# if we were on 3.12, this would use __buffer__
class _CBuffer:
  SIZE: int
  def __init__(self, mv:memoryview=None): self.mv, self._objects_ = mv.cast("B") if mv is not None else memoryview(bytearray(self.SIZE)), {}
  def __bytes__(self): return self.mv.tobytes()
  # backing ctypes.Structure, should this be a ClassVar?
  @classmethod
  def _cstruct(cls): return type(cls.__name__ + "_cstruct", (ctypes.Structure,), {'_fields_': [('_mem_', ctypes.c_byte * cls.SIZE)]})
  @classmethod
  def from_param(cls, obj):
    assert isinstance(obj, _CBuffer) and not obj.mv.readonly, "expected a _CBuffer with a writable memoryview"
    return obj._cstruct().from_buffer(obj.mv)
  @classmethod
  def from_buffer(cls, buf):
    _CBuffer.__init__(inst:=cls.__new__(cls), memoryview(buf))
    return inst
  @classmethod
  def from_mv(cls, mv, off=0): return cls.from_buffer(mv[off:off+cls.SIZE])
  @classmethod
  def set_into(cls, tgt:_CBuffer, off:int, v:_CBuffer): tgt.mv[off:off+cls.SIZE], tgt._objects_[off] = v.mv, v._objects_

CData = _SimpleCData | type[_CBuffer]
def sizeof(obj:CData|type[CData]) -> int:
  return (ctypes.sizeof(obj) if issubclass(obj, _SimpleCData) else obj.SIZE) if inspect.isclass(obj) else sizeof(obj.__class__)

class CFUNCTYPE:
  def __class_getitem__(cls, key):
    assert del_an(key[0]) is None or not issubclass(del_an(key[0]), Struct), "CFUNCTYPE returning struct-by-value is unsupported"
    return ctypes.CFUNCTYPE(del_an(key[0]), *(del_an(a) for a in key[1]))

class Array(_CBuffer):
  typ: CData
  length: int
  def __init__(self, *vals):
    super().__init__()
    self[:] = vals
  def __setitem__(self, k:int|slice, v):
    if isinstance(k, slice):
      for i,v in zip(range(*k.indices(self.length)), v): self[i] = v
    elif issubclass(self.typ, (POINTER, _CBuffer)): v.set_into(self, sizeof(self.typ) * k, v)
    else: struct.pack_into(self.typ._type_, self.mv, sizeof(self.typ) * k, v)
  def __getitem__(self, k:int|slice):
    if isinstance(k, slice): return [self[i] for i in range(*k.indices(self.length))]
    elif issubclass(self.typ, (POINTER, _CBuffer)): return self.typ.from_mv(self.mv, sizeof(self.typ) * k)
    else: return struct.unpack_from(self.typ._type_, self.mv, sizeof(self.typ) * k)[0]
  @classmethod
  def from_param(cls, obj): return ctypes.c_void_p(mv_address(obj.mv))
  def __class_getitem__(cls, key):
    typ, length = del_an(key[0]), key[1] if isinstance(key[1], int) else get_args(key[1])[0]
    return type(f"Array_{typ.__name__}_{length}", (cls,), {"typ": typ, 'length': length, "SIZE": sizeof(typ) * length})

class POINTER(ctypes.c_void_p):
  def __class_getitem__(cls, key): return type(f"POINTER_{del_an(key).__name__}", (cls,), {"typ": del_an(key), "SIZE": sizeof(ctypes.c_void_p)})
  def __init__(self, addr=None, _obj_=None):
    super().__init__(addr)
    self._objects_ = {"self": _obj_}
  def __setitem__(self, k:int, v):
    (arr:=Array[self.typ, k].from_buffer(to_mv(mv_address(self.mv), self.SIZE * k))).__setitem__(k, v)
    self._objects_[k] = arr._objects_
  @property
  def contents(self): return self.typ.from_buffer(to_mv(self.value, sizeof(self.typ)))
  @classmethod
  def from_param(cls, obj): return obj
  @classmethod
  def from_mv(cls, mv, off): return cls(struct.unpack_from("P", mv, off)[0])
  @classmethod
  def set_into(cls, tgt:_CBuffer, off:int, v:POINTER|_CBuffer):
    struct.pack_into("P", tgt.mv, off, v.value if isinstance(v, POINTER) else mv_address(v.mv))
    tgt._objects_[off] = {**v._objects_, **({"self": v} if isinstance(v, _CBuffer) else {})}

def pointer(obj:CData) -> POINTER: return POINTER[obj.__class__](mv_address(obj.mv) if isinstance(obj, _CBuffer) else ctypes.addressof(obj), obj)

class Struct(_CBuffer):
  def __init__(self, *args, **kwargs):
    super().__init__()
    for f,v in [*zip((rf[0] for rf in self._fields_), args), *kwargs.items()]: setattr(self, f, v)

def record(cls:type[Struct]) -> type[Struct]:
  _pending_records.append((cls, unwrap(sys._getframe().f_back).f_globals))
  return cls

def init_records():
  for cls, ns in _pending_records:
    setattr(cls, '_fields_', [])
    for nm, t in get_type_hints(cls, globalns=ns, include_extras=True).items():
      if nm == "SIZE": continue
      if t.__origin__ in (bool, bytes, str, int, float): setattr(cls, nm, Field(*(f:=t.__metadata__)))
      else: setattr(cls, nm, Field(*(f:=(del_an(t.__origin__), *t.__metadata__))))
      cls._fields_.append((nm,) + f)
  _pending_records.clear()

def i2b(i:int, sz:int) -> bytes: return i.to_bytes(sz, sys.byteorder)
def b2i(b:bytes) -> int: return int.from_bytes(b, sys.byteorder)

class Field(property):
  def __init__(self, typ:CData, off:int, bit_width=None, bit_off=0):
    if issubclass(typ, (_CBuffer, POINTER)): super().__init__(lambda self: typ.from_mv(self.mv, off), lambda self, v: typ.set_into(self, off, v))
    else:
      if bit_width is not None:
        sl, set_mask = slice(off,off+(sz:=ceildiv(bit_width+bit_off, 8))), ~((mask:=(1 << bit_width) - 1) << bit_off)
        # FIXME: signedness
        super().__init__(lambda self: (b2i(self.mv[sl]) >> bit_off) & mask,
                         lambda self,v: self.mv.__setitem__(sl, i2b((b2i(self.mv[sl]) & set_mask) | (v << bit_off), sz)))
      else: super().__init__(lambda self: struct.unpack_from(typ._type_, self.mv, off)[0],
                             lambda self,v: struct.pack_into(typ._type_, self.mv, off, v))
    self.offset = off

@functools.cache
def init_c_struct_t(sz:int, fields: tuple[tuple, ...]):
  CStruct = type("CStruct", (Struct,), {'_fields_': [('_mem_', ctypes.c_byte * sz)], '_real_fields_': []})
  for nm,ty,*args in fields:
    setattr(CStruct, nm, Field(*(f:=(del_an(ty), *args))))
    CStruct._real_fields_.append((nm,) + f) # type: ignore
  return CStruct
def init_c_var(ty, creat_cb): return (creat_cb(v:=del_an(ty)()), v)[1]

class DLL(ctypes.CDLL):
  _loaded_: set[str] = set()

  @staticmethod
  def findlib(nm:str, paths:list[str], extra_paths=[]):
    if nm == 'libc' and OSX: return '/usr/lib/libc.dylib'
    if pathlib.Path(path:=getenv(nm.replace('-', '_').upper()+"_PATH", '')).is_file(): return path
    for p in paths:
      libpaths = {"posix": ["/usr/lib64", "/usr/lib", "/usr/local/lib"], "nt": os.environ['PATH'].split(os.pathsep),
                  "darwin": ["/opt/homebrew/lib", f"/System/Library/Frameworks/{p}.framework", f"/System/Library/PrivateFrameworks/{p}.framework"],
                  'linux': ['/lib', '/lib64', f"/lib/{sysconfig.get_config_var('MULTIARCH')}", "/usr/lib/wsl/lib/"]}
      if (pth:=pathlib.Path(p)).is_absolute():
        if pth.is_file(): return p
        else: continue
      for pre in (pathlib.Path(pre) for pre in ([path] if path else []) + libpaths.get(os.name, []) + libpaths.get(sys.platform, []) + extra_paths):
        if not pre.is_dir(): continue
        if WIN or OSX:
          for base in ([f"lib{p}.dylib", f"{p}.dylib", str(p)] if OSX else [f"{p}.dll"]):
            if (l:=pre / base).is_file() or (OSX and 'framework' in str(l) and l.is_symlink()): return str(l)
        else:
          for l in (l for l in pre.iterdir() if l.is_file() and re.fullmatch(f"lib{p}\\.so\\.?[0-9]*", l.name)):
            # filter out linker scripts
            with open(l, 'rb') as f:
              if f.read(4) == b'\x7FELF': return str(l)

  def __init__(self, nm:str, paths:str|list[str], extra_paths=[], emsg="", **kwargs):
    self.nm, self.emsg = nm, emsg
    if (path:= DLL.findlib(nm, paths if isinstance(paths, list) else [paths], extra_paths if isinstance(extra_paths, list) else [extra_paths])):
      if DEBUG >= 3: print(f"loading {nm} from {path}")
      try:
        super().__init__(path, **kwargs)
        self._loaded_.add(self.nm)
      except OSError as e:
        self.emsg = str(e)
        if DEBUG >= 3: print(f"loading {nm} failed: {e}")
    elif DEBUG >= 3: print(f"loading {nm} failed: not found on system")

  def bind(self, fn):
    restype, argtypes = del_an((hints:=get_type_hints(fn, include_extras=True)).pop('return', None)), tuple(del_an(h) for h in hints.values())
    cfunc = None
    def wrapper(*args):
      nonlocal cfunc
      if cfunc is None: (cfunc:=getattr(self, fn.__name__)).argtypes, cfunc.restype = argtypes, restype
      return cfunc(*args)
    def struct_by_val(*args):
      nonlocal cfunc
      if cfunc is None: (cfunc:=getattr(self, fn.__name__)).argtypes, cfunc.restype = argtypes, restype._cstruct()
      return restype.from_buffer(cfunc(*args))
    return struct_by_val if restype is not None and issubclass(restype, Struct) else wrapper

  def __getattr__(self, nm):
    if self.nm not in self._loaded_:
      raise AttributeError(f"failed to load library {self.nm}: " + (self.emsg or f"try setting {self.nm.upper()+'_PATH'}?"))
    return super().__getattr__(nm)
