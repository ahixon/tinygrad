from __future__ import annotations

"""
Experimental PTX-ASIC backend that assembles PTX with the in-repo assembler
and runs programs on the Verilator RTL model using cocotb.

This integrates with Tinygrad's Compiled flow using the regular PTXRenderer.
Limitations: only a subset of PTX is supported by the in-tree assembler/RTL.
"""

from dataclasses import dataclass
from typing import Any, Sequence, Iterable
from pathlib import Path
import functools

from tinygrad.device import Compiled, BufferSpec, Allocator, Compiler, CompilerPairT
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import getenv


class PTXASICAllocator(Allocator['PTXASICDevice']):
  def _alloc(self, size:int, options:BufferSpec):
    # Host-resident bytearray backing store
    return bytearray(size)
  def _free(self, opaque, options:BufferSpec):
    # Python GC handles bytearray
    pass
  def _copyin(self, dest, src:memoryview):
    assert isinstance(dest, (bytearray, bytes)), "unexpected backing store"
    dest[:len(src)] = src
  def _copyout(self, dest:memoryview, src):
    assert isinstance(src, (bytearray, bytes)), "unexpected backing store"
    dest[:] = memoryview(src)[:len(dest)]

class PTXASMCompiler(Compiler):
  """No-op compiler that forwards PTX text as bytes.

  The PTX will be assembled by the RTL runner using the in-repo assembler.
  """
  def __init__(self, arch:str="sm_50"):
    super().__init__(cachekey=f"PTXASIC_{arch}")
    self.arch = arch
  def compile(self, src:str) -> bytes:
    # Pass PTX through without modification; the in-tree assembler handles supported ops.
    return src.encode('utf-8')

@dataclass
class PTXASICProgram:
  dev: 'PTXASICDevice'
  name: str
  lib: bytes  # PTX text

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    # Determine element count from input buffer size when available, else from local_size.
    # Compiled Tinygrad typically orders buffers as [output, input, ...].
    input_buf = bufs[1] if len(bufs) >= 2 else (bufs[0] if len(bufs) >= 1 else None)
    input_bytes = len(input_buf) if isinstance(input_buf, (bytearray, bytes)) else 0
    n = input_bytes // 4 if input_bytes >= 4 else max(1, int(local_size[0] or 1) * int(local_size[1] or 1) * int(local_size[2] or 1))

    # Extract input data from the input buffer (little-endian 32-bit words).
    inputs_32: list[int] = []
    if isinstance(input_buf, (bytearray, bytes)) and n > 0:
      mv = memoryview(input_buf)
      for i in range(n):
        start = i * 4
        if start + 4 > len(mv): break
        inputs_32.append(int.from_bytes(mv[start:start+4], byteorder='little', signed=False))

    # If inputs are empty, synthesize a simple pattern for smoke tests.
    if not inputs_32:
      inputs_32 = [0x3f800000 + 0x01000000 * (i % 8) for i in range(max(1, n))]

    # Compute dims: use one block with n threads by default, unless caller provided sizes.
    block_dim = (
      max(1, int(local_size[0] or (n if n > 0 else 1))),
      max(1, int(local_size[1] or 1)),
      max(1, int(local_size[2] or 1)),
    )
    grid_dim = (max(1, int(global_size[0] or 1)), max(1, int(global_size[1] or 1)), max(1, int(global_size[2] or 1)))

    # Map params for PTXRenderer convention: typically param0=output_base, param1=input_base.
    params64: list[int] = [0x200, 0x100]

    from tools.rtl_runner import run_kernel_on_rtl
    ptx_text = self.lib.decode('utf-8')
    # Enable strict verification when DEBUG is high to catch partial-lane execution early.
    strict = bool(getenv("DEBUG", 0) >= 3)
    outputs = run_kernel_on_rtl(
      ptx_text,
      inputs_32=inputs_32,
      grid_dim=grid_dim,
      block_dim=block_dim,
      params64=params64,
      strict_verify=strict,
    )

    # Write outputs back to the output buffer (first buffer) if provided
    if len(bufs) >= 1 and isinstance(bufs[0], bytearray) and outputs:
      out_mv = memoryview(bufs[0])
      needed = min(len(outputs) * 4, len(out_mv))
      for i, val in enumerate(outputs):
        if i*4 >= needed: break
        out_mv[i*4:(i+1)*4] = int(val & 0xFFFFFFFF).to_bytes(4, byteorder='little', signed=False)

    # Synchronous execution; return None similar to CUDAProgram when wait=False.
    return None

class PTXASICDevice(Compiled):
  def __init__(self, device:str="PTXASIC:0"):
    self.arch = "sm_50"
    compilers: list[CompilerPairT] = [(functools.partial(PTXRenderer, self.arch, device="NV"), functools.partial(PTXASMCompiler, self.arch))]
    super().__init__(device, PTXASICAllocator(self), compilers, functools.partial(PTXASICProgram, self))
  def synchronize(self):
    # Nothing to do; the RTL runner is synchronous per run
    return
  # Convenience direct PTX execution (bypass compiled graph)
  def run_ptx(self, ptx_text:str, inputs_32:Sequence[int], *, grid_dim:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), params64:Sequence[int]|None=None, workdir:Path|None=None) -> list[int]:
    from tools.rtl_runner import run_kernel_on_rtl
    block_dim = (max(1, int(local_size[0] or 1)), max(1, int(local_size[1] or 1)), max(1, int(local_size[2] or 1)))
    return run_kernel_on_rtl(ptx_text, inputs_32=list(inputs_32), grid_dim=grid_dim, block_dim=block_dim, params64=list(params64 or []), workdir=workdir)

# Convenience factory; mirrors other backends' naming
# PTXASIC() returns a constructed device instance.

def PTXASIC(device:str="PTXASIC:0") -> PTXASICDevice:  # noqa: N802 - mimic existing device factories
  return PTXASICDevice(device)
