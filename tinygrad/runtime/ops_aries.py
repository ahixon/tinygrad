from __future__ import annotations

import functools, os, re, shutil, subprocess, sys, tempfile
from dataclasses import dataclass
from pathlib import Path

from tinygrad.device import BufferSpec, Compiler, CompilerPair, CompilerSet, Compiled, LRUAllocator
from tinygrad.helpers import DEBUG, getenv
from tinygrad.renderer.ptx import PTXRenderer

# Aries simulator global memory is a fixed 4096-word (16 KiB) array in the testbench.
GMEM_WORDS = 4096
GMEM_BYTES = GMEM_WORDS * 4

# Reserve low GMEM for the kernel parameter block (PTX uses ld.param).
# If buffers are allocated at address 0, kernels will read buffer data as
# parameters/pointers and then dereference garbage addresses.
PARAM_BYTES = 0x100

# R239 is used as a spill/frame base by the Aries backend. Keep it well away
# from the buffer heap (which grows upward from PARAM_BYTES) so that allocas/
# spills don't alias user buffers.
STACK_BYTES = 0x400
STACK_BASE_DEFAULT = GMEM_BYTES - STACK_BYTES


_PTX_STRIP_DIRECTIVES = (
  # tinygrad emits this between ")" and "{"; ptx2ll expects "{" directly.
  "maxntid",
)


def _ptx_elem_size_bytes(ty: str) -> int:
  ty = ty.lower()
  if ty.endswith("64") or ty in {"b64", "u64", "s64", "f64"}:
    return 8
  if ty.endswith("16") or ty in {"b16", "u16", "s16", "f16"}:
    return 2
  if ty.endswith("8") or ty in {"b8", "u8", "s8"}:
    return 1
  return 4


def _ptx_add_offset(addr_expr: str, offset: int) -> str:
  addr_expr = addr_expr.strip()
  if offset == 0:
    return addr_expr

  # Common tinygrad PTX pattern: "%rdX+0" or "%rdX+123".
  m = re.match(r"^(?P<base>.*?)(?P<sign>[+-])(?P<imm>\d+)$", addr_expr)
  if m:
    base = m.group("base").rstrip()
    imm = int(m.group("imm"))
    if m.group("sign") == "-":
      imm = -imm
    new_imm = imm + offset
    sign = "+" if new_imm >= 0 else "-"
    return f"{base}{sign}{abs(new_imm)}"

  # Fallback: append "+<offset>".
  return f"{addr_expr}+{offset}"


_PTX_VEC_ST_RE = re.compile(
  r"^(?P<indent>\s*)st\.global\.v(?P<n>[24])\.(?P<ty>[A-Za-z0-9]+)\s+\[(?P<addr>[^\]]+)\]\s*,\s*\{(?P<regs>[^}]+)\}\s*;\s*$",
  re.IGNORECASE,
)
_PTX_VEC_LD_RE = re.compile(
  r"^(?P<indent>\s*)ld\.global\.v(?P<n>[24])\.(?P<ty>[A-Za-z0-9]+)\s+\{(?P<regs>[^}]+)\}\s*,\s*\[(?P<addr>[^\]]+)\]\s*;\s*$",
  re.IGNORECASE,
)


def _sanitize_ptx_for_ptx2ll(src: str) -> str:
  # 1) tinygrad uses placeholders.
  src = src.replace("TARGET", "sm_50").replace("VERSION", "7.0")

  # 2) Strip directives ptx2ll doesn't parse.
  out_lines: list[str] = []
  for line in src.splitlines():
    stripped = line.lstrip()
    if stripped.startswith("."):
      directive = stripped[1:].split(maxsplit=1)[0]
      if directive in _PTX_STRIP_DIRECTIVES:
        continue
    out_lines.append(line)
  src = "\n".join(out_lines)

  # 3) Scalarize vector global loads/stores into scalar ops.
  scalar_lines: list[str] = []
  for line in src.splitlines():
    if (m := _PTX_VEC_ST_RE.match(line)) is not None:
      n = int(m.group("n"))
      ty = m.group("ty")
      indent = m.group("indent")
      addr = m.group("addr")
      regs = [r.strip() for r in m.group("regs").split(",") if r.strip()]
      if len(regs) != n:
        scalar_lines.append(line)
        continue
      stride = _ptx_elem_size_bytes(ty)
      for i, reg in enumerate(regs):
        addr_i = _ptx_add_offset(addr, i * stride)
        scalar_lines.append(f"{indent}st.global.{ty} [{addr_i}], {reg};")
      continue

    if (m := _PTX_VEC_LD_RE.match(line)) is not None:
      n = int(m.group("n"))
      ty = m.group("ty")
      indent = m.group("indent")
      addr = m.group("addr")
      regs = [r.strip() for r in m.group("regs").split(",") if r.strip()]
      if len(regs) != n:
        scalar_lines.append(line)
        continue
      stride = _ptx_elem_size_bytes(ty)
      for i, reg in enumerate(regs):
        addr_i = _ptx_add_offset(addr, i * stride)
        scalar_lines.append(f"{indent}ld.global.{ty} {reg}, [{addr_i}];")
      continue

    scalar_lines.append(line)
  src = "\n".join(scalar_lines)

  # 4) Some tinygrad PTX strings have a stray trailing '%'. Drop it.
  src = src.rstrip()
  if src.endswith("%"):
    src = src[:-1].rstrip()
  return src + "\n"


def _find_aries_repo_root() -> Path:
  p = Path(__file__).resolve()
  for parent in p.parents:
    if (parent / "tools" / "ariesc.py").exists() and (parent / "sim" / "Makefile").exists():
      return parent
  raise RuntimeError("Failed to locate aries-rtl repo root (expected tools/ariesc.py and sim/Makefile)")


def _default_llc_path(aries_root: Path) -> str | None:
  cand = aries_root / "external" / "llvm-project" / "build-aries" / "bin" / "llc"
  return str(cand) if cand.exists() else None


def _u32(x: int) -> int:
  return x & 0xFFFF_FFFF


@dataclass
class _Block:
  addr: int
  size: int


class AriesAllocator(LRUAllocator['ARIESDevice']):
  def __init__(self, dev: 'ARIESDevice'):
    super().__init__(dev)
    self._next = PARAM_BYTES
    self._free_list: list[_Block] = []

  def _alloc(self, size: int, options: BufferSpec):
    if options.external_ptr is not None:
      return int(options.external_ptr)

    # Align allocations to 16B for a bit of sanity.
    size_aligned = (size + 15) & ~15

    for i, blk in enumerate(self._free_list):
      if blk.size >= size_aligned:
        self._free_list.pop(i)
        return blk.addr

    if self._next + size_aligned > GMEM_BYTES:
      raise RuntimeError(f"ARIES GMEM OOM: need {size_aligned} bytes, have {GMEM_BYTES - self._next} bytes left")

    addr = self._next
    self._next += size_aligned
    return addr

  def _free(self, opaque, options: BufferSpec):
    # Best-effort free list. Buffer sizes are tracked by tinygrad's allocator layer.
    # We don't know the size here, so the LRUAllocator will bypass _free for cached buffers.
    # For non-cached frees, we just leak (acceptable for now).
    pass

  def _copyin(self, dest, src: memoryview):
    dest_addr = int(dest)
    if dest_addr < 0 or dest_addr + len(src) > GMEM_BYTES:
      raise RuntimeError("ARIES copyin out of bounds")
    self.dev._gmem[dest_addr:dest_addr + len(src)] = src.tobytes()

  def _copyout(self, dest: memoryview, src):
    src_addr = int(src)
    if src_addr < 0 or src_addr + len(dest) > GMEM_BYTES:
      raise RuntimeError("ARIES copyout out of bounds")
    dest[:] = self.dev._gmem[src_addr:src_addr + len(dest)]

  def _offset(self, buf, size: int, offset: int):
    return int(buf) + offset


class AriesCompiler(Compiler):
  def __init__(self, aries_root: Path, llc: str | None = None):
    super().__init__(cachekey="ARIES")
    self.aries_root = aries_root
    self.ariesc = aries_root / "tools" / "ariesc.py"
    if not self.ariesc.exists():
      raise RuntimeError(f"ariesc.py not found at {self.ariesc}")
    self.llc = llc

  def compile(self, src: str) -> bytes:
    src = _sanitize_ptx_for_ptx2ll(src)

    keep = bool(int(getenv("ARIES_KEEP_TMP", "0")))
    if keep:
      # Put kept compilation artifacts inside the repo so they are easy to inspect.
      keep_dir = self.aries_root / "sim" / "logs"
      keep_dir.mkdir(parents=True, exist_ok=True)
      td = tempfile.mkdtemp(prefix="tinygrad-aries-", dir=os.fspath(keep_dir))
    else:
      td = tempfile.mkdtemp(prefix="tinygrad-aries-")
    td_path = Path(td)
    try:
      ptx_path = td_path / "kernel.ptx"
      lcb_path = td_path / "kernel.lcb"
      ptx_path.write_text(src)

      cmd = [
        sys.executable,
        os.fspath(self.ariesc),
        os.fspath(ptx_path),
        "-o",
        os.fspath(lcb_path),
      ]
      if keep:
        cmd += ["--keep-tmp"]
      llc = self.llc or os.environ.get("ARIES_LLC")
      if llc:
        cmd += ["--llc", llc]

      proc = subprocess.run(cmd, cwd=self.aries_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      if proc.returncode != 0:
        raise RuntimeError(
          "ariesc failed\n"
          f"tmp: {td_path}\n"
          f"cmd: {' '.join(cmd)}\n"
          f"stderr:\n{proc.stderr}\n"
          f"stdout:\n{proc.stdout}"
        )

      if DEBUG >= 2 and keep:
        print(f"[ARIES] kept compile tmp: {td_path}")

      return lcb_path.read_bytes()
    finally:
      if not keep:
        shutil.rmtree(td_path, ignore_errors=True)


class AriesProgram:
  def __init__(self, dev: 'ARIESDevice', name: str, lib: bytes, smem: int = 0):
    self.dev, self.name, self.lib, self.smem = dev, name, lib, smem
    if smem != 0:
      raise RuntimeError("ARIES simulator backend does not support dynamic shared memory yet")

  def __call__(
    self,
    *args,
    global_size: tuple[int, int, int] = (1, 1, 1),
    local_size: tuple[int, int, int] = (1, 1, 1),
    vals: tuple[int, ...] = (),
    wait: bool = False,
  ):
    # MVP constraints: one CTA, one warp (<=32 lanes).
    if local_size[1:] != (1, 1) or not (1 <= local_size[0] <= 32):
      raise RuntimeError(f"ARIES currently requires local_size=(N,1,1) with 1<=N<=32, got {local_size}")
    if global_size != (1, 1, 1):
      raise RuntimeError(f"ARIES currently requires global_size=(1,1,1), got {global_size}")

    # Aries bring-up ABI: kernel args are in R0.. as 32-bit values.
    # tinygrad PTX passes all pointers as .u64, but our simulated address space
    # is < 4GiB so we pass only the low 32 bits.
    regs: list[int] = []
    for a in args:
      regs += [_u32(int(a))]
    regs += [_u32(int(v)) for v in vals]
    if len(regs) > 8:
      raise RuntimeError(f"ARIES supports up to 8 arg registers (R0..R7); need {len(regs)}")
    regs += [0] * (8 - len(regs))

    self.dev._ensure_sim_built()

    with tempfile.TemporaryDirectory(prefix="aries-sim-") as td:
      td_path = Path(td)
      lcb_path = td_path / "kernel.lcb"
      gmem_init = td_path / "gmem_init.txt"
      gmem_dump = td_path / "gmem_dump.txt"

      lcb_path.write_bytes(self.lib)

      # Write full GMEM init (4096 words). Simulator reads sequential words.
      words = []
      mv = memoryview(self.dev._gmem)
      for wi in range(GMEM_WORDS):
        w = int.from_bytes(mv[wi*4:(wi+1)*4], "little", signed=False)
        words.append(f"{w:08x}\n")
      gmem_init.write_text(''.join(words))

      cycle_budget = int(getenv("ARIES_CYCLE_BUDGET", "20000"))
      r239 = int(getenv("ARIES_R239", hex(STACK_BASE_DEFAULT)), 0)
      make_cmd = [
        "make",
        "-C",
        os.fspath(self.dev._aries_root / "sim"),
        "run",
        "TEST=BASIC",
        f"FAST={1 if self.dev._fast else 0}",
        f"LCB={lcb_path}",
        f"GMEM_INIT={gmem_init}",
        f"GMEM_DUMP={gmem_dump}",
        "GMEM_DUMP_BASE=0",
        f"GMEM_DUMP_WORDS={GMEM_WORDS}",
        f"CYCLE_BUDGET={cycle_budget}",
        f"NUM_THREADS={local_size[0]}",
        f"ARG0={regs[0]:08x}",
        f"ARG1={regs[1]:08x}",
        f"ARG2={regs[2]:08x}",
        f"ARG3={regs[3]:08x}",
        f"ARG4={regs[4]:08x}",
        f"ARG5={regs[5]:08x}",
        f"ARG6={regs[6]:08x}",
        f"ARG7={regs[7]:08x}",
        f"R239={_u32(r239):08x}",
      ]

      if DEBUG >= 2:
        print("[ARIES] sim:", " ".join(make_cmd))

      proc = subprocess.run(make_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      if proc.returncode != 0:
        raise RuntimeError(f"ARIES simulator run failed\ncmd: {' '.join(make_cmd)}\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")

      if DEBUG >= 2:
        if proc.stdout.strip():
          print("[ARIES] sim stdout:\n" + proc.stdout)
        if proc.stderr.strip():
          print("[ARIES] sim stderr:\n" + proc.stderr)

      # Read back GMEM dump.
      dump_words = [int(line.strip(), 16) for line in gmem_dump.read_text().splitlines() if line.strip()]
      if len(dump_words) < GMEM_WORDS:
        dump_words += [0] * (GMEM_WORDS - len(dump_words))
      for wi in range(GMEM_WORDS):
        self.dev._gmem[wi*4:(wi+1)*4] = int(dump_words[wi] & 0xFFFF_FFFF).to_bytes(4, "little")

    return None


class ARIESDevice(Compiled):
  def __init__(self, device: str):
    self._aries_root = _find_aries_repo_root()
    self._fast = bool(int(getenv("ARIES_FAST", "1")))
    self._gmem = bytearray(GMEM_BYTES)

    llc = os.environ.get("ARIES_LLC") or _default_llc_path(self._aries_root)
    compilers = CompilerSet([
      CompilerPair(functools.partial(PTXRenderer, "sm_50", device="ARIES"), functools.partial(AriesCompiler, self._aries_root, llc)),
    ])
    super().__init__(device, AriesAllocator(self), compilers, functools.partial(AriesProgram, self), None)

  def _ensure_sim_built(self) -> None:
    sim_bin = (self._aries_root / "sim" / ("obj_dir_fast" if self._fast else "obj_dir") / "Vtb_aries_sm")
    if sim_bin.exists():
      return

    cmd = [
      "make",
      "-C",
      os.fspath(self._aries_root / "sim"),
      "verilator_build",
      f"FAST={1 if self._fast else 0}",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
      raise RuntimeError(f"Failed to build Aries simulator\ncmd: {' '.join(cmd)}\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")

  def synchronize(self):
    # All operations are synchronous (subprocess calls).
    return
