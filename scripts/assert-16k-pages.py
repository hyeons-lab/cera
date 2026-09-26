#!/usr/bin/env python3
"""Assert every PT_LOAD segment of an ELF is 16KB-aligned.

Why this exists
---------------
Android 15+ devices can use 16KB pages, and the dynamic linker refuses to
load native libraries whose LOAD segments are only 4KB-aligned. Google Play
requires 16KB-clean binaries for updates. NDK r28+ aligns to 16KB by
default, but this repo pins r27c (whose default is still 4KB), so every
Android target carries explicit `-z max-page-size=16384` linker flags (see
`.cargo/config.toml`), and this script enforces the result, because a flag
that silently stops reaching the link would ship a library that crashes on
first load for 16KB-page users, caught only by running on that hardware.

Usage: scripts/assert-16k-pages.py <elf> [...]

Pure stdlib (no readelf/llvm-readelf), so the same check runs on the Linux
CI runners, on macOS dev machines, and anywhere else with python3. Checks
32- and 64-bit, little- and big-endian ELFs.

Scope: host-loaded ELFs (libcera_ffi.so per ABI, the cera binary).
Hexagon DSP skels are embedded inside libcera_ffi.so and extracted to app
storage at runtime by HexagonNpu.setup, so all binaries staged in jniLibs
are checked.
"""

import struct
import sys

PAGE = 16384
PT_LOAD = 1


def load_alignments(path):
    """Return [(index, p_align)] for every PT_LOAD, or raise ValueError."""
    with open(path, "rb") as f:
        ident = f.read(16)
    if len(ident) < 16 or ident[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    ei_class, ei_data = ident[4], ident[5]
    if ei_class == 1:
        is64 = False
    elif ei_class == 2:
        is64 = True
    else:
        raise ValueError(f"unknown EI_CLASS {ei_class}")
    if ei_data == 1:
        endian = "<"
    elif ei_data == 2:
        endian = ">"
    else:
        raise ValueError(f"unknown EI_DATA {ei_data}")

    with open(path, "rb") as f:
        if is64:
            f.seek(0x20)
            (e_phoff,) = struct.unpack(endian + "Q", f.read(8))
            f.seek(0x36)
            e_phentsize, e_phnum = struct.unpack(endian + "HH", f.read(4))
            ph_fmt = endian + "IIQQQQQQ"
            ph_size = 56
        else:
            f.seek(0x1C)
            (e_phoff,) = struct.unpack(endian + "I", f.read(4))
            f.seek(0x2A)
            e_phentsize, e_phnum = struct.unpack(endian + "HH", f.read(4))
            ph_fmt = endian + "IIIIIIII"
            ph_size = 32
        loads = []
        for i in range(e_phnum):
            f.seek(e_phoff + i * e_phentsize)
            raw = f.read(ph_size)
            if len(raw) < ph_size:
                raise ValueError("truncated program header table")
            fields = struct.unpack(ph_fmt, raw)
            if is64:
                p_type, _, _, _, _, _, _, p_align = fields
            else:
                p_type, _, _, _, _, _, _, p_align = fields
            if p_type == PT_LOAD:
                loads.append((i, p_align))
    if not loads:
        raise ValueError("no PT_LOAD segments")
    return loads


def main(argv):
    if len(argv) < 2:
        print("usage: assert-16k-pages.py <elf> [...]", file=sys.stderr)
        return 2
    failed = False
    for path in argv[1:]:
        try:
            loads = load_alignments(path)
        except OSError as e:
            print(f"FAIL {path}: cannot read: {e}", file=sys.stderr)
            failed = True
            continue
        except ValueError as e:
            print(f"FAIL {path}: {e}", file=sys.stderr)
            failed = True
            continue
        bad = [(i, a) for i, a in loads if a < PAGE]
        if bad:
            failed = True
            detail = ", ".join(f"LOAD#{i} align {a:#x}" for i, a in bad)
            print(f"FAIL {path}: {detail} (need >= {PAGE:#x})", file=sys.stderr)
        else:
            smallest = min(a for _, a in loads)
            print(f"OK {path}: {len(loads)} LOAD segments, min align {smallest:#x}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
