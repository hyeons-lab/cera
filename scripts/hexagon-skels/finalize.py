#!/usr/bin/env python3
"""Turn a freshly built llama.cpp DSP skel into the file cera bundles.

    finalize.py libggml-htp-v79.so cera/src/backend/hexagon/skels/libcera-htp-v79.so
    finalize.py --as v85 libggml-htp-v81.so .../libcera-htp-v85.so     # v85 = renamed v81 build
    finalize.py --check cera/src/backend/hexagon/skels/libcera-htp-v79.so

llama.cpp ships the same skel names as cera, so an app embedding both would have one overwrite the
other in ADSP_LIBRARY_PATH. The bundled files are therefore renamed `libcera-htp-vNN.so` and the
SONAME inside is patched to match. `ggml` and `cera` are both four characters, so the patch is a
same-length byte substitution that moves nothing.

Upstream's build then appends a SWIV (software image validation) section: a CRC32 over the contents
of every loadable segment, BSS counted as zeros. Editing the SONAME after that leaves the CRC stale,
so this tool recomputes it. Before changing anything it recomputes the CRC of the input and requires
it to equal the one upstream's `htp-swiv.py` stored, which proves the computation here is the same one.
"""
import argparse, re, struct, sys, zlib

SWIV_MAGIC = b"SWIV"
SWIV_TYPE = 0xD3574956  # SHT_LOUSER plus ASCII "SWIV"
PT_LOAD = 1


def parse(data):
    """(ehdr dict, phdrs, shdrs) of an ELF32 or ELF64 little-endian image."""
    if data[:4] != b"\x7fELF" or data[5] != 1:
        raise SystemExit("not a little-endian ELF image")
    if data[4] == 1:  # ELF32
        e = dict(zip(("e_phoff", "e_shoff", "e_phentsize", "e_phnum", "e_shentsize", "e_shnum"),
                     (lambda t: (t[5], t[6], t[9], t[10], t[11], t[12]))(struct.unpack_from("<16sHHIIIIIHHHHHH", data))))
        ph = [dict(zip(("p_type", "p_offset", "p_filesz", "p_memsz"), (t[0], t[1], t[4], t[5])))
              for t in (struct.unpack_from("<8I", data, e["e_phoff"] + i * e["e_phentsize"]) for i in range(e["e_phnum"]))]
        sh = [dict(zip(("sh_type", "sh_offset", "sh_size"), (t[1], t[4], t[5])))
              for t in (struct.unpack_from("<10I", data, e["e_shoff"] + i * e["e_shentsize"]) for i in range(e["e_shnum"]))]
    else:  # ELF64
        t = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
        e = dict(zip(("e_phoff", "e_shoff", "e_phentsize", "e_phnum", "e_shentsize", "e_shnum"),
                     (t[5], t[6], t[9], t[10], t[11], t[12])))
        ph = [dict(zip(("p_type", "p_offset", "p_filesz", "p_memsz"), (t[0], t[2], t[5], t[6])))
              for t in (struct.unpack_from("<IIQQQQQQ", data, e["e_phoff"] + i * e["e_phentsize"]) for i in range(e["e_phnum"]))]
        sh = [dict(zip(("sh_type", "sh_offset", "sh_size"), (t[1], t[4], t[5])))
              for t in (struct.unpack_from("<IIQQQQIIQQ", data, e["e_shoff"] + i * e["e_shentsize"]) for i in range(e["e_shnum"]))]
    return e, ph, sh


def swiv_section(data):
    _, _, sh = parse(data)
    hits = [s for s in sh if s["sh_type"] == SWIV_TYPE]
    if len(hits) != 1 or hits[0]["sh_size"] != 16 or data[hits[0]["sh_offset"]:hits[0]["sh_offset"] + 4] != SWIV_MAGIC:
        raise SystemExit("expected exactly one 16-byte SWIV section (was the skel built with htp-swiv.py?)")
    return hits[0]["sh_offset"]


def crc_of(data):
    """CRC32 over the loadable segments' file bytes, BSS as zeros (htp-swiv.py `crc32_calculate`)."""
    _, ph, _ = parse(data)
    crc = 0
    for p in ph:
        if p["p_type"] != PT_LOAD:
            continue
        crc = zlib.crc32(data[p["p_offset"]:p["p_offset"] + p["p_filesz"]], crc)
        if p["p_memsz"] > p["p_filesz"]:
            crc = zlib.crc32(bytes(p["p_memsz"] - p["p_filesz"]), crc)
    return crc & 0xFFFFFFFF


def stored_crc(data):
    return struct.unpack_from("<I", data, swiv_section(data) + 4)[0]


def soname(data):
    names = set(re.findall(rb"lib[a-z]+-htp-v\d+\.so", data))
    return sorted(n.decode() for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the SWIV CRC of one file and exit")
    ap.add_argument("--as", dest="as_arch", help="architecture to name the output after (default: the input's)")
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    a = ap.parse_args()

    data = bytearray(open(a.src, "rb").read())
    want, got = stored_crc(data), crc_of(data)
    if want != got:
        sys.exit(f"{a.src}: SWIV CRC {want:#010x} does not match the contents ({got:#010x})")
    print(f"{a.src}: SWIV CRC {want:#010x} ok; names inside: {soname(data)}")
    if a.check:
        return
    if not a.dst:
        sys.exit("need an output path")

    m = re.search(rb"libggml-htp-(v\d+)\.so", data)
    if not m:
        sys.exit("no libggml-htp-vNN.so SONAME found in the input")
    old = m.group(0)
    new = b"libcera-htp-" + (a.as_arch.encode() if a.as_arch else m.group(1)) + b".so"
    assert len(old) == len(new), (old, new)
    count = data.count(old)
    data = bytearray(bytes(data).replace(old, new))
    struct.pack_into("<I", data, swiv_section(data) + 4, crc_of(data))
    assert stored_crc(data) == crc_of(data)
    open(a.dst, "wb").write(data)
    print(f"{a.dst}: {old.decode()} -> {new.decode()} ({count} occurrence(s)), SWIV CRC now {stored_crc(data):#010x}")


if __name__ == "__main__":
    main()
