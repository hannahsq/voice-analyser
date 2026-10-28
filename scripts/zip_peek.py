"""Look inside a remote zip without downloading it (AGENTS: downloads are untrusted; check sizes and paths before
unpacking). HTTP range requests fetch the archive's tail, which holds the central directory; the entries are listed
with their sizes and any unsafe paths (absolute, '..', drive letters) flagged. One member can then be fetched alone.
Handles zip64. Stored and deflated members only.
  python scripts/zip_peek.py list URL [--tail MB]        entries, sizes, extension counts, unsafe paths
  python scripts/zip_peek.py get URL MEMBER OUT [--tail MB]
The tail must reach back to the central directory's start (default 64 MB; the error says if it does not)."""
import argparse
import collections
import os
import struct
import subprocess
import zlib


def curl(url, rng=None, head=False):
    args = ["curl", "-sfL"] + (["-I"] if head else []) + (["-r", rng] if rng else []) + [url]
    return subprocess.run(args, capture_output=True, check=True).stdout


def remote_size(url):
    """from a one-byte range GET's Content-Range (some hosts, e.g. figshare's signed S3 redirects, refuse HEAD)"""
    args = ["curl", "-sfL", "-r", "0-0", "-o", os.devnull, "-D", "-", url]
    hdrs = subprocess.run(args, capture_output=True, check=True).stdout.splitlines()
    sizes = [int(l.rsplit(b"/", 1)[1]) for l in hdrs if l.lower().startswith(b"content-range:")]
    return sizes[-1]  # the last response after redirects


def entries(tail: bytes, total: int):
    """central directory entries (name, method, compressed size, size, local header offset) from the archive's last
    len(tail) bytes; total = the archive's size"""
    base = total - len(tail)
    e = tail.rfind(b"PK\x05\x06")
    assert e >= 0, "no end-of-central-directory record in the tail"
    off = struct.unpack("<I", tail[e + 16:e + 20])[0]
    if off == 0xFFFFFFFF:
        z64 = tail.rfind(b"PK\x06\x06")
        off = struct.unpack("<Q", tail[z64 + 48:z64 + 56])[0]
    p = off - base
    assert p >= 0, f"the central directory starts {-p} bytes before the fetched tail; raise --tail"
    out = []
    while tail[p:p + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", tail[p + 10:p + 12])
        csz, usz = struct.unpack("<II", tail[p + 20:p + 28])
        ln, le, lc = struct.unpack("<HHH", tail[p + 28:p + 34])
        lho, = struct.unpack("<I", tail[p + 42:p + 46])
        name = tail[p + 46:p + 46 + ln].decode("utf-8", "replace")
        extra = tail[p + 46 + ln:p + 46 + ln + le]
        q = 0
        while q + 4 <= len(extra):  # zip64 extra field: the 64-bit values, in this order, for the fields that overflowed
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                it = iter(struct.unpack("<" + "Q" * (hl // 8), extra[q + 4:q + 4 + hl]))
                if usz == 0xFFFFFFFF:
                    usz = next(it)
                if csz == 0xFFFFFFFF:
                    csz = next(it)
                if lho == 0xFFFFFFFF:
                    lho = next(it)
            q += 4 + hl
        out.append((name, meth, csz, usz, lho))
        p += 46 + ln + le + lc
    return out


def unsafe(name):
    return name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/") or ":" in name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "get"))
    ap.add_argument("url")
    ap.add_argument("member", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--tail", type=float, default=64)
    a = ap.parse_args()
    total = remote_size(a.url)
    tail = curl(a.url, f"-{int(min(a.tail * 2**20, total))}")
    ents = entries(tail, total)
    if a.cmd == "list":
        print(f"{total / 2**30:.2f} GB archive, {len(ents)} entries, {sum(e[3] for e in ents) / 2**30:.2f} GB unpacked")
        bad = [e[0] for e in ents if unsafe(e[0])]
        print(f"unsafe paths: {len(bad)}", *bad[:10], sep="\n  ")
        ext = collections.Counter(n.rsplit(".", 1)[-1] if "." in n.rsplit("/", 1)[-1] else "(none)" for n, *_ in ents)
        print("extensions:", ext.most_common(20))
        for n, _, _, usz, _ in ents:
            print(usz, n)
        return
    name, meth, csz, usz, lho = next(e for e in ents if e[0] == a.member)
    assert meth in (0, 8), f"compression method {meth} not handled"
    h = curl(a.url, f"{lho}-{lho + 30 + 65535 * 2 + csz}")
    n, m = struct.unpack("<HH", h[26:30])
    raw = h[30 + n + m:30 + n + m + csz]
    data = zlib.decompress(raw, -15) if meth == 8 else raw
    assert len(data) == usz, "size mismatch"
    with open(a.out, "wb") as f:
        f.write(data)
    print(f"{name}: {usz} bytes -> {a.out}")


if __name__ == "__main__":
    main()
