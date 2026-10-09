"""Reproducible corpus downloads: pinned URLs, checksums, safe extraction.

    va download hillenbrand ptdb_tug mir1k mdb_stem_synth   # gold sets
    va download vocalset                                    # training pools
    va download cmu_arctic                                  # EGG analysis set (bdl, jmk, slt)

Archives go to <data.root>/downloads/, and each corpus is extracted into its
own <data.root>/<corpus>/ directory. Nothing inside an archive is ever
executed; tar extraction uses the "data" filter, and zip/rar entries with
absolute or parent-relative paths are rejected.
"""

from __future__ import annotations

import hashlib
import http.client
import shutil
import subprocess
import tarfile
import time
import urllib.request
from fnmatch import fnmatch
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Source:
    url: str
    filename: str
    checksum: str | None  # "md5:<hex>" or "sha256:<hex>"; None = record on first download
    size: int | None = None
    inner_archives: tuple[str, ...] = ()  # archives inside the archive to extract too
    include: tuple[str, ...] = ()  # extract only members matching these patterns (fnmatch); () = all
    note: str = ""


def _arctic(voice: str, kind: str, sha256: str, size: int) -> Source:
    """CMU ARCTIC 0.95 (Kominek & Black 2003): the original stereo speech + EGG at 32 kHz ("WAVEGG"), or the
    release, of which only the phone labels and prompts are kept."""
    if kind == "WAVEGG":
        url = f"http://festvox.org/cmu_arctic/cmu_arctic/orig/cmu_us_{voice}_arctic-WAVEGG.tar.bz2"
        include = (f"cmu_us_{voice}_arctic/orig/*.wav", f"cmu_us_{voice}_arctic/COPYING", f"cmu_us_{voice}_arctic/README")
    else:
        url = f"http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_{voice}_arctic-0.95-release.tar.bz2"
        include = (f"cmu_us_{voice}_arctic/lab/*.lab", f"cmu_us_{voice}_arctic/etc/txt.done.data")
    return Source(url=url, filename=f"cmu_arctic/{url.rsplit('/', 1)[1]}", checksum=f"sha256:{sha256}", size=size,
                  include=include)


SOURCES: dict[str, Source | tuple[Source, ...]] = {
    "hillenbrand": Source(
        url="https://github.com/santiagobarreda/hillenbrand_et_al_1995/raw/6dd44decc8ef5b537cbb54732a3b00c8fa652e65/h95-alldata.zip",
        filename="h95-alldata.zip",
        checksum="sha256:2560548591e3a726c88549b6dc9d226995616c26874d601c69fee8b9eee9d730",
        inner_archives=("men.zip", "women.zip", "kids.zip"),
        note="Original distribution, mirrored by S. Barreda with permission from J. Hillenbrand "
        "(the wmich.edu page no longer resolves).",
    ),
    "ptdb_tug": Source(
        url="https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip",
        filename="SPEECH_DATA_ZIPPED.zip",
        checksum="md5:e590798aa0befaf7b458d3c38cda4030",
        size=4217094594,
    ),
    "mir1k": Source(
        url="https://ndownloader.figshare.com/files/10256751",
        filename="MIR-1K.rar",
        checksum="md5:1810d01457ccbb84a0b41c4da53eee74",
        note="figshare mirror (article 5802891); the mirlab.org link is dead and Zenodo 3532216 is restricted.",
    ),
    "mdb_stem_synth": Source(
        url="https://zenodo.org/api/records/1481172/files/MDB-stem-synth.tar.gz/content",
        filename="MDB-stem-synth.tar.gz",
        checksum="md5:31c1f6b4888e5fd108af91c69789a809",
        size=1845117819,
    ),
    # ---- training pools (plan §5.6): never gold, speakers disjoint from every gold set
    "vocalset": Source(
        url="https://zenodo.org/api/records/1442513/files/VocalSet11.zip/content",
        filename="VocalSet11.zip",
        checksum="md5:0c09396242f946e7111ad7d8fc649b81",
        size=2077243579,
        note="VocalSet v1.1 (Wilkins et al. 2018), CC BY 4.0, from Zenodo record 1442513 (v1.2). v1.2's own zip "
        "holds the same audio three times (by singer, technique, vowel); v1.1 has it once, by singer.",
    ),
    # ---- analysis sets
    # CMU ARCTIC: only bdl, jmk and slt have EGG (the other voices' lar/ directories are empty). Free for any
    # purpose with the copyright notice kept (COPYING). festvox.org serves these over plain HTTP only; the
    # checksums are pinned.
    "cmu_arctic": (
        _arctic("bdl", "WAVEGG", "bdae8de8689752caa5ef8ad2cb9b94fb0d9c2a3ee6dee6907d51dbba51c3b653", 402399651),
        _arctic("jmk", "WAVEGG", "8089374059b2706e1c7add9fee758d4052db7167edfbaea81358a7057e1be5b1", 414794292),
        _arctic("slt", "WAVEGG", "91e9c76429f2b5d8aa65f490af22af6b180be232811e5a133132d4e98cf8ed48", 362378154),
        _arctic("bdl", "release", "3c9ec89942056cb4609ea788a6a455d507f5fa3729fc5d0ffcb6ba7ae1d594c9", 94333107),
        _arctic("jmk", "release", "9f418da3db9dc037056990dbbb2ce6ad91b84e853ab7f9839c6db50c8ad6b411", 91729965),
        _arctic("slt", "release", "9fddec16fbfbfb7d4989dff0fe77ccbe31f80b07b57be49d09994aa7a67d6dba", 119914432),
    ),
}


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(src: Source, dest_dir: Path, log=print, retries: int = 8) -> Path:
    """Download with resume into dest_dir (retrying dropped connections); verify the checksum."""
    out = dest_dir / src.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        try:
            _fetch_once(src, out, log)
            break
        except (OSError, http.client.HTTPException) as e:  # URLError, resets and timeouts are OSErrors
            if attempt == retries:
                raise
            wait = min(60, 5 * 2**attempt)
            log(f"  {src.filename}: {type(e).__name__}: {e}; resuming in {wait}s ({attempt + 1}/{retries})")
            time.sleep(wait)
    if src.checksum:
        algo, want = src.checksum.split(":")
        got = _hash(out, algo)
        if got != want:
            raise ValueError(f"{out}: {algo} {got} != expected {want}; delete it and retry")
        log(f"  {src.filename}: {algo} verified")
    else:
        log(f"  {src.filename}: no pinned checksum; sha256 {_hash(out, 'sha256')} (pin this in SOURCES)")
    return out


def _fetch_once(src: Source, out: Path, log=print) -> None:
    part = out.with_suffix(out.suffix + ".part")
    if not out.exists():
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(src.url, headers={"User-Agent": "vocal-analysis/0.0.1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=60) as resp:
            mode = "ab" if have and resp.status == 206 else "wb"
            total = src.size or (int(resp.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0))
            done = have if mode == "ab" else 0
            next_report = done + (1 << 28)
            with part.open(mode) as f:
                while block := resp.read(1 << 20):
                    f.write(block)
                    done += len(block)
                    if done >= next_report:
                        log(f"  {src.filename}: {done / 1e9:.2f} / {total / 1e9:.2f} GB")
                        next_report += 1 << 28
        if total and done < total:
            raise ConnectionError(f"short read: {done} of {total} bytes")
        part.replace(out)


def _safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts and not (p.parts and p.parts[0].endswith(":"))


def _is_mac_junk(name: str) -> bool:
    """AppleDouble '._*' files, .DS_Store and __MACOSX/ from archives made on macOS."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return "__MACOSX" in parts or (bool(parts) and (parts[-1].startswith("._") or parts[-1] == ".DS_Store"))


def extract(archive: Path, dest: Path, include: tuple[str, ...] = ()) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    keep = (lambda n: any(fnmatch(n, pat) for pat in include)) if include else (lambda n: True)  # noqa: E731
    if include and name.endswith(".rar"):
        raise ValueError(f"{archive}: member filtering is not supported for RAR")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            bad = [n for n in z.namelist() if not _safe_member(n)]
            if bad:
                raise ValueError(f"{archive}: unsafe paths {bad[:3]}")
            z.extractall(dest, members=[n for n in z.namelist() if not _is_mac_junk(n) and keep(n)])
    elif name.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tbz2")):
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if not _is_mac_junk(m.name) and keep(m.name)]
            t.extractall(dest, members=members, filter="data")
    elif name.endswith(".rar"):
        # Windows' bundled bsdtar (libarchive) reads RAR; list first to vet paths.
        listing = subprocess.run(["tar", "-tf", str(archive)], capture_output=True, text=True, check=True).stdout
        bad = [n for n in listing.splitlines() if n and not _safe_member(n)]
        if bad:
            raise ValueError(f"{archive}: unsafe paths {bad[:3]}")
        subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)], check=True)
    else:
        raise ValueError(f"don't know how to extract {archive}")


def _flatten_single_dir(dest: Path) -> None:
    """If extraction produced exactly one top-level directory, move its contents up."""
    entries = [e for e in dest.iterdir() if e.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        tmp = dest.parent / f".{dest.name}.flatten"
        inner.rename(tmp)
        for child in tmp.iterdir():
            shutil.move(str(child), dest / child.name)
        tmp.rmdir()


def download(corpus: str, data_root: Path, log=print) -> Path:
    """Fetch and extract a corpus. A corpus of several archives is extracted into one directory, unflattened."""
    srcs = SOURCES[corpus]
    srcs = (srcs,) if isinstance(srcs, Source) else srcs
    dest = data_root / corpus
    marker = dest / ".extracted"
    if marker.exists():
        log(f"[{corpus}] already present at {dest}")
        return dest
    for src in srcs:
        log(f"[{corpus}] downloading {src.url}")
        archive = fetch(src, data_root / "downloads", log)
        log(f"[{corpus}] extracting {archive.name} to {dest}")
        extract(archive, dest, src.include)
    if len(srcs) == 1:
        _flatten_single_dir(dest)
    for src in srcs:
        for inner in src.inner_archives:
            hits = list(dest.rglob(inner))
            if not hits:
                raise FileNotFoundError(f"{corpus}: expected inner archive {inner}")
            for h in hits:
                extract(h, h.parent)
                h.unlink()
    marker.write_text("".join(s.url + "\n" for s in srcs), encoding="utf-8")
    log(f"[{corpus}] done")
    return dest
