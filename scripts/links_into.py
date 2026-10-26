"""Before moving or deleting a directory: list every junction / symlink under the roots (default C:\\github) that points
INTO it, and every link inside it. A link pointing in means the files live there and another tree depends on them
(main's reports/analysis was once a junction into a worktree, and removing the worktree deleted them); a link inside
means a recursive delete may follow it out. Removing a junction itself: `rmdir <link>` (cmd), which leaves its target.
  python scripts/links_into.py PATH [--root DIR ...] [--depth N]   exit 1 if anything is found
  python scripts/links_into.py --all [--root DIR ...]               list every link under the roots
Links are not followed; .git, venvs and __pycache__ are skipped. Depth 6 covers every link found so far (all <= 3)."""
import argparse
import os
import sys
from pathlib import Path

SKIP = {".git", ".venv", ".venv-dac", "__pycache__", "node_modules", ".pytest_cache"}


def is_link(e: os.DirEntry) -> bool:
    return e.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(e.path))


def links(root: Path, depth: int):
    """(link path, resolved target) for every junction / symlink under root, not following links"""
    stack = [(root, 0)]
    while stack:
        d, k = stack.pop()
        try:
            it = list(os.scandir(d))
        except OSError:
            continue
        for e in it:
            if is_link(e):
                yield Path(e.path), Path(os.path.realpath(e.path))
            elif k < depth and e.name not in SKIP and e.is_dir(follow_symlinks=False):
                stack.append((Path(e.path), k + 1))


def inside(p: Path, d: Path) -> bool:
    p, d = os.path.normcase(str(p)), os.path.normcase(str(d))
    return p == d or p.startswith(d.rstrip("\\/") + os.sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root", action="append")
    ap.add_argument("--depth", type=int, default=6)
    a = ap.parse_args()
    roots = [Path(r) for r in (a.root or [r"C:\github"])]
    found = [(l, t) for r in roots for l, t in links(r, a.depth)]
    if a.all or not a.path:
        for l, t in sorted(found):
            print(f"{l}  ->  {t}")
        return
    target = Path(os.path.realpath(a.path))
    into = [(l, t) for l, t in found if inside(t, target) and not inside(l, target)]
    within = [(l, t) for l, t in found if inside(l, target)]
    print(f"links pointing into {target}: {len(into)}", *[f"  {l}  ->  {t}" for l, t in into], sep="\n")
    print(f"links inside it: {len(within)}", *[f"  {l}  ->  {t}" for l, t in within], sep="\n")
    sys.exit(1 if into or within else 0)


if __name__ == "__main__":
    main()
