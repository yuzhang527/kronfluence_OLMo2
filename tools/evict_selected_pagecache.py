#!/usr/bin/env python3
"""
Drop page-cache for selected files only.

This does NOT delete files.
It asks Linux to evict clean cached pages for the selected paths by using
POSIX_FADV_DONTNEED.

Use --dry-run first.
Use --apply only after the torchrun job has completely exited.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
import time
from pathlib import Path
from typing import Iterable


def gib(n: int) -> float:
    return n / 1024 / 1024 / 1024


def cgroup_memory() -> None:
    usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    memstat = Path("/sys/fs/cgroup/memory/memory.stat")

    if usage.exists():
        try:
            value = int(usage.read_text().strip())
            print(f"cgroup usage before/after: {gib(value):.2f} GiB")
        except Exception:
            pass

    if memstat.exists():
        values = {}
        for line in memstat.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                values[parts[0]] = int(parts[1])

        if "cache" in values:
            print(f"cgroup cache before/after: {gib(values['cache']):.2f} GiB")
        if "rss" in values:
            print(f"cgroup rss before/after:   {gib(values['rss']):.2f} GiB")


def iter_regular_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen = set()

    for root in roots:
        root = root.resolve()

        if not root.exists():
            print(f"[skip] path does not exist: {root}", file=sys.stderr)
            continue

        if root.is_file():
            candidates = [root]
        else:
            candidates = (
                Path(dirpath) / name
                for dirpath, _, filenames in os.walk(root, followlinks=False)
                for name in filenames
            )

        for path in candidates:
            try:
                st = path.lstat()
            except FileNotFoundError:
                continue

            if not stat.S_ISREG(st.st_mode):
                continue

            try:
                real = path.resolve()
            except OSError:
                real = path

            key = str(real)
            if key in seen:
                continue

            seen.add(key)
            yield path


def advise_dontneed(path: Path) -> tuple[bool, int, str | None]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return False, 0, f"open failed: {exc}"

    try:
        size = os.fstat(fd).st_size

        if not hasattr(os, "posix_fadvise"):
            return False, size, "os.posix_fadvise is unavailable in this Python build"

        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True, size, None
    except OSError as exc:
        return False, 0, f"posix_fadvise failed: {exc}"
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Directory or file whose page-cache should be evicted. Repeatable.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually run POSIX_FADV_DONTNEED. Without this flag, only dry-run.",
    )
    parser.add_argument(
        "--show-largest",
        type=int,
        default=10,
        help="Show this many largest files during dry-run.",
    )
    args = parser.parse_args()

    roots = [Path(item) for item in args.root]
    files = list(iter_regular_files(roots))

    indexed = []
    total_bytes = 0

    for path in files:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        indexed.append((size, path))
        total_bytes += size

    print("=" * 80)
    print("Selected roots:")
    for root in roots:
        print(" ", root)

    print(f"Regular files found: {len(indexed):,}")
    print(f"Potential cache target: {gib(total_bytes):.2f} GiB")
    print()

    print(f"Largest {min(args.show_largest, len(indexed))} files:")
    for size, path in sorted(indexed, reverse=True)[: args.show_largest]:
        print(f"  {gib(size):8.2f} GiB  {path}")

    print()
    cgroup_memory()

    if not args.apply:
        print()
        print("Dry-run only. Re-run with --apply to request page-cache eviction.")
        return

    print()
    print("Requesting POSIX_FADV_DONTNEED on selected files...")

    ok_count = 0
    fail_count = 0
    ok_bytes = 0

    for i, (_, path) in enumerate(indexed, 1):
        ok, size, error = advise_dontneed(path)

        if ok:
            ok_count += 1
            ok_bytes += size
        else:
            fail_count += 1
            print(f"[warn] {path}: {error}", file=sys.stderr)

        if i % 500 == 0:
            print(
                f"processed {i:,}/{len(indexed):,} files; "
                f"successful target={gib(ok_bytes):.2f} GiB"
            )

    print()
    print("=" * 80)
    print(f"Finished. Successful files: {ok_count:,}")
    print(f"Failed files:             {fail_count:,}")
    print(f"Successful target bytes:  {gib(ok_bytes):.2f} GiB")
    print()

    # Give the kernel a moment to update cache accounting.
    time.sleep(3)
    cgroup_memory()
    print()
    print(
        "Note: POSIX_FADV_DONTNEED is advisory. "
        "A filesystem or kernel may retain some pages temporarily."
    )


if __name__ == "__main__":
    main()
