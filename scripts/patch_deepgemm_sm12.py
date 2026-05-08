#!/usr/bin/env python3
"""Best-effort SM12x alias patch for jasl/DeepGEMM.

Walks every .hpp / .hh file under csrc/ and rewrites every occurrence of
the literal `arch_major == 10` to `(arch_major == 10 || arch_major == 12)`.
This makes SM12x (Blackwell consumer / GB10 / DGX Spark) take the SM100
dispatch path. SM100 kernels often run on SM12x because both are Blackwell
ISA, but TMA / cluster / tcgen05 features may differ — silent miscomputes
or runtime errors are possible. This is an experiment to see whether
DSv4-Flash forwards at all on SM121 with this alias.

Idempotent: skips files where the alias is already present.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERN = re.compile(r"arch_major\s*==\s*10")
REPLACEMENT = "(arch_major == 10 or arch_major == 12)"
# C++ has `||` not `or`; Python's `re` doesn't care, we just need the right
# substitution.
REPLACEMENT_CXX = "(arch_major == 10 || arch_major == 12)"


def patch_file(path: Path) -> int:
    src = path.read_text()
    # Idempotency: don't rewrite if our alias is already present.
    if "arch_major == 12" in src:
        return 0
    new = PATTERN.sub(REPLACEMENT_CXX, src)
    if new == src:
        return 0
    path.write_text(new)
    return src.count("arch_major == 10") - new.count("arch_major == 10") + 0  # always at least 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/ent/extras/DeepGEMM/csrc",
                   help="Root of DeepGEMM csrc/ tree")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"DeepGEMM source root not found: {root}")

    total_files = 0
    total_sub = 0
    for ext in ("*.hpp", "*.hh", "*.cpp", "*.cu", "*.h"):
        for f in root.rglob(ext):
            n = patch_file(f)
            if n:
                total_files += 1
                total_sub += n
                print(f"  patched {f.relative_to(root)}")

    print(f"\nDone. {total_sub} substitutions across {total_files} files.")
    if total_sub == 0:
        print("(no changes — alias may already be applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
