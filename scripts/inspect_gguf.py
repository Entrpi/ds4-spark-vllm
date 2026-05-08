#!/usr/bin/env python3
"""Inspect a GGUF file's metadata fields and tensor headers."""

import argparse
import sys

import gguf


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--limit-tensors", type=int, default=30)
    args = p.parse_args()

    r = gguf.GGUFReader(args.path)

    print("=== METADATA FIELDS ===")
    for k in sorted(r.fields.keys()):
        # Skip the verbose tokenizer arrays.
        if "token" in k.lower() or "tokenizer" in k.lower():
            continue
        f = r.fields[k]
        try:
            vals = [f.parts[i] for i in f.data]
            if len(vals) == 1 and vals[0].size == 1:
                v = vals[0].item()
            elif len(vals) == 1:
                v = f"<array len={vals[0].size}>"
            else:
                v = f"<{len(vals)} parts>"
        except Exception as ex:
            v = f"<err: {ex}>"
        print(f"  {k} = {v}")

    print()
    all_t = list(r.tensors)
    print(f"=== {len(all_t)} TENSORS ===")
    print()
    print("-- routed-expert tensors (first 6) --")
    for t in [t for t in all_t if "_exps" in t.name][:6]:
        print(f"  {t.name}  shape={list(t.shape)} "
              f"dtype={int(t.tensor_type)} bytes={t.n_bytes}")

    print()
    print("-- blk.0 ALL tensors --")
    for t in [t for t in all_t if t.name.startswith("blk.0.")][:args.limit_tensors]:
        print(f"  {t.name}  shape={list(t.shape)} dtype={int(t.tensor_type)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
