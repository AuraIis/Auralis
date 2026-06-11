#!/usr/bin/env python3
"""Deterministic stride sampler: every Nth line so the sample spans the whole file."""
import sys, os
src, out, n_target = sys.argv[1], sys.argv[2], int(sys.argv[3])
size = os.path.getsize(src)
# estimate avg line length from first 50MB
tot = cnt = 0
with open(src, 'rb') as f:
    for line in f:
        tot += len(line); cnt += 1
        if tot > 50_000_000: break
est_lines = max(1, int(size / (tot / max(1, cnt))))
stride = max(1, est_lines // n_target)
kept = i = 0
with open(src, 'rb') as f, open(out, 'wb') as o:
    for line in f:
        if i % stride == 0:
            o.write(line); kept += 1
            if kept >= n_target: break
        i += 1
print(f"{os.path.basename(src)}: est_lines={est_lines:,} stride={stride} kept={kept}")
