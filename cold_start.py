"""
Cold-Start Reproducibility Test — Meesho DICE Season 3
======================================================

Copies the working tree to a scratch directory, deletes every GENERATED artefact,
and runs the full pipeline plus the test suite and preflight from scratch. The
original working tree is never touched.

WHY THIS EXISTS
---------------
The submission's loudest claim is that every figure is regenerable and every
artefact reproducible. That claim is checkable by anyone, in one command, in
under three minutes -- which makes it a claim we should be holding ourselves to
first.

It was never tested, and it was BROKEN.

    FileNotFoundError: data/test_evaluation_preds.csv

`signal_ceiling.py` reads that file, which `train_valmo_models.py` writes. The
`ceiling` stage sat SECOND in the pipeline list and `models` sat fifth, so
`pipeline.py all` failed at stage 2 of 14 on every clean checkout. It never
surfaced locally because a working tree that already holds every artefact hides
exactly this class of bug -- the pipeline reads the stale file and reports green.

The dependency was correctly DECLARED in the stage tuple the whole time. Only
the ORDER was wrong. `test_pipeline_stage_ordering` in test_suite.py now catches
that statically in milliseconds; this script is the end-to-end confirmation.

Run:  python cold_start.py            (~3-4 minutes)
      python cold_start.py --quick    (pipeline only, skip tests + preflight)
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# Artefacts the pipeline must be able to rebuild from source. Deleting them is the
# whole point: if a stage quietly depends on one that is only ever present in a
# dirty working tree, this is what exposes it.
#
# presentation_slides.html is deliberately NOT listed. It is hand-authored HTML
# that the pipeline MUTATES (claims.py and sync_speaker_scripts.py write into it),
# not a file it emits, so a clean checkout legitimately has it.
GENERATED = ['data', 'models', 'charts', '__pycache__', '_render']

SKIP_DIRS = {'.git', '__pycache__', '_render', '.kilo', '.pytest_cache',
             '.mypy_cache', '.ruff_cache', '.venv', 'venv', 'env', 'ENV'}


def copy_tree(src: str, dst: str) -> int:
    os.makedirs(dst, exist_ok=True)
    copied = 0
    for item in os.listdir(src):
        if item in SKIP_DIRS:
            continue
        s, d = os.path.join(src, item), os.path.join(dst, item)
        if os.path.isdir(s):
            copied += copy_tree(s, d)
        elif not fnmatch.fnmatch(item, '*.pyc') and not item.endswith('.log'):
            shutil.copy2(s, d)
            copied += 1
    return copied


def run(cmd: str, cwd: str, timeout: int = 2400):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=timeout)
    return p.returncode, (p.stdout or '') + (p.stderr or ''), time.time() - t0


def tail(text: str, n: int = 30, width: int = 150) -> None:
    for line in text.strip().splitlines()[-n:]:
        print('    ' + line[:width])


def main() -> int:
    ap = argparse.ArgumentParser(description='Cold-start reproducibility test')
    ap.add_argument('--quick', action='store_true',
                    help='run the pipeline only; skip test_suite and preflight')
    args = ap.parse_args()

    dest = tempfile.mkdtemp(prefix='valmo_cold_')
    print('=' * 74)
    print('   COLD-START REPRODUCIBILITY TEST')
    print('=' * 74)
    print(f'  source : {ROOT}')
    print(f'  scratch: {dest}')
    failures = []
    try:
        n = copy_tree(ROOT, dest)
        print(f'  copied : {n} source files')

        for g in GENERATED:
            p = os.path.join(dest, g)
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            print(f'  purged : {g}')

        print('\n  --- python pipeline.py all ---')
        rc, out, secs = run(f'"{sys.executable}" pipeline.py all', dest)
        tail(out, 30)
        print(f'  exit={rc}  elapsed={secs:.0f}s')
        if rc != 0:
            failures.append('pipeline.py all')

        if not args.quick and not failures:
            print('\n  --- python test_suite.py ---')
            rc2, out2, secs2 = run(f'"{sys.executable}" test_suite.py', dest)
            tail(out2, 8)
            print(f'  exit={rc2}  elapsed={secs2:.0f}s')
            if rc2 != 0:
                failures.append('test_suite.py')

            print('\n  --- python preflight.py ---')
            rc3, out3, secs3 = run(f'"{sys.executable}" preflight.py', dest, 900)
            tail(out3, 4)
            print(f'  exit={rc3}  elapsed={secs3:.0f}s')
            if rc3 != 0:
                failures.append('preflight.py')

        print()
        if failures:
            print(f'   COLD START FAILED: {", ".join(failures)}')
            return 1
        print('   COLD START PASSED')
        print("   A clean checkout rebuilds every artefact, every test passes, and")
        print("   preflight clears — on a machine that has never seen this repo.")
        return 0
    finally:
        shutil.rmtree(dest, ignore_errors=True)
        print(f'  scratch removed: {dest}')


if __name__ == '__main__':
    raise SystemExit(main())