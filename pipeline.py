"""
Cross-Platform Pipeline Driver — Meesho DICE Season 3 | Valmo RTO ML Suite
===========================================================================

This is a legacy prototype rebuild driver, not the submission-PDF builder. It
regenerates synthetic data and models and includes verification stages. Review
README.md's evidence boundary before using any resulting metrics:

    python pipeline.py all        full rebuild: data -> models -> bench -> charts -> claims
    python pipeline.py test       run the verification suite
    python pipeline.py preflight  everything to check before stepping on stage
    python pipeline.py serve      host presentation_slides.html on :8000

Every stage is incremental: it re-runs only if its outputs are missing or older
than its inputs. The full pipeline is not needed to review or export the current
HTML submission.

Ordering matters and is enforced here. Data must precede models, models must
precede the benchmark (it loads the artifacts), the benchmark must precede the
claims ledger (latency is a live measurement), and the charts must follow the
ledger because every figure is read out of it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# stage -> (command, [outputs that must exist afterwards], [inputs that trigger a rerun])
STAGES = {
    'data': (
        [sys.executable, 'generate_valmo_dataset.py'],
        ['data/valmo_orders_dataset.csv'],
        ['generate_valmo_dataset.py'],
    ),
    'models': (
        [sys.executable, 'train_valmo_models.py'],
        ['models/rto_metrics.json', 'models/uplift_metrics.json', 'models/telemetry_metrics.json'],
        ['train_valmo_models.py', 'metrics_lib.py'],
    ),
    'bench': (
        [sys.executable, 'benchmark_serving.py', '--runs', '2000'],
        ['models/benchmark_metrics.json'],
        ['benchmark_serving.py', 'predict.py'],
    ),
    # Extracts dated, marketplace-wide figures from Meesho's NSE-filed
    # prospectus. These are public context, not Valmo-specific RTO outcomes.
    'disclosure': (
        [sys.executable, 'meesho_disclosure.py'],
        ['models/meesho_disclosure.json'],
        ['meesho_disclosure.py'],
    ),
    'finance': (
        [sys.executable, 'financial_model.py'],
        ['models/financial_model.json'],
        # benchmark_metrics.json is an input because cost_of_ml() derives the
        # ML duty cycle from the measured P99: a re-bench without a re-finance
        # leaves models/financial_model.json quoting stale compute-hours while
        # the claims ledger (built live) quotes fresh ones.
        ['financial_model.py', 'models/rto_metrics.json', 'models/uplift_metrics.json',
         'models/benchmark_metrics.json'],
    ),
    'simulate': (
        [sys.executable, 'intervention_sim.py'],
        ['models/intervention_sim.json'],
        ['intervention_sim.py', 'models/rto_metrics.json',
         'models/uplift_metrics.json', 'models/financial_model.json',
         'data/valmo_orders_dataset.csv'],
    ),
    'ceiling': (
        [sys.executable, 'signal_ceiling.py'],
        ['models/signal_ceiling.json'],
        ['signal_ceiling.py', 'data/valmo_orders_dataset.csv',
         'data/test_evaluation_preds.csv'],
    ),
    'provenance': (
        [sys.executable, 'dataset_provenance.py'],
        ['models/dataset_provenance.json'],
        ['dataset_provenance.py', 'data/valmo_orders_dataset.csv'],
    ),
    'claims': (
        [sys.executable, 'claims.py'],
        ['models/claims.json'],
        ['claims.py', 'models/benchmark_metrics.json', 'models/rto_metrics.json',
         'models/uplift_metrics.json', 'models/telemetry_metrics.json',
         'models/dataset_summary.json', 'models/financial_model.json'],
    ),
    'charts': (
        [sys.executable, 'generate_deck_visuals.py'],
        ['charts/operating_point_lift.png', 'charts/qini_uplift_curve.png'],
        ['generate_deck_visuals.py'],
    ),
    'verify': (
        [sys.executable, 'claims.py', '--verify'],
        [],
        [],
    ),
}

# Stages run in dependency order. `all` stops at the first failure.
PIPELINE = ['data', 'models', 'ceiling', 'provenance', 'bench', 'disclosure', 'finance', 'simulate',
            'claims', 'charts', 'verify']


def _p(rel: str) -> str:
    return os.path.join(ROOT, rel)


def _newest(paths) -> float:
    """Newest mtime among existing paths; 0.0 if none exist."""
    best = 0.0
    for rel in paths:
        fp = _p(rel)
        if os.path.exists(fp):
            best = max(best, os.path.getmtime(fp))
    return best


def _needs_run(stage: str) -> bool:
    _, outputs, inputs = STAGES[stage]
    if not outputs:
        return True
    for out in outputs:
        if not os.path.exists(_p(out)):
            return True
    # A stage also re-runs when any of its declared inputs is newer than its
    # oldest output, which is what makes an edit to metrics_lib.py propagate.
    return _newest(inputs) > min(os.path.getmtime(_p(o)) for o in outputs
                                 if os.path.exists(_p(o)))


def run_stage(stage: str, force: bool = False) -> int:
    cmd, outputs, _ = STAGES[stage]
    if not force and not _needs_run(stage):
        print(f'  [skip] {stage:<8} outputs up to date')
        return 0
    print(f'  [run ] {stage:<8} {" ".join(os.path.basename(c) for c in cmd)}')
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f'  [FAIL] {stage:<8} exit {r.returncode} after {dt:.1f}s')
        return r.returncode
    missing = [o for o in outputs if not os.path.exists(_p(o))]
    if missing:
        print(f'  [FAIL] {stage:<8} did not produce: {", ".join(missing)}')
        return 1
    print(f'  [ok  ] {stage:<8} {dt:.1f}s')
    return 0


def cmd_all(force: bool) -> int:
    print('=' * 78)
    print('   VALMO ML PIPELINE — FULL REBUILD')
    print('=' * 78)
    t0 = time.time()
    for stage in PIPELINE:
        rc = run_stage(stage, force=force)
        if rc != 0:
            print(f'\nPipeline stopped at stage "{stage}".')
            return rc
    print('\n' + '=' * 78)
    print(f'   PIPELINE COMPLETE IN {time.time() - t0:.1f}s')
    print('   Technical models, simulations, and charts successfully verified.')
    print('=' * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='Valmo ML pipeline driver')
    ap.add_argument('target', nargs='?', default='all',
                    choices=list(STAGES) + ['all'],
                    help='stage to run, or "all" for the full pipeline')
    ap.add_argument('--force', action='store_true',
                    help='re-run stages even if their outputs are up to date')
    args = ap.parse_args()

    if args.target == 'all':
        return cmd_all(args.force)

    print(f'Running stage: {args.target}')
    return run_stage(args.target, force=args.force)


if __name__ == '__main__':
    raise SystemExit(main())
