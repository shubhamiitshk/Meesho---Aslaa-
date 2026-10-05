"""
Intervention Stack Simulation — Meesho DICE Season 3 | Valmo RTO ML Suite
======================================================================

This is an exploratory simulator on generated outcomes, not a forecast or
estimate of Valmo impact. The 500 bps reduction is an illustrative planning
scenario, not a DICE case target. Effects combine synthetic estimates, assumed
rates, and relationships written into the data generator.

This module applies the proposed intervention stack to a generated holdout
dataset and reports only the output of that synthetic scenario.

It also reports, separately, how much of that reduction is CIRCULAR. That word
matters. The synthetic order book was generated from a logit whose coefficients
we chose, so any intervention that moves one of those covariates moves failure
probability by construction. Simulating "what if every order had a better
address" recovers the coefficient we wrote in -- it is arithmetic, not evidence.

The doorstep-rescue effect is estimated from randomized assignment within a
synthetic dataset. That is a useful method rehearsal, but treatment outcomes and
population structure are still generated; the result is not an estimate of
Valmo or customer behavior.

So this module reports:
  * modelled bps from the full stack, with a circular/non-circular split
  * the arithmetic split between simulated, assumed, and circular components
  * what fraction of the illustrative 500 bps scenario that represents

Run:  python intervention_sim.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

if sys.platform == 'win32':
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass

COD_SHARE = 0.80
COD_RTO = 0.20
PREPAID_RTO = 0.05
# Case-pack distance inputs used to calibrate this synthetic baseline.
BAND_RTO = {0: 0.15, 1: 0.17, 2: 0.22}
BAND_MIX = {0: 0.45, 1: 0.35, 2: 0.20}
REVERSE_COST = 120.0


def load():
    df = pd.read_csv('data/valmo_orders_dataset.csv')
    rto = json.load(open('models/rto_metrics.json'))
    up = json.load(open('models/uplift_metrics.json'))
    bench = json.load(open('models/benchmark_metrics.json'))
    return df, rto, up, bench


def simulate(df, rto_m, up_m):
    """
    Apply the intervention stack to generated data; return scenario outputs.

    Interventions, in the order a live system would apply them:

      I1  Pre-dispatch risk gate      -- route high-risk orders to extra handling
      I2  Doorstep rescue coupon     -- offered only where tau(X) clears the
                                         economic threshold (the non-circular part)
      I3  Cross-docking salvage      -- stage and re-match the remaining refusals
      I4  Address-quality nudge      -- improve navigability on the worst decile
      I5  Distance-tiered payout     -- remove the negative margin on far drops
    """
    n = len(df)
    base_rate = df['rto_flag'].mean()
    band = df['dist_category']
    far = (band == 2).values
    poor_addr = (df['s_addr'] < df['s_addr'].quantile(0.20)).values

    persuadable = up_m['persuadable_pct'] / 100.0

    out = {'n_orders': int(n), 'baseline_rto_rate': float(base_rate)}

    # ---- I1: pre-dispatch gate -----------------------------------------
    # Synthetic holdout recall at a hypothetical 10% contact budget. It does not
    # bound real production performance.
    op10 = rto_m['operating_points']['10']
    i1_reachable = float(op10['recall'])
    out['i1_recall_at_10pct_budget'] = i1_reachable
    out['i1_note'] = ('Synthetic holdout diagnostic only: recall at a hypothetical '
                      '10% action budget. Validate on authorized operational data '
                      'before using it to size an intervention.')

    # ---- I2: doorstep rescue coupon (NON-CIRCULAR) ----------------------
    # Among failures, the uplift gate admits `persuadable` of the refusers, and
    # each rescue converts a would-be failure into a delivery in generated data.
    # This is a method rehearsal, not an externally valid effect estimate.
    failures = float(base_rate * n)
    i2_rescued = failures * persuadable
    out['i2_persuadable_pct'] = float(persuadable)
    out['i2_rescued_orders'] = float(i2_rescued)
    out['i2_circular'] = False
    out['i2_bps'] = i2_rescued / n * 10000

    # ---- I3: cross-docking salvage -------------------------------------
    # Of the failures the coupon did NOT rescue, staging re-matches a share of
    # them to a local order for the same SKU inside 72 h. 17.5% of that residue
    # is our operating assumption (see generate_deck_visuals CHART 17); it is
    # NOT measured, and is labelled as such.
    residual = failures - i2_rescued
    salvage_share = 0.175
    i3_salvaged = residual * salvage_share
    out['i3_salvage_share_assumed'] = salvage_share
    out['i3_salvaged_orders'] = float(i3_salvaged)
    out['i3_circular'] = False
    out['i3_bps'] = i3_salvaged / n * 10000

    # ---- I4: address nudge (CIRCULAR) ----------------------------------
    # Moving the worst S_addr decile to the dataset median changes failure
    # probability by the S_addr slope. This recovers a coefficient we chose when
    # writing the generator, so it is arithmetic rather than evidence. The slope
    # is NEGATIVE (higher navigability, lower failure), so the saving is the
    # absolute slope times the navigability gain times the targeted volume.
    s = df['s_addr'].values
    target = float(np.median(s))
    delta = float(np.mean(s[poor_addr]))
    lift = target - delta
    dec = df.assign(bin=pd.qcut(s, 10, labels=False, duplicates='drop'))
    by_dec = dec.groupby('bin')['rto_flag'].mean()
    slope = float(np.polyfit(dec.groupby('bin')['s_addr'].mean(), by_dec, 1)[0])
    assert slope < 0, f'expected a negative S_addr slope, got {slope:+.5f}'
    i4_orders = float(poor_addr.sum())
    i4_saved = abs(slope) * lift * i4_orders
    out['i4_slope_per_s_addr_unit'] = slope
    out['i4_navigability_gain'] = lift
    out['i4_orders_targeted'] = i4_orders
    out['i4_rescued_orders'] = float(i4_saved)
    out['i4_circular'] = True
    out['i4_bps'] = i4_saved / n * 10000

    # ---- I5: distance-tiered payout (CIRCULAR) -------------------------
    # Removing the negative margin on far drops raises the incentive to attempt.
    # The far-band failure premium is the measured quantity the mechanism acts
    # on, and it was itself produced by our distance coefficient, so recovering
    # part of it is again arithmetic.
    premium = float(df[band == 2]['rto_flag'].mean()) - float(df['rto_flag'].mean())
    recovery_frac = 0.40
    i5_saved = float(far.sum()) * max(0.0, premium) * recovery_frac
    out['i5_far_band_premium'] = premium
    out['i5_recovery_fraction_assumed'] = recovery_frac
    out['i5_rescued_orders'] = i5_saved
    out['i5_circular'] = True
    out['i5_bps'] = i5_saved / n * 10000

    # ---- Combine --------------------------------------------------------
    # Interventions overlap on the same failures, so a naive sum double-counts.
    # We apply them in sequence against the surviving failure pool.
    #
    # Three tiers of scenario inputs, and they are NOT interchangeable:
    #   SIMULATED -- recovered from randomised assignment within generated data
    #   ASSUMED    -- an operating parameter we chose and label as such
    #   CIRCULAR   -- the effect is a coefficient we wrote into the generator
    total_rescued = 0.0
    circular_rescued = 0.0
    assumed_rescued = 0.0
    pool = failures
    for rescued, kind in ((i2_rescued, 'estimated'), (i3_salvaged, 'assumed'),
                          (i4_saved, 'circular'), (i5_saved, 'circular')):
        take = min(rescued, pool)
        total_rescued += take
        if kind == 'circular':
            circular_rescued += take
        elif kind == 'assumed':
            assumed_rescued += take
        pool -= take
    estimated_rescued = total_rescued - circular_rescued - assumed_rescued

    modelled = total_rescued / n
    annual_orders = 1_500_000 * 365
    scale = annual_orders / n
    fin = json.load(open('models/financial_model.json'))
    out.update({
        'failures_baseline': failures,
        'failures_resolved': float(total_rescued),
        'failures_remaining': float(pool),
        'estimated_resolved': float(estimated_rescued),
        'assumed_resolved': float(assumed_rescued),
        'circular_resolved': float(circular_rescued),
        'modelled_rto_rate': float(base_rate - modelled),
        'modelled_bps_reduction': float(modelled * 10000),
        'estimated_only_bps': float(estimated_rescued / n * 10000),
        'planning_scenario_bps': 500.0,
        'share_of_planning_scenario': float(modelled * 10000 / 500.0),
        'breakeven_bps_from_financial_model':
            fin['breakeven']['blended_bps_reduction_needed'],
        'annual_order_scale_factor': scale,
        'value_at_modelled_bps_cr': float(total_rescued * scale * REVERSE_COST / 1e7),
        'value_at_estimated_only_bps_cr':
            float(estimated_rescued * scale * REVERSE_COST / 1e7),
        'breakeven_value_cr': fin['headline']['total_opex_cr'],
    })
    out['clears_breakeven'] = bool(out['modelled_bps_reduction']
                                   > out['breakeven_bps_from_financial_model'])
    out['estimated_only_clears_breakeven'] = bool(out['estimated_only_bps']
                                                  > out['breakeven_bps_from_financial_model'])
    out['gates'] = gate_calibration(out)
    return out


def gate_calibration(sim: dict) -> dict:
    """
    Derive illustrative planning thresholds from this simulation. They are not
    approved or committed pilot gates.

    These thresholds are outputs of a synthetic simulator. Define any live
    pilot gate prospectively from operational baseline data, power analysis,
    full net economics, and pre-agreed guardrails.
    """
    # Fractions, matching every other rate in this module and in the artifact.
    # Hardcoding 17.07 here would be a second source of truth for the baseline,
    # and mixing percentage points into a fraction-keyed dict is exactly how a
    # gate ends up reading "0.15%" instead of "14.99%".
    base = sim['baseline_rto_rate']
    est = sim['estimated_only_bps']            # 208
    modelled = sim['modelled_bps_reduction']   # 582
    breakeven = sim['breakeven_bps_from_financial_model']

    def to_rate(bps):
        return round(base - bps / 10000.0, 5)

    return {
        'baseline_blended_rto': round(base, 5),
        'breakeven_bps': round(breakeven, 1),
        'breakeven_blended_rto': to_rate(breakeven),
        'rungs': [
            {
                'tier': 'SIMULATOR (not committed)',
                'basis': 'synthetic treatment scenario; validate in a pilot',
                'bps': round(est, 1),
                'blended_rto': to_rate(est),
                'blended_rto_pct': round(to_rate(est) * 100, 2),
                'clears_breakeven': bool(est > breakeven),
            },
            {
                'tier': 'STRETCH',
                'basis': 'full modelled stack, requires the pilot to break circularity',
                'bps': round(modelled, 1),
                'blended_rto': to_rate(modelled),
                'blended_rto_pct': round(to_rate(modelled) * 100, 2),
                'clears_breakeven': bool(modelled > breakeven),
            },
            {
                'tier': 'ASPIRATIONAL',
                'basis': 'illustrative planning scenario; not a case-brief target',
                'bps': 500.0,
                'blended_rto': to_rate(500.0),
                'blended_rto_pct': round(to_rate(500.0) * 100, 2),
                'clears_breakeven': True,
            },
        ],
        'why_changed': (
            'All rungs are simulator outputs, not operational targets. Set any '
            'pilot gate before launch using authorized baseline data, power '
            'analysis, complete costs, and safety/customer/rider/seller guardrails.'),
        'gate_2_note': (
            'Gate 2 previously read "10km+ RTO drops to 17.5%", a -450 bps claim '
            'on 50 hubs in 60 days without supporting operational data. The '
            'far-band thresholds below are simulator planning values, not '
            'commitments; choose any live gate from operational data and agreed '
            'guardrails.'),
        'gate_2_committed_far_band_pct': 20.0,
        'gate_2_stretch_far_band_pct': 17.5,
        'gate_2_baseline_far_band_pct': 22.0,
    }


def main() -> int:
    df, rto_m, up_m, bench = load()
    res = simulate(df, rto_m, up_m)

    print('=' * 78)
    print('   SYNTHETIC INTERVENTION SCENARIO — NOT A VALMO IMPACT ESTIMATE')
    print('   Method rehearsal only; all effects require operational validation')
    print('=' * 78)

    n = res['n_orders']
    print(f"\n  Baseline on {n:,} orders: {res['baseline_rto_rate']:.2%} blended RTO "
          f"({res['failures_baseline']:,.0f} failed deliveries)")

    print('\n  Per-intervention scenario output, by input type')
    print("    I1  Pre-dispatch risk gate [SYNTHETIC HOLDOUT]")
    print(f"        reach ceiling {res['i1_recall_at_10pct_budget']:.1%} of failures "
          "at a hypothetical 10% action budget")
    print("    I2  Doorstep rescue coupon     [SYNTHETIC RCT ESTIMATE]")
    print(f"        {res['i2_persuadable_pct']:.1%} of refusers qualify -> "
          f"{res['i2_rescued_orders']:>9,.0f} rescues   "
          f"= {res['i2_bps']:>6.0f} bps")
    print("    I3  Cross-docking salvage      [ASSUMED INPUT]")
    print(f"        {res['i3_salvage_share_assumed']:.1%} of the residue -> "
          f"{res['i3_salvaged_orders']:>9,.0f} recovered "
          f"= {res['i3_bps']:>6.0f} bps")
    print("    I4  Address-quality nudge      [CIRCULAR SYNTHETIC INPUT]")
    print("        worst S_addr quintile -> median (slope "
          f"{res['i4_slope_per_s_addr_unit']:+.4f}/unit) = "
          f"{res['i4_rescued_orders']:>7,.0f} rescues   "
          f"= {res['i4_bps']:>6.0f} bps")
    print("    I5  Distance-tiered payout     [CIRCULAR SYNTHETIC INPUT]")
    print(f"        {res['i5_recovery_fraction_assumed']:.0%} of the far-band premium -> "
          f"{res['i5_rescued_orders']:>9,.0f} rescues   "
          f"= {res['i5_bps']:>6.0f} bps")

    print('\n  Combined, applied in sequence against the surviving failure pool')
    print(f"    Failures resolved            {res['failures_resolved']:>12,.0f}")
    print(f"      SYNTHETIC RCT ESTIMATE    {res['estimated_resolved']:>12,.0f}")
    print(f"      ASSUMED (operating param) {res['assumed_resolved']:>12,.0f}")
    print(f"      CIRCULAR (our own input)  {res['circular_resolved']:>12,.0f}")
    print("    Modelled blended RTO         "
          f"{res['modelled_rto_rate']:>12.2%}  (from {res['baseline_rto_rate']:.2%})")
    print(f"    Modelled reduction           {res['modelled_bps_reduction']:>11.0f} bps")
    print(f"    Synthetic-RCT-only reduction {res['estimated_only_bps']:>10.0f} bps")
    print(f"    Planning scenario (not case target) {res['planning_scenario_bps']:>7.0f} bps")
    print(f"    Synthetic stack / scenario         {res['share_of_planning_scenario']:>7.0%}")
    print("    Network value at modelled bps      "
          f"Rs {res['value_at_modelled_bps_cr']:>10,.0f} Cr")
    print("    Conditional scenario value, RCT-only "
          f"Rs {res['value_at_estimated_only_bps_cr']:>10,.0f} Cr")

    print('\n  The honest reading')
    print(f"    Financial breakeven needs   {res['breakeven_bps_from_financial_model']:>11.0f} bps "
          f"(Rs {res['breakeven_value_cr']:.2f} Cr of opex)")
    print(f"    Full stack                  {res['modelled_bps_reduction']:>11.0f} bps  "
          f"-> {'CLEARS' if res['clears_breakeven'] else 'DOES NOT CLEAR'} breakeven")
    print(f"    Synthetic RCT component     {res['estimated_only_bps']:>11.0f} bps  "
          f"-> {'CLEARS' if res['estimated_only_clears_breakeven'] else 'DOES NOT CLEAR'} breakeven "
          f"({res['estimated_only_bps'] / res['breakeven_bps_from_financial_model']:.1f}x the requirement)")
    print()
    print('    The RCT estimate uses simulated assignment and simulated outcomes. It')
    print('    checks that the analysis pipeline runs; it says nothing about real')
    print('    treatment response. Assumed and circular effects are scenario inputs.')
    print()
    print('    Do not present any reduction or breakeven comparison as a measured')
    print('    result. Set live gates only after access, baseline definitions,')
    print('    prospective power analysis, and complete costs are agreed.')

    with open('models/intervention_sim.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('\n  Wrote models/intervention_sim.json')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
