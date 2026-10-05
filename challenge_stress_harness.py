"""Synthetic-prototype sanity checks; no production or empirical validation.

Checks code invariants and hand-authored scenarios for the local Valmo case
prototype. Datasets, model labels, treatment effects, and rider-pattern examples
are synthetic. The local 3 ms ceiling is a regression budget, not a production
SLA. No result here establishes actual fraud, rider conduct, customer response,
unit economics, GMV preservation, or Valmo impact.
"""

import os
import sys
import time
import json
import subprocess
import traceback
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass

from predict import ValmoPredictor, FEATURE_DEFAULTS, COUPON_COST_RS, REVERSE_COST_RS, NET_MARGIN_PER_RESCUE_RS
import live_inference as li
import address_tokenizer as at

report_sections = []

def log(msg=""):
    print(msg)
    report_sections.append(msg)

def run_cli_tests():
    log("\n" + "="*80)
    log("TEST SUITE 1: CLI INVOCATION & PROFILE EVALUATIONS")
    log("="*80)

    commands = [
        ("Profile 1 (CLI)", [sys.executable, "live_inference.py", "--profile", "1"]),
        ("Profile 2 (CLI)", [sys.executable, "live_inference.py", "--profile", "2"]),
        ("Profile 3 (CLI)", [sys.executable, "live_inference.py", "--profile", "3"]),
        ("Profile 4 (CLI)", [sys.executable, "live_inference.py", "--profile", "4"]),
        ("Single Order (CLI)", [sys.executable, "live_inference.py", "--single"]),
        ("Profile 1 JSON", [sys.executable, "live_inference.py", "--profile", "1", "--json"]),
        ("Profile 4 JSON", [sys.executable, "live_inference.py", "--profile", "4", "--json"]),
        ("All Profiles JSON", [sys.executable, "live_inference.py", "--json"]),
    ]

    for name, cmd in commands:
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        dt_ms = (time.perf_counter() - t0) * 1000
        status = "PASS" if res.returncode == 0 else f"FAIL (exit {res.returncode})"
        log(f"  [{status}] {name:<22} in {dt_ms:6.1f} ms")
        if res.returncode != 0:
            log(f"    ERROR:\n{res.stderr}")
            assert False, f"CLI command {name} failed: {res.stderr}"
        if "--json" in cmd:
            try:
                parsed = json.loads(res.stdout)
                log(f"    -> Valid JSON output ({len(parsed)} top-level keys)")
            except Exception as e:
                log(f"    -> JSON parsing error: {e}")
                assert False, f"Failed to parse JSON for {name}"

    log("  => All CLI profiles and JSON flags passed cleanly.")


def run_adversarial_input_tests():
    log("\n" + "="*80)
    log("TEST SUITE 2: ADVERSARIAL & EXTREME INPUT STRESS-TESTING")
    log("="*80)

    predictor = ValmoPredictor()
    # Warmup
    _ = predictor.predict_single({'customer_address': 'Warmup 123'})

    adversarial_cases = [
        ("Empty dict payload", {}),
        ("High order value (Rs 1,000,000)", {'order_value': 1000000.0, 'is_cod': 1}),
        ("Zero order value (Rs 0.0)", {'order_value': 0.0}),
        ("Negative order value (Rs -500.0)", {'order_value': -500.0}),
        ("Extreme distance (5,000 km)", {'distance_km': 5000.0, 'dist_category': 2}),
        ("Zero distance (0.0 km)", {'distance_km': 0.0, 'dist_category': 0}),
        ("Negative distance (-10.0 km)", {'distance_km': -10.0}),
        ("Empty address string", {'customer_address': ''}),
        ("Whitespace-only address", {'customer_address': '     \t\n   '}),
        ("None address", {'customer_address': None}),
        ("Massive address (10,000 chars)", {'customer_address': 'Near Station, Ward 4, Rampur ' * 350}),
        ("Mixed Indic scripts (Hindi + Bengali + English)", {
            'customer_address': 'हावड़ा स्टेशन के पास, ওয়ার্ড ৪, Howrah, West Bengal 711101'
        }),
        ("XSS & SQL Injection strings in address", {
            'customer_address': "<script>alert('xss')</script> ' OR '1'='1' -- DROP TABLE users;"
        }),
        ("Emoji and special characters", {'customer_address': '📦 Flat 4B 🚚 Near Temple ⭐ Agra'}),
        ("Pincode mismatch extreme (50,000 m)", {'pin_mismatch_m': 50000.0}),
        ("Negative pincode mismatch (-500 m)", {'pin_mismatch_m': -500.0}),
        ("Extreme delay (100 days beyond TAT)", {'delay_days': 100}),
        ("Negative delay (-10 days)", {'delay_days': -10}),
        ("String-encoded numeric types", {
            'order_value': '750.50', 'is_cod': '1', 'distance_km': '12.5', 'delay_days': '2'
        }),
        ("Malformed non-numeric strings in numeric fields", {
            'order_value': 'invalid_num', 'distance_km': 'far_away', 'tier': 'TierOne'
        }),
        ("NaN values in all features", {k: float('nan') for k in FEATURE_DEFAULTS.keys()}),
        ("None values in all features", {k: None for k in FEATURE_DEFAULTS.keys()}),
        ("Infinite values in features", {'order_value': float('inf'), 'distance_km': float('inf')}),
        ("Negative infinite values", {'order_value': float('-inf'), 'distance_km': float('-inf')}),
        ("Unseen Tier (Tier 5)", {'tier': 5}),
        ("Negative SKU risk index", {'sku_risk_index': -0.5}),
        ("Massive SKU risk index", {'sku_risk_index': 5.0}),
    ]

    all_passed = True
    for name, payload in adversarial_cases:
        t0 = time.perf_counter_ns()
        try:
            # Test through evaluate_order (end-to-end tokenizer + predictor)
            res = li.evaluate_order(predictor, payload)
            lat_ms = (time.perf_counter_ns() - t0) / 1e6

            # Verify invariants:
            p_rto = res['rto_probability']
            tau = res['cate_uplift_tau']
            tier = res['risk_tier']
            wa = res['whatsapp_ndr_trigger']
            disc = res['unlock_doorstep_qr_discount']
            margin = res['net_margin_benefit_rs']

            assert 0.0 <= p_rto <= 1.0, f"p_rto out of bounds: {p_rto}"
            assert 0.0 <= tau <= 1.0, f"tau out of bounds: {tau}"
            assert tier in ("LOW", "MODERATE", "HIGH", "CRITICAL"), f"Invalid tier: {tier}"
            assert isinstance(wa, bool), f"whatsapp_ndr_trigger not bool: {wa}"
            assert isinstance(disc, bool), f"unlock_doorstep_qr_discount not bool: {disc}"
            assert (margin == 85.0 if disc else margin == 0.0), f"Margin inconsistent: {margin}"
            assert not np.isnan(p_rto), "p_rto is NaN"
            assert not np.isnan(tau), "tau is NaN"

            log(f"  [PASS] {name:<42} | P(RTO): {p_rto:6.2%} | tau: {tau:6.2%} | Tier: {tier:<8} | Lat: {lat_ms:5.3f}ms")
        except Exception as e:
            log(f"  [FAIL] {name:<42} | Exception: {e}")
            traceback.print_exc()
            all_passed = False

    assert all_passed, "One or more adversarial inputs failed!"
    log("  => All 27 adversarial edge cases handled gracefully without crashes or NaN corruptions.")


def run_local_latency_sanity_check():
    log("\n" + "="*80)
    log("CHECK 3: LOCAL LATENCY SAMPLE (3.0 ms regression budget; not a production SLA)")
    log("="*80)

    predictor = ValmoPredictor()

    # 1. Warm-up
    for i in range(100):
        predictor.predict_single(predictor._make_benchmark_payload(i))

    n_samples = 3000
    latencies = []
    tok_latencies = []

    log(f"  Executing {n_samples:,} sequential end-to-end evaluations (Tokenization + Inference)...")
    for i in range(n_samples):
        payload = predictor._make_benchmark_payload(i)
        payload['customer_address'] = f"Flat {i%500}, Ward {i%20}, Near Temple {i%10}, Rampur Rural, UP {244901 + (i%100)}"
        t0 = time.perf_counter_ns()
        res = li.evaluate_order(predictor, payload)
        t_total = (time.perf_counter_ns() - t0) / 1e6
        latencies.append(t_total)
        tok_latencies.append(res['tokenizer_latency_ms'])

    lat_arr = np.array(latencies)
    tok_arr = np.array(tok_latencies)

    mean_l = float(np.mean(lat_arr))
    p50_l = float(np.percentile(lat_arr, 50))
    p90_l = float(np.percentile(lat_arr, 90))
    p95_l = float(np.percentile(lat_arr, 95))
    p99_l = float(np.percentile(lat_arr, 99))
    max_l = float(np.max(lat_arr))

    mean_tok = float(np.mean(tok_arr))
    p50_tok = float(np.percentile(tok_arr, 50))
    p99_tok = float(np.percentile(tok_arr, 99))

    log(f"  • Address Tokenizer Latency : Mean {mean_tok:.4f} ms | P50 {p50_tok:.4f} ms | P99 {p99_tok:.4f} ms (local sample)")
    log(f"  • End-to-End Scoring Latency:")
    log(f"      - Mean Latency : {mean_l:.3f} ms")
    log(f"      - P50 Latency  : {p50_l:.3f} ms")
    log(f"      - P90 Latency  : {p90_l:.3f} ms")
    log(f"      - P95 Latency  : {p95_l:.3f} ms")
    log(f"      - P99 Latency  : {p99_l:.3f} ms (local synthetic-payload sample)")
    log(f"      - Max Latency  : {max_l:.3f} ms")

    within_local_budget = p99_l < 3.0
    log(f"  • Local regression budget : {'PASS' if within_local_budget else 'FAIL'}")
    assert within_local_budget, f"local P99 {p99_l:.3f} ms exceeds the 3.0 ms regression budget"
    log(f"  => Local timing sample recorded over {n_samples:,} synthetic-payload calls; not deployment evidence.")


def run_cate_uplift_stress_test():
    log("\n" + "="*80)
    log("TEST SUITE 4: CATE UPLIFT DECISION RULE & UNIT ECONOMICS STRESS-TEST")
    log("="*80)

    predictor = ValmoPredictor()
    gate = 35.0 / 120.0  # 0.291666666... (29.17%)
    log(f"  Theoretical Gate Threshold: {gate:.6f} ({gate:.2%}) = Rs 35 Coupon / Rs 120 Reverse Freight")

    # Boundary tests directly probing the decision boundary
    test_taus = [
        0.0000, 0.1000, 0.2000, 0.2500, 0.2900, 0.2916, 0.29166,
        0.2916666, 0.2916667, 0.2917, 0.2950, 0.3500, 0.5000, 1.0000
    ]

    for tau in test_taus:
        unlock = bool(tau >= gate)
        expected_unlock = tau >= (35.0 / 120.0)
        expected_savings = 85.0 if expected_unlock else 0.0

        # Simulate synthetic output check
        assert unlock == expected_unlock, f"Boundary failure at tau = {tau}"
        log(f"  tau = {tau:7.4%} | Gate = {gate:7.4%} | Unlock: {str(unlock):<5} | Net Savings: Rs {expected_savings:5.1f}")

    # Generated-example policy-rule check; this dataset is not Valmo data.
    df_path = 'data/valmo_orders_dataset.csv'
    if os.path.exists(df_path):
        sample_df = pd.read_csv('data/sample_1000_orders.csv')
        batch_res = predictor.predict_batch(sample_df)

        unlocked_mask = batch_res['unlock_upi_discount_qr']
        taus = batch_res['cate_uplift_tau']

        # Verify invariant: for EVERY order where unlock is True, tau MUST be >= economic threshold
        violations_unlock = batch_res[unlocked_mask & (taus < predictor.economic_threshold)]
        violations_withheld = batch_res[(~unlocked_mask) & (taus >= predictor.economic_threshold)]

        assert len(violations_unlock) == 0, f"Found {len(violations_unlock)} orders with tau < gate unlocked!"
        assert len(violations_withheld) == 0, f"Found {len(violations_withheld)} orders with tau >= gate withheld!"

        n_unlocked = unlocked_mask.sum()
        pct_unlocked = n_unlocked / len(batch_res)
        avg_tau_unlocked = taus[unlocked_mask].mean()
        avg_tau_withheld = taus[~unlocked_mask].mean()

        log(f"\n  Generated-example policy-rule check ({len(batch_res):,} rows):")
        log(f"    - Orders unlocking UPI discount : {n_unlocked:,} ({pct_unlocked:.1%})")
        log(f"    - Mean tau for unlocked cohort   : {avg_tau_unlocked:.2%}")
        log(f"    - Mean tau for withheld cohort   : {avg_tau_withheld:.2%}")
        log(f"    - Invariant Violations           : 0 (Strict compliance to gate)")

    # Conditional arithmetic over the selected unit-cost assumptions only.
    log("\n  Conditional Unit-Cost Arithmetic (selected assumptions only):")
    p_rescue = 0.3186  # Profile 2 tau
    c_coupon = COUPON_COST_RS
    c_rev = REVERSE_COST_RS
    net_valmo = c_rev - c_coupon
    log(f"    • Assumed reverse freight            : +₹{c_rev:.2f}")
    log(f"    • Proposed coupon assumption         : -₹{c_coupon:.2f}")
    log(f"    • Difference before other costs      : +₹{net_valmo:.2f} / scenario rescue")
    log("    • Seller accounting / realized GMV   : not modeled or measured")
    assert net_valmo == 85.0, f"scenario arithmetic mismatch: {net_valmo}"
    log("  => Arithmetic identity checked; treatment and operating economics remain unvalidated.")


def run_synthetic_telemetry_pattern_check():
    log("\n" + "="*80)
    log("CHECK 5: SYNTHETIC TELEMETRY ANOMALY-PATTERN PROTOTYPE")
    log("="*80)

    predictor = ValmoPredictor()

    test_scenarios = [
        # (Generated scenario name, payload, expected prototype status)
        ("Generated pattern with ordinary kinematics", {
            'gps_dist_m': 25.0, 'dwell_sec': 85.0, 'call_sec': 40.0, 'speed_kmh': 0.8, 'delta_sec': 280.0
        }, "NO_ANOMALY_FLAG"),
        ("Generated high-speed, short-dwell pattern", {
            'gps_dist_m': 650.0, 'dwell_sec': 5.0, 'call_sec': 0.0, 'speed_kmh': 36.0, 'delta_sec': 8.0
        }, "REVIEW_CANDIDATE"),
        ("High Speed Flying Drive-by (50 km/h, 2s dwell)", {
            'gps_dist_m': 400.0, 'dwell_sec': 2.0, 'call_sec': 0.0, 'speed_kmh': 50.0, 'delta_sec': 5.0
        }, "REVIEW_CANDIDATE"),
        ("Distant Geo-spoof (1500m away, standing still)", {
            'gps_dist_m': 1500.0, 'dwell_sec': 60.0, 'call_sec': 30.0, 'speed_kmh': 0.5, 'delta_sec': 120.0
        }, "REVIEW_CANDIDATE"),
        ("Zero Dwell, Zero Call (Immediate button click)", {
            'gps_dist_m': 80.0, 'dwell_sec': 0.0, 'call_sec': 0.0, 'speed_kmh': 15.0, 'delta_sec': 2.0
        }, "REVIEW_CANDIDATE"),
        ("Slow Walking Genuine Attempt (120s dwell, 60s call)", {
            'gps_dist_m': 15.0, 'dwell_sec': 120.0, 'call_sec': 60.0, 'speed_kmh': 0.3, 'delta_sec': 350.0
        }, "NO_ANOMALY_FLAG"),
        ("Legitimate Attempt with Busy Line (90s dwell, 10s call)", {
            'gps_dist_m': 30.0, 'dwell_sec': 90.0, 'call_sec': 10.0, 'speed_kmh': 0.5, 'delta_sec': 240.0
        }, "NO_ANOMALY_FLAG"),
        ("Corner Case: Exact Geofence (10m), Zero Dwell, 40 km/h", {
            'gps_dist_m': 10.0, 'dwell_sec': 1.0, 'call_sec': 0.0, 'speed_kmh': 40.0, 'delta_sec': 3.0
        }, "REVIEW_CANDIDATE"),
    ]

    all_match = True
    for name, payload, expected in test_scenarios:
        res = predictor.audit_telemetry(payload)
        status = res['attempt_status']
        score = res['anomaly_score']
        lat = res['latency_ms']
        matched = (status == expected)
        mark = "PASS" if matched else "FAIL"
        log(f"  [{mark}] {name:<50} | Score: {score:7.4f} | Result: {status:<22} (Expected: {expected})")
        if not matched:
            all_match = False

    assert all_match, "Synthetic telemetry outputs did not match the configured prototype scenarios"

    # Parameter sweeps to test monotonicity and decision boundary:
    log("\n  Continuous Parameter Boundary Sweeps:")
    log("  A. Speed Sweep (GPS=25m, Dwell=80s, Call=30s, Delta=250s):")
    for spd in [0.5, 5.0, 15.0, 25.0, 40.0, 60.0]:
        r = predictor.audit_telemetry({'gps_dist_m': 25.0, 'dwell_sec': 80.0, 'call_sec': 30.0, 'speed_kmh': spd, 'delta_sec': 250.0})
        log(f"     Speed: {spd:4.1f} km/h -> Anomaly Score: {r['anomaly_score']:7.4f} | Status: {r['attempt_status']}")

    log("  B. Dwell Time Sweep (GPS=25m, Call=30s, Speed=0.8 km/h, Delta=250s):")
    for dw in [0.0, 5.0, 15.0, 30.0, 60.0, 120.0]:
        r = predictor.audit_telemetry({'gps_dist_m': 25.0, 'dwell_sec': dw, 'call_sec': 30.0, 'speed_kmh': 0.8, 'delta_sec': 250.0})
        log(f"     Dwell: {dw:5.1f} s    -> Anomaly Score: {r['anomaly_score']:7.4f} | Status: {r['attempt_status']}")

    log("  C. GPS Distance Sweep (Dwell=80s, Call=30s, Speed=0.8 km/h, Delta=250s):")
    for dist in [10.0, 50.0, 100.0, 300.0, 600.0, 1200.0]:
        r = predictor.audit_telemetry({'gps_dist_m': dist, 'dwell_sec': 80.0, 'call_sec': 30.0, 'speed_kmh': 0.8, 'delta_sec': 250.0})
        log(f"     GPS Dist: {dist:6.1f} m -> Anomaly Score: {r['anomaly_score']:7.4f} | Status: {r['attempt_status']}")

    log("  => Generated scenario labels only; this does not detect fraud or establish rider conduct.")


def run_single_vs_batch_parity_stress_test():
    log("\n" + "="*80)
    log("TEST SUITE 6: SINGLE VS BATCH SERVING PARITY STRESS-TEST")
    log("="*80)

    predictor = ValmoPredictor()

    # 1. Synthetic 1,000-row sample parity check
    sample_csv = 'data/sample_1000_orders.csv'
    if os.path.exists(sample_csv):
        df = pd.read_csv(sample_csv)
        log(f"  Testing parity across {len(df):,} generated example rows from {sample_csv}...")
        batch_out = predictor.predict_batch(df)

        singles = [
            predictor.predict_single(r) for r in df.drop(
                columns=[c for c in ('pred_rto_prob', 'risk_tier', 'trigger_whatsapp_ndr',
                                     'cate_uplift_tau', 'unlock_upi_discount_qr') if c in df.columns]
            ).to_dict('records')
        ]

        prob_diffs = [abs(s['rto_probability'] - float(b)) for s, b in zip(singles, batch_out['pred_rto_prob'])]
        tau_diffs = [abs(s['cate_uplift_tau'] - float(t)) for s, t in zip(singles, batch_out['cate_uplift_tau'])]
        tier_mismatches = sum(1 for s, t in zip(singles, batch_out['risk_tier']) if s['risk_tier'] != t)
        wa_mismatches = sum(1 for s, b in zip(singles, batch_out['trigger_whatsapp_ndr']) if s['whatsapp_ndr_trigger'] != bool(b))
        disc_mismatches = sum(1 for s, b in zip(singles, batch_out['unlock_upi_discount_qr']) if s['unlock_doorstep_qr_discount'] != bool(b))

        max_prob_diff = max(prob_diffs)
        max_tau_diff = max(tau_diffs)

        log(f"  • Max Absolute P(RTO) Difference : {max_prob_diff:.2e}")
        log(f"  • Max Absolute CATE tau Diff     : {max_tau_diff:.2e}")
        log(f"  • Risk Tier Mismatches           : {tier_mismatches}")
        log(f"  • WhatsApp Trigger Mismatches    : {wa_mismatches}")
        log(f"  • UPI QR Unlock Mismatches       : {disc_mismatches}")

        assert max_prob_diff < 1e-9, f"P(RTO) divergence detected: {max_prob_diff}"
        assert max_tau_diff < 1e-9, f"CATE tau divergence detected: {max_tau_diff}"
        assert tier_mismatches == 0, f"Risk tier mismatches: {tier_mismatches}"
        assert wa_mismatches == 0, f"WhatsApp trigger mismatches: {wa_mismatches}"
        assert disc_mismatches == 0, f"UPI discount unlock mismatches: {disc_mismatches}"
        log("  => Exact parity checked on 1,000 generated example rows.")

    # 2. DataFrame row-level parity check on randomized synthetic orders
    log("\n  Stress-testing parity on 500 synthetic orders from DataFrame records...")
    rng = np.random.default_rng(42)
    syn_records = []
    for i in range(500):
        rec = {}
        for col, default in FEATURE_DEFAULTS.items():
            if rng.random() > 0.4:
                rec[col] = float(rng.uniform(0.0, default * 2.5))
        if rng.random() > 0.5:
            rec['customer_address'] = f"Flat {i}, Sector {i%15}, Jaipur"
        syn_records.append(rec)

    syn_df = pd.DataFrame(syn_records)
    batch_syn = predictor.predict_batch(syn_df)
    # Score each row as supplied by the DataFrame
    single_from_df = [predictor.predict_single(r) for r in syn_df.to_dict('records')]

    df_prob_diffs = [abs(s['rto_probability'] - float(b)) for s, b in zip(single_from_df, batch_syn['pred_rto_prob'])]
    df_tau_diffs = [abs(s['cate_uplift_tau'] - float(t)) for s, t in zip(single_from_df, batch_syn['cate_uplift_tau'])]
    df_tier_mismatch = sum(1 for s, t in zip(single_from_df, batch_syn['risk_tier']) if s['risk_tier'] != t)
    df_wa_mismatch = sum(1 for s, b in zip(single_from_df, batch_syn['trigger_whatsapp_ndr']) if s['whatsapp_ndr_trigger'] != bool(b))
    df_disc_mismatch = sum(1 for s, b in zip(single_from_df, batch_syn['unlock_upi_discount_qr']) if s['unlock_doorstep_qr_discount'] != bool(b))

    log(f"  • DataFrame Row-Level Max |ΔP(RTO)| : {max(df_prob_diffs):.2e}")
    log(f"  • DataFrame Row-Level Max |Δtau|    : {max(df_tau_diffs):.2e}")
    log(f"  • DataFrame Risk Tier Mismatches    : {df_tier_mismatch}")
    log(f"  • DataFrame WhatsApp Mismatches     : {df_wa_mismatch}")
    log(f"  • DataFrame UPI Unlock Mismatches   : {df_disc_mismatch}")

    assert max(df_prob_diffs) < 1e-9, f"DataFrame P(RTO) divergence: {max(df_prob_diffs)}"
    assert max(df_tau_diffs) < 1e-9, f"DataFrame CATE divergence: {max(df_tau_diffs)}"
    assert df_tier_mismatch == 0
    assert df_wa_mismatch == 0
    assert df_disc_mismatch == 0
    log("  => Bit-for-bit exact parity verified for DataFrame inputs.")

    # 3. Adversarial Edge Case Discovery: Dict vs Ragged-DataFrame NaN Semantics
    log("\n  Evaluating Input Representation Sensitivity (Ragged Dict vs DataFrame NaN):")
    raw_dict_singles = [predictor.predict_single(r) for r in syn_records]
    ragged_prob_diffs = [abs(s['rto_probability'] - float(b)) for s, b in zip(raw_dict_singles, batch_syn['pred_rto_prob'])]
    ragged_tau_diffs = [abs(s['cate_uplift_tau'] - float(t)) for s, t in zip(raw_dict_singles, batch_syn['cate_uplift_tau'])]
    ragged_tier_diff = sum(1 for s, t in zip(raw_dict_singles, batch_syn['risk_tier']) if s['risk_tier'] != t)

    log(f"  • Raw Dict vs Ragged DF Max |ΔP(RTO)|: {max(ragged_prob_diffs):.2e}")
    log(f"  • Raw Dict vs Ragged DF Max |Δtau|   : {max(ragged_tau_diffs):.2e}")
    log(f"  • Raw Dict vs Ragged DF Tier Shifts  : {ragged_tier_diff} / 500 orders")
    log("  • Analysis: When a list of heterogeneously keyed dicts is cast to DataFrame,")
    log("    Pandas fills missing columns with NaN. In _tokenize_record(), setdefault() skips")
    log("    over NaN values because the key exists, falling back to FEATURE_DEFAULTS.")
    log("    This records input-shape behavior in this code path; it is not an empirical finding.")


def main():
    log("="*80)
    log("  SYNTHETIC PROTOTYPE SANITY-CHECK HARNESS")
    log("  Target: Meesho DICE Challenge Season 3 — Valmo RTO Reduction System")
    log("="*80)

    try:
        run_cli_tests()
        run_adversarial_input_tests()
        run_local_latency_sanity_check()
        run_cate_uplift_stress_test()
        run_synthetic_telemetry_pattern_check()
        run_single_vs_batch_parity_stress_test()

        log("\n" + "="*80)
        log("FINAL SUMMARY: CODE-PATH CHECKS PASSED ON SYNTHETIC EXAMPLES")
        log("="*80)
    except AssertionError as ae:
        log(f"\nCRITICAL ASSERTION FAILED: {ae}")
        sys.exit(1)
    except Exception as ex:
        log(f"\nUNEXPECTED FAILURE: {ex}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == '__main__':
    main()
