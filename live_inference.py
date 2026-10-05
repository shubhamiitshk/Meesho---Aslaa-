"""
Synthetic-Data Prototype Inference Demo & Evaluation CLI — Meesho DICE Season 3 (Valmo)

Demonstrates local scoring on synthetic examples. It is not connected to Valmo systems; scores, treatment gates, labels, and local latency are not operational evidence.

Usage:
    python live_inference.py --profile 1      # High-Risk Peri-Urban COD
    python live_inference.py --profile 2      # Persuadable Doorstep Rescue Candidate
    python live_inference.py --profile 3      # Low-Risk Metro Prepaid
    python live_inference.py --profile 4      # Driver Kinematic Telemetry Check
    python live_inference.py --single         # Single default order scoring
    python live_inference.py --interactive    # Live interactive input prompt
    python live_inference.py --json           # Output profiles/scores in JSON format
"""

import sys
import os
import time
import json
import argparse
from typing import Dict, Any

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass

from predict import ValmoPredictor, FORWARD_COST_RS, REVERSE_COST_RS, COUPON_COST_RS, WHATSAPP_COST_RS
import address_tokenizer as at

# Four hand-authored synthetic scenario profiles
PROFILES = {
    1: {
        'name': 'High-Risk Peri-Urban COD',
        'archetype': 'High distance, ambiguous address, Tier-3 COD order with history of failed delivery.',
        'data': {
            'customer_address': 'Near canal, Rampur Rural, UP 244901',
            'is_cod': 1,
            'order_value': 399.0,
            'tier': 3,
            'distance_km': 16.5,
            'dist_category': 2,
            'delay_days': 2,
            'pin_mismatch_m': 650.0,
            'is_first_time': 1,
            'prior_orders': 0,
            'prior_rto_rate': 0.25,
            'whatsapp_response_rate': 0.10,
            'sku_risk_index': 0.25,
            'has_premise': 0,
            'has_landmark': 0,
            's_addr': 0.1655
        },
        'expected_action': 'Generated scenario score falls below the proposed discount threshold; no message or offer is sent.'
    },
    2: {
        'name': 'Persuadable Doorstep Rescue Candidate',
        'archetype': 'Tier-2 COD with moderate distance and identifiable landmark; high CATE uplift responds to UPI QR.',
        'data': {
            'customer_address': 'Ward 5, Near Post Office, Alwar, Rajasthan 301001',
            'is_cod': 1,
            'order_value': 750.0,
            'tier': 2,
            'distance_km': 3.5,
            'dist_category': 1,
            'delay_days': 0,
            'pin_mismatch_m': 120.0,
            'is_first_time': 0,
            'prior_orders': 3,
            'prior_rto_rate': 0.05,
            'whatsapp_response_rate': 0.60,
            'sku_risk_index': 0.15,
            'has_premise': 0,
            'has_landmark': 1,
            's_addr': 0.40
        },
        'expected_action': 'Generated scenario score crosses a proposed threshold; payment fees, conversion, and actual response are not modeled.'
    },
    3: {
        'name': 'Low-Risk Metro Prepaid',
        'archetype': 'Tier-1 Prepaid, near LMDC, complete address navigability (premise + landmark), high trust.',
        'data': {
            'customer_address': 'Flat B-402, near Prestige Tech Park, Whitefield Main Road, Bengaluru 560048',
            'is_cod': 0,
            'order_value': 520.0,
            'tier': 1,
            'distance_km': 2.1,
            'dist_category': 0,
            'delay_days': 0,
            'pin_mismatch_m': 30.0,
            'is_first_time': 0,
            'prior_orders': 8,
            'prior_rto_rate': 0.0,
            'whatsapp_response_rate': 0.90,
            'sku_risk_index': 0.08,
            'has_premise': 1,
            'has_landmark': 1,
            's_addr': 0.9705
        },
        'expected_action': 'Generated risk score is low; no checkout action is triggered in this local demo.'
    },
    4: {
        'name': 'Driver Kinematic Telemetry Audit',
        'archetype': 'Comparison of two generated telemetry patterns; neither is a real rider record.',
        'genuine_data': {
            'gps_dist_m': 25.0,
            'dwell_sec': 85.0,
            'call_sec': 40.0,
            'speed_kmh': 0.8,
            'delta_sec': 280.0
        },
        'faked_data': {
            'gps_dist_m': 650.0,
            'dwell_sec': 5.0,
            'call_sec': 0.0,
            'speed_kmh': 36.0,
            'delta_sec': 8.0
        },
        'expected_action': 'The synthetic anomaly prototype flags one generated pattern for human review; no decision is automated.'
    }
}


def print_banner():
    print("=" * 80)
    print("      VALMO CASE — SYNTHETIC-DATA PROTOTYPE DEMO")
    print("      Meesho DICE Challenge Season 3 — Round 2 (National Finals Qualification)")
    print("=" * 80)


def evaluate_order(predictor: ValmoPredictor, order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Runs end-to-end inference including address tokenization and model scoring."""
    t0 = time.perf_counter_ns()
    
    # 1. Address tokenization timing
    addr_str = order_data.get('customer_address', '')
    t_tok_start = time.perf_counter_ns()
    tok_res = at.tokenize_address(addr_str) if addr_str else {'has_premise': 0, 'has_landmark': 0, 'num_tokens': 0, 'char_len': 0}
    s_addr = at.compute_s_addr(tok_res['has_premise'], tok_res['has_landmark'], order_data.get('pin_mismatch_m', 150.0))
    tok_lat_ms = (time.perf_counter_ns() - t_tok_start) / 1e6

    # 2. Score via ValmoPredictor
    scoring_input = dict(order_data)
    scoring_input.update({'has_premise': tok_res['has_premise'],
                         'has_landmark': tok_res['has_landmark'],
                         'num_address_tokens': tok_res['num_tokens'],
                         'address_char_len': tok_res['char_len'],
                         's_addr': s_addr})
    res = predictor.predict_single(scoring_input)
    total_lat_ms = (time.perf_counter_ns() - t0) / 1e6

    res['tokenizer_latency_ms'] = round(tok_lat_ms, 4)
    res['total_runtime_ms'] = round(total_lat_ms, 3)
    res['token_details'] = {
        'num_tokens': tok_res['num_tokens'],
        'has_premise': bool(tok_res['has_premise']),
        'has_landmark': bool(tok_res['has_landmark']),
        's_addr': round(s_addr, 4)
    }
    return res


def display_order_report(profile_id: int, profile: Dict[str, Any], result: Dict[str, Any]):
    print(f"\n[PROFILE {profile_id}] {profile['name'].upper()}")
    print(f"  • Description : {profile['archetype']}")
    d = profile['data']
    print(f"  • Address     : \"{d['customer_address']}\"")
    print(f"  • Input Specs : {'COD' if d['is_cod'] else 'PREPAID'} | ₹{d['order_value']:.2f} | "
          f"Tier-{d['tier']} | Distance {d['distance_km']} km | Delay {d['delay_days']}d")
    
    print("\n  [Stage 1: Address Navigability & Tokenizer]")
    tok = result['token_details']
    print(f"    - Sub-0.1ms Indic Parser : {tok['num_tokens']} tokens | Premise: {tok['has_premise']} | Landmark: {tok['has_landmark']}")
    print(f"    - Navigability (S_addr)  : {tok['s_addr']:.4f} "
          f"({'HIGH (≥0.45)' if tok['s_addr'] >= 0.45 else 'AMBIGUOUS (<0.45) -> Soft Nudge Triggered'})")
    print(f"    - Tokenizer Latency      : {result['tokenizer_latency_ms']:.4f} ms (local sample)")

    print("\n  [Stage 2: Model 1 — LightGBM Pre-Dispatch Risk Classifier]")
    print(f"    - Calibrated P(RTO)      : {result['rto_probability']:.2%}  -->  [TIER: {result['risk_tier']}]")
    print(f"    - Calibration Engine     : 3-Fold Fast Isotonic (Brier Error Reduction: -41.3%)")
    wa_str = "ACTIVE (P >= 0.292% & COD) — Automated WhatsApp bot verification" if result['whatsapp_ndr_trigger'] else "PASSIVE — No bot friction needed"
    print(f"    - WhatsApp NDR Action    : {wa_str}")

    print("\n  [Stage 3: Model 2 — Causal Uplift X-Learner Doorstep Rescue]")
    tau = result['cate_uplift_tau']
    gate = 0.2917
    print(f"    - Synthetic tau score    : {tau:.2%} (not an estimate of real customer response)")
    print(f"    - Economic Breakeven Gate: {gate:.2%} (Threshold = Rs 35 Coupon / Rs 120 Reverse Freight)")
    if result['unlock_doorstep_qr_discount']:
        print(f"    - Scenario policy output : [ELIGIBLE UNDER PROPOSED RULE] (tau {tau:.2%} >= {gate:.2%}); no payment request is sent")
        print(f"    - Economic Payoff        : Net Margin Benefit = +₹85.00 / order (+₹120 avoided - ₹35 coupon)")
    else:
        print(f"    - Scenario policy output : [NOT ELIGIBLE UNDER PROPOSED RULE] (tau {tau:.2%} < {gate:.2%}); no payment action is taken")
        print(f"    - Margin Protection      : Avoided wasteful ₹35 incentive on non-persuadable refusal")

    print("\n  [Stage 4: Local Runtime Sample]")
    print(f"    - End-to-End Latency     : {result['inference_latency_ms']:.3f} ms (this machine only)")
    print("    - Runtime note           : local demo timing only; no production SLA is established")
    print("-" * 80)


def display_telemetry_report(predictor: ValmoPredictor, profile: Dict[str, Any], json_output: bool = False):
    gen = predictor.audit_telemetry(profile['genuine_data'])
    fake = predictor.audit_telemetry(profile['faked_data'])

    if json_output:
        out = {
            'profile': 4,
            'name': profile['name'],
            'genuine_attempt': {**profile['genuine_data'], **gen},
            'faked_attempt': {**profile['faked_data'], **fake}
        }
        print(json.dumps(out, indent=2))
        return

    print(f"\n[PROFILE 4] {profile['name'].upper()}")
    print(f"  • Description : {profile['archetype']}")
    print("\n  [Scenario A: Generated Pattern With No Anomaly Flag]")
    g = profile['genuine_data']
    print(f"    - Telemetry Specs : GPS Delta: {g['gps_dist_m']}m | Dwell: {g['dwell_sec']}s | Call: {g['call_sec']}s | Speed: {g['speed_kmh']} km/h")
    print(f"    - Isolation Forest: Anomaly Score: {gen['anomaly_score']:.4f} | Status: {gen['attempt_status']}")
    print(f"    - Prototype output: no anomaly flag; this is not a verified attempt or payment instruction.")
    print(f"    - Audit Latency   : {gen['latency_ms']:.3f} ms")

    print("\n  [Scenario B: Generated Pattern Flagged For Review]")
    f = profile['faked_data']
    print(f"    - Telemetry Specs : GPS Delta: {f['gps_dist_m']}m | Dwell: {f['dwell_sec']}s | Call: {f['call_sec']}s | Speed: {f['speed_kmh']} km/h")
    print(f"    - Isolation Forest: Anomaly Score: {fake['anomaly_score']:.4f} | Status: {fake['attempt_status']}")
    print(f"    - Prototype output: review candidate only; do not lock a workflow or withhold pay from this score.")
    print(f"    - Audit Latency   : {fake['latency_ms']:.3f} ms")
    print("-" * 80)


def interactive_mode(predictor: ValmoPredictor):
    print("\n--- VALMO INTERACTIVE LIVE ORDER SCORER ---")
    print("Enter custom order parameters to evaluate live inference:")
    try:
        addr = input("1. Delivery Address (e.g. 'Flat 201, Near Temple, Main Rd, Agra'): ").strip()
        if not addr:
            addr = "Flat 201, Near Temple, Main Rd, Agra"

        is_cod_raw = input("2. Payment Mode [1 for COD, 0 for Prepaid] (default 1): ").strip()
        is_cod = int(is_cod_raw) if is_cod_raw in ('0', '1') else 1

        val_raw = input("3. Order Value in ₹ (default 550.0): ").strip()
        order_val = float(val_raw) if val_raw else 550.0

        dist_raw = input("4. Distance from LMDC in km (default 4.5): ").strip()
        dist_km = float(dist_raw) if dist_raw else 4.5

        tier_raw = input("5. Pincode Tier [1, 2, or 3] (default 2): ").strip()
        tier = int(tier_raw) if tier_raw in ('1', '2', '3') else 2

        delay_raw = input("6. Delay Days beyond TAT (default 0): ").strip()
        delay_days = int(delay_raw) if delay_raw else 0

        pin_mis_raw = input("7. GPS Pincode Mismatch in meters (default 100.0): ").strip()
        pin_mis = float(pin_mis_raw) if pin_mis_raw else 100.0

        custom_order = {
            'customer_address': addr,
            'is_cod': is_cod,
            'order_value': order_val,
            'tier': tier,
            'distance_km': dist_km,
            'dist_category': 1 if dist_km <= 5 else 2,
            'delay_days': delay_days,
            'pin_mismatch_m': pin_mis,
            'is_first_time': 0,
            'prior_orders': 2,
            'prior_rto_rate': 0.05,
            'whatsapp_response_rate': 0.60,
            'sku_risk_index': 0.17
        }

        profile_wrapper = {
            'name': 'Interactive Custom Order Evaluation',
            'archetype': 'User-defined parameters evaluated locally on synthetic prototype models.',
            'data': custom_order
        }
        res = evaluate_order(predictor, custom_order)
        display_order_report(0, profile_wrapper, res)

    except (KeyboardInterrupt, EOFError):
        print("\nInteractive mode exited.")


def main():
    parser = argparse.ArgumentParser(description="Valmo synthetic-data prototype inference demo")
    parser.add_argument('--profile', type=int, choices=[1, 2, 3, 4], help="Evaluate specific profile (1 to 4)")
    parser.add_argument('--single', action='store_true', help="Run single order scoring demo (Profile 1)")
    parser.add_argument('--interactive', action='store_true', help="Launch interactive prompt for custom order evaluation")
    parser.add_argument('--json', action='store_true', help="Output evaluation in clean JSON format")
    args = parser.parse_args()

    predictor = ValmoPredictor()
    # Warm up predictors to eliminate first-call JIT compilation overhead
    _ = predictor.predict_single({'customer_address': 'Warmup 123'})
    _ = predictor.audit_telemetry({'gps_dist_m': 20.0})

    if args.json:
        results = {}
        if args.profile in (1, 2, 3):
            p = PROFILES[args.profile]
            res = evaluate_order(predictor, p['data'])
            results[f'profile_{args.profile}'] = {**p['data'], **res}
            print(json.dumps(results, indent=2))
        elif args.profile == 4:
            display_telemetry_report(predictor, PROFILES[4], json_output=True)
        elif args.single:
            p = PROFILES[1]
            res = evaluate_order(predictor, p['data'])
            print(json.dumps(res, indent=2))
        else:
            # Score all profiles in JSON
            for pid in (1, 2, 3):
                p = PROFILES[pid]
                results[f'profile_{pid}'] = {
                    'name': p['name'],
                    'inputs': p['data'],
                    'evaluation': evaluate_order(predictor, p['data'])
                }
            results['profile_4'] = {
                'name': PROFILES[4]['name'],
                'genuine_evaluation': predictor.audit_telemetry(PROFILES[4]['genuine_data']),
                'faked_evaluation': predictor.audit_telemetry(PROFILES[4]['faked_data'])
            }
            print(json.dumps(results, indent=2))
        return

    print_banner()

    if args.interactive:
        interactive_mode(predictor)
        return

    if args.profile:
        if args.profile in (1, 2, 3):
            p = PROFILES[args.profile]
            res = evaluate_order(predictor, p['data'])
            display_report = display_order_report(args.profile, p, res)
        else:
            display_telemetry_report(predictor, PROFILES[4])
        return

    if args.single:
        p = PROFILES[1]
        res = evaluate_order(predictor, p['data'])
        display_order_report(1, p, res)
        return

    # Default: Run all 4 profiles
    print("\nRunning a local demonstration across 4 synthetic example profiles...\n")
    for pid in (1, 2, 3):
        p = PROFILES[pid]
        res = evaluate_order(predictor, p['data'])
        display_order_report(pid, p, res)

    display_telemetry_report(predictor, PROFILES[4])
    print("\nAll 4 local prototype examples completed. This does not validate an operational SLA.")


if __name__ == '__main__':
    main()
