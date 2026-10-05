"""
Valmo Production ML Inference Engine & Latency Benchmarking — Meesho DICE Season 3
Provides sub-3ms real-time single-order scoring, batch prediction,
doorstep causal uplift CATE evaluation, and driver telemetry anomaly auditing.
Integrates live address tokenization via `address_tokenizer` and handles edge cases robustly.
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def _force_utf8_stdio() -> None:
    """
    Reconfigure the console to UTF-8 so the rupee glyph and Greek letters used
    in output do not raise UnicodeEncodeError on a cp1252 Windows console.

    Uses TextIOWrapper.reconfigure() on the EXISTING stream. Wrapping the stream
    again instead would create a second wrapper; when the first one is garbage
    collected it closes the underlying buffer and the second raises
    "I/O operation on closed file" mid-run. Calling this from more than one
    module therefore has to be idempotent, which reconfigure is and a new
    wrapper is not.
    """
    if sys.platform != 'win32':
        return
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


_force_utf8_stdio()

import address_tokenizer as at

MODELS_DIR = 'models'

# Feature defaults used whenever an inbound payload omits a field. These mirror
# the training-set medians and MUST be applied identically on the single-order
# and batch paths -- silently coercing a missing feature to 0.0 instead is a
# train/serve skew bug that produces confidently wrong risk scores.
FEATURE_DEFAULTS = {
    'is_cod': 1.0, 'order_value': 450.0, 'sku_risk_index': 0.17, 'is_first_time': 0.0,
    'prior_orders': 2.0, 'prior_rto_rate': 0.05, 'whatsapp_response_rate': 0.50,
    'tier': 2.0, 'has_premise': 1.0, 'has_landmark': 1.0, 'num_address_tokens': 4.0,
    'address_char_len': 65.0, 'pin_mismatch_m': 150.0, 's_addr': 0.70,
    'dist_category': 1.0, 'distance_km': 3.5, 'delay_days': 0.0, 'promised_tat_days': 3.0,
}

# Cost constants (see metrics_lib.py) — kept local so the serving path has no
# import-time dependency on the training module.
FORWARD_COST_RS = 50.0
REVERSE_COST_RS = 120.0
COUPON_COST_RS = 35.0
WHATSAPP_COST_RS = 0.35
NET_MARGIN_PER_RESCUE_RS = REVERSE_COST_RS - COUPON_COST_RS

def _safe_float(val, default: float = 0.0) -> float:
    """Robust conversion to float handling None, empty strings, and NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default

def _safe_int(val, default: int = 0) -> int:
    """Robust conversion to int handling None, empty strings, and NaN."""
    return int(_safe_float(val, float(default)))

def _risk_tier(p: float) -> str:
    if p < 0.10:
        return "LOW"
    if p < 0.20:
        return "MODERATE"
    if p < 0.35:
        return "HIGH"
    return "CRITICAL"


class _FastIsotonic:
    """
    Direct evaluation of a fitted IsotonicRegression via np.interp.

    sklearn's IsotonicRegression.predict costs ~127us for a single value --
    almost entirely input validation -- which dominates a sub-millisecond
    serving budget. The fitted model is a piecewise-linear map from
    X_thresholds_ to y_thresholds_, so np.interp reproduces it exactly
    (including out_of_bounds="clip" semantics, which are np.interp's default).
    Verified bit-for-bit against sklearn in test_suite.test_predict_engine.
    """
    __slots__ = ('_x', '_y')

    def __init__(self, fitted_isotonic):
        self._x = np.ascontiguousarray(fitted_isotonic.X_thresholds_, dtype=np.float64)
        self._y = np.ascontiguousarray(fitted_isotonic.y_thresholds_, dtype=np.float64)

    def predict(self, values):
        return np.interp(np.asarray(values, dtype=np.float64), self._x, self._y)

class ValmoPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self._load_models()

    def _load_models(self):
        # 1. Load Pre-Dispatch Classifier
        rto_model_path = os.path.join(self.models_dir, 'rto_risk_lightgbm_calibrated.joblib')
        if not os.path.exists(rto_model_path):
            rto_model_path = os.path.join(self.models_dir, 'rto_risk_model_calibrated.joblib')
        self.rto_clf = joblib.load(rto_model_path)
        self.rto_features = joblib.load(os.path.join(self.models_dir, 'rto_feature_names.joblib'))

        # Extract native C++ boosters and isotonic calibrators for sub-millisecond
        # execution. Calibrators are flattened to a direct np.interp lookup, which
        # is numerically identical to sklearn's IsotonicRegression.predict but
        # ~60x cheaper per call.
        self.boosters = [c.estimator.booster_ for c in self.rto_clf.calibrated_classifiers_]
        self.calibrators = [_FastIsotonic(c.calibrators[0])
                            for c in self.rto_clf.calibrated_classifiers_]
        self.n_calibrators = len(self.calibrators)

        # 2. Load Causal Uplift X-Learner Bundle
        uplift_path = os.path.join(self.models_dir, 'doorstep_uplift_xlearner.joblib')
        self.uplift_bundle = joblib.load(uplift_path)
        self.m0 = self.uplift_bundle['m0']
        self.m1 = self.uplift_bundle['m1']
        self.tau0 = self.uplift_bundle['tau0']
        self.tau1 = self.uplift_bundle['tau1']
        self.b_tau0 = self.tau0.booster_
        self.b_tau1 = self.tau1.booster_
        self.propensity = float(self.uplift_bundle['propensity'])
        self.uplift_features = self.uplift_bundle['feature_cols']
        self.economic_threshold = float(self.uplift_bundle.get(
            'economic_threshold', COUPON_COST_RS / REVERSE_COST_RS))

        # 3. Load Telemetry Isolation Forest
        tel_path = os.path.join(self.models_dir, 'telemetry_isolation_forest.joblib')
        self.telemetry_iso = joblib.load(tel_path)
        self.tel_features = ['gps_dist_m', 'dwell_sec', 'call_sec', 'speed_kmh', 'delta_sec']

        # Cost thresholds. Single rule, used identically everywhere:
        # fire an intervention of cost C when P(failure) > C / V_reverse_avoided.
        self.p_star_wa = WHATSAPP_COST_RS / REVERSE_COST_RS      # 0.00292
        self.p_star_disc = COUPON_COST_RS / REVERSE_COST_RS       # 0.29167

        self._fallback_features = [c for c in self.rto_features if c in FEATURE_DEFAULTS]
        self._missing_features = [c for c in self.rto_features if c not in FEATURE_DEFAULTS]

        # Pre-built random streams for latency benchmarking, so payload
        # construction never lands inside a timed region.
        self._bench_rngs = [np.random.default_rng(1000 + i) for i in range(64)]
        self._bench_pool = [self._make_benchmark_payload(i) for i in range(512)]

    # ------------------------------------------------------------------
    # Shared scoring core
    # ------------------------------------------------------------------
    def _score_rto(self, X: np.ndarray) -> np.ndarray:
        """
        Calibrated P(RTO) for a (n, len(rto_features)) matrix.

        This is the SINGLE scoring implementation. Both the single-order path
        and the batch path call it, so `predict_single(order)['rto_probability']`
        and the corresponding row of `predict_batch(df)['pred_rto_prob']` are
        identical by construction (asserted by `python predict.py --parity`).

        The raw LightGBM booster plus per-fold isotonic calibration is both the
        fast path and the numerically exact equivalent of
        CalibratedClassifierCV.predict_proba. For a single row we pin
        num_threads=1, because OpenMP thread-dispatch overhead dominates below
        roughly 100 rows.
        """
        out = np.zeros(len(X), dtype=np.float64)
        if len(X) == 0:
            return out
        single = len(X) == 1
        for booster, calibrator in zip(self.boosters, self.calibrators):
            margins = (booster.predict(X, raw_score=True, num_threads=1) if single
                       else booster.predict(X, raw_score=True))
            out += calibrator.predict(margins)
        return np.clip(out / self.n_calibrators, 0.0, 1.0)

    def _score_tau(self, U: np.ndarray) -> np.ndarray:
        """Propensity-weighted X-Learner CATE tau for a (n, len(uplift_features)) matrix."""
        if len(U) == 0:
            return np.empty(0, dtype=np.float64)
        single = len(U) == 1
        kw = {'num_threads': 1} if single else {}
        return np.clip(
            self.propensity * self.b_tau0.predict(U, **kw)
            + (1.0 - self.propensity) * self.b_tau1.predict(U, **kw), 0.0, 1.0)

    def _build_rto_matrix(self, records: list) -> np.ndarray:
        """
        Assemble the RTO feature matrix, applying FEATURE_DEFAULTS to gaps.

        A missing feature falls back to its training-set median, never to 0.0.
        Coercing a missing signal to zero would silently invert the model's
        reading of it (a missing address token is evidence of an ambiguous
        address, not evidence of a short one) and is a classic train/serve skew
        bug.
        """
        n = len(records)
        if n == 0:
            return np.empty((0, len(self.rto_features)), dtype=np.float32)
        if n == 1:
            rec = records[0]
            vec = [_safe_float(rec.get(c, FEATURE_DEFAULTS.get(c, 0.0)),
                              FEATURE_DEFAULTS.get(c, 0.0))
                   for c in self.rto_features]
            return np.array([vec], dtype=np.float32)

        X = np.empty((n, len(self.rto_features)), dtype=np.float32)
        df = pd.DataFrame(records)
        for j, col in enumerate(self.rto_features):
            default = FEATURE_DEFAULTS.get(col, 0.0)
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').to_numpy(
                    dtype=np.float32, na_value=np.nan)
                X[:, j] = np.nan_to_num(vals, nan=default)
            else:
                X[:, j] = default
        return X

    def _tokenize_record(self, d: dict) -> dict:
        """
        Returns a copy of `d` enriched with address-derived features when the
        caller supplied a raw address string. Explicitly-supplied values always
        win — we only fill gaps.
        """
        raw_addr = d.get('customer_address') or d.get('address') or d.get('raw_address')
        if not (raw_addr and isinstance(raw_addr, str)):
            return dict(d)
        tok = at.tokenize_address(raw_addr)
        out = dict(d)
        out.setdefault('has_premise', tok['has_premise'])
        out.setdefault('has_landmark', tok['has_landmark'])
        out.setdefault('num_address_tokens', tok['num_tokens'])
        out.setdefault('address_char_len', tok['char_len'])
        out.setdefault('s_addr', at.compute_s_addr(
            tok['has_premise'], tok['has_landmark'],
            _safe_float(d.get('pin_mismatch_m', 150.0), 150.0)))
        return out

    def predict_single(self, order_dict: dict) -> dict:
        """
        Runs sub-3ms inference on a single order payload.
        Shares its scoring core with predict_batch, so the two paths cannot drift.
        """
        t_start = time.perf_counter_ns()

        d = self._tokenize_record(order_dict)

        p_rto = float(self._score_rto(self._build_rto_matrix([d]))[0])

        is_cod_flag = _safe_int(d.get('is_cod', 1), 1)
        trigger_wa = bool(p_rto >= self.p_star_wa and is_cod_flag == 1)
        eligible_doorstep = bool(p_rto >= self.p_star_disc)

        u_vals = [_safe_float(d.get(c, FEATURE_DEFAULTS.get(c, 0.0)), FEATURE_DEFAULTS.get(c, 0.0))
                  for c in self.uplift_features]
        cate_tau = float(self._score_tau(np.array([u_vals], dtype=np.float32))[0])
        unlock_discount_qr = bool(cate_tau >= self.economic_threshold)

        latency_ms = (time.perf_counter_ns() - t_start) / 1e6

        return {
            'rto_probability': round(p_rto, 4),
            'risk_tier': _risk_tier(p_rto),
            'whatsapp_ndr_trigger': trigger_wa,
            'doorstep_discount_eligible': eligible_doorstep,
            'cate_uplift_tau': round(cate_tau, 4),
            'unlock_doorstep_qr_discount': unlock_discount_qr,
            'net_margin_benefit_rs': NET_MARGIN_PER_RESCUE_RS if unlock_discount_qr else 0.0,
            'inference_latency_ms': round(latency_ms, 3),
        }

    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        High-throughput batch scoring. Numerically identical to predict_single
        by construction (both delegate to _score_rto / _score_tau).
        """
        t_start = time.time()
        res = input_df.copy()
        records = [self._tokenize_record(rec) for rec in res.to_dict('records')]

        probs = self._score_rto(self._build_rto_matrix(records))

        u_df = np.empty((len(records), len(self.uplift_features)), dtype=np.float32)
        for j, col in enumerate(self.uplift_features):
            default = FEATURE_DEFAULTS.get(col, 0.0)
            for i, rec in enumerate(records):
                u_df[i, j] = _safe_float(rec.get(col, default), default)
        cate_scores = self._score_tau(u_df)

        res['pred_rto_prob'] = probs.round(4)
        res['risk_tier'] = [_risk_tier(p) for p in probs]
        is_cod = res['is_cod'] if 'is_cod' in res.columns else pd.Series(1, index=res.index)
        is_cod_num = pd.to_numeric(is_cod, errors='coerce').fillna(1).astype(int)
        res['trigger_whatsapp_ndr'] = (probs >= self.p_star_wa) & (is_cod_num == 1)
        res['cate_uplift_tau'] = cate_scores.round(4)
        res['unlock_upi_discount_qr'] = cate_scores >= self.economic_threshold

        t_total = time.time() - t_start
        print(f"Batch scored {len(res):,} orders in {t_total:.2f}s "
              f"({len(res) / max(t_total, 1e-6):,.0f} orders/sec).")
        return res

    def audit_telemetry(self, attempt_dict: dict) -> dict:
        """Evaluates delivery attempt physical kinematics."""
        t_start = time.perf_counter_ns()
        feat_vals = np.array([[
            _safe_float(attempt_dict.get('gps_dist_m', 20.0)),
            _safe_float(attempt_dict.get('dwell_sec', 90.0)),
            _safe_float(attempt_dict.get('call_sec', 35.0)),
            _safe_float(attempt_dict.get('speed_kmh', 1.0)),
            _safe_float(attempt_dict.get('delta_sec', 300.0))
        ]], dtype=np.float32)

        pred_label = self.telemetry_iso.predict(feat_vals)[0]  # 1 = Genuine, -1 = Anomaly
        score = float(-self.telemetry_iso.decision_function(feat_vals)[0])
        t_end = time.perf_counter_ns()

        is_fake = bool(pred_label == -1)
        return {
            'attempt_status': 'REVIEW_CANDIDATE' if is_fake else 'NO_ANOMALY_FLAG',
            'status_scope': 'synthetic prototype output; human review required; no adverse action',
            'is_anomaly': is_fake,
            'anomaly_score': round(score, 4),
            'latency_ms': round((t_end - t_start) / 1e6, 3)
        }

    def _make_benchmark_payload(self, seed: int = 0) -> dict:
        """
        One synthetic order drawn from the same ranges as the training data.
        `seed` indexes a pre-built pool so the benchmark harness can construct
        payloads OUTSIDE the timed region.
        """
        rng = self._bench_rngs[seed % len(self._bench_rngs)]
        return {
            'is_cod': int(rng.choice([0, 1], p=[0.2, 0.8])),
            'order_value': float(rng.uniform(200, 1200)),
            'sku_risk_index': float(rng.choice([0.08, 0.12, 0.19, 0.22])),
            'is_first_time': int(rng.choice([0, 1])),
            'prior_orders': int(rng.integers(0, 15)),
            'prior_rto_rate': float(rng.uniform(0.0, 0.3)),
            'whatsapp_response_rate': float(rng.uniform(0.1, 0.9)),
            'tier': int(rng.choice([1, 2, 3])),
            'has_premise': int(rng.choice([0, 1])),
            'has_landmark': int(rng.choice([0, 1])),
            'num_address_tokens': int(rng.integers(2, 6)),
            'address_char_len': int(rng.integers(30, 110)),
            'pin_mismatch_m': float(rng.exponential(150)),
            's_addr': float(rng.uniform(0.2, 0.95)),
            'dist_category': int(rng.choice([0, 1, 2])),
            'distance_km': float(rng.uniform(1.0, 15.0)),
            'delay_days': int(rng.choice([0, 1, 2])),
            'promised_tat_days': 3,
        }

    def benchmark_latency(self, n_samples: int = 1000, n_warmup: int = 50) -> dict:
        """Profiles inference latency across N simulated orders."""
        rng = np.random.default_rng()
        mock_payloads = [{
            'is_cod': int(rng.choice([0, 1], p=[0.2, 0.8])),
            'order_value': float(rng.uniform(200, 1200)),
            'sku_risk_index': float(rng.choice([0.08, 0.12, 0.19, 0.22])),
            'is_first_time': int(rng.choice([0, 1])),
            'prior_orders': int(rng.integers(0, 15)),
            'prior_rto_rate': float(rng.uniform(0.0, 0.3)),
            'whatsapp_response_rate': float(rng.uniform(0.1, 0.9)),
            'tier': int(rng.choice([1, 2, 3])),
            'has_premise': int(rng.choice([0, 1])),
            'has_landmark': int(rng.choice([0, 1])),
            'num_address_tokens': int(rng.integers(2, 6)),
            'address_char_len': int(rng.integers(30, 110)),
            'pin_mismatch_m': float(rng.exponential(150)),
            's_addr': float(rng.uniform(0.2, 0.95)),
            'dist_category': int(rng.choice([0, 1, 2])),
            'distance_km': float(rng.uniform(1.0, 15.0)),
            'delay_days': int(rng.choice([0, 1, 2])),
            'promised_tat_days': 3,
        } for _ in range(n_samples)]

        # Warm up on the payloads we actually have (never assume >= 50 samples).
        warmup = min(n_warmup, n_samples)
        for i in range(warmup):
            self.predict_single(mock_payloads[i])

        lat_arr = np.array([self.predict_single(p)['inference_latency_ms']
                            for p in mock_payloads])

        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        return {
            'n_runs': n_samples,
            'n_warmup': warmup,
            'mean_ms': round(float(lat_arr.mean()), 3),
            'p50_ms': round(p50, 3),
            'p90_ms': round(p90, 3),
            'p95_ms': round(p95, 3),
            'p99_ms': round(p99, 3),
            'local_p99_budget_ms': 3.0,
            'within_local_budget': bool(p99 < 3.0),
        }

def _run_parity_check(predictor, n: int = 500) -> dict:
    """
    Proves the single-order and batch serving paths cannot disagree.

    Rows are scored twice -- once through predict_single, once through
    predict_batch -- and the maximum absolute difference is reported. A passing
    run (max diff < 1e-4) is evidence against train/serve skew, which is the
    single most common way a low-latency serving path silently diverges from
    the validated model.
    """
    sample = pd.read_csv('data/sample_1000_orders.csv').head(n)
    batch = predictor.predict_batch(sample)

    singles = [predictor.predict_single(rec)
               for rec in sample.drop(columns=[c for c in ('pred_rto_prob', 'risk_tier',
                                                          'trigger_whatsapp_ndr', 'cate_uplift_tau',
                                                          'unlock_upi_discount_qr')
                                               if c in sample.columns]).to_dict('records')]

    d_prob = max(abs(s['rto_probability'] - float(b)) for s, b in
                 zip(singles, batch['pred_rto_prob']))
    d_tau = max(abs(s['cate_uplift_tau'] - float(t)) for s, t in
                zip(singles, batch['cate_uplift_tau']))
    d_tier = sum(1 for s, t in zip(singles, batch['risk_tier']) if s['risk_tier'] != t)

    # EXACT-EQUALITY GATE, matching benchmark_serving.run_parity and the claims
    # ledger's own description of this check. Both paths delegate to a single
    # _score_rto on the same matrix, so any non-zero difference means they are
    # NOT sharing a code path. A 1e-4 tolerance here would let a real train/serve
    # skew pass a check that is quoted by name in the Q&A rehearsal cards.
    ok = d_prob < 1e-9 and d_tau < 1e-9 and d_tier == 0
    return {'n': len(singles), 'max_abs_prob_dif': round(d_prob, 8),
            'max_abs_tau_dif': round(d_tau, 8), 'risk_tier_mismatches': int(d_tier),
            'parity_ok': bool(ok)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Valmo Production ML Predictor CLI")
    parser.add_argument('--single', action='store_true', help="Run single order scoring demo")
    parser.add_argument('--batch', type=str, default=None, help="Path to input CSV for batch scoring")
    parser.add_argument('--benchmark', action='store_true', help="Run 1,000 requests latency benchmark")
    parser.add_argument('--audit', action='store_true', help="Run edge telemetry audit demo")
    parser.add_argument('--parity', action='store_true',
                        help="Prove single-order and batch scoring paths agree")
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR)
    args = parser.parse_args()

    predictor = ValmoPredictor(models_dir=args.models_dir)

    if args.parity:
        print("=" * 75)
        print("   SERVING-PATH PARITY CHECK (single vs batch)   ")
        print("=" * 75)
        par = _run_parity_check(predictor)
        print(f"  • Rows compared        : {par['n']}")
        print(f"  • Max |ΔP(RTO)|        : {par['max_abs_prob_dif']:.2e}")
        print(f"  • Max |ΔCATE tau|      : {par['max_abs_tau_dif']:.2e}")
        print(f"  • Risk-tier mismatches : {par['risk_tier_mismatches']}")
        print(f"  • Verdict              : {'PASS — paths are numerically identical' if par['parity_ok'] else 'FAIL'}")

    if args.single or (not args.batch and not args.benchmark and not args.audit):
        print("=" * 75)
        print("   VALMO PRODUCTION REAL-TIME ORDER SCORING (MOCK ORDER DEMO)   ")
        print("=" * 75)
        mock_high_risk = {
            'customer_address': 'Flat 102, Shanti Niwas, Station Road, Kalyan Nagar, Jaipur, Rajasthan 302001',
            'is_cod': 1,
            'order_value': 480.0,
            'sku_risk_index': 0.22,
            'is_first_time': 1,
            'prior_orders': 0,
            'prior_rto_rate': 0.0,
            'whatsapp_response_rate': 0.15,
            'tier': 3,
            'pin_mismatch_m': 480.0,
            'dist_category': 2,
            'distance_km': 12.4,
            'delay_days': 1,
            'promised_tat_days': 3
        }
        res = predictor.predict_single(mock_high_risk)
        print("Order Profile:")
        print(f"  • Address: {mock_high_risk['customer_address']}")
        print("  • Payment: COD | Value: Rs. 480 | Tier: 3 | Distance: 12.4 km | Delay: 1 day")
        print("\nModel 1: Pre-Dispatch RTO Classifier:")
        print(f"  • Predicted Failure Risk  : {res['rto_probability']:.2%} ({res['risk_tier']} RISK)")
        print(f"  • WhatsApp NDR Pre-Alert  : {'[TRIGGER WHATSAPP BOT]' if res['whatsapp_ndr_trigger'] else '[NONE]'}")
        print("\nModel 2: Causal Uplift X-Learner (Doorstep Rescue):")
        print(f"  • Doorstep CATE Uplift (tau): {res['cate_uplift_tau']:.2%}")
        if res['unlock_doorstep_qr_discount']:
            print("  • Dynamic UPI QR Status   : [UNLOCK Rs. 35 DISCOUNT QR] "
                  f"(Uplift >= {predictor.economic_threshold:.2%} -> Persuadable!)")
            print("  • Economic Payoff         : Net Logistics Benefit = +Rs. 85.00 "
                  "(+Rs. 120 reverse saved - Rs. 35 discount)")
        else:
            print("  • Dynamic UPI QR Status   : [WITHHOLD DISCOUNT] "
                  f"(Uplift < {predictor.economic_threshold:.2%} -> Protect Margins)")
        print("\nInference Performance:")
        print(f"  • End-to-End Latency      : {res['inference_latency_ms']:.3f} ms (local demo timing; not an SLA)")

    if args.audit or not (args.batch or args.benchmark):
        print("\n" + "=" * 75)
        print("   MODEL 3: SYNTHETIC ATTEMPT-PATTERN REVIEW DEMO   ")
        print("=" * 75)
        genuine_mock = {'gps_dist_m': 28.0, 'dwell_sec': 85.0, 'call_sec': 45.0, 'speed_kmh': 0.5, 'delta_sec': 310.0}
        fake_mock = {'gps_dist_m': 650.0, 'dwell_sec': 5.0, 'call_sec': 0.0, 'speed_kmh': 36.0, 'delta_sec': 8.0}

        r_gen = predictor.audit_telemetry(genuine_mock)
        r_fake = predictor.audit_telemetry(fake_mock)

        print("Attempt A (GPS: 28m, Dwell: 85s, Call: 45s, Speed: 0.5 km/h):")
        print(f"  -> Prototype output: {r_gen['attempt_status']} (score: {r_gen['anomaly_score']:.4f}, local timing: {r_gen['latency_ms']} ms)")
        print("\nAttempt B (GPS: 650m, Dwell: 5s, Call: 0s, Speed: 36 km/h):")
        print(f"  -> Prototype output: {r_fake['attempt_status']} (score: {r_fake['anomaly_score']:.4f}, local timing: {r_fake['latency_ms']} ms; human review only)")

    if args.benchmark or not (args.batch or args.single):
        print("\n" + "=" * 75)
        print("   LOCAL LATENCY SAMPLE: 1,000 SYNTHETIC-PAYLOAD RUNS   ")
        print("=" * 75)
        bench = predictor.benchmark_latency(1000)
        print(f"  • Total Inference Runs: {bench['n_runs']:,}")
        print(f"  • Mean Latency        : {bench['mean_ms']:.3f} ms")
        print(f"  • P50 Median Latency  : {bench['p50_ms']:.3f} ms")
        print(f"  • P90 Latency         : {bench['p90_ms']:.3f} ms")
        print(f"  • P95 Latency         : {bench['p95_ms']:.3f} ms")
        print(f"  • P99 Latency          : {bench['p99_ms']:.3f} ms (local budget: < 3.0 ms; not an SLA)")
        print(f"  • Local Budget Check   : {'PASS' if bench['within_local_budget'] else 'FAIL'}")

    if args.batch:
        print(f"\nRunning batch scoring on '{args.batch}'...")
        in_df = pd.read_csv(args.batch)
        out_df = predictor.predict_batch(in_df)
        out_path = args.batch.replace('.csv', '_predictions.csv')
        out_df.to_csv(out_path, index=False)
        print(f"Saved batch predictions to '{out_path}'")
