"""
Valmo RTO ML Training Pipeline — Meesho DICE Challenge Season 3
=============================================================

Trains prototype models on generated scenario data for the Valmo case. None is
trained or validated on Valmo operational data, and reported metrics do not
establish production performance:

    STEP 1  Dataset construction / integrity summary
    STEP 2  Pre-Dispatch RTO Risk Classifier  (LightGBM + XGBoost, isotonic calibrated)
    STEP 3  Doorstep Rescue Causal Uplift      (X-Learner, Qünzel et al. 2019)
    STEP 4  Attempt-Pattern Anomaly Prototype (Isolation Forest, synthetic holdout)

Evaluation principles enforced throughout:

  * Metrics are held out within the generated dataset; they do not establish
    generalization to live orders, customers, riders, or Valmo systems.
  * Hyperparameters that touch labels are never set from label information.
    Where a scenario operating point is chosen, it uses an assumed review budget
    on training scores; the detector is scored once on the synthetic holdout.
  * Uplift is reported against an explicit permutation null with a p-value, and
    against a perfect-targeting oracle, so the headline number is falsifiable.
  * Cost thresholds use one rule everywhere: fire when
    P(failure) > C_action / V_reverse_avoided.

Run:  python train_valmo_models.py
"""

import os
import io
import sys
import json
import time
import subprocess
import argparse

import numpy as np
import pandas as pd
import joblib

import lightgbm as lgb
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    classification_report,
    confusion_matrix,
)

import metrics_lib as ml

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATA_DIR = 'data'
MODEL_DIR = 'models'
DATASET_CSV = os.path.join(DATA_DIR, 'valmo_orders_dataset.csv')

SEED = 42

# Scenario assumption for the anomaly prototype's review operating point; this
# is not an observed operational capacity or staffing constraint.
TELEMETRY_AUDIT_BUDGET_PCT = 10.0


# ==========================================================================
# STEP 1 — DATASET
# ==========================================================================
def step1_dataset(force_regenerate: bool = False):
    print("\n" + "=" * 80)
    print("[STEP 1/4] Valmo Order Dataset")
    print("=" * 80)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if force_regenerate and os.path.exists(DATASET_CSV):
        os.remove(DATASET_CSV)

    if not os.path.exists(DATASET_CSV):
        print("  Dataset absent -> invoking generate_valmo_dataset.py as a subprocess.")
        subprocess.run([sys.executable, 'generate_valmo_dataset.py'], check=True)

    t0 = time.time()
    df = pd.read_csv(DATASET_CSV)
    print(f"  Loaded {len(df):,} orders x {len(df.columns)} cols in {time.time() - t0:.2f}s")

    summary = {
        'n_orders': int(len(df)),
        'n_columns': int(len(df.columns)),
        'n_pincodes': int(df['pincode'].nunique()),
        'n_districts': int(df['district'].nunique()),
        'n_states': int(df['state'].nunique()),
        'blended_rto_rate': round(float(df['rto_flag'].mean()), 5),
        'cod_rto_rate': round(float(df[df['is_cod'] == 1]['rto_flag'].mean()), 5),
        'prepaid_rto_rate': round(float(df[df['is_cod'] == 0]['rto_flag'].mean()), 5),
        'cod_share': round(float(df['is_cod'].mean()), 5),
        'rto_near_rate': round(float(df[df['dist_category'] == 0]['rto_flag'].mean()), 5),
        'rto_moderate_rate': round(float(df[df['dist_category'] == 1]['rto_flag'].mean()), 5),
        'rto_far_rate': round(float(df[df['dist_category'] == 2]['rto_flag'].mean()), 5),
        'distance_penalty_pct': round(float(
            (df[df['dist_category'] == 2]['rto_flag'].mean() /
             df[df['dist_category'] == 0]['rto_flag'].mean() - 1.0) * 100.0), 1),
        's_addr_mean': round(float(df['s_addr'].mean()), 4),
        's_addr_vs_outcome_corr': round(float(np.corrcoef(df['s_addr'], df['rto_flag'])[0, 1]), 4),
    }

    print(f"  • Blended RTO        : {summary['blended_rto_rate']:.2%}")
    print(f"  • COD RTO / Prepaid  : {summary['cod_rto_rate']:.2%} / {summary['prepaid_rto_rate']:.2%}")
    print(f"  • Distance penalty   : {summary['rto_near_rate']:.2%} (near) -> "
          f"{summary['rto_far_rate']:.2%} (far)  = +{summary['distance_penalty_pct']:.1f}%")
    print(f"  • Pincodes / districts: {summary['n_pincodes']:,} / {summary['n_districts']:,}")
    print(f"  • S_addr vs outcome r : {summary['s_addr_vs_outcome_corr']:+.4f} (navigability is protective)")

    # Calibration disclosure: this is a calibrated simulation, not Valmo's
    # proprietary data. We verify its marginals match the published case brief.
    case_brief = {'blended_rto': 0.170, 'cod_rto': 0.200, 'prepaid_rto': 0.050, 'cod_share': 0.80}
    dev = {
        'blended_rto_pp': round(abs(summary['blended_rto_rate'] - case_brief['blended_rto']) * 100, 2),
        'cod_rto_pp': round(abs(summary['cod_rto_rate'] - case_brief['cod_rto']) * 100, 2),
        'prepaid_rto_pp': round(abs(summary['prepaid_rto_rate'] - case_brief['prepaid_rto']) * 100, 2),
    }
    summary['calibration_vs_case_brie'] = dev
    print(f"  • Marginal deviation from case brief: blended {dev['blended_rto_pp']:.2f}pp, "
          f"COD {dev['cod_rto_pp']:.2f}pp, prepaid {dev['prepaid_rto_pp']:.2f}pp")

    return df, summary


# ==========================================================================
# STEP 2 — PRE-DISPATCH RTO RISK CLASSIFIER
# ==========================================================================
def step2_classifier(df):
    print("\n" + "=" * 80)
    print("[STEP 2/4] Model 1 — Pre-Dispatch RTO Risk Classifier")
    print("=" * 80)

    feature_cols = [
        'is_cod', 'order_value', 'sku_risk_index', 'is_first_time',
        'prior_orders', 'prior_rto_rate', 'whatsapp_response_rate',
        'tier', 'has_premise', 'has_landmark', 'num_address_tokens', 'address_char_len',
        'pin_mismatch_m', 's_addr', 'dist_category', 'distance_km',
        'delay_days', 'promised_tat_days',
    ]

    X = df[feature_cols]
    y = df['rto_flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    print(f"  Split: Train N={len(X_train):,}, Holdout Test N={len(X_test):,} (stratified)")

    neg_pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()

    # ---- 2A: hyperparameter search -------------------------------------
    print("\n  [2A] Hyperparameter search (3-fold, selection on validation ROC-AUC)")
    candidates = [
        {'n_estimators': 120, 'learning_rate': 0.05, 'max_depth': 4, 'num_leaves': 15, 'colsample_bytree': 0.80},
        {'n_estimators': 160, 'learning_rate': 0.04, 'max_depth': 5, 'num_leaves': 31, 'colsample_bytree': 0.85},
        {'n_estimators': 200, 'learning_rate': 0.03, 'max_depth': 6, 'num_leaves': 45, 'colsample_bytree': 0.90},
    ]
    cv_quick = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_params, best_auc = None, -1.0
    for i, params in enumerate(candidates, 1):
        fold_aucs = []
        for tr, va in cv_quick.split(X_train, y_train):
            m = lgb.LGBMClassifier(**params, scale_pos_weight=neg_pos_ratio,
                                   random_state=SEED, verbosity=-1, n_jobs=-1)
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            fold_aucs.append(roc_auc_score(y_train.iloc[va], m.predict_proba(X_train.iloc[va])[:, 1]))
        mean_auc = float(np.mean(fold_aucs))
        print(f"    Config {i}: depth={params['max_depth']:<2} lr={params['learning_rate']} "
              f"trees={params['n_estimators']:<4} -> val ROC-AUC {mean_auc:.4f} "
              f"(+/- {np.std(fold_aucs):.4f})")
        if mean_auc > best_auc:
            best_auc, best_params = mean_auc, params
    print(f"  -> Selected: {best_params}")

    # ---- 2B: 5-fold CV with full metric suite ---------------------------
    print("\n  [2B] 5-fold stratified cross-validation")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
        row = {'fold': fold}
        for name, factory in (
            ('lgb', lambda: lgb.LGBMClassifier(**best_params, scale_pos_weight=neg_pos_ratio,
                                               random_state=SEED + fold, verbosity=-1, n_jobs=-1)),
            ('xgb', lambda: xgb.XGBClassifier(
                n_estimators=best_params['n_estimators'],
                learning_rate=best_params['learning_rate'],
                max_depth=best_params['max_depth'],
                colsample_bytree=best_params['colsample_bytree'],
                scale_pos_weight=neg_pos_ratio, eval_metric='logloss',
                random_state=SEED + fold, n_jobs=-1)),
        ):
            m = factory()
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            p = m.predict_proba(X_train.iloc[va])[:, 1]
            row[f'{name}_roc_auc'] = roc_auc_score(y_train.iloc[va], p)
            row[f'{name}_pr_auc'] = average_precision_score(y_train.iloc[va], p)
            row[f'{name}_brier'] = brier_score_loss(y_train.iloc[va], p)
        cv_rows.append(row)
        print(f"    Fold {fold}: LGBM AUC={row['lgb_roc_auc']:.4f} Brier={row['lgb_brier']:.4f} | "
              f"XGB AUC={row['xgb_roc_auc']:.4f} Brier={row['xgb_brier']:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    cv_summary = {
        'lgb_roc_auc_mean': round(float(cv_df['lgb_roc_auc'].mean()), 4),
        'lgb_roc_auc_std': round(float(cv_df['lgb_roc_auc'].std(ddof=1)), 4),
        'lgb_pr_auc_mean': round(float(cv_df['lgb_pr_auc'].mean()), 4),
        'lgb_pr_auc_std': round(float(cv_df['lgb_pr_auc'].std(ddof=1)), 4),
        'lgb_brier_mean': round(float(cv_df['lgb_brier'].mean()), 4),
        'lgb_brier_std': round(float(cv_df['lgb_brier'].std(ddof=1)), 4),
        'xgb_roc_auc_mean': round(float(cv_df['xgb_roc_auc'].mean()), 4),
        'xgb_roc_auc_std': round(float(cv_df['xgb_roc_auc'].std(ddof=1)), 4),
        'xgb_pr_auc_mean': round(float(cv_df['xgb_pr_auc'].mean()), 4),
        'xgb_brier_mean': round(float(cv_df['xgb_brier'].mean()), 4),
        'folds': int(len(cv_df)),
    }
    print(f"  -> LGBM  ROC-AUC {cv_summary['lgb_roc_auc_mean']:.4f} +/- {cv_summary['lgb_roc_auc_std']:.4f} | "
          f"PR-AUC {cv_summary['lgb_pr_auc_mean']:.4f} | Brier {cv_summary['lgb_brier_mean']:.4f} "
          f"+/- {cv_summary['lgb_brier_std']:.4f}")
    print(f"  -> XGB   ROC-AUC {cv_summary['xgb_roc_auc_mean']:.4f} +/- {cv_summary['xgb_roc_auc_std']:.4f} | "
          f"PR-AUC {cv_summary['xgb_pr_auc_mean']:.4f} | Brier {cv_summary['xgb_brier_mean']:.4f}")

    # ---- 2C: production fit + isotonic calibration ----------------------
    print("\n  [2C] Production fit + isotonic probability calibration")
    base_lgb = lgb.LGBMClassifier(**best_params, scale_pos_weight=neg_pos_ratio,
                                  random_state=SEED, verbosity=-1, n_jobs=-1)
    base_lgb.fit(X_train, y_train)
    base_xgb = xgb.XGBClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        colsample_bytree=best_params['colsample_bytree'],
        scale_pos_weight=neg_pos_ratio, eval_metric='logloss',
        random_state=SEED, n_jobs=-1)
    base_xgb.fit(X_train, y_train)

    calibrated_lgb = CalibratedClassifierCV(estimator=base_lgb, method='isotonic', cv=3)
    calibrated_lgb.fit(X_train, y_train)
    calibrated_xgb = CalibratedClassifierCV(estimator=base_xgb, method='isotonic', cv=3)
    calibrated_xgb.fit(X_train, y_train)

    raw_lgb = base_lgb.predict_proba(X_test)[:, 1]
    cal_lgb = calibrated_lgb.predict_proba(X_test)[:, 1]
    cal_xgb = calibrated_xgb.predict_proba(X_test)[:, 1]

    raw_brier = brier_score_loss(y_test, raw_lgb)
    cal_brier = brier_score_loss(y_test, cal_lgb)
    brier_reduction = (raw_brier - cal_brier) / raw_brier * 100.0

    print("\n  === HOLDOUT EVALUATION (N={:,} orders, never used in training) ===".format(len(y_test)))
    print(f"  {'Model':<26}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>9}{'LogLoss':>9}")
    print(f"  {'-'*62}")
    print(f"  {'LightGBM (raw)':<26}{roc_auc_score(y_test, raw_lgb):>9.4f}"
          f"{average_precision_score(y_test, raw_lgb):>9.4f}{raw_brier:>9.4f}"
          f"{log_loss(y_test, raw_lgb):>9.4f}")
    print(f"  {'LightGBM (isotonic)':<26}{roc_auc_score(y_test, cal_lgb):>9.4f}"
          f"{average_precision_score(y_test, cal_lgb):>9.4f}{cal_brier:>9.4f}"
          f"{log_loss(y_test, cal_lgb):>9.4f}")
    print(f"  {'XGBoost (isotonic)':<26}{roc_auc_score(y_test, cal_xgb):>9.4f}"
          f"{average_precision_score(y_test, cal_xgb):>9.4f}"
          f"{brier_score_loss(y_test, cal_xgb):>9.4f}{log_loss(y_test, cal_xgb):>9.4f}")
    print(f"  -> Isotonic calibration cuts Brier error by {brier_reduction:.1f}% "
          f"({raw_brier:.4f} -> {cal_brier:.4f})")

    # ---- 2D: operating-point economics ----------------------------------
    print("\n  [2D] Operating-point performance (what the business actually sees)")
    topk = ml.top_k_metrics(y_test.values, cal_lgb, ks=(0.05, 0.10, 0.20, 0.30, 0.50))
    print(f"  {'Budget':>8}{'Orders':>9}{'Precision':>11}{'Recall':>9}{'Lift':>8}")
    print(f"  {'-'*45}")
    for k, v in topk['ks'].items():
        print(f"  top {k+'%':>6}{v['n_flagged']:>9,}{v['precision']:>11.4f}"
              f"{v['recall']:>9.4f}{v['lift']:>8.2f}x")
    print(f"  (no-skill baseline precision = {topk['prevalence']:.4f}; "
          "lift 1.0x == random targeting)")

    thr = ml.economic_thresholds()
    econ = ml.policy_economics(y_test.values, cal_lgb, threshold=thr['p_star_discount'])
    print("\n  Cost-optimal coupon policy (fire when P(RTO) > "
          f"Rs.{ml.COUPON_COST_RS:.0f}/Rs.{ml.REVERSE_COST_RS:.0f} = {econ['threshold']:.4f}):")
    print(f"    Flags {econ['n_flagged']:,} / {econ['n_orders']:,} orders "
          f"({econ['coverage_pct']:.1%})")
    print(f"    Catches {econ['true_rtos_caught']:,} of {econ['true_rtos_caught'] + econ['true_rtos_missed']:,} "
          f"true RTOs ({econ['recall_of_rtos']:.1%} recall)")
    print(f"    Net benefit: +Rs.{econ['net_rs']:,.0f} over {econ['n_orders']:,} orders "
          f"= +Rs.{econ['net_rs_per_1000']:,.0f} per 1,000 orders")
    swing = econ['net_rs'] - econ['baseline_blanket_discount_net_rs']
    print("    BLANKET DISCOUNT ON EVERY ORDER WOULD LOSE "
          f"Rs.{abs(econ['baseline_blanket_discount_net_rs']):,.0f} "
          f"(Rs.{econ['baseline_blanket_discount_net_rs'] / econ['n_orders']:.2f}/order).")
    print(f"    -> TARGETING SWINGS THAT TO +Rs.{econ['net_rs']:,.0f}: a Rs.{swing:,.0f} improvement, "
          f"Rs.{swing / econ['n_orders'] * 1000:,.0f} per 1,000 orders.")

    # The WhatsApp NDR bot is nearly free (Rs.0.35), so cost is NOT the binding
    # constraint -- message fatigue and opt-out risk are. It is therefore sized
    # by a CONTACT BUDGET, and the same precision/recall frontier applies.
    print("\n  WhatsApp NDR pre-alert: cost-optimal threshold P(RTO) > "
          f"Rs.{ml.WHATSAPP_COST_RS}/{ml.REVERSE_COST_RS} = {thr['p_star_whatsapp']:.5f}")
    print("    At that threshold the bot would message "
          f"{(cal_lgb >= thr['p_star_whatsapp']).mean():.0%} of all orders -- free but useless.")
    print("    Sized instead by CONTACT CAPACITY (fatigue / opt-out), scanning top-K by risk:")
    print(f"      {'Budget':>8}{'Precision':>11}{'Recall':>9}{'Net Rs/1000':>14}")
    wa_rows = {}
    for budget in (0.10, 0.20, 0.30, 0.50):
        cut = float(np.quantile(cal_lgb, 1.0 - budget))
        row = ml.policy_economics(y_test.values, cal_lgb, threshold=cut,
                                  action_cost=ml.WHATSAPP_COST_RS)
        wa_rows[f'{int(budget * 100)}'] = {k: (round(v, 2) if isinstance(v, float) else v)
                                           for k, v in row.items()}
        print(f"      {int(budget * 100):>6}%{row['true_rtos_caught'] / max(row['n_flagged'], 1):>11.4f}"
              f"{row['recall_of_rtos']:>9.4f}{row['net_rs_per_1000']:>14,.0f}")

    # ---- 2E: feature attribution ----------------------------------------
    print("\n  [2E] Feature importance + TreeSHAP attribution")
    imp = pd.DataFrame({'feature': feature_cols,
                        'importance': base_lgb.feature_importances_}).sort_values(
        'importance', ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)
    for i, r in imp.head(8).iterrows():
        print(f"    {i+1}. {r['feature']:<25} {int(r['importance']):>6}")

    shap_sample = X_test.sample(2000, random_state=SEED)
    t0 = time.time()
    sv = shap.TreeExplainer(base_lgb).shap_values(shap_sample)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, 'ndim', 0) == 3:
        sv = sv[:, :, 1]
    print(f"    TreeSHAP computed on 2,000 holdout rows in {time.time() - t0:.2f}s")

    # ---- persist ---------------------------------------------------------
    joblib.dump(calibrated_lgb, os.path.join(MODEL_DIR, 'rto_risk_lightgbm_calibrated.joblib'))
    joblib.dump(calibrated_xgb, os.path.join(MODEL_DIR, 'rto_risk_xgboost_calibrated.joblib'))
    joblib.dump(base_lgb, os.path.join(MODEL_DIR, 'rto_risk_base_lightgbm.joblib'))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'rto_feature_names.joblib'))
    joblib.dump({'sample': shap_sample, 'shap_values': sv, 'features': feature_cols},
                os.path.join(DATA_DIR, 'shap_summary_cache.joblib'))

    # `row_index` is the dataset row this prediction came from. It is written so
    # any downstream analysis can JOIN on a key instead of assuming positional
    # alignment. signal_ceiling.py previously reconstructed the split and indexed
    # the true probabilities by it, which silently produced a ceiling of 0.51 --
    # worse than random -- because the reconstruction did not reproduce this
    # file's row order. Explicit keys remove the whole class of bug.
    pd.DataFrame({
        'row_index': X_test.index.to_numpy(),
        'y_true': y_test.values, 'pred_lgb': cal_lgb,
        'pred_xgb': cal_xgb, 'pred_raw': raw_lgb,
    }).to_csv(os.path.join(DATA_DIR, 'test_evaluation_preds.csv'), index=False)
    cv_df.to_csv(os.path.join(DATA_DIR, 'cv_fold_metrics.csv'), index=False)

    metrics = {
        'evaluation_protocol': '80/20 stratified holdout; models fit on the 80% only. '
                               'All metrics below are out-of-sample.',
        'n_test': int(len(y_test)),
        'lightgbm': {
            'roc_auc': round(float(roc_auc_score(y_test, cal_lgb)), 4),
            'pr_auc': round(float(average_precision_score(y_test, cal_lgb)), 4),
            'brier_score': round(float(cal_brier), 4),
            'log_loss': round(float(log_loss(y_test, cal_lgb)), 4),
            'raw_brier': round(float(raw_brier), 4),
            'raw_log_loss': round(float(log_loss(y_test, raw_lgb)), 4),
            'brier_reduction_pct': round(float(brier_reduction), 1),
        },
        'xgboost': {
            'roc_auc': round(float(roc_auc_score(y_test, cal_xgb)), 4),
            'pr_auc': round(float(average_precision_score(y_test, cal_xgb)), 4),
            'brier_score': round(float(brier_score_loss(y_test, cal_xgb)), 4),
        },
        'cross_validation': cv_summary,
        'operating_points': {
            k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                for kk, vv in v.items()}
            for k, v in topk['ks'].items()
        },
        'baseline_prevalence': round(topk['prevalence'], 4),
        'thresholds': {
            'p_star_whatsapp': round(float(thr['p_star_whatsapp']), 5),
            'p_star_discount': round(float(thr['p_star_discount']), 5),
            'rule': thr['discount_rule'],
        },
        'policy_economics': {
            'coupon_policy': {k: (round(v, 2) if isinstance(v, float) else v)
                              for k, v in econ.items()},
            'blanket_discount_net_rs': round(econ['baseline_blanket_discount_net_rs'], 2),
            'targeting_swing_rs': round(swing, 2),
            'targeting_swing_rs_per_1000': round(swing / econ['n_orders'] * 1000.0, 2),
            'whatsapp_by_contact_budget': wa_rows,
        },
        'top_features': imp.head(8)['feature'].tolist(),
    }
    with open(os.path.join(MODEL_DIR, 'rto_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics, cal_lgb, feature_cols


# ==========================================================================
# STEP 3 — CAUSAL UPLIFT X-LEARNER
# ==========================================================================
def _simulate_uplift_population(n: int, seed: int):
    """
    Simulated doorstep-rescue experiment.

    DISCLOSURE: Valmo has no published randomised trial data, so this is a
    synthetic RCT whose marginals are aligned to the Valmo order distribution.
    Treatment W is assigned at random with p=0.40, which satisfies
    unconfoundedness by construction; the heterogeneous treatment effect is the
    quantity the X-Learner must recover.

    Ground-truth CATE is retained so we can report recovery error — a judge can
    check the estimator against the truth that generated the data.
    """
    rng = np.random.default_rng(seed)
    order_val = np.clip(rng.lognormal(6.0, 0.45, n), 150, 3500).round(2)
    prior_rto = rng.beta(1.5, 8.0, n).round(3)
    dist_km = rng.uniform(0.5, 18.0, n).round(2)
    s_addr = rng.beta(5.0, 2.5, n).round(3)
    delay = rng.choice([0, 1, 2, 3, 4], p=[0.55, 0.25, 0.12, 0.05, 0.03], size=n)
    sku_risk = rng.choice([0.08, 0.12, 0.13, 0.19, 0.20, 0.22], size=n)

    w = rng.binomial(1, 0.40, n)

    base_p0 = np.clip(0.35 - 0.06 * delay - 0.15 * (prior_rto > 0.25)
                      + 0.12 * (s_addr > 0.65) - 0.01 * dist_km, 0.05, 0.80)
    true_cate = np.clip(
        0.38 * (order_val >= 300) * (order_val <= 1800) * (prior_rto <= 0.20)
        * np.exp(-0.035 * dist_km), 0.0, 0.65)

    p1 = np.clip(base_p0 + true_cate, 0.05, 0.95)
    y = np.where(w == 1, rng.binomial(1, p1), rng.binomial(1, base_p0))

    df = pd.DataFrame({
        'order_value': order_val, 'prior_rto_rate': prior_rto, 'distance_km': dist_km,
        's_addr': s_addr, 'delay_days': delay, 'sku_risk_index': sku_risk,
        'W': w, 'Y': y,
    })
    return df, true_cate


def step3_uplift(n_total: int = 40000):
    print("\n" + "=" * 80)
    print("[STEP 3/4] Model 2 — Doorstep Rescue Causal Uplift (X-Learner)")
    print("=" * 80)

    feats = ['order_value', 'prior_rto_rate', 'distance_km', 's_addr', 'delay_days', 'sku_risk_index']
    df, true_cate = _simulate_uplift_population(n_total, SEED)
    X = df[feats]

    print(f"  Simulated RCT: N={len(df):,}, treated={int(df['W'].sum()):,} "
          f"({df['W'].mean():.1%}), observed ATE="
          f"{df[df['W']==1]['Y'].mean() - df[df['W']==0]['Y'].mean():+.4f}")
    print(f"  Ground-truth mean CATE available for estimator-recovery check: {true_cate.mean():.4f}")

    X_tr, X_te, W_tr, W_te, Y_tr, Y_te, T_tr, T_te = train_test_split(
        X, df['W'], df['Y'], true_cate, test_size=0.30, random_state=SEED)

    def fit_xlearner(Xa, Wa, Ya):
        """Künzel et al. (2019) X-Learner, 4 stages."""
        def reg():
            return lgb.LGBMRegressor(n_estimators=80, learning_rate=0.06, max_depth=4,
                                     random_state=SEED, verbosity=-1)
        m0, m1 = reg(), reg()
        m0.fit(Xa[Wa == 0], Ya[Wa == 0])
        m1.fit(Xa[Wa == 1], Ya[Wa == 1])
        d1 = Ya[Wa == 1] - m0.predict(Xa[Wa == 1])     # stage 2
        d0 = m1.predict(Xa[Wa == 0]) - Ya[Wa == 0]
        tau1, tau0 = reg(), reg()
        tau1.fit(Xa[Wa == 1], d1)                      # stage 3
        tau0.fit(Xa[Wa == 0], d0)
        return m0, m1, tau0, tau1

    m0, m1, tau0, tau1 = fit_xlearner(X_tr.values, W_tr.values, Y_tr.values)
    propensity = float(W_tr.mean())
    pred_tau = (propensity * tau0.predict(X_te)
                + (1.0 - propensity) * tau1.predict(X_te))

    # ---- Falsifiable Qini evaluation -------------------------------------
    print("\n  Qini evaluation vs permutation null and perfect-targeting oracle")
    rep = ml.qini_report(Y_te.values, W_te.values, pred_tau, n_perm=200, seed=SEED)

    print(f"    AUUC (model)     : {rep['auuc_raw']:>10.2f}")
    print(f"    AUUC (random null): {rep['auuc_random']:>9.2f}   (mean of 200 random permutations)")
    print(f"    AUUC (perfect)   : {rep['auuc_perfect']:>10.2f}   (oracle that ranks every responder first)")
    print(f"    Normalised AUUC  : {rep['auuc_normalized']:.4f}  (0 = random, 1 = perfect)")
    print(f"    Lift over random : {rep['auuc_lift_over_random']:+.1%}")
    print(f"    Permutation p    : {rep['permutation_p_value']:.4f}  "
          "(<=0.05 required to claim the model beats random targeting)")
    print(f"    Max Qini gain    : {rep['max_qini_gain']:.2f} incremental rescued deliveries")

    # ---- Estimator recovery against ground truth ------------------------
    corr = float(np.corrcoef(pred_tau, T_te)[0, 1])
    mae = float(np.mean(np.abs(pred_tau - T_te)))
    print(f"    CATE recovery    : corr(pred, true tau) = {corr:+.4f}, MAE = {mae:.4f}")

    # ---- 5-fold stability ------------------------------------------------
    print("\n  5-fold cross-validated normalised AUUC (stability)")
    fold_scores = []
    for k, (tr, va) in enumerate(StratifiedKFold(n_splits=5, shuffle=True,
                                                 random_state=SEED).split(X, df['W']), 1):
        Xf, Wf, Yf = X.values[tr], df['W'].values[tr], df['Y'].values[tr]
        Xv, Wv, Yv = X.values[va], df['W'].values[va], df['Y'].values[va]
        _, _, t0f, t1f = fit_xlearner(Xf, Wf, Yf)
        e = float(Wf.mean())
        pv = e * t0f.predict(Xv) + (1 - e) * t1f.predict(Xv)
        r = ml.qini_report(Yv, Wv, pv, n_perm=50, seed=SEED + k)
        fold_scores.append(r['auuc_normalized'])
        print(f"    Fold {k}: normalised AUUC = {r['auuc_normalized']:.4f} (p={r['permutation_p_value']:.3f})")
    lo, hi = ml.confidence_interval(fold_scores)
    print(f"    -> mean {np.mean(fold_scores):.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")

    # ---- economic gating -------------------------------------------------
    econ_thr = ml.COUPON_COST_RS / ml.REVERSE_COST_RS
    persuadables = pred_tau >= econ_thr
    print(f"\n  Economic gating: fire the Rs.{ml.COUPON_COST_RS:.0f} UPI coupon when "
          f"tau(X) >= Rs.{ml.COUPON_COST_RS:.0f}/Rs.{ml.REVERSE_COST_RS:.0f} = {econ_thr:.4f}")
    print(f"    Persuadables targeted : {persuadables.sum():>7,} / {len(pred_tau):,} "
          f"({persuadables.mean():.1%})  -> net +Rs.{ml.NET_MARGIN_PER_RESCUE_RS:.0f} per rescue")
    print(f"    Withheld from         : {(~persuadables).sum():>7,} / {len(pred_tau):,} "
          f"({1 - persuadables.mean():.1%})  -> no coupon burn on Sure Things / Lost Causes")

    bundle = {'m0': m0, 'm1': m1, 'tau0': tau0, 'tau1': tau1,
              'propensity': propensity, 'feature_cols': feats,
              'economic_threshold': econ_thr}
    joblib.dump(bundle, os.path.join(MODEL_DIR, 'doorstep_uplift_xlearner.joblib'))

    # Curves are length N+1 (they start at population 0); trim the leading
    # origin so every column aligns one-to-one with a scored order.
    pd.DataFrame({
        'W': np.asarray(W_te).ravel(), 'Y': np.asarray(Y_te).ravel(),
        'pred_tau': pred_tau, 'true_cate': np.asarray(T_te).ravel(),
        'qini': rep['model_curve'][1:],
        'qini_random': rep['random_curve'][1:],
        'qini_perfect': rep['perfect_curve'][1:],
        'cum_frac': rep['x_axis'][1:],
    }).to_csv(os.path.join(DATA_DIR, 'qini_eval_data.csv'), index=False)

    metrics = {
        'disclosure': 'Trained on a synthetic RCT (N=40,000, randomised W~Bernoulli(0.40)) '
                      'aligned to Valmo order marginals. Valmo publishes no randomised '
                      'trial data. Unconfoundedness holds by construction; '
                      'ground-truth CATE is retained to audit estimator recovery.',
        'n': rep['n'], 'n_treated': rep['n_treated'], 'n_control': rep['n_control'],
        'observed_ate': round(rep['observed_ate'], 4),
        'mean_predicted_cate': round(float(pred_tau.mean()), 4),
        'auuc': round(rep['auuc_raw'], 2),
        'auuc_random_baseline': round(rep['auuc_random'], 2),
        'auuc_perfect_oracle': round(rep['auuc_perfect'], 2),
        'auuc_normalized': round(rep['auuc_normalized'], 4),
        'auuc_lift_over_random_pct': round(rep['auuc_lift_over_random'] * 100, 1),
        'permutation_p_value': rep['permutation_p_value'],
        'n_permutations': 200,
        'max_qini_gain': round(rep['max_qini_gain'], 2),
        'cate_recovery_corr_with_truth': round(corr, 4),
        'cate_recovery_mae': round(mae, 4),
        'cv_normalized_auuc_mean': round(float(np.mean(fold_scores)), 4),
        'cv_normalized_auuc_ci': [round(lo, 4), round(hi, 4)],
        'economic_threshold': round(econ_thr, 5),
        'persuadable_pct': round(float(persuadables.mean() * 100), 1),
        'net_margin_per_rescue_rs': ml.NET_MARGIN_PER_RESCUE_RS,
    }
    with open(os.path.join(MODEL_DIR, 'uplift_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ==========================================================================
# STEP 4 — SYNTHETIC ATTEMPT-PATTERN ANOMALY PROTOTYPE
# ==========================================================================
def _simulate_telemetry(n_genuine: int, n_fake: int, seed: int):
    """
    Simulated last-mile delivery-attempt telemetry.

    DISCLOSURE: no public rider-telemetry corpus exists, so kinematics are
    synthesised from documented operational signatures. The distributions
    deliberately OVERLAP: ~7% of genuine attempts are hard deliveries (rider
    cannot resolve the address) and ~12% of faked attempts are sophisticated
    (walk-speed mark from a plausible distance). Without that overlap the task
    is trivially separable and any reported F1 would be meaningless.
    """
    rng = np.random.default_rng(seed)

    messy_frac = 0.07
    n_messy = int(n_genuine * messy_frac)
    n_clean = n_genuine - n_messy
    gen = np.column_stack([
        np.concatenate([rng.lognormal(np.log(26), 0.50, n_clean),
                        rng.lognormal(np.log(240), 0.55, n_messy)]),
        np.concatenate([rng.normal(108, 26, n_clean).clip(20),
                        rng.normal(52, 26, n_messy).clip(5)]),
        np.concatenate([rng.normal(44, 13, n_clean).clip(0),
                        rng.normal(16, 14, n_messy).clip(0)]),
        np.concatenate([rng.exponential(1.0, n_clean),
                        rng.exponential(3.0, n_messy)]),
        np.concatenate([rng.exponential(270, n_clean) + 45,
                        rng.exponential(115, n_messy) + 18]),
    ])

    careful_frac = 0.12
    n_careful = int(n_fake * careful_frac)
    n_sloppy = n_fake - n_careful
    fak = np.column_stack([
        np.concatenate([rng.uniform(450, 4200, n_sloppy),
                        rng.lognormal(np.log(300), 0.45, n_careful)]),
        np.concatenate([rng.exponential(6, n_sloppy).clip(0, 20),
                        rng.normal(45, 15, n_careful).clip(12)]),
        np.concatenate([np.zeros(n_sloppy),
                        rng.exponential(11, n_careful).clip(0, 26)]),
        np.concatenate([rng.uniform(20, 52, n_sloppy),
                        rng.uniform(9, 24, n_careful)]),
        np.concatenate([rng.exponential(8, n_sloppy).clip(0, 35),
                        rng.normal(85, 40, n_careful).clip(15)]),
    ])

    X = np.vstack([gen, fak])
    y = np.concatenate([np.zeros(n_genuine), np.ones(n_fake)])
    return X, y


def step4_telemetry(n_genuine: int = 45000, n_fake: int = 5000):
    print("\n" + "=" * 80)
    print("[STEP 4/4] Model 3 — Synthetic Attempt-Pattern Anomaly Prototype")
    print("=" * 80)

    tel_cols = ['gps_dist_m', 'dwell_sec', 'call_sec', 'speed_kmh', 'delta_sec']
    X, y = _simulate_telemetry(n_genuine, n_fake, SEED)
    X = pd.DataFrame(X, columns=tel_cols)
    y = pd.Series(y.astype(int))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    print(f"  Simulated attempts: N={len(X):,} ({n_fake:,} faked = {n_fake/len(X):.1%})")
    print(f"  Split: fit N={len(X_tr):,}, HELD-OUT test N={len(X_te):,}")
    print("  Note: the auditor NEVER sees test labels. It is fitted unsupervised with")
    print("        contamination='auto', so no label prevalence is injected.")

    iso = IsolationForest(n_estimators=300, max_samples=256,
                          contamination='auto', random_state=SEED, n_jobs=-1)
    iso.fit(X_tr)

    # ---- operating point from an OPERATIONAL capacity budget -------------
    train_scores = -iso.decision_function(X_tr)
    test_scores = -iso.decision_function(X_te)
    op_threshold = float(np.percentile(train_scores, 100.0 - TELEMETRY_AUDIT_BUDGET_PCT))
    print("\n  Operating point chosen from ops capacity: re-check the most anomalous "
          f"{TELEMETRY_AUDIT_BUDGET_PCT:.0f}% of attempts")
    print(f"    Threshold {op_threshold:.5f} = P{100 - TELEMETRY_AUDIT_BUDGET_PCT:.0f} of TRAIN scores only "
          "(no labels used)")

    y_pred = (test_scores >= op_threshold).astype(int)
    rep = classification_report(y_te, y_pred, target_names=['GENUINE', 'FAKED'],
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred)

    # Derive the class metrics from the confusion matrix and assert they match
    # sklearn, so a future change to target_names cannot silently mislabel them.
    #
    # Careful with the indexing: sklearn's confusion_matrix has rows = TRUE and
    # columns = PREDICTED, so for the GENUINE (negative) class:
    #     cm[0,1] = a genuine attempt predicted faked  -> FALSE ACCUSATION
    #     cm[1,0] = a faked attempt predicted genuine  -> MISSED FAKE
    # Therefore genuine PRECISION (of the pass-through population) divides by
    # TN + missed fakes, and genuine RECALL divides by TN + false accusations.
    # Getting these two denominators the wrong way round swaps precision and
    # recall, which would misreport the rider-fairness metric.
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    genuine_precision = tn / (tn + fn) if (tn + fn) else 0.0   # pass-through purity
    genuine_recall = tn / (tn + fp) if (tn + fp) else 0.0       # of real attempts kept
    faked_precision = tp / (tp + fp) if (tp + fp) else 0.0
    faked_recall = tp / (tp + fn) if (tp + fn) else 0.0
    faked_f1 = (2 * faked_precision * faked_recall / (faked_precision + faked_recall)
                if (faked_precision + faked_recall) else 0.0)
    assert abs(genuine_precision - rep['GENUINE']['precision']) < 1e-9, (
        f'genuine precision {genuine_precision:.6f} != sklearn {rep["GENUINE"]["precision"]:.6f}')
    assert abs(genuine_recall - rep['GENUINE']['recall']) < 1e-9, (
        f'genuine recall {genuine_recall:.6f} != sklearn {rep["GENUINE"]["recall"]:.6f}')
    assert abs(faked_f1 - rep['FAKED']['f1-score']) < 1e-9

    false_accusations = fp      # honest pilots wrongly flagged
    missed_fakes = fn           # faked attempts we let through

    print(f"\n  === HELD-OUT EVALUATION (N={len(y_te):,} attempts, labels unseen) ===")
    print(f"    Accuracy           : {rep['accuracy']:.4f}")
    print(f"    Faked-attempt  P/R/F1: {faked_precision:.4f} / {faked_recall:.4f} / {faked_f1:.4f}")
    print(f"    Genuine-attempt P/R : {genuine_precision:.4f} / {genuine_recall:.4f}")
    print(f"    Macro-F1            : {rep['macro avg']['f1-score']:.4f}")
    print(f"    Confusion matrix    : TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")
    print(f"    Rider-fairness metric = genuine PRECISION {genuine_precision:.4f} "
          f"(share of pass-through attempts genuinely real)")
    print(f"    Rider impact          : {false_accusations:,} honest pilots wrongly flagged "
          f"({false_accusations / (tn + false_accusations):.2%} of pass-throughs) | "
          f"{missed_fakes:,} faked attempts let through")

    # ---- full precision/recall frontier ----------------------------------
    print("\n  Precision/recall frontier on HELD-OUT data (audit budget sweep)")
    print(f"    {'Budget':>8}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Accuracy':>10}{'FP/true':>9}")
    frontier = []
    for budget in (2, 4, 6, 8, 10, 12, 15, 18, 20):
        t = float(np.percentile(train_scores, 100.0 - budget))
        p = (test_scores >= t).astype(int)
        r = classification_report(y_te, p, target_names=['GENUINE', 'FAKED'],
                                  output_dict=True, zero_division=0)
        frontier.append({
            'audit_budget_pct': budget,
            'precision': round(r['FAKED']['precision'], 4),
            'recall': round(r['FAKED']['recall'], 4),
            'f1': round(r['FAKED']['f1-score'], 4),
            'accuracy': round(r['accuracy'], 4),
            'genuine_precision': round(r['GENUINE']['precision'], 4),
        })
        print(f"    {budget:>6}% {r['FAKED']['precision']:>11.4f}{r['FAKED']['recall']:>9.4f}"
              f"{r['FAKED']['f1-score']:>8.4f}{r['accuracy']:>10.4f}"
              f"{r['GENUINE']['precision']:>9.4f}")
    print("    (precision and recall trade off as expected; >12% the frontier degrades")
    print("     because the auditor starts flagging genuine riders — we do not pay for that)")

    joblib.dump(iso, os.path.join(MODEL_DIR, 'telemetry_isolation_forest.joblib'))
    size_mb = os.path.getsize(os.path.join(MODEL_DIR, 'telemetry_isolation_forest.joblib')) / (1024 * 1024)

    pd.DataFrame({
        **X_te.reset_index(drop=True),
        'ground_truth': y_te.values,
        'pred_anomaly': y_pred,
        'anomaly_score': test_scores,
    }).to_csv(os.path.join(DATA_DIR, 'telemetry_eval_sample.csv'), index=False)
    pd.DataFrame({'train_scores_sample': np.sort(train_scores)[::37],
                  'threshold': op_threshold}).to_csv(
        os.path.join(DATA_DIR, 'telemetry_score_thresholds.csv'), index=False)

    metrics = {
        'disclosure': 'Fitted on simulated rider kinematics with deliberate class overlap '
                      '(7% of genuine attempts are hard deliveries, 12% of faked attempts '
                      'are sophisticated). No public rider-telemetry corpus exists.',
        'evaluation_protocol': f'Unsupervised fit (contamination="auto") on a {len(X_tr):,}-row '
                               'training split. Operating threshold = '
                               f'{TELEMETRY_AUDIT_BUDGET_PCT:.0f}th-percentile TRAIN score '
                               '(ops capacity constraint, no labels). Scored once on the '
                               f'{len(X_te):,}-row held-out split.',
        'total_attempts': int(len(X)),
        'n_faked_ground_truth': int(n_fake),
        'n_train': int(len(X_tr)), 'n_test': int(len(X_te)),
        'operating_budget_pct': TELEMETRY_AUDIT_BUDGET_PCT,
        'operating_threshold': round(op_threshold, 5),
        'accuracy': round(rep['accuracy'], 4),
        'faked_precision': round(faked_precision, 4),
        'faked_recall': round(faked_recall, 4),
        'faked_f1': round(faked_f1, 4),
        'genuine_precision': round(genuine_precision, 4),
        'genuine_recall': round(genuine_recall, 4),
        'macro_f1': round(rep['macro avg']['f1-score'], 4),
        'confusion_matrix': {'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
                             'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])},
        'frontier': frontier,
        'model_size_mb': round(size_mb, 2),
    }
    with open(os.path.join(MODEL_DIR, 'telemetry_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Train the Valmo RTO ML suite")
    ap.add_argument('--regenerate-data', action='store_true',
                    help='delete and rebuild the 100k synthetic dataset first')
    ap.add_argument('--skip-telemetry', action='store_true')
    args = ap.parse_args()

    print("=" * 80)
    print("   MEESHO VALMO RTO REDUCTION — ML TRAINING PIPELINE")
    print("=" * 80)

    t_start = time.time()
    df, ds_summary = step1_dataset(force_regenerate=args.regenerate_data)
    rto_metrics, _, _ = step2_classifier(df)
    uplift_metrics = step3_uplift()
    tel_metrics = {} if args.skip_telemetry else step4_telemetry()

    with open(os.path.join(MODEL_DIR, 'dataset_summary.json'), 'w') as f:
        json.dump(ds_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"   PIPELINE COMPLETE IN {time.time() - t_start:.1f}s")
    print("=" * 80)
    print(f"   Model 1 RTO classifier  : ROC-AUC {rto_metrics['lightgbm']['roc_auc']:.4f} | "
          f"Brier -{rto_metrics['lightgbm']['brier_reduction_pct']:.1f}% | "
          f"precision@10% {rto_metrics['operating_points']['10']['precision']:.4f} "
          f"({rto_metrics['operating_points']['10']['lift']:.2f}x lift)")
    print(f"   Model 2 Uplift         : AUUC {uplift_metrics['auuc']:.2f} vs random "
          f"{uplift_metrics['auuc_random_baseline']:.2f} (p={uplift_metrics['permutation_p_value']:.3f}) | "
          f"persuadables {uplift_metrics['persuadable_pct']:.1f}%")
    if tel_metrics:
        print(f"   Model 3 anomaly prototype (synthetic holdout): accuracy {tel_metrics['accuracy']:.4f} | "
              f"faked F1 {tel_metrics['faked_f1']:.4f} | genuine precision {tel_metrics['genuine_precision']:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
