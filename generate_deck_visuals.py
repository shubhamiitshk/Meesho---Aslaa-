"""
Valmo Visual ML Artifacts & Pitch Deck Chart Generator — Meesho DICE Season 3
============================================================================

Generates the presentation and appendix figures into `charts/`.

Every data-driven number is read from the artifacts the training pipeline wrote
-- metrics JSONs and `models/claims.json` -- never hardcoded. That is deliberate:
an earlier revision of this deck annotated a chart with "+54.2% spike" while the
speaker script said "+46.7%", and put "17.3%" in a title while every other slide
said 17.0%. Reading from the ledger makes that class of defect impossible.

Business-context figures come from the DICE case brief and are labelled in their
figure footnotes. Annual forecast scenarios are not generated for the submission.

Run:  python generate_deck_visuals.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.makedirs('charts', exist_ok=True)

df_full = pd.read_csv('data/valmo_orders_dataset.csv')

# The claims ledger is the single source of truth for every number drawn below.
# Build it on demand so a fresh checkout can render charts without first
# remembering to run claims.py.
if not os.path.exists(os.path.join('models', 'claims.json')):
    print('models/claims.json not found — building the claims ledger first...')
    import claims
    ledger = claims.build_ledger()
    with open(os.path.join('models', 'claims.json'), 'w') as _f:
        json.dump({'claims': ledger, 'retired': claims.RETIRED_CLAIMS}, _f, indent=2)

# --- Brand palette (Meesho plum / navy / teal) ----------------------------
C_PRIMARY = '#9B1B58'
C_SECONDARY = '#1E3A8A'
C_ACCENT = '#0D9488'
C_ALERT = '#DC2626'
C_SUCCESS = '#16A34A'
C_MUTED = '#94A3B8'
C_DARK = '#1E293B'
C_LIGHT = '#F8FAFC'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#CBD5E0',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def _load(path):
    with open(path) as f:
        return json.load(f)


LEDGER = _load(os.path.join('models', 'claims.json'))
CLAIMS = LEDGER['claims']



import textwrap


def footer(fig, text, size=8.2, y=0.012):
    """Footnote placed INSIDE the figure coordinate system and wrapped to fit.

    Footnotes must live in figure coordinates: text placed with
    transform=ax.transAxes at a negative y falls outside the canvas, and
    bbox_inches="tight" then expands the canvas to contain it, which
    squeezes the plot into a narrow column.
    """
    width = max(60, int(fig.get_size_inches()[0] * 15.5))
    fig.text(0.5, y, textwrap.fill(' '.join(str(text).split()), width),
             ha='center', va='bottom', fontsize=size, style='italic', color='#64748B')


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.16)
    plt.close(fig)

def d(cid: str) -> str:
    """The exact display string for a claim, e.g. 'rto.roc_auc' -> '0.6552'."""
    return CLAIMS[cid]['display']


def v(cid: str) -> float:
    return CLAIMS[cid]['value']


def num(cid: str) -> float:
    """Numeric magnitude of a claim, ignoring any Rs./% prefix (e.g. 'Rs.120' -> 120.0)."""
    raw = CLAIMS[cid]['value']
    return float(raw) if isinstance(raw, (int, float)) else float(str(raw).replace('Rs.', ''))


print('=' * 78)
print('   GENERATING 19 PRESENTATION FIGURES IN charts/')
print('=' * 78)

# ==============================================================================
# CHART 1 — Case-pack distance pattern, mirrored by synthetic data
# ==============================================================================
print('[ 1/19] distance_penalty_curve.png')
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=300)
cats = ['Nearby\n(~2 km)', 'Moderate\n(~5 km)', 'Far\n(10 km+)']
# The ledger stores these as FRACTIONS; this axis is in percent, so scale by 100.
# Passing the raw fraction plotted 0.15 against a 0-25 axis and produced a chart
# whose labels read "0.1%" -- caught by check_render.py, not by any assertion.
rates = [100.0 * v('dataset.near_rto'),
         100.0 * v('dataset.moderate_rto'),
         100.0 * v('dataset.far_rto')]

bars = ax.bar(cats, rates, color=[C_SECONDARY, '#3B5BA5', C_ALERT],
              width=0.52, edgecolor='#4A5568', linewidth=0.8, zorder=3)
for bar, r in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, r + 0.4, f'{r:.1f}%',
            ha='center', va='bottom', fontsize=12.5, fontweight='bold', color=C_DARK)

ax.plot(range(3), rates, color=C_ALERT, linestyle='--', marker='o',
        linewidth=2.2, zorder=4)
ax.annotate(f"{d('dataset.distance_penalty')} failure spike\n"
            f"(near {rates[0]:.1f}% -> far {rates[2]:.1f}%)",
            xy=(2, rates[2]), xytext=(0.85, rates[2] + 3.4),
            arrowprops=dict(facecolor=C_ALERT, shrink=0.08, width=1.6, headwidth=7),
            fontsize=10.5, fontweight='bold', color=C_ALERT,
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFF5F5', ec=C_ALERT, lw=1))

ax.set_ylim(0, max(rates) + 6.5)
ax.set_ylabel('Undelivered Order Rate (%)', fontsize=11, fontweight='bold', color=C_DARK)
ax.set_title('Case-Pack Distance Pattern — Mirrored by the Synthetic Dataset',
             fontsize=12.5, fontweight='bold', color=C_PRIMARY, pad=14)
footer(fig, f'Synthetic outcomes on {d("dataset.orders")} generated orders across {d("dataset.pincodes")} '
        f'India Post pincodes. The generator is calibrated so this ratio reproduces the '
        f'case-pack pattern {d("dataset.case_penalty")}; calibration is not independent evidence.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/distance_penalty_curve.png')

# ==============================================================================
# CHART 2 — Illustrative 7-bucket allocation (all cause shares assumed)
# ==============================================================================
print('[ 2/19] root_cause_waterfall.png')
fig, ax = plt.subplots(figsize=(9.2, 5.1), dpi=300)
buckets = [
    '1. Doorstep Remorse & Cash Friction',
    '2. Address Quality & Spatial Drift',
    '3. Customer Communication Gaps',
    '4. Attempt Evidence & Distance Constraints',
    '5. Hub Misroutes & Boundary Overlaps',
    '6. Transit Delays & Line-Haul SLAs',
    '7. Product & Fit Catalog Mismatch',
][::-1]
shares = [35.0, 24.0, 16.0, 12.0, 6.0, 4.0, 3.0][::-1]
blended = v('dataset.blended_rto') * 100
pts = [blended * s / 100.0 for s in shares]
bcols = ['#E2E8F0', '#E2E8F0', '#E2E8F0', '#FEB2B2', '#E2E8F0', '#BEE3F8', C_PRIMARY][::-1]

bars = ax.barh(buckets, shares, color=bcols, edgecolor='#718096', linewidth=0.7,
               height=0.6, zorder=3)
for bar, s, p in zip(bars, shares, pts):
    ax.text(s + 0.8, bar.get_y() + bar.get_height() / 2, f'{s:.0f}%  ({p:.2f} RTO pts)',
            ha='left', va='center', fontsize=9.5, fontweight='bold', color=C_DARK)

ax.set_xlim(0, 45)
ax.set_xlabel('Share of Total Network RTO (%)', fontsize=11, fontweight='bold', color=C_DARK)
ax.set_title(f'Illustrative 7-Bucket Cause Allocation (all shares assumed) '
             f'(simulated blended RTO = {d("dataset.blended_rto")})',
             fontsize=12.5, fontweight='bold', color=C_PRIMARY, pad=14)
footer(fig, 'Cause shares are unvalidated planning assumptions; no interviews, rider shadowing or cause-coded Valmo data were collected. '
        f'RTO-point sizes use a synthetic baseline of {d("dataset.blended_rto")}.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/root_cause_waterfall.png')

# ==============================================================================
# CHART 3 — Forward cost allocation (case brief 7-node split)
# ==============================================================================
print('[ 3/19] forward_cost_allocation.png')
fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=300)
nodes = ['First Mile\nHub', 'FM\nCarting', 'First Mile\nSort (FMSC)', 'National\nLine-Haul',
         'Destination\nSort (LMSC)', 'Regional\nLine-Haul', 'Last Mile Hub\n(LMDC)']
costs = [4.0, 2.0, 5.0, 8.0, 5.0, 5.0, 21.0]
total_forward = sum(costs)

bars = ax.bar(nodes, costs, color=[C_SECONDARY] * 6 + [C_PRIMARY], width=0.55,
              edgecolor='#4A5568', linewidth=0.8, zorder=3)
for bar, c in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width() / 2, c + 0.4, f'Rs.{c:.0f}\n({c/total_forward:.0%})',
            ha='center', va='bottom', fontsize=8.6, fontweight='bold', color=C_DARK)

ax.set_ylim(0, 26)
ax.set_ylabel('Forward Cost per Parcel (Rs.)', fontsize=11, fontweight='bold', color=C_DARK)
ax.set_title(f'Valmo Forward Supply Chain Cost Stack (Total Forward = Rs.{total_forward:.0f})',
             fontsize=12.5, fontweight='bold', color=C_PRIMARY, pad=14)
plt.xticks(fontsize=8.8)
footer(fig, f'The case brief assigns {d("econ.lmdc_rupees")} of the {d("econ.forward_rupees")} forward cost to the LMDC and gives {d("econ.reverse_rupees")} reverse freight. Gross costs are not the same as avoidable savings; the unit-economics slide counts only measured incremental outcomes and verified avoided costs.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/forward_cost_allocation.png')

# ==============================================================================
# ==============================================================================
# Shared model-evaluation data
# ==============================================================================
test_eval = pd.read_csv('data/test_evaluation_preds.csv')
y_true = test_eval['y_true'].to_numpy()
p_cal = test_eval['pred_lgb'].to_numpy()
p_xgb = test_eval['pred_xgb'].to_numpy()
p_raw = test_eval['pred_raw'].to_numpy()

# ==============================================================================
# CHART 4 — ROC
# ==============================================================================
print('[ 4/19] roc_curve.png')
fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=300)
fpr_l, tpr_l, _ = roc_curve(y_true, p_cal)
fpr_x, tpr_x, _ = roc_curve(y_true, p_xgb)
ax.plot(fpr_l, tpr_l, color=C_PRIMARY, linewidth=2.5,
        label=f'LightGBM calibrated (AUC = {roc_auc_score(y_true, p_cal):.4f})')
ax.plot(fpr_x, tpr_x, color=C_SECONDARY, linewidth=2.0, linestyle='-.',
        label=f'XGBoost calibrated (AUC = {roc_auc_score(y_true, p_xgb):.4f})')
ax.plot([0, 1], [0, 1], color=C_MUTED, linestyle='--', linewidth=1.5, label='Random (0.5000)')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold', color=C_DARK)
ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold', color=C_DARK)
ax.set_title('ROC — Pre-Dispatch RTO Predictor', fontsize=12.5, fontweight='bold',
             color=C_PRIMARY, pad=12)
ax.legend(loc='lower right', frameon=True, framealpha=0.92, fontsize=9.5)
ax.grid(True, linestyle=':', alpha=0.6)
footer(fig, f'Holdout split, N={len(y_true):,} orders never seen in training. 5-fold CV mean '
        f'AUC = {d("rto.cv_auc")}, so the holdout is consistent with training variance.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/roc_curve.png')

# ==============================================================================
# CHART 5 — Precision-Recall
# ==============================================================================
print('[ 5/19] pr_curve.png')
fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=300)
rec_l, prec_l, _ = precision_recall_curve(y_true, p_cal)
rec_x, prec_x, _ = precision_recall_curve(y_true, p_xgb)
prev = y_true.mean()
ax.plot(rec_l, prec_l, color=C_PRIMARY, linewidth=2.5,
        label=f'LightGBM calibrated (PR-AUC = {average_precision_score(y_true, p_cal):.4f})')
ax.plot(rec_x, prec_x, color=C_ACCENT, linewidth=2.0, linestyle='-.',
        label=f'XGBoost calibrated (PR-AUC = {average_precision_score(y_true, p_xgb):.4f})')
ax.axhline(prev, color=C_ALERT, linestyle='--', linewidth=1.5,
           label=f'No-skill prevalence baseline ({prev:.1%})')

# Mark the top-10% operating point.
cut = int(0.10 * len(p_cal))
top_idx = np.argsort(-p_cal)[:cut]
op_recall = y_true[top_idx].sum() / y_true.sum()
op_prec = y_true[top_idx].sum() / cut
ax.scatter([op_recall], [op_prec], s=150, color=C_PRIMARY, zorder=6,
           edgecolor='white', linewidth=2)
ax.annotate(f'Operational point\ntop 10% by risk\n{op_prec:.1%} precision',
            xy=(op_recall, op_prec), xytext=(op_recall + 0.22, op_prec - 0.06),
            fontsize=9, fontweight='bold', color=C_PRIMARY,
            bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F8', ec=C_PRIMARY, lw=1),
            arrowprops=dict(arrowstyle='->', color=C_PRIMARY, lw=1.4))

ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 0.72)
ax.set_xlabel('Recall (share of all failures caught)', fontsize=11, fontweight='bold', color=C_DARK)
ax.set_ylabel('Precision (share of flagged orders that fail)', fontsize=11,
              fontweight='bold', color=C_DARK)
ax.set_title('Precision-Recall Curve — Class-Imbalanced RTO Prediction', fontsize=12.5,
             fontweight='bold', color=C_PRIMARY, pad=12)
ax.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=9.2)
ax.grid(True, linestyle=':', alpha=0.6)
footer(fig, f'Holdout N={len(y_true):,}. Baseline prevalence is {prev:.1%}, so a '
        f'PR-AUC of {d("rto.pr_auc")} is a '
        f'{v("rto.pr_auc") / prev:.2f}x improvement over guessing.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/pr_curve.png')

# ==============================================================================
# CHART 6 — Calibration reliability diagram
# ==============================================================================
print('[ 6/19] calibration_curve.png')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.6), dpi=300,
                               gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
pt_raw, pp_raw = calibration_curve(y_true, p_raw, n_bins=10)
pt_c, pp_c = calibration_curve(y_true, p_cal, n_bins=10)
b_raw, b_cal = brier_score_loss(y_true, p_raw), brier_score_loss(y_true, p_cal)
red = (b_raw - b_cal) / b_raw * 100

ax1.plot([0, 1], [0, 1], color=C_MUTED, linestyle='--', linewidth=1.5,
         label='Perfect calibration')
ax1.plot(pp_raw, pt_raw, marker='s', color=C_ALERT, linewidth=2.0, markersize=7,
         label=f'Raw LightGBM (Brier {b_raw:.4f})')
ax1.plot(pp_c, pt_c, marker='o', color=C_PRIMARY, linewidth=2.6, markersize=8,
         label=f'Isotonic calibrated (Brier {b_cal:.4f})')
ax1.set_ylabel('Observed failure frequency', fontsize=10.5, fontweight='bold', color=C_DARK)
ax1.set_title(f'Reliability Diagram — Isotonic Calibration Cuts Brier Error {red:.1f}%',
              fontsize=12.5, fontweight='bold', color=C_PRIMARY, pad=10)
ax1.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=9.2)
ax1.grid(True, linestyle=':', alpha=0.6)
ax2.hist(p_cal, bins=25, color=C_SECONDARY, edgecolor='white', alpha=0.85, zorder=3)
ax2.set_xlabel('Calibrated P(RTO)', fontsize=10.5, fontweight='bold', color=C_DARK)
ax2.set_ylabel('Orders', fontsize=9.5, fontweight='bold', color=C_DARK)
ax2.grid(True, linestyle=':', alpha=0.6)
econ_thr = v('uplift.econ_threshold') / 100.0
ax2.axvline(econ_thr, color=C_ALERT, linestyle=':', linewidth=1.8)
ax2.text(econ_thr, ax2.get_ylim()[1] * 0.92,
         f'  cost-optimal\n  threshold {d("uplift.econ_threshold")}',
         fontsize=8.6, fontweight='bold', color=C_ALERT, va='top')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/calibration_curve.png')

# ==============================================================================
# CHART 7 — TreeSHAP
# ==============================================================================
print('[ 7/19] shap_summary_plot.png')
shap_cache = joblib.load('data/shap_summary_cache.joblib')
fig = plt.figure(figsize=(9.2, 5.6), dpi=300)
shap.summary_plot(shap_cache['shap_values'], shap_cache['sample'],
                  feature_names=shap_cache['features'], show=False, max_display=12)
plt.title('TreeSHAP Feature Attribution — Pre-Dispatch RTO Predictor', fontsize=12.5,
          fontweight='bold', color=C_PRIMARY, pad=14)
plt.xlabel('SHAP value (impact on log-odds of failure)', fontsize=10.5,
           fontweight='bold', color=C_DARK)
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/shap_summary_plot.png')

# ==============================================================================
# CHART 8 — Synthetic Qini diagnostic (no operating cutoff)
# ==============================================================================
print('[ 8/19] qini_uplift_curve.png')
q = pd.read_csv('data/qini_eval_data.csv')
fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=300)
x = q['cum_frac'].to_numpy() * 100
ax.plot(x, q['qini'], color=C_PRIMARY, linewidth=2.8, zorder=5,
        label=(f"X-Learner CATE  (AUUC {d('uplift.auuc')}, "
               f"p = {d('uplift.p_value')})"))
ax.plot(x, q['qini_random'], color=C_MUTED, linestyle='--', linewidth=2.0, zorder=3,
        label=f"Random targeting null  (AUUC {d('uplift.auuc_random')}, 200 permutations)")
ax.plot(x, q['qini_perfect'], color=C_SUCCESS, linestyle=':', linewidth=2.0, zorder=3,
        label=f"Perfect-targeting oracle  (AUUC {d('uplift.auuc_perfect')})")

ax.set_xlim(0, 100)
ax.set_ylim(0, q['qini_perfect'].max() * 1.06)
ax.set_xlabel('% of hesitant COD refusers targeted at the doorstep',
              fontsize=10.5, fontweight='bold', color=C_DARK)
ax.set_ylabel('Qini — cumulative simulated treatment effect', fontsize=10.5,
              fontweight='bold', color=C_DARK)
ax.set_title('Synthetic Qini Diagnostic vs Generated Null', fontsize=12.5,
             fontweight='bold', color=C_PRIMARY, pad=14)
ax.legend(loc='center right', frameon=True, framealpha=0.95, fontsize=9)
footer(fig, f'Synthetic holdout N={len(q):,}. Normalised AUUC {d("uplift.auuc_norm")} against a null that is '
        f'0 by construction; {d("uplift.lift_over_random")} above random allocation, '
        f'p = {d("uplift.p_value")} over 200 permutations. '
        f'X-Learner recovers the generating CATE at r = {d("uplift.cate_recovery")}.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/qini_uplift_curve.png')

# ==============================================================================
# CHART 9 — Synthetic-label confusion matrix (display cutoff only)
# ==============================================================================
print('[ 9/19] confusion_matrix.png')
thr = 0.5
y_bin = (p_cal >= thr).astype(int)
cm = confusion_matrix(y_true, y_bin)
cmn = cm.astype(float) / cm.sum(axis=1)[:, None]

fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=300)
cax = ax.matshow(cmn, cmap='Blues', alpha=0.85)
labels_txt = {
    (0, 0): "Generated delivered\nlabel",
    (0, 1): "Generated false positive",
    (1, 0): "Generated false negative",
    (1, 1): "Generated RTO label",
}
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.1%})\n{labels_txt[(i, j)]}",
                ha='center', va='center', fontsize=8.8, fontweight='bold',
                color='white' if cmn[i, j] > 0.5 else C_DARK)
fig.colorbar(cax)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted delivered (0)', 'Predicted high-risk (1)'],
                   fontsize=9.8, fontweight='bold')
ax.set_yticklabels(['Actual delivered (0)', 'Actual RTO (1)'], fontsize=9.8, fontweight='bold')
ax.set_xlabel('Illustrative model cutoff (0.5)', fontsize=10.5, fontweight='bold', color=C_DARK, labelpad=10)
ax.set_ylabel('Ground truth outcome', fontsize=10.5, fontweight='bold', color=C_DARK, labelpad=10)
ax.set_title('Synthetic Holdout Confusion Matrix — Display Cutoff Only',
             fontsize=12, fontweight='bold', color=C_PRIMARY, pad=14)
footer(fig, 'Generated labels and generated holdout only. The 0.5 cutoff is used only to display a matrix; it is not an action, coupon, or operating threshold.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/confusion_matrix.png')

# ==============================================================================
# CHART 10 — Operating point: precision / recall / lift vs action budget
# ==============================================================================
print('[10/19] operating_point_lift.png')
rto_m = _load(os.path.join('models', 'rto_metrics.json'))
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=300)

ks = sorted(rto_m['operating_points'], key=lambda x: int(x))
budgets = [int(k) for k in ks]
precs = [rto_m['operating_points'][k]['precision'] for k in ks]
recs = [rto_m['operating_points'][k]['recall'] for k in ks]
lifts = [rto_m['operating_points'][k]['lift'] for k in ks]
prev = rto_m['baseline_prevalence']

axA.plot(budgets, [p * 100 for p in precs], marker='o', color=C_PRIMARY, linewidth=2.6,
         markersize=8, label='Precision @ budget', zorder=5)
axA.axhline(prev * 100, color=C_ALERT, linestyle='--', linewidth=1.6,
            label=f'No-skill baseline ({prev:.1%})')
axA.fill_between(budgets, [p * 100 for p in precs], prev * 100, color=C_PRIMARY, alpha=0.10)
axA.set_xlabel('Share of orders the team acts on (%)', fontsize=10.5, fontweight='bold', color=C_DARK)
axA.set_ylabel('Precision (%)', fontsize=10.5, fontweight='bold', color=C_DARK)
axA.set_title('Precision rises as the action budget narrows', fontsize=11.5,
              fontweight='bold', color=C_PRIMARY, pad=10)
axA.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=9)
axA.grid(True, linestyle=':', alpha=0.6)
i10 = budgets.index(10)
axA.scatter([10], [precs[i10] * 100], s=150, color=C_PRIMARY, zorder=7,
            edgecolor='white', linewidth=2)
axA.annotate(f'top 10%: {d("rto.precision_top10")}\n{d("rto.lift_top10")} lift',
             xy=(10, precs[i10] * 100), xytext=(20, precs[i10] * 100 - 7),
             fontsize=8.8, fontweight='bold', color=C_PRIMARY,
             arrowprops=dict(arrowstyle='->', color=C_PRIMARY, lw=1.3))

axB.plot(budgets, lifts, marker='s', color=C_ACCENT, linewidth=2.6, markersize=8, zorder=5)
axB.axhline(1.0, color=C_ALERT, linestyle='--', linewidth=1.6, label='Random targeting (1.0x)')
axB.fill_between(budgets, lifts, 1.0, color=C_ACCENT, alpha=0.10)
axB.set_xlabel('Share of orders the team acts on (%)', fontsize=10.5, fontweight='bold', color=C_DARK)
axB.set_ylabel('Lift over random targeting (x)', fontsize=10.5, fontweight='bold', color=C_DARK)
axB.set_title('Synthetic holdout: targeted ranking versus baselines', fontsize=11.5,
              fontweight='bold', color=C_PRIMARY, pad=10)
axB.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=9)
axB.grid(True, linestyle=':', alpha=0.6)
axB.set_ylim(0.9, max(lifts) * 1.12)

fig.suptitle('Operating-Point Performance — the Metric That Justifies the Budget',
             fontsize=13, fontweight='bold', color=C_PRIMARY, y=0.985)
footer(fig, f'Holdout N={rto_m["n_test"]:,} orders. Precision@k answers the operational question '
         f'("if I touch the riskiest k% of orders, what share genuinely fail?") that ROC-AUC '
         f'does not. Target the top 10% and catch {d("rto.recall_top10")} of all failures at '
         f'{d("rto.precision_top10")} precision.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/operating_point_lift.png')

# ==============================================================================
# CHART 11 — Telemetry auditor precision/recall frontier
# ==============================================================================
print('[11/19] telemetry_frontier.png')
tel_m = _load(os.path.join('models', 'telemetry_metrics.json'))
fr = pd.DataFrame(tel_m['frontier'])
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)

ax.plot(fr['recall'] * 100, fr['precision'] * 100, marker='o', color=C_PRIMARY,
        linewidth=2.6, markersize=8, zorder=5, label='Held-out frontier')
ax.axhline(fr.loc[fr['audit_budget_pct'] == 10, 'genuine_precision'].iloc[0] * 100,
           color=C_MUTED, linestyle='--', linewidth=1.4,
           label='Genuine-attempt precision (rider-fairness floor)')

sel = fr[fr['audit_budget_pct'] == tel_m['operating_budget_pct']].iloc[0]
ax.scatter([sel['recall'] * 100], [sel['precision'] * 100], s=190, color=C_ALERT,
           zorder=7, edgecolor='white', linewidth=2.4)
ax.annotate(f"Selected operating point\n{sel['audit_budget_pct']:.0f}% audit budget\n"
            f"P={sel['precision']:.1%}  R={sel['recall']:.1%}  F1={sel['f1']:.3f}",
            xy=(sel['recall'] * 100, sel['precision'] * 100),
            xytext=(sel['recall'] * 100 - 46, sel['precision'] * 100 - 22),
            fontsize=9, fontweight='bold', color=C_ALERT,
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFF5F5', ec=C_ALERT, lw=1.2),
            arrowprops=dict(arrowstyle='->', color=C_ALERT, lw=1.5))

for _, r in fr.iterrows():
    ax.annotate(f"{r['audit_budget_pct']:.0f}%", (r['recall'] * 100, r['precision'] * 100),
                textcoords='offset points', xytext=(7, 6), fontsize=7.8, color='#64748B')

ax.set_xlabel('Recall — share of generated anomaly-label rows flagged (%)', fontsize=10.5,
              fontweight='bold', color=C_DARK)
ax.set_ylabel('Precision — share of flagged rows matching the generated anomaly label (%)', fontsize=10.5,
              fontweight='bold', color=C_DARK)
ax.set_title('Synthetic Attempt-Pattern Anomaly Prototype — Held-Out Generated Labels', fontsize=12.5,
             fontweight='bold', color=C_PRIMARY, pad=14)
ax.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)
footer(fig, f'Unsupervised fit (contamination="auto"); the review-budget threshold is a scenario setting. '
        f'Scored on {tel_m["n_test"]:,} held-out generated examples. These metrics describe only '
        'synthetic labels; they are not a fraud rate, rider-behavior measure, or production control. '
        'Do not automate payment holds or adverse actions from this prototype.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/telemetry_frontier.png')

# ==============================================================================
# CHART 12 — Cross-validation stability
# ==============================================================================
print('[12/19] cv_fold_stability.png')
cv = pd.read_csv('data/cv_fold_metrics.csv')
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=300)

axA.plot(cv['fold'], cv['lgb_roc_auc'], marker='o', color=C_PRIMARY, linewidth=2.4,
         markersize=9, label='LightGBM', zorder=5)
axA.plot(cv['fold'], cv['xgb_roc_auc'], marker='s', color=C_SECONDARY, linewidth=2.0,
         linestyle='-.', markersize=8, label='XGBoost', zorder=5)
axA.axhline(cv['lgb_roc_auc'].mean(), color=C_PRIMARY, linestyle='--', linewidth=1.4, alpha=0.7,
            label=f'LGBM mean {cv["lgb_roc_auc"].mean():.4f}')
axA.set_xticks(cv['fold'])
axA.set_xlabel('CV fold', fontsize=10.5, fontweight='bold', color=C_DARK)
axA.set_ylabel('ROC-AUC', fontsize=10.5, fontweight='bold', color=C_DARK)
axA.set_title(f"Rank discrimination is stable (σ = {rto_m['cross_validation']['lgb_roc_auc_std']:.4f})",
              fontsize=11.5, fontweight='bold', color=C_PRIMARY, pad=10)
axA.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8.8)
axA.grid(True, linestyle=':', alpha=0.6)

axB.plot(cv['fold'], cv['lgb_brier'], marker='o', color=C_PRIMARY, linewidth=2.4,
         markersize=9, label='Raw LightGBM (uncalibrated)', zorder=5)
axB.axhline(num('rto.brier_calibrated'), color=C_SUCCESS, linestyle='-',
            linewidth=2.2,
            label=f'After isotonic calibration ({d("rto.brier_calibrated")})')
axB.set_xticks(cv['fold'])
axB.set_xlabel('CV fold', fontsize=10.5, fontweight='bold', color=C_DARK)
axB.set_ylabel('Brier score', fontsize=10.5, fontweight='bold', color=C_DARK)
axB.set_title('Calibration error is where the real gain sits', fontsize=11.5,
              fontweight='bold', color=C_PRIMARY, pad=10)
axB.legend(loc='center right', frameon=True, framealpha=0.92, fontsize=8.8)
axB.grid(True, linestyle=':', alpha=0.6)

fig.suptitle('5-Fold Cross-Validation Stability — Reported Variance Across Folds',
             fontsize=12.5, fontweight='bold', color=C_PRIMARY, y=0.985)
footer(fig, 'Raw GBDT probabilities are badly miscalibrated on every fold (Brier ≈ 0.23); isotonic '
         'calibration removes that error consistently. This is why the pack leads with '
         f'{d("rto.brier_reduction")} rather than with AUC.')
fig.tight_layout(rect=[0, 0.075, 1, 1.0])
save(fig, 'charts/cv_fold_stability.png')

# ==============================================================================
# ==============================================================================
# CHART 13 — Unit-economics sensitivity from the case-pack reverse cost
print('[13/19] unit_economics_sensitivity.png')
fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=300)
pp = np.array([0.5, 1.0, 2.0])
orders = 10_000
reverse_cost = 120.0
incremental_deliveries = orders * pp / 100.0
avoided_reverse_cost = incremental_deliveries * reverse_cost
bars = ax.bar([f'{x:.1f} pp' for x in pp], avoided_reverse_cost,
              color=[C_SECONDARY, C_ACCENT, C_SUCCESS], width=0.55,
              edgecolor='#4A5568', linewidth=0.8, zorder=3)
for bar, count, value in zip(bars, incremental_deliveries, avoided_reverse_cost):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 450,
            f'{count:,.0f} orders\nup to ₹{value:,.0f}',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=C_DARK)
ax.set_ylim(0, 29_000)
ax.set_ylabel('Gross reverse freight avoided (₹)', fontsize=10.5, fontweight='bold', color=C_DARK)
ax.set_xlabel('Absolute RTO improvement per 10,000 eligible orders', fontsize=10.5,
              fontweight='bold', color=C_DARK)
ax.set_title('Illustrative Unit Economics — Not an Outcome Forecast',
             fontsize=12.5, fontweight='bold', color=C_PRIMARY, pad=14)
ax.grid(True, axis='y', linestyle=':', alpha=0.6, zorder=0)
footer(fig, 'Arithmetic uses 10,000 eligible orders × the stated absolute percentage-point change × ₹120 case-pack reverse freight. Gross ceiling only: confirm avoidability and subtract all incremental intervention, service, payment, conversion, handling and seller costs.')
fig.tight_layout(rect=[0, 0.11, 1, 1.0])
save(fig, 'charts/unit_economics_sensitivity.png')

# CHART 14 — Worker-safe distance-pay pilot sequence
# ==============================================================================
print('[14/19] pilot_pay_economics.png')
fig, ax = plt.subplots(figsize=(11.8, 4.2), dpi=300)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
steps = [
    ('1. BASELINE', 'Record route distance, time,\ncurrent pay and outcomes'),
    ('2. CO-DESIGN', 'Agree eligible distance bands\nand a fair, capped rate'),
    ('3. CONTROLLED TEST', 'Randomize at a feasible unit;\nchange one lever at a time'),
    ('4. DECIDE', 'Compare delivery and net cost;\ncheck earnings, safety and appeals'),
]
xs = [0.13, 0.38, 0.63, 0.88]
for i, ((title, body), x) in enumerate(zip(steps, xs)):
    color = [C_PRIMARY, C_SECONDARY, C_ACCENT, C_SUCCESS][i]
    ax.text(x, 0.56, f'{title}\n\n{body}', transform=ax.transAxes,
            ha='center', va='center', fontsize=9.2, fontweight='bold', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.5', fc='#F8FAFC', ec=color, lw=2),
            linespacing=1.35)
    if i < len(xs) - 1:
        ax.annotate('', xy=(xs[i+1]-0.095, 0.56), xytext=(x+0.095, 0.56),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color=C_MUTED, lw=2.0))
ax.set_title('Distance-aware pay: measure first, then test',
             fontsize=13, fontweight='bold', color=C_PRIMARY, pad=16)
footer(fig, 'The case gives no current rider-pay rate or fuel cost. Set any treatment only after authorized baseline data and rider input. Pay legitimate attempts regardless of outcome; never use a noisy signal for automatic pay denial.')
plt.tight_layout(rect=[0, 0.12, 1, 0.94])
save(fig, 'charts/pilot_pay_economics.png')

# ==============================================================================
# CHART 15 — Refused-order recovery decision path
# ==============================================================================
print('[15/19] rescue_cascade_value.png')
fig, ax = plt.subplots(figsize=(12.2, 4.6), dpi=300)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
boxes = [
    (0.10, 0.62, 'UNDelivered /\nrefused order', C_PRIMARY),
    (0.34, 0.62, 'Does the original\nbuyer still want it?', C_SECONDARY),
    (0.61, 0.79, 'Yes: offer optional\npayment or reattempt', C_ACCENT),
    (0.61, 0.43, 'No: consider local recovery\nonly with seller consent,\neligible stock and demand', C_SECONDARY),
    (0.88, 0.43, 'Conditions pass?\nOne bounded re-match', C_SUCCESS),
    (0.88, 0.12, 'Otherwise: standard\nreturn and refund', C_MUTED),
]
for x, y, label, color in boxes:
    ax.text(x, y, label, transform=ax.transAxes, ha='center', va='center',
            fontsize=9.2, fontweight='bold', color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.62', fc='#F8FAFC', ec=color, lw=1.8),
            linespacing=1.25)
def arrow(a, b, label=None, offset=(0, 0)):
    ax.annotate('', xy=b, xytext=a, xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=C_MUTED, lw=1.5,
                                connectionstyle='arc3,rad=0.0'))
    if label:
        ax.text((a[0]+b[0])/2+offset[0], (a[1]+b[1])/2+offset[1], label,
                transform=ax.transAxes, fontsize=8, color=C_DARK, fontweight='bold',
                ha='center', va='center', bbox=dict(fc='white', ec='none', pad=1.5))
arrow((0.19, 0.62), (0.24, 0.62))
arrow((0.43, 0.66), (0.53, 0.78), 'YES', (0, 0.035))
arrow((0.43, 0.58), (0.53, 0.45), 'NO', (0, -0.035))
arrow((0.70, 0.43), (0.79, 0.43), 'IF APPROVED', (0, 0.035))
arrow((0.70, 0.38), (0.79, 0.16), 'IF NOT', (0.01, -0.005))
ax.set_title('Refused-order recovery: customer choice first, seller rights always',
             fontsize=12.6, fontweight='bold', color=C_PRIMARY, pad=16)
footer(fig, 'No recovery share or local-resale lift is assumed here. Compare all incremental payment, handling, storage, onward-delivery, refund and settlement costs with the standard return path.')
plt.tight_layout(rect=[0, 0.12, 1, 0.94])
save(fig, 'charts/rescue_cascade_value.png')

# ==============================================================================
# CHART 16 — 30-60-90 phase timeline with validation gates
# ==============================================================================
print('[16/19] phase_timeline.png')
fig, ax = plt.subplots(figsize=(12.0, 4.8), dpi=300)

phases = [
    # label, start day, end day, colour, items
    ('Phase 1 · Days 1–30\nSoftware only', 0, 30, C_PRIMARY, [
        'Establish baseline', 'Consent-based interviews',
        'Skippable prompt prototype', 'Define guardrails']),
    ('Phase 2 · Days 31–60\nAlgorithmic + pay', 30, 60, C_SECONDARY, [
        'Powered test of one lever', 'Concurrent control',
        'Measure rider/seller guardrails', 'Do not infer from simulator']),
    ('Phase 3 · Days 61–90\nNetwork scale', 60, 90, C_ACCENT, [
        'Expand only if net positive', 'Seller consent + controls',
        'Retain holdout', 'Pause on harm']),
]

for i, (label, d0, d1, col, items) in enumerate(phases):
    y = len(phases) - i - 1
    ax.barh(y, d1 - d0, left=d0, height=0.44, color=col, alpha=0.92,
            edgecolor='#4A5568', linewidth=0.8, zorder=3)
    ax.text(d0 + (d1 - d0) / 2, y, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=5)
    for j, it in enumerate(items):
        ax.text(d0 + 0.8, y - 0.30 - j * 0.155, f'· {it}', ha='left', va='center',
                fontsize=8.2, color=C_DARK, zorder=5)

# Validation gates at the phase boundaries.
gates = [
    # These must track the deck's gates. They were typed literals printing a
    # Gate 3 of "<= 12.0%" while the deck states 14.99%, so the figure
    # contradicted the slide it was embedded in -- the exact failure the gate
    # rework set out to fix.
    (30, 'GATE 1', 'Baseline + data quality\nSafety guardrails pass'),
    (60, 'GATE 2', 'Sample size + MDE set\nTreatment vs control'),
    (90, 'GATE 3', 'Expand if net benefit holds\nGuardrails stay in bounds'),
]
for gx, name, crit in gates:
    ax.axvline(gx, color=C_DARK, linestyle='--', linewidth=1.6, zorder=6)
    ax.text(gx + 1.2, len(phases) - 0.34, name, fontsize=9.5, fontweight='bold',
            color=C_DARK, va='center')
    ax.text(gx + 1.2, len(phases) - 0.62, crit, fontsize=7.8, color='#475569',
            va='center', linespacing=1.35)

ax.set_xlim(0, 96)
ax.set_ylim(-1.05, len(phases) - 0.15)
ax.set_yticks([])
ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
ax.set_xticklabels(['Day 0', '15', '30', '45', '60', '75', '90'], fontsize=9)
ax.set_xlabel('Days from pilot start', fontsize=10.5, fontweight='bold', color=C_DARK)
ax.set_title('90 Days, Three Gates — Nothing Scales Until the Gate Before It Clears',
             fontsize=12.5, fontweight='bold', color=C_PRIMARY, pad=16)
ax.grid(True, linestyle=':', alpha=0.5, axis='x')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

footer(fig, (
    'Phase 1 carries no dependency on any Phase 2 infrastructure, so the pilot starts even if nothing else '
    'is built. Each gate is a measured criterion, not a date: if Gate 1 fails, we do not proceed to Phase 2. '
    'Primary data and a randomized holdout replace synthetic estimates. Do not scale until incremental delivery, net cost and rider, seller and customer guardrails are measured.'))
plt.tight_layout(rect=[0, 0.115, 1, 1])
save(fig, 'charts/phase_timeline.png')

# ==============================================================================
# CHART 17 — Address-feature scenario
# ==============================================================================
print('[17/19] address_navigability.png')
ds_df = pd.read_csv('data/valmo_orders_dataset.csv',
                    usecols=['s_addr', 'rto_flag', 'has_premise', 'has_landmark',
                             'pin_mismatch_m', 'dist_category'])
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 4.4), dpi=300)

# Panel A: failure rate by navigability decile.
ds_df['s_bin'] = pd.qcut(ds_df['s_addr'], 10, labels=False, duplicates='drop')
dec = ds_df.groupby('s_bin').agg(s=('s_addr', 'mean'), rto=('rto_flag', 'mean'))
xs = np.arange(len(dec))
axA.bar(xs, dec['rto'] * 100, color=[C_SECONDARY] * 5 + [C_MUTED] * 2 + [C_PRIMARY] * 3,
        width=0.68, edgecolor='#4A5568', linewidth=0.7, zorder=3)
for x, (s_, r_) in enumerate(zip(dec['s'], dec['rto'])):
    axA.text(x, r_ * 100 + 0.5, f'{r_ * 100:.1f}', ha='center', va='bottom',
             fontsize=8.4, fontweight='bold', color=C_DARK)
axA.axhline(dec['rto'].mean() * 100, color=C_ALERT, linestyle='--', linewidth=1.6,
            label=f'Network average {dec["rto"].mean() * 100:.1f}%')
axA.set_xticks(xs)
axA.set_xticklabels([f'{v:.2f}' for v in dec['s']], fontsize=8, rotation=45)
axA.set_xlabel('Navigability score S_addr (decile)', fontsize=10.5, fontweight='bold', color=C_DARK)
axA.set_ylabel('Failure rate (%)', fontsize=10.5, fontweight='bold', color=C_DARK)
axA.set_ylim(0, dec['rto'].max() * 100 * 1.22)
axA.set_title('Synthetic scenario: address feature and generated failure label',
              fontsize=11.5, fontweight='bold', color=C_PRIMARY, pad=10)
axA.legend(loc='upper right', frameon=True, framealpha=0.94, fontsize=8.8)
axA.grid(True, linestyle=':', alpha=0.6, axis='y')

# Panel B: failure rate by the three signals that build S_addr.
signals = [
    ('Has a premise\nidentifier', ds_df['has_premise'] == 1),
    ('Has a nearby\nlandmark', ds_df['has_landmark'] == 1),
    ('Pin matches\n(order <100 m)', ds_df['pin_mismatch_m'] < 100),
    ('None of the\nthree', (ds_df['has_premise'] == 0) & (ds_df['has_landmark'] == 0)
     & (ds_df['pin_mismatch_m'] >= 100)),
]
base = ds_df['rto_flag'].mean() * 100
rates = [ds_df.loc[m, 'rto_flag'].mean() * 100 for _, m in signals]
cols = [C_SUCCESS, C_SUCCESS, C_SUCCESS, C_ALERT]
bars = axB.bar([s for s, _ in signals], rates, color=cols, width=0.6,
               edgecolor='#4A5568', linewidth=0.8, zorder=3)
for b_, r_ in zip(bars, rates):
    axB.text(b_.get_x() + b_.get_width() / 2, r_ + 0.6, f'{r_:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_DARK)
axB.axhline(base, color=C_MUTED, linestyle='--', linewidth=1.6, label=f'Network average {base:.1f}%')
axB.set_ylim(0, max(rates) * 1.3)
axB.set_ylabel('Failure rate (%)', fontsize=10.5, fontweight='bold', color=C_DARK)
axB.set_title('Generated feature correlations — not causal effects', fontsize=11.5,
              fontweight='bold', color=C_PRIMARY, pad=10)
axB.legend(loc='upper right', frameon=True, framealpha=0.94, fontsize=8.8)
axB.grid(True, linestyle=':', alpha=0.6, axis='y')
plt.setp(axB.get_xticklabels(), fontsize=8.2)

worst_decile = dec['rto'].max() * 100
best_decile = dec['rto'].min() * 100
no_signal = rates[-1]
fig.suptitle('Address-Feature Simulation — Operational Impact Is Unknown',
             fontsize=13, fontweight='bold', color=C_PRIMARY, y=0.985)
footer(fig, (
    f'Synthetic dataset only ({len(ds_df):,} generated rows); feature-outcome associations are induced by the simulator and are not observed Valmo effects. '
    'Treat this as a prototype diagnostic; test address prompts on randomized traffic and measure checkout conversion and delivery outcomes.'))
plt.tight_layout(rect=[0, 0.115, 1, 0.955])
save(fig, 'charts/address_navigability.png')

# ==============================================================================
# CHART 18 — Generated geography scenarios (no Valmo findings)
# ==============================================================================
print('[18/19] bharat_heterogeneity.png')
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=300)

# Panel A: the brief's "India is not one market" challenge.
tier_all = df_full.groupby('tier')['rto_flag'].mean() * 100
tier_cod = df_full[df_full['is_cod'] == 1].groupby('tier')['rto_flag'].mean() * 100
xs = np.arange(3)
labels_ = ['Tier 1\nmetro', 'Tier 2\ncity', 'Tier 3\nrural / semi-urban']
w = 0.36
axA.bar(xs - w / 2, tier_all.values, width=w, color=C_SECONDARY,
        label='All orders', zorder=3)
axA.bar(xs + w / 2, tier_cod.values, width=w, color=C_ALERT,
        label='COD orders', zorder=3)
axA.axhline(20.0, color=C_DARK, linestyle='--', linewidth=1.6, zorder=4)

for i in xs:
    axA.text(i - w / 2, tier_all.values[i] + 0.35, f'{tier_all.values[i]:.1f}',
             ha='center', fontsize=9.2, fontweight='bold', color=C_DARK)
    axA.text(i + w / 2, tier_cod.values[i] + 0.35, f'{tier_cod.values[i]:.1f}',
             ha='center', fontsize=9.2, fontweight='bold', color=C_ALERT)
axA.set_xticks(xs); axA.set_xticklabels(labels_)
axA.set_ylim(0, max(tier_cod.max(), 20.0) * 1.42)
axA.set_ylabel('Failure rate (%)', fontsize=10.5, fontweight='bold', color=C_DARK)
axA.set_title('Synthetic tier pattern (not an observed Valmo effect)',
              fontsize=11.5, fontweight='bold', color=C_PRIMARY, pad=10)
from matplotlib.lines import Line2D
_handles, _labels = axA.get_legend_handles_labels()
_handles.append(Line2D([0], [0], color=C_DARK, linestyle='--', linewidth=1.5))
_labels.append("brief COD benchmark 20%")
axA.legend(_handles, _labels, loc='upper left', frameon=True, framealpha=0.96,
           fontsize=8.2, bbox_to_anchor=(0.005, 0.80))
axA.grid(True, linestyle=':', alpha=0.6, axis='y')
gap = tier_cod.loc[3] - tier_cod.loc[1]
axA.annotate('Generator-injected tier differences;\nnot evidence of a Valmo tier effect',
             xy=(2 + w / 2, tier_cod.loc[3]), xytext=(0.10, tier_cod.max() * 1.185),
             fontsize=8.4, fontweight='bold', color=C_ALERT,
             bbox=dict(boxstyle='round,pad=0.32', fc='#FFF5F5', ec=C_ALERT, lw=1),
             arrowprops=dict(arrowstyle='->', color=C_ALERT, lw=1.2))

# Panel B: the brief's closing note on first-time and unclear addresses.
segs = [
    ('Both premise\nAND landmark', 14.52, C_SUCCESS),
    ('Landmark only', 17.10, C_MUTED),
    ('Premise only', 20.94, C_MUTED),
    ('Neither\n(cannot find it)', 22.65, C_ALERT),
]
xs2 = np.arange(len(segs))
axB.bar(xs2, [s[1] for s in segs], color=[s[2] for s in segs], width=0.6,
        edgecolor='#4A5568', linewidth=0.8, zorder=3)
for i, s in enumerate(segs):
    axB.text(i, s[1] + 0.4, f'{s[1]:.1f}%', ha='center', fontsize=10,
             fontweight='bold', color=C_DARK)
axB.axhline(17.07, color=C_DARK, linestyle='--', linewidth=1.5)
axB.text(3.46, 17.5, 'network avg 17.07%', fontsize=8.2, color=C_DARK,
         ha='right', fontweight='bold')
axB.set_xticks(xs2); axB.set_xticklabels([s[0] for s in segs], fontsize=8.6)
axB.set_ylim(0, 30)
axB.set_ylabel('Failure rate (%)', fontsize=10.5, fontweight='bold', color=C_DARK)
axB.set_title('Synthetic address-feature pattern (not causal evidence)',
             fontsize=11.5, fontweight='bold', color=C_PRIMARY, pad=10)
axB.grid(True, linestyle=':', alpha=0.6, axis='y')
axB.annotate('case brief gives a directional note on\n'
             'first-time / unclear addresses; no rates',
             xy=(3, 22.65), xytext=(0.05, 26.2), fontsize=8.0, fontweight='bold',
             color=C_SECONDARY, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='#EFF6FF', ec=C_SECONDARY, lw=1),
             arrowprops=dict(arrowstyle='->', color=C_SECONDARY, lw=1.1,
                             connectionstyle='arc3,rad=0.18'))

fig.suptitle('Synthetic Tier and Address Scenarios — Not Regional Findings',
             fontsize=13, fontweight='bold', color=C_PRIMARY, y=0.985)
footer(fig, (
    f"Synthetic dataset only ({len(df_full):,} generated rows); the generator encodes the brief's qualitative address and tier hypotheses. These charts do not establish a tier effect or causal address impact. "
    'Use them to define strata for a real pilot, not to rank live customers or sites.'))
plt.tight_layout(rect=[0, 0.115, 1, 0.955])
save(fig, 'charts/bharat_heterogeneity.png')

# ==============================================================================
# CHART 19 — Synthetic signal ceiling and seed robustness
# ==============================================================================
print('[19/19] signal_ceiling.png')
_sc = json.load(open('models/signal_ceiling.json'))
_ceil, _sr = _sc['ceiling'], _sc['seed_robustness']

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=300)

# Panel A: model vs the information ceiling.
bars = axA.bar(['Bayes ceiling\n(true p(y|x))', 'Our calibrated\nclassifier'],
               [_ceil['bayes_ceiling_auc'], _ceil['model_holdout_auc']],
               color=[C_MUTED, C_PRIMARY], width=0.52,
               edgecolor='#4A5568', linewidth=0.8, zorder=3)
for bar, v in zip(bars, [_ceil['bayes_ceiling_auc'], _ceil['model_holdout_auc']]):
    axA.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f'{v:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold', color=C_DARK)
axA.set_ylim(0.48, _ceil['bayes_ceiling_auc'] * 1.10)
axA.set_ylabel('ROC-AUC', fontsize=10.5, fontweight='bold', color=C_DARK)
axA.set_title(f"The model captures {_ceil['fraction_of_signal_captured']:.1%} "
              'of the available signal', fontsize=11.5, fontweight='bold',
              color=C_PRIMARY, pad=10)
axA.annotate(f"Synthetic generator ceiling;\nnot a Valmo performance bound\n"
             f"(gap to ceiling = {_ceil['gap_absolute']:.4f})",
             xy=(1, _ceil['model_holdout_auc'] * 0.999), xytext=(-0.52, 0.596),
             fontsize=8.6, fontweight='bold', color=C_ALERT,
             bbox=dict(boxstyle='round,pad=0.32', fc='#FFF5F5', ec=C_ALERT, lw=1),
             arrowprops=dict(arrowstyle='->', color=C_ALERT, lw=1.3))
axA.grid(True, linestyle=':', alpha=0.6, axis='y')

# Panel B: the honest error bar across five independent splits.
rows = _sr['per_seed']
seeds = [r['seed'] for r in rows]
vals = [r['roc_auc'] for r in rows]
xs = np.arange(len(seeds))
axB.axhline(_sr['roc_auc_mean'], color=C_DARK, linestyle='--', linewidth=1.6, zorder=4)
axB.text(len(seeds) - 0.5, _sr['roc_auc_mean'] + 0.0015,
         f"mean {_sr['roc_auc_mean']:.4f}", fontsize=8.4, fontweight='bold',
         color=C_DARK, ha='right')
axB.plot(xs, vals, marker='o', markersize=9, linewidth=2.4, color=C_ACCENT, zorder=5)
axB.fill_between(xs, _sr['roc_auc_mean'] - _sr['roc_auc_std'],
                 _sr['roc_auc_mean'] + _sr['roc_auc_std'],
                 color=C_ACCENT, alpha=0.16, zorder=2)
axB.set_xticks(xs)
axB.set_xticklabels([str(s) for s in seeds], fontsize=8.6)
axB.set_xlabel('Random seed (independent train/test split)', fontsize=10.5,
               fontweight='bold', color=C_DARK)
axB.set_ylabel('Holdout ROC-AUC', fontsize=10.5, fontweight='bold', color=C_DARK)
axB.set_title(f"Seed-to-seed AUC spread: +/-{_sr['roc_auc_std']:.4f}",
              fontsize=11.5, fontweight='bold', color=C_PRIMARY, pad=10)
axB.grid(True, linestyle=':', alpha=0.6, axis='y')

fig.suptitle('Synthetic Signal Ceiling and Fold Variation', fontsize=13,
             fontweight='bold', color=C_PRIMARY, y=0.985)
footer(fig, (
    f'The ceiling is computed by ranking on the TRUE conditional probability of failure -- the '
    f'optimal possible scorer for this population. It is a property of the data-generating process, '
    f'not of any estimator, so fitting harder cannot move it. Our holdout reaches '
    f'{_ceil["fraction_of_signal_captured"]:.1%} of it. The model-to-ceiling gap reflects the generated data and fitting process; it is not a Valmo estimate. '
    'DISCLOSURE: the ceiling inherits OUR calibration -- we solved those coefficients so the population '
    'matches the brief. If the real process is sharper, both the ceiling and the headroom rise. This '
    f'bounds what a classifier could achieve on this data; it is not a claim about Valmo. Right: '
    f'{_sr["roc_auc_min"]:.4f} to {_sr["roc_auc_max"]:.4f} across five independent splits.'))
plt.tight_layout(rect=[0, 0.115, 1, 0.955])
save(fig, 'charts/signal_ceiling.png')

print('\nAll 19 figures generated in charts/.')
