"""
Research-calibration figures for the Valmo proposal analysis.

Renders analytical figures for pilot methodology, NDR recovery window,
intervention prioritization, gross ceiling scenarios, and distance pattern analysis.

Run:  python generate_research_figures.py
Out:  charts/evidence_graded_causes.png
      charts/ndr_window.png
      charts/prioritisation_matrix.png
      charts/gross_ceiling_scenarios.png
      charts/case_pack_distance.png
"""

from __future__ import annotations

import io
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.makedirs('charts', exist_ok=True)

C_PRIMARY = '#9B1B58'
C_SECONDARY = '#1E3A8A'
C_ACCENT = '#0D9488'
C_ALERT = '#DC2626'
C_SUCCESS = '#16A34A'
C_WARN = '#B45309'
C_MUTED = '#94A3B8'
C_DARK = '#1E293B'
C_LIGHT = '#F8FAFC'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#CBD5E1',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def _load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)




def save(fig, name):
    path = os.path.join('charts', name)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.16,
                facecolor='white')
    plt.close(fig)
    print(f'  wrote charts/{name}')


# ---------------------------------------------------------------------------
# FIGURE 1 - evidence-graded root causes
# ---------------------------------------------------------------------------
def fig_evidence_graded_causes():
    """Explain why the submission does not show a Valmo cause-share waterfall.

    Public datasets use different categories and populations. The vendor
    "fake/fraudulent order" category is not a false courier-attempt-status
    measure. Those sources cannot be combined into a Valmo distribution.
    """
    fig, ax = plt.subplots(figsize=(11.8, 5.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.suptitle('Why the cause mix is still unknown', fontsize=17,
                 fontweight='bold', color=C_PRIMARY, y=0.98)

    cards = [
        (0.78, C_SECONDARY, 'CASE BRIEF',
         'Illustrative RTO and cost inputs.\nNo order-level Valmo causes.'),
        (0.57, C_WARN, 'PUBLIC SOURCES',
         'Other networks and definitions.\nUseful hypotheses; not a Valmo split.'),
        (0.36, C_ACCENT, 'PILOT BASELINE AUDIT',
         'Define the denominator, audit reason codes,\nand map route/carrier overlap before testing.'),
        (0.15, C_ALERT, 'VALUABLE NEXT STEP',
         'Define the denominator → audit codes →\ntest one intervention against control.'),
    ]
    for y, color, title, body in cards:
        box = FancyBboxPatch((0.08, y - 0.075), 0.84, 0.15,
                             boxstyle='round,pad=0.012,rounding_size=0.018',
                             facecolor='white', edgecolor='#CBD5E1', linewidth=1.1,
                             transform=ax.transAxes)
        ax.add_patch(box)
        ax.add_patch(Rectangle((0.08, y - 0.075), 0.018, 0.15,
                               facecolor=color, edgecolor='none',
                               transform=ax.transAxes))
        ax.text(0.13, y + 0.026, title, transform=ax.transAxes,
                fontsize=10.2, fontweight='bold', color=color, va='center')
        ax.text(0.48, y, body, transform=ax.transAxes,
                fontsize=10, color=C_DARK, va='center', linespacing=1.35)
    for y in (0.675, 0.465, 0.255):
        ax.annotate('', xy=(0.5, y - 0.04), xytext=(0.5, y + 0.04),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops={'arrowstyle': '-|>', 'color': '#94A3B8', 'lw': 1.3})
    fig.text(0.5, 0.015,
             'Do not combine unlike external categories without authorized order-level baselines.',
             ha='center', fontsize=9, color=C_MUTED)
    save(fig, 'evidence_graded_causes.png')

# FIGURE 2 - the NDR intervention window
# ---------------------------------------------------------------------------
def fig_ndr_window():
    """Show the proposed contact window as a pilot design, not a sourced effect."""
    fig, ax = plt.subplots(figsize=(14.6, 4.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.suptitle('Pilot hypothesis: make the next step available after a failed attempt',
                 fontsize=15, fontweight='bold', color=C_PRIMARY, y=0.96)

    nodes = [
        (0.12, '1. Attempt fails', 'Record the event and\nits source reason code', C_SECONDARY),
        (0.37, '2. Contact window', 'Offer a customer-chosen\nretry or reschedule', C_ACCENT),
        (0.63, '3. No crossover', 'Follow original assignment;\nlog delivery and costs', C_WARN),
        (0.88, '4. Compare outcomes', 'Treatment vs concurrent\ncontrol, with guardrails', C_PRIMARY),
    ]
    for x, title, body, color in nodes:
        ax.text(x, 0.62, title + '\n\n' + body, transform=ax.transAxes,
                ha='center', va='center', fontsize=9.4, fontweight='bold',
                color=C_DARK, linespacing=1.3,
                bbox={'boxstyle': 'round,pad=0.55', 'fc': 'white',
                      'ec': color, 'lw': 2})
    xs = [n[0] for n in nodes]
    for i in range(len(xs) - 1):
        ax.annotate('', xy=(xs[i+1] - 0.088, 0.62), xytext=(xs[i] + 0.088, 0.62),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops={'arrowstyle': '-|>', 'color': C_MUTED, 'lw': 1.8})

    # Balanced Dual Phase Containers (Replacing ambiguous floating arrow)
    # Container A: Customer Response Window (under Steps 1 & 2)
    rect_a = FancyBboxPatch((0.02, 0.16), 0.46, 0.23,
                            boxstyle='round,pad=0.012,rounding_size=0.02',
                            facecolor='#F0FDFA', edgecolor=C_ACCENT, linewidth=1.4,
                            transform=ax.transAxes)
    ax.add_patch(rect_a)
    ax.text(0.25, 0.32, 'STAGE A: PRE-SPECIFIED RESPONSE WINDOW (2h \u2013 24h)',
            ha='center', va='center', fontsize=8.8, fontweight='bold', color=C_ACCENT,
            transform=ax.transAxes)
    ax.text(0.25, 0.22, 'Contact window pre-specified from baseline operations (e.g., 2h \u2013 24h);\ntestable design choice to evaluate customer retry take-up without assumed lift.',
            ha='center', va='center', fontsize=7.8, color=C_DARK, linespacing=1.35,
            transform=ax.transAxes)

    # Container B: Concurrent Evaluation & Guardrails (under Steps 3 & 4)
    rect_b = FancyBboxPatch((0.52, 0.16), 0.46, 0.23,
                            boxstyle='round,pad=0.012,rounding_size=0.02',
                            facecolor='#EFF6FF', edgecolor=C_SECONDARY, linewidth=1.4,
                            transform=ax.transAxes)
    ax.add_patch(rect_b)
    ax.text(0.75, 0.32, 'STAGE B: CONCURRENT A/B EVALUATION & GUARDRAILS',
            ha='center', va='center', fontsize=8.8, fontweight='bold', color=C_SECONDARY,
            transform=ax.transAxes)
    ax.text(0.75, 0.22, 'Strict cluster RCT tracking: measures true incremental delivery lift\nand itemized operational costs vs control, with zero experimental crossover.',
            ha='center', va='center', fontsize=7.8, color=C_DARK, linespacing=1.35,
            transform=ax.transAxes)

    fig.text(0.5, 0.035,
             'External vendor timing claims are not used as effect sizes. Measure incremental delivery and every added cost in the pilot.',
             ha='center', fontsize=8.5, color=C_MUTED)
    save(fig, 'ndr_window.png')

# FIGURE 3 - prioritisation matrix
# ---------------------------------------------------------------------------
def fig_prioritisation_matrix():
    """
    Prioritise on the axes the review asked for: impact, confidence, cost and
    effort. Two of those four are plotted; the other two are encoded as ring
    thickness (cost) and marker shape (effort), so all four are visible at once
    without a table.

    Positions are ORDINAL JUDGEMENT, and the chart says so in its own subtitle.
    Confidence is the only axis with a defensible basis before we have data,
    because confidence here means "how much would we need to learn first", not
    "how good do we think this is".
    """
    # label, impact, evidence-needed, cost, effort, colour, dx, dy, ha, va
    items = [
        ('Agree the denominator\n& reason codes', 2.35, 4.85, 1.0, 1.2,
         C_PRIMARY, 0.0, 0.44, 'center', 'bottom'),
        ('Reason-code audit loop\n(are the codes right?)', 3.20, 4.30, 1.8, 2.0,
         C_PRIMARY, 0.0, -0.44, 'center', 'top'),
        ('First-failed-attempt outreach,\ntimed inside 24 h', 4.65, 3.30, 1.5, 2.4,
         C_SUCCESS, 0.0, 0.46, 'center', 'bottom'),
        ('Voluntary digital payment\nat the door', 3.90, 2.30, 2.4, 2.6,
         C_WARN, -0.30, -0.16, 'right', 'top'),
        ('Causal re-attempt targeting\n(which orders are worth a retry)',
         4.55, 1.75, 3.2, 4.0, C_ACCENT, 1.02, -0.26, 'left', 'top'),
        ('Distance-aware rider pay\n+ attempt verification',
         3.50, 1.32, 4.4, 3.8, C_SECONDARY, 1.06, -0.22, 'left', 'top'),
        ('Seller-approved local re-match\nof refused stock',
         2.60, 1.15, 4.6, 4.4, C_MUTED, 1.10, -0.18, 'left', 'top'),
    ]

    fig, ax = plt.subplots(figsize=(11.6, 6.9), dpi=300)

    # Quadrant shading: lead (little evidence still needed) vs must learn first.
    ax.axhspan(3.0, 5.55, xmin=0.0, xmax=0.50, color='#ECFDF5', alpha=0.60, zorder=0)
    ax.axhspan(3.0, 5.55, xmin=0.50, xmax=1.0, color='#FFFBEB', alpha=0.55, zorder=0)
    ax.axvline(2.95, color=C_MUTED, linewidth=1.1, linestyle='--', zorder=1)

    ax.text(1.78, 5.62, 'LEARN FIRST \u2014 definition and data foundations',
            ha='center', fontsize=10.0, fontweight='bold', color='#047857')
    ax.text(4.25, 5.62, 'TEST LATER \u2014 intervention hypotheses',
            ha='center', fontsize=10.0, fontweight='bold', color=C_WARN)

    for label, imp, need, cost, eff, col, dx, dy, ha, va in items:
        # Ring thickness encodes cost (thicker ring = more expensive).
        ax.scatter([need], [imp], s=290 + cost * 155, color=col,
                   alpha=0.15, edgecolor=col, linewidth=1.5, zorder=3)
        # Marker shape encodes effort.
        marker = 'o' if eff < 2.0 else ('s' if eff < 3.2 else '^')
        ax.scatter([need], [imp], s=86, color=col, marker=marker,
                   edgecolor='white', linewidth=1.5, zorder=4)
        ax.text(need + dx, imp + dy, label, ha=ha, va=va,
                fontsize=8.6, color=C_DARK, linespacing=1.4, zorder=5,
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='none', alpha=0.85))

    ax.set_xlim(0.52, 5.62)
    ax.set_ylim(1.72, 5.86)
    ax.set_xlabel('EVIDENCE STILL NEEDED  \u2192  how much must we learn '
                  'before this can be acted on?',
                  fontsize=10.0, fontweight='bold', color=C_DARK)
    ax.set_ylabel('POTENTIAL REACH  \u2192  ordinal hypothesis only',
                  fontsize=10.0, fontweight='bold', color=C_DARK)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([2, 3, 4, 5])
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # Encoding key.
    ax.text(0.5, 0.022,
            'ring thickness = cost   \u00b7   marker shape = effort   '
            '(\u25cf low, \u25a0 medium, \u25b2 high)',
            transform=ax.transAxes, ha='center', fontsize=8.5, color=C_MUTED,
            style='italic')

    ax.set_title('Why the first interventions lead',
                 fontsize=13.4, fontweight='bold', color=C_PRIMARY, pad=26)
    ax.text(0.5, 1.015,
            'Ordinal judgement, not measured effect \u2014 positions are our read, not an estimate',
            transform=ax.transAxes, ha='center', fontsize=9.4, color=C_MUTED,
            style='italic')

    fig.text(0.5, -0.045,
             'Test order is DEPENDENCY-FIRST, not lift-first: nothing can be tested before the denominator and the reason codes are agreed, '
             'because the reason code is what tells you\nwhich lever applies. '
             'All positions are team judgement, not measured impact, cost, or effort. '
             'Start with denominator and reason-code quality; set intervention order after baseline data.',
             ha='center', fontsize=8.1, color=C_MUTED, linespacing=1.6)

    save(fig, 'prioritisation_matrix.png')


# ---------------------------------------------------------------------------
# FIGURE 5 - the brief's distance pattern against published reality
# ---------------------------------------------------------------------------
def fig_gross_ceiling_scenarios():
    """Show conditional gross-ceiling arithmetic only; no lift is forecast."""
    orders = 10_000
    lifts_pp = [0.5, 1.0, 2.0]
    gross = [int(orders * pp / 100 * 120) for pp in lifts_pp]
    labels = [f'{pp:.1f} pp' for pp in lifts_pp]

    fig, ax = plt.subplots(figsize=(11.8, 4.7), dpi=300)
    bars = ax.bar(labels, gross, color=[C_SECONDARY, C_ACCENT, C_PRIMARY],
                  width=0.52, zorder=3)
    for bar, value in zip(bars, gross):
        ax.text(bar.get_x() + bar.get_width()/2, value + 500,
                f'up to ₹{value:,}', ha='center', fontsize=12,
                fontweight='bold', color=C_DARK)
    ax.set_ylim(0, 29_000)
    ax.set_ylabel('Gross reverse-cost ceiling (₹)', fontsize=10, color=C_DARK)
    ax.set_title('If a measured delivery lift occurred across 10,000 eligible first-failure cases',
                 fontsize=13, fontweight='bold', color=C_PRIMARY, pad=18)
    ax.text(0.5, 1.01, 'Illustrative arithmetic: eligible cases × absolute percentage-point lift × up to ₹120',
            transform=ax.transAxes, ha='center', fontsize=9.2, color=C_MUTED,
            style='italic')
    ax.grid(axis='y', alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.025,
             'Not a forecast or net benefit. The brief labels its ₹120 input illustrative. '
             'Subtract measured avoidable cost, intervention costs, and adverse effects.',
             ha='center', fontsize=8.5, color=C_MUTED)
    save(fig, 'gross_ceiling_scenarios.png')


def fig_case_pack_distance():
    """Reproduce only the case-pack bands and flag its conflicting labels."""
    names = ['Nearby\n(~2 km)', 'Moderate\n(~5 km)', 'Far\n(10 km+)']
    values = [15, 17, 22]
    fig, ax = plt.subplots(figsize=(10.2, 4.2), dpi=300)
    bars = ax.bar(names, values, color=[C_SUCCESS, C_WARN, C_ALERT],
                  width=0.55, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.35, f'{value}%',
                ha='center', fontsize=13, fontweight='bold', color=C_DARK)
    ax.set_ylim(0, 27)
    ax.set_ylabel('Case-pack reported share (%)', fontsize=10, color=C_DARK)
    ax.set_title('Illustrative distance pattern supplied in the case pack',
                 fontsize=13, fontweight='bold', color=C_PRIMARY, pad=20)
    ax.text(0.5, 1.015, 'Use as a planning hypothesis; no Valmo event-level observations supplied',
            transform=ax.transAxes, ha='center', fontsize=9, color=C_MUTED,
            style='italic')
    ax.grid(axis='y', alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.02,
             'Source table row says “orders returned undelivered”; its caption says “Delivery Success by Distance from Hub”. '
             'Confirm which measure was intended.', ha='center', fontsize=8.1,
             color=C_MUTED)
    save(fig, 'case_pack_distance.png')




def main() -> int:
    print('=' * 74)
    print('   RESEARCH METHODOLOGY FIGURES  (source and unit labelled per figure)')
    print('=' * 74)
    fig_evidence_graded_causes()
    fig_ndr_window()
    fig_prioritisation_matrix()
    fig_gross_ceiling_scenarios()
    fig_case_pack_distance()
    print('\n  5 figures written.')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
