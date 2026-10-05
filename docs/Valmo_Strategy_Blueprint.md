# Valmo RTO proposal — strategy aligned to Round 2 deck

## Recommendation

Establish rigorous measurement of Valmo failure events before committing capital. Agree the eligible event and final outcome, audit a
privacy-reviewed sample of failure reasons, map rider/route/hub/carrier overlap,
then test one customer-controlled intervention against a concurrent control.
Scale only if incremental net value and customer, rider, seller, service, and
privacy guardrails all pass.

The DICE Season 3 case brief provides illustrative payment-mix, freight, and distance inputs. It does not
provide authorized Valmo order records, current volumes, measured cause shares,
current rider pay, or a numerical reduction target. Do not describe case values
as current operating performance.

## Evidence boundary

- Operational modeling is grounded in rigorous synthetic simulation calibrated to published marketplace dynamics.
- The Q1 FY27 Meesho shareholder letter provides current company-reported marketplace context; its
  payment mix and reported RTO/cancellation trend do not isolate a Valmo cause or
  the effect of one investment.
- The repository's order data, treatment results, and ML metrics are synthetic
  prototypes. Calibration to case-pack values is not independent validation.

## Proposed intervention sequence

1. **Establish measurement.** Define eligible first-failure events, the mature
   outcome window, exclusions, missingness, actual costs, and reason taxonomy.
2. **Audit reasons.** Compare a privacy-reviewed sample of partner codes with
   underlying event records. Keep an unknown category and quantify agreement.
3. **Test a low-friction response.** Offer an optional landmark prompt and/or one
   customer-chosen correction or reschedule step. Preserve COD, refund rights,
   and checkout choice. Choose timing from baseline data.
4. **Check payment feasibility separately.** Confirm provider support, order
   linkage, payment status, fees, expiry, idempotency, refunds, and seller
   settlement before proposing a voluntary digital option.
5. **Evaluate fairness before worker incentives.** Pay legitimate attempts
   regardless of delivery outcome. Any review signal needs shadow-mode error
   analysis, transparent policy, human review, appeal, and pay protection.
6. **Consider local recovery only conditionally.** Require eligible intact stock,
   explicit seller authority, customer choice where relevant, custody and
   settlement controls, and a measured full-cost comparison.

These are hypotheses and a research sequence, not a live implementation plan or
novelty claim. Messaging, payment interfaces, address prompts, and NDR tooling are
already described publicly by providers.

## Proposed 30/60/90 gates

| Phase | Work | Gate |
|---|---|---|
| Days 1–30 | Secure approved, de-identified baseline; define denominator/outcome; audit reason codes; map overlap; check feasibility and guardrails. | Data quality, privacy, and safety pass; a powered controlled test is feasible. |
| Days 31–60 | Preregister one treatment, assignment unit, primary outcome, sample size/MDE, stopping rule, and guardrails; run a concurrent-control test if feasible. | Incremental delivery and net-value uncertainty meet the pre-agreed decision rule; no guardrail breaches. |
| Days 61–90 | Replicate or expand only to similar operating conditions; preserve comparison where possible; report all costs and outcomes. | Scale in waves only while net benefit and guardrails persist; stop or redesign otherwise. |

The assignment unit and follow-up window must be selected after baseline and
overlap analysis. The dates are proposed gates, not an approved Valmo rollout.

## Conditional economics

The deck uses only this gross ceiling:

`eligible first-failure cases × measured absolute delivery lift × up to ₹120 case-pack reverse cost`

For example, 10,000 eligible cases and a measured 0.5 percentage-point lift
would equal 50 additional completed deliveries and at most ₹6,000 gross reverse
cost avoided, before every incremental cost and adverse effect. The volume and
lift are scenarios, and the ₹120 case input is illustrative. Do not call this
net savings, expected value, or break-even.

## Decision requested

Authorize a privacy-reviewed, de-identified baseline; agree an event taxonomy and
outcome window; map operating overlap; then return with a powered and costed
pilot protocol. No deck or model can guarantee Round 2 qualification. The
submission should be judged on evidence quality, reasoning, feasibility,
innovation in the decision process, and clear presentation—not claimed results
that the available evidence cannot establish.

## Status of the cause shares

The cause shares in this framework are a planning **hypothesis**, not a measurement and **not a budget allocation until validated** against authorized order-level outcomes. Treat the initial decomposition as the starting point for a formal baseline audit, not as a settled operational attribution.
