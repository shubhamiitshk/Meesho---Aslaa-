# Meesho App Teardown — Desk Review Completed, Device Test Not Done

**Status: desk review of publicly documented flows only. No Meesho app was installed,
opened or navigated on a device for this submission, and no order was placed.** No
rider was contacted, no courier was contacted, and this document establishes no
Meesho or Valmo affiliation, access, or permission to test delivery operations.

This replaces the previous version, which was a proposed checklist for a first-hand
review that had not happened. The brief asks teams to "play with the Meesho-app to
identify flaws, place orders & observe what reasons could lead to an RTO". We did
not do that, and the honest response is to say what we did read, what it establishes,
and what it cannot.

## What was reviewed, and where

| Source | What it documents | Not observable here |
|---|---|---|
| Meesho customer-help content on pay on delivery | That COD availability is not uniform across pin codes; that customers are asked for exact cash because riders often lack change; that change may be credited to a wallet | Which flows the current app build actually shows, on which account states |
| Publicly documented doorstep payment paths | That digital payment at the door is offered as an option alongside cash | Whether it is presented as an equal option, prominent or buried, in the live build |
| Payment-provider developer documentation | What a doorstep collect requires technically, and that no provider publishes a success rate for it | Nothing about Meesho's own integration |
| Marketplace competitor help pages | That pay-on-delivery availability and change handling are common patterns | Anything Meesho-specific |

## What a desk review can and cannot establish

**It can establish that mechanisms are commodity.** This was the review's most
consequential finding, and it changed the proposal. Doorstep dynamic QR for COD is
documented by at least one large Indian marketplace and shipped as a named product
by three payment providers. First-attempt-success rider pay has been published
industry practice in India since at least 2017. COD risk gating at checkout, NDR
automation with reason codes, and address validation are all commercial products.

**It cannot establish prevalence, wording, or friction.** We do not know whether the
checkout already carries a landmark prompt, whether a skippable step would add
friction, whether the tracking screen explains a failed attempt, or whether there is
any customer-controlled next step after a refusal. Those are precisely the questions
that decide whether our proposed prompt is redundant.

**It cannot establish outcomes.** Nothing observed from documentation says anything
about delivery success, RTO, willingness to pay, conversion impact, or causality.

## The three flows, and what each would take

| Flow | Recorded from documentation | **Not observed** | Needs a device to settle |
|---|---|---|---|
| COD checkout | Payment options include COD; availability varies by pin code; exact-cash request exists because riders may lack change | Whether a landmark or address-quality prompt already exists, and whether any step is mandatory | Whether our prompt duplicates existing UI, and what it costs in completion |
| Order tracking and help | Notifications and support entry points are documented as available | Live status labels, the post-attempt screen, rescheduling controls | Which communication gap, if any, remains for the customer to act on |
| Missed attempt or refusal | Return and refund paths are documented | The in-app state after a failed attempt | Whether the app explains payment, return or support options at the point of failure |

## Safe protocol for the device test that was not run

1. Record date, app version if visible, device and OS, region at a coarse level,
   account state, and the exact flow entered.
2. Use an existing or otherwise authorised account. Do not place unnecessary
   orders, stage a delivery failure, evade app rules, contact a courier to
   engineer an event, or alter a real order in a way that could cause loss or
   inconvenience to anyone.
3. Inspect only normal, user-visible screens. Do not record or photograph another
   person's screen. Keep screenshots and notes free of names, phone numbers, full
   addresses, order IDs and payment details.
4. For any state that is not naturally available on an authorised account, record
   **not observed**. Do not simulate a state and report it as a product fact.
5. Keep direct observation separate from interpretation, and quote interface copy
   exactly only where it is publicly reproducible without personal data.

## Capture record

Keep completed notes in an access-controlled location; this repository is not a
participant-data intake system. A minimal record:

| Field | Purpose |
|---|---|
| `observation_id` | Random note identifier with no account or order linkage |
| `observed_at` | Date and time of observation |
| `app_version_device_region` | Coarse context needed to reproduce the observation |
| `flow_step` | Checkout, tracking/help, or missed-attempt/refusal |
| `observed_or_not_observed` | Prevents an inferred or simulated state being recorded as fact |
| `screen_behavior` | Factual description of what appeared and what action was available |
| `exact_copy_if_public` | Optional exact wording, redacted of personal information |
| `interpretation_and_limit` | Why it matters, and what one observation cannot establish |
| `evidence_reference` | Private reference only if a redacted, authorised screenshot exists |

Do not append app observations to the synthetic fieldwork CSVs, and do not present
an unobserved flow as a finding.

## How results may change the deck

An observation could reveal that our proposed checkout prompt already exists, in
which case we drop or redesign it and test conversion impact before recommending
anything. It could not establish feature prevalence across the marketplace, delivery
outcomes, willingness to pay, or causality. The app is one surface; the failure
mechanism we care about lives in the last mile, which no customer-facing screen
observes.

**Completion record:** desk review of publicly documented sources completed. First-hand
device walkthrough **not** completed. Update this line only after an authorised,
dated observation, and report its exact scope and limitations alongside it.