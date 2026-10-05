# Public-source and prior-art scan

**Reviewed:** 4 October 2026. Every capability below is taken from a **public source** — a provider’s own documentation or a published filing. This is a focused desk scan, not a full market census,
product teardown, or comparative performance study. Public product pages describe
capabilities and vendor claims; they do not establish causal effects for Valmo.

## Company context

Meesho's Q1 FY27 shareholder letter reports that about 37% of shipped orders were
prepaid and attributes lower RTO/cancellations year over year to its own fraud-detection and
predictive-routing investments. These are company-reported, marketplace-wide
statements. The letter does not quantify each investment's contribution or
provide a Valmo cause split. It is used as context only.
[Meesho Q1 FY27 shareholder letter](https://static-assets.meesho.com/investor-relations/1784806298911/Q1ShareholdersLetter.pdf)

## Relevant capabilities already described publicly

| Capability | Public material reviewed | What it establishes | What it does not establish |
|---|---|---|---|
| COD and payment-on-delivery workflows | [Delhivery SmartNDR](https://help.delhivery.com/docs/smartndr); [Cashfree UPI QR](https://www.cashfree.com/upi-qr-code/) | Providers document NDR workflows and merchant QR payment capabilities. | Support in Valmo, rider-app integration, order linkage, PSP fees, settlement/refund behavior, or incremental delivery lift. |
| Customer notifications, NDR, and delivery recovery | [Shiprocket support overview](https://support.shiprocket.in/support/solutions/articles/152000000948-how-can-i-reduce-my-rto-shipments-what-are-the-tools-offered-by-shiprocket-); [Delhivery NDR workflow](https://help.delhivery.com/docs/non-delivery-report-ndr) | Recovery tooling and configurable workflows are established prior art. | Comparable operating rules, independent effect estimates, or suitability for a multi-partner Valmo network. |
| RTO reason categories and vendor recommendations | [ClickPost 90-Day RTO Reduction Playbook](https://www.clickpost.ai/90-day-rto-reduction-playbook) | A vendor describes its own taxonomy, risk signals, and recommendations. It labels its 6–12% band as fake/fraudulent orders. | A false courier-attempt-status rate, independently audited category share, or transferable Valmo result. |
| Address quality and first-delivery outcomes | [UPU Think Tank Brief No. 2/2025](https://www.upu.int/UPU/media/upu/publications/202507dprmThinkTankBriefNo2-2025.pdf) | Global postal-operator analysis provides broader context on address quality. | An Indian e-commerce, Meesho, or Valmo effect size. |
| National addressing infrastructure | [India Post DIGIPIN](https://www.indiapost.gov.in/digipin); [UPU summary, 26 Aug 2026](https://www.upu.int/en/news/2026/august/indias-digipin-and-the-future-of-postal-addressing) | **India Post** publishes DIGIPIN as an open-source national addressing grid built with IIT Hyderabad and ISRO, released as foundational digital public infrastructure in March 2025 and in pilot since January 2026. A 10-character code resolves a roughly 3.82 m cell from coordinates alone, with no central registry and no personal data collected. | That Valmo has it, that it is production-ready for daily ingestion at network scale, or any coverage figure. No coverage number is published. It is **not a Valmo capability** and it verifies no address; it is an option to watch for the rural long tail, nothing more. |

## Provider retry behaviour, from the documentation

Read from the providers’ own developer documentation, because the distinction decides whether the option is buildable at all:

| Provider | What the docs say | Consequence for us |
|---|---|---|
| Cashfree | A static QR is a single **reusable** merchant code; the customer enters the amount. A dynamic QR is generated **per transaction** with the amount fixed, and the docs state it times out and must be regenerated. |
| PhonePe | One unique QR per transaction, pre-loaded with the exact amount, expiring after payment or a set time limit. |
| Paytm | Dynamic QR is enterprise-gated and requires a customer-facing display; the docs name door-step delivery collection as use case one. |

**A "reusable dynamic QR" is a contradiction in terms** and the deck must not use the phrase. Dynamic means per-transaction and time-limited; reusable means static with a customer-entered amount. What no provider documents is an order-bound QR pre-generated, cached in the rider app, created without a live server round-trip and reconciled later. That is the only defensible novelty claim in this area, and it is a reliability claim, not a product claim.

Offline payment is **not** an available fallback: the RBI offline digital payments framework caps a single offline transaction at ₹50 and an offline balance at ₹2,000 per instrument, and as of July 2026 NPCI was still developing offline UPI for conventional terminals with no published operating rules. A doorstep flow must not be designed against an offline rail that does not exist.

No provider publishes a success rate for doorstep QR collection. Every published RTO-reduction figure in this market comes from checkout-side COD gating or NDR automation, none of it randomised — which is why we treat the effect size as unmeasured rather than assuming one.

## Implications for the proposal

Do not present a QR code, messaging flow, generic RTO score, or NDR workflow as a
novel product feature. The defensible proposal is to validate the operating data,
map partner and rider overlap, then test one customer-controlled action against
a concurrent comparison with full costs and guardrails. Even this evaluation
design is a recommendation, not a claim that the rest of the market has never run
a randomized pilot.

Before any implementation, confirm partner-specific APIs, PSP participation,
idempotency, expiry and reconciliation, payment confirmation, refund and seller
settlement behavior, customer-consent rules, fees, support ownership, attempt
policy, and rider-pay protections. Treat these as discovery questions until
Valmo-side owners confirm them.

## Evidence limitations

Provider pages can change and may present vendor-authored claims. This scan does
not estimate market share, compare private RTO rates, audit trial methods, or
prove a feature is absent from other systems. No reviewed public source supplies
a transferable causal estimate for the proposed Valmo workflow. Recheck source
versions and access dates before a later presentation.
