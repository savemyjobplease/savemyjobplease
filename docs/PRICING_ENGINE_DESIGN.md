# Pricing engine — design notes & spec mapping

How `src/pricing/` implements the POC slice of **`09 - Price Estimation: Architecture & Roadmap`**.

## Thesis (held exactly)

A **transparent, auditable cost build-up engine** — not a statistical price predictor. For each
line item it assembles a unit cost from named, individually-sourced components, every number
tracing to a rule, a named data asset, or a bounded calibration, wrapped in an honest band.
The base is **current market rate only**; margin is a downstream overlay applied at render time,
never inside the engine.

## The build-up (doc A1)

```
unit_cost = raw_material + process + secondary_ops + overhead
          + electricity + transport + profit + other
```

- **raw_material** = `gross_weight × rm_rate(grade, condition, month, band)`. `gross ≥ net`
  enforced; yield checked against per-route plausible bands; a missing rate is **surfaced, never
  invented**.
- **process** = Σ route-steps (`route × machine × diameter-band`). Rates are **human-anchored
  placeholders** (no tabulated source exists) → Bucket-C is assistive-only until validated.
- **secondary_ops** = feasibility-driven surcharge (0 unless a tolerance gap needs it).
- **overhead / electricity / transport / profit / other** — ops-cost feeds; `profit` is the
  *supplier's* margin inside the market rate (not our markup); `other` carries the named
  `tooling_amortized` sub-line.

## Three bucket strategies (doc A2)

| Bucket | Strategy | Code |
|---|---|---|
| `assortment` | catalog lookup (shallow breakdown) | `engine._bucket_a` |
| `special_but_standard` | nearest reference + Σ modification deltas (length/finish/grade/drive) | `engine._bucket_b` + `_modification_deltas` |
| `to_the_print` | analog-first; else full route build-up from physics | `engine._bucket_c` |

## Modifiers (doc A4)

Feasibility → 3-value verdict `feasible | needs_secondary_op | needs_SME_review` with a reason
qualifier; **missing tolerance → two explicit scenarios** (`as_if_loose` / `as_if_tight`), never
assumed. L:D bend trigger. Quantity-tier sourcing basis + **MOQ-vs-requested rule** (price at MOQ,
show both unit costs). Geography recorded as a basis, **never blended**.

## Uncertainty (doc A5)

Per-category bands, **composed not assumed**, floor ±10%. Base by bucket (A ±10 / B ±15 / C ±25),
widened by logged triggers (stale rate, geometry weight, substituted grade, placeholder process,
FX staleness). Confidence is decomposed (the weak link is visible) with one scalar for the gate.

## Calibration (doc A6) — the hybrid

The chosen design: **an ML model supplies the bounded correction**, disciplined to remain a
*correction*, not the predictor:
`unit_price = rules_base × c[family, supplier, region]`, with

- **clamp** to `[0.85, 1.15]` (a fit wanting more → rules-gap routed to review),
- **minimum cell support** (`n < 5` → no fitted factor; hierarchical back-off → 1.0; band widened),
- **exclusion** of FX-volatile / brand-locked / import history from the training pool,
- **full visibility** in the trace (`rules X × 1.07 [family/supplier/region, n=6]`),
- the rules base always standing underneath, inspectable.

Implemented in `calibration.py` (`CalibrationModel`, sklearn backend; grid-mean fallback).

## Refusals (doc Part B) — `refusals.py`

Enforced as hard post-conditions that escalate status `auto_emit → assistive_only → review →
block`: never invent a tolerance; never reuse history/calibration for brand-locked/FX items;
never bury margin in the base; never blend multi-modal data; never apply calibration on
under-supported cells or past the clamp; never present a confident point on placeholder/stale
inputs; never report `gross < net`; never fabricate a missing rate; never present tooling-free
low-qty as final; the band is never an accuracy guarantee.

## Spec phase → code

| Doc phase | Module |
|---|---|
| 0 — costing logic & route templates | `reference_data.RouteTemplateKB` |
| 1 — pricing & ground-truth foundation | `reference_data.{RawMaterialMatrix,ReferencePrices,OpsCost}` |
| 2 — transparent base engine (3 buckets) | `engine.py`, `buildup.py` |
| 3 — cost-modifier layer | `buildup.py`, `engine._feasibility` |
| 4 — calibration & honest uncertainty | `calibration.py`, `uncertainty.py` |
| Part B — refusals | `refusals.py` |
| 5–9 (validation, alternatives, revision, freshness, autonomy) | **future work** |

## Contract

`EstimateResult` (`contracts.py`) is the output: base & calibrated unit price, per-category
`CostBreakdown`, gross/net weight, MOQ (+ gap), sourcing tier, geography basis, lead-time triplet,
feasibility, validity, calibration record, per-category confidence, flags, assumptions, full
reasoning trace, and a `status`. The contract is append-only — reliability detail lives in fields,
never as silent new numbers.

## Known limits

Process rates are placeholders → C is assistive-only; synthetic reference data ships for the demo;
calibration history is small/synthetic; bands are honest tolerances, not measured accuracy. Wire
real `ai_rfq_sample` tables + a converted-quote corpus, then implement Phases 5–9.
