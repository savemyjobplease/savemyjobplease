"""Builds notebooks/03_price_estimation.ipynb (driver/demo over src/pricing).
Cell sources are RAW triple strings; no triple-quotes inside cells.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def co(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 03 - Price Estimation (transparent cost build-up engine)

Implements the POC slice of **design doc 09 - Price Estimation: Architecture & Roadmap**.
This is **not** a statistical price predictor. It is an auditable cost build-up:

```
unit_cost = raw_material + process + secondary_ops + overhead
          + electricity + transport + profit + other
```

- **3 bucket strategies** - A `assortment` (catalog lookup), B `special_but_standard`
  (nearest reference + modification deltas), C `to_the_print` (analog-first, else full
  route build-up from physics).
- **Honest per-category bands** (floor +/-10%); **bounded calibration** (the hybrid ML
  correction, clamped to +/-15%, segment+n visible, OFF for flagged items);
  **hard refusal rules** (missing rate -> block, missing tolerance -> loose/tight + review,
  brand-locked -> vendor RFQ, placeholder process rates -> assistive-only).
- The base is **current market rate only**; margin is a downstream overlay, never in the engine.

The engine lives in the importable package `src/pricing/`; this notebook is a thin driver.
Reference data here is **synthetic** (no real data) - swap in the real `ai_rfq_sample` tables.
""")

md(r"### Bootstrap - put `src/` on the path and load keys (optional, for analog NLP)")
co(r"""
import sys, os
from pathlib import Path

def _find_root():
    for d in [Path.cwd(), *Path.cwd().resolve().parents]:
        if (d / "src" / "pricing").is_dir():
            return d
    return Path.cwd()

ROOT = _find_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
print("project root:", ROOT)

# On a bare Colab runtime without the repo, clone it instead, e.g.:
#   !git clone <your-repo-url> AI_RFQ && %cd AI_RFQ
# then re-run this cell.

try:
    from config import load_keys      # src/config.py - shared key loader
    load_keys(verbose=True)           # reads secret_keys.env -> OPENAI_API_KEY etc. (optional here)
except Exception as e:
    print("key loader not run (fine - analog falls back offline):", e)

from pricing import (PriceEstimationEngine, CalibrationModel, AnalogIndex,
                     LineItem, synthetic_data as sd)
print("pricing engine imported.")
""")

md(r"""
### 1. Reference data (doc Phase 1 + Phase 0 KB)

The single source of truth for price inputs: the raw-material rate matrix, ops-cost,
catalog/reference prices, and the route-template KB (process rates are **human-anchored
placeholders** by construction). Here it is synthetic.

**To use real data**, replace this with loaders over the schema tables, e.g.
`RawMaterialMatrix.from_schema(df_ref_raw_material_price)`.
""")
co(r"""
ref = sd.build_reference_data()
print("RM grades:", sorted({k[0] for k in ref.rm._rows}))
print("catalog SKUs:", list(ref.ref_prices.catalog.keys()))
print("route families:", list(ref.routes.templates.keys()))
print("FX USDINR:", ref.fx_usdinr, "| brand-locked:", ref.brand_locked_customers)
""")

md(r"""
### 2. Calibration layer (doc A6) - the HYBRID ML correction

An ML model supplies the bounded correction `c[family, supplier, region]`, but is
disciplined to stay a *correction*: clamped to +/-15%, suppressed to 1.0 below
minimum cell support, OFF for FX-volatile/brand-locked history, and fully visible
in the trace. The rules base always stands underneath.
""")
co(r"""
history = sd.build_calibration_history()
print("calibration history rows:", len(history), "| flagged (excluded):",
      int(history["flagged"].sum()))
calib = CalibrationModel(backend="ml").fit(history)
print("backend:", "ml" if calib.ml is not None else "grid",
      "| dense cells (n>=N_min):", [k for k, (m, n) in calib.grid.items() if len(k) == 3 and n >= calib.n_min])
""")

md(r"### 3. Analog index (doc A2.3) - Bucket-C front line (OpenAI embeddings if a key is set, else offline)")
co(r"""
analog = AnalogIndex(sd.build_analog_corpus(), use_openai=True)   # falls back offline w/o key
print("analog mode:", analog.mode)
""")

md(r"### 4. Build the engine")
co(r"""
engine = PriceEstimationEngine(ref, calibration=calib, analog_index=analog)
print("engine ready.")
""")

md(r"""
### 5. Estimate across all three buckets (+ the refusal cases)

Each estimate carries the full build-up, per-category bands, feasibility, validity,
calibration, and a status: `auto_emit | assistive_only | review | block`.
""")
co(r"""
estimates = []
for li in sd.sample_line_items():
    est = engine.estimate(li)
    estimates.append(est)
    print("=" * 90)
    print(est.summary())
""")

md(r"""
### 6. The single-item API (the black box)

`engine.estimate(row_dict)` accepts one line item (nulls allowed) and returns one
`EstimateResult` - the auditable cost sheet, not just a number. Below: a Bucket-B
line with several fields omitted.
""")
co(r"""
one_row = {
    "id": "demo-line",
    "bucket": "special_but_standard",
    "part_family": "shcs",
    "size": "M8x50",
    "nominal_diameter_mm": 8, "nominal_length_mm": 50,
    "material_grade": "10.9", "material_family": "alloy_steel",
    "finish": "znni", "quantity": 4000,
    "customer": "acme auto", "supplier": "s1", "region": "West",
    "ref_standard": "DIN 912", "ref_size": "M8x40", "ref_grade": "10.9",
    "ref_finish": "plain", "ref_length_mm": 40,
    # requested_tolerance_um, drawing dims, weights, tooling ... all omitted (None)
}
est = engine.estimate(one_row)
print(est.summary())
print("\n--- single float (current-market-rate unit price, INR) ---")
print(round(est.unit_price, 3))
print("status:", est.status, "| review:", est.needs_review, "| blocked:", est.blocked)
""")

md(r"""
### 7. The reasoning trace (auditability)

Every number traces to a rule, a named asset, or the bounded calibration. This is the
"shows its working" property the design doc requires.
""")
co(r"""
for line in est.reasoning_trace:
    print(" -", line)
print("\nassumptions:")
for a in est.assumptions:
    print("   *", a)
""")

md(r"""
### 8. Wiring to real data + how this maps to doc 09

**Real reference data** (replace the synthetic builder):

```python
import pandas as pd, sqlalchemy as sa
from pricing import RawMaterialMatrix, ReferenceData, OpsCost, ReferencePrices, RouteTemplateKB
eng = sa.create_engine("postgresql://user:pw@localhost:55432/ai_rfq_sample")
rm = RawMaterialMatrix.from_schema(pd.read_sql("SELECT * FROM ref_raw_material_price", eng))
ref = ReferenceData(rm=rm, ops=OpsCost(), ref_prices=ReferencePrices(), routes=RouteTemplateKB())
# populate ref.ref_prices.catalog from your reference-price / release tables
# seed route templates + per-step rates from the Phase-0 plant walkthrough
```

**Real line items** come from the extraction stage (notebook 02) - each masked drawing's
fastener record maps onto a `LineItem` (family, dims, grade, finish, tolerance, qty, ...).

**Calibration history** = converted RFQs (quote-date cost basis), with FX-volatile /
brand-locked rows flagged so the model excludes them.

| Doc phase | Where it lives |
|---|---|
| 0 - costing logic & route templates | `pricing/reference_data.py` (`RouteTemplateKB`) |
| 1 - pricing/ground-truth foundation | `RawMaterialMatrix`, `ReferencePrices`, `OpsCost` |
| 2 - transparent base engine (3 buckets) | `pricing/engine.py` + `buildup.py` |
| 3 - cost-modifier layer | `buildup.py` (L:D, qty-tier/MOQ), `engine._feasibility` |
| 4 - calibration + honest bands | `calibration.py`, `uncertainty.py` |
| (Part B) refusal rules | `refusals.py` |

**Limits (faithful to the doc):** process rates are human-anchored placeholders, so every
Bucket-C estimate is **assistive-only** until validated by the plant walkthrough; the band is
an honest tolerance, not an accuracy guarantee; Phases 5-9 (holdout validation, material
alternatives, revision delta, live freshness, autonomy) are future work.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.x"}}

out = Path(__file__).resolve().parents[1] / "notebooks" / "03_price_estimation.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

errs = 0
for j, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    if any(l.lstrip().startswith(("!", "%")) for l in c["source"].splitlines()):
        continue
    try:
        compile(c["source"], f"cell{j}", "exec")
    except SyntaxError as e:
        errs += 1; print("SYNTAX", j, e)
print(f"wrote {out} ({len(nb['cells'])} cells), syntax errors: {errs}")
