# AI RFQ — Fastener Quote Automation

End-to-end system for turning messy RFQ inputs (emails, PDFs, **engineering drawings**,
spreadsheets) into a **transparent, auditable price estimate** for fasteners (bolts, nuts,
washers, pins, studs, …). Parts span three classes: **catalog/standard**,
**special-but-standard** (a standard with modifications), and **to-the-print** (fully custom).

The pipeline has three stages:

```
 ┌─────────────────────┐     masked images      ┌────────────────────────┐    LineItem(s)   ┌───────────────────────┐
 │ 01  MASKING         │ ─────────────────────▶ │ 02  EXTRACTION         │ ───────────────▶ │ 03  PRICE ESTIMATION  │
 │ local VLM + OCR     │  data/interim/         │ GPT ‖ Claude → arbiter │   fastener facts │ transparent cost      │
 │ blacks out company  │  masked_drawings/      │ → structured record    │                  │ build-up engine       │
 │ name + logo         │                        │ (XLSX + audit JSON)    │                  │ (rules + bounded ML)  │
 └─────────────────────┘                        └────────────────────────┘                  └───────────────────────┘
   GPU · no API keys                              OpenAI + Anthropic keys                      OpenAI (optional, analog)
```

| Stage | Notebook | What it does | Needs |
|---|---|---|---|
| **01 Masking** | [`notebooks/01_masking.ipynb`](notebooks/01_masking.ipynb) | Local Qwen3-VL + OCR/PyMuPDF detect the owner **company name** (text) and **logo**, then blacken them out — *before* anything goes to the cloud. Outputs only the masked images. | GPU (Colab T4 ok). No keys. |
| **02 Extraction** | [`notebooks/02_extraction.ipynb`](notebooks/02_extraction.ipynb) | GPT + Claude extract a structured fastener record from each **masked** drawing in parallel; a GPT arbiter reconciles them → color-coded XLSX + per-drawing audit JSON. | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. |
| **03 Price estimation** | [`notebooks/03_price_estimation.ipynb`](notebooks/03_price_estimation.ipynb) | Transparent **cost build-up engine** (`src/pricing/`): raw material + process + … + bounded calibration. Per-category honest bands, hard refusal rules, full reasoning trace. | `OPENAI_API_KEY` (optional, for analog retrieval). |

The masking → extraction hand-off is **direct**: stage 01 writes the masked images to one
folder and stage 02 reads the *same* folder (`data/interim/masked_drawings/` locally,
`/content/masked_drawings/` on Colab). On separate runtimes, 01 also produces
`masked_drawings.zip` to upload into 02.

## The price engine in one line

It is **not** a statistical price predictor. It is an **auditable cost build-up** that assembles
a unit cost from named, individually-sourced components and shows its working:

```
unit_cost = raw_material + process + secondary_ops + overhead
          + electricity + transport + profit + other
```

Three bucket strategies (catalog lookup / nearest-reference + modification deltas / route
build-up), honest per-category uncertainty bands, a **bounded ±15% calibration correction**
(hybrid ML, segment-keyed, never the predictor), and hard refusal rules (missing rate → block,
missing tolerance → loose/tight scenarios, brand-locked → vendor RFQ). The base is **current
market rate only** — margin is a downstream overlay, never inside the engine. See
[`docs/PRICING_ENGINE_DESIGN.md`](docs/PRICING_ENGINE_DESIGN.md) and the governing spec
`docs/09_price_estimation_architecture.md`.

## Repository layout

```
.
├── notebooks/                 # the three pipeline stages (run in order)
│   ├── 01_masking.ipynb
│   ├── 02_extraction.ipynb
│   └── 03_price_estimation.ipynb
├── src/
│   ├── config.py              # shared: project paths, hand-off dir, secret-key loader
│   └── pricing/               # the cost build-up engine (importable package)
│       ├── contracts.py       # EstimateResult, CostBreakdown, ... (the data contract)
│       ├── reference_data.py  # RM matrix, ops-cost, reference prices, route-template KB
│       ├── buildup.py         # the 8-category arithmetic + modifiers (L:D, MOQ, weights)
│       ├── uncertainty.py     # per-category honest bands
│       ├── calibration.py     # bounded hybrid-ML correction (±15%, governed)
│       ├── refusals.py        # the hard "must refuse" rules
│       ├── engine.py          # orchestrator → EstimateResult
│       ├── analog.py          # Bucket-C analog retrieval (OpenAI embeddings / offline)
│       └── synthetic_data.py  # schema-shaped sample data (NOT real)
├── docs/                      # design spec + schema reference + design notes
├── data/                      # raw/ (gitignored), interim/ (hand-off), samples/
├── models/                    # saved calibration / model artifacts (gitignored)
├── experiments/               # original combined notebook + the ML-baseline regressor
├── scripts/                   # notebook generators (build/split/patch)
├── secret_keys.env.example    # template — copy to secret_keys.env and fill in
├── requirements*.txt          # top-level + per-stage
└── .gitignore
```

## Setup

```bash
# 1) secrets — copy the template and fill in your keys (NEVER commit secret_keys.env)
cp secret_keys.env.example secret_keys.env
#    then edit:  OPENAI_API_KEY=sk-...   ANTHROPIC_API_KEY=sk-ant-...

# 2) dependencies — install only what a stage needs
pip install -r requirements-pricing.txt        # to run stage 03 locally
# pip install -r requirements-extraction.txt   # for stage 02
# pip install -r requirements-masking.txt      # for stage 01 (GPU)
```

Secret keys resolve in priority order **`secret_keys.env` → OS env vars → Colab Secrets**,
via `src/config.load_keys()`. The key **names are identical everywhere**: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`.

## Quick start — the price engine

```python
import sys; sys.path.insert(0, "src")
from pricing import PriceEstimationEngine, CalibrationModel, synthetic_data as sd

ref    = sd.build_reference_data()                                   # swap for real schema tables
calib  = CalibrationModel().fit(sd.build_calibration_history())      # converted-quote history
engine = PriceEstimationEngine(ref, calibration=calib)

est = engine.estimate({
    "bucket": "special_but_standard", "part_family": "shcs", "size": "M8x45",
    "nominal_diameter_mm": 8, "nominal_length_mm": 45, "material_grade": "10.9",
    "finish": "znni", "quantity": 5000, "supplier": "s1", "region": "West",
    "ref_standard": "DIN 912", "ref_size": "M8x40", "ref_grade": "10.9",
    "ref_finish": "plain", "ref_length_mm": 40,
})
print(est.summary())          # full auditable cost sheet
print(est.unit_price)         # single float (current-market-rate, INR)
print(est.status)             # auto_emit | assistive_only | review | block
```

Run `notebooks/03_price_estimation.ipynb` for the full demo across all buckets, the refusal
cases, the reasoning trace, and instructions for wiring real `ai_rfq_sample` data.

## Data & privacy

- Real drawings, the raw dataset, models, and `secret_keys.env` are **gitignored** — only code,
  docs, and synthetic samples are committed.
- Stage 01 redacts owner branding **locally** before stage 02 ever sends a drawing to a cloud API.

## Status

POC slice. Process rates in the route-template KB are **human-anchored placeholders**, so every
to-the-print estimate is **assistive-only** until validated by a plant walkthrough. Uncertainty
bands are honest tolerances, not accuracy guarantees. Holdout validation, material alternatives,
revision-delta, live price feeds, and graduated autonomy are future work (doc Phases 5–9).
