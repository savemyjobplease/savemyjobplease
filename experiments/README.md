# experiments/

Exploratory / superseded work, kept for reference. **Not part of the production pipeline.**

| File | What it is |
|---|---|
| `original_combined_pipeline.ipynb` | The original single notebook that did masking **and** extraction together, before it was split into `notebooks/01_masking.ipynb` + `notebooks/02_extraction.ipynb`. |
| `ml_baseline_price_regressor.ipynb` | An early **statistical** price predictor (gradient boosting + a neural consolidation layer). The governing spec (`docs/09_…md`) rejects pure regression for the base price, so this is **not** the chosen engine. Its idea survives, demoted, as the bounded calibration layer in `src/pricing/calibration.py`. |

The production price engine is the transparent cost build-up in `src/pricing/`
(see `docs/PRICING_ENGINE_DESIGN.md`).
