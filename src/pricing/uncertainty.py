"""Per-category honest uncertainty (design doc A5).

The headline band is COMPOSED from per-category bands, never assumed — a known RM
rate is not blurred by an unknown process rate. Widening triggers are logged;
the floor is +/-10% (no false precision).
"""
from __future__ import annotations

BASE_BAND = {"assortment": 0.10, "special_but_standard": 0.15, "to_the_print": 0.25}
BAND_FLOOR = 0.10

# per-category starting half-widths by confidence label
CONF_BAND = {"high": 0.06, "medium": 0.12, "low": 0.30, "n/a": 0.30}

REVIEW = {
    "classification_conf_min": 0.60,
    "extraction_completeness_min": 0.50,
    "cost_uncertainty_max": 0.30,
}


def assign_category_bands(result, *, weight_from_geometry, rate_hit, fx_stale,
                          substituted_grade, missing_tolerance):
    """Set rel_band on each cost component and return (per_cat_conf, overall_band)."""
    cb = result.cost_breakdown
    bucket_base = BASE_BAND.get(result.bucket, 0.25)

    for comp in cb.components():
        if comp.value == 0 and comp.rate_source == "n/a":
            comp.rel_band = 0.0
            continue
        band = max(bucket_base, CONF_BAND.get(comp.confidence, 0.30))
        # category-specific widening
        if comp.category == "raw_material":
            if rate_hit is not None and rate_hit.found and rate_hit.note:
                band += 0.05; result.trace("RM: stale/nearest rate -> +5% + refresh flag")
            if weight_from_geometry:
                band += 0.05; result.trace("RM: weight from geometry -> +5%")
            if substituted_grade:
                band += 0.05; result.trace("RM: substituted grade -> +5%")
            if fx_stale:
                band += 0.10; result.trace("RM: FX-priced + stale snapshot -> +10%, validity 14->3")
        if comp.category == "process" and comp.rate_source == "placeholder":
            band = max(band, 0.30); result.trace("Process: placeholder rates dominate band")
        if comp.category == "secondary_ops" and missing_tolerance and comp.value:
            band += 0.05
        comp.rel_band = band

    per_cat_conf = {c.category: c.confidence for c in cb.components() if c.value or c.notes}

    total = cb.total()
    if total <= 0:
        return per_cat_conf, bucket_base
    # composed band = sum of per-category absolute bands / total
    half = sum(c.value * c.rel_band for c in cb.components())
    overall = max(BAND_FLOOR, half / total)
    return per_cat_conf, overall


def confidence_scalar(per_cat_conf):
    """Coarse overall scalar for the auto-handle gate (weakest link aware)."""
    if not per_cat_conf:
        return 0.0
    score = {"high": 0.95, "medium": 0.7, "low": 0.4, "n/a": 0.3}
    vals = [score.get(v, 0.4) for v in per_cat_conf.values()]
    return round(0.5 * min(vals) + 0.5 * (sum(vals) / len(vals)), 3)
