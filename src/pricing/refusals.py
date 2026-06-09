"""Part B of design doc 09 — what the engine MUST refuse to do.

These are enforced as hard post-conditions on an EstimateResult: they set flags,
escalate status (auto_emit -> assistive_only -> review -> block), and never let a
weak input pass silently. Ordering of severity: block > review > assistive_only.
"""
from __future__ import annotations

_SEVERITY = {"auto_emit": 0, "assistive_only": 1, "review": 2, "block": 3}


def _escalate(result, status):
    if _SEVERITY[status] > _SEVERITY[result.status]:
        result.status = status
    if status in ("review", "block"):
        result.needs_review = True
    if status == "block":
        result.blocked = True


def enforce(result, line, ref, *, missing_rm_rate, missing_tolerance, fx_stale,
            over_clamp, placeholder_process, weight_inferred_equal, low_qty_unknown_tooling):
    """Apply every refusal rule. Returns the (possibly escalated) result."""

    # #10 / #1.1: missing RM rate on an RM-dominated part -> BLOCK
    if missing_rm_rate:
        result.flags.append("missing_rm_rate")
        result.trace("REFUSE: no matrix rate for grade -> surface missing, route to vendor RFQ (BLOCK)")
        _escalate(result, "block")

    # #1: never invent a missing tolerance -> two scenarios + review
    if missing_tolerance:
        result.flags.append("missing_tolerance")
        result.trace("REFUSE: tolerance not given -> emit as_if_loose / as_if_tight, await human (review)")
        _escalate(result, "review")

    # #2: FX-volatile / brand-locked / import -> never reuse history/calibration; vendor RFQ
    if getattr(line, "_flagged", False):
        result.flags.append("brand_locked_or_fx_volatile")
        result.trace("REFUSE: constraint-flagged -> no history/calibration; route to vendor RFQ on every path")
        _escalate(result, "review")

    # #5: calibration beyond +/-15% clamp -> rules-gap to review (never absorbed)
    if over_clamp:
        result.flags.append("calibration_over_clamp")
        _escalate(result, "review")

    # #6 / #8: confident point on placeholder process rates -> assistive-only, never auto-emit
    if placeholder_process:
        result.flags.append("placeholder_process_rates")
        result.trace("REFUSE: process built on placeholder rates -> assistive-only (never auto-emit)")
        _escalate(result, "assistive_only")

    # #7: never infer a missing weight equal to the present one
    if weight_inferred_equal:
        result.flags.append("weight_inferred_equal")
        _escalate(result, "review")

    # #9: never present a tooling-free low-qty unit price as final
    if low_qty_unknown_tooling:
        result.flags.append("excludes_tooling_pending_supplier_quote")
        result.trace("REFUSE: low-qty + unknown tooling -> labelled 'excludes tooling', not final")
        _escalate(result, "review")

    # #3 invariant from build-up
    if "invariant_violation_gross_lt_net" in result.flags or "implausible_yield" in result.flags:
        _escalate(result, "review")

    # cost-uncertainty threshold (doc A5 locked review threshold)
    if result.overall_rel_band > 0.30:
        result.flags.append("uncertainty_gt_30pct")
        _escalate(result, "review")

    # #11: the band is an honest tolerance, never an accuracy guarantee
    result.assumptions.append("band = acceptable tolerance, NOT a measured accuracy guarantee")
    return result
