"""PriceEstimationEngine — orchestrates the cost build-up into one EstimateResult.

Flow (design doc A3): classify -> route by bucket (A/B/C) -> build the 8 categories
-> feasibility surcharge -> modifiers (L:D, qty-tier/MOQ, geography) -> compose
per-category bands -> bounded calibration -> enforce refusals -> set status.

Bucket A (assortment)        : catalog_lookup (shallow breakdown).
Bucket B (special_but_standard): nearest reference + sum of modification deltas.
Bucket C (to_the_print)      : analog-first; else full route build-up; assistive-only
                               while process rates are placeholders.
"""
from __future__ import annotations
from typing import Optional

from .contracts import (EstimateResult, CostBreakdown, CostComponent,
                        FeasibilityResult, LeadTimeBreakdown, QuoteValidity)
from .reference_data import ReferenceData
from . import buildup as bu
from . import uncertainty as unc
from . import refusals as ref_rules


class PriceEstimationEngine:
    def __init__(self, ref: ReferenceData, calibration=None, analog_index=None):
        self.ref = ref
        self.calibration = calibration      # optional CalibrationModel
        self.analog_index = analog_index    # optional AnalogIndex (Bucket C front line)

    # ---- public API ----
    def estimate(self, line) -> EstimateResult:
        if isinstance(line, dict):
            line = bu.LineItem.from_dict(line)
        line._flagged = self._is_flagged(line)

        res = EstimateResult(line_item_id=line.id or "item",
                             bucket=line.bucket or "to_the_print")
        res.geography_basis = line.geography
        res.trace(f"bucket={res.bucket}; geography basis={line.geography} (never blended)")

        if res.bucket == "assortment":
            self._bucket_a(line, res)
        elif res.bucket == "special_but_standard":
            self._bucket_b(line, res)
        else:
            self._bucket_c(line, res)
        return res

    # ---- constraint flagging (doc Part B #2) ----
    def _is_flagged(self, line) -> bool:
        cust = (line.customer or "").strip().lower()
        if any(b in cust for b in self.ref.brand_locked_customers):
            return True
        # FX-priced RM + stale snapshot is handled as a band/validity trigger, not a hard flag
        return False

    # =====================================================================
    # Bucket A - catalog lookup
    # =====================================================================
    def _bucket_a(self, line, res: EstimateResult):
        res.route_template_used = "none_assortment"
        price = self.ref.ref_prices.catalog_price(
            line.standard_reference, line.size, line.material_grade, line.finish)
        cb = res.cost_breakdown
        if price is None:
            cb.raw_material.notes = "no catalog price for canonical reference"
            res.flags.append("missing_reference_price")
            res.trace("REFUSE: catalog price not found -> surface missing, route to review")
            res.status = "review"; res.needs_review = True
            self._finalize(line, res, missing_rm_rate=False, missing_tolerance=False,
                           placeholder_process=False)
            return
        # shallow breakdown: we do not decompose a stocked price
        cb.raw_material.value = price
        cb.raw_material.driver = f"catalog reference price ({line.standard_reference} {line.size})"
        cb.raw_material.source = "reference-price layer"
        cb.raw_material.rate_source = "reference"; cb.raw_material.confidence = "high"
        res.trace(f"catalog_lookup -> reference_price = {price:.3f}")
        self._modifiers_and_finalize(line, res, route=None,
                                     missing_rm_rate=False, placeholder_process=False)

    # =====================================================================
    # Bucket B - reference + modification deltas
    # =====================================================================
    def _bucket_b(self, line, res: EstimateResult):
        rp = self.ref.ref_prices
        anchor = rp.catalog_price(line.ref_standard or line.standard_reference,
                                  line.ref_size or line.size,
                                  line.ref_grade or line.material_grade,
                                  line.ref_finish or "plain")
        cb = res.cost_breakdown
        if anchor is None:
            res.flags.append("missing_reference_anchor")
            res.trace("no nearest reference -> fall back to route build-up (treat as C-like)")
            return self._bucket_c(line, res)
        res.route_template_used = "inherited_from_reference"
        cb.raw_material.value = anchor
        cb.raw_material.driver = f"nearest reference {line.ref_standard} {line.ref_size} {line.ref_grade}"
        cb.raw_material.source = "reference-price layer"
        cb.raw_material.rate_source = "reference"; cb.raw_material.confidence = "high"
        res.trace(f"anchor reference_price = {anchor:.3f}")

        deltas = self._modification_deltas(line, res)
        proc = cb.process
        proc.value = sum(d for d in deltas.values())
        proc.driver = "modification deltas: " + ", ".join(f"{k} {v:+.3f}" for k, v in deltas.items())
        proc.source = "RM matrix + KB deltas"; proc.rate_source = "reference"
        proc.confidence = "medium"
        self._modifiers_and_finalize(line, res, route=None,
                                     missing_rm_rate=False, placeholder_process=False)

    def _modification_deltas(self, line, res) -> dict:
        out = {}
        rp, rm = self.ref.ref_prices, self.ref.rm
        # length delta -> recompute gross weight difference x rate
        dia, _ = bu.parse_size(line.size)
        L_req = line.nominal_length_mm or bu.parse_size(line.size)[1]
        L_ref = line.ref_length_mm or bu.parse_size(line.ref_size)[1]
        hit = rm.rate(line.material_grade, line.material_condition or "any")
        if dia and L_req and L_ref and hit.found and L_req != L_ref:
            dens = self.ref.density(line.material_family)
            dg = (bu.geometry_net_weight_kg(dia, L_req, dens)
                  - bu.geometry_net_weight_kg(dia, L_ref, dens))
            out["length"] = dg * hit.value
            res.trace(f"length delta {L_ref}->{L_req}: dGross x rate = {out['length']:+.3f}")
        # finish/plating delta
        if line.finish and line.ref_finish and line.finish != line.ref_finish:
            new = rp.plating_rate(line.finish) or 0.0
            old = rp.plating_rate(line.ref_finish) or 0.0
            out["finish"] = new - old
            res.trace(f"finish delta {line.ref_finish}->{line.finish} = {out['finish']:+.3f}")
        # grade delta -> material + heat-treat
        if line.material_grade and line.ref_grade and line.material_grade != line.ref_grade:
            ht = rp.heat_treat_rate(line.material_grade) - rp.heat_treat_rate(line.ref_grade)
            out["grade_heat_treat"] = ht
            res.trace(f"grade delta {line.ref_grade}->{line.material_grade} heat-treat = {ht:+.3f}")
        # drive/head delta
        if line.drive_head:
            out["drive_head"] = 0.30
            res.trace("drive/head change -> +0.30 forming/CNC")
        return out

    # =====================================================================
    # Bucket C - analog-first, else route build-up
    # =====================================================================
    def _bucket_c(self, line, res: EstimateResult):
        # 1) analog-first (the value-delivery front line)
        if self.analog_index is not None:
            analogs = self.analog_index.nearest(line, k=3)
            if analogs:
                res.trace("analog-first: presenting closest historical quotes (assistive)")
                res.assumptions.append(
                    "Bucket-C analogs available; build-up below grounds them: "
                    + "; ".join(f"{a['label']}={a['price']:.2f}" for a in analogs))
        # 2) full route build-up from physics
        route = self.ref.routes.get(line.part_family)
        res.route_template_used = route.process if route else "none"
        net, gross, dia, length, wfg = bu.resolve_weights(line, self.ref, route, res)
        res.net_weight_kg, res.gross_weight_kg = net, gross

        cb = res.cost_breakdown
        rm_comp, hit = bu.raw_material_component(line, self.ref, gross, wfg)
        cb.raw_material = rm_comp
        missing_rm = not hit.found

        cb.process = bu.process_component(line, self.ref, route, gross, dia, length)
        ld = bu.apply_ld_trigger(cb.process, route, dia, length, res)
        res.ld_trigger = ld

        # feasibility -> secondary ops
        feas, sec_cost = self._feasibility(line, route, res)
        res.feasibility = feas
        cb.secondary_ops = CostComponent(
            "secondary_ops", value=sec_cost,
            driver="tolerance-gap secondary machining" if sec_cost else "none (route holds tolerance)",
            source="feasibility engine", rate_source="human_anchored" if sec_cost else "n/a",
            confidence="low" if sec_cost else "n/a")

        ov, el, tr, pr = bu.overhead_electricity_transport_profit(
            cb.raw_material.value, cb.process.value, cb.secondary_ops.value, self.ref)
        cb.overhead, cb.electricity, cb.transport, cb.profit = ov, el, tr, pr
        cb.other = bu.other_component(line, line.quantity)

        # doc A1.2: Bucket-C on un-validated (placeholder/human-anchored) process
        # rates is assistive-only until the plant walkthrough validates them.
        placeholder_proc = cb.process.rate_source in ("placeholder", "human_anchored")
        self._modifiers_and_finalize(line, res, route=route,
                                     missing_rm_rate=missing_rm,
                                     placeholder_process=placeholder_proc,
                                     rate_hit=hit, weight_from_geometry=wfg)

    def _feasibility(self, line, route, res) -> tuple:
        native = route.native_tolerance_um if route else 100.0
        if line.requested_tolerance_um is None:
            # the ~70% case: bimodal -> two explicit scenarios, never assume
            loose = 0.0
            tight = 0.80 * (route.dia_multiplier(line.nominal_diameter_mm) if route else 1.0)
            feas = FeasibilityResult(verdict="needs_SME_review",
                                     reason_qualifier="pending_tolerance",
                                     requested_tolerance_um=None, route_native_um=native,
                                     scenarios={"as_if_loose": loose, "as_if_tight": tight},
                                     note="tolerance missing -> loose/tight scenarios")
            res.trace("feasibility: tolerance MISSING -> as_if_loose=0 / as_if_tight="
                      f"{tight:.2f}; customer field stays 'pending tolerance'")
            return feas, 0.0   # base build-up uses 0; scenarios carried separately
        gap = native - line.requested_tolerance_um
        if gap <= 0:
            return FeasibilityResult("feasible", None, line.requested_tolerance_um, native), 0.0
        # viable secondary chain assumed available -> add grinding/CNC pass
        sec = 0.80 * (route.dia_multiplier(line.nominal_diameter_mm) if route else 1.0)
        res.trace(f"feasibility: route native {native}um vs requested "
                  f"{line.requested_tolerance_um}um -> +secondary {sec:.2f}")
        return FeasibilityResult("needs_secondary_op", None,
                                 line.requested_tolerance_um, native, secondary_ops_cost=sec), sec

    # =====================================================================
    # shared finalize: modifiers + bands + calibration + refusals + status
    # =====================================================================
    def _modifiers_and_finalize(self, line, res, route, *, missing_rm_rate,
                                placeholder_process, rate_hit=None, weight_from_geometry=False):
        # quantity tier + MOQ rule (doc A4.3)
        tier, _ = bu.quantity_tier(line.quantity)
        res.sourcing_tier = tier
        if route is not None:
            moq = bu.route_moq(route)
            res.moq = moq
            if line.quantity and moq > line.quantity:
                base_unit = res.cost_breakdown.total()
                # tooling/setup amortization changes with the denominator
                tooling = (line.tooling_cost or 0.0)
                per_req = base_unit + (tooling / max(1, line.quantity) if tooling else 0.0)
                per_moq = base_unit + (tooling / max(1, moq) if tooling else 0.0)
                res.moq_gap = {"per_requested": per_req, "per_moq": per_moq}
                res.flags.append("moq_gt_requested")
                res.trace(f"MOQ {moq} > requested {line.quantity} -> price at MOQ, show both")
            # lead-time triplet (placeholder spans by process)
            mfg = {"forging": 18, "cold_heading": 12, "machining": 14,
                   "stamping": 10}.get(route.process, 12)
            res.lead_time = LeadTimeBreakdown(manufacturing_days=mfg, rm_days=7, transit_days=3)
        else:
            res.lead_time = LeadTimeBreakdown(manufacturing_days=7, rm_days=3, transit_days=2)

        self._finalize(line, res, missing_rm_rate=missing_rm_rate,
                       missing_tolerance=(line.requested_tolerance_um is None
                                          and res.bucket == "to_the_print"),
                       placeholder_process=placeholder_process,
                       rate_hit=rate_hit, weight_from_geometry=weight_from_geometry)

    def _finalize(self, line, res, *, missing_rm_rate, missing_tolerance,
                  placeholder_process, rate_hit=None, weight_from_geometry=False):
        # FX staleness (doc A4.6): foreign-priced RM + snapshot > 7 days
        fx_stale = (self.ref.fx_as_of_days > 7) and (line.geography.lower() != "india")

        # per-category honest bands
        per_cat, overall = unc.assign_category_bands(
            res, weight_from_geometry=weight_from_geometry, rate_hit=rate_hit,
            fx_stale=fx_stale, substituted_grade=False, missing_tolerance=missing_tolerance)
        res.per_category_confidence = per_cat
        res.overall_rel_band = overall
        res.overall_confidence = unc.confidence_scalar(per_cat)

        # base (rules) price
        res.base_unit_price = res.cost_breakdown.total()
        res.unit_price = res.base_unit_price

        # bounded calibration (HYBRID ML), governed
        if self.calibration is not None and res.base_unit_price > 0:
            cal, extras = self.calibration.factor(line, res.base_unit_price)
            res.calibration = cal
            if cal.applied:
                res.unit_price = res.base_unit_price * cal.factor
                res.trace(f"calibration: rules {res.base_unit_price:.3f} x {cal.factor:.3f} "
                          f"[{cal.segment}, n={cal.support_n}, {cal.backend}] = {res.unit_price:.3f}")
            else:
                res.trace(f"calibration: {cal.note}")
            if extras.get("widen_band"):
                res.overall_rel_band += 0.05
            over_clamp = extras.get("over_clamp", False)
        else:
            over_clamp = False

        # headline band around the (calibrated) unit price
        res.range_low = res.unit_price * (1 - res.overall_rel_band)
        res.range_high = res.unit_price * (1 + res.overall_rel_band)

        # validity (doc A4.6)
        res.validity = QuoteValidity(window_days=3 if fx_stale else 14,
                                     freshness_state="stale" if fx_stale else "fresh")
        if getattr(line, "_flagged", False):
            res.validity.freshness_state = "unreliable"

        # low-qty + unknown tooling on C
        low_qty_unknown_tooling = (res.bucket == "to_the_print"
                                   and (line.quantity or 0) < 2000
                                   and not line.tooling_cost)

        # margin discipline: displayed_price is base + overlay, set at render layer only
        res.displayed_price = None   # engine never bakes markup; rendering layer adds overlay

        # default status before refusals
        res.status = "auto_emit"; res.needs_review = False
        ref_rules.enforce(res, line, self.ref,
                          missing_rm_rate=missing_rm_rate,
                          missing_tolerance=missing_tolerance,
                          fx_stale=fx_stale, over_clamp=over_clamp,
                          placeholder_process=placeholder_process,
                          weight_inferred_equal=False,
                          low_qty_unknown_tooling=low_qty_unknown_tooling)
        return res
