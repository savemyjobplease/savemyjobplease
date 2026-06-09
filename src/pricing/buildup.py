"""The transparent cost build-up (design doc A1) and deterministic modifiers (A4).

Pure, auditable arithmetic: every term records its driver, source, and reliability.
Nothing here invents a price — missing inputs are surfaced upward as flags.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional
import math, re

from .contracts import CostComponent, CostBreakdown
from .reference_data import ReferenceData, RouteTemplate, YIELD_BANDS


# ---------------------------------------------------------------------------
@dataclass
class LineItem:
    id: str = ""
    bucket: str = ""
    part_family: str = "custom_special"
    standard_reference: Optional[str] = None
    size: Optional[str] = None
    nominal_diameter_mm: Optional[float] = None
    nominal_length_mm: Optional[float] = None
    material_family: Optional[str] = None
    material_grade: Optional[str] = None
    material_condition: Optional[str] = None
    finish: Optional[str] = None
    requested_tolerance_um: Optional[float] = None      # None => missing (the ~70% case)
    quantity: Optional[int] = None
    net_weight_kg: Optional[float] = None
    gross_weight_kg: Optional[float] = None
    customer: Optional[str] = None
    supplier: Optional[str] = None
    region: Optional[str] = None
    geography: str = "india"
    drive_head: Optional[str] = None
    tooling_cost: Optional[float] = None                # known tool cost (else unknown)
    tooling_amortize_qty: Optional[int] = None
    # Bucket-B reference anchor (the nearest catalog variant)
    ref_standard: Optional[str] = None
    ref_size: Optional[str] = None
    ref_grade: Optional[str] = None
    ref_finish: Optional[str] = None
    ref_length_mm: Optional[float] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "LineItem":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
def parse_size(size):
    """'M8x40' / 'M10x1.5x45' -> (diameter_mm, length_mm)."""
    if not size:
        return None, None
    nums = [float(x) for x in re.findall(r"[0-9]+\.?[0-9]*", str(size))]
    if not nums:
        return None, None
    dia = nums[0]
    length = nums[-1] if len(nums) >= 2 else None
    return dia, length


def geometry_net_weight_kg(dia_mm, length_mm, density_gcm3):
    """Cylinder approximation for a finished fastener (rough, flagged when used)."""
    if not dia_mm or not length_mm:
        return None
    vol_cm3 = math.pi * (dia_mm / 2.0) ** 2 * length_mm / 1000.0   # mm^3 -> cm^3
    return vol_cm3 * density_gcm3 / 1000.0                          # g -> kg


# ---- weight resolution + invariants (doc A1.1) ----------------------------
def resolve_weights(li: LineItem, ref: ReferenceData, route: Optional[RouteTemplate], result):
    dia = li.nominal_diameter_mm
    length = li.nominal_length_mm
    if dia is None or length is None:
        pd_, pl_ = parse_size(li.size)
        dia = dia if dia is not None else pd_
        length = length if length is not None else pl_

    net = li.net_weight_kg
    weight_from_geometry = False
    if net is None:
        net = geometry_net_weight_kg(dia, length, ref.density(li.material_family))
        if net is not None:
            weight_from_geometry = True
            result.assumptions.append("net_weight computed from geometry x density (+band)")

    yf = route.yield_factor if route else 1.2
    gross = li.gross_weight_kg
    if gross is None and net is not None:
        gross = net * yf
        result.assumptions.append(f"gross_weight = net x yield_factor({yf:.2f}) [{route.process if route else 'default'}]")

    # invariants
    if gross is not None and net is not None:
        if gross < net - 1e-12:
            result.flags.append("invariant_violation_gross_lt_net")
            result.trace("INVARIANT: gross < net -> route to review, widen RM band")
        eff_yield = gross / net if net else None
        band = YIELD_BANDS.get(route.process if route else "default", YIELD_BANDS["default"])
        if eff_yield is not None and not (band[0] <= eff_yield <= band[1] * 1.05):
            result.flags.append("implausible_yield")
            result.trace(f"yield {eff_yield:.2f} outside plausible {band} -> widen RM band")
    return net, gross, dia, length, weight_from_geometry


# ---- the eight categories --------------------------------------------------
def raw_material_component(li, ref, gross, weight_from_geometry):
    hit = ref.rm.rate(li.material_grade, li.material_condition or "any")
    comp = CostComponent("raw_material", driver="gross_weight x current grade rate",
                         source="raw-material matrix", rate_source="reference")
    if not hit.found:
        comp.value = 0.0
        comp.confidence = "n/a"
        comp.notes = hit.note
        return comp, hit
    comp.value = (gross or 0.0) * hit.value
    comp.driver = (f"{(gross or 0):.4f} kg x {hit.value:.1f} INR/kg "
                   f"[{li.material_grade}, as_of {hit.as_of}, {hit.volatility_class}]")
    comp.confidence = "high"
    if weight_from_geometry:
        comp.confidence = "medium"
    comp.notes = hit.note
    return comp, hit


def _step_cost(step, gross, dia_mult, length_mm):
    if step.basis == "per_piece":
        return step.rate * dia_mult
    if step.basis == "per_kg":
        return step.rate * (gross or 0.0) * dia_mult
    if step.basis == "per_minute":
        minutes = max(0.3, ((dia_mult * (length_mm or 20)) / 80.0))   # placeholder cycle time
        return step.rate * minutes
    return step.rate


def process_component(li, ref, route, gross, dia, length):
    comp = CostComponent("process", driver="sum of route steps (route x machine x dia band)",
                         source="route-template KB")
    if route is None:
        comp.notes = "no route template for family -> review"
        comp.confidence = "low"
        return comp
    dia_mult = route.dia_multiplier(dia)
    total, parts = 0.0, []
    for s in route.steps:
        c = _step_cost(s, gross, dia_mult, length)
        total += c
        parts.append(f"{s.name} {c:.2f}")
    comp.value = total
    comp.driver = f"{route.process}: " + " + ".join(parts)
    comp.rate_source = route.worst_rate_source()
    comp.confidence = "low" if comp.rate_source == "placeholder" else \
                      ("medium" if comp.rate_source == "human_anchored" else "high")
    comp.notes = f"dia_band_mult={dia_mult}"
    return comp


def overhead_electricity_transport_profit(rm, process, secondary, ref):
    base = rm + process + secondary
    ov = CostComponent("overhead", value=ref.ops.overhead_pct * base,
                       driver=f"{ref.ops.overhead_pct*100:.0f}% of (RM+process+secondary)",
                       source="ops-cost table", rate_source="reference", confidence="medium")
    el = CostComponent("electricity", value=ref.ops.electricity_inr_per_pc,
                       driver="region operating cost x consumption", source="ops-cost feed",
                       rate_source="reference", confidence="medium")
    tr = CostComponent("transport", value=ref.ops.transport_inr_per_pc,
                       driver="supplier geography -> destination freight", source="ops-cost feed",
                       rate_source="reference", confidence="medium")
    subtotal = base + ov.value + el.value + tr.value
    pr = CostComponent("profit", value=ref.ops.profit_pct * subtotal,
                       driver=f"{ref.ops.profit_pct*100:.0f}% supplier margin INSIDE market rate "
                              f"(NOT our markup)", source="template default",
                       rate_source="reference", confidence="medium")
    return ov, el, tr, pr


def other_component(li, qty):
    """Carries the named tooling_amortized sub-line (doc A1.4)."""
    comp = CostComponent("other", driver="tooling amortization / test / packing specials",
                         source="catch-all", rate_source="reference", confidence="medium")
    if li.tooling_cost and li.tooling_amortize_qty:
        per_pc = li.tooling_cost / max(1, li.tooling_amortize_qty)
        comp.value = per_pc
        comp.driver = f"tooling {li.tooling_cost} amortized / {li.tooling_amortize_qty} pcs"
        comp.subline = {"tooling_amortized": {"tool_cost": li.tooling_cost,
                        "amortize_qty": li.tooling_amortize_qty, "per_piece": per_pc}}
    else:
        comp.notes = "tooling unknown"
        comp.subline = {"tooling_amortized": None}
    return comp


# ---- modifiers (doc A4) ----------------------------------------------------
def apply_ld_trigger(process_comp, route, dia, length, result):
    if route is None or not dia or not length:
        return False
    ld = length / dia
    if ld >= route.ld_threshold:
        bend = route.ld_bend_cost * route.dia_multiplier(dia)
        process_comp.value += bend
        process_comp.driver += f" + L:D bend {bend:.2f} (L/D={ld:.1f})"
        result.trace(f"L:D trigger: L/D={ld:.1f} >= {route.ld_threshold} -> +{bend:.2f}")
        return True
    return False


def quantity_tier(qty):
    if qty is None:
        return "mixed", None
    if qty < 2000:
        return "distributor", None
    if qty >= 20000:
        return "direct_manufacturer", None
    return "mixed", None


def route_moq(route):
    table = {"forging": 10000, "cold_heading": 5000, "machining": 500,
             "stamping": 5000, "default": 1000}
    return table.get(route.process if route else "default", 1000)
