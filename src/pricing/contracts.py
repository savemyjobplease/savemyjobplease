"""Data contracts for the price-estimation cost engine.

Mirrors the `EstimateResult` shape from design doc 09: an additive, 8-category
cost build-up where every term carries its driver, source, and reliability, plus
an honest per-category uncertainty band, validity, feasibility, and a full
reasoning trace. The contract is append-only/frozen — reliability detail lives in
fields here, never as silent new top-level numbers.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---- controlled vocabularies -------------------------------------------------
BUCKETS = ("assortment", "special_but_standard", "to_the_print")
RATE_SOURCES = ("measured", "human_anchored", "placeholder", "reference", "n/a")
FEAS_VERDICTS = ("feasible", "needs_secondary_op", "needs_SME_review")
FEAS_QUALIFIERS = ("pending_tolerance", "no_viable_secondary_chain", None)
STATUSES = ("auto_emit", "assistive_only", "review", "block")

# the eight named cost categories, in build-up order
COST_CATEGORIES = ("raw_material", "process", "secondary_ops", "overhead",
                   "electricity", "transport", "profit", "other")


@dataclass
class CostComponent:
    category: str
    value: float = 0.0
    driver: str = ""
    source: str = ""
    rate_source: str = "n/a"          # measured | human_anchored | placeholder | reference | n/a
    confidence: str = "n/a"           # high | medium | low | n/a
    rel_band: float = 0.0             # fractional half-width of this term's band
    notes: str = ""
    subline: Optional[dict] = None    # e.g. {"tooling_amortized": {...}}

    @property
    def band_low(self) -> float:
        return self.value * (1.0 - self.rel_band)

    @property
    def band_high(self) -> float:
        return self.value * (1.0 + self.rel_band)


@dataclass
class CostBreakdown:
    raw_material: CostComponent = field(default_factory=lambda: CostComponent("raw_material"))
    process: CostComponent = field(default_factory=lambda: CostComponent("process"))
    secondary_ops: CostComponent = field(default_factory=lambda: CostComponent("secondary_ops"))
    overhead: CostComponent = field(default_factory=lambda: CostComponent("overhead"))
    electricity: CostComponent = field(default_factory=lambda: CostComponent("electricity"))
    transport: CostComponent = field(default_factory=lambda: CostComponent("transport"))
    profit: CostComponent = field(default_factory=lambda: CostComponent("profit"))
    other: CostComponent = field(default_factory=lambda: CostComponent("other"))

    def components(self):
        return [getattr(self, c) for c in COST_CATEGORIES]

    def total(self) -> float:
        return sum(c.value for c in self.components())

    def band_low(self) -> float:
        return sum(c.band_low for c in self.components())

    def band_high(self) -> float:
        return sum(c.band_high for c in self.components())


@dataclass
class FeasibilityResult:
    verdict: str = "feasible"                  # FEAS_VERDICTS
    reason_qualifier: Optional[str] = None     # FEAS_QUALIFIERS
    requested_tolerance_um: Optional[float] = None
    route_native_um: Optional[float] = None
    secondary_ops_cost: float = 0.0
    scenarios: Optional[dict] = None           # {"as_if_loose": cost, "as_if_tight": cost}
    note: str = ""


@dataclass
class LeadTimeBreakdown:
    manufacturing_days: float = 0.0
    rm_days: float = 0.0
    transit_days: float = 0.0

    @property
    def total_days(self) -> float:
        return self.manufacturing_days + self.rm_days + self.transit_days


@dataclass
class QuoteValidity:
    window_days: int = 14
    refresh_triggers: list = field(default_factory=lambda: ["time", "fx_move", "commodity_move"])
    freshness_state: str = "fresh"             # fresh | stale | unreliable
    note: str = ""


@dataclass
class Calibration:
    applied: bool = False
    factor: float = 1.0                        # clamped to [0.85, 1.15]
    segment: str = ""                          # e.g. "shcs|supplierX|West"
    support_n: int = 0
    backend: str = "none"                      # ml | grid | none
    note: str = ""


@dataclass
class EstimateResult:
    line_item_id: str = ""
    bucket: str = ""
    route_template_used: str = "none"

    # prices: base = current market rate; displayed adds any business overlay (never in base)
    base_unit_price: float = 0.0               # rules build-up (pre-calibration)
    unit_price: float = 0.0                    # after bounded calibration (still market rate)
    displayed_price: Optional[float] = None    # base + business overlay, set at render layer only
    range_low: float = 0.0                     # sum of per-category bands
    range_high: float = 0.0

    cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    gross_weight_kg: Optional[float] = None
    net_weight_kg: Optional[float] = None

    moq: Optional[int] = None
    moq_gap: Optional[dict] = None             # {"per_requested": x, "per_moq": y} when MOQ > qty
    sourcing_tier: str = ""                    # distributor | mixed | direct_manufacturer
    geography_basis: str = ""                  # never a blended cross-country number
    ld_trigger: bool = False

    lead_time: LeadTimeBreakdown = field(default_factory=LeadTimeBreakdown)
    feasibility: FeasibilityResult = field(default_factory=FeasibilityResult)
    validity: QuoteValidity = field(default_factory=QuoteValidity)
    calibration: Calibration = field(default_factory=Calibration)

    per_category_confidence: dict = field(default_factory=dict)
    overall_confidence: float = 0.0
    overall_rel_band: float = 0.0

    flags: list = field(default_factory=list)          # constraint/quality flags
    assumptions: list = field(default_factory=list)
    reasoning_trace: list = field(default_factory=list)
    status: str = "review"                              # STATUSES
    needs_review: bool = True
    blocked: bool = False

    def trace(self, msg: str):
        self.reasoning_trace.append(msg)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"[{self.line_item_id}] bucket={self.bucket} route={self.route_template_used} "
            f"status={self.status.upper()}",
            f"  unit_price = {self.unit_price:,.3f} INR   band [{self.range_low:,.3f}, "
            f"{self.range_high:,.3f}]  (+/-{self.overall_rel_band*100:.0f}%)",
        ]
        if self.base_unit_price and abs(self.base_unit_price - self.unit_price) > 1e-9:
            lines.append(f"  base(rules) = {self.base_unit_price:,.3f}  x calib "
                         f"{self.calibration.factor:.3f} [{self.calibration.segment}, "
                         f"n={self.calibration.support_n}]")
        lines.append("  cost build-up:")
        for c in self.cost_breakdown.components():
            if c.value or c.notes:
                tag = f" <{c.rate_source}/{c.confidence}>" if c.rate_source != "n/a" else ""
                lines.append(f"    {c.category:13s} {c.value:8.3f}  {c.driver}{tag}")
        if self.moq_gap:
            lines.append(f"  MOQ gap: per_requested={self.moq_gap['per_requested']:.3f} "
                         f"per_moq={self.moq_gap['per_moq']:.3f} (moq={self.moq})")
        lines.append(f"  lead-time = {self.lead_time.total_days:.0f} d "
                     f"(mfg {self.lead_time.manufacturing_days:.0f} + rm "
                     f"{self.lead_time.rm_days:.0f} + transit {self.lead_time.transit_days:.0f})")
        lines.append(f"  feasibility = {self.feasibility.verdict}"
                     + (f" ({self.feasibility.reason_qualifier})"
                        if self.feasibility.reason_qualifier else ""))
        lines.append(f"  validity = {self.validity.window_days} d ({self.validity.freshness_state})")
        if self.flags:
            lines.append(f"  FLAGS: {', '.join(self.flags)}")
        return "\n".join(lines)
