"""Reference / ground-truth data layer (design doc Phase 1 + Phase 0 KB).

Holds the single source of truth for every price input:
  * RawMaterialMatrix  - current-month grade x condition x supplier_band rate, with
    as_of + per-grade volatility_class. Missing keys are surfaced, never invented.
  * OpsCost            - overhead %, electricity, transport, profit defaults.
  * ReferencePrices    - catalog/stocked SKU prices + plating/heat-treat deltas.
  * RouteTemplateKB    - per part-family route (ordered steps, machine class,
    diameter-band multipliers, native tolerance, yield). Process rates are
    human-anchored placeholders by construction (no tabulated source exists).

All getters degrade gracefully and report whether a value was FOUND vs MISSING.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

MATERIAL_DENSITY_GCM3 = {
    "carbon_steel": 7.85, "alloy_steel": 7.85, "stainless_steel": 8.0,
    "brass": 8.5, "aluminium": 2.7, "titanium": 4.5, "default": 7.85,
}

# per-route plausible yield bands (gross/net) — invariant check, doc A1.1
YIELD_BANDS = {
    "forging": (1.05, 1.20), "cold_heading": (1.05, 1.20),
    "machining": (1.30, 2.00), "stamping": (1.10, 1.40), "default": (1.05, 2.00),
}


@dataclass
class RateHit:
    value: Optional[float]
    found: bool
    as_of: Optional[str] = None
    volatility_class: str = "monthly"   # stable | monthly | weekly
    note: str = ""


class RawMaterialMatrix:
    """Keyed lookup (grade, condition, month, supplier_band) -> INR/kg. Never guesses."""

    def __init__(self):
        self._rows = {}          # (grade, condition, month, band) -> dict
        self._by_grade = {}      # grade -> list of dicts (for nearest-month fallback)

    @staticmethod
    def _norm(s):
        return (str(s).strip().lower() if s is not None else "")

    def add(self, grade, rate, *, condition="any", month="current",
            supplier_band="any", as_of=None, volatility_class="monthly"):
        rec = dict(grade=self._norm(grade), condition=self._norm(condition),
                   month=self._norm(month), supplier_band=self._norm(supplier_band),
                   rate=float(rate), as_of=as_of, volatility_class=volatility_class)
        self._rows[(rec["grade"], rec["condition"], rec["month"], rec["supplier_band"])] = rec
        self._by_grade.setdefault(rec["grade"], []).append(rec)

    def rate(self, grade, condition="any", month="current", supplier_band="any") -> RateHit:
        g, c, m, b = (self._norm(grade), self._norm(condition),
                      self._norm(month), self._norm(supplier_band))
        # exact -> relax band -> relax condition -> any month for grade
        for key in [(g, c, m, b), (g, c, m, "any"), (g, "any", m, "any"),
                    (g, c, "current", "any"), (g, "any", "current", "any")]:
            r = self._rows.get(key)
            if r:
                return RateHit(r["rate"], True, r["as_of"], r["volatility_class"])
        if g in self._by_grade and self._by_grade[g]:
            r = self._by_grade[g][-1]
            return RateHit(r["rate"], True, r["as_of"], r["volatility_class"],
                           note="nearest-available month/condition")
        return RateHit(None, False, note=f"missing_rm_rate for grade={grade!r}")

    # ---- build from a schema-shaped ref_raw_material_price DataFrame ----
    @classmethod
    def from_schema(cls, df):
        m = cls()
        if df is None or len(df) == 0:
            return m
        import pandas as pd
        for _, row in df.iterrows():
            rate = pd.to_numeric(row.get("price"), errors="coerce")
            if rate != rate:   # NaN
                continue
            m.add(row.get("grade") or row.get("material"), rate,
                  as_of=str(row.get("as_of")), volatility_class="monthly")
        return m


@dataclass
class OpsCost:
    overhead_pct: float = 0.12          # of (material+process+secondary)
    electricity_inr_per_pc: float = 0.18
    transport_inr_per_pc: float = 0.15
    profit_pct: float = 0.08            # supplier's margin inside market rate (NOT our markup)


class ReferencePrices:
    """Catalog/stocked SKU prices + finish/heat-treat deltas (Bucket A/B)."""

    def __init__(self):
        self.catalog = {}                                  # (standard,size,grade,finish)->price
        self.plating = {"plain": 0.20, "zinc": 0.30, "geomet": 0.45,
                        "znni": 0.55, "passivation": 0.15}
        self.heat_treat = {"4.6": 0.0, "8.8": 0.0, "10.9": 0.30, "12.9": 0.45,
                           "a2-70": 0.10}

    @staticmethod
    def _k(standard, size, grade, finish):
        return tuple(str(x).strip().lower() for x in (standard, size, grade, finish))

    def add_catalog(self, standard, size, grade, finish, price):
        self.catalog[self._k(standard, size, grade, finish)] = float(price)

    def catalog_price(self, standard, size, grade, finish):
        return self.catalog.get(self._k(standard, size, grade, finish))

    def plating_rate(self, finish):
        return self.plating.get(str(finish).strip().lower(), None)

    def heat_treat_rate(self, grade):
        return self.heat_treat.get(str(grade).strip().lower(), 0.0)


@dataclass
class RouteStep:
    name: str
    basis: str            # per_piece | per_kg | per_minute
    rate: float
    machine_class: str = ""
    rate_source: str = "placeholder"     # measured | human_anchored | placeholder
    rate_confidence: str = "low"


@dataclass
class RouteTemplate:
    family: str
    process: str                          # forging | machining | cold_heading | stamping ...
    steps: list = field(default_factory=list)
    native_tolerance_um: float = 100.0
    yield_factor: float = 1.20
    ld_threshold: float = 10.0
    ld_bend_cost: float = 0.50
    diameter_bands: dict = field(default_factory=lambda: {  # multiplier by nominal dia (mm)
        (0, 6): 0.85, (6, 12): 1.0, (12, 20): 1.25, (20, 999): 1.6})

    def dia_multiplier(self, dia_mm):
        if dia_mm is None:
            return 1.0
        for (lo, hi), mult in self.diameter_bands.items():
            if lo <= dia_mm < hi:
                return mult
        return 1.0

    def worst_rate_source(self):
        order = {"measured": 0, "human_anchored": 1, "placeholder": 2}
        return max((s.rate_source for s in self.steps),
                   key=lambda r: order.get(r, 2), default="placeholder")


class RouteTemplateKB:
    """Part-family -> route template. Built-in placeholder rates (doc A1.2)."""

    def __init__(self):
        self.templates = {}
        self._seed_placeholders()

    def add(self, tpl: RouteTemplate):
        self.templates[tpl.family] = tpl

    def get(self, family) -> Optional[RouteTemplate]:
        return self.templates.get(str(family).strip().lower())

    def _seed_placeholders(self):
        ph = lambda n, b, r, mc: RouteStep(n, b, r, mc, "human_anchored", "low")
        # forged bolts / studs / pins
        forging = [ph("forging", "per_piece", 1.20, "press"),
                   ph("rolling", "per_piece", 0.40, "thread_roller"),
                   ph("heat_treat", "per_piece", 0.80, "furnace"),
                   ph("plating", "per_piece", 0.50, "plating_line"),
                   ph("sorting", "per_piece", 0.10, "sorter"),
                   ph("packing", "per_piece", 0.10, "manual")]
        machining = [ph("bar_cut", "per_piece", 0.30, "saw"),
                     ph("cnc_turning", "per_minute", 6.0, "cnc_lathe"),
                     ph("cnc_milling", "per_minute", 7.0, "cnc_mill"),
                     ph("heat_treat", "per_piece", 0.80, "furnace"),
                     ph("plating", "per_piece", 0.50, "plating_line"),
                     ph("sorting", "per_piece", 0.10, "sorter"),
                     ph("packing", "per_piece", 0.10, "manual")]
        cold_head = [ph("cold_heading", "per_piece", 0.60, "header"),
                     ph("rolling", "per_piece", 0.30, "thread_roller"),
                     ph("plating", "per_piece", 0.30, "plating_line"),
                     ph("sorting", "per_piece", 0.08, "sorter"),
                     ph("packing", "per_piece", 0.08, "manual")]
        for fam, proc, steps, tol, y in [
            ("hex_bolt", "forging", forging, 100.0, 1.15),
            ("shcs", "cold_heading", cold_head, 80.0, 1.15),
            ("stud", "forging", forging, 100.0, 1.15),
            ("hex_nut", "cold_heading", cold_head, 100.0, 1.12),
            ("flanged_nut", "cold_heading", cold_head, 100.0, 1.12),
            ("washer", "stamping", cold_head, 120.0, 1.30),
            ("pin", "machining", machining, 50.0, 1.50),
            ("bush", "machining", machining, 50.0, 1.50),
            ("custom_special", "machining", machining, 50.0, 1.50),
        ]:
            self.add(RouteTemplate(fam, proc, list(steps),
                                   native_tolerance_um=tol, yield_factor=y))


@dataclass
class ReferenceData:
    """Container bundling all reference layers, passed into the engine."""
    rm: RawMaterialMatrix = field(default_factory=RawMaterialMatrix)
    ops: OpsCost = field(default_factory=OpsCost)
    ref_prices: ReferencePrices = field(default_factory=ReferencePrices)
    routes: RouteTemplateKB = field(default_factory=RouteTemplateKB)
    fx_usdinr: float = 85.5
    fx_as_of_days: int = 1               # age of the FX snapshot, in days
    # items whose history must never be reused/calibrated (doc Part B #2)
    brand_locked_customers: tuple = ("cvl", "ejot", "lear")

    def density(self, material_family) -> float:
        return MATERIAL_DENSITY_GCM3.get(
            str(material_family).strip().lower(), MATERIAL_DENSITY_GCM3["default"])
