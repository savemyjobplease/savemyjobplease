"""Transparent, auditable cost build-up engine for fastener price estimation.

Implements design doc 09 (POC slice): an additive 8-category cost build-up with
3 per-bucket strategies, deterministic modifiers, honest per-category uncertainty
bands, a bounded (+/-15%) hybrid-ML calibration correction, and the hard refusal
rules. The base is always current-market-rate; margin is never inside the engine.

Typical use:

    from pricing import PriceEstimationEngine, CalibrationModel, synthetic_data as sd
    ref = sd.build_reference_data()
    calib = CalibrationModel().fit(sd.build_calibration_history())
    engine = PriceEstimationEngine(ref, calibration=calib)
    est = engine.estimate(sd.sample_line_items()[1])
    print(est.summary())
"""
from .contracts import (EstimateResult, CostBreakdown, CostComponent,
                        FeasibilityResult, LeadTimeBreakdown, QuoteValidity, Calibration,
                        BUCKETS, COST_CATEGORIES, STATUSES)
from .buildup import LineItem
from .reference_data import (ReferenceData, RawMaterialMatrix, OpsCost,
                             ReferencePrices, RouteTemplateKB, RouteTemplate, RouteStep)
from .calibration import CalibrationModel
from .analog import AnalogIndex
from .engine import PriceEstimationEngine
from . import synthetic_data

__all__ = [
    "PriceEstimationEngine", "CalibrationModel", "AnalogIndex",
    "LineItem", "EstimateResult", "CostBreakdown", "CostComponent",
    "FeasibilityResult", "LeadTimeBreakdown", "QuoteValidity", "Calibration",
    "ReferenceData", "RawMaterialMatrix", "OpsCost", "ReferencePrices",
    "RouteTemplateKB", "RouteTemplate", "RouteStep",
    "BUCKETS", "COST_CATEGORIES", "STATUSES", "synthetic_data",
]
