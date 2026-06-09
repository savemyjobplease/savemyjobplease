"""Synthetic, schema-shaped reference data + sample line items + calibration history.

NOT real data. Exists only so the engine and notebook run end-to-end. Replace the
builders with loaders over the real `ai_rfq_sample` tables (RawMaterialMatrix.from_schema
etc.) when wiring to production.
"""
from __future__ import annotations
import numpy as np

from .reference_data import (ReferenceData, RawMaterialMatrix, OpsCost,
                             ReferencePrices, RouteTemplateKB)


def build_reference_data() -> ReferenceData:
    rm = RawMaterialMatrix()
    # grade -> INR/kg (illustrative). SCM 435 deliberately ABSENT -> missing_rm_rate.
    for grade, rate, vol in [("8.8", 72.0, "monthly"), ("10.9", 95.0, "monthly"),
                             ("12.9", 110.0, "monthly"), ("a2-70", 230.0, "weekly"),
                             ("4140", 110.0, "monthly"), ("10b21", 67.5, "monthly")]:
        rm.add(grade, rate, as_of="2025-01-01", volatility_class=vol)

    rp = ReferencePrices()
    # catalog/stocked anchors for Bucket A/B
    rp.add_catalog("DIN 912", "M8x40", "10.9", "plain", 6.20)
    rp.add_catalog("DIN 933", "M6x20", "8.8", "zinc", 3.10)
    rp.add_catalog("DIN 934", "M8", "8.8", "zinc", 1.45)

    ref = ReferenceData(rm=rm, ops=OpsCost(), ref_prices=rp,
                        routes=RouteTemplateKB(), fx_usdinr=85.5, fx_as_of_days=1)
    return ref


def sample_line_items() -> list[dict]:
    return [
        # Bucket A - catalog hex bolt
        dict(id="A-hexbolt-M6x20", bucket="assortment", part_family="hex_bolt",
             standard_reference="DIN 933", size="M6x20", material_grade="8.8",
             material_family="carbon_steel", finish="zinc", quantity=20000,
             customer="acme auto", supplier="s1", region="West"),

        # Bucket B - SHCS longer + plating change (doc worked example a)
        dict(id="B-shcs-M8x45", bucket="special_but_standard", part_family="shcs",
             standard_reference="DIN 912", size="M8x45", nominal_diameter_mm=8,
             nominal_length_mm=45, material_grade="10.9", material_family="alloy_steel",
             finish="znni", quantity=5000, customer="acme auto", supplier="s1", region="West",
             ref_standard="DIN 912", ref_size="M8x40", ref_grade="10.9",
             ref_finish="plain", ref_length_mm=40),

        # Bucket C - to-the-print stepped pin, tolerance gap (doc worked example b),
        # but on 4140 (in matrix) so it builds; SCM435 variant below is blocked.
        dict(id="C-stepped-pin-4140", bucket="to_the_print", part_family="pin",
             size="D12x96", nominal_diameter_mm=12, nominal_length_mm=96,
             material_grade="4140", material_family="alloy_steel",
             requested_tolerance_um=10, quantity=8000, net_weight_kg=0.0240,
             customer="nova devices", supplier="s2", region="North"),

        # Bucket C - SCM 435 NOT in matrix -> missing_rm_rate -> BLOCK
        dict(id="C-pin-SCM435-missing-rate", bucket="to_the_print", part_family="pin",
             size="D12x96", nominal_diameter_mm=12, nominal_length_mm=96,
             material_grade="SCM 435", material_family="alloy_steel",
             requested_tolerance_um=10, quantity=8000, net_weight_kg=0.0240,
             customer="nova devices", supplier="s2", region="North"),

        # Bucket C - missing tolerance -> loose/tight scenarios -> review
        dict(id="C-custom-no-tol", bucket="to_the_print", part_family="custom_special",
             size="D10x60", nominal_diameter_mm=10, nominal_length_mm=60,
             material_grade="10.9", material_family="alloy_steel",
             requested_tolerance_um=None, quantity=1500, customer="nova devices",
             supplier="s3", region="South"),

        # Brand-locked customer -> calibration OFF, vendor RFQ
        dict(id="C-brandlocked-EJOT", bucket="to_the_print", part_family="hex_bolt",
             size="M10x40", nominal_diameter_mm=10, nominal_length_mm=40,
             material_grade="10.9", material_family="alloy_steel",
             requested_tolerance_um=50, quantity=12000, customer="EJOT", supplier="s1",
             region="West"),
    ]


def build_calibration_history(n_per_cell=8, seed=42):
    """Converted-quote history for the calibration layer. Includes flagged rows
    (excluded by the model) and a dense (shcs,s1,west) cell so calibration applies."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    cells = [("shcs", "s1", "West", 1.06), ("hex_bolt", "s1", "West", 1.03),
             ("pin", "s2", "North", 0.97), ("custom_special", "s3", "South", 1.10)]
    for fam, sup, reg, true_f in cells:
        for _ in range(n_per_cell):
            base = float(rng.uniform(4, 12))
            won = base * true_f * float(rng.normal(1.0, 0.04))
            rows.append(dict(part_family=fam, supplier=sup, region=reg,
                             rules_base=round(base, 3), won_price=round(won, 3),
                             flagged=False))
    # a few flagged (FX/brand-locked) rows that MUST be excluded
    for _ in range(6):
        base = float(rng.uniform(4, 12))
        rows.append(dict(part_family="hex_bolt", supplier="s9", region="West",
                         rules_base=round(base, 3),
                         won_price=round(base * 1.6, 3),  # EUR-favorable poison
                         flagged=True))
    return pd.DataFrame(rows)


def build_analog_corpus():
    return [
        dict(label="stepped pin 4140 D12x90 (won 2025)", price=9.40,
             part_family="pin", size="D12x90", material_grade="4140"),
        dict(label="stepped pin D10x80 (won 2024)", price=7.10,
             part_family="pin", size="D10x80", material_grade="4140"),
        dict(label="dowel pin D12x100 (lost 2025)", price=10.2,
             part_family="pin", size="D12x100", material_grade="ss"),
        dict(label="custom bush D20x30 (won 2025)", price=14.0,
             part_family="bush", size="D20x30", material_grade="brass"),
    ]
