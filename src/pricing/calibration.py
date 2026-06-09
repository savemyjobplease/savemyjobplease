"""Calibration layer (design doc A6) — HYBRID: an ML model supplies the bounded,
visible correction, governed by the doc's discipline so it stays a *correction*,
never the predictor.

    unit_price = rules_base x c[family, supplier, region],   c in [0.85, 1.15]

The ML (sklearn) learns the residual log-ratio log(won / rules_base) on
NON-flagged, quote-date-basis history. Every output is then:
  * clamped to +/-15% (a fit wanting more is a rules-gap routed to review),
  * suppressed to 1.0 when cell support n < N_min (hierarchical back-off + wider band),
  * forced OFF for FX-volatile / brand-locked / import-flagged items,
  * fully visible in the trace ("rules X x 1.07 [family/supplier/region, n=6]").
The rules base always stands underneath, inspectable.
"""
from __future__ import annotations
import math
import numpy as np

from .contracts import Calibration

CLAMP = (0.85, 1.15)
N_MIN = 5


class CalibrationModel:
    def __init__(self, backend="ml", n_min=N_MIN):
        self.backend = backend            # "ml" | "grid"
        self.n_min = n_min
        self.grid = {}                    # segment-tuple -> (mean_log_ratio, n) at each level
        self.counts = {}                  # (family,supplier,region) -> n
        self.global_log = 0.0
        self.ml = None
        self.fitted_ = False

    # ---- helpers ----
    @staticmethod
    def _norm(x):
        return str(x).strip().lower() if x is not None else "na"

    def _seg(self, fam, sup, reg):
        return (self._norm(fam), self._norm(sup), self._norm(reg))

    def cell_support(self, fam, sup, reg):
        return self.counts.get(self._seg(fam, sup, reg), 0)

    # ---- fit on converted history ----
    def fit(self, history):
        """history: DataFrame with part_family, supplier, region, rules_base,
        won_price, and optional 'flagged' (bool). Flagged rows are EXCLUDED."""
        import pandas as pd
        df = history.copy()
        if "flagged" in df.columns:
            n_flagged = int(df["flagged"].fillna(False).astype(bool).sum())
            df = df[~df["flagged"].fillna(False).astype(bool)]
            if n_flagged:
                print(f"[calib] excluded {n_flagged} flagged (FX/brand-locked/import) rows")
        df = df.dropna(subset=["rules_base", "won_price"])
        df = df[(df["rules_base"] > 0) & (df["won_price"] > 0)]
        if len(df) == 0:
            print("[calib] no usable history -> calibration will default to 1.0")
            return self
        df["log_ratio"] = np.log(df["won_price"] / df["rules_base"])
        self.global_log = float(df["log_ratio"].mean())

        f = df["part_family"].map(self._norm)
        s = df["supplier"].map(self._norm)
        r = df["region"].map(self._norm)
        for level_cols, key_fn in [
            (("f", "s", "r"), lambda a, b, c: (a, b, c)),
            (("f", "r"),      lambda a, b, c: (a, c)),
            (("f",),          lambda a, b, c: (a,)),
        ]:
            tmp = {}
            for a, b, c, lr in zip(f, s, r, df["log_ratio"]):
                k = key_fn(a, b, c)
                tmp.setdefault(k, []).append(lr)
            for k, v in tmp.items():
                self.grid[k] = (float(np.mean(v)), len(v))
        for a, b, c in zip(f, s, r):
            k = (a, b, c)
            self.counts[k] = self.counts.get(k, 0) + 1

        if self.backend == "ml" and len(df) >= self.n_min:
            try:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.linear_model import Ridge
                from sklearn.compose import ColumnTransformer
                X = pd.DataFrame({"part_family": f.values, "supplier": s.values, "region": r.values})
                pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"),
                                          ["part_family", "supplier", "region"])])
                self.ml = Pipeline([("pre", pre), ("reg", Ridge(alpha=2.0))])
                self.ml.fit(X, df["log_ratio"].values)
            except Exception as e:
                print("[calib] ML backend unavailable, using grid means:", e)
                self.ml = None
        self.fitted_ = True
        return self

    # ---- predict the bounded factor for one line ----
    def factor(self, line, rules_base):
        """Returns (Calibration, extras) where extras flags over_clamp / widen_band / excluded."""
        cal = Calibration()
        extras = {"over_clamp": False, "widen_band": False, "excluded": False}
        fam, sup, reg = line.part_family, line.supplier, line.region
        cal.segment = f"{self._norm(fam)}|{self._norm(sup)}|{self._norm(reg)}"

        if not self.fitted_:
            cal.note = "calibration not fitted -> 1.0 (pure rules)"
            extras["widen_band"] = True
            return cal, extras

        # constraint exclusion (doc Part B #2)
        if getattr(line, "_flagged", False):
            cal.note = "calibration OFF (constraint-flagged item)"
            extras["excluded"] = True
            return cal, extras

        n = self.cell_support(fam, sup, reg)
        cal.support_n = n
        if n < self.n_min:
            # hierarchical back-off, no fitted factor at the full segment
            for k in [(self._norm(fam), self._norm(reg)), (self._norm(fam),)]:
                if k in self.grid and self.grid[k][1] >= self.n_min:
                    cal.note = (f"calibration not applied at full segment (n={n}); "
                                f"backed off to {k} -> 1.0, band widened")
                    break
            else:
                cal.note = f"calibration not applied (n={n}); pure rules, band widened"
            extras["widen_band"] = True
            return cal, extras

        # enough support: ML (or grid) -> clamp
        if self.ml is not None:
            import pandas as pd
            X = pd.DataFrame([{"part_family": self._norm(fam),
                               "supplier": self._norm(sup), "region": self._norm(reg)}])
            log_r = float(self.ml.predict(X)[0]); cal.backend = "ml"
        else:
            log_r = self.grid.get((self._norm(fam), self._norm(sup), self._norm(reg)),
                                  (self.global_log, n))[0]
            cal.backend = "grid"
        raw = math.exp(log_r)
        clamped = min(CLAMP[1], max(CLAMP[0], raw))
        if abs(raw - clamped) > 1e-9:
            extras["over_clamp"] = True
            cal.note = (f"fit wanted x{raw:.3f} beyond +/-15% clamp -> capped at "
                        f"x{clamped:.3f} AND routed to review (rules-gap signal)")
        else:
            cal.note = f"applied x{clamped:.3f} [{cal.segment}, n={n}, {cal.backend}]"
        cal.factor = clamped
        cal.applied = True
        return cal, extras

    # ---- persistence ----
    def save(self, path):
        import joblib; joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib; return joblib.load(path)
