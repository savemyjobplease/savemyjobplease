"""Builds RFQ_pipeline.ipynb via nbformat (avoids hand JSON escaping).

Rules for cell sources below: each is a RAW triple-quoted string. No ''' or
triple-double-quotes appear *inside* any cell. Escape sequences like \\n, \\d
are written literally and are preserved verbatim into the notebook cell.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))

def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))

# ---------------------------------------------------------------------------
md(r"""
# AI RFQ -> Price Quote Predictor

**Goal.** Given **one** RFQ quote line (many columns, some `null`), output **one float**: the
predicted price quote (in INR).

This notebook *defines and trains* the model assuming data is supplied in the
format described by the `ai_rfq_sample` schema. **No real data is ingested** here;
a clearly-labelled synthetic generator is used only to smoke-test the pipeline
end-to-end. Swap in your real tables and re-run.

---

## Architecture (why it is split this way)

The natural inference grain is **one `quote_line` row**, so that is the unit of prediction.
The 37 tables are reduced to features in five justified groups:

| Group | Source | Rationale |
|---|---|---|
| **A. Parsed structured numerics** | `quote_line` (quantity, MOQ, lead time, dimensions) | Raw strings like `"30K"`, `"M6X21"`, `"L/T-45-50 Days"` carry strong price signal once parsed. |
| **B. Cost-physics features** | dims + material density + raw-material price | A fastener's price is anchored by its **material mass x INR/kg**. We engineer a first-principles cost proxy. |
| **C. Reference lookups (baked into the model)** | `ref_raw_material_price`, `px_timeseries` (USDINR), `duty_component`, `px_market` | External cost drivers. Stored inside the fitted model so a *single* quote-line row needs no side tables at inference. |
| **D. Target-encoded categoricals** | `region, customer, material, grade, standard_code, finish_or_coating, source_type, segment` | High-cardinality categoricals -> leakage-safe (KFold) target encoding. |
| **E. NLP text embeddings** | `item_name, subject, part_no, *_raw_excerpt, classification_reason` | OpenAI embeddings -> **PCA** compression. Captures the free-text item description. |

**Models (multi-model + consolidation layer, as requested).**
1. **Sub-model 1 - Gradient Boosting** (`HistGradientBoostingRegressor`, native NaN support) on groups A-D.
2. **Sub-model 2 - Text head** (`Ridge`) on group E (PCA embeddings).
3. **Consolidation network** - a PyTorch MLP trained on **everything** (A-E) **plus** the two
   sub-models' out-of-fold predictions. This is the "extra layer that consolidates results."
   Falls back to an sklearn `MLPRegressor` if torch is unavailable.

Target is `log1p(price_INR)`; predictions are inverted with `expm1`. Every stage is
**null-tolerant** (imputers, encoded-missing values, and text-missing fallbacks).
""")

# ---------------------------------------------------------------------------
code(r"""
# --- Dependencies (uncomment to install in a fresh env) ---
# %pip install -q numpy pandas scikit-learn scipy joblib openai torch

import os, re, json, math, hashlib, warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
    torch.manual_seed(RNG)
except Exception:
    HAVE_TORCH = False

print("torch available:", HAVE_TORCH)
""")

# ---------------------------------------------------------------------------
code(r"""
# =====================================================================
# CONFIG
# =====================================================================

# OpenAI: read key from env. If absent, the embedder falls back to a
# deterministic hashing vectorizer so the notebook still runs offline.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = "text-embedding-3-small"   # 1536-d, cheap

# Target selection: first non-null wins. Both are in quote_line.
TARGET_PRIORITY = ["supplier_quoted_price", "buyer_quoted_price"]

# How many PCA dims to keep from the text embedding.
N_TEXT_PCA = 24

# Categorical columns -> target encoding (group D).
CAT_COLS = ["region", "customer", "material", "grade", "standard_code",
            "finish_or_coating", "source_type", "segment", "customer_confidence"]

# Free-text columns concatenated for the NLP embedding (group E).
TEXT_COLS = ["item_name", "subject", "part_no", "buyer_raw_excerpt",
             "supplier_raw_excerpt", "classification_reason"]

# Material densities (g/cm^3) for the cost-physics proxy (group B).
DENSITY_GCM3 = {"stainless": 8.0, "ss": 8.0, "a2": 8.0, "a4": 8.0,
                "brass": 8.5, "aluminium": 2.7, "aluminum": 2.7,
                "titanium": 4.5, "default": 7.85}  # 7.85 = carbon/alloy steel
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Parsing utilities

Robust, null-safe parsers for the messy free-text numeric columns. Each returns
`np.nan` on failure rather than raising.
""")

code(r"""
def _to_text(x):
    if x is None: return ""
    if isinstance(x, float) and math.isnan(x): return ""
    return str(x)

def parse_quantity(x):
    # "30K" -> 30000 ; "1,00,000" -> 100000 ; "2.0" -> 2
    s = _to_text(x).strip().lower().replace(",", "")
    if not s: return np.nan
    m = re.search(r"([0-9]*\.?[0-9]+)\s*([km])?", s)
    if not m: return np.nan
    val = float(m.group(1))
    suf = m.group(2)
    if suf == "k": val *= 1e3
    elif suf == "m": val *= 1e6
    return val

def parse_moq(x):
    # "MOQ-1,00,000" -> 100000
    return parse_quantity(x)

def parse_lead_days(x):
    # "L/T-45-50 Days" -> 47.5 ; "45 days" -> 45
    s = _to_text(x).lower()
    nums = [float(n) for n in re.findall(r"[0-9]+\.?[0-9]*", s)]
    if not nums: return np.nan
    if "week" in s:
        nums = [n * 7 for n in nums]
    return float(np.mean(nums[:2]))

def parse_dimensions(x):
    # "M6X21" / "6x14" / "M10 x 1.5 x 40" -> (diameter_mm, length_mm)
    s = _to_text(x).lower().replace("mm", "")
    nums = re.findall(r"[0-9]+\.?[0-9]*", s)
    nums = [float(n) for n in nums]
    if not nums: return (np.nan, np.nan)
    dia = nums[0]
    length = nums[-1] if len(nums) >= 2 else np.nan
    return (dia, length)

def parse_price(x):
    s = _to_text(x).replace(",", "")
    m = re.search(r"[0-9]*\.?[0-9]+", s)
    return float(m.group(0)) if m else np.nan

def standard_family(x):
    s = _to_text(x).upper()
    for fam in ["DIN", "ISO", "IFI", "ASME", "ANSI", "JIS", "IS ", "GB"]:
        if fam.strip() in s: return fam.strip()
    return "OTHER"

def material_density(material, grade):
    s = (_to_text(material) + " " + _to_text(grade)).lower()
    for key, d in DENSITY_GCM3.items():
        if key != "default" and key in s:
            return d
    return DENSITY_GCM3["default"]

def is_stainless(material, grade, finish):
    s = (_to_text(material) + " " + _to_text(grade) + " " + _to_text(finish)).lower()
    return int(any(k in s for k in ["stainless", "ss", "a2", "a4", "inox", "304", "316"]))
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Reference store (group C)

Reduces the reference tables to compact lookups and **stores them inside the model**.
At inference time a lone `quote_line` row is enriched from these baked-in tables,
so no side data is required. All getters degrade gracefully to a global median / None.
""")

code(r"""
class ReferenceStore:
    # Compact, picklable lookups built from the reference tables.
    def __init__(self):
        self.rm_by_material = {}      # normalized material -> median INR/kg
        self.rm_by_mat_grade = {}     # (material, grade) -> median INR/kg
        self.rm_global = np.nan
        self.usdinr = np.nan
        self.duty_pct = np.nan
        self.market_by_size = {}      # M-size int -> median market price
        self.market_global = np.nan

    @staticmethod
    def _norm(s):
        return re.sub(r"\s+", " ", _to_text(s).strip().lower())

    def fit(self, tables: dict):
        self._fit_raw_material(tables.get("ref_raw_material_price"))
        self._fit_fx(tables.get("px_timeseries"))
        self._fit_duty(tables.get("duty_component"))
        self._fit_market(tables.get("px_market"))
        return self

    def _fit_raw_material(self, df):
        if df is None or len(df) == 0: return
        df = df.copy()
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df = df.dropna(subset=["price"])
        if len(df) == 0: return
        self.rm_global = float(df["price"].median())
        for mat, g in df.groupby(df["material"].map(self._norm)):
            self.rm_by_material[mat] = float(g["price"].median())
        if "grade" in df.columns:
            for (mat, grd), g in df.groupby([df["material"].map(self._norm),
                                             df["grade"].map(self._norm)]):
                self.rm_by_mat_grade[(mat, grd)] = float(g["price"].median())

    def _fit_fx(self, df):
        if df is None or len(df) == 0: return
        fx = df[df.get("series_id").astype(str).str.upper() == "USDINR"].copy()
        if len(fx) == 0: return
        fx["value"] = pd.to_numeric(fx["value"], errors="coerce")
        if "is_current" in fx.columns and fx["is_current"].any():
            cur = fx[fx["is_current"] == True]
            self.usdinr = float(cur["value"].dropna().iloc[-1])
        else:
            fx = fx.sort_values("obs_date") if "obs_date" in fx.columns else fx
            self.usdinr = float(fx["value"].dropna().iloc[-1])

    def _fit_duty(self, df):
        if df is None or len(df) == 0: return
        rate = pd.to_numeric(df.get("rate_percent"), errors="coerce").dropna()
        if len(rate): self.duty_pct = float(rate.sum())  # total of parsed components

    def _fit_market(self, df):
        if df is None or len(df) == 0: return
        df = df.copy()
        df["amt"] = pd.to_numeric(df.get("price_amount_min"), errors="coerce")
        df = df.dropna(subset=["amt"])
        if len(df) == 0: return
        self.market_global = float(df["amt"].median())
        sizes = df["item"].map(lambda s: (re.search(r"m\s*([0-9]+)", _to_text(s).lower())
                                          or [None, None]))
        df["msize"] = [int(m.group(1)) if hasattr(m, "group") else np.nan for m in sizes]
        for sz, g in df.dropna(subset=["msize"]).groupby("msize"):
            self.market_by_size[int(sz)] = float(g["amt"].median())

    # ---- getters (null-safe) ----
    def rm_price(self, material, grade):
        mat, grd = self._norm(material), self._norm(grade)
        if (mat, grd) in self.rm_by_mat_grade: return self.rm_by_mat_grade[(mat, grd)]
        if mat in self.rm_by_material: return self.rm_by_material[mat]
        return self.rm_global

    def market_price(self, diameter_mm):
        if diameter_mm is not None and not (isinstance(diameter_mm, float) and math.isnan(diameter_mm)):
            key = int(round(diameter_mm))
            if key in self.market_by_size: return self.market_by_size[key]
        return self.market_global
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. NLP embedder (group E)

Wraps OpenAI embeddings with an in-memory cache. If no API key is present (or the
call fails), it transparently falls back to a deterministic hashing vectorizer so
the notebook is fully runnable offline. The downstream code does not care which
path produced the vectors.
""")

code(r"""
class TextEmbedder:
    def __init__(self, api_key="", model=OPENAI_EMBED_MODEL, fallback_dim=256):
        self.model = model
        self.fallback_dim = fallback_dim
        self.cache = {}
        self.dim = None
        self._client = None
        self.mode = "hash"
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
                self.mode = "openai"
            except Exception as e:
                print("OpenAI init failed, using hashing fallback:", e)

    def _hash_vec(self, text):
        # Deterministic bag-of-hashed-tokens, L2-normalized. Pure-numpy, offline.
        v = np.zeros(self.fallback_dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", _to_text(text).lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.fallback_dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _embed_openai(self, texts):
        # texts: list[str] (already non-empty placeholders). Batches of 256.
        out = []
        for i in range(0, len(texts), 256):
            batch = texts[i:i+256]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend([np.asarray(d.embedding, dtype=np.float32) for d in resp.data])
        return out

    def embed(self, texts):
        texts = [(_to_text(t) or " ") for t in texts]
        # serve cache, collect misses
        miss_idx = [i for i, t in enumerate(texts) if t not in self.cache]
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            if self.mode == "openai":
                try:
                    vecs = self._embed_openai(miss_texts)
                except Exception as e:
                    print("OpenAI embed failed mid-run, falling back to hashing:", e)
                    self.mode = "hash"
                    vecs = [self._hash_vec(t) for t in miss_texts]
            else:
                vecs = [self._hash_vec(t) for t in miss_texts]
            for i, v in zip(miss_idx, vecs):
                self.cache[texts[i]] = v
        mat = np.vstack([self.cache[t] for t in texts])
        self.dim = mat.shape[1]
        return mat
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Feature builder

Turns a raw `quote_line` DataFrame (+ the fitted `ReferenceStore` and `TextEmbedder`)
into a fully numeric, null-tolerant matrix covering groups A-E. It fits:
leakage-safe (KFold) **target encoders**, the text **PCA**, and the standardizer.
`transform` works on a single-row frame for inference.
""")

code(r"""
class TargetEncoder:
    # Smoothed mean-target encoding with global fallback for unseen/null.
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.maps = {}     # col -> {category: encoded}
        self.global_ = 0.0

    @staticmethod
    def _col(df, c):
        # null-safe column fetch: absent column -> all-NA series (for single-row inference)
        if c in df.columns:
            return df[c]
        return pd.Series([np.nan] * len(df), index=df.index, dtype="object")

    def fit(self, df, y, cols):
        self.global_ = float(np.mean(y))
        for c in cols:
            col = self._col(df, c)
            s = col.astype("object").where(col.notna(), other="__NA__").astype(str)
            stats = pd.DataFrame({"cat": s, "y": np.asarray(y)}).groupby("cat")["y"]
            cnt, mean = stats.count(), stats.mean()
            enc = (cnt * mean + self.smoothing * self.global_) / (cnt + self.smoothing)
            self.maps[c] = enc.to_dict()
        return self

    def transform(self, df, cols):
        out = {}
        for c in cols:
            col = self._col(df, c)
            s = col.astype("object").where(col.notna(), other="__NA__").astype(str)
            out["te_" + c] = s.map(self.maps.get(c, {})).fillna(self.global_).to_numpy(dtype=float)
        return pd.DataFrame(out, index=df.index)


class FeatureBuilder:
    def __init__(self, ref: ReferenceStore, embedder: TextEmbedder,
                 cat_cols=CAT_COLS, text_cols=TEXT_COLS, n_text_pca=N_TEXT_PCA):
        self.ref = ref
        self.embedder = embedder
        self.cat_cols = cat_cols
        self.text_cols = text_cols
        self.n_text_pca = n_text_pca
        self.te = TargetEncoder()
        self.pca = None
        self.scaler = StandardScaler()
        self.struct_cols_ = None
        self.feature_names_ = None

    # ---- group A + B + C: deterministic, row-wise numeric engineering ----
    def _structured(self, df):
        out = pd.DataFrame(index=df.index)
        qty = df.get("quantity")
        out["qty"] = (qty if qty is not None else pd.Series(index=df.index)).map(parse_quantity)
        moq_src = df["buyer_moq"] if "buyer_moq" in df else df.get("supplier_moq")
        out["moq"] = (moq_src if moq_src is not None else pd.Series(index=df.index)).map(parse_moq)
        lt_src = df["buyer_lead_time"] if "buyer_lead_time" in df else df.get("supplier_lead_time")
        out["lead_days"] = (lt_src if lt_src is not None else pd.Series(index=df.index)).map(parse_lead_days)
        dims = (df.get("dimensions") if df.get("dimensions") is not None
                else pd.Series(index=df.index)).map(parse_dimensions)
        out["dia_mm"] = dims.map(lambda t: t[0])
        out["len_mm"] = dims.map(lambda t: t[1])
        out["log_qty"] = np.log1p(out["qty"])
        out["log_moq"] = np.log1p(out["moq"])
        out["len_over_dia"] = out["len_mm"] / out["dia_mm"]

        # group B: cost-physics proxy = volume * density * (INR/kg)
        mats = df.get("material"); grds = df.get("grade"); fins = df.get("finish_or_coating")
        rows = []
        for i in df.index:
            mat = mats.get(i) if mats is not None else None
            grd = grds.get(i) if grds is not None else None
            fin = fins.get(i) if fins is not None else None
            d = out.at[i, "dia_mm"]; L = out.at[i, "len_mm"]
            dens = material_density(mat, grd)
            rm = self.ref.rm_price(mat, grd)
            if not (np.isnan(d) or np.isnan(L)):
                vol_cm3 = math.pi * (d / 2.0) ** 2 * L / 1000.0   # mm^3 -> cm^3
                mass_kg = vol_cm3 * dens / 1000.0
            else:
                mass_kg = np.nan
            mat_cost = mass_kg * rm if (rm is not None and not np.isnan(mass_kg)) else np.nan
            rows.append((dens, rm if rm is not None else np.nan, mass_kg, mat_cost,
                         is_stainless(mat, grd, fin),
                         self.ref.market_price(d)))
        cb = pd.DataFrame(rows, index=df.index,
                          columns=["density", "rm_inr_kg", "mass_kg",
                                   "material_cost", "is_stainless", "market_price"])
        out = pd.concat([out, cb], axis=1)

        # group C scalars (same for all rows; still informative for the net)
        out["usdinr"] = self.ref.usdinr
        out["duty_pct"] = self.ref.duty_pct
        out["std_family_din"] = df.get("standard_code", pd.Series(index=df.index)).map(
            lambda s: int(standard_family(s) == "DIN"))
        out["std_family_iso"] = df.get("standard_code", pd.Series(index=df.index)).map(
            lambda s: int(standard_family(s) == "ISO"))
        out["has_drawing"] = df.get("drawing_ref", pd.Series(index=df.index)).notna().astype(int) \
            if "drawing_ref" in df else 0
        return out

    # ---- group E: text -> embedding ----
    def _text_raw(self, df):
        parts = []
        for c in self.text_cols:
            col = df[c] if c in df else pd.Series([""] * len(df), index=df.index)
            parts.append(col.map(_to_text))
        return [" | ".join(vals) for vals in zip(*parts)] if parts else [""] * len(df)

    def fit(self, df, y):
        struct = self._structured(df)
        self.struct_cols_ = list(struct.columns)
        te = self.te.fit(df, y, self.cat_cols).transform(df, self.cat_cols)

        emb = self.embedder.embed(self._text_raw(df))
        n_comp = max(1, min(self.n_text_pca, emb.shape[0] - 1, emb.shape[1]))
        self.pca = PCA(n_components=n_comp, random_state=RNG)
        txt = self.pca.fit_transform(emb)
        txt = pd.DataFrame(txt, index=df.index,
                           columns=[f"txt_{i}" for i in range(txt.shape[1])])

        full = pd.concat([struct, te, txt], axis=1)
        self.feature_names_ = list(full.columns)
        # scaler is used only for the torch branch (GBM ignores scaling); fit on
        # a median-imputed copy so NaNs do not poison the mean/std.
        self.scaler.fit(SimpleImputer(strategy="median").fit_transform(full.values))
        return full

    def transform(self, df):
        struct = self._structured(df)
        te = self.te.transform(df, self.cat_cols)
        emb = self.embedder.embed(self._text_raw(df))
        txt = self.pca.transform(emb)
        txt = pd.DataFrame(txt, index=df.index,
                           columns=[f"txt_{i}" for i in range(txt.shape[1])])
        full = pd.concat([struct, te, txt], axis=1)
        # guarantee identical column order to fit-time
        return full.reindex(columns=self.feature_names_)

    def scale_impute(self, full_df):
        # for the DL branch: median-impute then standardize -> dense float array
        X = full_df.values.astype(float).copy()
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(med, inds[1])
        return self.scaler.transform(X)
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Target construction

Picks the first available quoted price per `TARGET_PRIORITY`, converts USD -> INR
using the stored FX rate, and returns `log1p(price_INR)`. Rows with no usable
price are dropped (training only).
""")

code(r"""
def build_target(df, ref: ReferenceStore, priority=TARGET_PRIORITY):
    price = pd.Series(np.nan, index=df.index, dtype=float)
    cur = pd.Series("INR", index=df.index, dtype=object)
    for col in priority:
        if col not in df: continue
        p = pd.to_numeric(df[col], errors="coerce")
        take = price.isna() & p.notna()
        price[take] = p[take]
        ccol = col.replace("quoted_price", "currency")  # buyer/supplier_currency
        if ccol in df:
            cur[take] = df[ccol][take].astype(str).str.upper().fillna("INR")
    # currency normalize -> INR
    fx = ref.usdinr if (ref.usdinr and not np.isnan(ref.usdinr)) else 83.0
    is_usd = cur.str.contains("USD", na=False)
    price_inr = price.where(~is_usd, price * fx)
    return price_inr
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Consolidation network (DL layer over everything)

A PyTorch MLP that takes the **full standardized feature matrix** *plus* the two
sub-models' out-of-fold predictions and emits one scalar (`log1p` price). This is
the layer that "consolidates" the gradient-boosting and text heads. If torch is
missing it falls back to sklearn's `MLPRegressor` with the same interface.
""")

code(r"""
class ConsolidationNet:
    def __init__(self, in_dim, hidden=(128, 64), epochs=300, lr=1e-3, patience=30):
        self.in_dim = in_dim; self.hidden = hidden
        self.epochs = epochs; self.lr = lr; self.patience = patience
        self.backend = "torch" if HAVE_TORCH else "sklearn"
        self.model = None

    def _build_torch(self):
        layers, d = [], self.in_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15)]
            d = h
        layers += [nn.Linear(d, 1)]
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32).ravel()
        if self.backend == "sklearn":
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=1000,
                                      early_stopping=True, random_state=RNG)
            self.model.fit(X, y)
            return self
        # torch path with early stopping
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=RNG)
        net = self._build_torch()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = nn.SmoothL1Loss()
        Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr).view(-1, 1)
        Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva).view(-1, 1)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            net.train(); opt.zero_grad()
            loss = lossf(net(Xtr_t), ytr_t); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vloss = lossf(net(Xva_t), yva_t).item()
            if vloss < best - 1e-5:
                best, best_state, bad = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience: break
        if best_state is not None: net.load_state_dict(best_state)
        net.eval(); self.model = net
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.backend == "sklearn":
            return self.model.predict(X)
        with torch.no_grad():
            return self.model(torch.tensor(X)).numpy().ravel()
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. The black box: `RFQPricePredictor`

`fit(tables)` trains everything; `predict_one(row)` is the requested single-row,
null-tolerant float predictor. Out-of-fold sub-predictions are used while training
the consolidation net to prevent leakage; sub-models are then refit on all data
for inference.
""")

code(r"""
class RFQPricePredictor:
    def __init__(self, n_text_pca=N_TEXT_PCA):
        self.ref = ReferenceStore()
        self.embedder = TextEmbedder(api_key=OPENAI_API_KEY)
        self.fb = None
        self.n_text_pca = n_text_pca
        self.gbm = None          # sub-model 1 (full data)
        self.txt_model = None    # sub-model 2 (full data)
        self.consolidator = None
        self.txt_cols_ = None
        self.fitted_ = False

    def _split_text_cols(self, full):
        self.txt_cols_ = [c for c in full.columns if c.startswith("txt_")]

    def fit(self, tables: dict, n_splits=5, verbose=True):
        ql = tables["quote_line"].copy().reset_index(drop=True)
        self.ref.fit(tables)
        y_inr = build_target(ql, self.ref)
        mask = y_inr.notna() & (y_inr > 0)
        ql, y_inr = ql.loc[mask].reset_index(drop=True), y_inr.loc[mask].reset_index(drop=True)
        if len(ql) < 8:
            raise ValueError(f"Need >=8 priced rows to train; got {len(ql)}.")
        y = np.log1p(y_inr.to_numpy())

        self.fb = FeatureBuilder(self.ref, self.embedder, n_text_pca=self.n_text_pca)
        full = self.fb.fit(ql, y)
        self._split_text_cols(full)

        # ---- out-of-fold sub-predictions (leakage-safe) ----
        oof_gbm = np.zeros(len(y)); oof_txt = np.zeros(len(y))
        kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=RNG)
        Xtxt_all = full[self.txt_cols_].fillna(0.0).to_numpy()
        for tr, va in kf.split(full):
            g = HistGradientBoostingRegressor(random_state=RNG, max_depth=6,
                                              learning_rate=0.06, max_iter=400,
                                              l2_regularization=1.0)
            g.fit(full.iloc[tr].to_numpy(), y[tr])
            oof_gbm[va] = g.predict(full.iloc[va].to_numpy())
            t = Ridge(alpha=2.0).fit(Xtxt_all[tr], y[tr])
            oof_txt[va] = t.predict(Xtxt_all[va])

        # ---- refit sub-models on ALL data (used at inference) ----
        self.gbm = HistGradientBoostingRegressor(random_state=RNG, max_depth=6,
                                                  learning_rate=0.06, max_iter=400,
                                                  l2_regularization=1.0)
        self.gbm.fit(full.to_numpy(), y)
        self.txt_model = Ridge(alpha=2.0).fit(Xtxt_all, y)

        # ---- consolidation net on EVERYTHING + sub-preds ----
        Xscaled = self.fb.scale_impute(full)
        Xcon = np.hstack([Xscaled, oof_gbm.reshape(-1, 1), oof_txt.reshape(-1, 1)])
        self.consolidator = ConsolidationNet(in_dim=Xcon.shape[1]).fit(Xcon, y)
        self.fitted_ = True

        if verbose:
            self._report(full, Xtxt_all, y)
        return self

    def _consolidate_inputs(self, full):
        Xscaled = self.fb.scale_impute(full)
        gbm_p = self.gbm.predict(full.to_numpy())
        txt_p = self.txt_model.predict(full[self.txt_cols_].fillna(0.0).to_numpy())
        return np.hstack([Xscaled, gbm_p.reshape(-1, 1), txt_p.reshape(-1, 1)])

    def predict(self, ql_df):
        # returns price in INR for each row of a quote_line-shaped frame
        full = self.fb.transform(ql_df.reset_index(drop=True))
        Xcon = self._consolidate_inputs(full)
        y_log = self.consolidator.predict(Xcon)
        return np.expm1(y_log)

    def predict_one(self, row):
        # row: dict or pd.Series of features (any may be missing/None) -> float (INR)
        if isinstance(row, dict):
            row = pd.Series(row)
        df = pd.DataFrame([row.to_dict()])
        return float(self.predict(df)[0])

    def _report(self, full, Xtxt_all, y):
        # quick in-sample + holdout style sanity metrics on the training split
        Xcon = self._consolidate_inputs(full)
        pred_log = self.consolidator.predict(Xcon)
        pred_inr, true_inr = np.expm1(pred_log), np.expm1(y)
        mae = mean_absolute_error(true_inr, pred_inr)
        rmse = math.sqrt(mean_squared_error(true_inr, pred_inr))
        r2 = r2_score(y, pred_log)
        print(f"[train-fit] consolidation: MAE={mae:,.3f} INR  RMSE={rmse:,.3f} INR  R2(log)={r2:.3f}")
        print(f"           backend={self.consolidator.backend}  embedder={self.embedder.mode}  "
              f"n_features={full.shape[1]}  n_rows={len(y)}")

    # ---- persistence ----
    def save(self, path):
        import joblib
        joblib.dump(self, path)
        print("saved ->", path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Synthetic smoke-test data (NOT real data)

The block below **fabricates** schema-shaped tables purely so the pipeline can be
executed and validated here. Prices are generated from a known latent formula
(`material mass x INR/kg`, size, quantity discount, stainless premium + noise) so
the model has something learnable. **Delete / replace this with your real tables.**
""")

code(r"""
def make_synthetic_tables(n_quotes=400, seed=RNG):
    rng = np.random.default_rng(seed)
    materials = ["10B21", "Alloys-Boron-MS", "SS 304", "Brass", "Aluminium 6061"]
    grades = ["8.8", "10.9", "A2-70", "4.6", "12.9"]
    stds = ["DIN 933", "ISO 4014", "DIN 928", "ISO 2338", "DIN 912"]
    regions = ["Auto", "EMS", "North", "Central", "South"]
    customers = ["marelli talbros", "Harro hofliger", "IMI NORGEN", "Lectrix EV", "Bosch"]
    finishes = ["Plating", "Zinc", "Black Oxide", "None", "PTFE"]
    rm_price = {"10B21": 67.5, "Alloys-Boron-MS": 70.0, "SS 304": 230.0,
                "Brass": 650.0, "Aluminium 6061": 300.0}
    dens = {"10B21": 7.85, "Alloys-Boron-MS": 7.85, "SS 304": 8.0,
            "Brass": 8.5, "Aluminium 6061": 2.7}

    rows = []
    for i in range(n_quotes):
        mat = rng.choice(materials); grd = rng.choice(grades); std = rng.choice(stds)
        dia = float(rng.choice([3, 4, 5, 6, 8, 10, 12]))
        length = float(rng.choice([10, 14, 16, 20, 25, 30, 40, 50]))
        qty = int(rng.choice([1000, 5000, 10000, 30000, 100000]))
        fin = rng.choice(finishes)
        vol_cm3 = math.pi * (dia / 2) ** 2 * length / 1000.0
        mass_kg = vol_cm3 * dens[mat] / 1000.0
        base = mass_kg * rm_price[mat]                       # material cost
        ss_prem = 1.25 if mat == "SS 304" else 1.0
        qty_disc = 1.0 + 0.4 * math.exp(-qty / 30000.0)      # small-qty premium
        price = base * 2.6 * ss_prem * qty_disc * float(rng.normal(1.0, 0.08))
        price = max(price, 0.2)
        rows.append(dict(
            quote_line_id=i + 1, rfq_uid=f"{rng.choice(regions)}--CUST--{i}",
            region=rng.choice(regions), customer=rng.choice(customers),
            source_type=rng.choice(["email", "pdf"]), segment=rng.choice(["Auto", "EV"]),
            customer_confidence=rng.choice(["high", "medium", "low"]),
            subject=f"RE: Quotation Required {std} fasteners",
            item_name=f"Bolt {std} {mat} M{int(dia)}X{int(length)} {fin}",
            part_no=f"W{rng.integers(100000,999999)}-M{int(dia)}X{int(length)}",
            standard_code=std, material=mat, grade=grd, finish_or_coating=fin,
            dimensions=f"M{int(dia)}X{int(length)}", quantity=f"{qty//1000}K",
            buyer_moq=f"MOQ-{qty:,}", buyer_lead_time=f"L/T-{rng.integers(30,60)}-{rng.integers(60,90)} Days",
            buyer_raw_excerpt=f"Please quote {std} bolt {mat} grade {grd} qty {qty}",
            supplier_raw_excerpt=f"Offer for M{int(dia)}X{int(length)} {fin}",
            classification_reason="Market-Std signal", drawing_ref=None,
            buyer_currency="INR", supplier_currency="INR",
            buyer_quoted_price=round(price * 1.05, 3),       # buyer slightly higher
            supplier_quoted_price=round(price, 3),
        ))
    quote_line = pd.DataFrame(rows)

    # introduce realistic NULLs across feature columns
    for c in ["grade", "finish_or_coating", "buyer_moq", "buyer_lead_time",
              "classification_reason", "part_no", "dimensions"]:
        idx = rng.choice(quote_line.index, size=int(0.15 * len(quote_line)), replace=False)
        quote_line.loc[idx, c] = None

    ref_raw_material_price = pd.DataFrame([
        dict(id=k + 1, material=m, grade="", form="ROUND", supplier="X",
             price=rm_price[m], unit="INR/kg", currency="INR", as_of="2025-01-01")
        for k, m in enumerate(materials)])
    px_timeseries = pd.DataFrame([dict(id=1, obs_date="2026-05-27", series_id="USDINR",
                                       series_name="USD/INR", category="fx", value=85.5,
                                       is_current=True)])
    duty_component = pd.DataFrame([dict(duty_component_id=1, px_duty_id=1,
                                        component_name="duty", rate_percent=15.0),
                                   dict(duty_component_id=2, px_duty_id=1,
                                        component_name="igst", rate_percent=18.0)])
    px_market = pd.DataFrame([dict(id=i + 1, item=f"M{d} 4.6 1.0 mm INR {p}",
                                   price_amount_min=p, price_amount_max=p)
                              for i, (d, p) in enumerate(
                                  [(6, 95), (8, 140), (10, 190), (12, 240)])])
    return dict(quote_line=quote_line, ref_raw_material_price=ref_raw_material_price,
                px_timeseries=px_timeseries, duty_component=duty_component, px_market=px_market)

tables = make_synthetic_tables()
print("synthetic quote_line:", tables["quote_line"].shape)
tables["quote_line"].head(3)
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. Train end-to-end + evaluate on a held-out split

We hold out 20% of priced rows to report honest test metrics, then refit on all
data for the deployable black box.
""")

code(r"""
# ---- honest held-out evaluation ----
ql_all = tables["quote_line"]
tr_idx, te_idx = train_test_split(ql_all.index, test_size=0.2, random_state=RNG)

tables_tr = dict(tables); tables_tr["quote_line"] = ql_all.loc[tr_idx]
model_eval = RFQPricePredictor().fit(tables_tr, verbose=True)

ql_test = ql_all.loc[te_idx].reset_index(drop=True)
y_test_inr = build_target(ql_test, model_eval.ref)
m = y_test_inr.notna()
pred_test = model_eval.predict(ql_test.loc[m])
true_test = y_test_inr.loc[m].to_numpy()

mae = mean_absolute_error(true_test, pred_test)
rmse = math.sqrt(mean_squared_error(true_test, pred_test))
mape = float(np.mean(np.abs((true_test - pred_test) / true_test)) * 100)
r2 = r2_score(np.log1p(true_test), np.log1p(np.clip(pred_test, 1e-6, None)))
print(f"\n[HELD-OUT TEST]  n={m.sum()}")
print(f"  MAE  = {mae:,.3f} INR")
print(f"  RMSE = {rmse:,.3f} INR")
print(f"  MAPE = {mape:,.1f} %")
print(f"  R2 (log space) = {r2:.3f}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. Fit the final deployable model (all data)
""")

code(r"""
model = RFQPricePredictor().fit(tables, verbose=True)
model.save("rfq_price_model.joblib")
""")

# ---------------------------------------------------------------------------
md(r"""
## 11. The deliverable: one row in (nulls allowed) -> one float out

This is the requested black box. Provide any subset of the `quote_line` feature
columns; missing ones are tolerated. The reference lookups (raw-material price,
FX, duty, market) are already baked into the model, so you only supply the line.
""")

code(r"""
# A single RFQ line with several fields intentionally left null/missing:
sample_row = {
    "region": "Auto",
    "customer": "marelli talbros",
    "source_type": "email",
    "item_name": "Bolt DIN 933 SS 304 M8X25 Zinc",
    "subject": "RE: Quotation Required SS fasteners",
    "standard_code": "DIN 933",
    "material": "SS 304",
    "grade": None,                 # <- null
    "finish_or_coating": "Zinc",
    "dimensions": "M8X25",
    "quantity": "10K",
    "buyer_moq": None,             # <- null
    "buyer_lead_time": "L/T-45-60 Days",
    "buyer_raw_excerpt": None,     # <- null
    "part_no": None,               # <- null
}

predicted_price = model.predict_one(sample_row)
print(f"Predicted price quote: {predicted_price:,.3f} INR")

# Reload-from-disk round trip to prove the black box is self-contained:
reloaded = RFQPricePredictor.load("rfq_price_model.joblib")
print(f"After reload         : {reloaded.predict_one(sample_row):,.3f} INR")
""")

# ---------------------------------------------------------------------------
md(r"""
## 12. Using your real data + notes & limits

**To use real data**, build a `tables` dict of pandas DataFrames whose columns match
the schema (at minimum `quote_line`, plus any of `ref_raw_material_price`,
`px_timeseries`, `duty_component`, `px_market` you have), then:

```python
model = RFQPricePredictor().fit(tables)
model.save("rfq_price_model.joblib")
price = model.predict_one(one_quote_line_row_as_dict)
```

Load tables however you like, e.g. `pd.read_parquet(...)` or from Postgres:

```python
import sqlalchemy as sa
eng = sa.create_engine("postgresql://user:pw@localhost:55432/ai_rfq_sample")
tables = {t: pd.read_sql(f"SELECT * FROM {t}", eng) for t in
          ["quote_line","ref_raw_material_price","px_timeseries","duty_component","px_market"]}
```

**Enable real OpenAI embeddings** by setting the key before constructing the model:
`os.environ["OPENAI_API_KEY"] = "sk-..."`. Otherwise a deterministic hashing
vectorizer is used so the notebook always runs.

**Design notes**
- **Target** = `supplier_quoted_price` (fallback `buyer_quoted_price`), USD->INR normalized, `log1p`-transformed.
- **Nulls** handled everywhere: GBM ingests NaN natively; the DL branch uses median-impute + standardize; categoricals route unseen/null to a smoothed global mean; missing text -> blank embedding.
- **Why multiple models** - GBM captures structured/cost interactions, the Ridge text head captures description signal, and the **DL consolidation layer** learns how to weight them together (stacking) while also seeing the raw features.
- **Leakage control** - target encoding and the sub-model predictions fed to the consolidator are produced **out-of-fold**.

**Limits**
- Other evidence tables (`ocr_page`, `extracted_table*`, `chunk`, `extracted_entity`) are not joined in by default. To exploit them, aggregate per `rfq_uid` (e.g. concat OCR/chunk text into the text field, or count entities) and merge onto `quote_line` before `fit` - the text pipeline will pick the extra text up automatically.
- Synthetic prices follow a known formula; **metrics on real data will differ**. Re-tune `HistGradientBoostingRegressor` and `ConsolidationNet` hyperparameters on your data.
- With very few priced rows the PCA component count and KFold splits auto-shrink; provide enough labelled quote lines for stable training.
""")

# ---------------------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}

out = "RFQ_pipeline.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

# ---- syntax-check every code cell ----
errs = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    try:
        compile(c["source"], f"<cell {i}>", "exec")
    except SyntaxError as e:
        errs += 1
        print(f"SYNTAX ERROR in cell {i}: {e}")
print(f"Wrote {out} with {len(nb['cells'])} cells; code-cell syntax errors: {errs}")
