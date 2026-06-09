"""Bucket-C analog retrieval (design doc A2.3 front line).

Presents the closest historical supplier-quoted prices for a to-the-print part so a
sourcing engineer can reason by comparison. Uses OpenAI embeddings over the item
text when a key is available (the project's NLP touchpoint); otherwise falls back
to a deterministic family+size matcher so it always runs offline.
"""
from __future__ import annotations
import os, re
import numpy as np


def _item_text(d: dict) -> str:
    return " | ".join(str(d.get(k, "")) for k in
                      ("part_family", "standard_reference", "size", "material_grade",
                       "material_family", "finish", "notes"))


class AnalogIndex:
    def __init__(self, corpus, use_openai=True, model="text-embedding-3-small"):
        """corpus: list of dicts each with at least {label, price, part_family, size, ...}."""
        self.corpus = corpus
        self.model = model
        self.mode = "offline"
        self._client = None
        self._emb = None
        if use_openai and os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self._client = OpenAI()
                self.mode = "openai"
                self._emb = self._embed([_item_text(c) for c in corpus])
            except Exception as e:
                print("[analog] OpenAI unavailable, offline matcher:", e)

    def _embed(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            r = self._client.embeddings.create(model=self.model, input=texts[i:i+256])
            out.extend([np.asarray(d.embedding, np.float32) for d in r.data])
        return np.vstack(out) if out else np.zeros((0, 1), np.float32)

    def nearest(self, line, k=3):
        ld = line if isinstance(line, dict) else line.__dict__
        if self.mode == "openai" and self._emb is not None and len(self.corpus):
            q = self._embed([_item_text(ld)])[0]
            sims = self._emb @ q / (np.linalg.norm(self._emb, axis=1) * np.linalg.norm(q) + 1e-9)
            order = np.argsort(-sims)[:k]
            return [dict(self.corpus[i], score=float(sims[i])) for i in order]
        # offline: same family, nearest nominal size
        fam = str(ld.get("part_family", "")).lower()
        dia = _first_num(ld.get("size"))
        cands = [c for c in self.corpus if str(c.get("part_family", "")).lower() == fam]
        if not cands:
            return []
        cands.sort(key=lambda c: abs((_first_num(c.get("size")) or 0) - (dia or 0)))
        return [dict(c, score=None) for c in cands[:k]]


def _first_num(s):
    if not s:
        return None
    m = re.findall(r"[0-9]+\.?[0-9]*", str(s))
    return float(m[0]) if m else None
