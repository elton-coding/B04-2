"""型号波形指纹：生成式朴素贝叶斯（每型号 RF/PW/PRI 的核密度 + 抖动标志），无类别先验。

D2 规则：同型号波形定义相同，同类两型号参数部分重叠 → 落在重叠区的 CPI 似然近似相等，
自动降权；对象级型号 = 各 CPI 对数似然之和的最大者（多 CPI 证据累积）。
判别式分类器会把训练场景的类别比例当作先验，换场景（B 榜）时有偏，故此处采用似然。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

FEATS = {  # name: (lo, hi, step, sigma)
    "rf": (27500.0, 37000.0, 0.25, 1.0),
    "pw": (0.0, 4.0, 0.01, 0.03),
    "pri": (0.0, 1500.0, 0.1, 0.5),
}
FLOOR = 1e-7


def _feat(cp: pd.DataFrame, name: str) -> np.ndarray:
    if name == "pri":
        return cp.pri.fillna(-1.0).values
    return cp[name].values


def fit_fingerprint(cp: pd.DataFrame, y: pd.Series, min_n: int = 8) -> dict:
    m = (cp.n.values >= min_n) & y.notna().values
    model = {"classes": sorted(y[m].unique()), "feats": FEATS, "tables": {}, "jit": {}}
    for c in model["classes"]:
        sel = m & (y.values == c)
        tabs = {}
        for name, (lo, hi, step, sig) in FEATS.items():
            v = _feat(cp, name)[sel]
            v = v[(v >= lo) & (v < hi)]
            nb = int(round((hi - lo) / step))
            h = np.bincount(((v - lo) / step).astype(np.int64), minlength=nb)[:nb].astype(float)
            h = gaussian_filter1d(h, sig / step, mode="constant")
            dens = h / max(h.sum(), 1e-12) / step
            tabs[name] = np.log(dens + FLOOR).astype(np.float32)
        model["tables"][c] = tabs
        j = cp.jitter.values[sel]
        model["jit"][c] = (j.sum() + 1.0) / (len(j) + 2.0)
    return model


def loglik(model: dict, cp: pd.DataFrame) -> pd.DataFrame:
    out = np.zeros((len(cp), len(model["classes"])))
    for name, (lo, hi, step, _) in model["feats"].items():
        v = _feat(cp, name)
        nb = int(round((hi - lo) / step))
        idx = np.clip(((v - lo) / step).astype(np.int64), 0, nb - 1)
        bad = (v < lo) | (v >= hi) | ~np.isfinite(v)
        for k, c in enumerate(model["classes"]):
            ll = model["tables"][c][name][idx].astype(float)
            ll[bad] = np.log(FLOOR)
            out[:, k] += ll
    jit = cp.jitter.values.astype(bool)
    for k, c in enumerate(model["classes"]):
        p = model["jit"][c]
        out[:, k] += np.where(jit, np.log(p), np.log(1 - p))
    return pd.DataFrame(out, index=cp.index, columns=model["classes"])


def proba_from_ll(ll: pd.DataFrame, n: np.ndarray, cap: float = 12.0) -> pd.DataFrame:
    """CPI 级后验（等先验）；单 CPI 对数似然差截断，防止个别异常 CPI 主导对象投票。"""
    z = ll.values - ll.values.max(axis=1, keepdims=True)
    z = np.maximum(z, -cap)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    small = n < 4  # 过短 CPI 参数估计不可靠：向均匀分布收缩
    p[small] = 0.5 * p[small] + 0.5 / p.shape[1]
    return pd.DataFrame(p, index=ll.index, columns=ll.columns)
