"""对象级迭代精修：以驻留组为单元，按"对象型号似然 + 几何连续/航迹残差 + 时序互斥"重新指派。

对象成形后对象级型号与航迹已很可靠，用它们纠正单元级（尤其同频段相近型号、密集编队中）的错误归属。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angdiff


def _object_track(cp: pd.DataFrame, obj: np.ndarray) -> dict:
    """对象的锚定航迹：过半脉冲锚定于同一航迹时取之。"""
    out = {}
    if "anchor" not in cp.columns:
        return out
    d = pd.DataFrame({"o": obj, "j": cp.anchor.values, "n": cp.n.values})
    tot = d.groupby("o").n.sum()
    w = d[d.j >= 0].groupby(["o", "j"]).n.sum()
    for (o, j), v in w.items():
        if v >= 0.5 * tot.loc[o]:
            out[o] = int(j)
    return out


def refine_objects(cp: pd.DataFrame, obj: np.ndarray, unit: np.ndarray, proba: pd.DataFrame, cfg: dict,
                   R: np.ndarray | None = None, rgate: np.ndarray | None = None,
                   locked: np.ndarray | None = None) -> np.ndarray:
    rc = cfg.get("refine", {})
    fc, tc = cfg["fusion"], cfg["track"]
    obj = obj.copy()
    n = len(cp)
    ll = np.log(np.clip(proba.values, 1e-6, 1.0))
    ll = ll - ll.max(axis=1, keepdims=True)               # 每 CPI 相对最优型号的对数似然（≤0）
    w = np.minimum(np.log2(cp.n.values + 1), 8.0)
    classes = list(proba.columns)
    t0, t1, doa, rf = cp.t0.values, cp.t1.values, cp.doa.values, cp.rf.values
    pri = np.where(cp.jitter.values > 0, -1.0, cp.pri.fillna(-1).values)
    plat = cp.platform_id.values
    air_node = cp.sh.values > 100
    err = np.where(air_node, cfg["cpi"]["doa_err_air"], cfg["cpi"]["doa_err_sea"])
    sig = np.where(air_node, cfg["anchor"]["sigma_air"], cfg["anchor"]["sigma_sea"])
    margin = fc["max_delay_us"]
    maxdur = float((cp.t1 - cp.t0).max()) + 1.0
    U = int(unit.max()) + 1
    for it in range(rc.get("iters", 2)):
        objs = np.unique(obj)
        # 对象型号（对象级证据累积）与航迹
        lw = pd.DataFrame(ll * w[:, None]).groupby(obj).sum()
        otype = {o: int(np.argmax(lw.loc[o].values)) for o in objs}
        otrk = _object_track(cp, obj)
        S = np.full((U, len(objs)), -np.inf)
        # 单元自身各型号累计似然（相对 CPI 最优），用于型号硬门限
        LLu = np.zeros((U, ll.shape[1]))
        np.add.at(LLu, unit, ll * w[:, None])
        LLu_best = LLu.max(axis=1)
        for k, o in enumerate(objs):
            # 1) 型号项
            c_type = rc.get("type_w", 1.0) * ll[:, otype[o]] * w
            # 2) 几何项
            if R is not None and o in otrk:
                r = R[:, otrk[o]]
                z2 = np.where(np.isfinite(r), np.minimum((r / sig) ** 2, rc.get("geo_cap", 9.0)), rc.get("noinfo_pen", 2.0))
                c_geo = z2 * w
            else:
                c_geo = np.full(n, rc.get("noinfo_pen", 2.0)) * w
            conf = np.zeros(n)
            om = obj == o
            for p in np.unique(plat):
                oi = np.where(om & (plat == p))[0]
                ci = np.where(plat == p)[0]
                if len(oi) == 0:
                    continue
                oi = oi[np.argsort(t0[oi])]
                T0 = t0[oi]
                # 互斥：查找时间重叠的对象 CPI（不同单元、参数不同）
                lo = np.searchsorted(T0, t0[ci] - maxdur)
                hi = np.searchsorted(T0, t1[ci])
                for q in range(len(ci)):
                    c = ci[q]
                    for m in range(lo[q], hi[q]):
                        j = oi[m]
                        if unit[j] == unit[c]:
                            continue
                        if min(t1[j], t1[c]) - max(t0[j], t0[c]) > margin:
                            same = abs(rf[j] - rf[c]) <= fc["rf_tol"] and (abs(pri[j] - pri[c]) <= fc["pri_tol"] or pri[j] < 0 or pri[c] < 0)
                            if not same:
                                conf[c] = 1.0
                                break
                if not (R is not None and o in otrk):
                    # 无航迹对象：同节点时间最近（不同单元）CPI 的方位连续性
                    kk = np.clip(np.searchsorted(T0, t0[ci]), 1, len(oi) - 1) if len(oi) > 1 else np.zeros(len(ci), int)
                    best = np.full(len(ci), np.inf)
                    for cand in (kk - 1, kk) if len(oi) > 1 else (kk,):
                        j = oi[np.clip(cand, 0, len(oi) - 1)]
                        ok = unit[j] != unit[ci]
                        gap = np.abs(t0[j] - t0[ci]) / 1e6
                        tol = 2 * err[ci] + tc["doa_rate_air"] * gap
                        dev = (np.abs(angdiff(doa[ci], doa[j])) / tol) ** 2
                        dev = np.where(ok & (gap <= rc.get("cont_max_gap_s", 30.0)), np.minimum(dev * 4.0, rc.get("geo_cap", 9.0)), np.inf)
                        best = np.minimum(best, dev)
                    c_geo[ci] = np.where(np.isfinite(best), best, rc.get("noinfo_pen", 2.0)) * w[ci]
            score_c = c_type - c_geo - rc.get("conflict_pen", 30.0) * conf * w
            sk = np.bincount(unit, weights=score_c, minlength=U)
            # 型号硬门限：单元在对象型号下的似然比其最可能型号低太多则不可归入（杜绝跨型号/跨频段错归）
            sk[LLu[:, otype[o]] - LLu_best < -rc.get("type_gate", 15.0)] = -np.inf
            S[:, k] = sk
        cur = pd.Series(obj).groupby(unit).agg(lambda s: s.value_counts().index[0]).reindex(range(U)).values
        cur_k = np.searchsorted(objs, cur)
        S[np.arange(U), cur_k] = np.where(np.isfinite(S[np.arange(U), cur_k]), S[np.arange(U), cur_k], -1e9)
        best_k = np.argmax(S, axis=1)
        gain = S[np.arange(U), best_k] - S[np.arange(U), cur_k]
        move = (best_k != cur_k) & (gain > rc.get("min_gain", 2.0))
        if locked is not None and rc.get("lock_anchored", True):
            lu = np.bincount(unit, weights=locked.astype(float), minlength=U) > 0
            move &= ~lu
        if not move.any():
            break
        new_u = np.where(move, objs[best_k], cur)
        obj = new_u[unit]
    return pd.factorize(obj)[0]
