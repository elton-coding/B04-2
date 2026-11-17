"""航迹锚定聚类：以 B 类航迹为几何锚点，把跨节点"发射"（同源 CPI 组）指派到航迹。

证据：
- 方位残差：发射在每个观测节点上的 DOA 与"节点→航迹插值位置"方位之差（多节点联合）；
- 型号一致：一条航迹（一个平台）只挂一部雷达 → 航迹型号由唯一匹配的发射投票确定；
- 领域/散射等级规则（D2）；
- 时序互斥：同一辐射源同一时刻只发射一个 CPI → 同一航迹上时间重叠且参数不同的发射至多保留一个。
无法锚定的发射（无航迹辐射源、歧义）交由节点内串接 + 融合处理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angdiff, bearing_deg


class DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r


def build_emissions(cp: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """同源 CPI 连通分量 → 发射号；约束每个发射在每个节点至多一个 CPI。"""
    pos = pd.Series(np.arange(len(cp)), index=cp.cpi.values)
    plat = cp.platform_id.values
    dsu = DSU(len(cp))
    members = {i: {plat[i]} for i in range(len(cp))}
    if len(pairs):
        ia = pos.loc[pairs.cpi_a.values].values
        ib = pos.loc[pairs.cpi_b.values].values
        dt = np.abs(cp.t0.values[ia] - cp.t0.values[ib])
        for k in np.argsort(dt, kind="stable"):
            ra, rb = dsu.find(ia[k]), dsu.find(ib[k])
            if ra == rb or members[ra] & members[rb]:
                continue
            dsu.p[rb] = ra
            members[ra] |= members.pop(rb)
    roots = np.array([dsu.find(i) for i in range(len(cp))])
    return pd.factorize(roots)[0]


def track_residuals(cp: pd.DataFrame, tracks: pd.DataFrame) -> tuple[np.ndarray, list]:
    """每个 CPI 对每条航迹的方位残差（度），航迹时间范围外为 NaN。"""
    tm = ((cp.t0 + cp.t1) / 2).values
    tids = sorted(tracks.track_id.unique())
    R = np.full((len(cp), len(tids)), np.nan)
    for j, tid in enumerate(tids):
        x = tracks[tracks.track_id == tid]
        tt = x.t_us.values
        lon = np.interp(tm, tt, x.target_longitude.values)
        lat = np.interp(tm, tt, x.target_latitude.values)
        r = angdiff(cp.doa.values, bearing_deg(cp.slon.values, cp.slat.values, lon, lat))
        inside = (tm >= tt[0] - 1e6) & (tm <= tt[-1] + 1e6)
        R[inside, j] = r[inside]
    return R, tids


def anchor_assign(cp: pd.DataFrame, proba: pd.DataFrame, emis: np.ndarray, tracks: pd.DataFrame,
                  knowledge: dict, cfg: dict) -> tuple[np.ndarray, dict]:
    """返回每个 CPI 的锚定航迹号（-1 表示未锚定）与航迹型号估计。"""
    ac = cfg["anchor"]
    if tracks is None or len(tracks) == 0:
        return np.full(len(cp), -1, np.int64), {}
    R, tids = track_residuals(cp, tracks)
    J = len(tids)
    air_node = cp.sh.values > 100
    sig = np.where(air_node, ac["sigma_air"], ac["sigma_sea"])
    gate = np.where(air_node, ac["gate_air"], ac["gate_sea"])
    Z2 = (R / sig[:, None]) ** 2
    OUT = np.abs(R) > gate[:, None]
    w = np.minimum(np.log2(cp.n.values + 1), 8.0)

    # 领域/散射等级可行性（按 CPI 自身型号概率加权）
    classes = list(proba.columns)
    models = knowledge.get("models", {})
    tinfo = tracks.groupby("track_id").agg(h=("target_height", "median"), rcs=("target_type", "first"))
    feas_ct = np.ones((len(classes), J), bool)
    for ci, c in enumerate(classes):
        m = models.get(c, {})
        for j, tid in enumerate(tids):
            if not m:
                continue
            ok = (m.get("domain") == "air") == (tinfo.h.loc[tid] > 50)
            rcs = tinfo.rcs.loc[tid]
            if isinstance(rcs, str) and rcs in ("small", "medium", "large"):
                ok &= rcs in m.get("rcs", [rcs])
            feas_ct[ci, j] = ok

    # 发射级聚合
    E = int(emis.max()) + 1
    df = pd.DataFrame({"e": emis, "w": w})
    lp = np.log(np.clip(proba.values, 1e-4, 1)) * w[:, None]
    LP = np.zeros((E, len(classes)))
    np.add.at(LP, emis, lp)
    LP -= LP.max(axis=1, keepdims=True)
    PE = np.exp(LP)
    PE /= PE.sum(axis=1, keepdims=True)
    valid = ~np.isnan(R)
    Zs = np.zeros((E, J)); Ws = np.zeros((E, J)); Out = np.zeros((E, J))
    np.add.at(Zs, emis, np.where(valid, Z2, 0) * w[:, None])
    np.add.at(Ws, emis, valid * w[:, None])
    np.add.at(Out, emis, (OUT & valid).astype(float))
    cost = np.where(Ws > 0, Zs / np.maximum(Ws, 1e-9), np.inf)
    geo_ok = (Ws > 0) & (Out == 0)
    # 型号可行：发射型号分布中可行型号的概率质量
    pfeas = PE @ feas_ct.astype(float)
    geo_ok &= pfeas >= 0.2

    # 1) 航迹型号：用几何上唯一可行的发射投票
    uniq = geo_ok.sum(axis=1) == 1
    TT = np.zeros((J, len(classes)))
    ew = np.bincount(emis, weights=cp.n.values, minlength=E)
    # 按脉冲数加权的硬投票：航迹上占主导的辐射源决定航迹型号（避免邻近小目标的大量短单元左右结果）
    for e in np.where(uniq)[0]:
        j = int(np.argmax(geo_ok[e]))
        TT[j, int(np.argmax(PE[e]))] += ew[e]
    support = np.bincount(np.argmax(geo_ok[uniq], axis=1), weights=ew[uniq], minlength=J) if uniq.any() else np.zeros(J)
    ttype = {}
    for j in range(J):
        if support[j] >= ac["min_track_support"]:
            ttype[j] = int(np.argmax(TT[j]))
    # 2) 指派：几何可行 + 与航迹型号一致；多候选取综合代价最小且需有足够间隔
    score = np.full((E, J), np.inf)
    for j, ci in ttype.items():
        pj = PE[:, ci]
        ok = geo_ok[:, j] & (pj >= ac["min_type_prob"])
        score[ok, j] = cost[ok, j] - 2.0 * np.log(np.clip(pj[ok], 1e-4, 1))
    best = np.argmin(score, axis=1)
    s_sorted = np.sort(score, axis=1)
    has = np.isfinite(s_sorted[:, 0])
    margin = s_sorted[:, 1] - s_sorted[:, 0] if J > 1 else np.full(E, np.inf)
    lab_e = np.where(has & (margin >= ac["min_margin"]), best, -1)
    amb_e = has & (margin < ac["min_margin"])
    lab = lab_e[emis]
    info = {"tids": tids, "ttype": {tids[j]: classes[c] for j, c in ttype.items()},
            "amb": amb_e[emis], "cand": score[emis] < np.inf, "score": score[emis], "R": R, "gate": gate}
    return lab, info


def resolve_conflicts(cp: pd.DataFrame, lab: np.ndarray, unit: np.ndarray, score: np.ndarray, cfg: dict) -> np.ndarray:
    """同一航迹标签内：不同单元的 CPI 在时间上重叠且参数不同 → 单元间冲突，去掉代价较大的单元。

    逐 CPI 比较（单元可跨多个 CPI，其均值参数无意义）。"""
    fc = cfg["fusion"]
    lab = lab.copy()
    sc_cpi = score[np.arange(len(cp)), np.clip(lab, 0, None)]
    usc = pd.Series(sc_cpi).groupby(unit).mean()
    pri = np.where(cp.jitter.values > 0, -1.0, cp.pri.fillna(-1).values)
    margin = fc["max_delay_us"]
    bad = {}
    for j in np.unique(lab[lab >= 0]):
        ii = np.where(lab == j)[0]
        ii = ii[np.argsort(cp.t0.values[ii])]
        t0, t1, rf, pr, u = cp.t0.values[ii], cp.t1.values[ii], cp.rf.values[ii], pri[ii], unit[ii]
        run_end, run_k = -np.inf, -1
        for k in range(len(ii)):
            if run_k >= 0 and t0[k] < run_end - margin and u[k] != u[run_k]:
                same = abs(rf[k] - rf[run_k]) <= fc["rf_tol"] and (abs(pr[k] - pr[run_k]) <= fc["pri_tol"] or pr[k] < 0 or pr[run_k] < 0)
                if not same:
                    pair = (min(u[k], u[run_k]), max(u[k], u[run_k]))
                    bad[pair] = bad.get(pair, 0) + 1
            if t1[k] > run_end:
                run_end, run_k = t1[k], k
    drop = set()
    for (ua, ub), c in sorted(bad.items(), key=lambda x: -x[1]):
        if ua in drop or ub in drop:
            continue
        drop.add(ua if usc.get(ua, 0) >= usc.get(ub, 0) else ub)
    if drop:
        lab[np.isin(unit, list(drop))] = -1
    return lab


def propagate_labels(cp: pd.DataFrame, lab: np.ndarray, frag: np.ndarray, proba: pd.DataFrame,
                     ttype: dict, tids: list, cfg: dict, R: np.ndarray | None = None,
                     gate: np.ndarray | None = None) -> np.ndarray:
    """沿节点内片段的时间序列，把锚定标签传播给相邻的未锚定 CPI（需时间近、方位连续、型号相容）。"""
    ac = cfg["anchor"]
    lab = lab.copy()
    cls_idx = {c: i for i, c in enumerate(proba.columns)}
    P = proba.values
    d = pd.DataFrame({"i": np.arange(len(cp)), "frag": frag, "t0": cp.t0.values, "t1": cp.t1.values,
                      "doa": cp.doa.values, "lab": lab}).sort_values(["frag", "t0"])
    for _, g in d.groupby("frag", sort=False):
        if (g.lab >= 0).all() or (g.lab < 0).all():
            continue
        L = g.lab.values.copy()
        t0, t1, doa, ii = g.t0.values, g.t1.values, g.doa.values, g.i.values
        for direction in (1, -1):
            rng = range(len(g)) if direction == 1 else range(len(g) - 1, -1, -1)
            last = -1
            for k in rng:
                if L[k] >= 0 and lab[ii[k]] >= 0:
                    last = k
                    continue
                if L[k] >= 0 or last < 0:
                    continue
                gap = (t0[k] - t1[last]) if direction == 1 else (t0[last] - t1[k])
                if gap > ac["prop_gap_s"] * 1e6 or abs(float(angdiff(doa[k], doa[last]))) > ac["prop_doa_deg"]:
                    continue
                if R is not None:
                    r = R[ii[k], L[last]]
                    if np.isfinite(r) and abs(r) > gate[ii[k]]:
                        continue
                tj = ttype.get(tids[L[last]])
                if tj is not None and P[ii[k], cls_idx[tj]] < ac["min_type_prob"]:
                    continue
                L[k] = L[last]
        lab[ii] = L
    return lab


def contiguity_links(cp: pd.DataFrame, tol: float = 0.05, amax: int = 3, bmax: int = 2) -> pd.DataFrame:
    """同节点 CPI 首尾相接：后一 CPI 首脉冲 = 前一 CPI 末脉冲 + a·PRI_A + b·PRI_B（a≥1，容忍漏脉冲）。

    同一雷达按 CPI 连续发射，相邻 CPI 间隔恰为 PRI（μs 级精确），训练集实测精度 99.99%。
    仅用于固定 PRI 的 CPI；每个 CPI 至多一个后继/前驱（取偏差最小者）。
    """
    out = []
    pri_all = np.where(cp.jitter.values > 0, np.nan, cp.pri.values)
    for _, idx in cp.groupby("platform_id").indices.items():
        idx = idx[np.isfinite(pri_all[idx]) & (cp.n.values[idx] >= 2)]
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(cp.t0.values[idx])]
        t0 = cp.t0.values[idx]
        t1 = cp.t1.values[idx]
        pa = pri_all[idx]
        best_dev = np.full(len(idx), np.inf)
        best_j = np.full(len(idx), -1)
        for a in range(1, amax + 1):
            for bb in range(0, bmax + 1):
                # 先按 a·PRI_A 定位候选，再用候选自身 PRI 校正 b·PRI_B
                tgt = t1 + a * pa
                lo = np.searchsorted(t0, tgt - 1.0)
                hi = np.searchsorted(t0, tgt + bb * 1500.0 + 1.0)
                for i in np.where(hi > lo)[0]:
                    js = np.arange(lo[i], hi[i])
                    dev = np.abs(t0[js] - (t1[i] + a * pa[i] + bb * pa[js]))
                    k = int(np.argmin(dev))
                    if dev[k] <= tol and dev[k] < best_dev[i] and js[k] != i:
                        best_dev[i], best_j[i] = dev[k], js[k]
        ok = best_j >= 0
        src, dst = idx[ok], idx[best_j[ok]]
        # 每个后继只保留一个前驱
        d = pd.DataFrame({"a": src, "b": dst, "dev": best_dev[ok]}).sort_values("dev").drop_duplicates("b")
        out.append(d[["a", "b"]])
    if not out:
        return pd.DataFrame(columns=["a", "b"])
    return pd.concat(out, ignore_index=True)


def build_groups(cp: pd.DataFrame, emis: np.ndarray, links: pd.DataFrame) -> np.ndarray:
    """发射（跨节点）∪ 首尾相接（节点内）的连通分量 → 纯度极高的"驻留组"。"""
    dsu = DSU(len(cp))
    first = {}
    for i, e in enumerate(emis):
        if e in first:
            dsu.p[dsu.find(i)] = dsu.find(first[e])
        else:
            first[e] = i
    for a_, b_ in zip(links.a.values, links.b.values):
        ra, rb = dsu.find(int(a_)), dsu.find(int(b_))
        if ra != rb:
            dsu.p[rb] = ra
    return pd.factorize(np.array([dsu.find(i) for i in range(len(cp))]))[0]
