"""节点内时间互斥重分配。

物理事实：一部雷达同一时刻只发射一串脉冲，因此同一辐射源在同一接收节点上的 CPI 在时间上互不重叠
（频率相同的重叠除外：抖动重频脉冲串偶被串接成交错的两半，训练集真值下约 3%，两半频率一致）。
当若干同频段、同型号族的辐射源都以高占空比持续照射某节点时，一个 CPI 只能嵌进其中某一个辐射源的
发射间隙：若它与对象 o 的 CPI 不重叠、而 o 在其附近的占空比为 b，则"属于 o"相对"与 o 独立"的
似然比约为 1/(1-b)；若与 o 重叠（频率不同），则几乎不可能属于 o。

以驻留组（单元）为粒度累加上述证据，满足型号门限与航迹几何门限时把单元改归到证据最强的对象。
只在占空比高、证据强时才会改归；低占空比节点上证据自然趋零，不影响其他场景。
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd


@numba.njit(cache=True)
def _overlap_pairs(t0, t1, rf, margin, rf_tol):
    """t0 已升序。返回时间重叠超过门限的 CPI 对（局部下标）及是否同频。"""
    n = len(t0)
    cap = 1024
    oi = np.empty(cap, np.int64)
    oj = np.empty(cap, np.int64)
    sf = np.empty(cap, np.bool_)
    m = 0
    for k in range(n):
        dk = t1[k] - t0[k]
        if dk <= 0:
            continue
        j = k + 1
        while j < n and t0[j] < t1[k]:
            dj = t1[j] - t0[j]
            if dj > 0:
                g = min(margin, 0.5 * min(dk, dj))
                if min(t1[j], t1[k]) - t0[j] > g:
                    if m == cap:
                        cap *= 2
                        a = np.empty(cap, np.int64)
                        b = np.empty(cap, np.int64)
                        c = np.empty(cap, np.bool_)
                        a[:m] = oi[:m]
                        b[:m] = oj[:m]
                        c[:m] = sf[:m]
                        oi, oj, sf = a, b, c
                    oi[m] = k
                    oj[m] = j
                    sf[m] = abs(rf[j] - rf[k]) <= rf_tol
                    m += 1
            j += 1
    return oi[:m], oj[:m], sf[:m]


def node_overlap_pairs(cp: pd.DataFrame, margin: float, rf_tol: float):
    """全体 CPI 中同节点、时间重叠的对（全局行号）及是否同频。"""
    PI, PJ, SF = [], [], []
    t0, t1, rf = cp.t0.values.astype(float), cp.t1.values.astype(float), cp.rf.values.astype(float)
    for pid in np.unique(cp.platform_id.values):
        idx = np.where(cp.platform_id.values == pid)[0]
        idx = idx[np.argsort(t0[idx], kind="stable")]
        a, b, f = _overlap_pairs(t0[idx], t1[idx], rf[idx], margin, rf_tol)
        PI.append(idx[a])
        PJ.append(idx[b])
        SF.append(f)
    return np.concatenate(PI), np.concatenate(PJ), np.concatenate(SF)


def _window_sum(key_sorted, csum, q_lo, q_hi):
    return csum[np.searchsorted(key_sorted, q_hi, "right")] - csum[np.searchsorted(key_sorted, q_lo, "left")]


def _unit_mode(unit: np.ndarray, obj: np.ndarray, U: int) -> np.ndarray:
    """每个单元内 CPI 数最多的对象。"""
    cnt = pd.DataFrame({"u": unit, "o": obj}).value_counts().reset_index(name="c")
    cnt = cnt.sort_values(["u", "c"], ascending=[True, False]).drop_duplicates("u")
    out = np.full(U, -1, np.int64)
    out[cnt.u.values] = cnt.o.values
    return out


def exclusion_repair(cp: pd.DataFrame, obj: np.ndarray, unit: np.ndarray, proba: pd.DataFrame, cfg: dict,
                     R: np.ndarray | None = None, rgate: np.ndarray | None = None, debug: dict | None = None) -> tuple[np.ndarray, int]:
    ec = cfg.get("exclusion", {})
    obj = obj.copy()
    n = len(cp)
    if n == 0:
        return obj
    margin = ec.get("margin_us", 1000.0)
    W = ec.get("busy_win_s", 1.0) * 1e6
    bmax = ec.get("busy_max", 0.95)
    bmin = ec.get("busy_min", 0.5)
    pen = ec.get("overlap_pen", 3.0)
    min_gain = ec.get("min_gain", 6.0)
    rc = cfg.get("refine", {})
    t0, t1 = cp.t0.values.astype(float), cp.t1.values.astype(float)
    dur = np.maximum(t1 - t0, 0.0)
    mid = 0.5 * (t0 + t1)
    w = np.minimum(np.log2(cp.n.values + 1), 8.0)
    PI, PJ, SF = node_overlap_pairs(cp, margin, cfg["fusion"]["rf_tol"])
    DI, DJ, QI, QJ = PI[~SF], PJ[~SF], PI[SF], PJ[SF]
    nodes, node_i = np.unique(cp.platform_id.values, return_inverse=True)
    U = int(unit.max()) + 1
    ll = np.log(np.clip(proba.values, 1e-6, 1.0))
    ll = ll - ll.max(axis=1, keepdims=True)
    LLu = np.zeros((U, ll.shape[1]))
    np.add.at(LLu, unit, ll * w[:, None])
    LLu_best = LLu.max(axis=1)
    Wu = np.bincount(unit, weights=w, minlength=U)
    gate_mean = rc.get("type_gate_mean", 4.0)
    BIG = 1e12
    # 单元自身在窗口内的占用（计算当前对象占空比时扣除，避免单元自证）
    ukey = (node_i.astype(float) * U + unit) * BIG + t0
    uo = np.argsort(ukey, kind="stable")
    ucs = np.r_[0.0, np.cumsum(dur[uo])]
    ubase = (node_i.astype(float) * U + unit) * BIG
    own = _window_sum(ukey[uo], ucs, ubase + mid - W, ubase + mid + W) / (2 * W)
    anchor = cp["anchor"].values if "anchor" in cp.columns else np.full(n, -1)
    direct = cp["anchor_direct"].values if "anchor_direct" in cp.columns else np.full(n, -1)
    total_moved = 0
    for _ in range(int(ec.get("iters", 4))):
        objs = np.unique(obj)
        K = len(objs)
        ok_ = np.searchsorted(objs, obj)
        # 重叠指示：CPI i 与对象 k 的（其他）CPI 重叠；频率不同 → 冲突，频率相同 → 中性（可能是同一串被拆成两半）
        C = np.zeros((n, K), bool)
        C[DI, ok_[DJ]] = True
        C[DJ, ok_[DI]] = True
        Q = np.zeros((n, K), bool)
        Q[QI, ok_[QJ]] = True
        Q[QJ, ok_[QI]] = True
        # 占空比：对象 k 在 CPI 所在节点 ±W 内的 CPI 时长占比
        okey = (node_i.astype(float) * K + ok_) * BIG + t0
        oo = np.argsort(okey, kind="stable")
        ocs = np.r_[0.0, np.cumsum(dur[oo])]
        ks = okey[oo]
        B = np.empty((n, K))
        for k in range(K):
            base = (node_i.astype(float) * K + k) * BIG
            B[:, k] = _window_sum(ks, ocs, base + mid - W, base + mid + W) / (2 * W)
        rows = np.arange(n)
        B[rows, ok_] = B[rows, ok_] - own
        B = np.clip(B, 0.0, bmax)
        fit = np.where(B >= bmin, -np.log1p(-B), 0.0)  # 低占空比时间隙多且有结构，不重叠不构成证据
        s = np.where(C, -pen, np.where(Q, 0.0, fit)) * w[:, None]
        Su = np.zeros((U, K))
        np.add.at(Su, unit, s)
        # 型号门限（与精修一致：按单元内 CPI 加权平均）
        lw = pd.DataFrame(ll * w[:, None]).groupby(obj).sum()
        otype = np.array([int(np.argmax(lw.loc[o].values)) for o in objs])
        dLL = LLu[:, otype] - LLu_best[:, None]
        Su[dLL < -gate_mean * Wu[:, None]] = -np.inf
        # 对象的航迹：过半脉冲锚定于同一航迹
        d = pd.DataFrame({"o": obj, "j": anchor, "n": cp.n.values})
        tot = d.groupby("o").n.sum()
        wj = d[d.j >= 0].groupby(["o", "j"]).n.sum()
        otrk = np.full(K, -1)
        for (o, j), v in wj.items():
            if v >= 0.5 * tot.loc[o]:
                otrk[int(np.searchsorted(objs, o))] = int(j)
        # 锁定：单元过半直接锚定于其当前对象的航迹（航迹几何/TDOA 已确认），不因时序证据改归
        on_trk = (direct >= 0) & (direct == otrk[ok_])
        locked = np.bincount(unit, weights=on_trk * w, minlength=U) > 0.5 * Wu
        # 几何门限：目标对象锚定于某航迹时，单元在该航迹上有残差的 CPI 须过半落在门限内
        if R is not None and rgate is not None:
            for k in np.where(otrk >= 0)[0]:
                j = otrk[k]
                r = np.abs(R[:, int(j)])
                fin = np.isfinite(r)
                inside = fin & (r <= rgate)
                nf = np.bincount(unit, weights=fin * w, minlength=U)
                ni = np.bincount(unit, weights=inside * w, minlength=U)
                bad = (nf > 0) & (ni < 0.5 * nf)
                Su[bad, k] = -np.inf
        cur = _unit_mode(unit, obj, U)
        cur_k = np.searchsorted(objs, cur)
        sc = Su[np.arange(U), cur_k]
        sc = np.where(np.isfinite(sc), sc, -1e9)
        best = np.argmax(Su, axis=1)
        sb = Su[np.arange(U), best]
        move = (best != cur_k) & (sb > 0) & (sb - sc > min_gain) & ~locked
        if debug is not None:
            debug.update(Su=Su, objs=objs, cur_k=cur_k, s=s, C=C, B=B)
        if not move.any():
            break
        total_moved += int(np.bincount(unit, weights=cp.n.values, minlength=U)[move].sum())
        new_u = np.where(move, objs[best], cur)
        obj = new_u[unit]
    return pd.factorize(obj)[0], total_moved


def _object_tracks(obj: np.ndarray, anchor: np.ndarray, n: np.ndarray) -> dict:
    """对象 → 过半脉冲锚定的航迹下标（无则不在字典中）。"""
    d = pd.DataFrame({"o": obj, "j": anchor, "n": n})
    tot = d.groupby("o").n.sum()
    wj = d[d.j >= 0].groupby(["o", "j"]).n.sum()
    return {int(o): int(j) for (o, j), v in wj.items() if v >= 0.5 * tot.loc[o]}


def _busy(t0: np.ndarray, t1: np.ndarray, lo: float, hi: float) -> float:
    return float(np.clip(np.minimum(t1, hi) - np.maximum(t0, lo), 0.0, None).sum() / (hi - lo))


def episode_repair(cp: pd.DataFrame, obj: np.ndarray, unit: np.ndarray, proba: pd.DataFrame, cfg: dict,
                   R: np.ndarray | None = None, rgate: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """整段补洞（见配置 exclusion.episode）。返回 (obj, 改归脉冲数)。

    片段内逐驻留组判定：组的工作模式（脉宽、重频）更接近 Y 两侧的 CPI 才改归，X 自身穿插其中的波束留在 X。
    """
    ec = cfg.get("exclusion", {})
    obj = obj.copy()
    if len(cp) == 0:
        return obj, 0
    G = ec.get("ep_gap_s", 0.1) * 1e6
    span_min = ec.get("ep_min_span_s", 0.6) * 1e6
    bmin_ep = ec.get("ep_min_busy", 0.5)
    F = ec.get("ep_flank_s", 2.0) * 1e6
    fmin = ec.get("ep_flank_min", 0.4)
    hmax = ec.get("ep_hole_max", 0.2)
    t0, t1 = cp.t0.values.astype(float), cp.t1.values.astype(float)
    n = cp.n.values
    pid = cp.platform_id.values
    pw = cp.pw.values.astype(float)
    lpri = np.log(np.maximum(cp.pri.values.astype(float), 1e-3))
    w = np.minimum(np.log2(n + 1), 8.0)
    ll = np.log(np.clip(proba.values, 1e-6, 1.0))
    otype = pd.DataFrame(ll * w[:, None]).groupby(obj).sum().idxmax(axis=1).to_dict()
    anchor = cp["anchor"].values if "anchor" in cp.columns else np.full(len(cp), -1)
    direct = cp["anchor_direct"].values if "anchor_direct" in cp.columns else np.full(len(cp), -1)
    otrk = _object_tracks(obj, anchor, n)
    obj0 = obj.copy()
    moved = 0

    def mode_dist(i_u, i_ref):
        if len(i_ref) == 0:
            return np.inf
        d = np.abs(pw[i_ref] - np.median(pw[i_u])) / 0.1 + np.abs(lpri[i_ref] - np.median(lpri[i_u])) / 0.1
        return float(np.min(d))

    for p in np.unique(pid):
        on = np.where(pid == p)[0]
        objs_p = np.unique(obj0[on])
        idx_of = {o: on[obj0[on] == o] for o in objs_p}
        for X in objs_p:
            ix = idx_of[X]
            if len(ix) < 5:
                continue
            ix = ix[np.argsort(t0[ix], kind="stable")]
            ends = np.maximum.accumulate(t1[ix])
            eid = np.cumsum(np.r_[True, t0[ix][1:] - ends[:-1] > G]) - 1
            for e in np.unique(eid):
                m = ix[eid == e]
                lo, hi = t0[m].min(), t1[m].max()
                if hi - lo < span_min or _busy(t0[m], t1[m], lo, hi) < bmin_ep:
                    continue
                xo = ix[eid != e]
                if (_busy(t0[xo], t1[xo], lo - F, lo) + _busy(t0[xo], t1[xo], hi, hi + F)) / 2 >= fmin:
                    continue
                for Y in objs_p:
                    if Y == X or otype.get(Y) != otype.get(X):
                        continue
                    iy = idx_of[Y]
                    if min(_busy(t0[iy], t1[iy], lo - F, lo), _busy(t0[iy], t1[iy], hi, hi + F)) < fmin \
                            or _busy(t0[iy], t1[iy], lo, hi) > hmax:
                        continue
                    yf = iy[((t1[iy] > lo - F) & (t0[iy] < lo)) | ((t0[iy] < hi + F) & (t1[iy] > hi))]
                    xf = xo[((t1[xo] > lo - F) & (t0[xo] < lo)) | ((t0[xo] < hi + F) & (t1[xo] > hi))]
                    jy = otrk.get(int(Y), -1)
                    jx = otrk.get(int(X), -1)
                    for u in np.unique(unit[m]):
                        iu = m[unit[m] == u]
                        if jx >= 0 and (direct[iu] == jx).sum() * 2 > len(iu):  # 直接锚定于 X 的航迹：锁定
                            continue
                        if jy >= 0 and R is not None and rgate is not None:
                            r = np.abs(R[iu, jy])
                            fin = np.isfinite(r)
                            if fin.any() and (r[fin] <= rgate[iu][fin]).mean() < 0.5:
                                continue
                        if mode_dist(iu, yf) < mode_dist(iu, xf):
                            obj[iu] = Y
                            moved += int(n[iu].sum())
                    break
    return (pd.factorize(obj)[0] if moved else obj), moved


def dissolved_remnants(before: np.ndarray, after: np.ndarray, n: np.ndarray, frac: float) -> set:
    """重分配后的对象中，主体来自某原对象、但只剩该原对象不足 (1-frac) 脉冲的残余对象编号。"""
    d = pd.DataFrame({"b": before, "a": after, "n": n})
    size_b = d.groupby("b").n.sum()
    ab = d.groupby(["a", "b"]).n.sum()
    out = set()
    for a_, g in ab.groupby(level=0):
        b_ = g.droplevel(0).idxmax()
        if g.sum() < (1.0 - frac) * size_b.loc[b_]:
            out.add(int(a_))
    return out


def geometry_veto_repair(cp: pd.DataFrame, obj: np.ndarray, proba: pd.DataFrame, tracks: pd.DataFrame,
                         cfg: dict) -> tuple[np.ndarray, int]:
    """几何否决改归：对象锚定于某航迹时，方位残差超过 k 倍单 CPI 门限的 CPI 不可能来自该对象
    （测向误差有界：海面 ±3°、空中 ±1.5°，CPI 级平均后更小）。常见来源是同参数的两部同型号雷达被跨节点
    配成同一发射、或传播标签越过了无航迹目标。改归到型号相容（对象型号或 CPI 自身判型）且几何一致的对象：
    有航迹者残差在门限内，无航迹者与其同节点前后 win 内的 CPI 方位连续；取代价最小者，无候选则不动。"""
    from .anchor import track_residuals
    from .geo import angdiff
    from .recognize import vote_object_type

    gc = cfg.get("geo_veto", {})
    if not gc.get("enabled", True) or tracks is None or not len(tracks) or len(cp) == 0:
        return obj, 0
    ac = cfg["anchor"]
    obj = obj.copy()
    n = cp.n.values
    pid = cp.platform_id.values
    t0 = cp.t0.values.astype(float)
    doa = cp.doa.values.astype(float)
    air = cp.sh.values > 100
    gate = np.where(air, ac["gate_air"], ac["gate_sea"])
    derr = np.where(air, cfg["cpi"]["doa_err_air"], cfg["cpi"]["doa_err_sea"])
    R, tids = track_residuals(cp, tracks)
    anchor = cp["anchor"].values if "anchor" in cp.columns else np.full(len(cp), -1)
    otrk = _object_tracks(obj, anchor, n)
    r_own = np.full(len(cp), np.nan)
    for o_, j in otrk.items():
        m = obj == o_
        r_own[m] = np.abs(R[m, j])
    flag = np.nan_to_num(r_own, nan=0.0) > gc.get("k", 2.0) * gate
    if not flag.any():
        return obj, 0
    otype = vote_object_type(cp.assign(object_id=obj), proba)
    ptype = cp.ptype.values if "ptype" in cp.columns else proba.idxmax(axis=1).values
    win = gc.get("win_s", 5.0) * 1e6
    # 无航迹候选对象逐节点的有序 (t0, doa)，不含被否决的 CPI
    keep = ~flag
    nodes = {}
    for o_ in np.unique(obj):
        if o_ in otrk:
            continue
        for p in np.unique(pid[(obj == o_) & keep]):
            ii = np.where((obj == o_) & keep & (pid == p))[0]
            ii = ii[np.argsort(t0[ii], kind="stable")]
            nodes[(o_, p)] = (t0[ii], doa[ii])
    moved = 0
    new = obj.copy()
    for i in np.where(flag)[0]:
        types = {otype.get(obj[i]), ptype[i]}
        best, bc = None, np.inf
        for y in otype.index:
            if y == obj[i] or otype[y] not in types:
                continue
            if y in otrk:
                r = abs(R[i, otrk[y]])
                if not np.isfinite(r) or r > gate[i]:
                    continue
                c = r / gate[i]
            else:
                key = (y, pid[i])
                if key not in nodes:
                    continue
                tt, dd = nodes[key]
                lo, hi = np.searchsorted(tt, t0[i] - win), np.searchsorted(tt, t0[i] + win)
                if hi <= lo:
                    continue
                dm = float(np.median(np.abs(angdiff(dd[lo:hi], doa[i]))))
                if dm > 2.0 * derr[i]:
                    continue
                c = dm / (2.0 * derr[i])
            if c < bc:
                best, bc = y, c
        if best is not None:
            new[i] = best
            moved += int(n[i])
    # 第二步（反向）：CPI 紧贴某个型号相容的有航迹对象 X 的航迹（残差 ≤ k2 × 门限），而与当前对象不一致
    # （当前对象有航迹时残差超门限；无航迹时同节点前后 win 内没有方位连续的 CPI）→ 改归 X
    k2 = gc.get("k_fit", 0.5)
    obj2 = new
    otrk2 = _object_tracks(obj2, anchor, n)
    otype2 = vote_object_type(cp.assign(object_id=obj2), proba)
    tr_objs = [(x, j) for x, j in otrk2.items()]
    if tr_objs:
        Rt = np.abs(np.stack([R[:, j] for _, j in tr_objs], axis=1))
        fit = np.nan_to_num(Rt, nan=99.0) <= k2 * gate[:, None]
        cand_rows = np.where(fit.any(axis=1))[0]
        idx_on = {}
        for i in cand_rows:
            y = obj2[i]
            xs = [q for q, (x, _) in enumerate(tr_objs) if x != y and fit[i, q]
                  and otype2.get(x) in (otype2.get(y), ptype[i])]
            if not xs:
                continue
            if y in otrk2:
                ry = abs(R[i, otrk2[y]])
                if not (np.isfinite(ry) and ry > gate[i]):
                    continue
            else:
                key = (y, pid[i])
                if key not in idx_on:
                    jj = np.where((obj2 == y) & (pid == pid[i]))[0]
                    jj = jj[np.argsort(t0[jj], kind="stable")]
                    idx_on[key] = jj
                jj = idx_on[key]
                lo, hi = np.searchsorted(t0[jj], t0[i] - win), np.searchsorted(t0[jj], t0[i] + win)
                nb = jj[lo:hi]
                nb = nb[nb != i]
                if len(nb) and float(np.median(np.abs(angdiff(doa[nb], doa[i])))) <= 2.0 * derr[i]:
                    continue
            q = min(xs, key=lambda q: Rt[i, q])
            new[i] = tr_objs[q][0]
            moved += int(n[i])
    return (pd.factorize(new)[0] if moved else new), moved
