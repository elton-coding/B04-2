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
    P = proba.values
    col = {c: q for q, c in enumerate(proba.columns)}
    p_min = gc.get("type_p_min", 0.4)

    def compat(i, ty, ty_own):
        """型号相容：对象型号、CPI 判型，或 CPI 在该型号上的概率不低于 p_min（指纹相近型号的少脉冲 CPI 常判错）。"""
        return ty in (ty_own, ptype[i]) or (ty in col and P[i, col[ty]] >= p_min)

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
        best, bc = None, np.inf
        for y in otype.index:
            if y == obj[i] or not compat(i, otype[y], otype.get(obj[i])):
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
        frac_cache = {}
        assigned = np.array(sorted({j for _, j in tr_objs}), np.int64)
        orphan_cache = {}

        def favors(i, q, y):
            """CPI 判型明显偏向 X（X 型号概率比当前对象型号高出 type_margin）。"""
            tx, ty_ = otype2.get(tr_objs[q][0]), otype2.get(y)
            return tx in col and ty_ in col and P[i, col[tx]] >= P[i, col[ty_]] + gc.get("type_margin", 0.2)
        for i in cand_rows:
            y = obj2[i]
            xs = [q for q, (x, _) in enumerate(tr_objs) if x != y and fit[i, q]
                  and compat(i, otype2.get(x), otype2.get(y))]
            if not xs:
                continue
            if y not in otrk2 and gc.get("max_fit_frac", 0.4) < 1.0:
                # 当前对象（无航迹）在该节点上本就大量贴合 X 的航迹 → 此节点上几何分不开两者；除非 CPI 判型更偏向 X，否则不动
                xs2 = []
                for q in xs:
                    kf = (y, pid[i], q)
                    if kf not in frac_cache:
                        jy = np.where((obj2 == y) & (pid == pid[i]))[0]
                        frac_cache[kf] = float(n[jy][fit[jy, q]].sum() / max(n[jy].sum(), 1))
                    if frac_cache[kf] <= gc.get("max_fit_frac", 0.4) or favors(i, q, y):
                        xs2.append(q)
                xs = xs2
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
                # 同样紧贴 X 航迹的 CPI 不算"当前对象的方位连续邻居"（整串驻留一起被错归时互为邻居）
                # 脉冲数远少于本 CPI 的零散 CPI 也不算（常是别处错挂来的 1–2 脉冲碎片）
                if gc.get("nb_excl_fit", True):
                    nb = nb[(nb != i) & ~fit[nb][:, xs].any(axis=1) & (n[nb] >= gc.get("nb_min_frac", 0.25) * n[i])]
                else:
                    nb = nb[nb != i]
                if len(nb) and float(np.median(np.abs(angdiff(doa[nb], doa[i])))) <= 2.0 * derr[i]:
                    continue
            if y not in otrk2 and gc.get("local_s", 30.0) > 0 and not any(favors(i, q, y) for q in xs):
                # 当前对象在该 CPI 前后 [1 s, local_s] 内（去掉紧邻的同串驻留）过半也贴合 X 的航迹 → 它本身此时就在这条航迹的
                # 方位上，几何分不开；除非 CPI 判型明显偏向 X
                jj = idx_on.get((y, pid[i]))
                if jj is None:
                    jj = np.where((obj2 == y) & (pid == pid[i]))[0]
                    jj = jj[np.argsort(t0[jj], kind="stable")]
                    idx_on[(y, pid[i])] = jj
                dt = np.abs(t0[jj] - t0[i])
                lw = jj[(dt >= gc.get("local_excl_s", 1.0) * 1e6) & (dt <= gc["local_s"] * 1e6)]
                if len(lw):
                    fr = n[lw][fit[lw][:, xs].any(axis=1)].sum() / n[lw].sum()
                    if fr > gc.get("local_frac", 0.5):
                        continue
            q = min(xs, key=lambda q: Rt[i, q])
            ug = gc.get("uniq_gate", 1.0)
            if ug > 0:
                # 还有一条未分给任何对象的航迹也在 ug × 门限内，且当前对象整体大量贴合它 → 它可能正是当前对象自己
                # （未锚定上）的航迹，几何有歧义，不动
                ra = np.abs(np.nan_to_num(R[i], nan=99.0))
                ra[assigned] = 99.0
                amb = False
                for j in np.where(ra <= ug * gate[i])[0]:
                    # 只有当前对象整体也大量贴合这条孤立航迹时，它才可能是当前对象自己的航迹
                    kj = (y, j)
                    if kj not in orphan_cache:
                        jy = np.where(obj2 == y)[0]
                        ok = np.abs(np.nan_to_num(R[jy, j], nan=99.0)) <= gate[jy]
                        orphan_cache[kj] = float(n[jy][ok].sum() / max(n[jy].sum(), 1))
                    if orphan_cache[kj] >= gc.get("orphan_min_hit", 0.5):
                        amb = True
                        break
                if amb:
                    continue
            new[i] = tr_objs[q][0]
            moved += int(n[i])
    return (pd.factorize(new)[0] if moved else new), moved


def absorb_tiny_local(cp: pd.DataFrame, obj: np.ndarray, proba: pd.DataFrame, cfg: dict,
                      force: set | None = None) -> tuple[np.ndarray, int]:
    """节点上极小的局部簇（对象在该节点上的脉冲数 < min_pulses）通常是零散 CPI 被错挂到别处成形的对象上
    （训练集真值中单节点局部簇最小约 250 脉冲，而这类碎簇只有几到几十个脉冲、纯度很低）。逐 CPI 改归到
    同节点上型号相容、前后 win 内方位连续的对象（取方位偏差最小者）；无候选则不动。
    force：这些对象在所有节点上都按极小局部簇处理（如改归后瓦解的残余对象，内含多个辐射源的零散 CPI）。"""
    from .geo import angdiff
    from .recognize import vote_object_type

    tc = cfg.get("tiny_local", {})
    if not tc.get("enabled", True) or len(cp) == 0:
        return obj, 0
    n = cp.n.values
    pid = cp.platform_id.values
    t0 = cp.t0.values.astype(float)
    doa = cp.doa.values.astype(float)
    derr = np.where(cp.sh.values > 100, cfg["cpi"]["doa_err_air"], cfg["cpi"]["doa_err_sea"])
    size = pd.Series(n).groupby([obj, pid]).sum()
    tiny = size[(size < tc.get("min_pulses", 150)) | size.index.get_level_values(0).isin(list(force or ()))]
    if not len(tiny):
        return obj, 0
    otype = vote_object_type(cp.assign(object_id=obj), proba)
    ptype = cp.ptype.values if "ptype" in cp.columns else proba.idxmax(axis=1).values
    P = proba.values
    col = {c: q for q, c in enumerate(proba.columns)}
    p_min = tc.get("type_p_min", 0.2)
    win = tc.get("win_s", 5.0) * 1e6
    tiny_keys = set(tiny.index)
    new = obj.copy()
    moved = 0
    for p in np.unique([k[1] for k in tiny_keys]):
        on = np.where(pid == p)[0]
        hosts = {}
        for o_ in np.unique(obj[on]):
            if (o_, p) in tiny_keys:
                continue
            ii = on[obj[on] == o_]
            ii = ii[np.argsort(t0[ii], kind="stable")]
            hosts[o_] = (t0[ii], doa[ii])
        for i in on[np.array([(o_, p) in tiny_keys for o_ in obj[on]], bool)]:
            best, bc = None, np.inf
            for y, (tt, dd) in hosts.items():
                ty = otype.get(y)
                # 几个脉冲的 CPI 指纹判型不可靠，只看方位
                if n[i] >= tc.get("free_type_n", 8) and not (ty == ptype[i] or (ty in col and P[i, col[ty]] >= p_min)):
                    continue
                lo, hi = np.searchsorted(tt, t0[i] - win), np.searchsorted(tt, t0[i] + win)
                if hi <= lo:
                    continue
                dm = float(np.median(np.abs(angdiff(dd[lo:hi], doa[i]))))
                if dm <= 2.0 * derr[i] and dm < bc:
                    best, bc = y, dm
            if best is not None:
                new[i] = best
                moved += int(n[i])
    return (pd.factorize(new)[0] if moved else new), moved


def phase_repair(cp: pd.DataFrame, obj: np.ndarray, proba: pd.DataFrame, tracks: pd.DataFrame,
                 cfg: dict) -> tuple[np.ndarray, int]:
    """到达时刻相位改归（单节点测距证据，见 tdoa.toa_phase_residuals）：CPI 对当前对象航迹的相位残差 > bad_us，
    而对另一个有航迹对象 X 的航迹相位残差 ≤ fit_us、方位残差在单 CPI 门限内、型号相容 → 改归 X（X 唯一时）。

    方位分不开的同型号雷达对（两部 R01 / 两部 R05 方位相近时整串驻留错挂）主要靠它纠正。当前对象有航迹时
    要求对象型号相同（训练集验证：放宽到 CPI 判型会把编队中判型互错的 R07/R08 CPI 改坏）；当前对象无航迹时
    无从比较自身相位，按几何否决同样的型号相容规则，并要求该对象在此节点上整体不贴合 X 的相位与方位（否则
    两者此处不可分）。自检：直接锚定 CPI 对其航迹的相位命中率须明显高于随机（数据不具备公共时间格时自动关闭）。"""
    from .anchor import track_residuals
    from .recognize import vote_object_type
    from .tdoa import smooth_tracks, toa_phase_residuals

    pc = cfg.get("phase", {})
    if not pc.get("enabled", True) or tracks is None or not len(tracks) or len(cp) == 0:
        return obj, 0
    ac = cfg["anchor"]
    n = cp.n.values
    pid = cp.platform_id.values
    air = cp.sh.values > 100
    gate = np.where(air, ac["gate_air"], ac["gate_sea"])
    R, tids = track_residuals(cp, tracks)
    tsm = smooth_tracks(tracks, ac.get("track_smooth_s", 10.0), ac.get("track_smooth_air_s", 3.0))
    E = toa_phase_residuals(cp, tsm, tids, pc.get("lattice_us", 64.0))
    if pc.get("head_chain", True):
        # 抖动重频（R05/R06）驻留的前 1–2 个脉冲常被切成单独的 CPI，后面的主 CPI 起点不在时间格上：
        # 沿同节点首尾相接的 CPI 回溯到驻留起点，起点与自身任一贴合即算贴合
        h = dwell_heads(cp, pc.get("head_gap_mul", 2.6), pc.get("head_gap_us", 1700.0), pc.get("head_dur_ms", 30.0) * 1e3,
                        np.where(air, cfg["cpi"]["doa_err_air"], cfg["cpi"]["doa_err_sea"]))
        E = np.fmin(E, E[h])
    fit_us, bad_us = pc.get("fit_us", 0.6), pc.get("bad_us", 2.0)
    anchor = cp["anchor"].values if "anchor" in cp.columns else np.full(len(cp), -1)
    direct = cp["anchor_direct"].values if "anchor_direct" in cp.columns else anchor
    ok = direct >= 0
    if ok.sum() < pc.get("check_min_cpi", 50):
        return obj, 0
    e_dir = E[np.where(ok)[0], direct[ok]]
    e_dir = e_dir[np.isfinite(e_dir)]
    if not len(e_dir) or float(np.mean(e_dir <= fit_us)) < pc.get("check_min_hit", 0.5):
        return obj, 0
    # 逐航迹自检：航迹时间戳与 A 数据有偏移时（空中目标每秒偏移折合数百米），其相位预测失效，此航迹不参与
    ii, dj = np.where(ok)[0], direct[ok]
    rel = set()
    for j in np.unique(dj):
        e = E[ii[dj == j], j]
        e = e[np.isfinite(e)]
        if len(e) >= pc.get("track_min_cpi", 20) and float(np.mean(e <= fit_us)) >= pc.get("track_min_hit", 0.5):
            rel.add(int(j))
    otrk_all = _object_tracks(obj, anchor, n)
    otrk = {o: j for o, j in otrk_all.items() if j in rel}
    if not otrk:
        return obj, 0
    otype = vote_object_type(cp.assign(object_id=obj), proba)
    ptype = cp.ptype.values if "ptype" in cp.columns else proba.idxmax(axis=1).values
    P = proba.values
    col = {c: q for q, c in enumerate(proba.columns)}
    p_min = cfg.get("geo_veto", {}).get("type_p_min", 0.4)
    p_tight = pc.get("tight_p_min", 0.25)  # 当前对象无航迹且相位紧贴（≤ cross_fit_us）时的型号概率下限（R05/R06 共用模式约 0.33）
    tr_objs = list(otrk.items())
    js = np.array([j for _, j in tr_objs])
    Ex = np.nan_to_num(E[:, js], nan=99.0)
    Rx = np.abs(np.nan_to_num(R[:, js], nan=99.0))
    fit = (Ex <= fit_us) & (Rx <= gate[:, None])
    frac_cache = {}
    xr, xf, xb, xk = pc.get("cross_geo", False), pc.get("cross_fit_us", 0.2), pc.get("cross_bad_us", 1.0), pc.get("cross_ratio", 5.0)

    def cross_ok(i, q, tx, y):
        """跨型号改归（当前对象有航迹）：① 当前航迹方位也超门限（几何上已不可能属于当前对象），X 相位与方位都贴合；
        ② 或 CPI 在 X 型号上概率 ≥ p_min，且相位明显更贴合 X（≤ cross_fit_us，当前航迹 ≥ cross_bad_us 且 ≥ cross_ratio 倍）
        ——编队两航迹距离差仅 0.3–3 μs、指纹又互判错时靠它区分。"""
        jy = otrk[y]
        if xr and np.isfinite(R[i, jy]) and abs(R[i, jy]) > gate[i]:
            return True
        ey = E[i, jy]
        return (tx in col and P[i, col[tx]] >= p_min and np.isfinite(ey) and Ex[i, q] <= xf and ey >= xb
                and ey >= xk * max(Ex[i, q], 1e-3))

    new = obj.copy()
    moved = 0
    for i in np.where(fit.any(axis=1))[0]:
        y = obj[i]
        ty = otype.get(y)
        if y in otrk_all and y not in otrk:
            continue
        if y in otrk:
            ey = E[i, otrk[y]]
            if not (np.isfinite(ey) and ey > min(bad_us, xb)):
                continue
        xs = []
        for q, (x, _) in enumerate(tr_objs):
            if x == y or not fit[i, q]:
                continue
            tx = otype.get(x)
            if y in otrk:
                if tx != ty:
                    if not cross_ok(i, q, tx, y):
                        continue
                elif not E[i, otrk[y]] > bad_us:
                    continue
            elif not (tx in (ty, ptype[i]) or (tx in col and P[i, col[tx]] >= (
                    p_tight if Ex[i, q] <= xf else p_min))):
                continue
            xs.append(q)
        if len(xs) != 1:
            continue
        q = xs[0]
        if y not in otrk:
            kf = (y, pid[i], q)
            if kf not in frac_cache:
                jy = np.where((obj == y) & (pid == pid[i]))[0]
                frac_cache[kf] = float(n[jy][fit[jy, q]].sum() / max(n[jy].sum(), 1))
            if frac_cache[kf] > pc.get("max_fit_frac", 0.4):
                continue
        new[i] = tr_objs[q][0]
        moved += int(n[i])
    return (pd.factorize(new)[0] if moved else new), moved


@numba.njit(cache=True)
def _dwell_heads(order, t0, t1, pid, ty, doa, derr, pri, mul, gap, dur, jit, npl):
    n = len(order)
    h = np.arange(len(t0))
    for q in range(n):
        i = order[q]
        if not jit[i]:
            continue
        best, bt = -1, -1e300
        for r in range(q - 1, max(q - 40, -1), -1):
            j = order[r]
            if pid[j] != pid[i] or t0[i] - t0[j] > dur:
                break
            d = abs((doa[j] - doa[i] + 180.0) % 360.0 - 180.0)
            if t1[j] < t0[i] and t1[j] > bt and (jit[j] or npl[j] <= 2) and (ty[j] == ty[i] or d <= 2.0 * derr[i]):
                bt, best = t1[j], j
        if best >= 0 and t0[i] - bt <= min(mul * pri[i], gap) and t0[i] - t0[h[best]] <= dur:
            h[i] = h[best]
    return h


def dwell_heads(cp: pd.DataFrame, mul: float, gap: float, dur: float, derr: np.ndarray) -> np.ndarray:
    """每个抖动重频 CPI 所在驻留的起始 CPI（行号；非抖动 CPI 为自身——编队 R07/R08 相邻驻留首尾相接、方位又相近，
    回溯会串到另一部雷达）：同节点上前一个（抖动或 ≤ 2 脉冲的）首尾间隔 ≤ min(mul × PRI, gap) 、型号相同或方位连续的
    CPI 视为同一驻留的前段，递推到起点（驻留总长 ≤ dur）。"""
    pid = cp.platform_id.values.astype(np.int64)
    t0 = cp.t0.values.astype(np.float64)
    order = np.lexsort((t0, pid)).astype(np.int64)
    ty = pd.factorize(cp.ptype.values)[0].astype(np.int64) if "ptype" in cp.columns else np.zeros(len(cp), np.int64)
    pri = np.nan_to_num(cp.pri.values.astype(np.float64), nan=gap / max(mul, 1e-9))
    return _dwell_heads(order, t0, cp.t1.values.astype(np.float64), pid, ty, cp.doa.values.astype(np.float64),
                        np.asarray(derr, np.float64), pri, float(mul), float(gap), float(dur),
                        cp.jitter.values > 0, cp.n.values.astype(np.int64))
