"""L3 节点内 CPI 串接（局部辐射源片段）与 L6 跨平台融合。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from .geo import angdiff


# ---------------------------------------------------------------- 节点内片段
def build_fragments(cp: pd.DataFrame, proba: pd.DataFrame, domain_air: dict, cfg: dict,
                    phase: tuple | None = None) -> np.ndarray:
    """逐节点按时间顺序做 CPI→辐射源的门限关联，返回每个 CPI 的片段号。

    phase=(主相位, 备选相位, 门限 μs, 漂移率 μs/s, 最大有效门限 μs, 格长 μs)：驻留起点 mod 64 μs 的相位（与 CPI 行对齐，NaN 不用）。
    同一辐射源在同一节点上相位只随距离缓慢漂移，不同辐射源相位随机；方位/型号门限内有多个候选时，
    只在相位延续的候选中选（无一延续则照旧）。
    """
    tc = cfg["track"]
    doa_err = {True: cfg["cpi"]["doa_err_air"], False: cfg["cpi"]["doa_err_sea"]}
    classes = list(proba.columns)
    air_mask = np.array([domain_air.get(c, True) for c in classes])
    P = proba.values
    frag = np.full(len(cp), -1, np.int64)
    nxt = 0
    K = P.shape[1]
    max_gap, eps, min_compat = tc["max_gap_s"], tc["overlap_eps_us"], tc["min_type_compat"]
    r_air, r_sea = tc["doa_rate_air"], tc["doa_rate_sea"]
    if phase is not None:
        ph1, ph2, p_fit, p_rate, p_max, L = phase
        p_veto_k = int(tc.get("frag_phase_veto_k", 0))
        p_veto_us = float(tc.get("frag_phase_veto_us", 2.0))
        p_veto_n = int(tc.get("frag_phase_veto_n", 8))
        p_rate_unc = float(tc.get("frag_phase_rate_unc", 0.05))
        p_acc = float(tc.get("frag_phase_acc", 0.01))
        p_rw_min = float(tc.get("frag_phase_rw_min", 3.0))
        p_rw_cap = float(tc.get("frag_phase_rw_cap", 30.0))
        ph1, ph2 = np.asarray(ph1, float), np.asarray(ph2, float)
    for pid, idx in cp.groupby("platform_id", sort=True).indices.items():
        idx = idx[np.argsort(cp.t0.values[idx], kind="stable")]
        node_air = float(np.nanmedian(cp.sh.values[idx])) > 100
        e = doa_err[node_air]
        # 存活轨迹以数组保存（顺序与逐条遍历一致：过期剔除保持次序、新轨迹追加在末尾），每个 CPI 对全部轨迹向量化门限判断。
        # 杂散脉冲会形成大量单脉冲 CPI 并各自开轨迹，逐条 Python 遍历在 1–2% 杂散时会慢到数小时。
        cap = 64
        T_id = np.empty(cap, np.int64); T_t1 = np.empty(cap); T_doa = np.empty(cap); T_sd = np.empty(cap)
        T_p = np.empty((cap, K))
        T_ph = np.empty(cap); T_pht = np.empty(cap); T_pm = np.empty(cap, np.int64); T_pk = np.zeros(cap, np.int64)
        T_pr = np.zeros(cap); T_pw = np.zeros(cap)   # 相位漂移率估计（μs/s）及其累计时长（s）
        m = 0
        for i in idx:
            t0, t1, doa, n = cp.t0.values[i], cp.t1.values[i], cp.doa.values[i], cp.n.values[i]
            p = P[i]
            sd_c = e / np.sqrt(3.0) / np.sqrt(max(n, 1))
            best = -1
            if m:
                gap = (t0 - T_t1[:m]) / 1e6
                alive = gap <= max_gap
                if not alive.all():
                    keep = np.where(alive)[0]
                    m2 = len(keep)
                    T_id[:m2], T_t1[:m2], T_doa[:m2], T_sd[:m2] = T_id[keep], T_t1[keep], T_doa[keep], T_sd[keep]
                    T_p[:m2] = T_p[keep]
                    T_ph[:m2], T_pht[:m2], T_pm[:m2], T_pk[:m2] = T_ph[keep], T_pht[keep], T_pm[keep], T_pk[keep]
                    T_pr[:m2], T_pw[:m2] = T_pr[keep], T_pw[keep]
                    gap = gap[keep]
                    m = m2
            if m:
                cand = ~(t0 < T_t1[:m] - eps)
                pt = T_p[:m] / T_p[:m].sum(axis=1, keepdims=True)
                compat = pt @ p
                cand &= ~(compat < min_compat)
                rate = np.where(pt @ air_mask > 0.5, r_air, r_sea)
                g = np.maximum(gap, 0.0)
                gate = 3.0 * np.hypot(sd_c, T_sd[:m]) + 0.5 + rate * g
                dd = np.abs(angdiff(doa, T_doa[:m]))
                cand &= ~(dd > gate)
                pok = None
                if phase is not None and cand.sum() > 0 and not np.isnan(ph1[i]):
                    dts = np.maximum(t0 - T_pht[:m], 0.0) / 1e6
                    est = T_pw[:m] >= p_rw_min
                    ptol = p_fit + np.where(est, p_rate_unc, p_rate) * dts + 0.5 * p_acc * dts * dts
                    php = T_ph[:m] + np.where(est, T_pr[:m], 0.0) * dts
                    d1 = np.abs((ph1[i] - php + 0.5 * L) % L - 0.5 * L)
                    d2 = np.abs((ph2[i] - php + 0.5 * L) % L - 0.5 * L) if not np.isnan(ph2[i]) else d1
                    pok = cand & (ptol <= p_max) & (np.fmin(d1, d2) <= ptol)
                    if pok.any() and cand.sum() > 1:
                        cand = pok
                    elif p_veto_k > 0 and n >= p_veto_n:
                        # 候选相位已连续多次延续（可靠）而本 CPI 明显不延续 → 不是同一辐射源，另起片段
                        cand &= ~((T_pk[:m] >= p_veto_k) & (ptol <= p_veto_us) & (np.fmin(d1, d2) > ptol))
                if cand.any():
                    cost = np.where(cand, dd / gate + 0.5 * (1.0 - compat) + 0.2 * g / max_gap, np.inf)
                    best = int(np.argmin(cost))
            if best < 0:
                if m == cap:
                    cap *= 2
                    T_id = np.resize(T_id, cap); T_t1 = np.resize(T_t1, cap); T_doa = np.resize(T_doa, cap)
                    T_sd = np.resize(T_sd, cap); T_p = np.resize(T_p, (cap, K))
                    T_ph = np.resize(T_ph, cap); T_pht = np.resize(T_pht, cap); T_pm = np.resize(T_pm, cap); T_pk = np.resize(T_pk, cap)
                    T_pr = np.resize(T_pr, cap); T_pw = np.resize(T_pw, cap)
                T_id[m], T_t1[m], T_doa[m], T_sd[m], T_p[m] = nxt, t1, doa, sd_c, p * np.log2(n + 1)
                T_ph[m], T_pht[m], T_pm[m], T_pk[m] = (ph1[i] if phase is not None else np.nan), t0, 0, 0
                T_pr[m], T_pw[m] = 0.0, 0.0
                frag[i] = nxt
                nxt += 1
                m += 1
            else:
                T_t1[best] = max(T_t1[best], t1)
                T_doa[best] = doa
                T_sd[best] = sd_c
                T_p[best] = T_p[best] + p * np.log2(n + 1)
                if phase is not None and not np.isnan(ph1[i]):
                    if pok is not None and pok[best]:
                        dtb = (t0 - T_pht[best]) / 1e6
                        phm = ph1[i] if d1[best] <= d2[best] else ph2[i]
                        if dtb >= 0.3:
                            res = (phm - (T_ph[best] + T_pr[best] * dtb) + 0.5 * L) % L - 0.5 * L
                            T_pr[best] += res / (T_pw[best] + dtb)
                            T_pw[best] = min(T_pw[best] + dtb, p_rw_cap)
                        T_ph[best], T_pht[best], T_pm[best] = phm, t0, 0
                        T_pk[best] += 1
                    else:
                        T_pk[best] = 0
                        T_pm[best] += 1   # 不延续：多为起点不在格上的 CPI，保留原相位；连续两次不延续或相位过旧则重置
                        if T_pm[best] >= 2 or np.isnan(T_ph[best]) or (pok is not None and not (ptol[best] <= p_max)):
                            T_ph[best], T_pht[best], T_pm[best] = ph1[i], t0, 0
                            T_pr[best], T_pw[best] = 0.0, 0.0
                frag[i] = T_id[best]
    return frag


def fragment_phase(cp: pd.DataFrame, cfg: dict):
    """片段串接用的 CPI 相位：主相位 = 驻留起点（抖动重频 CPI 回溯到驻留首段）mod 格长，备选 = 自身起点 mod 格长。"""
    from .exclusion import dwell_heads
    pc = cfg["phase"]
    L = float(pc.get("lattice_us", 64.0))
    t0 = cp.t0.values.astype(float)
    ph = t0 % L
    derr = np.where(cp.sh.values > 100, cfg["cpi"]["doa_err_air"], cfg["cpi"]["doa_err_sea"])
    h = dwell_heads(cp, pc.get("head_gap_mul", 2.6), pc.get("head_gap_us", 1700.0), pc.get("head_dur_ms", 30.0) * 1e3, derr)
    ph1 = ph[h]
    ph2 = np.where(h != np.arange(len(cp)), ph, np.nan)
    small = cp.n.values < int(cfg["track"].get("frag_phase_min_n", 3))
    ph1 = np.where(small, np.nan, ph1)
    return ph1, ph2


def fragment_phase_hit(cp: pd.DataFrame, frag: np.ndarray, ph1: np.ndarray, fit: float, rate: float, L: float,
                       max_gap_s: float = 3.0) -> tuple[float, int]:
    """自检：片段内相邻 CPI（同节点、间隔 ≤ max_gap_s）相位延续的比例（随机约 2–5%，训练/测试集约 0.9）。"""
    d = pd.DataFrame({"f": frag, "t": cp.t0.values.astype(float), "ph": ph1}).dropna().sort_values(["f", "t"])
    same = d.f.values[1:] == d.f.values[:-1]
    dt = np.diff(d.t.values) / 1e6
    m = same & (dt <= max_gap_s) & (dt > 0)
    if not m.any():
        return 0.0, 0
    dp = np.abs((np.diff(d.ph.values) + 0.5 * L) % L - 0.5 * L)
    return float(np.mean(dp[m] <= fit + rate * dt[m])), int(m.sum())


# ---------------------------------------------------------------- 跨节点同源 CPI
def _seq_aligned(ta: np.ndarray, tb: np.ndarray, tol: float, k: int = 3) -> float:
    """两节点脉冲到达时刻序列能否整体平移对齐：同一次发射在两节点上只差一个固定传播时延。

    候选平移量取两端前 k 个脉冲的两两差（容忍首尾漏脉冲），返回最佳平移下的命中比例。"""
    if len(ta) == 0 or len(tb) == 0:
        return 0.0
    k = min(k, len(ta), len(tb))
    offs = np.concatenate([(tb[:k, None] - ta[None, :k]).ravel(), (tb[-k:, None] - ta[None, -k:]).ravel()])
    best = 0
    for o in np.unique(np.round(offs, 1)):
        q = ta + o
        j = np.clip(np.searchsorted(tb, q), 1, len(tb) - 1) if len(tb) > 1 else np.zeros(len(q), int)
        d = np.minimum(np.abs(tb[j] - q), np.abs(tb[j - 1] - q)) if len(tb) > 1 else np.abs(tb[0] - q)
        best = max(best, int((d <= tol).sum()))
    return best / min(len(ta), len(tb))


def verify_jitter_pairs(cp: pd.DataFrame, pairs: pd.DataFrame, toa: np.ndarray, cpi_of: np.ndarray,
                        tol: float = 2.0, min_hit: float = 0.6) -> pd.DataFrame:
    """抖动 PRI 的跨节点配对只凭时间/RF/PW 门限，两部同型号雷达同时发射时会被误配；
    同一发射的抖动序列在两节点上完全一致（仅差传播时延），用脉冲级序列对齐复核。"""
    if not len(pairs):
        return pairs
    jit = pd.Series(cp.jitter.values > 0, index=cp.cpi.values)
    m = jit.loc[pairs.cpi_a.values].values
    if not m.any():
        return pairs
    order = np.lexsort((toa, cpi_of))
    cs, ts = cpi_of[order], toa[order].astype(float)
    keep = np.ones(len(pairs), bool)
    for q in np.where(m)[0]:
        a_, b_ = pairs.cpi_a.values[q], pairs.cpi_b.values[q]
        ta = ts[np.searchsorted(cs, a_):np.searchsorted(cs, a_, side="right")]
        tb = ts[np.searchsorted(cs, b_):np.searchsorted(cs, b_, side="right")]
        keep[q] = _seq_aligned(ta, tb, tol) >= min_hit
    # 保护：训练集抖动配对 85–98% 为真；若大半被否决，说明到达时刻噪声超出 tol（未知场景），放弃复核
    if (~keep[m]).mean() > 0.5:
        return pairs
    return pairs[keep]


def match_cross_platform(cp: pd.DataFrame, cfg: dict, toa: np.ndarray | None = None,
                         cpi_of: np.ndarray | None = None) -> pd.DataFrame:
    """同一发射在不同节点上形成的 CPI：时间近同（传播时延差）、RF/PW/PRI 一致；抖动 PRI 另做脉冲序列对齐复核。"""
    fc = cfg["fusion"]
    d = cp[["cpi", "platform_id", "t0", "t1", "rf", "pw", "pri", "jitter", "n"]].copy()
    d["prikey"] = np.where(d.jitter > 0, -1, np.round(d.pri.fillna(-1))).astype(np.int64)
    d = d.sort_values("t0")
    plats = sorted(d.platform_id.unique())
    res = []
    for i, pa in enumerate(plats):
        for pb in plats[i + 1:]:
            A = d[d.platform_id == pa]
            B = d[d.platform_id == pb]
            for direction in ("backward", "forward"):
                m = pd.merge_asof(A, B, on="t0", by="prikey", direction=direction, tolerance=fc["max_delay_us"],
                                  suffixes=("_a", "_b"))
                m = m.dropna(subset=["cpi_b"])
                ok = ((m.rf_a - m.rf_b).abs() <= fc["rf_tol"]) & ((m.pw_a - m.pw_b).abs() <= fc["pw_tol"])
                ok &= (m.jitter_a > 0) | ((m.pri_a - m.pri_b).abs() <= fc["pri_tol"])
                res.append(m.loc[ok, ["cpi_a", "cpi_b"]])
    if not res:
        return pd.DataFrame(columns=["cpi_a", "cpi_b"])
    out = pd.concat(res).astype(np.int64).drop_duplicates()
    if toa is not None and cpi_of is not None and fc.get("seq_check", True):
        out = verify_jitter_pairs(cp, out, toa, cpi_of, fc.get("seq_tol_us", 2.0), fc.get("seq_min_hit", 0.6))
    return out


class _DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r


def _overlap_count(iv: np.ndarray, eps: float) -> int:
    if len(iv) < 2:
        return 0
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    run_max = np.maximum.accumulate(iv[:-1, 1])
    return int((iv[1:, 0] < run_max - eps).sum())


def fuse_fragments(cp: pd.DataFrame, frag: np.ndarray, pairs: pd.DataFrame, cfg: dict) -> np.ndarray:
    """以同源 CPI 数为权重，在"同节点片段时间不重叠"约束下贪心合并片段 → 对象。"""
    fc, tc = cfg["fusion"], cfg["track"]
    cp = cp.assign(frag=frag)
    f_of_cpi = pd.Series(frag, index=cp.cpi.values)
    if len(pairs):
        fp = pd.DataFrame({"fa": f_of_cpi.loc[pairs.cpi_a.values].values, "fb": f_of_cpi.loc[pairs.cpi_b.values].values})
        fp = fp[fp.fa != fp.fb]
        w = fp.groupby(["fa", "fb"]).size().rename("w").reset_index()
        w = w[w.w >= fc["min_pairs"]].sort_values("w", ascending=False)
    else:
        w = pd.DataFrame(columns=["fa", "fb", "w"])
    nf = int(frag.max()) + 1
    dsu = _DSU(nf)
    # 组件 -> {platform: intervals}
    comp_iv = {}
    for (f, pid), x in cp.groupby(["frag", "platform_id"]):
        comp_iv.setdefault(int(f), {})[int(pid)] = x[["t0", "t1"]].values
    eps = tc["overlap_eps_us"]
    for fa, fb, _ in w.itertuples(index=False):
        ra, rb = dsu.find(int(fa)), dsu.find(int(fb))
        if ra == rb:
            continue
        A, B = comp_iv[ra], comp_iv[rb]
        ok = True
        for pid in set(A) & set(B):
            iv = np.vstack([A[pid], B[pid]])
            if _overlap_count(iv, eps) > 0.01 * min(len(A[pid]), len(B[pid])):
                ok = False
                break
        if not ok:
            continue
        dsu.p[rb] = ra
        for pid, iv in B.items():
            A[pid] = np.vstack([A[pid], iv]) if pid in A else iv
        del comp_iv[rb]
    roots = np.array([dsu.find(int(f)) for f in frag])
    _, obj = np.unique(roots, return_inverse=True)
    return obj.astype(np.int64)


# ---------------------------------------------------------------- 对象级合并
@njit(cache=True)
def _conflicts(a0, a1, arf, apri, b0, b1, brf, bpri, maxdur, margin, rf_tol, pri_tol):
    """统计 B 中与 A 在时间上重叠（> margin）且参数不一致的 CPI 数。

    同一辐射源任一时刻只发射一个 CPI：不同节点上的同一 CPI 参数一致、时间仅差传播时延；
    若两对象的 CPI 在时间上重叠而参数不同，则二者不可能是同一辐射源。
    """
    cnt = 0
    na = a0.shape[0]
    for j in range(b0.shape[0]):
        lo = np.searchsorted(a0, b0[j] - maxdur)
        hi = np.searchsorted(a0, b1[j])
        for i in range(lo, min(hi, na)):
            ov = min(a1[i], b1[j]) - max(a0[i], b0[j])
            if ov > margin:
                same = abs(arf[i] - brf[j]) <= rf_tol and (abs(apri[i] - bpri[j]) <= pri_tol or apri[i] < 0 or bpri[j] < 0)
                if not same:
                    cnt += 1
                    break
    return cnt


def merge_objects(cp: pd.DataFrame, obj: np.ndarray, obj_type: pd.Series, cfg: dict,
                  proba: pd.DataFrame | None = None) -> np.ndarray:
    """型号相容、全局发射时序无冲突、且同节点方位衔接连续的对象合并。

    型号相容：合并后最优型号的对数似然相对各自最优之和的损失（按较小对象 CPI 权重归一）
    不超过 type_loss_max；无 proba 时退化为型号相等。"""
    tc, fc = cfg["track"], cfg["fusion"]
    cp = cp.assign(obj=obj)
    pri = np.where(cp.jitter.values > 0, -1.0, cp.pri.fillna(-1).values)
    cp = cp.assign(pri_k=pri)
    maxdur = float((cp.t1 - cp.t0).max()) + 1.0
    margin = fc["max_delay_us"]
    groups = {}
    for o, x in cp.groupby("obj"):
        x = x.sort_values("t0")
        groups[int(o)] = x
    types = obj_type.to_dict()
    S, W = {}, {}
    if proba is not None:
        wv = np.log2(cp.n.values + 1)
        LLw = pd.DataFrame(np.log(np.clip(proba.values, 1e-4, 1)) * wv[:, None], index=cp.index, columns=proba.columns)
        for k, x in groups.items():
            S[k] = LLw.loc[x.index].sum().values
            W[k] = wv[cp.index.get_indexer(x.index)].sum()
    classes = list(proba.columns) if proba is not None else []
    loss_max = tc.get("type_loss_max", 0.5)

    def type_ok(ka, kb):
        if proba is None:
            return types.get(ka) == types.get(kb), types.get(ka)
        joint = S[ka] + S[kb]
        loss = (S[ka].max() + S[kb].max() - joint.max()) / max(min(W[ka], W[kb]), 1e-9)
        return loss <= loss_max, classes[int(np.argmax(joint))]

    doa_err = {True: cfg["cpi"]["doa_err_air"], False: cfg["cpi"]["doa_err_sea"]}

    def arrays(x, t):
        # 只用自身判型与对象型号一致的 CPI 计冲突，避免少量误归属 CPI 造成伪冲突
        if "ptype" in x.columns:
            y = x[x.ptype == t]
            if len(y) >= 0.5 * len(x):
                x = y
        return (x.t0.values, x.t1.values, x.rf.values, x.pri_k.values)

    def doa_consistent(xa, xb):
        """共享节点上，时间最近的首尾 CPI 方位差是否在门限内；返回 (有证据, 一致)。"""
        evid, good = 0, 0
        for pid in set(xa.platform_id) & set(xb.platform_id):
            A = xa[xa.platform_id == pid]
            B = xb[xb.platform_id == pid]
            # A 在前 B 在后 或 反之：取边界
            for P, Q in ((A, B), (B, A)):
                if P.t1.max() <= Q.t0.min() + margin:
                    p, q = P.iloc[-1], Q.iloc[0]
                    gap_s = max(q.t0 - p.t1, 0) / 1e6
                    air = p.sh > 100
                    gate = 2.0 * doa_err[air] + tc["doa_rate_air"] * gap_s
                    evid += 1
                    good += abs(float(angdiff(p.doa, q.doa))) <= gate
        return evid, good

    def pair_score(ka, kb):
        ok, t = type_ok(ka, kb)
        if not ok:
            return None
        xa, xb = groups[ka], groups[kb]
        small = min(len(xa), len(xb))
        c = _conflicts(*arrays(xa, t), *arrays(xb, t), maxdur, margin, fc["rf_tol"], fc["pri_tol"])
        if c > max(2, fc.get("conflict_frac", 0.01) * small):
            return None
        ev, good = doa_consistent(xa, xb)
        if ev > 0 and good == 0:
            return None
        return (good + 1) * np.log1p(small)

    # 成对判定只依赖两个对象自身，缓存后每轮只需重算与上轮合并对象相关的对（杂散脉冲造成数百个碎对象时避免平方级重算）
    cache = {}
    changed = True
    while changed:
        changed = False
        keys = sorted(groups, key=lambda k: -len(groups[k]))[: tc.get("max_merge_objects", 200)]
        best = None
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                if (ka, kb) not in cache:
                    cache[(ka, kb)] = pair_score(ka, kb)
                score = cache[(ka, kb)]
                if score is None:
                    continue
                if best is None or score > best[0]:
                    best = (score, ka, kb)
        if best is not None:
            _, ka, kb = best
            for key in [k_ for k_ in cache if ka in k_ or kb in k_]:
                del cache[key]
            groups[ka] = pd.concat([groups[ka], groups[kb]]).sort_values("t0")
            del groups[kb]
            if proba is not None:
                S[ka] = S[ka] + S.pop(kb)
                W[ka] = W[ka] + W.pop(kb)
                types[ka] = classes[int(np.argmax(S[ka]))]
            changed = True
    new = pd.Series(0, index=cp.index, dtype=np.int64)
    for k, (o, x) in enumerate(sorted(groups.items())):
        new.loc[x.index] = k
    return new.loc[cp.index].values


@njit(cache=True)
def _rf_conflicts(a0, a1, arf, b0, b1, brf, maxdur, eps, rf_tol):
    """B 中与 A（按 t0 排序）时间重叠超过 eps 且 RF 不一致的 CPI 数。"""
    cnt = 0
    for j in range(b0.shape[0]):
        lo = np.searchsorted(a0, b0[j] - maxdur)
        hi = np.searchsorted(a0, b1[j])
        for i in range(lo, hi):
            if min(a1[i], b1[j]) - max(a0[i], b0[j]) > eps and abs(arf[i] - brf[j]) > rf_tol:
                cnt += 1
                break
    return cnt


def merge_phase_objects(cp: pd.DataFrame, obj: np.ndarray, proba: pd.DataFrame, cfg: dict,
                        ph1: np.ndarray, ph2: np.ndarray) -> tuple[np.ndarray, int]:
    """同节点上时间交错、相位轨迹一致的两个对象是同一辐射源（驻留起点 mod 64 μs 只随距离缓慢漂移，
    不同辐射源偶然一致约 2%）：常见于同一雷达的部分 CPI 因方位略超航迹门限未被锚定、另成对象。

    B 的每个（有相位的）CPI 与 A 在同节点上时间最近（≤ win_s）的 CPI 比相位；所有比较数足够的共享节点上
    吻合比例都 ≥ node_frac、总体 ≥ min_frac，且型号相容、同节点无时间重叠、不是锚定于两条不同航迹 → B 并入 A。
    """
    pc = cfg["track"]
    win = float(pc.get("phase_merge_win_s", 1.5)) * 1e6
    nb = int(pc.get("phase_merge_nb", 3))
    fit, rate = float(pc["frag_phase_fit_us"]), float(pc["frag_phase_rate"])
    L = float(cfg["phase"].get("lattice_us", 64.0))
    min_cmp, min_frac = int(pc.get("phase_merge_min_cmp", 8)), float(pc.get("phase_merge_min_frac", 0.6))
    node_cmp, node_frac = int(pc.get("phase_merge_node_cmp", 5)), float(pc.get("phase_merge_node_frac", 0.5))
    loss_max = float(pc.get("phase_merge_type_loss", 1.0))
    eps = float(pc["overlap_eps_us"])
    rf = cp.rf.values.astype(float)
    rf_tol = float(cfg["fusion"]["rf_tol"])
    conf_frac = float(pc.get("phase_merge_conflict_frac", 0.02))
    maxdur = float((cp.t1 - cp.t0).max()) + 1.0
    obj = obj.copy()
    n = cp.n.values
    pid = cp.platform_id.values
    t0, t1 = cp.t0.values.astype(float), cp.t1.values.astype(float)
    ad = cp["anchor_direct"].values if "anchor_direct" in cp.columns else np.full(len(cp), -1)
    wv = np.log2(n + 1)
    LLc = np.log(np.clip(proba.values, 1e-4, 1)) * wv[:, None]
    has = ~np.isnan(ph1)
    merged = 0
    for _ in range(int(pc.get("phase_merge_max", 20))):
        size = pd.Series(n).groupby(obj).sum().sort_values()
        anc = {}
        for o, v in pd.Series(ad[ad >= 0]).groupby(obj[ad >= 0]):
            anc[o] = int(v.value_counts().index[0])
        best = None
        for b_o in size.index:
            xb = np.where((obj == b_o) & has)[0]
            if len(xb) < min_cmp:
                continue
            for a_o in size.index[::-1]:
                if a_o == b_o or size[a_o] < size[b_o]:
                    continue
                if b_o in anc and a_o in anc and anc[a_o] != anc[b_o]:
                    continue
                M = C = 0
                ok = True
                for p in np.unique(pid[xb]):
                    xa = np.where((obj == a_o) & (pid == p) & has)[0]
                    if not len(xa):
                        continue
                    xa = xa[np.argsort(t0[xa])]
                    ib = xb[pid[xb] == p]
                    # 与 A 在此节点上前后各 nb 个 CPI 比较，过半吻合才算吻合（A 中混入的少量 B 辐射源 CPI 不能单独促成合并）
                    k = np.searchsorted(t0[xa], t0[ib])
                    hit = np.zeros(len(ib)); cnt = np.zeros(len(ib))
                    for off in range(-nb, nb):
                        q = k + off
                        okq = (q >= 0) & (q < len(xa))
                        j = xa[np.clip(q, 0, len(xa) - 1)]
                        dt = np.abs(t0[j] - t0[ib])
                        okq &= dt <= win
                        dd = np.full(len(ib), np.inf)
                        for u in (ph1, ph2):
                            for v in (ph1, ph2):
                                x = np.abs((u[ib] - v[j] + 0.5 * L) % L - 0.5 * L)
                                dd = np.fmin(dd, np.where(np.isnan(x), np.inf, x))
                        cnt += okq
                        hit += okq & (dd <= fit + rate * dt / 1e6)
                    sel = cnt > 0
                    if not sel.any():
                        continue
                    ib = ib[sel]
                    mt = int((hit[sel] >= 0.5 * cnt[sel]).sum())
                    if len(ib) >= node_cmp and mt < node_frac * len(ib):
                        ok = False
                        break
                    M += mt; C += len(ib)
                if not ok or C < min_cmp or M < min_frac * C:
                    continue
                la, lb = LLc[obj == a_o].sum(0), LLc[obj == b_o].sum(0)
                loss = (la.max() + lb.max() - (la + lb).max()) / max(wv[obj == b_o].sum(), 1e-9)
                if loss > loss_max:
                    continue
                # 同节点时间重叠且 RF 不同（同一辐射源同一时刻只有一个频点；多重频驻留会被串成时间交叠、同频的几条 CPI）
                ovl = 0
                for p in np.unique(pid[obj == b_o]):
                    xa = np.where((obj == a_o) & (pid == p))[0]
                    if not len(xa):
                        continue
                    xa = xa[np.argsort(t0[xa])]
                    ib = np.where((obj == b_o) & (pid == p))[0]
                    ovl += _rf_conflicts(t0[xa], t1[xa], rf[xa], t0[ib], t1[ib], rf[ib], maxdur, eps, rf_tol)
                if ovl > conf_frac * (obj == b_o).sum():
                    continue
                score = M / C
                if best is None or score > best[0]:
                    best = (score, a_o, b_o, M, C)
            if best is not None:
                break
        if best is None:
            break
        obj[obj == best[2]] = best[1]
        merged += 1
    return pd.factorize(obj)[0], merged


def absorb_small(cp: pd.DataFrame, obj: np.ndarray, proba: pd.DataFrame, cfg: dict,
                 force: set | None = None) -> np.ndarray:
    """小对象（脉冲数低于阈值）并入同节点上方位最连续、型号相容的大对象；找不到则保留。
    force：无论大小都按小对象处理的对象（如被时间互斥重分配瓦解后的残余）。"""
    tc = cfg["track"]
    obj = obj.copy()
    size = pd.Series(cp.n.values).groupby(obj).sum()
    is_small = size < tc["min_obj_pulses"]
    if force:
        is_small |= size.index.isin(list(force))
    small = size[is_small].index
    big = size[~is_small].index
    if len(small) == 0 or len(big) == 0:
        return obj
    wv = np.log2(cp.n.values + 1)
    LL = pd.DataFrame(np.log(np.clip(proba.values, 1e-4, 1)) * wv[:, None]).groupby(obj).sum()
    Wt = pd.Series(wv).groupby(obj).sum()
    doa_err = {True: cfg["cpi"]["doa_err_air"], False: cfg["cpi"]["doa_err_sea"]}
    tiny_thr = tc.get("tiny_obj_pulses", 2000)
    order = size.loc[small].sort_values(ascending=False).index
    cap = tc.get("max_absorb_objects", 500)
    small, rest = order[:cap], order[cap:]
    if len(rest):  # 超量小对象：按同节点时间最近的大对象 CPI 快速归并（复杂度保护）
        bigmask = np.isin(obj, big)
        for pid in np.unique(cp.platform_id.values):
            bi = np.where(bigmask & (cp.platform_id.values == pid))[0]
            si = np.where(np.isin(obj, rest) & (cp.platform_id.values == pid))[0]
            if len(bi) == 0 or len(si) == 0:
                continue
            bi = bi[np.argsort(cp.t0.values[bi])]
            k = np.clip(np.searchsorted(cp.t0.values[bi], cp.t0.values[si]), 0, len(bi) - 1)
            obj[si] = obj[bi[k]]
    # 大对象逐节点的有序 (t0, doa) 缓存；大对象吸收小对象后作废其缓存（结果与逐 CPI 循环逐位一致，复杂度从 O(小×大×N) 降到按需构建）
    t0v, doav, pidv = cp.t0.values, cp.doa.values, cp.platform_id.values
    tolbase = np.where(cp.sh.values > 100, 2 * doa_err[True], 2 * doa_err[False])
    LLv = LL.values
    li = {o: i for i, o in enumerate(LL.index)}
    cache = {}

    def big_node(b_o, pid):
        key = (b_o, pid)
        if key not in cache:
            xb = np.where((obj == b_o) & (pidv == pid))[0]
            xb = xb[np.argsort(t0v[xb])]
            cache[key] = (t0v[xb], doav[xb])
        return cache[key]

    for s_o in small:
        xs = np.where(obj == s_o)[0]
        cands = []  # (cost, b_o, devmed, loss)
        ls = LLv[li[s_o]]
        nodes = np.unique(pidv[xs])
        for b_o in big:
            lb = LLv[li[b_o]]
            loss = (ls.max() + lb.max() - (ls + lb).max()) / Wt.loc[s_o]
            devs = []
            for pid in nodes:
                tb, db = big_node(b_o, pid)
                if len(tb) == 0:
                    continue
                ii = xs[pidv[xs] == pid]
                ti = t0v[ii]
                k = np.searchsorted(tb, ti)
                km1 = np.clip(k - 1, 0, len(tb) - 1)
                kk = np.clip(k, 0, len(tb) - 1)
                use_prev = (k == len(tb)) | ((k > 0) & (np.abs(tb[km1] - ti) <= np.abs(tb[kk] - ti)))
                q = np.where(use_prev, km1, kk)
                gap_s = np.abs(tb[q] - ti) / 1e6
                tol = tolbase[ii] + tc["doa_rate_air"] * gap_s
                devs.append(np.abs(angdiff(doav[ii], db[q])) / tol)
            if not devs:
                continue
            dm = float(np.median(np.concatenate(devs)))
            cands.append((dm + loss, b_o, dm, loss))
        if not cands:
            continue
        ok = [c for c in cands if c[2] <= 1.0 and c[3] <= tc.get("absorb_type_loss_max", 3.0)]
        tgt = None
        if ok:
            tgt = min(ok)[1]
        elif size.loc[s_o] < tiny_thr:  # 极小对象：独立成对象证据不足，并入综合代价最小者
            tgt = min(cands)[1]
        if tgt is not None:
            obj[xs] = tgt
            for key in [k_ for k_ in cache if k_[0] == tgt]:
                del cache[key]
    return pd.factorize(obj)[0]


def _doa_agreement(ta, da, tb, db, win, bin_us):
    """B 的每个 CPI 与 A 在同一节点上的方位（用 A 前后 win 内最近的 CPI 线性插值，消除目标运动的影响）之差，
    按时间格取中位数。返回各格的 |方位差| 中位数。"""
    if len(ta) < 2 or len(tb) == 0:
        return np.array([])
    ua = np.rad2deg(np.unwrap(np.deg2rad(da)))
    k = np.searchsorted(ta, tb)
    ok = (k > 0) & (k < len(ta))
    k = np.clip(k, 1, len(ta) - 1)
    tl, tr = ta[k - 1], ta[k]
    ok &= (tb - tl <= win) & (tr - tb <= win)
    if not ok.any():
        return np.array([])
    f = np.where(tr > tl, (tb - tl) / np.maximum(tr - tl, 1e-9), 0.5)
    pred = ua[k - 1] + f * (ua[k] - ua[k - 1])
    d = np.abs(angdiff(db[ok], pred[ok]))
    return pd.Series(d).groupby((tb[ok] // bin_us).astype(np.int64)).median().values


def merge_split_objects(cp: pd.DataFrame, obj: np.ndarray, proba: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, int]:
    """拆分对象重合并：同一辐射源被拆成两个对象时（常见于无航迹锚定的目标），两半在每个共享节点上
    同一时刻的方位相同、全局发射时序无冲突。以同时段方位一致为正证据（比 merge_objects 的首尾衔接更严格）；
    锚定于不同航迹的对象不合并。返回 (obj, 合并次数)。"""
    tc, fc = cfg["track"], cfg["fusion"]
    mc = cfg.get("merge_split", {})
    bin_us = mc.get("bin_s", 5.0) * 1e6
    win = mc.get("interp_win_s", 5.0) * 1e6
    min_bins = mc.get("min_bins", 10)
    frac = mc.get("min_frac", 0.6)
    gscale = mc.get("gate_scale", 0.5)
    obj = pd.factorize(obj)[0]
    n = cp.n.values
    pid = cp.platform_id.values
    t0, t1 = cp.t0.values.astype(float), cp.t1.values.astype(float)
    tm = 0.5 * (t0 + t1)
    rf = cp.rf.values.astype(float)
    pri = np.where(cp.jitter.values > 0, -1.0, cp.pri.fillna(-1).values).astype(float)
    doa = cp.doa.values.astype(float)
    node_air = pd.Series(cp.sh.values).groupby(pid).median() > 100
    gate = {p: (cfg["cpi"]["doa_err_air"] if node_air.loc[p] else cfg["cpi"]["doa_err_sea"]) * gscale for p in node_air.index}
    anchor = cp["anchor"].values if "anchor" in cp.columns else np.full(len(cp), -1)
    maxdur = float((t1 - t0).max()) + 1.0
    wv = np.log2(n + 1)
    LL = np.log(np.clip(proba.values, 1e-4, 1)) * wv[:, None]
    loss_max = tc.get("type_loss_max", 0.5)
    merges = 0
    for _ in range(mc.get("max_merges", 10)):
        size = pd.Series(n).groupby(obj).sum()
        S = pd.DataFrame(LL).groupby(obj).sum()
        W = pd.Series(wv).groupby(obj).sum()
        d = pd.DataFrame({"o": obj, "j": anchor, "n": n})
        wj = d[d.j >= 0].groupby(["o", "j"]).n.sum()
        otrk = {int(o): int(j) for (o, j), v in wj.items() if v >= 0.5 * size.loc[o]}
        big = [o for o in size.sort_values(ascending=False).index[: tc.get("max_merge_objects", 200)] if size.loc[o] >= tc["min_obj_pulses"]]
        per = {}
        for o in big:
            io_ = np.where(obj == o)[0]
            for p in np.unique(pid[io_]):
                ii = io_[pid[io_] == p]
                ii = ii[np.argsort(tm[ii], kind="stable")]
                per[(o, p)] = (tm[ii], doa[ii])
        best = None
        for i, a in enumerate(big):
            for b in big[i + 1:]:
                ja, jb = otrk.get(int(a), -1), otrk.get(int(b), -1)
                if ja >= 0 and jb >= 0 and ja != jb:
                    continue
                sa, sb = S.loc[a].values, S.loc[b].values
                if (sa.max() + sb.max() - (sa + sb).max()) / max(min(W.loc[a], W.loc[b]), 1e-9) > loss_max:
                    continue
                med, g = [], []
                for p in node_air.index:
                    if (a, p) not in per or (b, p) not in per:
                        continue
                    (ta, da), (tb, db) = per[(a, p)], per[(b, p)]
                    # 以 CPI 少的一方为查询点
                    x = _doa_agreement(ta, da, tb, db, win, bin_us) if len(tb) <= len(ta) else _doa_agreement(tb, db, ta, da, win, bin_us)
                    med.append(x)
                    g.append(np.full(len(x), gate[p]))
                if not med:
                    continue
                med, g = np.concatenate(med), np.concatenate(g)
                if len(med) < min_bins:
                    continue
                good = med <= g
                if good.mean() < frac:
                    continue
                ia = np.where(obj == a)[0]
                ib = np.where(obj == b)[0]
                ia, ib = ia[np.argsort(t0[ia])], ib[np.argsort(t0[ib])]
                c = _conflicts(t0[ia], t1[ia], rf[ia], pri[ia], t0[ib], t1[ib], rf[ib], pri[ib], maxdur,
                               fc["max_delay_us"], fc["rf_tol"], fc["pri_tol"])
                if c > max(2, mc.get("conflict_frac", 0.06) * min(len(ia), len(ib))):
                    continue
                sc = good.sum()
                if best is None or sc > best[0]:
                    best = (sc, a, b)
        if best is None:
            break
        obj = np.where(obj == best[2], best[1], obj)
        merges += 1
    return pd.factorize(obj)[0], merges
