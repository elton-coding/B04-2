"""L3 节点内 CPI 串接（局部辐射源片段）与 L6 跨平台融合。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from .geo import angdiff


# ---------------------------------------------------------------- 节点内片段
def build_fragments(cp: pd.DataFrame, proba: pd.DataFrame, domain_air: dict, cfg: dict) -> np.ndarray:
    """逐节点按时间顺序做 CPI→辐射源的门限关联，返回每个 CPI 的片段号。"""
    tc = cfg["track"]
    doa_err = {True: cfg["cpi"]["doa_err_air"], False: cfg["cpi"]["doa_err_sea"]}
    classes = list(proba.columns)
    air_mask = np.array([domain_air.get(c, True) for c in classes])
    P = proba.values
    frag = np.full(len(cp), -1, np.int64)
    nxt = 0
    for pid, idx in cp.groupby("platform_id", sort=True).indices.items():
        idx = idx[np.argsort(cp.t0.values[idx], kind="stable")]
        node_air = float(np.nanmedian(cp.sh.values[idx])) > 100
        e = doa_err[node_air]
        tracks = []  # dict: id, t1, doa, sd, p(sum), n
        for i in idx:
            t0, t1, doa, n = cp.t0.values[i], cp.t1.values[i], cp.doa.values[i], cp.n.values[i]
            p = P[i]
            sd_c = e / np.sqrt(3.0) / np.sqrt(max(n, 1))
            best, best_cost = None, 1e18
            alive = []
            for tr in tracks:
                gap_s = (t0 - tr["t1"]) / 1e6
                if gap_s > tc["max_gap_s"]:
                    continue
                alive.append(tr)
                if t0 < tr["t1"] - tc["overlap_eps_us"]:
                    continue
                pt = tr["p"] / tr["p"].sum()
                compat = float(np.dot(p, pt))
                if compat < tc["min_type_compat"]:
                    continue
                is_air = float(np.dot(pt, air_mask)) > 0.5
                rate = tc["doa_rate_air"] if is_air else tc["doa_rate_sea"]
                gate = 3.0 * np.hypot(sd_c, tr["sd"]) + 0.5 + rate * max(gap_s, 0.0)
                dd = abs(float(angdiff(doa, tr["doa"])))
                if dd > gate:
                    continue
                cost = dd / gate + 0.5 * (1.0 - compat) + 0.2 * max(gap_s, 0.0) / tc["max_gap_s"]
                if cost < best_cost:
                    best, best_cost = tr, cost
            tracks = alive
            if best is None:
                best = {"id": nxt, "t1": t1, "doa": doa, "sd": sd_c, "p": p * np.log2(n + 1)}
                nxt += 1
                tracks.append(best)
            else:
                best["t1"] = max(best["t1"], t1)
                best["doa"] = doa
                best["sd"] = sd_c
                best["p"] = best["p"] + p * np.log2(n + 1)
            frag[i] = best["id"]
    return frag


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

    changed = True
    while changed:
        changed = False
        keys = sorted(groups, key=lambda k: -len(groups[k]))[: tc.get("max_merge_objects", 200)]
        best = None
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                ok, t = type_ok(ka, kb)
                if not ok:
                    continue
                xa, xb = groups[ka], groups[kb]
                small = min(len(xa), len(xb))
                c = _conflicts(*arrays(xa, t), *arrays(xb, t), maxdur, margin, fc["rf_tol"], fc["pri_tol"])
                if c > max(2, fc.get("conflict_frac", 0.01) * small):
                    continue
                ev, good = doa_consistent(xa, xb)
                if ev > 0 and good == 0:
                    continue
                score = (good + 1) * np.log1p(small)
                if best is None or score > best[0]:
                    best = (score, ka, kb)
        if best is not None:
            _, ka, kb = best
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
    for s_o in small:
        xs = np.where(obj == s_o)[0]
        cands = []  # (cost, b_o, devmed, loss)
        for b_o in big:
            loss = (LL.loc[s_o].max() + LL.loc[b_o].max() - (LL.loc[s_o] + LL.loc[b_o]).max()) / Wt.loc[s_o]
            devs = []
            for pid in np.unique(cp.platform_id.values[xs]):
                xb = np.where((obj == b_o) & (cp.platform_id.values == pid))[0]
                if len(xb) == 0:
                    continue
                xb = xb[np.argsort(cp.t0.values[xb])]
                for i in xs[cp.platform_id.values[xs] == pid]:
                    k = np.searchsorted(cp.t0.values[xb], cp.t0.values[i])
                    near = [q for q in (k - 1, k) if 0 <= q < len(xb)]
                    q = min(near, key=lambda q: abs(cp.t0.values[xb[q]] - cp.t0.values[i]))
                    gap_s = abs(cp.t0.values[xb[q]] - cp.t0.values[i]) / 1e6
                    tol = 2 * doa_err[cp.sh.values[i] > 100] + tc["doa_rate_air"] * gap_s
                    devs.append(abs(float(angdiff(cp.doa.values[i], cp.doa.values[xb[q]]))) / tol)
            if not devs:
                continue
            dm = float(np.median(devs))
            cands.append((dm + loss, b_o, dm, loss))
        if not cands:
            continue
        ok = [c for c in cands if c[2] <= 1.0 and c[3] <= tc.get("absorb_type_loss_max", 3.0)]
        if ok:
            obj[xs] = min(ok)[1]
        elif size.loc[s_o] < tiny_thr:  # 极小对象：独立成对象证据不足，并入综合代价最小者
            obj[xs] = min(cands)[1]
    return pd.factorize(obj)[0]
