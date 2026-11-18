"""L4 型号识别（判别式备选）、L5 工作状态识别（两级分类 + 多节点一致 + 跟踪段序号）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TYPE_FEATS = ["rf", "pw", "pri_eff", "jitter", "logn"]


def type_features(cp: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=cp.index)
    f["rf"] = cp.rf
    f["pw"] = cp.pw
    f["pri_eff"] = cp.pri.fillna(-1.0)
    f["jitter"] = cp.jitter
    f["logn"] = np.log2(cp.n.clip(lower=1))
    return f


def fit_type_model(cp: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    m = (cp.n >= 3).values
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(type_features(cp)[TYPE_FEATS].values[m], y.values[m],
            sample_weight=np.log2(cp.n.values[m] + 1))
    return clf


def predict_type_proba(clf, cp: pd.DataFrame) -> pd.DataFrame:
    p = clf.predict_proba(type_features(cp)[TYPE_FEATS].values)
    return pd.DataFrame(p, index=cp.index, columns=clf.classes_)


def vote_object_type(cp: pd.DataFrame, proba: pd.DataFrame, key: str = "object_id") -> pd.Series:
    """对象级证据累积：按脉冲数加权的对数概率求和（重叠区 CPI 自然降权）。"""
    lp = np.log(proba.clip(lower=1e-4))
    w = np.log2(cp.n.clip(lower=1) + 1)
    s = lp.mul(w, axis=0).groupby(cp[key]).sum()
    return s.idxmax(axis=1)


# ---------------------------------------------------------------- 状态识别
# 物理依据（训练集验证）：
# - 雷达边扫边跟：跟踪驻留按固定重访周期（0.2/0.4 s）插入搜索序列，相邻 CPI 首尾相接；
# - 同一发射在各节点状态相同 → 概率按发射平均；
# - 粗跟位于"对某被跟踪目标"跟踪段的前若干次驻留；被跟踪目标 = 该发射幅度最大的节点。
BASE_FEATS = ["logn", "pri_eff", "pw", "jitter", "amp", "amp_rel", "amp_rel_roll", "amp_sd",
              "log_gap_prev", "log_gap_next", "burst_n", "burst_pulses", "burst_pos", "drf_prev",
              "drf_next", "type_code", "sensor_air", "pw_rel"]
CTX_FEATS = ["has_pred", "has_succ", "amp_d_pred", "amp_d_succ", "chain_len", "chain_pos", "chain_rpos",
             "enodes", "amp_minus_emax", "amp_rank_e"]
P_FEATS = ["pf_run", "pb_run", "p_tot", "p_tot_l", "p_node", "p_half"]
STATE_FEATS = BASE_FEATS + CTX_FEATS + P_FEATS
EP_FEATS = ["ek", "etin", "elen", "erem", "egap"]
STATE2_FEATS = STATE_FEATS + EP_FEATS


def _revisit_period(em: pd.DataFrame, lo=0.05e6, hi=2.0e6, bin_us=5000.0) -> float:
    """跟踪重访周期：同 (对象, 节点) 相邻短脉宽驻留间隔的众数。

    取全场景全局估计：单对象估计会被密集的短脉宽搜索驻留带偏（训练集上实测会估出 54/82 ms）。
    """
    em = em.sort_values("t0")
    d = pd.DataFrame({"obj": em.obj.values, "dt": em.groupby(["obj", "node"]).t0.diff().values}).dropna()
    d = d[(d.dt >= lo) & (d.dt <= hi)]
    if len(d) < 20:
        return 0.4e6
    edges = np.arange(lo, hi + bin_us, bin_us)
    # 每对象直方图做 ±1 格平滑（真值周期恰在格边界时不被劈成两半），按对象内占比求和（每对象等权）：
    # 跟踪重访周期为多部雷达共有且各自集中，某部雷达自身的短脉宽搜索节拍（如 53.76 ms）只在个别对象上占优。
    # 训练集三折得分 400 ms ≈2.8–3.8，次高 ≈1.3；改为全局计数时去掉一个节点会误选 53.76 ms。
    score = np.zeros(len(edges) - 1)
    for _, x in d.groupby("obj"):
        h = np.histogram(x.dt.values, bins=edges)[0].astype(float)
        score += (h + np.r_[0, h[:-1]] + np.r_[h[1:], 0]) / len(x)
    i = int(np.argmax(score))
    c = 0.5 * (edges[i] + edges[i + 1])
    sel = np.abs(d.dt.values - c) <= 1.5 * bin_us
    return float(np.median(d.dt.values[sel]))


PW_TRACK_MAX = 1.3   # 跟踪驻留固定用最短一档脉宽：相对脉宽高于此的 CPI 不作为周期命中的候选


def period_hit_features(d: pd.DataFrame, M: int = 5, tight_us: float = 1000.0, loose_frac: float = 0.1) -> pd.DataFrame:
    """跟踪驻留严格按重访周期 P 重复（P 由数据估计）：统计同对象短脉宽驻留在 t±mP（m=1..M）处的精确命中。

    搜索驻留即便用到最短脉宽，也只会偶然落在 ±1 ms 内；跟踪驻留则几乎逐周期命中。
    pf_run/pb_run：向后/向前连续命中数；p_tot：精确命中总数；p_tot_l：±0.1P 宽松命中（容忍调度相位跳变）；
    p_node：同节点精确命中；p_half：±P/2 处命中（交错的另一条跟踪）。
    """
    n = len(d)
    out = np.zeros((n, len(P_FEATS)))
    cand = d.pw_rel.values < PW_TRACK_MAX
    if cand.sum() < 2:
        return pd.DataFrame(out, index=d.index, columns=P_FEATS)
    t = d.t0.values.astype(float)
    of = pd.factorize(d.object_id.values)[0]
    on = pd.factorize(pd.Series(of).astype(str).values + "_" + pd.Series(d.platform_id.values).astype(str).values)[0]
    P = _revisit_period(pd.DataFrame({"obj": of[cand], "node": d.platform_id.values[cand], "t0": t[cand]}))
    key, keyn = of * 1e13 + t, on * 1e13 + t            # 按对象（对象×节点）分块后整体有序
    kc, kcn = np.sort(key[cand]), np.sort(keyn[cand])

    def hit(base, arr, off, tol):
        q = base + off
        return (np.searchsorted(arr, q + tol, side="right") > np.searchsorted(arr, q - tol)).astype(float)

    fr, br = np.ones(n), np.ones(n)
    for m in range(1, M + 1):
        hf, hb = hit(key, kc, m * P, tight_us), hit(key, kc, -m * P, tight_us)
        fr, br = fr * hf, br * hb
        out[:, 0] += fr
        out[:, 1] += br
        out[:, 2] += hf + hb
        out[:, 3] += hit(key, kc, m * P, loose_frac * P) + hit(key, kc, -m * P, loose_frac * P)
        out[:, 4] += hit(keyn, kcn, m * P, tight_us) + hit(keyn, kcn, -m * P, tight_us)
    out[:, 5] = hit(key, kc, P / 2, tight_us) + hit(key, kc, -P / 2, tight_us)
    return pd.DataFrame(out, index=d.index, columns=P_FEATS)


def state_features(cp: pd.DataFrame, type_codes: dict) -> pd.DataFrame:
    """cp 需含 local_cluster_id、obj_type、emission、object_id 列；返回按 cp.index 对齐的特征表。"""
    from .anchor import DSU, contiguity_links

    d = cp.sort_values(["local_cluster_id", "t0"]).copy()
    g = d.groupby("local_cluster_id", sort=False)
    d["gap_prev"] = d.t0 - g.t1.shift()
    d["gap_next"] = g.t0.shift(-1) - d.t1
    d["log_gap_prev"] = np.log10(d.gap_prev.clip(lower=1).fillna(1e9))
    d["log_gap_next"] = np.log10(d.gap_next.clip(lower=1).fillna(1e9))
    d["drf_prev"] = (d.rf - g.rf.shift()).abs().fillna(-1)
    d["drf_next"] = (d.rf - g.rf.shift(-1)).abs().fillna(-1)
    newb = (d.gap_prev.isna()) | (d.gap_prev > 5000.0)
    d["burst"] = newb.cumsum()
    gb = d.groupby("burst")
    d["burst_n"] = gb.n.transform("size")
    d["burst_pulses"] = gb.n.transform("sum")
    d["burst_pos"] = gb.cumcount()
    d["amp_rel"] = d.amp - g.amp.transform("median")
    d["tsec"] = d.t0 / 1e6
    roll = []
    for _, x in d.groupby("local_cluster_id", sort=False):
        s = pd.Series(x.amp.values, index=pd.to_datetime(x.tsec.values, unit="s"))
        roll.append(pd.Series(s.rolling("30s", center=True).median().values, index=x.index))
    d["amp_rel_roll"] = d.amp - pd.concat(roll)
    d["logn"] = np.log2(d.n.clip(lower=1))
    d["pri_eff"] = d.pri.fillna(-1.0)
    d["amp_sd"] = d.amp_sd.fillna(0)
    d["type_code"] = d.obj_type.map(type_codes).fillna(-1)
    d["sensor_air"] = (d.sh > 100).astype(int)
    # 相对脉宽：雷达跟踪驻留固定使用其脉宽集中最短的一档（训练集 8 型全部成立），与具体型号参数无关
    big = d.n >= 8
    pmin = d[big].groupby("object_id").pw.quantile(0.02)
    d["pw_rel"] = d.pw / d.object_id.map(pmin).fillna(d.groupby("object_id").pw.transform("min"))
    d = d.loc[cp.index]
    # 首尾相接链（同节点，同一雷达连续发射）
    c = cp.reset_index(drop=True)
    L = contiguity_links(c)
    n = len(c)
    pred_of = np.full(n, -1)
    succ_of = np.full(n, -1)
    pred_of[L.b.values.astype(int)] = L.a.values.astype(int)
    succ_of[L.a.values.astype(int)] = L.b.values.astype(int)
    amp = c.amp.values
    d["has_pred"] = (pred_of >= 0).astype(float)
    d["has_succ"] = (succ_of >= 0).astype(float)
    d["amp_d_pred"] = np.where(pred_of >= 0, amp - amp[np.clip(pred_of, 0, None)], 0.0)
    d["amp_d_succ"] = np.where(succ_of >= 0, amp - amp[np.clip(succ_of, 0, None)], 0.0)
    dsu = DSU(n)
    for x, y in zip(L.a.values, L.b.values):
        dsu.p[dsu.find(int(y))] = dsu.find(int(x))
    ch = np.array([dsu.find(i) for i in range(n)])
    tmp = pd.DataFrame({"ch": ch, "t0": c.t0.values}).sort_values("t0")
    gc = tmp.groupby("ch")
    tmp["len"] = gc.t0.transform("size")
    tmp["pos"] = gc.cumcount()
    tmp = tmp.sort_index()
    d["chain_len"] = tmp["len"].values
    d["chain_pos"] = tmp["pos"].values
    d["chain_rpos"] = d.chain_len - 1 - d.chain_pos
    # 同一发射的多节点上下文
    ge = cp.groupby("emission")
    d["enodes"] = ge.platform_id.transform("nunique").values
    d["amp_minus_emax"] = (cp.amp - ge.amp.transform("max")).values
    d["amp_rank_e"] = ge.amp.rank(ascending=False).values
    d["emission"] = cp.emission.values
    d["object_id"] = cp.object_id.values
    ph = period_hit_features(d)
    for col in P_FEATS:
        d[col] = ph[col].values
    return d


def _emission_mean_prob(p: np.ndarray, emission: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = pd.Series(np.log(p / (1 - p))).groupby(emission).transform("mean").values
    return 1.0 / (1.0 + np.exp(-lo))


def episode_features(feat: pd.DataFrame, trk: np.ndarray, gap_s: float = 1.5) -> pd.DataFrame:
    """对每个 (对象, 被跟踪节点) 的跟踪驻留序列计算段内序号/时间（以发射为单位）。"""
    out = np.full((len(feat), len(EP_FEATS)), -1.0)
    d = pd.DataFrame({"obj": feat.object_id.values, "em": feat.emission.values, "t0": feat.t0.values,
                      "amp": feat.amp.values, "plat": feat.platform_id.values})[trk]
    if len(d):
        i_max = d.groupby("em").amp.idxmax()
        em = d.groupby("em").agg(obj=("obj", "first"), t0=("t0", "min"))
        em["node"] = d.plat.loc[i_max.loc[em.index].values].values
        em = em.sort_values("t0")
        g = em.groupby(["obj", "node"])
        em["gap"] = g.t0.diff() / 1e6
        em["new"] = em.gap.isna() | (em.gap > gap_s)
        em["ep"] = em.groupby(["obj", "node"]).new.cumsum()
        ge = em.groupby(["obj", "node", "ep"])
        em["ek"] = ge.cumcount()
        em["etin"] = (em.t0 - ge.t0.transform("min")) / 1e6
        em["elen"] = ge.t0.transform("size")
        em["erem"] = em.elen - 1 - em.ek
        em["egap"] = em.gap.fillna(99).clip(upper=99)
        vals = em[EP_FEATS].reindex(feat.emission.values).values
        m = trk & ~np.isnan(vals[:, 0])
        out[m] = vals[m]
    return pd.DataFrame(out, index=feat.index, columns=EP_FEATS)


CH_FEATS = ["ck", "ctin", "clen", "crem", "tdt_prev", "tdt_next", "node_same", "node_k", "ampn", "eamax"]


def chain_features(feat: pd.DataFrame, trk: np.ndarray, tol_frac: float = 0.075) -> pd.DataFrame:
    """跟踪驻留链：同一对象的跟踪驻留按重访周期首尾衔接（不依赖幅度最大节点，避免节点跳变把一段跟踪切碎）。

    ck/ctin：链内序号/时间；node_same/node_k：链上前驱是否同一被跟踪节点、同节点连续步数；
    ampn：被跟踪节点幅度相对该节点常态（主瓣照射强弱）；tdt_*：与同对象相邻跟踪驻留的时间间隔。
    """
    out = np.full((len(feat), len(CH_FEATS)), -1.0)
    d = pd.DataFrame({"obj": feat.object_id.values, "em": feat.emission.values, "t0": feat.t0.values,
                      "amp": feat.amp.values, "plat": feat.platform_id.values})[trk]
    if not len(d):
        return pd.DataFrame(out, index=feat.index, columns=CH_FEATS)
    i_max = d.groupby("em").amp.idxmax()
    em = d.groupby("em").agg(obj=("obj", "first"), t0=("t0", "min"))
    em["node"] = d.plat.loc[i_max.loc[em.index].values].values
    em["amax"] = d.amp.loc[i_max.loc[em.index].values].values
    em = em.sort_values(["obj", "t0"])
    P = _revisit_period(em)
    tol = tol_frac * P
    ck, ctin, clen, crem, nsame, nk = (np.zeros(len(em)) for _ in range(6))
    tprev, tnext = np.full(len(em), 99.0), np.full(len(em), 99.0)
    pos = 0
    for _, x in em.groupby("obj", sort=False):
        t, nd = x.t0.values, x.node.values
        n = len(t)
        j = np.searchsorted(t, t - P - tol)
        jj = np.clip(j, 0, n - 1)
        prev = np.where((j < np.arange(n)) & (np.abs(t[jj] - (t - P)) <= tol), jj, -1)
        k = np.zeros(n); st = t.copy(); same = np.zeros(n); kn = np.zeros(n); root = np.arange(n)
        for i in range(n):
            q = prev[i]
            if q >= 0:
                k[i] = k[q] + 1; st[i] = st[q]; root[i] = root[q]
                same[i] = float(nd[q] == nd[i]); kn[i] = kn[q] + 1 if nd[q] == nd[i] else 0
        sl = slice(pos, pos + n)
        ck[sl], ctin[sl], nsame[sl], nk[sl] = k, (t - st) / 1e6, same, kn
        cl = pd.Series(root).map(pd.Series(root).value_counts()).values
        clen[sl], crem[sl] = cl, cl - 1 - k
        dtp = np.diff(t) / 1e6
        tprev[pos + 1:pos + n] = dtp; tnext[pos:pos + n - 1] = dtp
        pos += n
    em["ck"], em["ctin"], em["clen"], em["crem"] = ck, ctin, clen, crem
    em["tdt_prev"], em["tdt_next"] = np.minimum(tprev, 99), np.minimum(tnext, 99)
    em["node_same"], em["node_k"] = nsame, nk
    em["ampn"] = em.amax - em.groupby(["obj", "node"]).amax.transform("median")
    em["eamax"] = em.amax
    vals = em[CH_FEATS].reindex(feat.emission.values).values
    m = trk & ~np.isnan(vals[:, 0])
    out[m] = vals[m]
    return pd.DataFrame(out, index=feat.index, columns=CH_FEATS)


G_FEATS = ["gk", "gtin", "grem", "gpred", "gsucc", "glen"]


def link_dwells(dw: pd.DataFrame, P: float, tol: float = 0.2, mmax: int = 3, node_pen: float = 0.2,
                miss_pen: float = 0.3) -> pd.DataFrame:
    """驻留全局配对：同对象候选 (前驱, 后继) 间隔 ≈ mP（m≤mmax 容忍漏收，±tol·P 容忍调度相位跳变），
    代价 = 相位偏差 + 漏收数惩罚 + 被跟踪节点变化惩罚；按代价升序贪心，每个驻留至多一个前驱/后继。

    比"就近首个匹配"更稳：交错跟踪、新跟踪插入导致相位跳变时，不会把两条跟踪串成一条。
    dw 需含 obj、t、node；返回 k（链内周期序号）、tin（链内时间 s）、rem（距链尾周期数）、haspred、hassucc。
    """
    t, nd, ob = dw.t.values.astype(float), dw.node.values, dw.obj.values
    order = np.lexsort((t, ob))
    t, nd, ob = t[order], nd[order], ob[order]
    N = len(t)
    key = pd.factorize(ob)[0] * 1e13 + t
    I, J, C, Mm = [], [], [], []
    for m in range(1, mmax + 1):
        lo = np.searchsorted(key, key - m * P - tol * P)
        hi = np.searchsorted(key, key - m * P + tol * P, side="right")
        for i in np.where(hi > lo)[0]:
            for j in range(lo[i], min(hi[i], i)):
                I.append(i); J.append(j); Mm.append(m)
                C.append(abs(t[i] - t[j] - m * P) / P + (m - 1) * miss_pen + (node_pen if nd[i] != nd[j] else 0.0))
    pred, pm, hs = np.full(N, -1), np.zeros(N), np.zeros(N, bool)
    for q in np.argsort(np.asarray(C), kind="stable"):
        i, j = I[q], J[q]
        if pred[i] < 0 and not hs[j]:
            pred[i], pm[i], hs[j] = j, Mm[q], True
    k, st, root = np.zeros(N), t.copy(), np.arange(N)
    for i in range(N):              # 对象内时间序，前驱必在前
        j = pred[i]
        if j >= 0:
            k[i], st[i], root[i] = k[j] + pm[i], st[j], root[j]
    kmax = pd.Series(k).groupby(root).transform("max").values
    out = pd.DataFrame({"k": k, "tin": (t - st) / 1e6, "rem": kmax - k, "haspred": (pred >= 0).astype(float),
                        "hassucc": hs.astype(float)}, index=dw.index.values[order])
    return out.reindex(dw.index)


def global_chain_features(feat: pd.DataFrame, trk: np.ndarray, merge_us: float = 1500.0) -> pd.DataFrame:
    """跟踪 CPI 合并为驻留（同对象起始相差 ≤1.5 ms，即不同节点接收同一次驻留），再做全局配对成链。"""
    out = np.full((len(feat), len(G_FEATS)), -1.0)
    d = pd.DataFrame({"obj": feat.object_id.values, "t0": feat.t0.values, "amp": feat.amp.values,
                      "plat": feat.platform_id.values, "row": np.arange(len(feat))})[trk]
    if len(d) < 2:
        return pd.DataFrame(out, index=feat.index, columns=G_FEATS)
    d = d.sort_values(["obj", "t0"])
    d["dw"] = ((d.obj != d.obj.shift()) | (d.t0.diff() > merge_us)).cumsum().values
    g = d.groupby("dw")
    dw = g.agg(obj=("obj", "first"), t=("t0", "min"))
    dw["node"] = d.plat.loc[g.amp.idxmax().values].values
    P = _revisit_period(dw.rename(columns={"t": "t0"}))
    c = link_dwells(dw, P)
    v = np.c_[c.k, c.tin, c.rem, c.haspred, c.hassucc, c.k + c.rem + 1]
    out[d.row.values] = v[np.searchsorted(dw.index.values, d.dw.values)]
    return pd.DataFrame(out, index=feat.index, columns=G_FEATS)


USE_CHAIN = True
MASK_STAGE1 = False  # 伪影只影响段内序号类特征（第二级）；第一级屏蔽验证略差
OPT_PWRULE = False  # 验证无增益（GBDT 已由脉宽+型号学到），保留开关


def _stage2(feat: pd.DataFrame, trk: np.ndarray):
    parts = [feat, episode_features(feat, trk)]
    cols = STATE2_FEATS
    if USE_CHAIN:
        parts += [chain_features(feat, trk), global_chain_features(feat, trk)]
        cols = STATE2_FEATS + CH_FEATS + G_FEATS
    return pd.concat(parts, axis=1), cols, feat.emission.values


def _hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def fit_state_models(feat: pd.DataFrame, y: pd.Series, keep: np.ndarray | None = None):
    """keep：参与训练的样本（屏蔽数据中段截断处的伪影样本）；特征仍在全量上计算。"""
    w0 = np.log2(feat.n.values + 1)
    w = np.where(keep, w0, 0.0) if keep is not None else w0
    yv = y.values
    m1 = _hgb().fit(feat[STATE_FEATS].values, (yv > 0).astype(int), sample_weight=w if MASK_STAGE1 else w0)
    p = _emission_mean_prob(m1.predict_proba(feat[STATE_FEATS].values)[:, 1], feat.emission.values)
    trk = p > 0.5
    f2, cols, _ = _stage2(feat, trk)
    sel = trk & (yv > 0) & (w > 0)
    m2 = _hgb().fit(f2[cols].values[sel], (yv[sel] == 2).astype(int), sample_weight=w[sel])
    return m1, m2


def predict_states(m1, m2, feat: pd.DataFrame, oracle: np.ndarray | None = None) -> pd.Series:
    p = _emission_mean_prob(m1.predict_proba(feat[STATE_FEATS].values)[:, 1], feat.emission.values)
    trk = p > 0.5 if oracle is None else oracle   # oracle：仅供分析（真值第一级）
    f2, cols, grp = _stage2(feat, trk)
    p2 = _emission_mean_prob(m2.predict_proba(f2[cols].values)[:, 1], grp)
    st = np.where(trk, np.where(p2 > 0.5, 2, 1), 0)
    if OPT_PWRULE:
        st = np.where(feat.pw_rel.values > PW_TRACK_MAX, 0, st)   # 非最短脉宽 → 必为搜索
    return pd.Series(st, index=feat.index, dtype=np.int64)
