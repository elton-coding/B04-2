"""L4 型号识别、L5 工作状态识别（CPI 级分类 + 对象级证据累积 + 时序平滑）。"""
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
STATE_FEATS = ["logn", "pri_eff", "pw", "jitter", "amp", "amp_rel", "amp_rel_roll", "amp_sd",
               "log_gap_prev", "log_gap_next", "burst_n", "burst_pulses", "burst_pos", "drf_prev",
               "drf_next", "type_code", "sensor_air"]
STATE2_FEATS = STATE_FEATS + ["p_track", "ep_t", "ep_k", "ep_len_t", "ep_frac"]


def state_features(cp: pd.DataFrame, type_codes: dict) -> pd.DataFrame:
    """cp 需含 local_cluster_id、obj_type 列；返回按 cp.index 对齐的特征表。"""
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
    # 30 s 滚动中位数幅度（相对抬升）
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
    return d.loc[cp.index]


def episode_features(d: pd.DataFrame, p_track: np.ndarray, gap_s: float = 3.0) -> pd.DataFrame:
    """跟踪段（连续 p_track>0.5 的 CPI）内的相对时间：粗跟通常位于段首。"""
    d = d.copy()
    d["p_track"] = p_track
    d = d.sort_values(["local_cluster_id", "t0"])
    trk = d.p_track > 0.5
    g = d.groupby("local_cluster_id", sort=False)
    prev_trk_t = d.t1.where(trk).groupby(d.local_cluster_id).ffill().groupby(d.local_cluster_id).shift()
    new_ep = trk & ((prev_trk_t.isna()) | ((d.t0 - prev_trk_t) > gap_s * 1e6))
    ep = new_ep.cumsum().where(trk, -1)
    d["ep"] = ep
    e = d[trk].groupby("ep")
    d["ep_t"] = np.nan
    d.loc[trk, "ep_t"] = (d.t0 - e.t0.transform("min")).loc[trk[trk].index] / 1e6
    d.loc[trk, "ep_k"] = e.cumcount()
    d.loc[trk, "ep_len_t"] = ((e.t1.transform("max") - e.t0.transform("min")) / 1e6).loc[trk[trk].index]
    d["ep_t"] = d.ep_t.fillna(-1)
    d["ep_k"] = d.ep_k.fillna(-1)
    d["ep_len_t"] = d.ep_len_t.fillna(-1)
    d["ep_frac"] = np.where(d.ep_len_t > 0, d.ep_t / d.ep_len_t.clip(lower=1e-3), -1)
    return d


def _hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)


def fit_state_models(feat: pd.DataFrame, y: pd.Series):
    w = np.log2(feat.n.values + 1)
    m1 = _hgb().fit(feat[STATE_FEATS].values, (y.values > 0).astype(int), sample_weight=w)
    p = m1.predict_proba(feat[STATE_FEATS].values)[:, 1]
    d = episode_features(feat, p)
    yy = y.loc[d.index]
    sel = (d.p_track > 0.5).values & (yy.values > 0)
    m2 = _hgb().fit(d[STATE2_FEATS].values[sel], (yy.values[sel] == 2).astype(int),
                    sample_weight=np.log2(d.n.values[sel] + 1))
    return m1, m2


def predict_states(m1, m2, feat: pd.DataFrame) -> pd.Series:
    p = m1.predict_proba(feat[STATE_FEATS].values)[:, 1]
    d = episode_features(feat, p)
    out = pd.Series(0, index=d.index, dtype=np.int64)
    trk = (d.p_track > 0.5).values
    if trk.any():
        p2 = m2.predict_proba(d[STATE2_FEATS].values[trk])[:, 1]
        out[trk] = np.where(p2 > 0.5, 2, 1)
    return out.loc[feat.index]
