"""本地评分器（官方公式未公布，多指标并行，均对编号置换不敏感）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score, v_measure_score


def _pair_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    ct = pd.crosstab(truth, pred).values.astype(np.float64)
    tp = (ct * (ct - 1) / 2).sum()
    a = ct.sum(1)
    b = ct.sum(0)
    tp_fp = (b * (b - 1) / 2).sum()
    tp_fn = (a * (a - 1) / 2).sum()
    p = tp / tp_fp if tp_fp else 0.0
    r = tp / tp_fn if tp_fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _majority_acc(truth: np.ndarray, pred: np.ndarray) -> float:
    """每个预测簇映射到其多数真值后的准确率（纯度）。"""
    ct = pd.crosstab(pred, truth)
    return ct.max(axis=1).sum() / ct.values.sum()


def _macro(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """按簇宏平均（小簇与大簇同权）：(每个真值簇的最佳 F1 均值, 每个预测簇的纯度均值)。"""
    d = pd.DataFrame({"t": truth, "p": pred}).groupby(["t", "p"]).size()
    nt = d.groupby(level=0).sum()
    npd = d.groupby(level=1).sum()
    f1 = 2 * d / (nt.reindex(d.index.get_level_values(0)).values + npd.reindex(d.index.get_level_values(1)).values)
    return float(f1.groupby(level=0).max().mean()), float((d.groupby(level=1).max() / npd).mean())


def score(a: pd.DataFrame, b: pd.DataFrame, pred: pd.DataFrame, links: pd.DataFrame) -> dict:
    """a 需含 target_id/target_class/target_state 真值；b 需含 target_id。"""
    m = a[["pulse_id", "platform_id", "target_id", "target_class", "target_state"]].merge(
        pred, on="pulse_id", suffixes=("", "_p"))
    out = {}
    ari, pf1 = [], []
    for _, x in m.groupby("platform_id"):
        ari.append(adjusted_rand_score(x.target_id, x.local_cluster_id))
        pf1.append(_pair_f1(x.target_id.values, x.local_cluster_id.values))
    out["cluster_ARI"] = float(np.mean(ari))
    out["cluster_pairF1"] = float(np.mean(pf1))
    out["cluster_purity"] = float(_majority_acc(m.target_id.values, m.local_cluster_id.values))
    lt = m.platform_id.astype(str) + "/" + m.target_id.astype(str)
    lp = m.platform_id.astype(str) + "/" + m.local_cluster_id.astype(str)
    out["cluster_macroF1"], out["cluster_macroPur"] = _macro(lt.values, lp.values)
    out["fusion_macroF1"], out["fusion_macroPur"] = _macro(m.target_id.values, m.object_id.values)
    out["type_acc"] = float((m.target_class == m.target_class_p).mean())
    out["state_acc"] = float((m.target_state == m.target_state_p).mean())
    out["state_macroF1"] = float(f1_score(m.target_state, m.target_state_p, average="macro"))
    out["fusion_ARI"] = float(adjusted_rand_score(m.target_id, m.object_id))
    out["fusion_pairF1"] = float(_pair_f1(m.target_id.values, m.object_id.values))
    out["fusion_V"] = float(v_measure_score(m.target_id, m.object_id))
    out["n_objects"] = int(m.object_id.nunique())
    out["n_true"] = int(m.target_id.nunique())
    # 关联：对象→多数真值目标；判定其关联航迹是否正确（或正确 no_match）
    if "target_id" in b.columns and b.target_id.notna().any():
        trk2tid = b.groupby("track_id").target_id.first().to_dict()
        tids_with_track = set(b.target_id.dropna().astype(np.int64))
        maj = m.groupby("object_id").target_id.agg(lambda s: s.value_counts().index[0])
        size = m.groupby("object_id").size()
        ok, w = [], []
        for r in links.itertuples(index=False):
            t = maj.get(r.object_id)
            if r.link_status == "matched":
                good = trk2tid.get(int(r.track_id)) == t
            else:
                good = t not in tids_with_track
            ok.append(good)
            w.append(size.get(r.object_id, 0))
        ok = np.array(ok, float)
        out["link_acc_obj"] = float(ok.mean())
        out["link_acc_pulse"] = float(np.average(ok, weights=w))
        # 航迹视角：每条航迹的判定是否正确
        tp = {int(r.track_id): maj.get(r.object_id) for r in links.itertuples(index=False) if r.link_status == "matched"}
        tk = []
        for trk, tid in trk2tid.items():
            tid_has_pulses = tid in set(m.target_id)
            if trk in tp:
                tk.append(tp[trk] == tid)
            else:
                tk.append(not tid_has_pulses)
        out["link_acc_track"] = float(np.mean(tk))
    return out
