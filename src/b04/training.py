"""可复用的模型训练：输入带真值的 A/B 数据（任意时间子集），输出与推理一致的模型字典。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn

from .cpi import extract_cpis
from .fingerprint import fit_fingerprint
from .pipeline import cluster_and_type
from .recognize import fit_state_models, fit_type_model, state_features


def cpi_labels(a: pd.DataFrame, cpi_of: np.ndarray) -> pd.DataFrame:
    x = pd.DataFrame({"cpi": cpi_of, "tid": a.target_id.values, "cls": a.target_class.values,
                      "st": a.target_state.values})
    return x.groupby("cpi").agg(tid=("tid", lambda s: s.mode().iloc[0]), cls=("cls", lambda s: s.mode().iloc[0]),
                                st=("st", lambda s: s.mode().iloc[0]))


def artifact_mask(t0: np.ndarray, gap_us: float = 10e6, span_us: float = 6e6) -> np.ndarray:
    """训练数据中段起点（前有 >10 s 空洞且不是场景零点，如时间分折训练）之后 6 s 内的样本。

    这些位置上正在进行的跟踪段被截断，"段内序号"类特征会把精跟误呈现为新段开头，作为训练样本是伪影，需屏蔽；
    完整场景（零点起、无空洞）训练时不屏蔽任何样本。
    """
    u = np.sort(np.asarray(t0, float))
    m = np.zeros(len(t0), bool)
    if not len(u):
        return m
    starts = list(u[1:][np.diff(u) > gap_us])
    if u[0] > 1e6:
        starts.append(u[0])
    for s0 in starts:
        m |= (t0 >= s0) & (t0 < s0 + span_us)
    return m


def fit_models(a: pd.DataFrame, b: pd.DataFrame, knowledge: dict, cfg: dict) -> dict:
    """a 需含 target_id/target_class/target_state；状态模型在预测得到的局部簇上训练（与推理一致）。"""
    a = a.reset_index(drop=True)
    cpi_of, cp = extract_cpis(a, cfg)
    cp = cp.join(cpi_labels(a, cpi_of), on="cpi")
    classes = sorted(knowledge["models"].keys())
    models = {"type": fit_type_model(cp, cp.cls), "fp": fit_fingerprint(cp, cp.cls),
              "type_codes": {c: i for i, c in enumerate(classes)}}
    cpi_of2, cp2, _ = cluster_and_type(a, models, knowledge, cfg, b)
    cp2 = cp2.join(cpi_labels(a, cpi_of2), on="cpi")
    feat = state_features(cp2, models["type_codes"])
    m1, m2 = fit_state_models(feat, cp2.st, ~artifact_mask(cp2.t0.values))
    models.update({"state1": m1, "state2": m2, "sklearn": sklearn.__version__})
    return models
