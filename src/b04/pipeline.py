"""单场景端到端流水线：A/B 数据 → 两张结果表（+ 中间结果供本体/报告使用）。"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .association import associate, scenario_base_time
from .cpi import extract_cpis
from .recognize import predict_states, predict_type_proba, state_features, vote_object_type
from .tracking import build_fragments, fuse_fragments, match_cross_platform, merge_objects

log = logging.getLogger("b04")


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self, msg):
        now = time.time()
        log.info("%-28s %6.1f s", msg, now - self.t)
        self.t = now


def cluster_and_type(a: pd.DataFrame, models: dict, knowledge: dict, cfg: dict):
    """L2-L4 + L6：CPI 提取、局部聚类、跨平台融合、型号判定。返回 (cpi_of, cp, proba)。"""
    tm = Timer()
    cpi_of, cp = extract_cpis(a, cfg)
    tm.lap(f"CPI 提取 ({len(cp)})")
    proba = predict_type_proba(models["type"], cp)
    cp["ptype"] = proba.idxmax(axis=1)
    domain_air = {k: v.get("domain") == "air" for k, v in knowledge.get("models", {}).items()}
    frag = build_fragments(cp, proba, domain_air, cfg)
    tm.lap(f"节点内片段 ({frag.max() + 1})")
    pairs = match_cross_platform(cp, cfg)
    obj = fuse_fragments(cp, frag, pairs, cfg)
    tm.lap(f"跨平台融合 ({obj.max() + 1}, 同源对 {len(pairs)})")
    cp["object_id"] = obj
    ot = vote_object_type(cp, proba)
    obj = merge_objects(cp, obj, ot, cfg)
    cp["object_id"] = obj
    tm.lap(f"对象合并 ({obj.max() + 1})")
    cp["frag"] = frag
    cp["obj_type"] = cp.object_id.map(vote_object_type(cp, proba))
    cp["local_cluster_id"] = cp.groupby(["object_id", "platform_id"], sort=True).ngroup().values
    return cpi_of, cp, proba


def run_scenario(a: pd.DataFrame, b: pd.DataFrame, models: dict, knowledge: dict, cfg: dict):
    tm = Timer()
    cpi_of, cp, proba = cluster_and_type(a, models, knowledge, cfg)
    feat = state_features(cp, models["type_codes"])
    cp["state"] = predict_states(models["state1"], models["state2"], feat)
    tm.lap("状态识别")
    base = scenario_base_time(a)
    links, scores = associate(cp, cp.groupby("object_id").obj_type.first(), b, base, knowledge, cfg)
    tm.lap(f"A-B 关联 (matched {int((links.link_status == 'matched').sum())})")

    c = cp.set_index("cpi")
    sid, ver = str(a.scenario_id.iloc[0]), str(a.data_version.iloc[0])
    pred = pd.DataFrame({
        "scenario_id": sid, "data_version": ver, "pulse_id": a.pulse_id.values,
        "local_cluster_id": c.local_cluster_id.loc[cpi_of].values.astype(np.int64) + 1,
        "object_id": c.object_id.loc[cpi_of].values.astype(np.int64) + 1,
        "target_class": c.obj_type.loc[cpi_of].values,
        "target_state": c.state.loc[cpi_of].values.astype(np.int64),
    })
    links = links.assign(scenario_id=sid, data_version=ver, object_id=links.object_id.astype(np.int64) + 1)
    links = links[["scenario_id", "data_version", "object_id", "track_id", "link_status"]]
    extras = {"cp": cp, "proba": proba, "scores": scores, "base_time": base}
    return pred, links, extras
