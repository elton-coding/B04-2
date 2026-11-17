"""单场景端到端流水线：A/B 数据 → 两张结果表（+ 中间结果供本体/报告使用）。"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .anchor import (anchor_assign, build_emissions, build_groups, contiguity_links, propagate_labels,
                     resolve_conflicts, veto_by_fragment)
from .association import associate, scenario_base_time, track_table
from .cpi import extract_cpis
from .fingerprint import loglik, proba_from_ll
from .refine import refine_objects
from .recognize import predict_states, predict_type_proba, state_features, vote_object_type
from .tracking import absorb_small, build_fragments, fuse_fragments, match_cross_platform, merge_objects

log = logging.getLogger("b04")


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self, msg):
        now = time.time()
        log.info("%-28s %6.1f s", msg, now - self.t)
        self.t = now


def _unify_by_emission(lab: np.ndarray, emis: np.ndarray) -> np.ndarray:
    """同一发射的各节点 CPI 取同一标签（多数）。"""
    d = pd.DataFrame({"e": emis, "lab": lab})
    m = d[d.lab >= 0].groupby("e").lab.agg(lambda s: s.value_counts().index[0])
    out = pd.Series(emis).map(m).fillna(-1).astype(np.int64).values
    return np.where(out >= 0, out, lab)


def cluster_and_type(a: pd.DataFrame, models: dict, knowledge: dict, cfg: dict, b: pd.DataFrame | None = None):
    """L2-L4 + L6：CPI 提取、航迹锚定、局部聚类、跨平台融合、型号判定。返回 (cpi_of, cp, proba)。"""
    tm = Timer()
    cpi_of, cp = extract_cpis(a, cfg)
    tm.lap(f"CPI 提取 ({len(cp)})")
    if cfg["recognize"]["type_model"] == "fingerprint" and "fp" in models:
        proba = proba_from_ll(loglik(models["fp"], cp), cp.n.values)
    else:
        proba = predict_type_proba(models["type"], cp)
    cp["ptype"] = proba.idxmax(axis=1)
    domain_air = {k: v.get("domain") == "air" for k, v in knowledge.get("models", {}).items()}
    frag = build_fragments(cp, proba, domain_air, cfg)
    pairs = match_cross_platform(cp, cfg)
    emis = build_emissions(cp, pairs)
    groups = build_groups(cp, emis, contiguity_links(cp))
    tm.lap(f"片段 {frag.max() + 1} / 发射 {emis.max() + 1} / 驻留组 {groups.max() + 1}")

    lab = np.full(len(cp), -1, np.int64)
    ttype = {}
    R = rgate = None
    if cfg["anchor"].get("enabled", True) and b is not None and len(b):
        tracks = track_table(b, scenario_base_time(a))
        lab, info = anchor_assign(cp, proba, groups, tracks, knowledge, cfg)
        lab = resolve_conflicts(cp, lab, groups, info["score"], cfg)
        if cfg["anchor"].get("frag_veto", True):
            lab = veto_by_fragment(cp, lab, frag, info["R"], info["gate"], cfg["anchor"].get("veto_min_hit", 0.6))
        lab = _unify_by_emission(lab, groups)
        cp["anchor_direct"] = lab  # 直接锚定（未经传播），精修时锁定
        lab = propagate_labels(cp, lab, frag, proba, info["ttype"], info["tids"], cfg, info["R"], info["gate"])
        lab = _unify_by_emission(lab, groups)
        ttype = info["ttype"]
        R, rgate = info["R"], info["gate"]
        w = cp.n.values
        tm.lap(f"航迹锚定 (锚定脉冲 {w[lab >= 0].sum() / w.sum():.1%}, 航迹型号 {len(ttype)})")
    cp["anchor"] = lab
    tids_arr = np.array(info["tids"]) if (lab >= 0).any() else np.array([-1])
    cp["anchor_track"] = np.where(lab >= 0, tids_arr[np.clip(lab, 0, len(tids_arr) - 1)], -1)

    # 未锚定部分：节点内串接 → 同源融合
    obj = np.full(len(cp), -1, np.int64)
    obj[lab >= 0] = lab[lab >= 0]
    rest = np.where(lab < 0)[0]
    if len(rest):
        cr = cp.iloc[rest].reset_index(drop=True)
        pr = proba.iloc[rest].reset_index(drop=True)
        fr = build_fragments(cr, pr, domain_air, cfg)
        prs = pairs[pairs.cpi_a.isin(cr.cpi) & pairs.cpi_b.isin(cr.cpi)]
        ob = fuse_fragments(cr, fr, prs, cfg)
        obj[rest] = ob + (lab.max() + 1 if (lab >= 0).any() else 0)
    obj = pd.factorize(obj)[0]
    cp["object_id"] = obj
    ot = vote_object_type(cp, proba)
    obj = merge_objects(cp, obj, ot, cfg, proba)
    obj = absorb_small(cp, obj, proba, cfg)
    if cfg.get("refine", {}).get("enabled", True) and R is not None:  # 无航迹时仅靠方位连续，证据不足，验证显示有害
        t_r = time.time()
        locked = cp["anchor_direct"].values >= 0 if "anchor_direct" in cp.columns else None
        obj = refine_objects(cp, obj, groups, proba, cfg, R, rgate, locked)
        obj = absorb_small(cp, obj, proba, cfg)
        log.info("对象级精修 %.1f s", time.time() - t_r)
    cp["object_id"] = obj
    tm.lap(f"对象形成 ({obj.max() + 1})")
    cp["frag"] = frag
    cp["emission"] = emis
    cp["obj_type"] = cp.object_id.map(vote_object_type(cp, proba))
    cp["local_cluster_id"] = cp.groupby(["object_id", "platform_id"], sort=True).ngroup().values
    return cpi_of, cp, proba


def run_scenario(a: pd.DataFrame, b: pd.DataFrame, models: dict, knowledge: dict, cfg: dict):
    tm = Timer()
    cpi_of, cp, proba = cluster_and_type(a, models, knowledge, cfg, b)
    feat = state_features(cp, models["type_codes"])
    cp["state"] = predict_states(models["state1"], models["state2"], feat)
    tm.lap("状态识别")
    base = scenario_base_time(a)
    # 锚定结果作为关联先验：对象过半脉冲锚定于同一航迹时优先该航迹
    prefer = {}
    if (cp.anchor_track >= 0).any():
        w = cp[cp.anchor_track >= 0].groupby(["object_id", "anchor_track"]).n.sum()
        tot = cp.groupby("object_id").n.sum()
        for (o, t), v in w.items():
            if v >= 0.5 * tot.loc[o]:
                prefer[o] = int(t)
    links, scores = associate(cp, cp.groupby("object_id").obj_type.first(), b, base, knowledge, cfg, prefer)
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
