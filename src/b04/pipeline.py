"""单场景端到端流水线：A/B 数据 → 两张结果表（+ 中间结果供本体/报告使用）。"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .anchor import (anchor_assign, build_emissions, build_groups, contiguity_links, propagate_labels,
                     resolve_conflicts, track_residuals, veto_by_fragment)
from .association import associate, scenario_base_time, track_table
from .cpi import extract_cpis
from .fingerprint import loglik, proba_from_ll
from .refine import refine_objects
from .tdoa import prepare_tdoa
from .exclusion import (absorb_tiny_local, dissolved_remnants, episode_repair, exclusion_repair, geometry_veto_repair,
                        phase_repair)
from .recognize import predict_states, predict_type_proba, state_features, vote_object_type
from .tracking import absorb_small, build_fragments, fuse_fragments, match_cross_platform, merge_objects, merge_split_objects

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


def _bearing_confirmed(cp, tracks, dl: dict, sig, gate, ac: dict) -> dict:
    """TDOA 估出的航迹时间偏移中，方位残差也支持的那些（平移后半门限内的 CPI 加权数 ≥ 平移前 × (1+min_gain)）。"""
    out = {}
    w = cp.n.values.astype(float)
    for tid, d in dl.items():
        x = tracks[tracks.track_id == tid]
        r0, _ = track_residuals(cp, x, sig, ac.get("pos_sigma_m", 0.0))
        r1, _ = track_residuals(cp, x.assign(t_us=x.t_us + d * 1e6), sig, ac.get("pos_sigma_m", 0.0))
        m0 = float((w * (np.abs(np.nan_to_num(r0[:, 0], nan=99.0)) <= 0.5 * gate)).sum())
        m1 = float((w * (np.abs(np.nan_to_num(r1[:, 0], nan=99.0)) <= 0.5 * gate)).sum())
        if m1 >= m0 * (1.0 + ac.get("offset_bearing_gain", 0.05)):
            out[tid] = d
    return out


def cluster_and_type(a: pd.DataFrame, models: dict, knowledge: dict, cfg: dict, b: pd.DataFrame | None = None,
                     extra: dict | None = None):
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
    pairs = match_cross_platform(cp, cfg, a.toa.values, cpi_of)
    emis = build_emissions(cp, pairs, frag if cfg["fusion"].get("emis_frag_support", True) else None)
    groups = build_groups(cp, emis, contiguity_links(cp))
    tm.lap(f"片段 {frag.max() + 1} / 发射 {emis.max() + 1} / 驻留组 {groups.max() + 1}")

    lab = np.full(len(cp), -1, np.int64)
    ttype = {}
    R = rgate = None
    tdoa = None
    track_dl = {}
    if cfg["anchor"].get("enabled", True) and b is not None and len(b):
        tracks = track_table(b, scenario_base_time(a))
        if cfg["anchor"].get("tdoa_w", 0) > 0:
            ac = cfg["anchor"]
            sig = np.where(cp.sh.values > 100, ac["sigma_air"], ac["sigma_sea"])
            R0, _ = track_residuals(cp, tracks, sig, ac.get("pos_sigma_m", 0.0))
            g0 = np.where(cp.sh.values > 100, ac["gate_air"], ac["gate_sea"])
            ep, Rt, dl = prepare_tdoa(cp, emis, a.toa.values, cpi_of, tracks, np.abs(np.nan_to_num(R0, nan=99.0)) <= g0[:, None], cfg)
            tdoa = (ep, Rt)
            if dl and cfg["anchor"].get("offset_to_bearing", True):
                # 航迹时间戳偏移同样作用于方位几何（锚定、精修、关联），否则空中航迹的方位残差随偏移系统性增大。
                # 只采用方位也能印证的偏移：平移后残差落在半门限内的 CPI（按脉冲数加权）须明显增多
                dlb = _bearing_confirmed(cp, tracks, dl, sig, g0, ac)
                if dlb:
                    tracks = tracks.assign(t_us=tracks.t_us + tracks.track_id.map(dlb).fillna(0.0).values * 1e6)
                    track_dl.update(dlb)
            tm.lap(f"TDOA 配对 ({len(ep)})，航迹时间偏移 {dl}，方位采用 {track_dl}")
        lab, info = anchor_assign(cp, proba, groups, tracks, knowledge, cfg, tdoa)
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
        obj = refine_objects(cp, obj, groups, proba, cfg, R, rgate, locked, tdoa)
        obj = absorb_small(cp, obj, proba, cfg)
        log.info("对象级精修 %.1f s", time.time() - t_r)
    if cfg.get("exclusion", {}).get("enabled", True):
        t_r = time.time()
        obj0 = obj
        obj, moved = exclusion_repair(cp, obj, groups, proba, cfg, R, rgate)
        if moved:
            gone = dissolved_remnants(obj0, obj, cp.n.values, cfg["exclusion"].get("dissolve_frac", 0.7))
            obj = absorb_small(cp, obj, proba, cfg, gone)
        log.info("时间互斥重分配：改归 %d 脉冲，%.1f s", moved, time.time() - t_r)
        if cfg["exclusion"].get("episode", True):
            t_r = time.time()
            obj, moved = episode_repair(cp, obj, groups, proba, cfg, R, rgate)
            log.info("整段补洞：改归 %d 脉冲，%.1f s", moved, time.time() - t_r)
    if cfg.get("merge_split", {}).get("enabled", True):
        obj, nm = merge_split_objects(cp, obj, proba, cfg)
        log.info("拆分对象重合并：%d 次", nm)
    if R is not None:
        obj, mv = geometry_veto_repair(cp, obj, proba, tracks, cfg)
        log.info("几何否决改归：%d 脉冲", mv)
        obj, mv = phase_repair(cp, obj, proba, tracks, cfg)
        log.info("到达相位改归：%d 脉冲", mv)
    obj, mv = absorb_tiny_local(cp, obj, proba, cfg)
    log.info("节点碎簇改归：%d 脉冲", mv)
    cp["object_id"] = obj
    tm.lap(f"对象形成 ({obj.max() + 1})")
    cp["frag"] = frag
    cp["unit"] = groups
    cp["emission"] = emis
    cp["obj_type"] = cp.object_id.map(vote_object_type(cp, proba))
    cp["local_cluster_id"] = cp.groupby(["object_id", "platform_id"], sort=True).ngroup().values
    if extra is not None:
        extra["tdoa"] = tdoa
        extra["track_dl"] = track_dl
    return cpi_of, cp, proba


def run_scenario(a: pd.DataFrame, b: pd.DataFrame, models: dict, knowledge: dict, cfg: dict):
    tm = Timer()
    ext = {}
    cpi_of, cp, proba = cluster_and_type(a, models, knowledge, cfg, b, ext)
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
    dl = ext.get("track_dl") or {}
    if dl:
        b = b.assign(datetime=b.datetime + pd.to_timedelta(b.track_id.map(dl).fillna(0.0).values, unit="s"))
    links, scores = associate(cp, cp.groupby("object_id").obj_type.first(), b, base, knowledge, cfg, prefer,
                              ext.get("tdoa"))
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
    extras = {"cp": cp, "proba": proba, "scores": scores, "base_time": base, "cpi_of": cpi_of}
    return pred, links, extras
