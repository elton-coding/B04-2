"""B 榜兜底：输入清洗、分级降级运行、输出修复与应急基线。

B 榜在未公开的场景上直接运行容器，任何一处异常都会使该场景无输出（0 分）。这里保证"总能写出格式合法的结果"：
1. 输入清洗：数值列缺失 / 非法值按节点内时间顺序前后填补，缺失的可选列补默认值（不删除任何脉冲）；
2. 分级运行：完整流程 → 安全模式（关闭各后处理改归环节）→ 安全模式且不用 B 航迹 → 应急基线；
3. 输出修复：补齐缺失脉冲、空值，局部簇跨节点 / 跨对象时拆开或取多数，关联表去重并剔除不存在的航迹。
"""
from __future__ import annotations

import copy
import logging
import os
import time
import traceback

import numpy as np
import pandas as pd

log = logging.getLogger("b04")

A_NUM = ["toa", "param1", "param2", "param3", "doa", "sensor_longitude", "sensor_latitude", "sensor_height"]


def sanitize_a(a: pd.DataFrame) -> pd.DataFrame:
    """数值列中的 NaN / inf：同节点按 pulse_id 顺序前后填补；整节点缺失时用全局中位数。返回新表（行与顺序不变）。"""
    bad = ~np.isfinite(a[A_NUM].to_numpy(dtype=np.float64, na_value=np.nan))
    if "platform_id" in a.columns and a.platform_id.isna().any():
        a = a.assign(platform_id=a.platform_id.fillna(a.platform_id.mode().iloc[0]))
    n_bad = int(bad.any(axis=1).sum())
    if n_bad == 0:
        return a
    log.warning("A 类数据 %d 行数值缺失/非法，按节点内顺序填补", n_bad)
    a = a.copy()
    x = a[A_NUM].replace([np.inf, -np.inf], np.nan)
    order = np.lexsort((a.pulse_id.values, a.platform_id.values))
    xs = x.iloc[order]
    xs = xs.groupby(a.platform_id.values[order]).transform(lambda s: s.ffill().bfill())
    x.iloc[order] = xs.values
    a[A_NUM] = x.fillna(x.median()).fillna(0.0).values
    return a


def sanitize_b(b: pd.DataFrame) -> pd.DataFrame:
    """B 类：去掉坐标 / 时间无效的点（航迹点不在输出里，可删）；缺失的可选列补默认值。"""
    if b is None or not len(b):
        return b
    b = b.copy()
    for c, v in (("target_type", "unknown"), ("target_class", ""), ("target_height", 0.0)):
        if c not in b.columns:
            b[c] = v
    b["target_height"] = pd.to_numeric(b["target_height"], errors="coerce").fillna(0.0)
    ok = b.datetime.notna() & np.isfinite(b.target_longitude) & np.isfinite(b.target_latitude)
    if (~ok).any():
        log.warning("B 类数据 %d 个航迹点时间/坐标无效，已忽略", int((~ok).sum()))
        b = b[ok]
    # 少于 2 个点的航迹无法插值
    cnt = b.groupby("track_id").track_id.transform("size")
    return b[cnt >= 2].reset_index(drop=True)


def safe_cfg(cfg: dict) -> dict:
    """安全模式：保留 CPI、指纹、锚定、融合、状态、关联主干，关闭各后处理改归与自适应环节。"""
    c = copy.deepcopy(cfg)
    for sec in ("doa_calib", "refine", "exclusion", "geo_veto", "phase", "tiny_local", "merge_split", "locate"):
        if sec in c and isinstance(c[sec], dict):
            c[sec]["enabled"] = False
    c["track"]["frag_phase"] = False
    c["track"]["phase_merge"] = False
    c["anchor"]["tdoa_w"] = 0.0
    c["anchor"]["tdoa_time_offset"] = False
    c["link"]["tdoa_w"] = 0.0
    return c


def emergency_baseline(a: pd.DataFrame, knowledge: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """应急基线（主流程全部失败时）：每节点按 RF / 脉宽粗分档为局部簇，每个局部簇单独成对象，型号取知识库首个、
    状态全为搜索（训练集约 77% 脉冲），全部 no_match。只为保证格式合法、得到部分分数。"""
    sid, ver = str(a.scenario_id.iloc[0]), str(a.data_version.iloc[0])
    rf = np.round(a.param1.to_numpy(np.float64) / 50.0)
    pw = np.round(np.log2(np.maximum(a.param2.to_numpy(np.float64), 1e-3)))
    key = pd.Series(a.platform_id.astype(str).values) + "/" + pd.Series(rf.astype(np.int64).astype(str)) + "/" + pd.Series(pw.astype(np.int64).astype(str))
    lc = pd.factorize(key)[0].astype(np.int64) + 1
    typ = sorted((knowledge.get("models") or {"R01": {}}).keys())[0]
    pred = pd.DataFrame({"scenario_id": sid, "data_version": ver, "pulse_id": a.pulse_id.values, "local_cluster_id": lc,
                         "object_id": lc, "target_class": typ, "target_state": 0})
    links = pd.DataFrame({"scenario_id": sid, "data_version": ver, "object_id": np.unique(lc), "track_id": pd.NA,
                          "link_status": "no_match"})
    return pred, links


def repair_outputs(a: pd.DataFrame, b: pd.DataFrame, pred: pd.DataFrame, links: pd.DataFrame,
                   knowledge: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """把结果修成格式合法（每脉冲一行、局部簇只属一个节点和一个对象、关联一对一）。返回 (pred, links, 修改数)。"""
    sid, ver = str(a.scenario_id.iloc[0]), str(a.data_version.iloc[0])
    fixes = 0
    pred = pred.drop_duplicates("pulse_id")
    p = pd.DataFrame({"pulse_id": a.pulse_id.values, "platform_id": a.platform_id.values}).merge(
        pred.drop(columns=["scenario_id", "data_version"], errors="ignore"), on="pulse_id", how="left")
    miss = p.local_cluster_id.isna() | p.object_id.isna()
    if miss.any():
        fixes += int(miss.sum())
        # 缺失脉冲：每节点一个新局部簇 / 新对象
        base = int(np.nanmax(p.local_cluster_id)) if p.local_cluster_id.notna().any() else 0
        obase = int(np.nanmax(p.object_id)) if p.object_id.notna().any() else 0
        nid = pd.factorize(p.platform_id[miss])[0]
        p.loc[miss, "local_cluster_id"] = base + 1 + nid
        p.loc[miss, "object_id"] = obase + 1 + nid
    typ = sorted((knowledge.get("models") or {"R01": {}}).keys())[0]
    valid = set((knowledge.get("models") or {}).keys())
    bad_t = p.target_class.isna() | (~p.target_class.isin(valid) if valid else False)
    if bad_t.any():
        fixes += int(bad_t.sum())
        mode = p.loc[~bad_t].groupby("object_id").target_class.agg(lambda s: s.mode().iloc[0])
        p.loc[bad_t, "target_class"] = p.loc[bad_t, "object_id"].map(mode).fillna(typ)
    bad_s = ~p.target_state.isin([0, 1, 2])
    if bad_s.any():
        fixes += int(bad_s.sum())
        p.loc[bad_s, "target_state"] = 0
    p["local_cluster_id"] = p.local_cluster_id.astype(np.int64)
    p["object_id"] = p.object_id.astype(np.int64)
    p["target_state"] = p.target_state.astype(np.int64)
    # 局部簇跨节点：按 (节点, 局部簇) 重新编号
    if (p.groupby("local_cluster_id").platform_id.nunique() > 1).any():
        fixes += 1
        p["local_cluster_id"] = pd.factorize(p.platform_id.astype(str) + "/" + p.local_cluster_id.astype(str))[0] + 1
    # 局部簇对应多个对象：取脉冲数最多的对象
    if (p.groupby("local_cluster_id").object_id.nunique() > 1).any():
        fixes += 1
        maj = p.groupby("local_cluster_id").object_id.agg(lambda s: s.value_counts().index[0])
        p["object_id"] = p.local_cluster_id.map(maj).astype(np.int64)
    # 同一对象的型号须一致（评分器按对象成员评价，统一为多数型号）
    if (p.groupby("object_id").target_class.nunique() > 1).any():
        fixes += 1
        maj = p.groupby("object_id").target_class.agg(lambda s: s.value_counts().index[0])
        p["target_class"] = p.object_id.map(maj)
    pred = pd.DataFrame({"scenario_id": sid, "data_version": ver, "pulse_id": p.pulse_id.values,
                         "local_cluster_id": p.local_cluster_id.values, "object_id": p.object_id.values,
                         "target_class": p.target_class.values, "target_state": p.target_state.values})
    # 关联表：每个对象一行；matched 的航迹须存在且一对一
    objs = np.unique(pred.object_id.values)
    lk = links.drop_duplicates("object_id").set_index("object_id") if len(links) else pd.DataFrame(columns=["track_id", "link_status"])
    tids = set(b.track_id.astype(np.int64)) if b is not None and len(b) else set()
    used, rows = set(), []
    for o in objs:
        st, tr = "no_match", pd.NA
        if o in lk.index and lk.loc[o, "link_status"] == "matched" and pd.notna(lk.loc[o, "track_id"]):
            t = int(lk.loc[o, "track_id"])
            if t in tids and t not in used:
                st, tr = "matched", t
                used.add(t)
            else:
                fixes += 1
        rows.append((sid, ver, int(o), tr, st))
    fixes += int(len(set(lk.index) - set(objs))) if len(lk) else 0
    links = pd.DataFrame(rows, columns=["scenario_id", "data_version", "object_id", "track_id", "link_status"])
    return pred, links, fixes


def run_with_fallback(a, b, models, knowledge, cfg, run_scenario):
    """分级运行，返回 (pred, links, extras, 模式名)。"""
    from . import io

    a = sanitize_a(a)
    try:
        b = sanitize_b(b)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        b = b.iloc[0:0] if b is not None else b
    b_empty = b.iloc[0:0] if b is not None and len(b) else b
    stages = [("完整流程", cfg, b), ("安全模式", safe_cfg(cfg), b), ("安全模式（不用 B 航迹）", safe_cfg(cfg), b_empty)]
    for name, c, bb in stages:
        t0 = time.time()
        try:
            fault = os.environ.get("B04_FAULT", "")  # 仅供降级路径自测：full = 完整流程报错，all = 各级都报错
            if fault == "all" or (fault == "full" and name == "完整流程"):
                raise RuntimeError(f"B04_FAULT={fault} 注入故障")
            pred, links, extras = run_scenario(a, bb, models, knowledge, c)
            pred, links, nfix = repair_outputs(a, b, pred, links, knowledge)
            errs = io.validate_outputs(a, b, pred, links)
            if errs:
                raise ValueError(f"格式自检未通过: {errs}")
            if nfix:
                log.warning("%s：输出修复 %d 处", name, nfix)
            if name != "完整流程":
                log.warning("已降级为%s（%.0f s）", name, time.time() - t0)
            return pred, links, extras, name
        except Exception:  # noqa: BLE001
            log.error("%s失败，尝试降级：\n%s", name, traceback.format_exc())
    log.error("主流程全部失败，使用应急基线")
    pred, links = emergency_baseline(a, knowledge)
    pred, links, _ = repair_outputs(a, b, pred, links, knowledge)
    return pred, links, {}, "应急基线"
