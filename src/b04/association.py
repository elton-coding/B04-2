"""L7 A-B 跨源关联：方位残差似然 + 领域/散射等级规则约束 + 匈牙利指派。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .geo import angdiff, bearing_deg


def scenario_base_time(a: pd.DataFrame) -> pd.Timestamp:
    """场景零点：datetime - toa 的中位数（toa 以场景起点为零，单位 μs）。"""
    s = a[["datetime", "toa"]].dropna().sample(min(len(a), 200000), random_state=0)
    base = s.datetime - pd.to_timedelta(s.toa, unit="us")
    return base.median().round("s")  # floor 会因 toa>0 落到前一秒，造成 1 s 时间偏差


def track_table(b: pd.DataFrame, base: pd.Timestamp) -> pd.DataFrame:
    t = b.copy()
    t["t_us"] = (t.datetime - base).dt.total_seconds() * 1e6
    return t.sort_values(["track_id", "t_us"])


def _interp_track(tr: pd.DataFrame, t_us: np.ndarray):
    tt = tr.t_us.values
    inside = (t_us >= tt[0]) & (t_us <= tt[-1])
    lon = np.interp(t_us, tt, tr.target_longitude.values)
    lat = np.interp(t_us, tt, tr.target_latitude.values)
    return lon, lat, inside


def association_scores(cp: pd.DataFrame, tracks: pd.DataFrame, cfg: dict, seed: int = 0) -> pd.DataFrame:
    """返回每个 (object_id, track_id) 的方位命中率、残差中位数与使用的 CPI 数。"""
    lc = cfg["link"]
    doa_err = {True: cfg["cpi"]["doa_err_air"], False: cfg["cpi"]["doa_err_sea"]}
    rng = np.random.default_rng(seed)
    rows = []
    trk = {k: v for k, v in tracks.groupby("track_id")}
    for oid, x in cp[cp.n >= 8].groupby("object_id"):
        if len(x) > lc["max_cpis_per_object"]:
            x = x.iloc[rng.choice(len(x), lc["max_cpis_per_object"], replace=False)]
        tm = ((x.t0 + x.t1) / 2).values
        err = np.where(x.sh.values > 100, doa_err[True], doa_err[False])
        tol = err + lc["res_margin_deg"]
        for tid, tr in trk.items():
            lon, lat, inside = _interp_track(tr, tm)
            if inside.sum() < lc["min_cpis"]:
                continue
            brg = bearing_deg(x.slon.values, x.slat.values, lon, lat)
            sres = angdiff(x.doa.values, brg)[inside]
            res = np.abs(sres)
            rows.append((oid, tid, float((res <= tol[inside]).mean()), float(np.median(res)), int(inside.sum()),
                         _dispersion(sres, x.platform_id.values[inside], err[inside])))
    return pd.DataFrame(rows, columns=["object_id", "track_id", "hit", "med_res", "n_used", "disp"])


def _dispersion(sres: np.ndarray, plat: np.ndarray, err: np.ndarray, min_n: int = 10) -> float:
    """残差离散度：各节点带符号残差的 MAD（去掉节点固定偏差）/ 该节点脉冲级测向误差，按 CPI 数加权。

    同一目标：CPI 级方位是多脉冲平均，残差只含少量噪声，MAD 远小于脉冲级误差（实测 ≤0.15）；
    近旁的另一目标（编队伴随、静默航迹）方位随时间的变化形状不同，MAD 明显增大（实测 ≥0.4）。
    """
    d = pd.DataFrame({"p": plat, "r": sres, "e": err})
    g = d.groupby("p")
    n = g.size()
    mad = g.r.agg(lambda v: np.median(np.abs(v - np.median(v)))) / g.e.first()
    use = n >= min_n
    if not use.any():
        use = n > 0
    return float(np.average(mad[use], weights=n[use]))


def associate(cp: pd.DataFrame, obj_type: pd.Series, b: pd.DataFrame, base: pd.Timestamp,
              knowledge: dict, cfg: dict, prefer: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    lc = cfg["link"]
    objects = sorted(cp.object_id.unique())
    links = pd.DataFrame({"object_id": objects, "track_id": pd.array([pd.NA] * len(objects), dtype="Int64"),
                          "link_status": "no_match"})
    if b is None or len(b) == 0:
        return links, pd.DataFrame()
    tracks = track_table(b, base)
    sc = association_scores(cp, tracks, cfg)
    if sc.empty:
        return links, sc
    # 规则约束：领域（航迹高度>0 ↔ 空中型号）、散射等级候选
    tinfo = tracks.groupby("track_id").agg(h=("target_height", "median"), rcs=("target_type", "first"))
    models = knowledge.get("models", {})
    feas = []
    for r in sc.itertuples(index=False):
        m = models.get(obj_type.get(r.object_id), {})
        ok = True
        if m:
            track_air = tinfo.h.get(r.track_id, 0) > 50
            ok &= (m.get("domain") == "air") == track_air
            rcs = tinfo.rcs.get(r.track_id)
            if isinstance(rcs, str) and rcs in ("small", "medium", "large"):
                ok &= rcs in m.get("rcs", [rcs])
        feas.append(ok)
    sc["feasible"] = np.array(feas) & (sc.disp.values <= lc.get("max_disp", 0.35))
    sc["cost"] = np.where(sc.feasible & (sc.hit >= lc["min_hit"]), 1.0 - sc.hit + 0.1 * sc.med_res + 0.5 / np.sqrt(sc.n_used), 10.0)
    if prefer:
        pm = np.array([prefer.get(o) == t for o, t in zip(sc.object_id, sc.track_id)])
        sc.loc[pm & (sc.cost < 10.0), "cost"] -= lc.get("anchor_bonus", 0.5)
    oi = {o: i for i, o in enumerate(objects)}
    tids = sorted(tracks.track_id.unique())
    ti = {t: i for i, t in enumerate(tids)}
    # 代价矩阵 + 每个对象一个"不关联"虚拟列（代价 null_cost），避免匈牙利算法为凑满指派而牺牲正确关联
    no = len(objects)
    C = np.full((no, len(tids) + no), 1e6)
    C[:, :len(tids)] = 10.0
    C[np.arange(no), len(tids) + np.arange(no)] = lc.get("null_cost", 0.5)
    for r in sc.itertuples(index=False):
        C[oi[r.object_id], ti[r.track_id]] = r.cost
    rr, cc = linear_sum_assignment(C)
    for i, j in zip(rr, cc):
        if j < len(tids) and C[i, j] < lc.get("null_cost", 0.5):
            links.loc[i, "track_id"] = tids[j]
            links.loc[i, "link_status"] = "matched"
    return links, sc
