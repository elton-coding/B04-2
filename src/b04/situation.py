"""态势要素抽取：把结果表 + 原始观测整理为可核对的"事实"（报告与本体共用）。

所有事实带来源：object_id / track_id / platform_id / 时段 / 数值，报告只引用这里的事实。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angdiff, dist_m, enu

STATE_NAME = {0: "搜索", 1: "粗跟踪", 2: "精跟踪"}


def _in_poly(x, y, poly):
    """射线法判断点是否在多边形内（经纬度平面近似）。"""
    inside = np.zeros(len(x), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cond
    return inside


def _smooth_velocity(t, X, Y, win_s=20.0):
    """滑动窗口线性拟合求速度（抑制约 65 m 的航迹位置噪声）。"""
    vx = np.full(len(t), np.nan)
    vy = np.full(len(t), np.nan)
    for k in range(len(t)):
        m = np.abs(t - t[k]) <= win_s / 2
        if m.sum() >= 4:
            vx[k] = np.polyfit(t[m], X[m], 1)[0]
            vy[k] = np.polyfit(t[m], Y[m], 1)[0]
    return vx, vy


def motion_pattern(t, X, Y, domain):
    win = 20.0 if domain in ("air", "空中") else 60.0  # 海面目标慢，需更长窗口压噪
    vx, vy = _smooth_velocity(t, X, Y, win)
    # 以半窗口间隔抽样计算航向变化，避免逐点噪声累积
    step = max(1, int(round((win / 2) / max(np.median(np.diff(t)), 1e-3)))) if len(t) > 1 else 1
    vx, vy = vx[::step], vy[::step]
    spd = np.hypot(vx, vy)
    hdg = np.degrees(np.arctan2(vx, vy)) % 360
    ok = np.isfinite(hdg)
    dh = angdiff(hdg[ok][1:], hdg[ok][:-1]) if ok.sum() > 1 else np.array([0.0])
    total_turn = float(np.nansum(dh))
    abs_turn = float(np.nansum(np.abs(dh)))
    extent = float(np.hypot(np.ptp(X), np.ptp(Y)))
    med_spd = float(np.nanmedian(spd)) if ok.any() else 0.0
    # 航向直方图：单峰→直线；双峰相反→往复/跑道形；均匀且累计转角大→盘旋
    h = np.histogram(hdg[ok], bins=12, range=(0, 360))[0] if ok.any() else np.zeros(12)
    h = h / max(h.sum(), 1)
    top = np.argsort(h)[::-1]
    opposite = abs(((top[0] - top[1]) % 12) - 6) <= 1 and h[top[1]] > 0.15
    if med_spd < 0.5:
        pat = "静止/漂泊"
    elif abs(total_turn) >= 300 and extent < 15000:
        pat = "盘旋"
    elif opposite and abs_turn >= 300:
        pat = "往复/跑道形"
    elif h[top[0]] + h[top[1] if abs(top[0] - top[1]) in (1, 11) else top[0]] >= 0.7 and abs_turn < 120:
        pat = "直线航行"
    else:
        pat = "机动"
    return {"speed_med": round(med_spd, 1), "speed_p10": round(float(np.nanpercentile(spd, 10)), 1) if ok.any() else 0.0,
            "speed_p90": round(float(np.nanpercentile(spd, 90)), 1) if ok.any() else 0.0,
            "heading_start": round(float(hdg[ok][0]), 0) if ok.any() else None,
            "heading_end": round(float(hdg[ok][-1]), 0) if ok.any() else None,
            "total_turn_deg": round(total_turn, 0), "abs_turn_deg": round(abs_turn, 0),
            "extent_km": round(extent / 1000, 1), "pattern": pat}


def node_facts(a: pd.DataFrame) -> list[dict]:
    out = []
    for pid, x in a.groupby("platform_id"):
        x = x.sort_values("toa")
        air = float(x.sensor_height.median()) > 100
        s = x.iloc[np.linspace(0, len(x) - 1, min(len(x), 400)).astype(int)]
        X, Y = enu(s.sensor_longitude.values, s.sensor_latitude.values, s.sensor_longitude.mean(), s.sensor_latitude.mean())
        mp = motion_pattern(s.toa.values / 1e6, X, Y, "air" if air else "sea")
        out.append({"platform_id": int(pid), "domain": "空中" if air else "海面", "height_m": float(x.sensor_height.median()),
                    "pulses": int(len(x)), "t_first_s": round(x.toa.min() / 1e6, 1), "t_last_s": round(x.toa.max() / 1e6, 1),
                    "pos_start": [round(float(x.sensor_longitude.iloc[0]), 5), round(float(x.sensor_latitude.iloc[0]), 5)],
                    "pos_end": [round(float(x.sensor_longitude.iloc[-1]), 5), round(float(x.sensor_latitude.iloc[-1]), 5)],
                    **mp})
    return out


def track_facts(tracks: pd.DataFrame, poly=None) -> dict:
    out = {}
    for tid, x in tracks.groupby("track_id"):
        x = x.sort_values("t_us")
        t = x.t_us.values / 1e6
        lon0, lat0 = x.target_longitude.mean(), x.target_latitude.mean()
        X, Y = enu(x.target_longitude.values, x.target_latitude.values, lon0, lat0)
        dom = "空中" if x.target_height.median() > 50 else "海面"
        f = {"track_id": int(tid), "domain": dom, "height_m": float(x.target_height.median()),
             "rcs": str(x.target_type.iloc[0]), "points": int(len(x)), "t_first_s": round(t[0], 1), "t_last_s": round(t[-1], 1),
             "dt_s": float(np.median(np.diff(t))) if len(t) > 1 else None,
             "pos_start": [round(float(x.target_longitude.iloc[0]), 5), round(float(x.target_latitude.iloc[0]), 5)],
             "pos_end": [round(float(x.target_longitude.iloc[-1]), 5), round(float(x.target_latitude.iloc[-1]), 5)],
             **motion_pattern(t, X, Y, dom)}
        if poly is not None:
            ins = _in_poly(x.target_longitude.values, x.target_latitude.values, poly)
            f["in_area_frac"] = round(float(ins.mean()), 2)
            ch = np.where(np.diff(ins.astype(int)) != 0)[0]
            f["area_crossings_s"] = [round(float(t[i + 1]), 0) for i in ch]
        out[int(tid)] = f
    return out


def track_pair_relations(tracks: pd.DataFrame, step_s: float = 5.0) -> list[dict]:
    """航迹两两关系：伴随（持续近距同向同速）与近距接近事件。"""
    T = {k: v.sort_values("t_us") for k, v in tracks.groupby("track_id")}
    keys = sorted(T)
    t0 = max(min(v.t_us.min() for v in T.values()), 0)
    t1 = min(max(v.t_us.max() for v in T.values()), max(v.t_us.max() for v in T.values()))
    grid = np.arange(t0, t1, step_s * 1e6)
    P = {}
    for k, v in T.items():
        m = (grid >= v.t_us.min()) & (grid <= v.t_us.max())
        lon = np.interp(grid, v.t_us, v.target_longitude)
        lat = np.interp(grid, v.t_us, v.target_latitude)
        lon[~m] = np.nan
        lat[~m] = np.nan
        P[k] = (lon, lat, v.target_height.median() > 50)
    V = {}
    for k, v in T.items():
        tt = v.t_us.values / 1e6
        X, Y = enu(v.target_longitude.values, v.target_latitude.values, 142.1, 24.7)
        win = 20.0 if v.target_height.median() > 50 else 60.0
        vx, vy = _smooth_velocity(tt, X, Y, win)
        ok = np.isfinite(vx)
        V[k] = (np.interp(grid / 1e6, tt[ok], vx[ok]), np.interp(grid / 1e6, tt[ok], vy[ok]))
    rel = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            la, pa, aa = P[ka]
            lb, pb, ab = P[kb]
            if aa != ab:
                continue
            d = dist_m(la, pa, lb, pb)
            ok = np.isfinite(d)
            if ok.sum() < 5:
                continue
            dv = np.hypot(V[ka][0] - V[kb][0], V[ka][1] - V[kb][1])
            dmed = float(np.nanmedian(d))
            near = d < (1500 if aa else 1000)
            near_frac = float(np.mean(near[ok]))
            r = {"track_a": int(ka), "track_b": int(kb), "domain": "空中" if aa else "海面",
                 "dist_med_m": round(dmed), "dist_min_m": round(float(np.nanmin(d))),
                 "t_min_s": round(float(grid[np.nanargmin(d)] / 1e6), 0), "near_frac": round(near_frac, 2),
                 "rel_speed_med": round(float(np.nanmedian(dv[ok])), 1)}
            if near_frac >= 0.6 and r["rel_speed_med"] < (10.0 if aa else 2.0):
                r["relation"] = "伴随/编队"
            elif r["dist_min_m"] < (1500 if aa else 600):
                r["relation"] = "近距接近"
            else:
                continue
            rel.append(r)
    return rel


def tracked_nodes(cp: pd.DataFrame) -> pd.DataFrame:
    """跟踪驻留（状态 1/2）中幅度最大的节点视为被照射跟踪的对象 → (object, node, 时段, 驻留数)。"""
    t = cp[cp.state > 0]
    if t.empty:
        return pd.DataFrame(columns=["object_id", "node", "t_first_s", "t_last_s", "dwells", "coarse", "fine"])
    i = t.groupby("emission").amp.idxmax()
    e = t.loc[i.values, ["object_id", "platform_id", "t0", "state", "amp"]]
    # 主瓣照射判据：被跟踪节点的幅度显著高于该对象在该节点的中位幅度
    med = cp.groupby(["object_id", "platform_id"]).amp.median()
    e = e[e.amp.values - med.reindex(list(zip(e.object_id, e.platform_id))).values > 6.0]
    g = e.groupby(["object_id", "platform_id"])
    out = g.agg(t_first_s=("t0", lambda s: round(s.min() / 1e6, 1)), t_last_s=("t0", lambda s: round(s.max() / 1e6, 1)),
                dwells=("t0", "size"), coarse=("state", lambda s: int((s == 1).sum())),
                fine=("state", lambda s: int((s == 2).sum()))).reset_index().rename(columns={"platform_id": "node"})
    return out[out.dwells >= 3]


def object_facts(cp: pd.DataFrame, pred: pd.DataFrame, links: pd.DataFrame, scores: pd.DataFrame,
                 a: pd.DataFrame, knowledge: dict) -> list[dict]:
    models = knowledge.get("models", {})
    out = []
    pp = pred.assign(platform_id=a.platform_id.values, toa=a.toa.values)
    L = links.set_index("object_id")
    for oid, x in pp.groupby("object_id"):
        cpo = cp[cp.object_id == oid - 1]
        typ = x.target_class.iloc[0]
        m = models.get(typ, {})
        nodes = []
        for (pid, lc), y in x.groupby(["platform_id", "local_cluster_id"]):
            nodes.append({"platform_id": int(pid), "local_cluster_id": int(lc), "pulses": int(len(y)),
                          "t_first_s": round(y.toa.min() / 1e6, 1), "t_last_s": round(y.toa.max() / 1e6, 1)})
        st = x.target_state.value_counts(normalize=True)
        f = {"object_id": int(oid), "type": typ, "platform_model": m.get("platform_model"), "category": m.get("category"),
             "domain": {"air": "空中", "sea": "海面"}.get(m.get("domain"), "未知"), "pulses": int(len(x)),
             "t_first_s": round(x.toa.min() / 1e6, 1), "t_last_s": round(x.toa.max() / 1e6, 1), "nodes": nodes,
             "state_frac": {STATE_NAME[k]: round(float(st.get(k, 0.0)), 3) for k in (0, 1, 2)},
             "cpis": int(len(cpo)), "anchored_frac": round(float((cpo.anchor >= 0).mean()), 2) if len(cpo) else 0.0}
        lk = L.loc[oid]
        f["link_status"] = lk.link_status
        f["track_id"] = None if pd.isna(lk.track_id) else int(lk.track_id)
        if len(scores):
            s = scores[scores.object_id == oid - 1].sort_values("hit", ascending=False)
            f["link_candidates"] = [{"track_id": int(r.track_id), "hit": round(float(r.hit), 3), "med_res_deg": round(float(r.med_res), 2),
                                     "feasible": bool(r.feasible)} for r in s.head(3).itertuples()]
        out.append(f)
    return out


def type_confidence(cp: pd.DataFrame, proba: pd.DataFrame) -> dict:
    """对象级型号把握度：最优与次优型号的平均每 CPI 对数似然差。"""
    lp = np.log(proba.clip(lower=1e-4))
    s = lp.groupby(cp.object_id.values).mean()
    out = {}
    for o, r in s.iterrows():
        v = r.sort_values(ascending=False)
        out[int(o) + 1] = {"best": v.index[0], "second": v.index[1], "margin_per_cpi": round(float(v.iloc[0] - v.iloc[1]), 2)}
    return out


def passive_facts(cp: pd.DataFrame, objects: list[dict], tracks: pd.DataFrame, cfg: dict, poly=None) -> list[dict]:
    """A 类数据独立的被动定位（locate.passive_tracks）→ 每对象的定位事实：可靠时段、运动方式、与 B 航迹的距离。

    已关联对象：被动轨迹与关联航迹的距离是独立于方位命中率的关联佐证；无航迹对象：给出其位置与活动。"""
    from .locate import passive_tracks

    dom = {o["object_id"] - 1: o["domain"] == "空中" for o in objects}
    P = passive_tracks(cp, dom, cfg)
    out = []
    link = {o["object_id"] - 1: o["track_id"] for o in objects}
    for o0, r in sorted(P.items()):
        rel = r["reliable"]
        if rel.sum() < 5:
            continue
        tk, lon, lat = r["tk"][rel], r["lon"][rel], r["lat"][rel]
        X, Y = enu(lon, lat, float(np.mean(lon)), float(np.mean(lat)))
        air = dom[o0]
        f = {"object_id": o0 + 1, "domain": "空中" if air else "海面", "n_cpi": r["n_cpi"], "nodes": r["nodes"],
             "phase_used": bool(r["phase_used"]), "phase_hit": round(float(r["phase_hit"]), 2),
             "reliable_frac": round(float(rel.mean()), 2), "t_first_s": round(float(tk[0]), 1), "t_last_s": round(float(tk[-1]), 1),
             "sd_med_m": round(float(np.median(r["sd"][rel])), 0),
             "pos_start": [round(float(lon[0]), 5), round(float(lat[0]), 5)], "pos_end": [round(float(lon[-1]), 5), round(float(lat[-1]), 5)],
             **motion_pattern(tk, X, Y, "air" if air else "sea")}
        if air and r["phase_used"]:
            f["height_m"] = round(r["z"], -2)
        if poly is not None:
            f["in_area_frac"] = round(float(_in_poly(lon, lat, poly).mean()), 2)
        step = max(1, int(round(10.0 / max(np.median(np.diff(tk)), 1e-3))))
        f["path"] = [[round(float(t), 1), round(float(x), 5), round(float(y), 5)] for t, x, y in zip(tk[::step], lon[::step], lat[::step])]
        ds = []
        if tracks is not None and len(tracks):
            for tid, x in tracks.groupby("track_id"):
                tt = x.t_us.values / 1e6
                m = (tk >= tt[0]) & (tk <= tt[-1])
                if m.sum() < 5:
                    continue
                d = dist_m(lon[m], lat[m], np.interp(tk[m], tt, x.target_longitude.values), np.interp(tk[m], tt, x.target_latitude.values))
                ds.append({"track_id": int(tid), "dist_med_m": round(float(np.median(d)), 0), "dist_p90_m": round(float(np.percentile(d, 90)), 0),
                           "dist_min_m": round(float(d.min()), 0), "overlap_s": round(float(m.sum() * np.median(np.diff(tk))), 0)})
        ds.sort(key=lambda d: d["dist_med_m"])
        f["nearest_tracks"] = ds[:3]
        lk = link.get(o0)
        if lk is not None:
            f["linked_track_dist"] = next((d for d in ds if d["track_id"] == lk), None)
        out.append(f)
    return out


def build_facts(a, b, pred, links, extras, knowledge, tracks) -> dict:
    sid, ver = str(a.scenario_id.iloc[0]), str(a.data_version.iloc[0])
    ann = knowledge.get("announcements", {}).get(f"{sid}_{ver}")
    poly = ann["polygon"] if ann else None
    cp = extras["cp"]
    facts = {
        "scenario": {"scenario_id": sid, "data_version": ver, "base_time": str(extras["base_time"]),
                     "duration_s": round(a.toa.max() / 1e6, 1), "pulses": int(len(a)), "nodes": int(a.platform_id.nunique()),
                     "tracks": int(b.track_id.nunique()) if len(b) else 0, "announcement": ann},
        "nodes": node_facts(a),
        "tracks": track_facts(tracks, poly) if len(tracks) else {},
        "objects": object_facts(cp, pred, links, extras["scores"], a, knowledge),
        "type_confidence": type_confidence(cp, extras["proba"]),
        "tracked_nodes": tracked_nodes(cp).assign(object_id=lambda d: d.object_id + 1).to_dict("records"),
        "track_relations": track_pair_relations(tracks) if len(tracks) else [],
    }
    try:
        from .config import load_config
        facts["passive"] = passive_facts(cp, facts["objects"], tracks, extras.get("cfg") or load_config(), poly)
    except Exception as e:  # 定位失败不影响其余要素
        facts["passive"] = []
        facts["passive_error"] = repr(e)
    linked = {o["track_id"] for o in facts["objects"] if o["track_id"] is not None}
    facts["unlinked_tracks"] = sorted(set(facts["tracks"]) - linked)
    return facts
