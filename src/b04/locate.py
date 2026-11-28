"""辐射源被动定位：多站测向交会 + 到达时刻相位测距（态势报告用，不影响结果表）。

- 测向：每个 CPI 给出节点→辐射源方位，脉冲级测向误差为给定半宽的均匀分布，CPI 平均后误差按 1/√n 缩小，
  另加 0.3° 的残余系统误差下限。
- 相位测距（见 tdoa.toa_phase_residuals）：驻留起点的发射时刻落在公共 64 μs 时间格上，故驻留起点到达时刻
  对 64 μs 取模 = 节点—辐射源斜距 / c 对 64 μs 取模，单节点即可测距（模糊周期 19.2 km，由测向交会与
  轨迹连续性消解）。空中目标高度由空中、海面节点斜距之差解出。
- 轨迹模型：分段线性（空中 2 s、海面 5 s 一个结点）+ 加速度先验；迭代重加权（Huber）只作用于观测残差，
  先验不做稳健化（否则轨迹可整周期跳变）。
- 训练集验证（500 s，按真值分组、不用航迹，与真航迹比较）：可靠时段水平误差中位 64–190 m、90% 分位 110–250 m、
  99% 分位 ≤ 290 m，与 B 航迹自身约 100 m 的位置误差同量级；空中目标高度误差 ≤ 0.1 km。相位自检不通过（数据无公共
  时间格）时只用测向。只有单节点的观测段离可交会段过远时可能整周期错位，标为不可靠（见 _cross_fix_ok）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from .geo import enu

C_M_PER_US = 299.792458
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0

DEFAULTS = {"knot_air_s": 2.0, "knot_sea_s": 5.0, "acc_air": 3.0, "acc_sea": 0.5, "bias_deg": 0.3,
            "phase_sigma_us": 0.3, "lattice_us": 64.0, "phase_check_hit": 0.3, "max_sd_m": 300.0, "gap_s": 3.0,
            "max_cpi": 40000, "z0_air": 3000.0, "z_prior_m": 2000.0, "min_cpi": 30,
            "restart_hit": 0.8, "restart_gain": 0.1, "restart_max_cpi": 3000, "restart_min_m": 1000.0,
            "restart_max_m": 30000.0, "restart_step_m": 2000.0, "seg_gap_s": 10.0, "max_link_air_s": 30.0,
            "max_link_sea_s": 120.0}


def _sparsity(n, K, k, use_phase, air):
    rows, cols = [], []
    for bk in range(2 if use_phase else 1):
        base = bk * n
        for d in range(2):
            for kk in (k, k + 1):
                rows.append(base + np.arange(n)); cols.append(2 * kk + d)
        if air and bk == 1:
            rows.append(base + np.arange(n)); cols.append(np.full(n, 2 * K))
    base = (2 if use_phase else 1) * n
    q = np.arange(K - 2)
    for d in range(2):
        for off in range(3):
            rows.append(base + 2 * q + d); cols.append(2 * (q + off) + d)
    nr = base + 2 * (K - 2)
    if air:
        rows.append(np.array([nr])); cols.append(np.array([2 * K])); nr += 1
    rr, cc = np.concatenate(rows), np.concatenate(cols)
    return coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nr, 2 * K + (1 if air else 0)))


def fit_track(t, xs, ys, zs, doa_deg, n, half_w, ph_us, air: bool, lc: dict, use_phase: bool = True,
              init_range_m: float | None = None):
    """单个辐射源的被动轨迹。输入为 CPI 级数组（t 秒、节点平面坐标 m、方位°、脉冲数、测向半宽°、驻留起点到达时刻 μs）。
    返回 dict(tk, X(K,2), z, sd(K), phase_hit) 或 None。"""
    o = np.argsort(t)
    t, xs, ys, zs, doa_deg, n, half_w, ph_us = (v[o] for v in (t, xs, ys, zs, doa_deg, n, half_w, ph_us))
    th = np.radians(doa_deg)
    N0 = len(t)
    sig = np.radians(np.sqrt((half_w / np.sqrt(3)) ** 2 / np.maximum(n, 1) + lc["bias_deg"] ** 2))
    knot = lc["knot_air_s"] if air else lc["knot_sea_s"]
    acc = lc["acc_air"] if air else lc["acc_sea"]
    tk = np.arange(t.min(), t.max() + knot, knot)
    K = len(tk)
    if K < 3:
        tk = np.array([t.min(), t.min() + knot, t.min() + 2 * knot]); K = 3
    k = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, K - 2)
    al = (t - tk[k]) / knot
    # 初值：结点附近窗口内测向线加权最小二乘交会（至少两个节点），其余结点线性插值
    X0 = np.full((K, 2), np.nan)
    lo = np.searchsorted(t, tk - max(knot, 10.0))
    hi = np.searchsorted(t, tk + max(knot, 10.0))
    for q in range(K):
        m = slice(lo[q], hi[q])
        if hi[q] - lo[q] < 3 or np.ptp(xs[m]) + np.ptp(ys[m]) < 1.0:
            continue
        nx, ny = np.cos(th[m]), -np.sin(th[m])
        w = 1.0 / sig[m]
        A = np.c_[nx, ny] * w[:, None]
        if np.linalg.cond(A) > 1e4:
            continue
        X0[q] = np.linalg.lstsq(A, (nx * xs[m] + ny * ys[m]) * w, rcond=None)[0]
    if init_range_m is not None:
        # 多起点：沿最近 CPI 的测向线放在给定距离上（单节点为主、交会初值不可用时）
        j = np.clip(np.searchsorted(t, tk), 0, N0 - 1)
        X0 = np.c_[xs[j] + init_range_m * np.sin(th[j]), ys[j] + init_range_m * np.cos(th[j])]
    ok = np.isfinite(X0[:, 0])
    if ok.sum() < 2:
        return None
    for d in (0, 1):
        X0[:, d] = np.interp(tk, tk[ok], X0[ok, d])
    z0 = lc["z0_air"] if air else 0.0
    sp, L = lc["phase_sigma_us"], lc["lattice_us"]
    N = len(t)
    wd = np.ones(N)
    wp = np.zeros(N)

    def unpack(p):
        return p[:2 * K].reshape(K, 2), (p[2 * K] if air else 0.0)

    def raw(p):
        X, z = unpack(p)
        P = X[k] * (1 - al)[:, None] + X[k + 1] * al[:, None]
        dx, dy = P[:, 0] - xs, P[:, 1] - ys
        r1 = ((th - np.arctan2(dx, dy) + np.pi) % (2 * np.pi) - np.pi) / sig
        R = np.sqrt(dx ** 2 + dy ** 2 + (z - zs) ** 2) / C_M_PER_US
        e = ((ph_us - R + L / 2) % L - L / 2) / sp
        return X, z, r1, e

    def resid(p):
        X, z, r1, e = raw(p)
        out = [r1 * np.sqrt(wd)]
        if use_phase:
            out.append(e * np.sqrt(wp))
        out.append(((X[:-2] - 2 * X[1:-1] + X[2:]) / (acc * knot ** 2)).ravel())
        if air:
            out.append(np.array([(z - z0) / lc["z_prior_m"]]))
        return np.concatenate(out)

    sps = _sparsity(N, K, k, use_phase, air)
    p = np.r_[X0.ravel(), [z0] if air else []]
    passes = [False, False, True, True, True] if use_phase else [False, False, False]
    for q, use_p in enumerate(passes + [None]):
        r = least_squares(resid, p, method="trf", max_nfev=40, jac_sparsity=sps, x_scale=1000.0)
        p = r.x
        if use_p is None:
            break
        _, _, r1, e = raw(p)
        hc = 2.0 if use_p else 3.0
        wd = np.minimum(1.0, hc / np.maximum(np.abs(r1), 1e-9))
        if use_phase:
            ae = np.abs(e)
            wp = np.where(ae < 10.0, np.minimum(1.0, 2.0 / np.maximum(ae, 1e-9)), 0.0) if use_p else (ae * sp < 1.5).astype(float)
    X, z, _, e = raw(p)
    hit = float(np.mean(np.abs(e[n >= 8]) * sp <= 0.6)) if (n >= 8).any() else 0.0
    J = r.jac.toarray() if hasattr(r.jac, "toarray") else np.asarray(r.jac)
    try:
        Cv = np.linalg.inv(J.T @ J + 1e-9 * np.eye(J.shape[1]))
        i = np.arange(K)
        sd = np.sqrt(np.maximum(Cv[2 * i, 2 * i] + Cv[2 * i + 1, 2 * i + 1], 0.0))
    except np.linalg.LinAlgError:
        sd = np.full(K, np.inf)
    return {"tk": tk, "X": X, "z": float(z), "sd": sd, "phase_hit": hit, "t_obs": t}


def _cross_fix_ok(tk, t, pid, lc: dict, air: bool) -> np.ndarray:
    """模糊周期可靠性：观测按 > seg_gap_s 的空档分段；至少两个节点各有 ≥3 个 CPI 的段可交会定位；只有单节点的段
    只能靠相位测距 + 连续性，与最近的可交会段相隔超过 max_link_s（空中 30 s、海面 120 s）时距离可能整周期
    （19.2 km）错位，标为不可靠。训练集中一部 R01 长空档后只被单节点收到的末段即整段偏 19.3 km。"""
    o = np.argsort(t)
    t, pid = t[o], pid[o]
    cut = np.where(np.diff(t) > lc["seg_gap_s"])[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(t) - 1]
    multi = np.zeros(len(starts), bool)
    for q, (a, b) in enumerate(zip(starts, ends)):
        _, cnt = np.unique(pid[a:b + 1], return_counts=True)
        multi[q] = (cnt >= 3).sum() >= 2
    link = lc["max_link_air_s"] if air else lc["max_link_sea_s"]
    ok_seg = multi.copy()
    if multi.any():
        ms, me = t[starts[multi]], t[ends[multi]]
        for q in np.where(~multi)[0]:
            g = np.min(np.maximum(0.0, np.maximum(ms - t[ends[q]], t[starts[q]] - me)))
            ok_seg[q] = g <= link
    # 结点归属：落在某段 [起, 止] ± gap_s 内取该段，否则不可靠（near 条件另行处理）
    seg = np.searchsorted(t[starts], tk, side="right") - 1
    seg = np.clip(seg, 0, len(starts) - 1)
    return ok_seg[seg]


def passive_tracks(cp: pd.DataFrame, domain_air: dict, cfg: dict) -> dict:
    """逐对象被动轨迹。domain_air: {object_id(0 基): 是否空中}。返回 {object_id: 结果 dict（含经纬度）}。"""
    from .exclusion import dwell_heads

    lc = {**DEFAULTS, **(cfg.get("locate") or {})}
    if not lc.get("enabled", True) or not len(cp):
        return {}
    air_node = cp.sh.values > 100
    hw = np.where(air_node, cfg["cpi"]["doa_err_air"], cfg["cpi"]["doa_err_sea"]).astype(float)
    pc = cfg.get("phase", {})
    h = dwell_heads(cp, pc.get("head_gap_mul", 2.6), pc.get("head_gap_us", 1700.0), pc.get("head_dur_ms", 30.0) * 1e3, hw)
    t0h = cp.t0.values[h].astype(float)
    lon0, lat0 = float(cp.slon.mean()), float(cp.slat.mean())
    xs, ys = enu(cp.slon.values, cp.slat.values, lon0, lat0)
    obj = cp.object_id.values
    out = {}
    rng = np.random.default_rng(0)
    for o, air in domain_air.items():
        idx = np.where(obj == o)[0]
        if len(idx) < lc["min_cpi"]:
            continue
        if len(idx) > lc["max_cpi"]:
            idx = np.sort(rng.choice(idx, lc["max_cpi"], replace=False))
        args = (cp.t0.values[idx] / 1e6, xs[idx], ys[idx], cp.sh.values[idx].astype(float), cp.doa.values[idx].astype(float),
                cp.n.values[idx].astype(float), hw[idx], t0h[idx], bool(air), lc)
        try:
            res = fit_track(*args, use_phase=True)
            if res is not None and res["phase_hit"] < lc["restart_hit"]:
                # 以单节点为主时测向交会初值差，相位落在错误的模糊周期上：沿测向线按不同距离多起点重拟合，取相位命中率最高者
                sub = idx if len(idx) <= lc["restart_max_cpi"] else np.sort(rng.choice(idx, lc["restart_max_cpi"], replace=False))
                sargs = (cp.t0.values[sub] / 1e6, xs[sub], ys[sub], cp.sh.values[sub].astype(float),
                         cp.doa.values[sub].astype(float), cp.n.values[sub].astype(float), hw[sub], t0h[sub], bool(air), lc)
                best_r, best_h = None, res["phase_hit"] + lc["restart_gain"]
                for R0 in np.arange(lc["restart_min_m"], lc["restart_max_m"] + 1, lc["restart_step_m"]):
                    rr = fit_track(*sargs, use_phase=True, init_range_m=float(R0))
                    if rr is not None and rr["phase_hit"] > best_h:
                        best_r, best_h = float(R0), rr["phase_hit"]
                if best_r is not None:
                    rr = fit_track(*args, use_phase=True, init_range_m=best_r)
                    if rr is not None and rr["phase_hit"] > res["phase_hit"]:
                        res = rr
            if res is not None and res["phase_hit"] < lc["phase_check_hit"]:
                res = fit_track(*args, use_phase=False)
                if res is not None:
                    res["phase_used"] = False
            elif res is not None:
                res["phase_used"] = True
        except (ValueError, np.linalg.LinAlgError):
            res = None
        if res is None:
            continue
        tk, X = res["tk"], res["X"]
        to = np.sort(res["t_obs"])
        q = np.searchsorted(to, tk)
        near = np.minimum(np.abs(tk - to[np.clip(q - 1, 0, len(to) - 1)]), np.abs(to[np.clip(q, 0, len(to) - 1)] - tk))
        res["reliable"] = (near <= lc["gap_s"]) & (res["sd"] <= lc["max_sd_m"]) & _cross_fix_ok(
            tk, cp.t0.values[idx] / 1e6, cp.platform_id.values[idx], lc, bool(air))
        res["lon"] = lon0 + X[:, 0] / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat0)))
        res["lat"] = lat0 + X[:, 1] / M_PER_DEG_LAT
        res["nodes"] = sorted(int(v) for v in np.unique(cp.platform_id.values[idx]))
        res["n_cpi"] = int(len(idx))
        out[int(o)] = res
    return out


PSEUDO_ID_BASE = 900000


def pseudo_tracks(cp: pd.DataFrame, knowledge: dict, cfg: dict, base, sid: str, ver: str) -> pd.DataFrame:
    """航迹缺失时的伪航迹：对锚定脉冲占比低的对象做被动定位，可靠结点写成 B 类格式（track_id ≥ PSEUDO_ID_BASE，
    散射等级 unknown），供第二轮锚定 / 精修 / 几何否决 / 相位改归使用；不参与 A-B 关联输出。"""
    pc = cfg.get("pseudo_anchor", {})
    models = knowledge.get("models", {})
    w = cp.n.values.astype(float)
    anc = (cp["anchor"].values >= 0) if "anchor" in cp.columns else np.zeros(len(cp), bool)
    tot = pd.Series(w).groupby(cp.object_id.values).sum()
    got = pd.Series(w * anc).groupby(cp.object_id.values).sum()
    otype = cp.groupby("object_id").obj_type.first()
    sel = {int(o): models.get(otype.loc[o], {}).get("domain") == "air" for o in tot.index
           if got.loc[o] < pc.get("obj_max_anchor", 0.5) * tot.loc[o] and tot.loc[o] >= pc.get("min_pulses", 2000)}
    if not sel:
        return pd.DataFrame()
    P = passive_tracks(cp, sel, cfg)
    rows = []
    for o, r in P.items():
        rel = r["reliable"]
        if rel.sum() < pc.get("min_knots", 10) or r["phase_hit"] < pc.get("min_phase_hit", 0.5):
            continue
        t = r["tk"][rel]
        rows.append(pd.DataFrame({
            "scenario_id": sid, "data_version": ver, "track_id": PSEUDO_ID_BASE + int(o),
            "datetime": base + pd.to_timedelta(t, unit="s"), "target_longitude": r["lon"][rel], "target_latitude": r["lat"][rel],
            "target_height": max(r["z"], 0.0), "target_type": "unknown", "target_class": "",
            "preset_type": str(otype.loc[o])}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
