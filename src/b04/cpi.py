"""L2：在单节点交错脉冲流中按"RF/PW 恒定 + PRI 恒定或抖动"串接出 CPI。

依据（训练集实测）：CPI 内 RF 均匀误差约 ±2.5 MHz、PW ±0.1 μs、幅度 ±0.5 dB、
DOA 空中 ±1.5° / 海面 ±3°；非抖动型号 CPI 内 PRI 精确恒定（<0.01 μs）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from .geo import angdiff

MODE_ONE, MODE_TENT, MODE_FIXED, MODE_JIT = 0, 1, 2, 3


@njit(cache=True)
def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


@njit(cache=True)
def chain_pulses(toa, rf, pw, amp, doa, doa_w, min_pri, max_pri, pri_tol, jit_frac, rf_w, pw_w, amp_w, kmax):
    """贪心多链串接。参数门限采用"区间宽度"判据：均匀有界噪声下，同一 CPI 内
    任意脉冲参数的极差不超过噪声全宽（rf_w/pw_w/amp_w/doa_w）。
    时序上容忍漏脉冲：间隔可为 PRI 的整数倍（k<=kmax）；若首个间隔本身是倍数，后续自动修正 PRI。"""
    n = toa.shape[0]
    out = np.empty(n, np.int64)
    K = 1024
    a_id = np.empty(K, np.int64)
    a_last = np.empty(K)
    a_n = np.empty(K, np.int64)
    a_pri = np.empty(K)
    a_mode = np.empty(K, np.int64)
    a_dt = np.empty(K)
    a_dref = np.empty(K)
    # 每链 4 个参数的 [sum, min, max]：0 rf, 1 pw, 2 amp, 3 doa(相对 dref)
    a_s = np.empty((K, 4))
    a_lo = np.empty((K, 4))
    a_hi = np.empty((K, 4))
    w = np.array([rf_w, pw_w, amp_w, doa_w])
    v = np.empty(4)
    na = 0
    next_id = 0
    for i in range(n):
        t = toa[i]
        j = 0
        while j < na:
            gap = t - a_last[j]
            m = a_mode[j]
            if m == MODE_FIXED or m == MODE_TENT:
                lim = a_pri[j] * (kmax + 0.5) + 1.0
            elif m == MODE_JIT:
                lim = a_pri[j] * (3.0 + 0.5) + 1.0
            else:
                lim = max_pri
            if gap > lim:
                na -= 1
                a_id[j] = a_id[na]; a_last[j] = a_last[na]; a_n[j] = a_n[na]; a_pri[j] = a_pri[na]
                a_mode[j] = a_mode[na]; a_dt[j] = a_dt[na]; a_dref[j] = a_dref[na]
                for k in range(4):
                    a_s[j, k] = a_s[na, k]; a_lo[j, k] = a_lo[na, k]; a_hi[j, k] = a_hi[na, k]
            else:
                j += 1
        best = -1
        best_s = 1e18
        best_cls = -1
        for j in range(na):
            v[0] = rf[i]; v[1] = pw[i]; v[2] = amp[i]; v[3] = _wrap(doa[i] - a_dref[j])
            ok = True
            s = 0.0
            for k in range(4):
                lo = min(a_lo[j, k], v[k])
                hi = max(a_hi[j, k], v[k])
                if hi - lo > w[k]:
                    ok = False
                    break
                s += abs(v[k] - a_s[j, k] / a_n[j]) / w[k]
            if not ok:
                continue
            gap = t - a_last[j]
            m = a_mode[j]
            cls = -1
            p = a_pri[j]
            if m == MODE_FIXED or m == MODE_TENT:
                k = np.floor(gap / p + 0.5)
                if k >= 1 and k <= kmax and abs(gap - k * p) <= pri_tol * k:
                    cls = 0 if k == 1 else 1
                else:
                    k2 = np.floor(p / gap + 0.5)
                    if k2 >= 2 and k2 <= kmax and abs(p - k2 * gap) <= pri_tol * k2:
                        cls = 1  # 先前 PRI 估计为倍数，接受并修正
                    elif m == MODE_TENT and abs(gap - p) <= jit_frac * p:
                        cls = 2
            elif m == MODE_JIT:
                for kk in range(1, 4):
                    if abs(gap - kk * p) <= jit_frac * kk * p:
                        cls = 1 if kk == 1 else 2
                        break
            else:
                if gap >= min_pri and gap <= max_pri:
                    cls = 3
            if cls < 0:
                continue
            s += cls * 100.0
            if s < best_s:
                best_s = s
                best = j
                best_cls = cls
        if best < 0:
            if na >= K:
                na -= 1
            j = na
            na += 1
            a_id[j] = next_id; next_id += 1
            a_last[j] = t; a_n[j] = 1; a_pri[j] = 0.0; a_mode[j] = MODE_ONE; a_dt[j] = 0.0
            a_dref[j] = doa[i]
            v[0] = rf[i]; v[1] = pw[i]; v[2] = amp[i]; v[3] = 0.0
            for k in range(4):
                a_s[j, k] = v[k]; a_lo[j, k] = v[k]; a_hi[j, k] = v[k]
            out[i] = a_id[j]
        else:
            j = best
            gap = t - a_last[j]
            m = a_mode[j]
            p = a_pri[j]
            if m == MODE_ONE:
                a_mode[j] = MODE_TENT
                a_pri[j] = gap
            elif m == MODE_TENT or m == MODE_FIXED:
                if gap < p * 0.75 and abs(p - np.floor(p / gap + 0.5) * gap) <= pri_tol * np.floor(p / gap + 0.5):
                    a_pri[j] = gap
                    a_mode[j] = MODE_FIXED
                elif best_cls == 2:
                    a_mode[j] = MODE_JIT
                    a_pri[j] = 0.5 * (p + gap)
                else:
                    a_mode[j] = MODE_FIXED
            elif m == MODE_JIT:
                kk = np.floor(gap / p + 0.5)
                if kk == 1:
                    a_pri[j] = 0.9 * p + 0.1 * gap
            a_last[j] = t
            a_n[j] += 1
            a_dt[j] += gap
            v[0] = rf[i]; v[1] = pw[i]; v[2] = amp[i]; v[3] = _wrap(doa[i] - a_dref[j])
            for k in range(4):
                a_s[j, k] += v[k]
                a_lo[j, k] = min(a_lo[j, k], v[k])
                a_hi[j, k] = max(a_hi[j, k], v[k])
            out[i] = a_id[j]
    return out


def extract_cpis(a: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """返回 (每个脉冲的 cpi_id 数组(与 a 行对齐), CPI 表)。"""
    c = cfg["cpi"]
    cpi_of = np.full(len(a), -1, np.int64)
    offset = 0
    for pid, idx in a.groupby("platform_id", sort=True).indices.items():
        order = np.argsort(a.toa.values[idx], kind="stable")  # 只取所需列，不复制该节点整表
        ii = idx[order]
        air = float(np.nanmedian(a.sensor_height.values[idx])) > 100.0
        doa_err = c["doa_err_air"] if air else c["doa_err_sea"]
        tq = a.toa.values[ii].astype(np.float64)
        if len(tq) > 1 and (np.diff(tq) <= 0).any():
            # 同节点到达时刻重复（碰撞 / 填补的缺失值）：串接只需严格递增，错开 1 ns（不改原数据）
            k = np.arange(len(tq)) * 1e-3
            tq = np.maximum.accumulate(tq - k) + k
        ch = chain_pulses(tq, a.param1.values[ii], a.param2.values[ii], a.param3.values[ii],
                          a.doa.values[ii], 2.0 * doa_err + c["doa_w_extra"], c["min_pri"], c["max_pri"],
                          c["pri_tol"], c["jit_frac"], c["rf_w"], c["pw_w"], c["amp_w"], c["kmax"])
        cpi_of[ii] = ch + offset
        offset += int(ch.max()) + 1
    # 按 (cpi, toa) 稳定排序后直接按排列取列（不先建整表再排序复制，大场景内存峰值更低；顺序与原 sort_values 相同）
    o = np.lexsort((a.toa.values, cpi_of))
    t = pd.DataFrame({
        "cpi": cpi_of[o], "platform_id": a.platform_id.values[o], "toa": a.toa.values[o], "rf": a.param1.values[o],
        "pw": a.param2.values[o], "amp": a.param3.values[o], "doa": a.doa.values[o],
        "slon": a.sensor_longitude.values[o], "slat": a.sensor_latitude.values[o], "sh": a.sensor_height.values[o],
    }, index=o)
    del o
    first = np.r_[True, t.cpi.values[1:] != t.cpi.values[:-1]]
    dtv = np.diff(t.toa.values, prepend=np.nan)
    dtv[first] = np.nan
    t["dt"] = dtv
    # DOA 圆周均值：以首脉冲为参考累加差值
    ref = t.doa.values[np.maximum.accumulate(np.where(first, np.arange(len(t)), 0))]
    t["ddoa"] = angdiff(t.doa.values, ref)
    del ref, dtv, first
    g = t.groupby("cpi", sort=True)
    cp = g.agg(platform_id=("platform_id", "first"), n=("toa", "size"), t0=("toa", "first"), t1=("toa", "last"),
               rf=("rf", "mean"), pw=("pw", "mean"), amp=("amp", "mean"), amp_sd=("amp", "std"),
               doa_ref=("doa", "first"), ddoa=("ddoa", "mean"), pri=("dt", "median"), pri_min=("dt", "min"),
               pri_max=("dt", "max"), slon=("slon", "mean"), slat=("slat", "mean"), sh=("sh", "mean"))
    cp["doa"] = (cp.doa_ref + cp.ddoa) % 360.0
    # 漏脉冲鲁棒的 PRI/抖动估计：以最小间隔为基准，取 1 倍间隔的中位数；抖动 = 间隔偏离整数倍的中位数
    t["pmin"] = t.cpi.map(cp.pri_min)
    one = t.dt <= 1.5 * t.pmin
    cp["pri"] = t.dt[one].groupby(t.cpi[one]).median().reindex(cp.index)
    t["pri1"] = t.cpi.map(cp.pri)
    dev = (t.dt - np.round(t.dt / t.pri1) * t.pri1).abs()
    cp["pri_dev"] = dev.groupby(t.cpi).median().reindex(cp.index)
    cp["jitter"] = (cp.pri_dev > 1.0).astype(np.int8)
    cp["drop_frac"] = (1.0 - (cp.n - 1) / ((cp.t1 - cp.t0) / cp.pri)).clip(0, 1).fillna(0)
    cp = cp.drop(columns=["doa_ref", "ddoa", "pri_dev"]).reset_index()
    return cpi_of, cp


def estimate_doa_halfwidth(a: pd.DataFrame, min_pairs: int = 2000) -> dict:
    """逐节点估计脉冲测向误差半宽（均匀分布）：取同一 PRI 链上相邻脉冲（连续三个脉冲间隔相等、RF/PW 一致，
    基本可确定是同一雷达）的方位差，两独立 U(-h, h) 之差为三角分布：|差| 的 99.9% 分位 = 2h(1 - √0.001) 给出噪声支撑半宽
    （决定 CPI 内方位极差上限），90% 分位 = 2h(1 - √0.1) 给出与标准差相当的等效半宽（决定 CPI 级方位误差）。
    训练/测试集两种估计都是空中 1.5°、海面 3.0°，与赛题给定一致。返回 {platform_id: (支撑半宽, 等效半宽)}。"""
    out = {}
    for pid, idx in a.groupby("platform_id", sort=True).indices.items():
        ii = idx[np.argsort(a.toa.values[idx], kind="stable")]  # 只取所需列，不复制整表
        dt = np.diff(a.toa.values[ii].astype(float))
        if len(dt) < 3:
            continue
        drf = np.abs(np.diff(a.param1.values[ii]))
        dpw = np.abs(np.diff(a.param2.values[ii]))
        dv = a.doa.values[ii]
        dd = np.abs(angdiff(dv[1:], dv[:-1]))
        same = (np.abs(dt[1:] - dt[:-1]) <= 0.05) & (dt[1:] <= 2000.0) & (drf[1:] <= 3.0) & (drf[:-1] <= 3.0) & (dpw[1:] <= 0.15)
        x = dd[1:][same]
        if len(x) < min_pairs:
            continue
        q90, q999 = np.quantile(x, [0.9, 0.999])
        out[pid] = (float(q999 / (2.0 * (1.0 - np.sqrt(0.001)))), float(q90 / (2.0 * (1.0 - np.sqrt(0.1)))))
    return out


def calibrate_doa(a: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    """实测测向误差明显大于给定值（> ratio_min 倍）时按比例放大方位门限：测向半宽（CPI 串接窗等）按支撑半宽之比，
    锚定 σ 与单 CPI 门限按等效半宽之比。给定值是赛题物理误差模型，实测不大于它时保持不变。
    返回 (cfg, {"air": (支撑倍数, 等效倍数), ...})。"""
    import copy

    if cfg.get("_doa_calibrated"):
        return cfg, {}
    cfg = dict(cfg, _doa_calibrated=True)   # 只校准一次（run_scenario 与 cluster_and_type 都会调用）
    dc = cfg.get("doa_calib", {})
    if not dc.get("enabled", True):
        return cfg, {}
    h = estimate_doa_halfwidth(a, dc.get("min_pairs", 2000))
    if not h:
        return cfg, {}
    air = a.groupby("platform_id").sensor_height.median() > 100.0
    c = cfg["cpi"]
    k = {}
    for kind, nom in (("air", c["doa_err_air"]), ("sea", c["doa_err_sea"])):
        v = [x for p, x in h.items() if bool(air.get(p, False)) == (kind == "air")]
        if v:
            rs = max(x[0] for x in v) / nom
            rq = max(x[1] for x in v) / nom
            if max(rs, rq) > dc.get("ratio_min", 1.1):
                cap = dc.get("ratio_max", 3.0)
                k[kind] = (min(max(rs, 1.0), cap), min(max(rq, 1.0), cap))
    if not k:
        return cfg, {}
    cfg = copy.deepcopy(cfg)
    for kind, (rs, rq) in k.items():
        cfg["cpi"][f"doa_err_{kind}"] *= rs
        cfg["anchor"][f"sigma_{kind}"] *= rq
        cfg["anchor"][f"gate_{kind}"] *= rq
    return cfg, k
