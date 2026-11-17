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
        sub = a.iloc[idx]
        order = np.argsort(sub.toa.values, kind="stable")
        ii = idx[order]
        air = float(np.nanmedian(sub.sensor_height.values)) > 100.0
        doa_err = c["doa_err_air"] if air else c["doa_err_sea"]
        ch = chain_pulses(a.toa.values[ii], a.param1.values[ii], a.param2.values[ii], a.param3.values[ii],
                          a.doa.values[ii], 2.0 * doa_err + c["doa_w_extra"], c["min_pri"], c["max_pri"],
                          c["pri_tol"], c["jit_frac"], c["rf_w"], c["pw_w"], c["amp_w"], c["kmax"])
        cpi_of[ii] = ch + offset
        offset += int(ch.max()) + 1
    t = pd.DataFrame({
        "cpi": cpi_of, "platform_id": a.platform_id.values, "toa": a.toa.values, "rf": a.param1.values,
        "pw": a.param2.values, "amp": a.param3.values, "doa": a.doa.values,
        "slon": a.sensor_longitude.values, "slat": a.sensor_latitude.values, "sh": a.sensor_height.values,
    }).sort_values(["cpi", "toa"], kind="stable")
    t["dt"] = t.groupby("cpi").toa.diff()
    # DOA 圆周均值：以首脉冲为参考累加差值
    ref = t.groupby("cpi").doa.transform("first")
    t["ddoa"] = angdiff(t.doa.values, ref.values)
    g = t.groupby("cpi", sort=True)
    cp = g.agg(platform_id=("platform_id", "first"), n=("toa", "size"), t0=("toa", "first"), t1=("toa", "last"),
               rf=("rf", "mean"), pw=("pw", "mean"), amp=("amp", "mean"), amp_sd=("amp", "std"),
               doa_ref=("doa", "first"), ddoa=("ddoa", "mean"), pri=("dt", "median"), pri_min=("dt", "min"),
               pri_max=("dt", "max"), slon=("slon", "mean"), slat=("slat", "mean"), sh=("sh", "mean"))
    cp["doa"] = (cp.doa_ref + cp.ddoa) % 360.0
    # 漏脉冲鲁棒的 PRI/抖动估计：以最小间隔为基准，取 1 倍间隔的中位数；抖动 = 间隔偏离整数倍的中位数
    t["pmin"] = t.cpi.map(cp.pri_min)
    one = t.dt <= 1.5 * t.pmin
    cp["pri"] = t[one].groupby("cpi").dt.median().reindex(cp.index)
    t["pri1"] = t.cpi.map(cp.pri)
    dev = (t.dt - np.round(t.dt / t.pri1) * t.pri1).abs()
    cp["pri_dev"] = dev.groupby(t.cpi).median().reindex(cp.index)
    cp["jitter"] = (cp.pri_dev > 1.0).astype(np.int8)
    cp["drop_frac"] = (1.0 - (cp.n - 1) / ((cp.t1 - cp.t0) / cp.pri)).clip(0, 1).fillna(0)
    cp = cp.drop(columns=["doa_ref", "ddoa", "pri_dev"]).reset_index()
    return cpi_of, cp
