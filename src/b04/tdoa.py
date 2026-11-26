"""跨节点到达时间差（TDOA）几何证据。

同一次发射到达两节点的时刻只差传播时延：Δt = (|P − S_b| − |P − S_a|) / c。各节点 toa 处于同一时间基准，
同一发射在两节点的脉冲序列平移对齐后逐脉冲时差几乎恒定（训练集实测离散度≈0），TDOA 精度只受航迹预测限制，
远比测向（0.3–0.55°，数十千米处折合数百米）精细，能区分间距一两百米的编队航迹。

航迹侧两项误差需处理（训练集真值标定）：
- 位置噪声约 100 m：局部线性平滑，空中目标机动多用短窗（3 s），海面目标用长窗（10 s）；
- 个别空中航迹时间戳整体偏移（T4 约 −1 s，按 100 m/s 折合 0.3–0.7 μs）：逐航迹在 δ 网格（±10 s，A/B 时间基准
  整体偏移时也能找回）上估计，
  仅当命中率显著提高时采用（避免把编队伙伴航迹"平移"过来）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from .geo import enu

C_M_PER_US = 299.792458


def smooth_tracks(tracks: pd.DataFrame, win_s: float = 8.0, win_air_s: float | None = None) -> pd.DataFrame:
    """逐航迹做高斯加权局部线性回归（经度、纬度、高度），抑制航迹点位置噪声。win_air_s：空中航迹窗口。"""
    if win_s <= 0 or tracks is None or not len(tracks):
        return tracks
    out = []
    for _, x in tracks.groupby("track_id", sort=False):
        x = x.sort_values("t_us").copy()
        win = win_air_s if (win_air_s and float(x.target_height.median()) > 50) else win_s
        t = x.t_us.values / 1e6
        if len(t) < 5:
            out.append(x)
            continue
        dt = t[None, :] - t[:, None]
        w = np.exp(-0.5 * (dt / win) ** 2)
        s0, s1, s2 = w.sum(1), (w * dt).sum(1), (w * dt * dt).sum(1)
        den = s0 * s2 - s1 * s1
        for col in ("target_longitude", "target_latitude", "target_height"):
            y = x[col].values.astype(float)
            t0, t1 = w @ y, (w * dt) @ y
            x[col] = np.where(den > 1e-9, (s2 * t0 - s1 * t1) / np.where(den > 1e-9, den, 1.0), y)
        out.append(x)
    return pd.concat(out).sort_values(["track_id", "t_us"])


@njit(cache=True)
def _best_offset(ta, tb, tol, k):
    """两段脉冲序列的最佳整体平移（tb ≈ ta + o）：候选取两端前 k 个脉冲两两差，返回 (o, 命中比例)。"""
    na, nb = len(ta), len(tb)
    kk = min(k, na, nb)
    best_h, best_o, best_r = -1, 0.0, np.inf
    for side in range(2):
        for i in range(kk):
            for j in range(kk):
                if side == 0:
                    o = tb[j] - ta[i]
                else:
                    o = tb[nb - 1 - j] - ta[na - 1 - i]
                h = 0
                s = 0.0
                s2 = 0.0
                jj = 0
                for q in range(na):
                    v = ta[q] + o
                    while jj < nb - 1 and tb[jj + 1] <= v:
                        jj += 1
                    d0 = tb[jj] - v
                    if jj < nb - 1 and abs(tb[jj + 1] - v) < abs(d0):
                        d0 = tb[jj + 1] - v
                    if abs(d0) <= tol:
                        h += 1
                        s += d0
                        s2 += d0 * d0
                if h > best_h:
                    best_h = h
                    mu = s / h if h > 0 else 0.0
                    best_o = o + mu
                    best_r = np.sqrt(max(s2 / h - mu * mu, 0.0)) if h > 1 else np.inf
    return best_o, best_h / max(min(na, nb), 1), best_r


@njit(cache=True)
def _pair_offsets(ts, starts, ends, ia, ib, tol, k):
    n = len(ia)
    off = np.empty(n)
    hit = np.empty(n)
    rms = np.empty(n)
    for q in range(n):
        a0, a1 = starts[ia[q]], ends[ia[q]]
        b0, b1 = starts[ib[q]], ends[ib[q]]
        off[q], hit[q], rms[q] = _best_offset(ts[a0:a1], ts[b0:b1], tol, k)
    return off, hit, rms


def emission_pairs(cp: pd.DataFrame, unit: np.ndarray, toa: np.ndarray, cpi_of: np.ndarray,
                   tol: float = 0.6, k: int = 3, min_hit: float = 0.6, max_rms: float = np.inf) -> pd.DataFrame:
    """同一发射单元内、不同节点的 CPI 两两配对，测量脉冲级到达时间差。

    返回列：ia, ib（cp 行号）, unit, t（时刻 μs）, off（tb − ta, μs）, period（PRI 模糊周期，抖动为 inf）, hit,
    rms（对齐后逐脉冲时差的标准差：同一发射只含到达时刻噪声，约 0.1–0.2 μs；偏大说明序列对齐不可靠）。"""
    order = np.lexsort((toa, cpi_of))
    cs, ts = cpi_of[order], toa[order].astype(np.float64)
    ids = cp.cpi.values
    starts = np.searchsorted(cs, ids, side="left")
    ends = np.searchsorted(cs, ids, side="right")
    df = pd.DataFrame({"u": unit, "i": np.arange(len(cp)), "p": cp.platform_id.values})
    multi = df.groupby("u").p.transform("nunique") > 1
    df = df[multi]
    if not len(df):
        return pd.DataFrame(columns=["ia", "ib", "unit", "t", "off", "period", "hit", "rms"])
    m = df.merge(df, on="u", suffixes=("a", "b"))
    m = m[m.pa < m.pb]
    ia, ib = m.ia.values.astype(np.int64), m.ib.values.astype(np.int64)
    off, hit, rms = _pair_offsets(ts, starts.astype(np.int64), ends.astype(np.int64), ia, ib, tol, k)
    jit = (cp.jitter.values[ia] > 0) | (cp.jitter.values[ib] > 0)
    period = np.where(jit, np.inf, cp.pri.values[ia])
    t = (np.maximum(cp.t0.values[ia], cp.t0.values[ib]) + np.minimum(cp.t1.values[ia], cp.t1.values[ib])) / 2
    out = pd.DataFrame({"ia": ia, "ib": ib, "unit": m.u.values, "t": t, "off": off, "period": period, "hit": hit,
                        "rms": rms})
    return out[(out.hit >= min_hit) & (out.rms <= max_rms)].reset_index(drop=True)


def tdoa_residuals(cp: pd.DataFrame, ep: pd.DataFrame, tracks: pd.DataFrame, tids: list) -> np.ndarray:
    """每个配对对每条航迹的 TDOA 残差（μs，观测 − 预测，按 PRI 模糊周期折叠）；航迹时段外为 NaN。"""
    R = np.full((len(ep), len(tids)), np.nan)
    if not len(ep):
        return R
    lon0, lat0 = float(cp.slon.mean()), float(cp.slat.mean())
    ia, ib = ep.ia.values, ep.ib.values
    xa, ya = enu(cp.slon.values[ia], cp.slat.values[ia], lon0, lat0)
    xb, yb = enu(cp.slon.values[ib], cp.slat.values[ib], lon0, lat0)
    za, zb = cp.sh.values[ia], cp.sh.values[ib]
    t = ep.t.values
    per = ep.period.values
    for j, tid in enumerate(tids):
        x = tracks[tracks.track_id == tid]
        tt = x.t_us.values
        px, py = enu(np.interp(t, tt, x.target_longitude.values), np.interp(t, tt, x.target_latitude.values), lon0, lat0)
        pz = np.interp(t, tt, x.target_height.values)
        ra = np.sqrt((px - xa) ** 2 + (py - ya) ** 2 + (pz - za) ** 2)
        rb = np.sqrt((px - xb) ** 2 + (py - yb) ** 2 + (pz - zb) ** 2)
        r = ep.off.values - (rb - ra) / C_M_PER_US
        fin = np.isfinite(per) & (per > 0)
        r[fin] = (r[fin] + per[fin] / 2) % per[fin] - per[fin] / 2
        inside = (t >= tt[0] - 1e6) & (t <= tt[-1] + 1e6)
        R[inside, j] = r[inside]
    return R


def estimate_time_offsets(cp: pd.DataFrame, ep: pd.DataFrame, tracks: pd.DataFrame, tids: list, ok_doa: np.ndarray,
                          grid=np.arange(-2.5, 2.51, 0.25), hit_us: float = 0.15, min_pairs: int = 50,
                          min_gain: float = 0.15, min_hit: float = 0.5, prom_s: float = 2.0, min_prom: float = 0.15,
                          edge_s: float = 0.5) -> dict:
    """逐空中航迹估计时间戳偏移 δ（秒）：候选配对 = 两端 CPI 对该航迹方位残差均在门限内（ok_doa: CPI × 航迹）；
    取使 |TDOA 残差| < hit_us 比例最大的 δ，需比 δ=0 高 min_gain 且自身 ≥ min_hit，否则为 0。"""
    out = {}
    if not len(ep):
        return out
    h = tracks.groupby("track_id").target_height.median()
    i0 = int(np.argmin(np.abs(grid)))
    for j, tid in enumerate(tids):
        if h.loc[tid] <= 50:
            continue
        cand = ok_doa[ep.ia.values, j] & ok_doa[ep.ib.values, j]
        if cand.sum() < min_pairs:
            continue
        e = ep[cand].reset_index(drop=True)
        x = tracks[tracks.track_id == tid]
        fr = np.array([np.nanmean(np.abs(tdoa_residuals(cp, e, x.assign(t_us=x.t_us + d * 1e6), [tid])[:, 0]) < hit_us)
                       for d in grid])
        k = int(np.nanargmax(fr))
        # 真偏移是网格内部的尖峰；落在网格边界或在 ±prom_s 处仍不回落的平缓高地多为假峰（航迹恰好扫过别处的同源配对）
        w = int(round(prom_s / max(grid[1] - grid[0], 1e-9)))
        side = [fr[q] for q in (k - w, k + w) if 0 <= q < len(fr)]
        interior = 0 < k < len(fr) - 1 and abs(grid[k]) <= grid.max() - edge_s
        prom = fr[k] - (max(side) if side else 0.0)
        if fr[k] >= min_hit and fr[k] - fr[i0] >= min_gain and (k == i0 or (interior and prom >= min_prom)):
            out[tid] = float(grid[k])
    return out


def prepare_tdoa(cp: pd.DataFrame, emis: np.ndarray, toa: np.ndarray, cpi_of: np.ndarray, tracks: pd.DataFrame,
                 ok_doa: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """配对 → 平滑航迹 → 逐航迹时间偏移 → 残差矩阵。返回 (配对表, 残差矩阵, {航迹: δ 秒})。"""
    ac = cfg["anchor"]
    tids = sorted(tracks.track_id.unique())
    ep = emission_pairs(cp, emis, toa, cpi_of)
    tsm = smooth_tracks(tracks, ac.get("track_smooth_s", 10.0), ac.get("track_smooth_air_s", 3.0))
    m = ac.get("tdoa_offset_max_s", 10.0)
    grid = np.arange(-m, m + 1e-9, 0.25)
    dl = estimate_time_offsets(cp, ep, tsm, tids, ok_doa, grid=grid) if ac.get("tdoa_time_offset", True) else {}
    if dl:
        tsm = tsm.copy()
        tsm["t_us"] = tsm.t_us + tsm.track_id.map(dl).fillna(0.0).values * 1e6
    return ep, tdoa_residuals(cp, ep, tsm, tids), dl


def toa_phase_residuals(cp: pd.DataFrame, tracks: pd.DataFrame, tids: list, lattice_us: float = 64.0) -> np.ndarray:
    """单节点到达时刻相位残差 |((t0 − r/c) mod L)|（μs，折叠到 [0, L/2]）：CPI × 航迹，航迹时段外为 NaN。

    训练集实测：各辐射源驻留的发射起点落在 L = 64 μs 的公共时间格上（与到达时间同一基准），故一个从驻留起点开始的
    CPI，其首脉冲到达时刻对 L 取模恰等于 节点—辐射源 距离 / c 对 L 取模（真航迹残差中位 0.03–0.5 μs，相当于
    10–150 m）；而对任意其他航迹，残差在 [0, L/2] 上近似均匀，落入 0.6 μs 的概率只有约 2%。与 TDOA 不同，
    它不要求两节点同时收到同一串脉冲，能区分分时照射各节点的同型号雷达（如两部 R01 的整串驻留错挂）。"""
    E = np.full((len(cp), len(tids)), np.nan)
    if not len(cp) or tracks is None or not len(tracks):
        return E
    lon0, lat0 = float(cp.slon.mean()), float(cp.slat.mean())
    xs, ys = enu(cp.slon.values, cp.slat.values, lon0, lat0)
    zs = cp.sh.values.astype(float)
    tq = cp.t0.values.astype(float)
    L = float(lattice_us)
    for j, tid in enumerate(tids):
        x = tracks[tracks.track_id == tid]
        if not len(x):
            continue
        tt = x.t_us.values
        px, py = enu(np.interp(tq, tt, x.target_longitude.values), np.interp(tq, tt, x.target_latitude.values), lon0, lat0)
        pz = np.interp(tq, tt, x.target_height.values)
        r = np.sqrt((px - xs) ** 2 + (py - ys) ** 2 + (pz - zs) ** 2) / C_M_PER_US
        e = np.abs((tq - r + L / 2) % L - L / 2)
        inside = (tq >= tt[0] - 1e6) & (tq <= tt[-1] + 1e6)
        E[inside, j] = e[inside]
    return E
