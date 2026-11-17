"""默认参数（物理误差模型 + 训练集统计），可用 config/*.yaml 覆盖。"""
from __future__ import annotations

import copy
import os

import yaml

DEFAULT = {
    "cpi": {
        "doa_err_air": 1.5,   # 空中节点 DOA 半宽误差（D3）
        "doa_err_sea": 3.0,   # 海面节点
        "min_pri": 30.0,
        "max_pri": 1200.0,
        "pri_tol": 0.02,      # 固定 PRI 链内允许偏差 μs
        "jit_frac": 0.08,     # 抖动型 PRI 相对窗口
        "kmax": 6,            # 允许的连续漏脉冲倍数
        # 同一 CPI 内参数极差上限（噪声全宽 + 0.1 量化余量），训练/测试实测一致
        "rf_w": 6.0, "pw_w": 0.25, "amp_w": 1.3, "doa_w_extra": 0.5,
    },
    "track": {
        "max_gap_s": 20.0,          # 节点内 CPI 串接最大时间间隔
        "doa_rate_air": 1.5,        # 允许的方位变化率 deg/s（被观测为空中目标）
        "doa_rate_sea": 0.6,
        "overlap_eps_us": 5.0,
        "min_type_compat": 0.05,
    },
    "fusion": {
        "max_delay_us": 400.0,      # 跨节点同源 CPI 起止时刻差上限（基线/光速 + 截断余量）
        "rf_tol": 3.5,
        "pw_tol": 0.15,
        "pri_tol": 0.05,
        "min_pairs": 3,
    },
    "link": {
        "res_margin_deg": 2.0,      # 方位残差容差 = 节点 DOA 误差 + 余量
        "min_hit": 0.85,
        "min_cpis": 5,
        "max_cpis_per_object": 3000,
    },
}


def _merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def load_knowledge(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
