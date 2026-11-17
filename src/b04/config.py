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
        "type_loss_max": 0.5,       # 对象合并的型号似然损失上限（nats / CPI 权重）
        "min_obj_pulses": 5000,     # 小于此脉冲数的对象尝试并入大对象
        "absorb_type_loss_max": 3.0,
        "tiny_obj_pulses": 2000,
        "max_merge_objects": 200,   # 复杂度保护：两两合并只在最大的若干对象间进行
        "max_absorb_objects": 500,    # 小于此脉冲数的对象强制并入最相近对象  # 小对象证据少，型号约束放宽，以方位连续为主
    },
    "recognize": {
        "type_model": "fingerprint",  # fingerprint（生成式、无先验）| gbdt
    },
    "fusion": {
        "max_delay_us": 400.0,      # 跨节点同源 CPI 起止时刻差上限（基线/光速 + 截断余量）
        "rf_tol": 3.5,
        "pw_tol": 0.15,
        "pri_tol": 0.05,
        "min_pairs": 3,
        "conflict_frac": 0.12,      # 合并时允许的冲突 CPI 比例（同一辐射源因抖动串接伪影约 1–6%，不同辐射源 >30%）
    },
    "anchor": {
        "sigma_air": 0.3,           # 发射-航迹方位残差标准差（空中/海面节点，训练集实测）
        "sigma_sea": 0.55,
        "gate_air": 1.2,            # 单个 CPI 残差硬门限
        "gate_sea": 2.6,
        "min_track_support": 5000,  # 航迹判型所需唯一匹配脉冲数
        "min_type_prob": 0.05,
        "rel_type_prob": 0.3,       # 航迹型号概率 ≥ 该单元最可能型号概率 × 此比例
        "vote_min_conf": 0.8,       # 参与航迹判型投票的单元最低型号把握度
        "frag_veto": True, "veto_min_hit": 0.6,
        "min_margin": 4.0,          # 最优/次优航迹综合代价差
        "prop_gap_s": 3.0,          # 标签沿片段传播的最大时间间隔
        "prop_doa_deg": 2.0,
    },
    "refine": {
        "enabled": True,
        "iters": 2,
        "min_gain": 6.0,
        "type_w": 1.0,              # 型号项权重
        "geo_cap": 9.0,             # 几何项（残差平方/连续性偏差）单 CPI 上限            # 改归所需最小得分增益
        "conflict_pen": 8.0,        # 时序互斥冲突惩罚（每 CPI 权重；软证据，抖动型号串接有自重叠伪影）
        "type_gate": 15.0,          # 单元在对象型号下的似然比最优型号低于此值（nats）则禁止归入
        "noinfo_pen": 2.0,          # 无几何信息时的中性惩罚
        "cont_max_gap_s": 30.0,     # 方位连续性的最大时间间隔
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
