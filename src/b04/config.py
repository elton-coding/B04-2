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
    "doa_calib": {                  # 测向误差自校准：实测（同 PRI 链相邻脉冲方位差）明显大于给定值时按比例放大方位门限
        "enabled": True,
        "ratio_min": 1.1,
        "ratio_max": 3.0,
        "min_pairs": 2000,
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
        "max_merge_objects": 80,    # 复杂度保护：两两合并只在最大的若干对象间进行（真实场景进入合并的对象 ≤50；杂散脉冲会产生数百个碎对象）
        "frag_phase": True,         # 片段串接时优先相位延续的候选（驻留起点 mod 64 μs 只随距离缓慢漂移）
        "frag_phase_fit_us": 0.6,   # 相位延续门限
        "frag_phase_rate": 1.0,     # 允许的相位漂移率 μs/s（训练集同源相邻 CPI 中位 0.04–0.6）
        "frag_phase_rate_unc": 0.05,  # 已估出漂移率（累计 ≥ rw_min 秒）后的漂移率不确定度 μs/s
        "frag_phase_acc": 0.01,     # 漂移率变化（径向加速度）μs/s²
        "frag_phase_rw_min": 3.0,
        "frag_phase_rw_cap": 30.0,
        "frag_phase_max_us": 8.0,   # 门限放宽到超过此值（间隔过长）不再用相位
        "frag_phase_min_n": 3,      # 少于此脉冲数的 CPI 不用相位
        "frag_phase_veto_k": 3,     # 候选片段相位连续延续 ≥ 此次数、本 CPI 不延续 → 不并入（0 关闭）
        "frag_phase_veto_us": 4.0,  #   仅在相位门限 ≤ 此值时否决
        "frag_phase_veto_n": 8,     #   本 CPI 脉冲数下限
        "frag_phase_check_hit": 0.5,  # 自检：首轮片段内相邻 CPI 相位延续比例下限（随机约 2–5%）
        "frag_phase_full": False,   # 首轮（全部 CPI）片段也用相位（验证：扰动锚定与 R05/R06 共用模式的改归，关闭）
        "frag_phase_rest": True,    # 未锚定部分的片段用相位
        "frag_phase_max_anchor": 0.6,  #   仅当锚定脉冲占比低于此值（航迹缺失多）时；测试集 0.89、完整航迹 CV 0.83、半数航迹约 0.4
        "phase_merge": False,       # 同节点时间交错、相位轨迹一致的对象合并（验证：对象已混入另一部同型号雷达的 CPI 时会把两部雷达并成一个，关闭）
        "phase_merge_win_s": 1.5,
        "phase_merge_nb": 3,        # 与 A 前后各此数个 CPI 比相位，过半吻合才算吻合
        "phase_merge_min_cmp": 8,
        "phase_merge_min_frac": 0.6,
        "phase_merge_node_cmp": 5,
        "phase_merge_node_frac": 0.5,
        "phase_merge_type_loss": 1.0,
        "phase_merge_max": 20,
        "phase_merge_conflict_frac": 0.1,  # 同节点时间重叠且 RF 不同的 CPI 占比上限（被并对象常混有少量另一部同型号雷达的 CPI）
        "phase_merge_final": False,  # 末轮（各改归步骤之后）再做一次
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
        "emis_frag_support": True,  # 发射构建时按片段对支持度优先配对
        "seq_check": True,          # 抖动 PRI 跨节点配对的脉冲序列对齐复核
        "seq_tol_us": 2.0,
        "seq_min_hit": 0.6,
        "conflict_frac": 0.12,      # 合并时允许的冲突 CPI 比例（同一辐射源因抖动串接伪影约 1–6%，不同辐射源 >30%）
    },
    "anchor": {
        "sigma_air": 0.3,           # 发射-航迹方位残差标准差（空中/海面节点，训练集实测）
        "sigma_sea": 0.55,
        "pos_sigma_m": 100.0,       # 航迹位置误差（米）：近距时折算的方位误差计入残差尺度
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
        "track_smooth_s": 10.0,     # 海面航迹局部线性平滑窗口（高斯 σ，秒）；0 关闭。用于 TDOA 预测
        "track_smooth_air_s": 3.0,  # 空中航迹（机动多）平滑窗口
        "tdoa_time_offset": True,   # 逐空中航迹估计时间戳偏移
        "tdoa_offset_max_s": 10.0,  # 偏移搜索范围 ±（A/B 时间基准整体偏移的稳健性）
        "offset_to_bearing": True,  # 估出的航迹时间偏移同时用于方位锚定与关联
        "offset_bearing_gain": 0.05,  # 方位印证：平移后半门限内 CPI 加权数至少增加此比例
        "tdoa_w": 1.0,              # TDOA 残差在锚定代价中的权重（每个跨节点配对按较小 CPI 权重计）；0 关闭
        "tdoa_sigma_us": 0.12,      # 平滑航迹下真航迹 TDOA 残差 σ（训练集实测 ≈0.10 μs ≈ 30 m）
        "tdoa_cap": 9.0,            # 单配对残差平方上限
        "tdoa_cost": True,          # TDOA 残差计入锚定代价（False 时仅作几何否决）
        "tdoa_track_min_pairs": 20, # 航迹级否决：支撑单元配对数下限
        "tdoa_track_min_hit": 0.3,  # 航迹判型：候选型号支撑配对吻合比例低于此值 → 剔除（该型号雷达不在此航迹上）
        "tdoa_fine_us": 0.6,        # "在航迹上"的确认门限（0.2 μs 细门限验证过：真航迹吻合率仅 0.6–0.99，编队测试变差）
        "tdoa_on_track_hit": 0.8,   # 吻合比例 ≥ 此值视为证实在航迹上；两个以上候选都被证实时按支撑量取多数
        "tdoa_dup_gap": 0.15,       # 编队两航迹争同一辐射源：细门限吻合比例相差 ≥ 此值时取高者
        "tdoa_gate_us": 0.6,
        "tdoa_out_frac": 0.5,       # 单元内超出门限的配对权重占比超过此值 → 该航迹几何不可行
    },
    "refine": {
        "enabled": True,
        "iters": 2,
        "min_gain": 6.0,
        "type_w": 1.0,              # 型号项权重
        "geo_cap": 9.0,             # 几何项（残差平方/连续性偏差）单 CPI 上限            # 改归所需最小得分增益
        "conflict_pen": 8.0,        # 时序互斥冲突惩罚（每 CPI 权重；软证据，抖动型号串接有自重叠伪影）
        "type_gate_mean": 4.0,      # 若设定：按单元内 CPI 加权平均的型号似然差门限（nats/权重）
        "type_gate": 15.0,          # 单元在对象型号下的似然比最优型号低于此值（nats）则禁止归入
        "noinfo_pen": 2.0,          # 无几何信息时的中性惩罚
        "cont_max_gap_s": 30.0,     # 方位连续性的最大时间间隔
        "tdoa": False,              # 精修几何项是否计入 TDOA（验证有害：扰动无锚定融合，关闭）
    },
    "exclusion": {                  # 节点内时间互斥重分配（同一辐射源同一时刻只发一串脉冲）
        "enabled": True,
        "margin_us": 1000.0,        # 重叠超过此时长（且频率不同）才算冲突
        "busy_win_s": 1.0,          # 局部占空比窗口半宽
        "busy_max": 0.95,
        "busy_min": 0.5,            # 局部占空比低于此值时"不重叠"不计为证据
        "overlap_pen": 3.0,         # 每个冲突 CPI 的惩罚（nats × CPI 权重）
        "min_gain": 2.0,
        "iters": 8,
        "dissolve_frac": 0.7,       # 对象被改归走超过此比例 → 残余按小对象并入他处
        # 整段补洞：对象 Y 在某节点前后都高占空比照射、中间出现空洞，而同型号对象 X 恰在空洞内有一段
        # 自身前后都没有的高占空比片段 → 该片段是 Y 的（逐 CPI 打分时片段内单元互相抬高 X 的占空比，修不动）
        "episode": True,
        "ep_gap_s": 0.1,            # X 的 CPI 间隔小于此值则串成同一片段
        "ep_min_span_s": 0.6,       # 片段最短时长
        "ep_min_busy": 0.5,         # 片段自身占空比下限
        "ep_flank_s": 2.0,          # 前后侧窗长度
        "ep_flank_min": 0.4,        # Y 两侧占空比均不低于此值；X 两侧均值须低于此值
        "ep_hole_max": 0.2,         # Y 在片段时段内的占空比上限（空洞）
    },
    "geo_veto": {                   # 几何否决改归：对象航迹残差 > k × 单 CPI 门限的 CPI 改归几何一致的型号相容对象
        "enabled": True,
        "k": 2.0,                   # CV 中 k=2 命中的 CPI 全部为错归（零误伤）
        "win_s": 5.0,               # 无航迹候选对象：同节点前后此范围内的 CPI 方位须连续（≤ 2 倍测向半宽误差）
        "type_p_min": 0.4,          # 型号相容另含：CPI 在候选对象型号上的概率 ≥ 此值（R05/R06 共用参数模式的 CPI 概率约 0.33/0.67，不含型号信息，不能据此跨型号改归）
        "nb_excl_fit": True,        # 反向规则判"方位连续邻居"时排除同样紧贴该航迹的 CPI
        "nb_min_frac": 0.25,        # 以及脉冲数不足本 CPI 此比例的零散 CPI
        "max_fit_frac": 0.4,        # 当前无航迹对象在该节点上贴合 X 航迹的脉冲占比超过此值 → 几何分不开，除非 CPI 判型更偏向 X 否则不改归（CV 改对的情形 ≤0.31）
        "local_s": 30.0,            # 反向规则：当前无航迹对象在该 CPI 前后 [local_excl_s, local_s] 内的 CPI 过 local_frac 也贴合 X 的航迹 → 不改归
        "local_excl_s": 1.0,
        "local_frac": 0.5,
        "uniq_gate": 1.0,           # 反向规则：另有未分给任何对象的航迹残差 ≤ 此倍数 × 门限，且当前对象脉冲至少
        "orphan_min_hit": 0.5,      # 此比例在该航迹门限内 → 可能是当前对象自己的航迹，不改归
        "type_margin": 0.2,         # "判型更偏向 X"：CPI 在 X 型号上的概率至少高出此值
        "k_fit": 0.5,               # 反向：紧贴另一有航迹对象的航迹（残差 ≤ 此倍数 × 门限）且与当前对象不一致 → 改归（CV 精度约 99.5%）
    },
    "phase": {                      # 到达时刻相位改归（单节点测距：驻留起点在公共 64 μs 时间格上）
        "enabled": True,
        "lattice_us": 64.0,
        "fit_us": 0.6,              # 相位贴合门限（真航迹命中 86–96%，其他航迹约 2%）
        "bad_us": 2.0,              # 当前对象航迹相位残差超过此值才考虑改归
        "max_fit_frac": 0.4,        # 无航迹当前对象：此节点上整体贴合 X 的脉冲占比超过此值 → 不可分，不动
        "cross_geo": False,         # 跨型号：当前航迹方位也超门限时，相位+方位贴合 X 即可改归（验证有害：把无航迹 R06 整片并入 R07，关闭）
        "cross_fit_us": 0.2,        # 跨型号（方位未超门限）：X 相位残差上限
        "cross_bad_us": 1.0,        #   当前航迹相位残差下限
        "cross_ratio": 5.0,         #   且当前 / X 相位残差之比下限
        "tight_p_min": 0.25,        # 当前对象无航迹、相位 ≤ cross_fit_us 时型号概率下限（R05/R06 共用参数模式下真 R05 CPI 约 0.33）
        "check_min_cpi": 50,        # 自检：直接锚定 CPI 数下限
        "check_min_hit": 0.5,       # 自检：直接锚定 CPI 相位命中率下限（随机约 2%）
        "track_min_cpi": 20,        # 逐航迹自检：该航迹直接锚定 CPI 数下限（不足则不用这条航迹）
        "track_min_hit": 0.5,       # 逐航迹自检：相位命中率下限
        "head_chain": True,         # 回溯驻留起点（抖动重频驻留前段常被切成单独 CPI）
        "head_gap_mul": 2.6,        # 首尾间隔 ≤ min(倍数 × PRI, head_gap_us) 视为同一驻留（容许漏一个脉冲）
        "head_gap_us": 1700.0,
        "head_dur_ms": 30.0,        # 驻留总长上限
    },
    "tiny_local": {                 # 节点上极小局部簇（零散错挂的 CPI）逐 CPI 改归同节点方位连续的型号相容对象
        "enabled": True,
        "min_pulses": 150,          # 训练集真值单节点局部簇最小约 250 脉冲；碎簇 2–22 脉冲
        "win_s": 5.0,
        "type_p_min": 0.2,
        "free_type_n": 8,           # 脉冲数少于此值的 CPI 不做型号约束（指纹不可靠）
    },
    "merge_split": {                # 拆分对象重合并（精修/互斥之后）：同时段方位一致 + 发射无冲突 + 型号相容
        "enabled": True,
        "bin_s": 5.0,               # 同时段方位比较的时间格（格内取中位数）
        "interp_win_s": 5.0,        # 用对方前后此范围内最近的 CPI 插值方位
        "min_bins": 10,             # 可比较的（节点, 时间格）至少这么多
        "min_frac": 0.6,            # 方位差在门限内的格占比下限（CV/压力测试：同源拆分 0.64–0.98，异源 ≤0.25）
        "gate_scale": 0.5,          # 门限 = 节点 DOA 半宽误差 × 此系数（格内中位数）
        "conflict_frac": 0.06,      # 允许的冲突 CPI 比例（同源抖动串接伪影约 1–6%）
        "max_merges": 10,
    },
    "link": {
        "res_margin_deg": 2.0,      # 方位残差容差 = 节点 DOA 误差 + 余量
        "min_hit": 0.85,
        "min_cpis": 5,
        "max_cpis_per_object": 3000,
        "max_disp": 0.25,            # 关联残差离散度门限（MAD/测向误差），排除近旁静默航迹（同目标实测 ≤0.15，伴随 ≥0.33）
        "max_med_res": 2.0,          # 关联方位残差中位数上限（deg；同目标实测 ≤1.05）
        "tdoa_w": 1.0,               # TDOA 命中率在关联代价中的权重；0 关闭
        "tdoa_gate_us": 0.6,
        "tdoa_min_pairs": 30,        # 对象内 TDOA 配对数达到此值才使用
        "tdoa_min_hit": 0.3,         # 吻合比例低于此值 → 不可关联
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
