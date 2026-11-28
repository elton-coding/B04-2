"""数据读取、字段兼容、结果写出与格式自检。"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# 实际数据列名存在拼写差异（lontitude），统一为 longitude
_RENAME = {
    "sensor_lontitude": "sensor_longitude",
    "target_lontitude": "target_longitude",
    "lontitude": "target_longitude",
    "longitude": "target_longitude",
    "latitude": "target_latitude",
    "height": "target_height",
}

A_COLS = ["scenario_id", "data_version", "pulse_id", "datetime", "toa", "param1", "param2",
          "param3", "doa", "sensor_longitude", "sensor_latitude", "sensor_height", "platform_id"]
B_COLS = ["scenario_id", "data_version", "track_id", "datetime", "target_longitude",
          "target_latitude", "target_height", "target_type", "target_class"]


def _read_csv(path: str, usecols: list[str] | None = None) -> pd.DataFrame:
    if usecols is not None:
        have = _sniff_header(path)
        usecols = [c for c in have if c in usecols] or None
    try:
        # 分块流式读取（64 MB 一块）再拼接：峰值内存约为结果表的 1.7 倍，整表读取约 3–5 倍
        import pyarrow.csv as pacsv

        r = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=64 << 20),
                           convert_options=pacsv.ConvertOptions(include_columns=usecols) if usecols else None)
        parts = [batch.to_pandas() for batch in r]
        if not parts:
            raise ValueError("空文件")
        df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
        del parts
        return df
    except Exception:  # 类型推断跨块不一致等：回退为整表读取
        try:
            return pd.read_csv(path, engine="pyarrow", usecols=usecols)
        except Exception:
            return pd.read_csv(path, low_memory=False, usecols=usecols)


def _sniff_header(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        return [c.strip() for c in f.readline().split(",")]


def discover_inputs(data_root: str) -> tuple[list[str], list[str]]:
    """按表头识别 A 类（含 pulse_id）与 B 类（含 track_id）CSV。"""
    files = sorted(glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True))
    a_files, b_files = [], []
    for f in files:
        cols = _sniff_header(f)
        if "pulse_id" in cols:
            a_files.append(f)
        elif "track_id" in cols:
            b_files.append(f)
    return a_files, b_files


def normalize_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns and k.startswith("sensor")})
    df = _dedup_columns(df)
    for c in ["scenario_id", "data_version"]:  # 类别型：千万行时比逐行字符串省约 1 GB
        df[c] = df[c].astype("category").cat.rename_categories(lambda x: str(x))
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["toa", "param1", "param2", "param3", "doa", "sensor_longitude", "sensor_latitude", "sensor_height"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    pid = pd.to_numeric(df["platform_id"], errors="coerce")
    df["platform_id"] = pid.fillna(pid.mode().iloc[0] if pid.notna().any() else 0).astype("int64")
    df["pulse_id"] = df["pulse_id"].astype("int64")
    # toa 单位自检：赛题约定 μs；若同节点相邻脉冲间隔中位数远超正常量级（交错脉冲流通常为数十~数百 μs），按 ns 换算。
    # 只对 (节点, toa) 两列排序，避免整表复制（大场景内存峰值）
    pv, tv = df["platform_id"].to_numpy(), df["toa"].to_numpy()
    o = np.lexsort((tv, pv))
    d = np.diff(tv[o])
    d = d[(pv[o][1:] == pv[o][:-1]) & np.isfinite(d)]
    med = float(np.median(d)) if len(d) else np.nan
    if np.isfinite(med) and med > 5000:
        import logging
        logging.getLogger("b04").warning("toa 相邻间隔中位数 %.0f 异常，按 ns→μs 换算", med)
        df["toa"] = df["toa"] / 1000.0
    return df


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """改名后出现同名列（两种拼写并存）：逐行取第一个非空值。"""
    if not df.columns.duplicated().any():
        return df
    out = {}
    for c in dict.fromkeys(df.columns):
        x = df.loc[:, df.columns == c]
        out[c] = x.bfill(axis=1).iloc[:, 0] if x.shape[1] > 1 else x.iloc[:, 0]
    return pd.DataFrame(out)


def normalize_b(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns and not k.startswith("sensor")})
    df = _dedup_columns(df)
    for c, v in (("target_type", "unknown"), ("target_class", ""), ("target_height", 0.0)):
        if c not in df.columns:  # 可选列缺失时补默认值
            df[c] = v
    for c in ["scenario_id", "data_version"]:
        df[c] = df[c].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["target_longitude", "target_latitude", "target_height"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    tid = pd.to_numeric(df["track_id"], errors="coerce")
    df = df[tid.notna()].copy()
    df["track_id"] = tid[tid.notna()].astype("int64").values
    df["target_type"] = df["target_type"].astype(str).str.strip().str.lower()
    return df


A_READ = A_COLS + ["sensor_lontitude"]
TRUTH = ["target_id", "target_class", "target_state"]


def load_inputs(data_root: str, keep_truth: bool = False) -> dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]]:
    """返回 {(scenario_id, data_version): (A, B)}。推理时只读所需列（真值列仅训练 / 评估用，keep_truth=True），
    单文件不做 concat、单场景不再复制，大场景内存峰值约减半。"""
    a_files, b_files = discover_inputs(data_root)
    if not a_files:
        raise FileNotFoundError(f"在 {data_root} 下未找到 A 类脉冲 CSV")
    cols = A_READ + (TRUTH if keep_truth else [])
    parts = [_read_csv(f, cols) for f in a_files]
    A = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    del parts
    A = normalize_a(A)
    B = pd.DataFrame(columns=B_COLS)
    if b_files:
        try:
            B = normalize_b(pd.concat([_read_csv(f) for f in b_files], ignore_index=True))
        except Exception:  # B 类读取失败不影响 A 类处理（按无航迹运行）
            import logging
            import traceback
            logging.getLogger("b04").error("B 类数据读取失败，按无航迹运行：%s", traceback.format_exc())
            B = pd.DataFrame(columns=B_COLS)
    out = {}
    if A.scenario_id.nunique() == 1 and A.data_version.nunique() == 1:
        key = (A.scenario_id.iloc[0], A.data_version.iloc[0])
        b = B[(B.scenario_id == key[0]) & (B.data_version == key[1])] if len(B) else B
        out[key] = (A, b.reset_index(drop=True))
        return out
    for key, a in A.groupby(["scenario_id", "data_version"], sort=False):
        b = B[(B.scenario_id == key[0]) & (B.data_version == key[1])] if len(B) else B
        out[key] = (a.reset_index(drop=True), b.reset_index(drop=True))
    return out


# ---------------------------------------------------------------- 输出
PRED_COLS = ["scenario_id", "data_version", "pulse_id", "local_cluster_id", "object_id",
             "target_class", "target_state"]
LINK_COLS = ["scenario_id", "data_version", "object_id", "track_id", "link_status"]


def validate_outputs(a: pd.DataFrame, b: pd.DataFrame, pred: pd.DataFrame, links: pd.DataFrame) -> list[str]:
    errs = []
    if list(pred.columns) != PRED_COLS:
        errs.append(f"pulse_predictions 列不符: {list(pred.columns)}")
    if len(pred) != len(a) or not pred.pulse_id.is_unique or set(pred.pulse_id) != set(a.pulse_id):
        errs.append("pulse_id 未做到逐一且唯一覆盖")
    if pred[["local_cluster_id", "object_id", "target_class", "target_state"]].isna().any().any():
        errs.append("pulse_predictions 存在空值")
    # 同一 local_cluster 只能属于一个 platform 与一个 object
    m = pred[["local_cluster_id", "object_id"]].assign(platform_id=a.set_index("pulse_id").loc[pred.pulse_id, "platform_id"].values)
    if (m.groupby("local_cluster_id").platform_id.nunique() > 1).any():
        errs.append("local_cluster_id 跨 platform")
    if (m.groupby("local_cluster_id").object_id.nunique() > 1).any():
        errs.append("local_cluster_id 对应多个 object_id")
    if list(links.columns) != LINK_COLS:
        errs.append(f"object_track_links 列不符: {list(links.columns)}")
    if set(links.object_id) != set(pred.object_id) or not links.object_id.is_unique:
        errs.append("links 中 object_id 与预测对象不一致或重复")
    if not set(links.link_status).issubset({"matched", "no_match"}):
        errs.append("link_status 取值非法")
    mt = links[links.link_status == "matched"]
    if mt.track_id.isna().any() or not set(mt.track_id.astype("int64")).issubset(set(b.track_id)):
        errs.append("matched 航迹不存在")
    if not mt.track_id.is_unique:
        errs.append("一个航迹关联了多个对象")
    if links[links.link_status == "no_match"].track_id.notna().any():
        errs.append("no_match 的 track_id 非空")
    return errs


def write_outputs(out_dir: str, scenario: str, version: str, pred: pd.DataFrame, links: pd.DataFrame) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, f"{scenario}_{version}_pulse_predictions.csv")
    p2 = os.path.join(out_dir, f"{scenario}_{version}_object_track_links.csv")
    pred.to_csv(p1, index=False, lineterminator="\n")  # 与 Linux 评测环境一致，避免 Windows 下写出 CRLF
    links = links.copy()
    links["track_id"] = links["track_id"].astype("Int64")
    links.to_csv(p2, index=False, lineterminator="\n")
    return p1, p2
