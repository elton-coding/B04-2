#!/usr/bin/env python3
"""泛化验证：时间分折交叉验证 + 压力测试/消融（模拟 B 榜未知场景）。

用法：
  python tools/robustness.py --data-root <训练数据目录> --out docs/泛化验证结果.md [--folds 3] [--no-stress] [--no-cv]
每折：用其余时段训练全部模型，只把该折时段的数据（截成独立"新场景"）送入推理，按真值评分。
压力测试在最后一折上进行：加大漏脉冲、去掉节点、去掉/伪造航迹、关闭锚定、无 B 数据等。
"""
import argparse
import copy
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from b04 import io  # noqa: E402
from b04.config import load_config, load_knowledge  # noqa: E402
from b04.evaluate import score  # noqa: E402
from b04.pipeline import run_scenario  # noqa: E402
from b04.training import fit_models  # noqa: E402

COLS = ["cluster_ARI", "cluster_pairF1", "type_acc", "state_acc", "state_macroF1", "state_F1_ex5s", "fusion_ARI",
        "link_acc_obj", "link_acc_track", "n_objects", "n_true", "sec"]


def window(a, b, t0, t1):
    aw = a[(a.toa >= t0) & (a.toa < t1)].reset_index(drop=True)
    if len(b):
        base = (a.datetime - pd.to_timedelta(a.toa, unit="us")).median().round("s")
        tb = (b.datetime - base).dt.total_seconds() * 1e6
        bw = b[(tb >= t0 - 2e6) & (tb < t1 + 2e6)].reset_index(drop=True)
    else:
        bw = b
    return aw, bw


def evaluate(name, a, b, models, kn, cfg, log, skip_us=0.0):
    """skip_us：窗口不从场景零点开始时，开头这段的跟踪段被截断（真实场景不存在），另报去掉它后的状态宏 F1。"""
    t = time.time()
    pred, links, _ = run_scenario(a, b, models, kn, cfg)
    errs = io.validate_outputs(a, b, pred, links)
    r = score(a, b, pred, links)
    m = a[["pulse_id", "toa", "target_state"]].merge(pred[["pulse_id", "target_state"]], on="pulse_id", suffixes=("", "_p"))
    m = m[m.toa >= a.toa.min() + skip_us]
    r["state_F1_ex5s"] = float(f1_score(m.target_state, m.target_state_p, average="macro"))
    r["sec"] = round(time.time() - t, 1)
    r["format"] = "通过" if not errs else ";".join(errs)
    log.info("[%s] %s", name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
    return r


def fake_tracks(b, rng, near_m=300.0, far_m=3000.0):
    """复制部分航迹并平移：近距（伴随假象）与远距（虚假航迹），target_id 置空。"""
    out = [b]
    tids = sorted(b.track_id.unique())
    nid = max(tids) + 100
    for k, tid in enumerate(rng.choice(tids, size=min(4, len(tids)), replace=False)):
        x = b[b.track_id == tid].copy()
        off = near_m if k % 2 == 0 else far_m
        ang = rng.uniform(0, 2 * np.pi)
        x["target_longitude"] += off * np.sin(ang) / (111320 * np.cos(np.radians(x.target_latitude)))
        x["target_latitude"] += off * np.cos(ang) / 110574
        x["track_id"] = nid + k
        x["target_id"] = np.nan
        out.append(x)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "泛化验证结果.md"))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--drop-rate", type=float, default=0.03)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--no-stress", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("b04").setLevel(logging.WARNING)
    log = logging.getLogger("robust")
    cfg = load_config()
    kn = load_knowledge(os.path.join(ROOT, "config", "knowledge.yaml"))
    (a, b), = io.load_inputs(args.data_root).values()
    rng = np.random.default_rng(0)
    a = a[rng.random(len(a)) >= args.drop_rate].reset_index(drop=True)
    a["target_id"] = a.target_id.astype(np.int64)
    a["target_state"] = a.target_state.astype(np.int64)
    T = a.toa.max() + 1
    edges = np.linspace(0, T, args.folds + 1)
    rows = []
    models_last = None
    for k in range(args.folds):
        t0, t1 = edges[k], edges[k + 1]
        if args.no_cv and k < args.folds - 1:
            continue
        log.info("== 第 %d 折：测试 %.0f–%.0f s", k + 1, t0 / 1e6, t1 / 1e6)
        a_tr = a[(a.toa < t0) | (a.toa >= t1)]
        models = fit_models(a_tr, b, kn, cfg)
        aw, bw = window(a, b, t0, t1)
        rows.append({"实验": f"CV 第{k + 1}折 ({t0 / 1e6:.0f}–{t1 / 1e6:.0f}s)", **evaluate(f"fold{k + 1}", aw, bw, models, kn, cfg, log, 5e6 if k > 0 else 0.0)})
        models_last, win = models, (t0, t1)
    if not args.no_stress and models_last is not None:
        aw, bw = window(a, b, *win)
        srng = np.random.default_rng(1)
        variants = []
        c = copy.deepcopy(cfg); c["anchor"]["enabled"] = False
        variants.append(("关闭航迹锚定", aw, bw, c))
        variants.append(("无 B 数据", aw, bw.iloc[0:0], cfg))
        variants.append(("漏脉冲 10%", aw[srng.random(len(aw)) >= 0.072].reset_index(drop=True), bw, cfg))
        for pid in sorted(aw.platform_id.unique())[::3]:
            variants.append((f"去掉节点 {pid}", aw[aw.platform_id != pid].reset_index(drop=True), bw, cfg))
        keep = srng.choice(sorted(bw.track_id.unique()), size=bw.track_id.nunique() // 2, replace=False)
        variants.append(("去掉一半航迹", aw, bw[bw.track_id.isin(keep)].reset_index(drop=True), cfg))
        variants.append(("增加 4 条伪造航迹（近 300 m / 远 3 km）", aw, fake_tracks(bw, srng), cfg))
        for name, av, bv, cv in variants:
            log.info("== 压力测试：%s", name)
            try:
                rows.append({"实验": name, **evaluate(name, av, bv, models_last, kn, cv, log, 5e6)})
            except Exception as e:  # noqa: BLE001
                log.exception("失败")
                rows.append({"实验": name, "format": f"异常：{e}"})
    df = pd.DataFrame(rows)
    cols = ["实验"] + [c for c in COLS if c in df.columns] + ["format"]
    df = df[cols]
    md = ["# 泛化验证结果", "",
          f"数据：P1A_train（模拟 {args.drop_rate:.0%} 漏脉冲）；{args.folds} 折时间交叉验证：其余时段训练，测试时段截为独立场景推理。",
          "压力测试在最后一折的模型与测试时段上进行。",
          "state_F1_ex5s：去掉窗口开头 5 s 后的状态宏 F1（第 2、3 折窗口不从场景零点开始，开头处进行中的跟踪段被截断，是切分伪影；真实场景从零点开始）。", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        md.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v) for v in r) + " |")
    cv = df[df["实验"].str.startswith("CV")]
    if len(cv) > 1:
        md += ["", "CV 均值：" + "，".join(f"{c} {cv[c].mean():.4f}（最差 {cv[c].min():.4f}）"
                                          for c in ["cluster_ARI", "type_acc", "state_acc", "state_macroF1", "state_F1_ex5s", "fusion_ARI", "link_acc_obj"])]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
