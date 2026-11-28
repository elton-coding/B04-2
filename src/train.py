#!/usr/bin/env python3
"""训练型号/状态模型并做本地评估。

用法：
  python src/train.py --data-root <训练数据目录> [--split-s 300] [--save models/b04_models.joblib]
  --split-s>0 时：仅用 toa<split 的数据训练，并在 toa>=split 的脉冲上报告指标（模拟新场景泛化）。
"""
import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b04 import io  # noqa: E402
from b04.config import load_config, load_knowledge  # noqa: E402
from b04.cpi import extract_cpis  # noqa: E402
from b04.evaluate import score  # noqa: E402
from b04.fingerprint import fit_fingerprint  # noqa: E402
from b04.pipeline import cluster_and_type, run_scenario  # noqa: E402
from b04.recognize import fit_state_models, fit_type_model, state_features  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpi_labels(a: pd.DataFrame, cpi_of: np.ndarray) -> pd.DataFrame:
    x = pd.DataFrame({"cpi": cpi_of, "tid": a.target_id.values, "cls": a.target_class.values,
                      "st": a.target_state.values})
    return x.groupby("cpi").agg(tid=("tid", lambda s: s.mode().iloc[0]), cls=("cls", lambda s: s.mode().iloc[0]),
                                st=("st", lambda s: s.mode().iloc[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-s", type=float, default=0.0)
    ap.add_argument("--save", default="")
    ap.add_argument("--drop-rate", type=float, default=0.03,
                    help="训练时随机丢弃脉冲比例，模拟测试集约 3%% 的漏脉冲")
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "default.yaml"))
    ap.add_argument("--type-model", default="", help="覆盖配置：fingerprint | gbdt")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    if args.type_model:
        cfg["recognize"]["type_model"] = args.type_model
    kn = load_knowledge(os.path.join(ROOT, "config", "knowledge.yaml"))
    data = io.load_inputs(args.data_root, keep_truth=True)
    for (sid, ver), (a, b) in data.items():
        logging.info("场景 %s_%s: A=%d B=%d", sid, ver, len(a), len(b))
        if args.drop_rate > 0:
            keep = np.random.default_rng(0).random(len(a)) >= args.drop_rate
            a = a[keep].reset_index(drop=True)
            logging.info("模拟漏脉冲 %.1f%% → 保留 %d", 100 * args.drop_rate, len(a))
        a["target_id"] = a.target_id.astype(np.int64)
        a["target_state"] = a.target_state.astype(np.int64)
        split_us = args.split_s * 1e6
        # 1) 型号模型
        cpi_of, cp = extract_cpis(a, cfg)
        cp = cp.join(cpi_labels(a, cpi_of), on="cpi")
        trn = (cp.t0 < split_us) if split_us > 0 else np.ones(len(cp), bool)
        type_clf = fit_type_model(cp[trn], cp.cls[trn])
        classes = sorted(kn["models"].keys())
        fp = fit_fingerprint(cp[trn], cp.cls[trn])
        models = {"type": type_clf, "fp": fp, "type_codes": {c: i for i, c in enumerate(classes)}}
        # 2) 状态模型：在预测得到的局部簇上构造特征，保证训练/推理一致
        _, cp2, _ = cluster_and_type(a, models, kn, cfg, b)
        cp2 = cp2.join(cpi_labels(a, cpi_of), on="cpi")
        feat = state_features(cp2, models["type_codes"])
        trn2 = (feat.t0 < split_us) if split_us > 0 else np.ones(len(feat), bool)
        m1, m2 = fit_state_models(feat[trn2], cp2.st[trn2])
        models.update({"state1": m1, "state2": m2, "sklearn": sklearn.__version__})
        # 3) 端到端评估
        pred, links, extras = run_scenario(a, b, models, kn, cfg)
        errs = io.validate_outputs(a, b, pred, links)
        logging.info("格式自检: %s", errs or "通过")
        ev_a = a[a.toa >= split_us] if split_us > 0 else a
        res = score(ev_a, b, pred[pred.pulse_id.isin(ev_a.pulse_id)], links)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        m = ev_a[["pulse_id", "target_id"]].merge(pred, on="pulse_id")
        summ = m.groupby("object_id").agg(pulses=("pulse_id", "size"), type=("target_class", "first"),
                                          tid=("target_id", lambda s: s.value_counts().index[0]),
                                          share=("target_id", lambda s: round(s.value_counts(normalize=True).iloc[0], 3)))
        summ = summ.join(links.set_index("object_id")[["track_id", "link_status"]])
        sc = extras["scores"]
        if len(sc):
            sc = sc.assign(object_id=sc.object_id + 1).sort_values("hit", ascending=False).drop_duplicates("object_id")
            summ = summ.join(sc.set_index("object_id")[["track_id", "hit", "med_res", "feasible"]].add_prefix("best_"))
        summ["true_track"] = summ.tid.map({v: k for k, v in b.groupby("track_id").target_id.first().items()})
        print(summ.sort_values("pulses", ascending=False).to_string())
        if args.save:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            joblib.dump(models, args.save, compress=3)
            logging.info("模型已保存: %s", args.save)


if __name__ == "__main__":
    main()
