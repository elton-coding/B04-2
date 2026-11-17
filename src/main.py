#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B04 方案入口。

评测服务通过 run.sh 调用：--task <值> [--test] --out-dir /output
输入：DATA_ROOT（默认 /input）下的 A/B 类传感器 CSV（按表头自动识别）
输出：[场景]_[版本]_pulse_predictions.csv、[场景]_[版本]_object_track_links.csv
"""
import argparse
import logging
import os
import sys
import traceback

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b04 import io  # noqa: E402
from b04.config import load_config, load_knowledge  # noqa: E402
from b04.pipeline import run_scenario  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/input"))
    ap.add_argument("--out-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "default.yaml"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models", "b04_models.joblib"))
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("b04")
    log.info("DATA_ROOT=%s out_dir=%s task=%s", args.data_root, args.out_dir, args.task)
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)
    kn = load_knowledge(os.path.join(ROOT, "config", "knowledge.yaml"))
    models = joblib.load(args.models)
    data = io.load_inputs(args.data_root)
    failed = 0
    for (sid, ver), (a, b) in data.items():
        log.info("场景 %s_%s: A=%d 脉冲, B=%d 点 / %d 航迹", sid, ver, len(a), len(b),
                 b.track_id.nunique() if len(b) else 0)
        try:
            pred, links, _ = run_scenario(a, b, models, kn, cfg)
            errs = io.validate_outputs(a, b, pred, links)
            if errs:
                log.warning("格式自检问题: %s", errs)
            p1, p2 = io.write_outputs(args.out_dir, sid, ver, pred, links)
            log.info("已写出 %s, %s", p1, p2)
        except Exception:
            failed += 1
            traceback.print_exc()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
