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

# Arrow 默认内存池（jemalloc / mimalloc）释放后不归还系统，CSV 读入后常驻内存多出 2–3 GB；改用系统分配器
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b04 import io  # noqa: E402
from b04.config import load_config, load_knowledge  # noqa: E402
from b04.pipeline import run_scenario  # noqa: E402
from b04.safety import run_with_fallback  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/input"))
    ap.add_argument("--out-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "default.yaml"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models", "b04_models.joblib"))
    ap.add_argument("--report", action="store_true",
                    help="额外生成态势要素、本体图谱与态势报告到 <out-dir>/report/（--task report 等效）")
    return ap.parse_args()


def make_report(a, b, pred, links, extras, kn, out_dir):
    import json

    from b04 import agent, ontology, report
    from b04.association import track_table
    from b04.situation import build_facts

    log = logging.getLogger("b04")
    tracks = track_table(b, extras["base_time"]) if len(b) else b
    facts = build_facts(a, b, pred, links, extras, kn, tracks)
    os.makedirs(out_dir, exist_ok=True)
    sc = facts["scenario"]
    stem = os.path.join(out_dir, f"{sc['scenario_id']}_{sc['data_version']}")
    with open(stem + "_facts.json", "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=1, default=str)
    info = ontology.export(facts, kn, out_dir)
    nar = agent.narrative(facts)
    log.info("本体三元组 %d；大模型研判：%s", info["triples"], "已生成" if nar else "未启用/降级为模板")
    path = report.write_report(facts, a, pred, tracks, kn, out_dir, nar)
    log.info("态势报告：%s", path)


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
            # 分级降级：完整流程 → 安全模式 → 不用 B 航迹 → 应急基线；输出先修复再自检，保证总能写出合法结果
            pred, links, extras, mode = run_with_fallback(a, b, models, kn, cfg, run_scenario)
            errs = io.validate_outputs(a, b, pred, links)
            if errs:
                log.warning("格式自检问题: %s", errs)
            p1, p2 = io.write_outputs(args.out_dir, sid, ver, pred, links)
            log.info("已写出 %s, %s", p1, p2)
            if args.report or args.task in ("report", "full") or os.environ.get("B04_REPORT") == "1":
                try:
                    if extras:
                        make_report(a, b, pred, links, extras, kn, os.path.join(args.out_dir, "report"))
                except Exception:  # 报告失败不影响结果表
                    traceback.print_exc()
        except Exception:
            failed += 1
            traceback.print_exc()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
