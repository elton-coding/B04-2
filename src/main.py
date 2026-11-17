#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B04 方案入口。

评测服务通过 run.sh 调用：--task <值> [--test] --out-dir /output
输入：DATA_ROOT（默认 /input）下的 A/B 类传感器 CSV
输出：[场景]_[版本]_pulse_predictions.csv、[场景]_[版本]_object_track_links.csv

算法流水线见 docs/技术方案.md，模块按里程碑逐步实现。
"""
import argparse
import os


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/input"))
    ap.add_argument("--out-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    return ap.parse_args()


def main():
    args = parse_args()
    print("DATA_ROOT =", args.data_root)
    print("out_dir   =", args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    # TODO(里程碑1): 读取 -> 预分选 -> 局部聚类 -> 型号/状态 -> 跨平台融合 -> A-B 关联 -> 写出并自检


if __name__ == "__main__":
    main()
