#!/usr/bin/env python3
"""打包 A 榜结果文件：先按赛题提交规则逐项自检，全部通过才写出 dist/<team>_result_<version>.zip。

zip 根目录直接放 [场景编号]_[数据版本]_pulse_predictions.csv 与 [场景编号]_[数据版本]_object_track_links.csv。
无效提交（格式不符/命名不规范）仍计入提交次数，故自检不通过时拒绝打包。

用法：python tools/pack_results.py --team <队名> --version v8 --data-root <测试数据目录> [--out-dir output]
"""
import argparse
import hashlib
import os
import sys
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from b04 import io  # noqa: E402


def check(a: pd.DataFrame, b: pd.DataFrame, f_pred: str, f_link: str, sid: str, ver: str) -> list[str]:
    errs = []
    pred = pd.read_csv(f_pred)
    links = pd.read_csv(f_link, dtype={"track_id": "Int64"})
    errs += io.validate_outputs(a, b, pred, links)
    for name, df in (("pulse_predictions", pred), ("object_track_links", links)):
        if set(df.scenario_id.astype(str)) != {sid} or set(df.data_version.astype(str)) != {ver}:
            errs.append(f"{name} 的 scenario_id/data_version 与输入不一致")
    for c in ("local_cluster_id", "object_id", "target_state"):
        if not pd.api.types.is_integer_dtype(pred[c]):
            errs.append(f"pulse_predictions.{c} 不是整数")
    if not set(pred.target_state.unique()).issubset({0, 1, 2}):
        errs.append("target_state 取值超出 0/1/2")
    for f in (f_pred, f_link):
        with open(f, "rb") as fh:
            head = fh.read(1 << 16)
        if head.startswith(b"\xef\xbb\xbf"):
            errs.append(f"{os.path.basename(f)} 带 BOM")
        if b"\r\n" in head:
            errs.append(f"{os.path.basename(f)} 为 CRLF 换行")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    args = ap.parse_args()
    files = []
    for (sid, ver), (a, b) in io.load_inputs(args.data_root).items():
        sid, ver = str(sid), str(ver)
        f_pred = os.path.join(args.out_dir, f"{sid}_{ver}_pulse_predictions.csv")
        f_link = os.path.join(args.out_dir, f"{sid}_{ver}_object_track_links.csv")
        for f in (f_pred, f_link):
            if not os.path.exists(f):
                sys.exit(f"缺少结果文件：{f}")
        errs = check(a, b, f_pred, f_link, sid, ver)
        if errs:
            sys.exit(f"场景 {sid} 自检未通过：" + "；".join(errs))
        pred = pd.read_csv(f_pred, usecols=["object_id"])
        links = pd.read_csv(f_link)
        print(f"场景 {sid}/{ver}：脉冲 {len(pred)}（输入 {len(a)}），对象 {pred.object_id.nunique()}，"
              f"matched {int((links.link_status == 'matched').sum())} / no_match {int((links.link_status == 'no_match').sum())}，自检通过")
        files += [f_pred, f_link]
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", f"{args.team}_result_{args.version}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.basename(f))
    md5 = hashlib.md5(open(out, "rb").read()).hexdigest()
    print(f"{out}  {os.path.getsize(out) / 1e6:.1f} MB  md5 {md5}")
    for f in files:
        print(f"  {os.path.basename(f)}  md5 {hashlib.md5(open(f, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
