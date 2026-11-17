#!/usr/bin/env python3
"""打包提交代码：dist/<team>_code_<version>.zip，内含同名顶层目录（与官方模板一致）。

用法：python tools/pack.py --team <队名> --version v1
"""
import argparse
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = ["Dockerfile", "run.sh", "requirements.txt", "README.md", "src", "config", "models", "output/.gitkeep"]
LF_EXT = (".sh", ".py", ".yaml", ".txt", ".md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    name = f"{args.team}_code_{args.version}"
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    out = os.path.join(ROOT, "dist", name + ".zip")
    files = []
    for item in INCLUDE:
        p = os.path.join(ROOT, item)
        if os.path.isdir(p):
            for d, dirs, fs in os.walk(p):
                dirs[:] = [x for x in dirs if x != "__pycache__"]
                files += [os.path.join(d, f) for f in fs if not f.endswith(".pyc")]
        elif os.path.exists(p):
            files.append(p)
        else:
            raise FileNotFoundError(item)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo(name + "/"), "")
        for f in sorted(files):
            rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
            data = open(f, "rb").read()
            if rel.endswith(LF_EXT) or rel == "Dockerfile":
                data = data.replace(b"\r\n", b"\n")  # 容器内脚本必须为 LF
            info = zipfile.ZipInfo(f"{name}/{rel}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if rel.endswith(".sh") else 0o644) << 16
            z.writestr(info, data)
    print(out, f"{os.path.getsize(out) / 1e6:.2f} MB, {len(files)} files")


if __name__ == "__main__":
    main()
