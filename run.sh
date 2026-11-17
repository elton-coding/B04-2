#!/bin/bash
# 评测服务入口：ENTRYPOINT 调用本脚本并追加 --task / --test / --out-dir 等参数
set -e
echo "=== B04 方案启动 ==="
echo "DATA_ROOT = ${DATA_ROOT:-/input}"
echo "参数: $*"
cd /app/program/solution
exec python3 src/main.py "$@"
