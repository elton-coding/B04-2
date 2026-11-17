#!/bin/bash
# 按评测服务流程在 Linux/Docker 下验证提交包：解压 → docker build → 只读挂载 /input 运行 → 检查 /output。
# 用法（在 WSL Ubuntu 中）：
#   bash tools/validate_docker.sh <提交zip> <测试数据目录> [工作目录(默认 ~/b04-validate)] [--network none]
set -euo pipefail
ZIP="$1"; DATA="$2"; WORK="${3:-$HOME/b04-validate}"; NET="${4:-none}"
TAG="b04-validate:latest"

echo "== 清理工作目录 $WORK"
if [ -d "$WORK" ]; then docker run --rm -v "$WORK":/w python:3.11-slim rm -rf /w/pkg /w/out; fi
mkdir -p "$WORK/pkg" "$WORK/out"
python3 -m zipfile -e "$ZIP" "$WORK/pkg"
PKG=$(ls -d "$WORK"/pkg/*/ | head -1)
echo "== 包目录 $PKG"
for f in Dockerfile run.sh requirements.txt README.md src; do
  [ -e "$PKG/$f" ] || { echo "缺少必选项: $f"; exit 1; }
done
if grep -q $'\r' "$PKG/run.sh"; then echo "run.sh 含 CRLF"; exit 1; fi

echo "== docker build"
t0=$(date +%s)
docker build -q -t "$TAG" "$PKG"
echo "build 用时 $(( $(date +%s) - t0 )) s, 镜像大小 $(docker image inspect -f '{{.Size}}' "$TAG" | awk '{printf "%.0f MB", $1/1e6}')"

echo "== docker run (network=$NET)"
t0=$(date +%s)
CID=$(docker run -d --network "$NET" -v "$DATA":/input:ro -v "$WORK/out":/output "$TAG" --task all --test --out-dir /output)
peak=0
while [ "$(docker inspect -f '{{.State.Running}}' "$CID")" = "true" ]; do
  m=$(docker stats --no-stream --format '{{.MemUsage}}' "$CID" 2>/dev/null | awk '{print $1}')
  mb=$(echo "$m" | awk '/GiB/{sub("GiB","");printf "%d",$1*1024} /MiB/{sub("MiB","");printf "%d",$1} /KiB/{print 0}')
  [ -n "$mb" ] && [ "$mb" -gt "$peak" ] && peak=$mb
  sleep 2
done
rc=$(docker inspect -f '{{.State.ExitCode}}' "$CID")
docker logs "$CID" 2>&1 | tail -15
docker rm "$CID" >/dev/null
echo "run 用时 $(( $(date +%s) - t0 )) s, 退出码 $rc, 峰值内存约 ${peak} MB"
echo "== 输出"
ls -la "$WORK/out"
exit "$rc"
