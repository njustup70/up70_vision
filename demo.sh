#!/usr/bin/env bash
# 从 clone 到看见第一张结果图：一条命令跑完。
#
# 用大白话说
#   以前要照 README 的顺序手敲四条命令（建镜像 -> 环境自检 -> OpenCV 冒烟 -> 跑一批合成图），
#   中途哪一步挂了还得自己判断挂在哪。这个脚本把四步串起来，谁挂了就报谁，
#   最后再核一遍产物齐不齐——"跑完了但图没存下来"也算失败。
#
# 用法（在仓库根目录）
#   ./demo.sh                 # 完整四步
#   SKIP_BUILD=1 ./demo.sh    # 镜像已经建过时跳过第 1 步，省 2~3 分钟
#
# 跑完应该看到
#   output/demo/ 里四种产物：frame_*.png / results.json / detections.csv / summary.json
#   末行「✅ demo 全部通过」
#
# 退出码
#   0 全过；非 0 = 第一个失败步骤的退出码（步骤号与可复制的命令都会打出来）

set -u -o pipefail

COMPOSE=(docker compose -f .devcontainer/docker-compose.yml)
SERVICE=up70-vision
STEP=0

# 跑一步：失败就报"第几步、哪条命令、退出码多少"，然后按该命令的退出码退出。
# 不用 `set -e`：要自己接住退出码，才能把"挂在第几步"打清楚。
step() {
    local title="$1"
    shift
    STEP=$((STEP + 1))
    printf '\n==> [%d/4] %s\n' "$STEP" "$title"
    "$@"
    local code=$?
    if [ "$code" -ne 0 ]; then
        printf '\n❌ 第 %d 步失败：%s（退出码 %d）\n' "$STEP" "$title" "$code" >&2
        printf '   单独重跑：%s\n' "$*" >&2
        printf '   排查顺序见 README「快速开始」与「已知限制」两节。\n' >&2
        exit "$code"
    fi
}

if [ "${SKIP_BUILD:-0}" = "1" ]; then
    printf '\n==> [1/4] 跳过镜像构建（SKIP_BUILD=1）\n'
    STEP=1
else
    step "构建镜像（首次约 2~3 分钟）" "${COMPOSE[@]}" build
fi

step "环境自检（末行应是「结果：全部通过」）" \
    "${COMPOSE[@]}" run --rm "$SERVICE" python -m up70_vision doctor

step "OpenCV 冒烟（末行应是「OpenCV 冒烟测试 -> PASS」）" \
    "${COMPOSE[@]}" run --rm "$SERVICE" python -m up70_vision.tools.demo

step "跑一批合成图（产出 output/demo/）" \
    "${COMPOSE[@]}" run --rm "$SERVICE" python -m up70_vision run --set input.source=synthetic

# ---- 产物核对：四样缺一样都算没跑成 ----
missing=""
for name in results.json detections.csv summary.json; do
    [ -f "output/demo/$name" ] || missing="$missing $name"
done
ls output/demo/frame_*.png >/dev/null 2>&1 || missing="$missing frame_*.png"
if [ -n "$missing" ]; then
    printf '\n❌ 产物缺失：%s\n' "$missing" >&2
    printf '   落盘开关在 configs/default.yaml 的 output 段（save_overlay/save_json/save_csv）。\n' >&2
    exit 1
fi

printf '\n✅ demo 全部通过：output/demo/ 有 4 种产物（PNG / results.json / detections.csv / summary.json）\n'
