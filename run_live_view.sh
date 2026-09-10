#!/usr/bin/env bash
# Terminale 2: finestra live su /camera/rgb (rqt_image_view non e' installato).
# Stesso isolamento ROS del terminale 1: vedi il commento in run_sim_824.sh.
set -euo pipefail

ENV_DIR=/home/vodka/miniconda3/envs/foundgraph
REPO=/home/vodka/Desktop/TiagoSara/FOUND-Dataset

exec env -i \
  HOME="$HOME" \
  USER="${USER:-vodka}" \
  TERM="${TERM:-xterm}" \
  PATH="$ENV_DIR/bin:/usr/bin:/bin" \
  DISPLAY="${DISPLAY:-:1}" \
  XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  "$ENV_DIR/bin/python" "$REPO/live_view.py" "$@"
