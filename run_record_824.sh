#!/usr/bin/env bash
# Terminale 3: registra la sequenza dinamica della scena 824.
# Stesso isolamento ROS del terminale 1: vedi il commento in run_sim_824.sh.
#
# Il nodo (terminale 1) deve essere gia' in esecuzione: il recorder aspetta
# il primo frame su /camera/rgb e /camera/depth e fallisce dopo 15 s se il
# simulatore non sta pubblicando.
set -euo pipefail

ENV_DIR=/home/vodka/miniconda3/envs/foundgraph
REPO=/home/vodka/Desktop/TiagoSara/FOUND-Dataset
OUT="${OUT:-/home/vodka/dynamic-gsg/data/FOUND/00824_dynamic}"

# Nessun --frames-per-waypoint/--step-m qui sotto: usa i default dello
# script (camminata lenta, sosta lunga davanti a ogni cambiamento).
# Sovrascrivili passandoli a questo comando, es.:
#   ./run_record_824.sh --step-m 0.35 --walk-pause-s 0.1
exec env -i \
  HOME="$HOME" \
  USER="${USER:-vodka}" \
  TERM="${TERM:-xterm}" \
  PATH="$ENV_DIR/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  "$ENV_DIR/bin/python" "$REPO/record_dynamic_sequence.py" \
    --out "$OUT" \
    --script "$REPO/scripts/generated/household_experiments_scene_824.json" \
    "$@"
