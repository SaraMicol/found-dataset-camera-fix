#!/usr/bin/env bash
# Terminale 1: nodo Habitat sulla scena 824.
#
# Perche' 'env -i': l'env conda foundgraph contiene un ROS 2 Humble RoboStack
# completo e coerente. Se il terminale ha anche /ros2_humble/install/setup.bash
# sourceato, Python carica la meta' Python di nav_msgs da un prefisso e la
# libreria nativa dall'altro -> "undefined symbol:
# nav_msgs__msg__goals__convert_from_py" (quel nav_msgs ha Goals.msg, quello
# conda no). Qui si riparte da un ambiente vuoto e si reintroducono solo le
# variabili necessarie: nessuna contaminazione possibile, qualunque cosa sia
# stata sourceata nella shell chiamante.
set -euo pipefail

ENV_DIR=/home/vodka/miniconda3/envs/foundgraph
REPO=/home/vodka/Desktop/TiagoSara/FOUND-Dataset

# La quota della camera la calcola gia' record_dynamic_sequence.py
# (HUMAN_EYE_HEIGHT_M = 1.55 m, altezza occhi di una persona in piedi,
# fissa: non dipende dall'altezza dell'oggetto). Qui va lasciato 0, perche'
# make_cfg() SOMMA sensor_height sopra la quota di "eye" gia' scelta dal
# recorder: un valore diverso da zero la alza di nuovo e riproduce lo stesso
# bug gia' visto (prima 1.5 m -> vista da drone a ~63 gradi; poi 0.35 m ->
# vista da terra verso l'alto). Da toccare solo se si pilota la camera
# direttamente via /habitat/set_agent_pose senza passare dal recorder:
#   HABITAT_SENSOR_HEIGHT=0.2 ./run_sim_824.sh
SENSOR_H="${HABITAT_SENSOR_HEIGHT:-0.0}"

exec env -i \
  HOME="$HOME" \
  USER="${USER:-vodka}" \
  TERM="${TERM:-xterm}" \
  PATH="$ENV_DIR/bin:/usr/bin:/bin" \
  DISPLAY="${DISPLAY:-:1}" \
  XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  HABITAT_SENSOR_HEIGHT="$SENSOR_H" \
  HABITAT_VIEWER_EXAMPLES=/home/vodka/build_habitat/habitat-sim/examples \
  HABITAT_SCENE="$REPO/habitat/hm3d-val-habitat-v0.2/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb" \
  HABITAT_SCENE_DATASET="$REPO/habitat/hm3d-val-semantic-configs-v0.2/hm3d_annotated_basis.scene_dataset_config.json" \
  "$ENV_DIR/bin/python" "$REPO/habitat_camera_objects_node.py"
