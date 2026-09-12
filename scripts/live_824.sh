#!/usr/bin/env bash
# LIVE: nodo Habitat + pipeline dgsg (bridge ROS) + monitor live, poi il
# recorder che cammina e pubblica ogni frame in tempo reale. Niente file
# intermedi su disco: la pipeline processa il frame appena arriva, il
# gt-log si scrive nello stesso istante in cui l'evento avviene davvero.
#
# ORDINE DI AVVIO (non intercambiabile):
#   1. nodo Habitat        (run_sim_824.sh)       -- pubblica /camera/*
#   2. pipeline dgsg        (dgsg_live.py)          -- si iscrive, aspetta
#   3. monitor live         (watch_live_progress)   -- legge cio' che 2 scrive
#   4. recorder che cammina (run_record_824_live.sh) -- SOLO dopo che 2 e'
#      pronta: ogni frame pubblicato senza un consumatore aspetta l'ack
#      fino a --frame-ack-timeout-s (60s) prima di procedere con un warning.
#
# Applicate le stesse correzioni della notte del 2026-09-11/12 (vedi
# configs/found/dgsg_live.py): use_dam=False e removal_opacity_threshold
# alzata a 0.3. Lo stride qui non si applica: il bridge riceve un frame
# alla volta in tempo reale, non un elenco di file da campionare.
set -uo pipefail

REPO=/home/vodka/dynamic-gsg
FOUND=/home/vodka/Desktop/TiagoSara/FOUND-Dataset
PY=/home/vodka/miniconda3/envs/dgsg/bin/python
# Il run del 2026-09-12 17:14 e' morto per CUDA OOM nell'encoder SAM al
# frame 158 (1.55 GiB riservati da PyTorch ma non allocati, frammentazione
# -- vedi il messaggio d'errore stesso). expandable_segments riduce la
# frammentazione dell'allocatore e non ha altro effetto sulla pipeline.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR"
MAIN=$LOGDIR/live824_${STAMP}.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

D=$REPO/experiments/FOUND/00824_live_0
mkdir -p "$D"
GT=$D/gt_events.jsonl
rm -f "$GT"
SCRIPT_PLAN="$FOUND/scripts/generated/household_experiments_scene_824.json"
cp "$SCRIPT_PLAN" "$D/household_experiments_scene_824_plan.json"

# Pulizia preventiva: un residuo del nodo Habitat da un run precedente
# farebbe ripartire la scena con oggetti gia' spawnati da un test passato
# (vedi Troubleshooting in README_FOUND_824.md).
pkill -f "habitat_camera_objects_node.py" 2>/dev/null
pkill -f "record_dynamic_sequence_live.py" 2>/dev/null
sleep 2

say "=== LIVE 824: nodo Habitat + pipeline + monitor + camminata ==="

# ---------------------------------------------------------- 1. simulatore
say "1/4: nodo Habitat"
"$FOUND/run_sim_824.sh" > "$LOGDIR/sim824live_${STAMP}.log" 2>&1 &
SIM=$!
say "  avviato (PID $SIM), attendo che sia pronto"
ready=0
for _ in $(seq 1 60); do
  grep -q "Habitat ROS viewer ready" "$LOGDIR/sim824live_${STAMP}.log" 2>/dev/null && { ready=1; break; }
  kill -0 "$SIM" 2>/dev/null || break
  sleep 2
done
[ "$ready" -eq 1 ] || { say "  ERRORE: simulatore non pronto"; kill "$SIM" 2>/dev/null; exit 1; }
say "  pronto"

# ---------------------------------------------------------- 2. pipeline
say "2/4: pipeline dgsg (bridge ROS + finestra live)"
DISPLAY=:1 "$PY" scripts/dynamic_gsg_real_ssim.py "$REPO/configs/found/dgsg_live.py" \
  > "$LOGDIR/pipeline824live_${STAMP}.log" 2>&1 &
PIPE=$!
say "  avviata (PID $PIPE), attendo il caricamento dei modelli (~45s)"
sleep 45
kill -0 "$PIPE" 2>/dev/null || { say "  ERRORE: pipeline morta subito, vedi $LOGDIR/pipeline824live_${STAMP}.log"; kill "$SIM" 2>/dev/null; exit 1; }

# ---------------------------------------------------------- 3. monitor live
say "3/4: monitor live"
"$PY" "$REPO/scripts/compute_gt_text_embeddings.py" \
  --script "$D/household_experiments_scene_824_plan.json" \
  --out "$D/gt_text_embeddings.json" >> "$MAIN" 2>&1 \
  && say "  embedding label pronti" || say "  embedding falliti (fallback su match per parole)"

"$PY" "$REPO/scripts/watch_live_progress.py" \
  --script "$D/household_experiments_scene_824_plan.json" \
  --graph-stream "$D/graph_stream.jsonl" \
  --gt-log "$GT" \
  --text-embeddings "$D/gt_text_embeddings.json" \
  --scene 00824 --out-dir "$D" --interval 5 \
  > "$LOGDIR/watch824live_${STAMP}.log" 2>&1 &
WATCH=$!
say "  avviato (PID $WATCH) -> $D/00824_live.json"

# ---------------------------------------------------------- 4. camminata
say "4/4: recorder che cammina (pubblica al bridge, --gt-log attivo)"
"$FOUND/run_record_824_live.sh" --gt-log "$GT" > "$LOGDIR/record824live_${STAMP}.log" 2>&1
say "  camminata finita (codice $?)"

# ------------------------------------------------------------ chiusura
say "attendo che la pipeline finisca di processare gli ultimi frame in coda (60s)"
sleep 60
kill "$WATCH" 2>/dev/null
kill "$PIPE" 2>/dev/null
sleep 5
kill "$SIM" 2>/dev/null
kill -9 "$PIPE" "$SIM" 2>/dev/null

say "report finale"
"$PY" "$REPO/scripts/compute_gt_text_embeddings.py" \
  --script "$D/household_experiments_scene_824_plan.json" \
  --graph-stream "$D/graph_stream.jsonl" \
  --out "$D/gt_text_embeddings.json" >> "$MAIN" 2>&1
"$PY" "$REPO/scripts/report_scene_results.py" \
  --script "$D/household_experiments_scene_824_plan.json" \
  --graph-stream "$D/graph_stream.jsonl" \
  --gt-log "$GT" \
  --text-embeddings "$D/gt_text_embeddings.json" \
  --scene 00824 --out-dir "$D" >> "$MAIN" 2>&1 \
  && say "  report OK" || say "  report FALLITO"

# Deduplica gli oggetti che il matching cross-frame ha spezzato in piu'
# nodi (es. 11 "pillow" per un solo cuscino poco visibile in molti frame
# -- vedi la docstring di dedup_graph_objects.py per il meccanismo). Non
# tocca graph_stream.jsonl ne' i numeri gia' scritti sopra: e' una vista
# derivata in piu', utile per capire quanto della frammentazione osservata
# e' rumore di detection piuttosto che oggetti fisici davvero distinti.
"$PY" "$REPO/scripts/dedup_graph_objects.py" \
  --graph-stream "$D/graph_stream.jsonl" \
  --out "$D/00824_graph_deduped.json" >> "$MAIN" 2>&1 \
  && say "  dedup grafo OK" || say "  dedup grafo FALLITO"

say "=== FINE ==="
[ -f "$D/00824_summary.json" ] && "$PY" -c "
import json; d=json.load(open('$D/00824_summary.json'))
print(f\"  spawn {d['spawn']}  move {d['move']}  remove {d['remove']}\")
" | tee -a "$MAIN"
