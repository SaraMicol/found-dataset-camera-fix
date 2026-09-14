#!/usr/bin/env bash
# Watchdog per la run live della scena 824.
#
# Controlla periodicamente che la pipeline dgsg sia viva e la fa ripartire
# se muore. Gira sulla macchina in background, indipendente da qualsiasi
# sessione interattiva: continua a funzionare anche a terminale chiuso.
#
# Distingue tre casi:
#   - pipeline viva            -> non fa nulla
#   - run finita regolarmente  -> esce (marker "=== FINE ===" nel log main)
#   - pipeline morta a meta'   -> rilancia live_824.sh
#
# Uso:
#   nohup bash scripts/watchdog_824.sh > logs/watchdog_824.log 2>&1 &
#   disown
#
# Per fermarlo:
#   pkill -f watchdog_824.sh
set -uo pipefail

REPO=/home/vodka/dynamic-gsg
LOGDIR=$REPO/logs
STATE=$LOGDIR/watchdog_824_state.log

# Ogni quanto controllare. 30s e' un compromesso: abbastanza reattivo da
# non perdere minuti di lavoro, abbastanza raro da non pesare.
INTERVAL=30

# Oltre questo numero di riavvii il watchdog si ferma: se la pipeline muore
# subito e ripetutamente, riavviarla all'infinito non risolve nulla e
# riempie il disco di log. Meglio fermarsi e lasciare la diagnosi a un umano.
MAX_RESTARTS=10

# Se la pipeline muore prima di questo tempo dall'avvio, il riavvio viene
# comunque contato ma segnalato come sospetto: indica un errore all'avvio
# (config rotto, GPU occupata) piuttosto che un OOM a meta' run.
MIN_HEALTHY_S=120

restarts=0

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATE"
}

pipeline_alive() {
  pgrep -f "dynamic_gsg_real_ssim" >/dev/null 2>&1
}

run_finished() {
  # La run e' completata solo se lo script principale ha scritto il marker
  # finale. Senza questo controllo il watchdog rilancerebbe in eterno una
  # run andata a buon fine.
  local main
  main=$(ls -t "$LOGDIR"/live824_*.log 2>/dev/null | head -1)
  [ -n "$main" ] && grep -q "=== FINE ===" "$main" 2>/dev/null
}

last_frame() {
  local pl
  pl=$(ls -t "$LOGDIR"/pipeline824live_*.log 2>/dev/null | head -1)
  [ -n "$pl" ] && grep -o "frame [0-9]* num of objects: [0-9]*" "$pl" 2>/dev/null | tail -1
}

crash_cause() {
  local pl
  pl=$(ls -t "$LOGDIR"/pipeline824live_*.log 2>/dev/null | head -1)
  [ -n "$pl" ] && grep -E "Error|Traceback|out of memory|StandardGpuResources" "$pl" 2>/dev/null | tail -2 | tr '\n' ' '
}

log "watchdog avviato (controllo ogni ${INTERVAL}s, max ${MAX_RESTARTS} riavvii)"

while true; do
  if run_finished; then
    log "RUN COMPLETATA regolarmente: watchdog non serve piu', esco."
    exit 0
  fi

  if ! pipeline_alive; then
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      log "CRASH ma raggiunto il limite di $MAX_RESTARTS riavvii: mi fermo."
      log "  ultimo frame: $(last_frame)"
      log "  causa:        $(crash_cause)"
      exit 1
    fi

    restarts=$((restarts + 1))
    log "CRASH rilevato (riavvio $restarts/$MAX_RESTARTS)"
    log "  ultimo frame: $(last_frame)"
    log "  causa:        $(crash_cause)"

    # Pulizia: senza questo, i processi orfani della run morta (nodo
    # Habitat, recorder, monitor) restano vivi e la nuova run trova la
    # porta ROS occupata e la GPU parzialmente allocata.
    pkill -9 -f "live_824.sh" 2>/dev/null
    pkill -9 -f "habitat_camera_objects_node.py" 2>/dev/null
    pkill -9 -f "record_dynamic_sequence_live.py" 2>/dev/null
    pkill -9 -f "watch_live_progress.py" 2>/dev/null
    sleep 5

    started=$(date +%s)
    cd "$REPO" || exit 1
    nohup bash "$REPO/scripts/live_824.sh" > "$LOGDIR/live824_restart_${restarts}.log" 2>&1 &
    disown
    log "  rilanciata live_824.sh"

    # Attesa di avvio: la pipeline impiega ~45s a caricare i modelli, prima
    # di allora pgrep la troverebbe assente e il watchdog conterebbe un
    # secondo crash inesistente.
    sleep 90

    if ! pipeline_alive; then
      elapsed=$(( $(date +%s) - started ))
      if [ "$elapsed" -lt "$MIN_HEALTHY_S" ]; then
        log "  ATTENZIONE: morta di nuovo dopo ${elapsed}s -- errore all'avvio, non un OOM a meta' run"
      fi
    else
      log "  pipeline ripartita correttamente"
    fi
  fi

  sleep "$INTERVAL"
done
