#!/usr/bin/env python3
"""Monitor live del progresso di una run: riscrive un file di stato ogni
N secondi mentre la pipeline gira, e stampa a schermo un riepilogo tipo
"spawn 3/10" con le label.

Riusa la STESSA logica di scripts/report_scene_results.py (build_final_comparison
+ build_summary): non e' un calcolo diverso, e' lo stesso confronto rifatto
a ripetizione mentre graph_stream.jsonl cresce. Ogni rifresco legge
SOLO l'ultima riga scritta finora nel grafo (vedi
compare_final_state.load_final_graph_state), quindi il numero cresce mano
a mano che la pipeline mappa nuovi oggetti -- e' un vero progresso, non
uno snapshot fisso.

Non sostituisce il visualizzatore grafico: quello (utils/live_viewer.py,
finestra OpenCV con camera + maschere + grafo) parte da solo insieme alla
pipeline, quando config['live_viewer'] non e' messo a False. Questo script
e' il secondo canale che chiedevi: un file di testo/JSON leggibile mentre
la run va avanti, utile quando non si ha (o non si vuole tenere aperta)
una finestra grafica, o quando serve leggere i numeri via script.

Uso:
    python3 scripts/watch_live_progress.py \\
        --script experiments/FOUND/00824_dynamic_0/household_experiments_scene_824_plan.json \\
        --graph-stream experiments/FOUND/00824_dynamic_0/graph_stream.jsonl \\
        --gt-log experiments/FOUND/00824_dynamic_0/gt_events.jsonl \\
        --text-embeddings experiments/FOUND/00824_dynamic_0/gt_text_embeddings.json \\
        --scene 00824 --out-dir experiments/FOUND/00824_dynamic_0 \\
        --interval 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from scripts.compare_final_state import (
    MAX_DIST_M,
    build_final_comparison,
    load_ever_observed_categories,
)
from scripts.report_scene_results import (
    build_map_snapshot,
    build_summary,
    planned_actions,
    write_log,
)
from scripts.watch_gt_vs_graph import (
    build_comparison as build_event_comparison,
    load_name_to_template_category,
    load_planned_names,
    load_planned_totals,
)


def _clear_line():
    # \r + spazi: sovrascrive la riga precedente nel terminale invece di
    # accumulare una stampa per refresh, che con --interval basso
    # diventerebbe illeggibile in pochi minuti.
    sys.stdout.write("\r" + " " * 100 + "\r")


def print_progress_line(summary):
    m = summary["meta"]
    p = summary["progress"]
    ts = datetime.now().strftime("%H:%M:%S")
    _clear_line()
    line = (f"[{ts}] frame {m['ultimo_frame_elaborato']}  |  "
            f"spawn {p['spawn']}  move {p['move']}  remove {p['remove']}  |  "
            f"{summary['totale_oggetti_nel_grafo']} oggetti nel grafo")
    sys.stdout.write(line)
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, type=Path)
    ap.add_argument("--graph-stream", required=True, type=Path)
    ap.add_argument("--gt-log", type=Path, default=None)
    ap.add_argument("--text-embeddings", type=Path, default=None)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-dist", type=float, default=MAX_DIST_M)
    ap.add_argument("--interval", type=float, default=5.0,
                    help="secondi tra un refresh e l'altro")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    live_path = args.out_dir / f"{args.scene}_live.json"
    log_path = args.out_dir / f"{args.scene}_live_log.txt"
    map_path = args.out_dir / f"{args.scene}_mappa.json"

    actions, categories = planned_actions(args.script)
    planned_totals = load_planned_totals(args.script)
    planned_names = load_planned_names(args.script)
    name_to_template_category = load_name_to_template_category(args.script)

    print(f"Monitor live: {args.scene}")
    print(f"  grafo:      {args.graph_stream}")
    print(f"  progresso:  {live_path}")
    print(f"  mappa:      {map_path}")
    print(f"  refresh ogni {args.interval}s -- Ctrl+C per fermare")
    print()

    embeddings = (json.loads(args.text_embeddings.read_text())
                  if args.text_embeddings and args.text_embeddings.exists() else {})

    last_frame_seen = None
    # Categorie viste in QUALSIASI frame, per verificare gli spawn di oggetti
    # poi rimossi (vedi report_scene_results.spawn_seen_in_stream). Ricalcolarle
    # a ogni refresh significherebbe rileggere l'intero graph_stream.jsonl
    # (centinaia di MB) ogni --interval secondi: si aggiorna solo quando il
    # frame e' avanzato, come gia' si fa per il log e la mappa.
    ever_observed = set()
    while True:
        try:
            if not args.graph_stream.exists():
                print(f"\r[in attesa che {args.graph_stream} venga creato...]", end="")
                time.sleep(args.interval)
                continue

            comparison = build_final_comparison(
                args.script, args.graph_stream, args.gt_log,
                args.text_embeddings, args.max_dist,
            )
            curr_frame = comparison["meta"]["last_frame_in_graph"]
            # Va aggiornato PRIMA di build_summary, che lo usa per verificare
            # gli spawn poi rimossi.
            if curr_frame != last_frame_seen:
                ever_observed = load_ever_observed_categories(args.graph_stream)
            summary = build_summary(args.scene, args.script, comparison, actions, categories,
                                    gt_log_path=args.gt_log,
                                    ever_observed_categories=ever_observed,
                                    embeddings=embeddings)

            # Confronto A EVENTI (non a stato finale): per ogni spawn/move/
            # remove gia' avvenuto nel simulatore finora, dice se il grafo lo
            # ha gia' confermato -- si aggiorna man mano che gli oggetti
            # appaiono, invece di restare piatto finche' la run non finisce
            # (progress sopra rigioca l'intero script e confronta contro lo
            # stato FINALE atteso, quindi resta a "0/10" anche quando il
            # simulatore ha gia' spawnato 4 oggetti se nessuno di quei 4
            # sopravvive fino alla fine). Stessa logica gia' in
            # watch_gt_vs_graph.py, qui incorporata cosi' un solo file
            # (00824_live.json) porta entrambe le viste invece di richiedere
            # un secondo processo mai avviato da live_824.sh.
            if args.gt_log is not None:
                event_comparison = build_event_comparison(
                    args.gt_log, args.graph_stream, planned_totals,
                    planned_names, name_to_template_category,
                )
                summary["progress_live"] = event_comparison["progress"]

            summary["aggiornato_alle"] = datetime.now().isoformat(timespec="seconds")

            tmp = live_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            tmp.replace(live_path)

            # Il log dettagliato (chi e' stato trovato, chi manca e perche')
            # si riscrive solo quando il frame elaborato e' cambiato: e' il
            # file piu' pesante da generare e da leggere, non serve
            # rigenerarlo se il grafo non e' avanzato dall'ultimo refresh.
            if curr_frame != last_frame_seen:
                write_log(log_path, args.scene, args.script, comparison,
                          actions, categories, summary)
                map_snapshot = build_map_snapshot(args.scene, args.graph_stream,
                                                  args.gt_log, comparison, args.max_dist)
                map_tmp = map_path.with_suffix(".tmp")
                with open(map_tmp, "w") as f:
                    json.dump(map_snapshot, f, indent=2, ensure_ascii=False)
                map_tmp.replace(map_path)
                last_frame_seen = curr_frame

            print_progress_line(summary)
        except Exception as exc:
            # Diagnostico: un refresh fallito (es. graph_stream a meta'
            # scrittura) non deve mai fermare il monitor.
            _clear_line()
            print(f"[watch_live_progress] refresh fallito, riprovo ({exc})")

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nmonitor fermato.")
