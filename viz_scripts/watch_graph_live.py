#!/usr/bin/env python3

"""Segue in tempo reale lo scene graph costruito da dynamic_gsg_real_ssim.py.

La pipeline scrive una riga JSON per frame in graph_stream.jsonl; questo
script la segue (come `tail -f`) e ridisegna lo stato del grafo a ogni
aggiornamento: quali oggetti sono stati rilevati finora, quanti gaussiani
compongono la mappa, e quali oggetti vengono rimossi quando la scena cambia.

Uso (in un secondo terminale, mentre la pipeline gira):
    python3 viz_scripts/watch_graph_live.py experiments/FOUND/00824_dynamic_0
"""

import argparse
import json
import os
import time
from pathlib import Path

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def render(state, events):
    os.system("clear")
    frame = state.get("frame", 0)
    objs = state.get("objects", [])
    gauss = state.get("num_gaussians", 0)

    print(f"{BOLD}DynamicGSG — scene graph live{RESET}")
    print(f"{DIM}{'─' * 62}{RESET}")
    print(f"  frame {BOLD}{frame}{RESET}    "
          f"oggetti {BOLD}{len(objs)}{RESET}    "
          f"gaussiani {BOLD}{gauss:,}{RESET}")
    print(f"{DIM}{'─' * 62}{RESET}")

    if not objs:
        print(f"  {DIM}nessun oggetto ancora rilevato…{RESET}")
    for o in objs:
        cat = o.get("category") or f"(classe {o.get('class_id')})"
        det = o.get("detections", 0)
        bar = "▪" * min(det, 20)
        print(f"  {CYAN}[{o['idx']:>3}]{RESET} {cat:<26} {DIM}{bar} {det}{RESET}")

    if events:
        print(f"\n{DIM}{'─' * 62}{RESET}")
        print(f"  {BOLD}cambiamenti nella scena{RESET}")
        for ev in events[-8:]:
            print(f"  {RED}frame {ev['frame']}: rimossi {ev['removed']}{RESET}")

    print(f"\n{DIM}Ctrl+C per uscire{RESET}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="cartella della run, es. experiments/FOUND/00824_dynamic_0")
    args = parser.parse_args()

    path = Path(args.run_dir) / "graph_stream.jsonl"
    print(f"In attesa di {path} …")
    while not path.exists():
        time.sleep(0.5)

    state, events = {}, []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("removed"):
                events.append(entry)
            state = entry
            render(state, events)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto.")
