#!/usr/bin/env python3
"""Confronto tra lo STATO FINALE atteso della scena e lo stato finale del
grafo costruito dalla pipeline dgsg.

Differenza rispetto a watch_gt_vs_graph.py (che resta valido e indipendente):
quello confronta EVENTI nel tempo ("allo spawn a frame 11 corrisponde un
added entro 40 frame?"), questo confronta due FOTOGRAFIE finali ("alla fine
della run il grafo assomiglia a come la scena dovrebbe essere?").

Il confronto a eventi dipende da una run completa e non interrotta, e rende
i "remove" quasi invisibili (si confermano solo se la pipeline emette il
removed nella finestra giusta). Il confronto finale non ha questi problemi:
un oggetto rimosso e' semplicemente un oggetto che NON deve comparire nello
stato finale, indipendentemente da quando la pipeline se ne accorge.

Lo stato atteso si ricostruisce rigiocando lo script pianificato dall'inizio
alla fine -- non leggendo il gt-log degli eventi eseguiti: la domanda e'
"com'e' la scena SE fosse successo tutto quello che doveva".

Match per posizione (entro --max-dist metri) + similarita' semantica delle
label (CLIP testo<->testo, vedi compute_gt_text_embeddings.py).

Uso:
    python3 scripts/compare_final_state.py \
        --script experiments/FOUND/00824_live_0/household_experiments_scene_824_plan.json \
        --graph-stream experiments/FOUND/00824_live_0/graph_stream.jsonl \
        --gt-log experiments/FOUND/00824_live_0/gt_events.jsonl \
        --text-embeddings experiments/FOUND/00824_live_0/gt_text_embeddings.json \
        --out experiments/FOUND/00824_live_0/live_comparison.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

# Riuso diretto delle funzioni gia' scritte e verificate, invece di
# duplicarle: la trasformata map->Habitat in particolare porta con se'
# una lunga derivazione (vedi i docstring in watch_gt_vs_graph.py) che
# non va reimplementata a mano.
from scripts.watch_gt_vs_graph import (
    centroid_distance,
    frame0_offset_map_to_habitat,
    load_jsonl,
    map_centroid_to_habitat,
    name_matches_category,
)
from scripts.compute_gt_text_embeddings import template_to_category


# Oltre questa distanza due oggetti non sono plausibilmente lo stesso
# oggetto fisico.
#
# 0.9m, non 3m: con la trasformata corretta (vedi
# frame0_offset_map_to_habitat in watch_gt_vs_graph.py) la GT viene da un
# simulatore ed e' esatta per costruzione, quindi un oggetto ben
# ricostruito cade a POCHI CENTIMETRI dalla sua posizione vera --
# misurato: banana 1.2cm, bowl 1.2cm. I 3m iniziali servivano solo a
# compensare l'errore sistematico della trasformata sbagliata, e a quella
# distanza in una stanza di 81m2 il criterio di posizione non
# discriminava piu' nulla.
#
# Il margine fino a 0.9m copre il caso reale degli oggetti che la
# pipeline frammenta in piu' nodi: la scatola di zucchero e' stata
# spezzata in 6 oggetti distinti e i suoi frammenti veri cadono a
# 38/39/67cm (ognuno ricostruito da meno osservazioni, quindi con un
# centro meno preciso), mentre i tre falsi stanno a 2.5/3.0/5.2m e
# restano correttamente esclusi.
MAX_DIST_M = 0.9
# Sotto questa cosine similarity tra le due label (testo CLIP) il match e'
# considerato semanticamente implausibile anche se la distanza e' piccola.
# Calibrata sui dati reali di questa scena: le coppie che descrivono lo
# stesso oggetto stanno a >=0.877 ("rubiks cube" vs "puzzle cube", la
# coppia piu' difficile: parole diverse, stesso oggetto) o a 1.0 se la
# stringa coincide; le coppie non correlate stanno tutte tra 0.50 e 0.66
# ("gelatin box" vs "sugar box" 0.645 -- il falso match che il confronto
# per sole parole produceva, visto che condividono "box"; "spoon" vs
# "bowl" 0.658; "sponge" vs "pillow" 0.607). C'e' un vallone netto tra
# 0.66 e 0.877: 0.75 sta nel mezzo, con margine su entrambi i lati.
MIN_LABEL_SIM = 0.75
# Pesi del punteggio combinato: la label pesa piu' della distanza perche'
# la distanza soffre del drift, la label no.
W_SIM = 1.0
W_DIST = 0.5


def replay_script_to_final_state(script_path: Path):
    """Rigioca lo script dall'inizio alla fine e ritorna lo stato finale.

    DEVE essere sequenziale, non una mappa nome->ultima posizione vista:
    un oggetto puo' essere spawnato, mosso piu' volte e infine rimosso
    (verificato su questo script: golden_twitch_banana viene mossa DUE
    volte e poi rimossa, quindi alla fine NON deve esserci). Una mappa
    presa all'ultima occorrenza lo darebbe erroneamente presente.

    Nota sui campi: "spawn" identifica l'oggetto con "name" e porta il
    "template" (da cui si ricava la categoria fisica), mentre "move" e
    "remove" lo referenziano con "object" e NON hanno il template -- la
    categoria va recuperata dallo spawn corrispondente.

    Ritorna (expected_present, expected_absent, absent_categories):
      expected_present:  {name: {"position": [x,y,z], "category": str}}
      expected_absent:   [name, ...]  oggetti spawnati e poi rimossi
      absent_categories: {name: category}  per gli stessi nomi, serve a
        controllare se sono mai stati osservati nel grafo (vedi
        match_final_states / load_ever_observed_categories)
    """
    data = json.loads(script_path.read_text())
    state = {}
    categories = {}
    removed = []
    for step in data.get("steps", []):
        action = step.get("action")
        if action == "spawn":
            name = step.get("name")
            if not name:
                continue
            categories[name] = template_to_category(step["template"]) if step.get("template") else None
            state[name] = step.get("position")
        elif action == "move":
            name = step.get("object")
            if name in state and step.get("position") is not None:
                state[name] = step["position"]
        elif action == "remove":
            name = step.get("object")
            if name in state:
                del state[name]
                removed.append(name)

    expected_present = {
        name: {"position": pos, "category": categories.get(name)}
        for name, pos in state.items()
    }
    absent_categories = {name: categories.get(name) for name in removed}
    return expected_present, removed, absent_categories


def load_final_graph_state(graph_stream_path: Path, offset):
    """Ultimo frame di graph_stream.jsonl, con i centroidi portati nel
    frame Habitat (lo stesso in cui e' espressa la "position" dello script).

    Il file supera i 100MB: si itera a righe tenendo solo l'ultima, senza
    caricarlo in memoria.

    offset=None (gt-log senza 'frame0_anchor') -> i centroidi restano
    non convertiti e valgono None: il match ricadra' sulla sola label.
    """
    last = None
    with open(graph_stream_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None, []
    entry = json.loads(last)
    objects = []
    for o in entry.get("objects", []):
        objects.append({
            "idx": o.get("idx"),
            "category": o.get("category"),
            "detections": o.get("detections"),
            "centroid_habitat": map_centroid_to_habitat(o.get("centroid"), offset),
        })
    return entry.get("frame"), objects


def load_ever_observed_categories(graph_stream_path: Path):
    """Categorie DISTINTE apparse in QUALSIASI frame dello stream, non solo
    nell'ultimo.

    Serve a distinguere "rimosso davvero" da "mai rilevato" per gli
    expected_absent (vedi match_final_states): un oggetto remove-ato ha
    successo solo se la pipeline lo aveva prima osservato e poi lo ha
    fatto sparire, non se semplicemente non lo ha mai visto. Verificato
    sulla run 00824_live_0 (2026-09-12): "banana" non compare in nessuno
    dei 158 frame -- il remove veniva contato "riuscito" solo perche' la
    pipeline non ha mai rilevato l'oggetto, non perche' lo abbia rimosso.

    File grande (>100MB): si scorre a righe senza tenerle in memoria."""
    cats = set()
    if graph_stream_path is None or not graph_stream_path.exists():
        return cats
    with open(graph_stream_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for o in entry.get("objects", []):
                if o.get("category"):
                    cats.add(o["category"])
    return cats


def _cosine(a, b):
    if not a or not b:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def label_similarity(gt_name, gt_category, graph_category, embeddings):
    """Similarita' tra la label GT e quella del grafo.

    Preferisce CLIP testo<->testo (le due stringhe descrivono lo stesso
    oggetto fisico con parole diverse -- "rubiks cube" vs "puzzle cube" --
    e solo un embedding condiviso puo' misurarlo). Se gli embedding non
    sono disponibili ricade sul confronto per parole gia' usato dal
    confronto a eventi, che e' piu' grezzo ma non blocca l'esecuzione.

    Ritorna (similarity, method): similarity in [0,1], method per il report.
    """
    gt_ft = (embeddings.get("gt", {}).get(gt_name) or {}).get("clip_text_ft")
    graph_ft = embeddings.get("graph_categories", {}).get(graph_category)
    sim = _cosine(gt_ft, graph_ft)
    if sim is not None:
        return sim, "clip_text"
    # graph_ft manca per QUESTA categoria specifica (es. embedding
    # calcolati prima che il grafo esistesse, poi mai ricalcolati -- vedi
    # graph_categories vuoto nella run del 2026-09-12 17:14): se le due
    # stringhe coincidono esattamente e' comunque un match certo, non
    # serve CLIP per saperlo. Senza questo controllo un "rubiks cube"
    # atteso veniva scartato anche a fronte di un nodo "rubiks cube" nel
    # grafo, perche' _cosine(gt_ft, None) e' sempre None.
    if gt_category and graph_category and gt_category.strip().lower() == graph_category.strip().lower():
        return 1.0, "exact_string"
    # Fallback testuale, usato solo se mancano gli embedding. Le parole
    # generiche vanno escluse: "gelatin box" e "sugar box" condividono
    # "box" e name_matches_category le darebbe per equivalenti (verificato:
    # tart_setting_pack veniva matchato con una "sugar box" a 1.56m con
    # similarita' 1.0). Un match su una sola parola generica non e' un
    # match -- serve che coincida la parola che identifica l'oggetto.
    GENERIC = {"box", "cube", "block", "large", "small", "container", "object"}
    if gt_category and graph_category:
        gt_words = set(gt_category.lower().split())
        graph_words = set(graph_category.lower().split())
        shared = gt_words & graph_words
        if shared and shared - GENERIC:
            return 1.0, "word_overlap"
        return 0.0, "word_overlap"
    matched = name_matches_category(gt_name, graph_category, gt_category)
    return (1.0 if matched else 0.0), "word_overlap"


def match_final_states(expected_present, expected_absent, graph_objects,
                       embeddings, max_dist_m, ever_observed_categories=None,
                       absent_categories=None):
    """Match globale greedy tra oggetti GT attesi e oggetti del grafo finale.

    Greedy sul punteggio decrescente calcolato su TUTTE le coppie, non
    oggetto-per-oggetto nell'ordine dello script: altrimenti sarebbe
    l'ordine degli step a decidere chi si prende un candidato conteso.
    Ogni oggetto del grafo (idx) conferma al piu' un oggetto GT, come gia'
    fa used_idx nel confronto a eventi.
    """
    pairs = []
    diagnostics = {name: {"candidates_in_range": 0, "best_sim": None, "best_dist": None}
                   for name in expected_present}

    for name, exp in expected_present.items():
        for obj in graph_objects:
            dist = centroid_distance(exp["position"], obj["centroid_habitat"])
            # Centroide non convertibile: si tiene la coppia ma senza
            # contributo di distanza, il match dipendera' dalla sola label.
            if dist is not None and dist > max_dist_m:
                continue
            sim, method = label_similarity(name, exp["category"], obj["category"], embeddings)
            d = diagnostics[name]
            d["candidates_in_range"] += 1
            if d["best_sim"] is None or sim > d["best_sim"]:
                d["best_sim"] = sim
            if dist is not None and (d["best_dist"] is None or dist < d["best_dist"]):
                d["best_dist"] = dist
            if sim < MIN_LABEL_SIM:
                continue
            score = W_SIM * sim - (W_DIST * (dist / max_dist_m) if dist is not None else 0.0)
            pairs.append({
                "gt_name": name, "idx": obj["idx"], "graph_category": obj["category"],
                "distance_m": dist, "label_similarity": sim, "label_method": method,
                "score": score,
            })

    pairs.sort(key=lambda p: p["score"], reverse=True)
    matched = {}
    used_idx = set()
    for p in pairs:
        if p["gt_name"] in matched or p["idx"] in used_idx:
            continue
        matched[p["gt_name"]] = p
        used_idx.add(p["idx"])

    found, missing = [], []
    for name, exp in expected_present.items():
        if name in matched:
            m = dict(matched[name])
            m["gt_category"] = exp["category"]
            m["gt_position"] = exp["position"]
            found.append(m)
        else:
            d = diagnostics[name]
            if d["candidates_in_range"] == 0:
                reason = f"nessun oggetto del grafo entro {max_dist_m}m"
            elif d["best_sim"] is not None and d["best_sim"] < MIN_LABEL_SIM:
                reason = (f"{d['candidates_in_range']} candidati entro {max_dist_m}m ma "
                          f"label incompatibile (sim max {d['best_sim']:.2f} < {MIN_LABEL_SIM})")
            else:
                reason = "candidato gia' assegnato a un altro oggetto GT"
            missing.append({
                "gt_name": name, "gt_category": exp["category"],
                "gt_position": exp["position"], "reason": reason,
                "candidates_in_range": d["candidates_in_range"],
                "best_label_similarity": d["best_sim"],
                "nearest_candidate_m": d["best_dist"],
            })

    # Attesi-assenti: un oggetto rimosso dallo script che compare ancora
    # nel grafo finale e' un falso positivo (la pipeline non ha rimosso
    # cio' che doveva). Qui non si puo' usare la posizione: l'oggetto e'
    # stato rimosso, una "posizione finale" non esiste -- si cerca per
    # sola label tra gli oggetti non ancora assegnati.
    #
    # "assente nell'ultimo frame" da solo NON e' un remove riuscito: puo'
    # voler dire "mai rilevato" (la pipeline non ha mai visto l'oggetto,
    # quindi non ha mai potuto rimuoverlo) invece di "rilevato e poi
    # rimosso". Visto in produzione sulla run 00824_live_0 (2026-09-12):
    # "banana" non compare in NESSUNO dei 158 frame dello stream, eppure
    # veniva contato come remove riuscito. ever_observed_categories (da
    # load_ever_observed_categories, scorre TUTTO lo stream, non solo
    # l'ultimo frame) permette di distinguere i due casi. None disabilita
    # il controllo (retrocompatibile con chi non lo passa).
    false_positives, correctly_absent, not_evaluable = [], [], []
    for name in expected_absent:
        hits = []
        for obj in graph_objects:
            if obj["idx"] in used_idx:
                continue
            sim, method = label_similarity(name, None, obj["category"], embeddings)
            if sim >= MIN_LABEL_SIM:
                hits.append({"idx": obj["idx"], "graph_category": obj["category"],
                             "label_similarity": sim, "label_method": method})
        if hits:
            hits.sort(key=lambda h: h["label_similarity"], reverse=True)
            false_positives.append({"gt_name": name, "still_in_graph": hits})
            continue
        if ever_observed_categories is not None:
            gt_category = (absent_categories or {}).get(name)
            ever_seen = any(
                label_similarity(name, gt_category, cat, embeddings)[0] >= MIN_LABEL_SIM
                for cat in ever_observed_categories
            )
            if not ever_seen:
                not_evaluable.append({
                    "gt_name": name,
                    "motivo": ("mai osservato in nessun frame dello stream: un remove "
                               "\"riuscito\" contro un oggetto mai rilevato non e' "
                               "informativo, non conta come successo"),
                })
                continue
        correctly_absent.append(name)

    unassigned = [
        {"idx": o["idx"], "category": o["category"], "detections": o["detections"]}
        for o in graph_objects if o["idx"] not in used_idx
    ]
    return found, missing, false_positives, correctly_absent, unassigned, not_evaluable


def _transform_check(expected_present, found):
    """Auto-verifica della trasformata map->Habitat sui match trovati.

    La GT viene da un simulatore: e' esatta per costruzione, non e' una
    misura rumorosa. Quindi un oggetto ben ricostruito deve cadere a POCHI
    CENTIMETRI dalla sua posizione vera, e un residuo sistematicamente
    grande su TUTTI gli oggetti non e' imprecisione della pipeline ma un
    difetto nella catena di conversione delle coordinate.

    E' esattamente il modo in cui il bug precedente e' sfuggito a lungo:
    l'offset era sbagliato di ~3.8m e le statistiche uscivano vuote senza
    che nulla segnalasse il perche'. Qui il residuo mediano viene
    misurato e giudicato a ogni esecuzione, cosi' se la catena TF cambia
    in futuro il file lo dice invece di produrre numeri sbagliati in
    silenzio.
    """
    dists = sorted(f["distance_m"] for f in found if f.get("distance_m") is not None)
    if not dists:
        return {
            "status": "unknown",
            "detail": ("nessun match con distanza misurabile: la trasformata non e' "
                       "verificabile su questi dati (serve almeno un oggetto GT "
                       "ritrovato nel grafo)"),
        }
    n = len(dists)
    median = dists[n // 2] if n % 2 else (dists[n // 2 - 1] + dists[n // 2]) / 2
    if median <= 0.15:
        status, detail = "ok", "residuo mediano da simulatore: trasformata coerente"
    elif median <= 0.9:
        status, detail = "degraded", (
            "residuo mediano oltre i pochi centimetri attesi da una GT di "
            "simulatore: plausibile drift di tracking o oggetti frammentati in "
            "piu' nodi, ma la trasformata non e' grossolanamente sbagliata")
    else:
        status, detail = "suspect", (
            "residuo mediano troppo grande per una GT esatta per costruzione: "
            "sospetta trasformata map->Habitat sbagliata (rotazione o offset), "
            "NON imprecisione della pipeline -- verificare "
            "frame0_offset_map_to_habitat in watch_gt_vs_graph.py")
    return {
        "status": status,
        "detail": detail,
        "median_residual_m": round(median, 4),
        "min_residual_m": round(dists[0], 4),
        "max_residual_m": round(dists[-1], 4),
        "samples": n,
    }


def build_final_comparison(script_path, graph_stream_path, gt_log_path,
                           text_embeddings_path, max_dist_m):
    expected_present, expected_absent, absent_categories = replay_script_to_final_state(script_path)

    offset = frame0_offset_map_to_habitat(load_jsonl(gt_log_path)) if gt_log_path else None
    last_frame, graph_objects = load_final_graph_state(graph_stream_path, offset)
    ever_observed_categories = load_ever_observed_categories(graph_stream_path)

    embeddings = {}
    if text_embeddings_path and text_embeddings_path.exists():
        embeddings = json.loads(text_embeddings_path.read_text())

    # Segnale esplicito per un caso visto in produzione (run 00824_live_0,
    # 2026-09-12 17:14): gt_text_embeddings.json esisteva con "gt" popolato
    # ma "graph_categories" vuoto perche' calcolato prima che graph_stream
    # esistesse e mai ricalcolato dopo (la run e' crashata prima). Il
    # sintomo a valle era silenzioso: ogni sim CLIP tornava None e ogni
    # match veniva scartato, indistinguibile da un metodo che non
    # riconosce nulla. Qui il caso si nomina invece di lasciarlo degradare.
    graph_categories_missing = (
        bool(embeddings)
        and not embeddings.get("graph_categories")
        and len(graph_objects) > 0
    )

    found, missing, false_positives, correctly_absent, unassigned, not_evaluable = match_final_states(
        expected_present, expected_absent, graph_objects, embeddings, max_dist_m,
        ever_observed_categories=ever_observed_categories,
        absent_categories=absent_categories)

    n_present, n_absent = len(expected_present), len(expected_absent)
    # Il denominatore di expected_absent_ok esclude i remove non
    # valutabili (oggetto mai osservato in nessun frame -- vedi
    # match_final_states): altrimenti un remove ineseguibile per la
    # pipeline (non ha mai visto l'oggetto) abbasserebbe un rapporto che
    # dovrebbe misurare solo i remove che la pipeline poteva davvero
    # sbagliare o azzeccare.
    n_absent_evaluable = n_absent - len(not_evaluable)
    return {
        "expected_present": f"{len(found)}/{n_present}",
        "expected_absent_ok": f"{len(correctly_absent)}/{n_absent_evaluable}",
        "found": found,
        "missing": missing,
        "false_positives": false_positives,
        "correctly_absent": correctly_absent,
        "not_evaluable": not_evaluable,
        "graph_objects_unassigned": unassigned,
        "transform_check": _transform_check(expected_present, found),
        "meta": {
            "script_path": str(script_path),
            "graph_stream_path": str(graph_stream_path),
            "last_frame_in_graph": last_frame,
            "graph_objects_total": len(graph_objects),
            "max_dist_m": max_dist_m,
            "min_label_similarity": MIN_LABEL_SIM,
            "habitat_offset_available": offset is not None,
            "text_embeddings_available": bool(embeddings),
            "graph_categories_missing": graph_categories_missing,
            "generated_at_unix": time.time(),
            "note": (
                "Confronto tra STATI FINALI, non tra eventi. expected_present = "
                "oggetti che devono esistere alla fine, ottenuti rigiocando lo "
                "script pianificato in ordine (spawn inserisce, move sovrascrive "
                "la posizione, remove cancella) -- NON dal gt-log degli eventi "
                "eseguiti: la domanda e' 'com'e' la scena se fosse successo tutto "
                "quello che doveva'. Il replay deve essere sequenziale perche' un "
                "oggetto puo' essere spawnato, mosso piu' volte e infine rimosso "
                "(golden_twitch_banana: 2 move poi remove -> alla fine assente). "
                "Ogni atteso si matcha con l'ultimo frame di graph_stream.jsonl "
                "per posizione (centroidi portati in Habitat con la trasformata "
                "di watch_gt_vs_graph.py) entro max_dist_m E similarita' delle "
                "label via CLIP testo<->testo (necessaria perche' 'rubiks cube' e "
                "'puzzle cube' sono la stessa cosa con parole diverse). "
                "L'assegnazione e' greedy globale sui punteggi, cosi' non e' "
                "l'ordine dello script a decidere i candidati contesi, e ogni idx "
                "del grafo conferma al piu' un oggetto GT. false_positives = "
                "oggetti che lo script rimuove ma che sono ancora nel grafo. "
                "not_evaluable = oggetti rimossi dallo script ma MAI osservati "
                "in nessun frame dello stream: un remove 'riuscito' contro un "
                "oggetto mai rilevato non e' informativo (la pipeline non lo "
                "ha rimosso, semplicemente non lo ha mai visto), quindi non "
                "entra nel denominatore di expected_absent_ok."
            ),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, type=Path)
    ap.add_argument("--graph-stream", required=True, type=Path)
    ap.add_argument("--gt-log", type=Path, default=None,
                    help="serve solo per l'evento frame0_anchor, da cui si calibra "
                         "l'offset map->Habitat dei centroidi")
    ap.add_argument("--text-embeddings", type=Path, default=None,
                    help="output di compute_gt_text_embeddings.py; se assente si "
                         "ricade sul confronto delle label per parole")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-dist", type=float, default=MAX_DIST_M)
    args = ap.parse_args()

    comparison = build_final_comparison(args.script, args.graph_stream, args.gt_log,
                                        args.text_embeddings, args.max_dist)

    # Si aggiunge una sezione al file esistente senza toccare le altre:
    # il watcher a eventi continua a scrivere le sue e i due confronti
    # convivono nello stesso live_comparison.json.
    payload = {}
    if args.out.exists():
        try:
            payload = json.loads(args.out.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload["final_state_comparison"] = comparison

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(args.out)

    print(f"Scritto: {args.out}")
    if comparison["meta"]["graph_categories_missing"]:
        print("  ATTENZIONE: gt_text_embeddings.json non ha 'graph_categories' "
              "(o e' vuoto) pur avendo oggetti nel grafo -- ogni similarita' "
              "CLIP testo<->testo tornera' 0.00/None e i match verranno "
              "scartati quasi tutti. Rilanciare compute_gt_text_embeddings.py "
              "con --graph-stream prima di fidarsi di questi numeri.")
    print(f"  oggetti attesi presenti trovati: {comparison['expected_present']}")
    print(f"  oggetti attesi assenti, corretti: {comparison['expected_absent_ok']}")
    if comparison["false_positives"]:
        names = [f["gt_name"] for f in comparison["false_positives"]]
        print(f"  FALSI POSITIVI (rimossi ma ancora nel grafo): {names}")
    if comparison["not_evaluable"]:
        names = [n["gt_name"] for n in comparison["not_evaluable"]]
        print(f"  NON VALUTABILI (mai osservati in nessun frame, remove non "
              f"significativo): {names}")


if __name__ == "__main__":
    main()
