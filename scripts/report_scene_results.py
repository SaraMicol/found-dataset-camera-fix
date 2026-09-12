#!/usr/bin/env python3
"""Due file di risultato per una run: un riepilogo semplice e un log diagnostico.

    <scena>_summary.json   quanti spawn/move/remove il grafo ha ritrovato,
                           con le label -- "spawn 7/10" e i nomi.
    <scena>_log.txt        perche' ognuno e' andato come e' andato: distanze,
                           similarita', candidati scartati e il motivo.

La separazione e' voluta: il primo si legge in dieci secondi per sapere
com'e' andata, il secondo si legge quando il primo dice qualcosa di strano
e si vuole capire il perche'.

Lo stato atteso si ricostruisce rigiocando lo script pianificato (spawn
inserisce, move sposta, remove cancella), quindi funziona anche se la run
si e' interrotta a meta': gli oggetti mai apparsi risultano semplicemente
non trovati, con il motivo scritto nel log.

Uso:
    python3 scripts/report_scene_results.py \\
        --script experiments/FOUND/00824_dynamic_0/household_experiments_scene_824_plan.json \\
        --graph-stream experiments/FOUND/00824_dynamic_0/graph_stream.jsonl \\
        --gt-log experiments/FOUND/00824_dynamic_0/gt_events.jsonl \\
        --text-embeddings experiments/FOUND/00824_dynamic_0/gt_text_embeddings.json \\
        --scene 00824 --out-dir experiments/FOUND/00824_dynamic_0
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
    MIN_LABEL_SIM,
    build_final_comparison,
    frame0_offset_map_to_habitat,
    label_similarity,
    load_ever_observed_categories,
    load_final_graph_state,
    load_jsonl,
    replay_script_to_final_state,
)
from scripts.compute_gt_text_embeddings import template_to_category


def gt_progress_so_far(gt_log_path, actions):
    """Quanti spawn/move/remove sono REALMENTE avvenuti finora nel
    simulatore, letti da gt_events.jsonl -- non "quanti il grafo ha
    ritrovato" (quello e' progress.spawn/move/remove, il confronto con la
    pipeline), ma "quanti il simulatore ha effettivamente emesso finora".

    Serve a poter monitorare la run mentre gira: se qui c'e' scritto
    "spawn 4/10" e sotto in progress.spawn c'e' "1/10", vuol dire che il
    simulatore ha gia' spawnato 4 oggetti ma la pipeline ne ha ritrovato
    uno solo -- un disallineamento vero da seguire in tempo reale, diverso
    dal "7/10" di progress.spawn che conta sullo stato FINALE atteso (e
    quindi resta piatto finche' la run non finisce). Senza questo blocco,
    gt_events.jsonl (i fatti) e 00824_live.json (il confronto) non si
    potevano leggere fianco a fianco.
    """
    events = load_jsonl(gt_log_path) if gt_log_path else []
    happened = {"spawn": [], "move": [], "remove": []}
    for e in events:
        action = e.get("action")
        if action in happened and e.get("name"):
            happened[action].append(e["name"])

    out = {}
    for action in ("spawn", "move", "remove"):
        total = len(actions.get(action, []))
        done = happened[action]
        out[action] = f"{len(done)}/{total}"
        out[f"{action}_avvenuti"] = done
    return out


def planned_actions(script_path: Path):
    """Azioni pianificate nello script, in ordine, con nome e categoria.

    spawn identifica l'oggetto con "name" e porta il "template" (da cui la
    categoria fisica); move e remove lo referenziano con "object" e non
    hanno template -- la categoria si recupera dallo spawn.
    """
    data = json.loads(script_path.read_text())
    categories = {}
    actions = {"spawn": [], "move": [], "remove": []}
    for step in data.get("steps", []):
        action = step.get("action")
        if action == "spawn":
            name = step.get("name")
            if not name:
                continue
            if step.get("template"):
                categories[name] = template_to_category(step["template"])
            actions["spawn"].append(name)
        elif action in ("move", "remove"):
            name = step.get("object")
            if name:
                actions[action].append(name)
    return actions, categories


def _matched_event(f):
    """Un evento GT trovato, con A QUALE oggetto del grafo e' stato
    associato -- idx, categoria rilevata, posizione, distanza. Stessa idea
    di "matched_graph_events" in live_comparison.json (watch_gt_vs_graph.py):
    non basta dire "trovato", va detto CON COSA, altrimenti l'unico modo
    per saperlo e' aprire il log diagnostico."""
    return {
        "nome": f["gt_name"],
        "label_attesa": f["gt_category"],
        "posizione_attesa": [round(x, 3) for x in f["gt_position"]],
        "matched_graph_object": {
            "idx": f["idx"],
            "label": f["graph_category"],
            "distanza_m": round(f["distance_m"], 3) if f.get("distance_m") is not None else None,
            "somiglianza_label": round(f["label_similarity"], 3),
        },
    }


def _missing_event(m, categories):
    """Un evento GT non trovato: nessun matched_graph_object (lista vuota,
    non assente) cosi' la forma e' la stessa di un evento trovato -- e il
    motivo per cui non c'e' nessuna associazione."""
    return {
        "nome": m["gt_name"],
        "label_attesa": m.get("gt_category") or categories.get(m["gt_name"], m["gt_name"]),
        "posizione_attesa": [round(x, 3) for x in m["gt_position"]] if m.get("gt_position") else None,
        "matched_graph_object": None,
        "motivo": m["reason"],
    }


def spawn_seen_in_stream(name, category, ever_observed_categories, embeddings):
    """Lo spawn di un oggetto POI RIMOSSO e' stato osservato dalla pipeline?

    Serve per gli oggetti che il piano spawna e poi cancella: il loro spawn
    e' un evento realmente avvenuto e va verificato, ma non si puo' cercarlo
    nello stato FINALE del grafo (li' l'oggetto, giustamente, non c'e' piu').
    Si guarda quindi se la sua categoria e' comparsa in QUALSIASI frame dello
    stream -- cioe' se la pipeline lo ha visto mentre c'era.

    Limite noto: il match e' per SOLA LABEL, senza posizione, perche'
    load_ever_observed_categories conserva solo le stringhe delle categorie e
    non i centroidi per frame. E' piu' debole del match posizionale usato per
    gli spawn che sopravvivono fino alla fine: una categoria gia' presente
    nell'arredamento della scena (es. un "bowl" preesistente) puo' confermare
    uno spawn che la pipeline non ha mai davvero visto. Per questo l'esito e'
    marcato nel JSON con il metodo usato, invece di essere spacciato per una
    conferma piena.
    """
    for observed in ever_observed_categories:
        sim, _ = label_similarity(name, category, observed, embeddings)
        if sim >= MIN_LABEL_SIM:
            return True, observed
    return False, None


def build_summary(scene, script_path, comparison, actions, categories, gt_log_path=None,
                  ever_observed_categories=None, embeddings=None):
    """Il riepilogo live: quanti su quanti, e per ognuno A QUALE oggetto
    del grafo e' stato associato (o perche' no) -- stessa struttura di
    live_comparison.json (watch_gt_vs_graph.py), cosi' i due file si
    leggono allo stesso modo anche se calcolano il confronto in modo
    diverso (qui: stato finale via replay dello script; li': eventi nel
    tempo). "spawn" conta gli oggetti spawnati che il grafo ha
    effettivamente ritrovato nello stato finale. Gli oggetti poi rimossi
    dallo script non devono esserci alla fine, quindi non entrano in
    questo conteggio: per loro conta "remove", soddisfatto quando
    l'oggetto NON e' nel grafo finale.
    """
    found_by_name = {f["gt_name"]: f for f in comparison["found"]}
    missing_by_name = {m["gt_name"]: m for m in comparison["missing"]}
    absent_ok = set(comparison["correctly_absent"])
    not_evaluable_by_name = {n["gt_name"]: n for n in comparison.get("not_evaluable", [])}
    removed_names = set(actions["remove"])

    # Un evento non ancora avvenuto nel simulatore (gt_progress qui sotto:
    # "spawn 2/10" -> solo i primi 2 nomi di actions["spawn"] sono avvenuti)
    # non e' "mancante": e' semplicemente non ancora successo. Prima di
    # questo controllo, un oggetto non ancora spawnato veniva comunque
    # cercato nel grafo finale e riportato con un motivo tecnico dettagliato
    # ("14 candidati ma label incompatibile") -- fuorviante durante una run
    # live, perche' sembra un fallimento della pipeline mentre e' solo un
    # evento futuro. Qui si sa SOLO se e' avvenuto o no, non se e' stato
    # ritrovato: quel giudizio resta a found_by_name/missing_by_name sotto.
    gtp = gt_progress_so_far(gt_log_path, actions)
    happened = {
        "spawn": set(gtp["spawn_avvenuti"]),
        "move": set(gtp["move_avvenuti"]),
        "remove": set(gtp["remove_avvenuti"]),
    }

    # Oggetti che devono esistere alla fine: spawnati e mai rimossi.
    expected_present_names = [n for n in actions["spawn"] if n not in removed_names]
    # Un "move" e' verificabile solo se l'oggetto sopravvive fino alla fine:
    # se e' stato spostato e poi rimosso, la sua posizione finale non esiste.
    moved_checkable = [n for n in actions["move"] if n not in removed_names]

    def block(label, names, total_override=None):
        eventi = []
        trovati = 0
        trovati_label = []
        valutabili = 0
        non_avvenuti = []
        for n in names:
            # L'ordine conta: "avvenuto?" va controllato PRIMA di "trovato?".
            # Il matching a stato-finale cerca per categoria+posizione senza
            # sapere quando l'evento accade, quindi puo' abbinare un oggetto
            # non ancora spawnato a un oggetto della scena che gli somiglia
            # (visto: apple_pie_board dichiarato trovato come "wood block"
            # mentre gt_progress diceva che erano avvenuti solo banana e
            # sugar -- il "wood block" del grafo era arredamento
            # preesistente, non l'oggetto dello script). Un evento non
            # ancora avvenuto non puo' essere stato trovato, mai.
            if n not in happened[label]:
                # Non ancora avvenuto: nessun motivo tecnico, non c'e'
                # ancora nulla da spiegare. Non entra nel denominatore (vedi
                # valutabili): un evento che il simulatore non ha mai emesso
                # non e' un fallimento della pipeline, e contarlo come tale
                # renderebbe "0/10" indistinguibile da un vero 0 su 10.
                non_avvenuti.append(n)
                eventi.append({"nome": n, "label_attesa": categories.get(n, n),
                              "posizione_attesa": None, "matched_graph_object": None,
                              "non_ancora_avvenuto": True,
                              "motivo": f"non ancora eseguito dal simulatore (vedi gt_progress.{label})"})
            elif n in found_by_name:
                valutabili += 1
                f = found_by_name[n]
                eventi.append(_matched_event(f))
                trovati += 1
                # "spawn": "1/10" da solo non dice QUALE dei 10 -- bisognava
                # aprire spawn_eventi ed eventualmente scorrerlo tutto per
                # scoprirlo. Qui accanto al contatore c'e' subito nome e
                # label di ognuno dei trovati.
                trovati_label.append(f"{n} ({f['gt_category']})")
            elif n in missing_by_name:
                valutabili += 1
                eventi.append(_missing_event(missing_by_name[n], categories))
            else:
                # move su un oggetto poi rimosso, o altro caso non
                # verificabile: nessun dato di matching disponibile.
                eventi.append({"nome": n, "label_attesa": categories.get(n, n),
                              "posizione_attesa": None, "matched_graph_object": None,
                              "motivo": "non verificabile (vedi denominatore)"})
        # Denominatore = eventi REALMENTE avvenuti nel simulatore e quindi
        # verificabili, non il totale pianificato. Stesso principio gia'
        # applicato ai remove (n_absent_evaluable in
        # compare_final_state.build_final_comparison): un evento mai emesso
        # non puo' contare come fallimento della pipeline. Il totale
        # pianificato resta leggibile in gt_progress e in
        # <label>_pianificati_totale.
        total = total_override if total_override is not None else len(names)
        return {label: f"{trovati}/{valutabili}", f"{label}_trovati": trovati_label,
                f"{label}_valutabili": valutabili,
                f"{label}_pianificati_totale": total,
                f"{label}_non_ancora_avvenuti": non_avvenuti,
                f"{label}_eventi": eventi}

    progress = {}
    progress.update(block("spawn", expected_present_names, total_override=len(actions["spawn"])))
    # Numeratore/denominatore degli spawn di oggetti poi rimossi, sommati
    # sotto a quelli calcolati da block(): sono spawn a tutti gli effetti e
    # vanno nello stesso contatore, ma il loro esito si verifica in modo
    # diverso (sull'intero stream invece che sullo stato finale), quindi si
    # accumulano qui e non dentro block().
    # spawn_confermati parte dai trovati nello stato finale (block li ha gia'
    # contati nel numeratore di progress["spawn"]); il loop sotto vi somma
    # gli spawn poi-rimossi confermati sullo stream.
    progress["spawn_confermati"] = int(progress["spawn"].split("/")[0])
    # Gli spawn di oggetti POI RIMOSSI vanno verificati come tutti gli altri.
    # Prima venivano saltati ("non conta come spawn atteso a fine scena")
    # perche' il confronto guardava solo lo stato finale, dove quegli oggetti
    # giustamente non ci sono. Ma lo spawn E' un evento realmente avvenuto:
    # la banana compare a frame 11 e resta visibile per un pezzo di run, e la
    # pipeline deve averla mappata. Saltarlo significa non testare mai uno
    # spawn vero -- nel caso di questa run, l'UNICO spawn che il simulatore
    # abbia davvero eseguito.
    #
    # Ogni oggetto spawnato-e-poi-rimosso conta quindi DUE volte, con due
    # domande diverse: qui "e' apparso ed e' stato mappato?", e in
    # remove_eventi "e' poi sparito?". Sono due proprieta' indipendenti: la
    # pipeline puo' azzeccarne una e sbagliare l'altra.
    for n in actions["spawn"]:
        if n not in removed_names:
            continue
        label = categories.get(n, n)
        if n not in happened["spawn"]:
            progress["spawn_eventi"].append({
                "nome": n, "label_attesa": label,
                "posizione_attesa": None, "matched_graph_object": None,
                "poi_rimosso": True, "non_ancora_avvenuto": True,
                "motivo": "non ancora eseguito dal simulatore (vedi gt_progress.spawn)",
            })
            continue
        seen, as_category = spawn_seen_in_stream(
            n, label, ever_observed_categories or set(), embeddings or {})
        progress["spawn_valutabili"] += 1
        if seen:
            progress["spawn_confermati"] += 1
            # Senza questa riga un oggetto poi-rimosso confermato (es.
            # baker_b@gmail_sugar) alza il numeratore di "spawn" ma il suo
            # nome non compare in spawn_trovati -- lettura confusa: "4/7"
            # con solo 2 nomi elencati, gli altri 2 visibili solo aprendo
            # spawn_eventi per intero.
            progress["spawn_trovati"].append(f"{n} ({label})")
        progress["spawn_eventi"].append({
            "nome": n, "label_attesa": label,
            "posizione_attesa": None, "matched_graph_object": None,
            "poi_rimosso": True, "spawn_confermato": seen,
            "osservato_come": as_category,
            "verificato_per": "sola label sull'intero stream (senza posizione)",
            "motivo": (
                f"spawnato e poi rimosso dal piano: lo spawn e' stato verificato "
                f"sull'intero stream, non sullo stato finale. La pipeline ha "
                f"osservato «{as_category}» in almeno un frame."
                if seen else
                "spawnato e poi rimosso dal piano: la pipeline non ha mai osservato "
                "questa categoria in nessun frame dello stream, quindi lo spawn non "
                "risulta confermato. L'esito della rimozione e' in remove_eventi."
            ),
        })
    # Contatore spawn ricomposto: include ora anche gli spawn di oggetti poi
    # rimossi, verificati sopra sull'intero stream.
    progress["spawn"] = f"{progress['spawn_confermati']}/{progress['spawn_valutabili']}"
    progress["spawn_pianificati_totale"] = len(actions["spawn"])

    progress.update(block("move", moved_checkable, total_override=len(actions["move"])))

    # Remove ha una forma diversa: l'obiettivo e' l'ASSENZA, non un
    # matched_graph_object. Se fallisce (false_positives), pero', va detto
    # con quale idx l'oggetto e' ancora presente -- stessa importanza di
    # sapere "a chi" per uno spawn trovato.
    fp_by_name = {fp["gt_name"]: fp for fp in comparison["false_positives"]}
    remove_eventi = []
    remove_trovati = 0
    remove_valutabili = 0
    for n in actions["remove"]:
        if n not in happened["remove"]:
            # Non ancora rimosso dal simulatore: non e' ne' un successo ne'
            # un fallimento, e' un evento futuro. Stesso spirito di
            # not_evaluable (non entra nel numeratore/denominatore), ma con
            # un motivo che dice la cosa vera invece di quello tecnico
            # ("mai osservato nello stream") che si applicherebbe comunque
            # ma suggerirebbe un problema di detection invece che il fatto
            # ovvio che l'evento non e' ancora successo.
            remove_eventi.append({
                "nome": n, "label_attesa": categories.get(n, n),
                "assente_dal_grafo": None, "ancora_presente_come": [],
                "non_valutabile": True, "non_ancora_avvenuto": True,
                "motivo": "non ancora eseguito dal simulatore (vedi gt_progress.remove)",
            })
            continue
        if n in not_evaluable_by_name:
            # Mai osservato in nessun frame dello stream: un "assente dal
            # grafo" qui non significa "rimosso", significa "mai visto".
            # Non entra ne' al numeratore ne' al denominatore di
            # progress["remove"] (vedi n_absent_evaluable in
            # compare_final_state.build_final_comparison).
            remove_eventi.append({
                "nome": n, "label_attesa": categories.get(n, n),
                "assente_dal_grafo": None, "ancora_presente_come": [],
                "non_valutabile": True,
                "motivo": not_evaluable_by_name[n]["motivo"],
            })
            continue
        remove_valutabili += 1
        if n in absent_ok:
            remove_eventi.append({"nome": n, "label_attesa": categories.get(n, n),
                                  "assente_dal_grafo": True, "ancora_presente_come": []})
            remove_trovati += 1
        else:
            fp = fp_by_name.get(n)
            still = fp["still_in_graph"] if fp else []
            remove_eventi.append({
                "nome": n, "label_attesa": categories.get(n, n),
                "assente_dal_grafo": False,
                "ancora_presente_come": [
                    {"idx": h["idx"], "label": h["graph_category"],
                     "somiglianza_label": round(h["label_similarity"], 3)}
                    for h in still
                ],
            })
    progress["remove"] = f"{remove_trovati}/{remove_valutabili}"
    progress["remove_eventi"] = remove_eventi

    meta = comparison["meta"]
    check = comparison["transform_check"]
    embeddings_broken = meta.get("graph_categories_missing", False)
    nota = (
        "progress.spawn/move/remove = eventi confermati / eventi VERIFICABILI, "
        "cioe' quelli che il simulatore ha davvero eseguito finora (vedi "
        "gt_progress). Gli eventi non ancora avvenuti non entrano nel "
        "denominatore -- contarli come fallimenti renderebbe '0/10' "
        "indistinguibile da un vero zero su dieci; il totale pianificato resta "
        "in <azione>_pianificati_totale e i nomi in "
        "<azione>_non_ancora_avvenuti. Ogni evento ha 'matched_graph_object' "
        "con idx, label e distanza dell'oggetto del grafo a cui e' stato "
        "associato (null se non trovato, col motivo). Gli oggetti spawnati e "
        "POI RIMOSSI dal piano contano DUE volte, con due domande diverse: lo "
        "spawn qui ('e' apparso ed e' stato mappato?', marcato poi_rimosso con "
        "spawn_confermato) e la rimozione in remove_eventi ('e' poi sparito?'). "
        "Il loro spawn si verifica sull'INTERO stream e per sola label, non "
        "sullo stato finale dove giustamente non compaiono: e' una conferma "
        "piu' debole, perche' una categoria gia' presente nell'arredamento "
        "puo' confermare uno spawn mai visto davvero."
    )
    if embeddings_broken:
        # Visto in produzione (00824_live_0, 2026-09-12): senza questo
        # avviso i numeri sotto sembrano un fallimento del metodo, mentre
        # e' solo un file di embedding non ricalcolato dopo che il grafo
        # esisteva. Va in cima, non in coda, perche' e' la prima cosa da
        # controllare prima di fidarsi di spawn/move/remove qui sopra.
        nota = (
            "ATTENZIONE: gt_text_embeddings.json non ha 'graph_categories' "
            "valorizzato pur essendoci oggetti nel grafo -> ogni similarita' "
            "di label e' tornata 0.00 e i match sono stati scartati quasi "
            "tutti. Rilanciare compute_gt_text_embeddings.py con "
            "--graph-stream prima di leggere i numeri sotto come un giudizio "
            "sul metodo. " + nota
        )
    return {
        "scena": scene,
        # Quanti eventi il SIMULATORE ha davvero gia' emesso finora, letti
        # da gt_events.jsonl -- non il confronto con la pipeline (quello e'
        # "progress" sotto). Se qui spawn e' 4/10 e in progress.spawn c'e'
        # 1/10, il simulatore e' avanti alla pipeline: il disallineamento
        # da monitorare mentre la run gira, che "progress" da solo non
        # mostra perche' confronta contro lo stato FINALE atteso. Gia'
        # calcolato sopra come gtp, per sapere quali nomi sono "avvenuti".
        "gt_progress": gtp,
        "progress": progress,
        "totale_oggetti_nel_grafo": meta["graph_objects_total"],
        "meta": {
            "ultimo_frame_elaborato": meta["last_frame_in_graph"],
            "trasformata": check["status"],
            "residuo_mediano_m": check.get("median_residual_m"),
            "soglia_distanza_m": meta["max_dist_m"],
            "graph_categories_missing": embeddings_broken,
        },
        "nota": nota,
    }


def write_log(path, scene, script_path, comparison, actions, categories, summary):
    """Il log diagnostico: tutto quello che serve per capire cosa e' successo."""
    c = comparison
    meta = c["meta"]
    check = c["transform_check"]
    L = []
    w = L.append

    w("=" * 78)
    w(f"SCENA {scene} — diagnostica della run")
    w(f"generato: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 78)
    w("")
    gtp = summary.get("gt_progress", {})
    w("QUANTO E' AVANZATO IL SIMULATORE (fatti, da gt_events.jsonl)")
    w(f"  spawn   {gtp.get('spawn', 'n/d')}   avvenuti: {gtp.get('spawn_avvenuti', [])}")
    w(f"  move    {gtp.get('move', 'n/d')}   avvenuti: {gtp.get('move_avvenuti', [])}")
    w(f"  remove  {gtp.get('remove', 'n/d')}   avvenuti: {gtp.get('remove_avvenuti', [])}")
    w("")
    w("COM'E' ANDATA (confronto pipeline vs stato finale atteso)")
    w(f"  spawn   {summary['progress']['spawn']}")
    w(f"  move    {summary['progress']['move']}")
    w(f"  remove  {summary['progress']['remove']}")
    w("")
    w("DA DOVE VENGONO QUESTI NUMERI")
    w("  Lo script pianificato viene rigiocato dall'inizio alla fine (spawn")
    w("  inserisce, move sposta, remove cancella) per sapere come DEVE essere")
    w("  la scena alla fine. Quello stato si confronta con l'ultimo frame del")
    w("  grafo costruito dalla pipeline.")
    w("")
    w(f"  script:        {script_path}")
    w(f"  grafo:         {meta['graph_stream_path']}")
    w(f"  ultimo frame:  {meta['last_frame_in_graph']}")
    w(f"  oggetti nel grafo finale: {meta['graph_objects_total']}")
    w("")

    w("-" * 78)
    w("TRASFORMATA DI COORDINATE (sanity check)")
    w("-" * 78)
    w("  La ground truth viene da un simulatore: e' esatta per costruzione.")
    w("  Quindi un oggetto ben ricostruito deve cadere a POCHI CENTIMETRI dalla")
    w("  sua posizione vera. Un residuo grande su TUTTI gli oggetti non e'")
    w("  imprecisione della pipeline: e' un difetto nella conversione fra i")
    w("  sistemi di coordinate.")
    w("")
    w(f"  esito: {check['status'].upper()} — {check['detail']}")
    if check.get("median_residual_m") is not None:
        w(f"  residuo mediano {check['median_residual_m']} m "
          f"(min {check['min_residual_m']}, max {check['max_residual_m']}, "
          f"su {check['samples']} oggetti)")
    w(f"  offset calibrato da frame0_anchor: {meta['habitat_offset_available']}")
    w(f"  embedding testuali CLIP disponibili: {meta['text_embeddings_available']}")
    w("")

    w("-" * 78)
    w("OGGETTI RITROVATI")
    w("-" * 78)
    if not c["found"]:
        w("  nessuno.")
    for f in c["found"]:
        w(f"  {f['gt_name']}  ({f['gt_category']})")
        w(f"      trovato come: idx {f['idx']} «{f['graph_category']}»")
        w(f"      distanza dalla posizione vera: {f['distance_m']:.3f} m"
          if f.get("distance_m") is not None else "      distanza: non misurabile")
        w(f"      somiglianza della label: {f['label_similarity']:.3f} ({f['label_method']})")
        w(f"      atteso in: {[round(x, 2) for x in f['gt_position']]}")
    w("")

    w("-" * 78)
    w("OGGETTI NON RITROVATI — e perche'")
    w("-" * 78)
    if not c["missing"]:
        w("  nessuno.")
    for m in c["missing"]:
        w(f"  {m['gt_name']}  ({m['gt_category']})")
        w(f"      atteso in: {[round(x, 2) for x in m['gt_position']]}")
        w(f"      motivo: {m['reason']}")
        if m.get("candidates_in_range"):
            near = m.get("nearest_candidate_m")
            w(f"      candidati entro {meta['max_dist_m']} m: {m['candidates_in_range']}"
              + (f", il piu' vicino a {near:.2f} m" if near is not None else ""))
            sim = m.get("best_label_similarity")
            if sim is not None:
                w(f"      migliore somiglianza di label fra quelli: {sim:.3f} "
                  f"(serve >= {MIN_LABEL_SIM})")
    w("")

    w("-" * 78)
    w("RIMOZIONI")
    w("-" * 78)
    w("  Un remove e' riuscito quando l'oggetto NON e' piu' nel grafo finale --")
    w("  ma SOLO se la pipeline lo aveva prima osservato: un oggetto mai visto")
    w("  in nessun frame non e' un remove riuscito, e' un remove non valutabile")
    w("  (non conta ne' come OK ne' come NO, vedi sotto).")
    w("")
    for name in c["correctly_absent"]:
        w(f"  OK   {name} — rimosso dallo script e assente dal grafo")
    for fp in c["false_positives"]:
        w(f"  NO   {fp['gt_name']} — rimosso dallo script ma ANCORA nel grafo:")
        for h in fp["still_in_graph"][:6]:
            w(f"           idx {h['idx']} «{h['graph_category']}» "
              f"(somiglianza {h['label_similarity']:.3f})")
    for ne in c.get("not_evaluable", []):
        w(f"  ?    {ne['gt_name']} — non valutabile: {ne['motivo']}")
    w("")

    w("-" * 78)
    w("OGGETTI DEL GRAFO NON ASSOCIATI A NULLA")
    w("-" * 78)
    w("  Sono oggetti che la pipeline ha mappato ma che non corrispondono a")
    w("  nessun oggetto dello script: arredamento gia' presente nella scena,")
    w("  oppure frammenti dello stesso oggetto visto piu' volte.")
    w("")
    unassigned = c["graph_objects_unassigned"]
    w(f"  totale: {len(unassigned)}")
    by_cat = {}
    for o in unassigned:
        by_cat.setdefault(o["category"], []).append(o["idx"])
    for cat, idxs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        marker = "   <-- stesso oggetto diviso in piu' nodi?" if len(idxs) > 2 else ""
        w(f"  {len(idxs):>3}x {cat:<24} idx {idxs[:12]}{marker}")
    w("")
    w("=" * 78)

    Path(path).write_text("\n".join(L) + "\n")


def build_map_snapshot(scene, graph_stream_path, gt_log_path, comparison, max_dist_m):
    """Tutti gli oggetti della mappa, non solo quelli associati a un GT.

    E' il secondo file richiesto: mentre 'associazioni' in summary guarda
    dal lato GT ("quale oggetto della mappa ho trovato per la banana?"),
    questo guarda dal lato mappa ("cosa c'e' nel grafo, punto e basta,
    posizione compresa") -- utile per vedere anche gli oggetti che NON
    corrispondono a nessun evento dello script (arredamento preesistente,
    o duplicati dello stesso oggetto fisico visto da piu' angoli).

    Rilegge lo stesso ultimo frame di graph_stream.jsonl che ha gia' letto
    build_final_comparison: non e' un dato diverso, e' lo stesso stato con
    tutti gli oggetti invece dei soli associati.
    """
    offset = frame0_offset_map_to_habitat(load_jsonl(gt_log_path)) if gt_log_path else None
    last_frame, graph_objects = load_final_graph_state(graph_stream_path, offset)

    # idx -> con quale GT e' stato associato, se lo e' stato (stesso
    # risultato del matching gia' fatto in build_final_comparison, non
    # un secondo calcolo: si guarda solo comparison["found"]).
    matched_to = {f["idx"]: f["gt_name"] for f in comparison["found"]}

    objects = []
    for o in graph_objects:
        objects.append({
            "idx": o["idx"],
            "label": o["category"],
            "posizione": [round(x, 3) for x in o["centroid_habitat"]] if o.get("centroid_habitat") else None,
            "osservazioni": o.get("detections"),
            "associato_a_gt": matched_to.get(o["idx"]),
        })

    return {
        "scena": scene,
        "ultimo_frame_elaborato": last_frame,
        "totale_oggetti": len(objects),
        "oggetti_associati_a_un_gt": sum(1 for o in objects if o["associato_a_gt"]),
        "soglia_distanza_usata_per_associare_m": max_dist_m,
        "nota": (
            "Ogni oggetto della mappa 3D, indipendentemente dal fatto che "
            "corrisponda o meno a un evento dello script. 'posizione' e' il "
            "centroide delle sue gaussiane, gia' convertito nel frame Habitat "
            "(stesso sistema di coordinate della ground truth) -- None se "
            "l'oggetto non ha piu' gaussiane assegnate. 'associato_a_gt' e' "
            "valorizzato solo per gli oggetti che il confronto ha effettivamente "
            "riconosciuto come uno degli spawn/move pianificati."
        ),
        "oggetti": objects,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, type=Path)
    ap.add_argument("--graph-stream", required=True, type=Path)
    ap.add_argument("--gt-log", type=Path, default=None)
    ap.add_argument("--text-embeddings", type=Path, default=None)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-dist", type=float, default=MAX_DIST_M)
    args = ap.parse_args()

    comparison = build_final_comparison(args.script, args.graph_stream, args.gt_log,
                                        args.text_embeddings, args.max_dist)
    actions, categories = planned_actions(args.script)
    # Categorie viste in QUALSIASI frame (non solo l'ultimo): servono a
    # verificare gli spawn di oggetti poi rimossi, che nello stato finale
    # giustamente non compaiono. Vedi spawn_seen_in_stream.
    ever_observed = load_ever_observed_categories(args.graph_stream)
    embeddings = (json.loads(args.text_embeddings.read_text())
                  if args.text_embeddings and args.text_embeddings.exists() else {})
    summary = build_summary(args.scene, args.script, comparison, actions, categories,
                            gt_log_path=args.gt_log,
                            ever_observed_categories=ever_observed,
                            embeddings=embeddings)
    map_snapshot = build_map_snapshot(args.scene, args.graph_stream, args.gt_log,
                                      comparison, args.max_dist)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.scene}_summary.json"
    log_path = args.out_dir / f"{args.scene}_log.txt"
    map_path = args.out_dir / f"{args.scene}_mappa.json"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_log(log_path, args.scene, args.script, comparison, actions, categories, summary)
    map_path.write_text(json.dumps(map_snapshot, indent=2, ensure_ascii=False))

    print(f"scena {args.scene}")
    print(f"  spawn   {summary['progress']['spawn']}")
    print(f"  move    {summary['progress']['move']}")
    print(f"  remove  {summary['progress']['remove']}")
    print(f"  totale oggetti nel grafo: {summary['totale_oggetti_nel_grafo']}")
    print(f"  riepilogo -> {summary_path}")
    print(f"  log       -> {log_path}")
    print(f"  mappa     -> {map_path}  ({map_snapshot['totale_oggetti']} oggetti)")


if __name__ == "__main__":
    main()
