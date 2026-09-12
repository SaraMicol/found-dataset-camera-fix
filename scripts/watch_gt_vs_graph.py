#!/usr/bin/env python3
"""Riscrive periodicamente un file di confronto live tra la ground truth
(eventi spawn/move/remove REALMENTE eseguiti dallo script di cammino,
FOUND-Dataset/record_dynamic_sequence_live.py --gt-log ...) e il grafo
costruito dalla pipeline dgsg (graph_stream.jsonl, scritto da
log_graph_state in scripts/dynamic_gsg_real_ssim.py).

Entrambi i file usano lo stesso frame_idx come chiave (il commit al bridge
e il time_idx della pipeline coincidono per costruzione, vedi ros_live.py),
quindi il confronto è diretto: per ogni evento GT a un dato frame si guarda
se in quello stesso frame (o entro una piccola finestra) il grafo mostra
un added/removed coerente.

Uso:
    python3 scripts/watch_gt_vs_graph.py \
        --script /path/to/household_experiments_scene_824.json \
        --gt-log /tmp/gt_events.jsonl \
        --graph-stream experiments/FOUND/00824_live_0/graph_stream.jsonl \
        --out experiments/FOUND/00824_live_0/live_comparison.json \
        --interval 3
"""

import argparse
import json
import time
from pathlib import Path


def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_planned_totals(script_path: Path):
    """Conta quanti spawn/move/remove sono pianificati in TOTALE nello
    script della traiettoria (non quanti eseguiti finora) -- il
    denominatore fisso per il progress tipo "3/10"."""
    if script_path is None or not script_path.exists():
        return {"spawn": None, "move": None, "remove": None}
    data = json.loads(script_path.read_text())
    totals = {"spawn": 0, "move": 0, "remove": 0}
    for step in data.get("steps", []):
        action = step.get("action")
        if action in totals:
            totals[action] += 1
    return totals


def load_planned_names(script_path: Path):
    """Nomi (in ordine) degli oggetti pianificati per ogni azione, per
    mostrare le label accanto al conteggio x/y."""
    if script_path is None or not script_path.exists():
        return {"spawn": [], "move": [], "remove": []}
    data = json.loads(script_path.read_text())
    names = {"spawn": [], "move": [], "remove": []}
    for step in data.get("steps", []):
        action = step.get("action")
        if action in names:
            names[action].append(step.get("name") or step.get("object"))
    return names


def load_name_to_template_category(script_path: Path):
    """Mappa name -> categoria plausibile ESTRATTA DAL TEMPLATE (non dal
    nome): il "name" (es. "zest_tightener", "apple_pie_board") e' una
    stringa creativa assegnata dall'LLM che ha generato lo script e NON
    corrisponde in 5 casi su 10 alla categoria fisica reale dell'oggetto
    (verificato: zest_tightener -> template "051_large_clamp", niente a
    che vedere con "tightener"). Il "template" (es. "051_large_clamp",
    presente solo negli step spawn) e' invece sempre coerente con
    l'oggetto fisico -- e' quello che va confrontato con la category
    rilevata dalla pipeline, non il name. move/remove non hanno il
    template nel loro step: lo recuperano dallo spawn con lo stesso name."""
    if script_path is None or not script_path.exists():
        return {}
    data = json.loads(script_path.read_text())
    mapping = {}
    for step in data.get("steps", []):
        if step.get("action") == "spawn" and step.get("template"):
            name = step.get("name")
            # "051_large_clamp" -> "large clamp" (rimuove il prefisso numerico)
            template = step["template"]
            parts = template.split("_")
            if parts and parts[0].isdigit():
                parts = parts[1:]
            mapping[name] = " ".join(parts)
    return mapping


def name_matches_category(gt_name: str, category: str, template_category: str = None) -> bool:
    """Confronto testuale tra l'oggetto ground truth e la categoria
    generica rilevata dalla pipeline (es. "banana"). Usa PRIMA la
    categoria derivata dal template (sempre coerente con l'oggetto
    fisico, es. "large clamp" per zest_tightener), e solo se non
    disponibile ripiega sul name grezzo (utile solo per i casi in cui
    name e template coincidono per caso, es. golden_twitch_banana).
    Euristica su stringhe, non un id univoco -- in caso di piu' oggetti
    della stessa categoria nel campo visivo resta ambiguo."""
    if not category:
        return False
    cat_words = set(category.lower().split())
    if template_category:
        template_words = set(template_category.lower().split())
        if cat_words.issubset(template_words) or (template_words & cat_words):
            return True
    if not gt_name:
        return False
    name_words = set(gt_name.lower().replace("@", "_").split("_"))
    return cat_words.issubset(name_words) or bool(name_words & cat_words)


def centroid_distance(pos_a, pos_b):
    """Distanza euclidea tra due posizioni [x,y,z], o None se una manca."""
    if pos_a is None or pos_b is None:
        return None
    return sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)) ** 0.5


# Rotazione FISSA "map" (ROS, frame in cui la pipeline logga i centroidi via
# first_frame_c2w, vedi log_graph_state) -> Habitat world (frame in cui e'
# espressa "position" nel gt-log). Derivata analiticamente componendo
# habitat_pose_to_ros (Habitat Y-up -> ROS Z-up, in
# habitat_camera_objects_node.py) con la rotazione statica camera_optical
# (REP-103), poi VERIFICATA empiricamente sulla run reale: la distanza
# relativa tra due centroidi osservati (banana, sugar box) ruotata con
# questa matrice combacia con la distanza relativa tra le stesse due
# "position" GT a meno di ~0.37m (drift di tracking normale). E' una
# permutazione pura degli assi, nessuna riflessione: map(x,y,z) ->
# habitat(z,x,y).
def map_to_habitat_rotation_only(p_map):
    x, y, z = p_map
    return (z, x, y)


def inv_habitat_pose_to_ros_position(p_ros):
    """Inversa ESATTA di habitat_pose_to_ros (solo posizione, vedi
    habitat_camera_objects_node.py: ros=[-hz,-hx,hy]): dato un punto
    espresso nel frame ROS in cui e' pubblicato base_link (quello che la
    funzione produce), restituisce lo stesso punto in Habitat world.
    Round-trip esatto verificato: inv(habitat_pose_to_ros(h)) == h per
    qualunque h. NON e' la stessa rotazione di map_to_habitat_rotation_only
    (quella e' calibrata sui centroidi delle gaussiane, che passano anche
    per la rotazione ottica camera->camera_optical; questa e' per
    base_link, senza quella rotazione aggiuntiva)."""
    rx, ry, rz = p_ros
    return (-ry, rz, -rx)


def frame0_offset_map_to_habitat(gt_events):
    """Offset costante che manca alla sola rotazione
    (map_to_habitat_rotation_only) per portare un centroide da "map" a
    Habitat world: le due origini NON coincidono (verificato: la distanza
    RELATIVA tra due centroidi ruotati combaciava con la GT a meno di
    ~0.37m di drift, quella assoluta no -- mancava l'offset tra le
    origini).

    offset = inv_habitat_pose_to_ros(base_link_in_map@frame0)
             - map_to_habitat_rotation_only(base_link_in_map@frame0)

    "base_link_in_map@frame0" e' l'evento 'frame0_anchor' loggato da
    record_dynamic_sequence_live.py::_log_frame0_anchor (posizione
    dell'agente Habitat, letta da TF map->base_link, appena arriva il
    primo frame -- prima che il camminatore lo muova, quindi coincide con
    l'istante in cui la pipeline fissa first_frame_c2w). Lo stesso punto
    fisico (l'agente al frame 0), letto con le DUE rotazioni diverse
    (quella esatta per base_link, e quella calibrata sui centroidi delle
    gaussiane) da' due risultati diversi -- la loro differenza E' l'offset
    tra le origini dei due mondi, senza bisogno di nessun altro dato GT.

    Ritorna None se il gt-log non ha ancora l'evento 'frame0_anchor' (run
    vecchie, prima di questa modifica, o camminatore non ancora arrivato
    a quel punto)."""
    anchor = None
    for ev in gt_events:
        if ev.get("action") == "frame0_anchor":
            anchor = ev["base_link_in_map"]
            break
    if anchor is None:
        return None
    # CORREZIONE (2026-09-11). La versione precedente calcolava
    #     inv_habitat_pose_to_ros_position(anchor) - map_to_habitat_rotation_only(anchor)
    # cioe' la DIFFERENZA tra due rotazioni diverse applicate allo stesso
    # punto fisico. Non e' un offset tra origini: e' la distanza tra due
    # modi di ruotare lo stesso vettore, una quantita' senza significato
    # geometrico qui. Produceva un errore di ~3.8m su X e Z (la Y restava
    # quasi giusta perche' entrambe le rotazioni mandano l'altezza sullo
    # stesso asse, mascherando il problema).
    #
    # Derivazione corretta: i centroidi loggati dalla pipeline vivono nel
    # frame "map", la cui origine E' la posizione dell'agente al frame 0
    # (relative_pose=True: il mondo del pipeline e' ancorato alla camera
    # del primo frame). Per portarli in Habitat world basta ruotarli e poi
    # sottrarre quella stessa origine, ruotata con la STESSA rotazione:
    #     p_habitat = rot(p_map) - rot(anchor)
    # quindi offset = -rot(anchor).
    #
    # Verificato su tre oggetti con posizione GT nota (banana e bowl,
    # presenti in copia unica nel grafo, cadono a 1.2cm dalla GT; vedi
    # anche la stima indipendente ricavata dai dati, che coincide con
    # questa formula a 1.8cm).
    rotated_anchor = map_to_habitat_rotation_only(anchor)
    return tuple(-r for r in rotated_anchor)


def map_centroid_to_habitat(p_map, offset):
    """Applica rotazione fissa + offset per portare un centroide dal frame
    "map" (come loggato in graph_stream.jsonl) al world Habitat (come in
    "position" nel gt-log). offset=None -> nessuna conversione possibile
    (frame0_anchor non disponibile, run vecchia): ritorna None."""
    if p_map is None or offset is None:
        return None
    rotated = map_to_habitat_rotation_only(p_map)
    return tuple(r + o for r, o in zip(rotated, offset))


def build_comparison(gt_log_path: Path, graph_stream_path: Path, planned_totals: dict,
                      planned_names: dict, name_to_template_category: dict):
    gt_events_raw = load_jsonl(gt_log_path)
    # Offset (map -> Habitat world) calibrato una volta per run
    # dall'evento 'frame0_anchor' -- vedi frame0_offset_map_to_habitat.
    # None se il gt-log non ha ancora quell'evento: in quel caso i
    # centroidi non vengono convertiti (restano non comparabili, vedi
    # map_centroid_to_habitat) e il matching ricade sulla sola
    # finestra+categoria, com'era prima di questa modifica.
    habitat_offset = frame0_offset_map_to_habitat(gt_events_raw)
    # 'frame0_anchor' (scritto da record_dynamic_sequence_live.py per
    # calibrare l'offset sopra) NON e' un evento spawn/move/remove: va
    # escluso qui, altrimenti il resto della funzione (che assume
    # action in {spawn,move,remove} e un campo "name") crasha con
    # KeyError sul primo accesso a e["name"].
    gt_events = [ev for ev in gt_events_raw if ev.get("action") != "frame0_anchor"]

    ever_seen = {}
    observed_events = []
    last_frame = None
    for entry in load_jsonl(graph_stream_path):
        frame = entry["frame"]
        last_frame = frame
        curr_objs = {o["idx"]: o for o in entry["objects"]}
        for idx, o in curr_objs.items():
            if idx not in ever_seen:
                ever_seen[idx] = (frame, o["category"])
                observed_events.append({
                    "frame": frame, "action": "added", "idx": idx,
                    "category": o["category"],
                    "centroid": map_centroid_to_habitat(o.get("centroid"), habitat_offset),
                })
        for idx in entry.get("removed", []):
            o = curr_objs.get(idx)
            cat = (o["category"] if o else None) or ever_seen.get(idx, (None, "?"))[1]
            centroid = map_centroid_to_habitat(o.get("centroid") if o else None, habitat_offset)
            observed_events.append({
                "frame": frame, "action": "removed", "idx": idx,
                "category": cat, "centroid": centroid,
            })

    # Match per evento GT: due criteri, non uno solo.
    #
    # 1) FINESTRA ASIMMETRICA (solo in avanti): un oggetto spawnato/mosso a
    #    un frame GT puo' comparire nel grafo solo DOPO, mai prima -- la
    #    fusione nella mappa richiede che l'agente veda fisicamente
    #    l'oggetto, il che richiede tempo (visto: banana spawnata a frame
    #    11, fusa nel grafo a frame 18 -- 7 frame di ritardo, normale). Una
    #    finestra centrata (vecchio comportamento, +-3) accettava/rifiutava
    #    simmetricamente e non aveva margine per questo ritardo. Qui invece
    #    si guarda avanti fino a FORWARD_WINDOW frame.
    #
    # 2) POSIZIONE 3D quando disponibile (vedi "centroid" in
    #    log_graph_state, scripts/dynamic_gsg_real_ssim.py): la sola
    #    category (o l'embedding CLIP visuale, clip_ft) NON distingue due
    #    oggetti della stessa categoria fisica -- verificato: una banana
    #    preesistente in scena e la banana spawnata dallo script hanno
    #    cosine similarity sui clip_ft di ~1.0, quindi qualunque euristica
    #    solo-per-categoria o solo-per-embedding puo' "confermare" la
    #    banana SBAGLIATA (quella gia' in scena PRIMA dello spawn). Il
    #    centroide invece separa le due per posizione reale.
    #    Se il centroide non e' disponibile (run vecchie, prima di questa
    #    modifica, o oggetto senza gaussiane assegnate) si ricade sul solo
    #    controllo di categoria + finestra: comportamento precedente,
    #    diagnosticamente piu' debole ma non bloccante.
    #
    # Un oggetto della mappa (idx) puo' essere usato per confermare UN SOLO
    # evento GT: senza questo, 10 "move" della stessa banana matcherebbero
    # tutti lo stesso idx e il conteggio sarebbe falsamente alto.
    FORWARD_WINDOW = 40
    MAX_CENTROID_DIST_M = 1.0  # oltre 1m non e' plausibile che sia lo stesso oggetto fisico
    matched = []
    unmatched = []
    used_idx = set()
    for ev in gt_events:
        action = "added" if ev["action"] in ("spawn", "move") else "removed"
        frame = ev["frame"]
        gt_name = ev.get("name")
        gt_pos = ev.get("position")
        template_cat = name_to_template_category.get(gt_name)

        candidates = [
            o for o in observed_events
            if o["idx"] not in used_idx
            and o["action"] == action
            and 0 <= (o["frame"] - frame) <= FORWARD_WINDOW
            and name_matches_category(gt_name, o["category"], template_cat)
        ]

        # Se piu' di un candidato passa il filtro testuale (es. due banane),
        # la posizione decide -- e se il centroide manca su tutti i lati non
        # si puo' disambiguare: si tiene il piu' vicino nel tempo (comportamento
        # precedente) ma va segnalato come ambiguo.
        ambiguous = False
        if len(candidates) > 1 and gt_pos is not None:
            with_dist = [
                (c, centroid_distance(gt_pos, c.get("centroid")))
                for c in candidates
            ]
            with_dist = [(c, d) for c, d in with_dist if d is not None]
            if with_dist:
                with_dist.sort(key=lambda cd: cd[1])
                best_c, best_d = with_dist[0]
                if best_d <= MAX_CENTROID_DIST_M:
                    candidates = [best_c]
                else:
                    # nessun candidato e' vicino a sufficienza: nessun match onesto
                    candidates = []
            else:
                ambiguous = True  # nessun centroid disponibile per disambiguare
        elif len(candidates) == 1 and gt_pos is not None and candidates[0].get("centroid") is not None:
            d = centroid_distance(gt_pos, candidates[0]["centroid"])
            if d is not None and d > MAX_CENTROID_DIST_M:
                candidates = []  # candidato unico ma troppo lontano: probabile falso positivo per categoria

        entry = dict(ev)
        if candidates:
            entry["matched_graph_events"] = candidates
            entry["ambiguous"] = ambiguous
            matched.append(entry)
            used_idx.add(candidates[0]["idx"])
        else:
            entry["matched_graph_events"] = []
            unmatched.append(entry)

    gt_done = {
        "spawn": sum(1 for e in gt_events if e["action"] == "spawn"),
        "move": sum(1 for e in gt_events if e["action"] == "move"),
        "remove": sum(1 for e in gt_events if e["action"] == "remove"),
    }
    done_names = {
        "spawn": [e["name"] for e in gt_events if e["action"] == "spawn"],
        "move": [e["name"] for e in gt_events if e["action"] == "move"],
        "remove": [e["name"] for e in gt_events if e["action"] == "remove"],
    }


    def fmt(action):
        total = planned_totals.get(action)
        done = gt_done[action]
        return f"{done}/{total}" if total is not None else f"{done}/?"

    # Confermati nel grafo: tra gli eventi GT di quel tipo, quanti hanno
    # trovato un match (added/removed) nel grafo entro la finestra.
    matched_by_action = {"spawn": 0, "move": 0, "remove": 0}
    matched_names_by_action = {"spawn": [], "move": [], "remove": []}
    for e in matched:
        matched_by_action[e["action"]] += 1
        matched_names_by_action[e["action"]].append(e["name"])
    unmatched_names_by_action = {"spawn": [], "move": [], "remove": []}
    for e in unmatched:
        unmatched_names_by_action[e["action"]].append(e["name"])

    def fmt_confirmed(action):
        total = planned_totals.get(action)
        done = matched_by_action[action]
        return f"{done}/{total}" if total is not None else f"{done}/?"

    return {
        "progress": {
            "spawn": fmt("spawn"),
            "spawn_labels": done_names["spawn"],
            "move": fmt("move"),
            "move_labels": done_names["move"],
            "remove": fmt("remove"),
            "remove_labels": done_names["remove"],
            "confirmed_in_graph": {
                "spawn": fmt_confirmed("spawn"),
                "spawn_labels": matched_names_by_action["spawn"],
                "spawn_not_confirmed_labels": unmatched_names_by_action["spawn"],
                "move": fmt_confirmed("move"),
                "move_labels": matched_names_by_action["move"],
                "move_not_confirmed_labels": unmatched_names_by_action["move"],
                "remove": fmt_confirmed("remove"),
                "remove_labels": matched_names_by_action["remove"],
                "remove_not_confirmed_labels": unmatched_names_by_action["remove"],
            },
        },
        "meta": {
            "gt_log_path": str(gt_log_path),
            "graph_stream_path": str(graph_stream_path),
            "last_frame_in_graph": last_frame,
            "match_window_frames_forward": FORWARD_WINDOW,
            "max_centroid_dist_m": MAX_CENTROID_DIST_M,
            "generated_at_unix": time.time(),
            "note": (
                "progress.spawn/move/remove = eventi eseguiti dallo script di "
                "cammino / totale pianificato nello script. confirmed_in_graph = "
                "quanti di quegli eventi hanno un evento added/removed nel grafo "
                "entro FORWARD_WINDOW frame DOPO l'evento GT (mai prima: la "
                "fusione nella mappa arriva sempre dopo lo spawn/move reale) "
                "E category che corrisponde per parola alla categoria del "
                "TEMPLATE Habitat (es. zest_tightener -> template "
                "051_large_clamp -> confrontato con category \"clamp\", non con "
                "la parola \"tightener\" che non ha nulla a che vedere con "
                "l'oggetto fisico). Quando ci sono piu' candidati con la stessa "
                "categoria (es. due banane) si usa il centroide 3D dell'oggetto "
                "(vedi 'centroid' in log_graph_state) per scegliere quello "
                "davvero vicino alla position GT -- necessario perche' la sola "
                "category, e persino il clip_ft visuale, sono praticamente "
                "identici tra due oggetti della stessa categoria fisica "
                "(verificato: cosine ~1.0 tra una banana preesistente in scena "
                "e quella spawnata dallo script). Ogni oggetto della mappa "
                "(idx) conferma AL PIU' UN evento GT (vedi used_idx)."
            ),
        },
        "summary": {
            "graph_added_total": sum(1 for o in observed_events if o["action"] == "added"),
            "graph_removed_total": sum(1 for o in observed_events if o["action"] == "removed"),
        },
        "gt_events_matched": matched,
        "gt_events_unmatched": unmatched,
        "graph_observed_events": observed_events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=Path, default=None,
                    help="script trajectory (household_experiments_scene_824.json) da cui "
                         "leggere i totali pianificati spawn/move/remove per il progress x/y")
    ap.add_argument("--gt-log", required=True, type=Path)
    ap.add_argument("--graph-stream", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=3.0, help="secondi tra un refresh e l'altro")
    args = ap.parse_args()

    planned_totals = load_planned_totals(args.script)
    planned_names = load_planned_names(args.script)
    name_to_template_category = load_name_to_template_category(args.script)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scrivo confronto live in: {args.out}")
    print(f"  gt_log:       {args.gt_log}")
    print(f"  graph_stream: {args.graph_stream}")
    print(f"  totali pianificati: {planned_totals}")
    print(f"  refresh ogni {args.interval}s -- Ctrl+C per fermare")

    while True:
        try:
            comparison = build_comparison(args.gt_log, args.graph_stream, planned_totals,
                                           planned_names, name_to_template_category)
            # Riscrive SOLO le proprie sezioni, preservando quelle scritte da
            # altri (es. "final_state_comparison" di compare_final_state.py):
            # questo loop gira ogni pochi secondi, e un json.dump secco del
            # solo "comparison" cancellerebbe il lavoro dell'altro script al
            # primo refresh utile.
            payload = {}
            if args.out.exists():
                try:
                    payload = json.loads(args.out.read_text())
                except json.JSONDecodeError:
                    payload = {}
            payload.update(comparison)
            tmp = args.out.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(args.out)
        except Exception as exc:
            print(f"[watch_gt_vs_graph] refresh fallito (continuo): {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
