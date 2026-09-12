#!/usr/bin/env python3
"""Deduplica per post-processing gli oggetti del grafo che sono lo stesso
oggetto fisico spezzato in piu' nodi (es. 11 "pillow" per un solo cuscino).

Perche' qui e non nella pipeline: il matching cross-frame in
utils/map_objects_utils_up_with_groupv3.py (compute_similarities_and_merge)
decide se una detection e' un oggetto gia' visto SOLO renderizzando quell'
oggetto dal punto di vista della camera nel frame corrente (IoU 2D +
similarita' CLIP) -- se l'oggetto e' poco visibile/occluso in quel preciso
frame (sotto la soglia di 200 pixel in render_curr_frame_with_idx), la
detection non trova nulla con cui confrontarsi e diventa un nuovo oggetto,
anche se un oggetto identico esiste gia' altrove nel grafo. Non esiste un
fallback per categoria + posizione 3D globale in quel matching, e
aggiungerlo li' richiederebbe riordinare come objects/params si passano i
centroidi -- un cambiamento rischioso nell'algoritmo di matching in tempo
reale, mentre invece i centroidi 3D sono gia' calcolati e disponibili nello
stream (log_graph_state, graph_stream.jsonl).

Questo script legge l'ultimo frame di graph_stream.jsonl (o uno stream
intermedio, per essere usato mentre la run gira) e fonde in gruppi gli
oggetti con la STESSA category e centroide entro --max-dist metri --
esattamente il criterio "stessa etichetta, stesso posto" che un umano
userebbe per riconoscere che due nodi sono in realta' un solo oggetto.
Produce un nuovo grafo deduplicato, senza toccare graph_stream.jsonl.

Uso:
    python3 scripts/dedup_graph_objects.py \\
        --graph-stream experiments/FOUND/00824_live_0/graph_stream.jsonl \\
        --out experiments/FOUND/00824_live_0/graph_deduped.json \\
        --max-dist 0.3
"""

import argparse
import json
from pathlib import Path


def load_last_frame(graph_stream_path: Path):
    """Solo l'ultima riga: il file puo' superare i 100MB, non si carica
    tutto in memoria (stesso pattern di compare_final_state.load_final_graph_state)."""
    last = None
    with open(graph_stream_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    return json.loads(last)


def _dist(a, b):
    if a is None or b is None:
        return None
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _cosine(a, b):
    if not a or not b:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


# Sotto questa soglia due clip_ft della stessa category non bastano a
# dire "stesso oggetto fisico": due pillow diversi hanno gia' cosine
# alto per via della sola categoria condivisa (vedi il commento in
# log_graph_state: "due oggetti della stessa categoria fisica hanno
# clip_ft praticamente identico, ~1.0 -- da solo NON basta a
# distinguerli"). Qui pero' e' l'UNICO segnale quando il centroide manca
# (oggetto con una sola detection, mai ancora renderizzato con abbastanza
# gaussiane per calcolare il centroide in log_graph_state), quindi si
# tiene una soglia alta apposta: preferibile lasciare due nodi separati
# per un dubbio, piuttosto che fondere due oggetti fisici diversi solo
# perche' condividono la label.
MIN_CLIP_SIM_NO_CENTROID = 0.92


def dedup_objects(objects, max_dist_m, min_clip_sim=MIN_CLIP_SIM_NO_CENTROID):
    """Raggruppa per (category, centroide vicino) con un merge greedy union-find:
    se A e' vicino a B e B e' vicino a C, A/B/C finiscono nello stesso gruppo
    anche se A e C non sono direttamente entro max_dist_m -- lo stesso oggetto
    fisico osservato da piu' angolazioni forma spesso una catena di centroidi
    leggermente spostati, non un singolo cluster compatto.

    Quando il centroide manca (visto in produzione: un oggetto con una sola
    detection non ha ancora abbastanza gaussiane per calcolarlo, vedi
    _centroid_habitat in dynamic_gsg_real_ssim.py) il confronto ricade sulla
    similarita' CLIP visuale, con la soglia alta MIN_CLIP_SIM_NO_CENTROID.

    Ritorna (merged_objects, groups): merged_objects e' la lista deduplicata
    (un oggetto per gruppo, con "merged_from" = lista degli idx originali,
    centroide = media dei centroidi noti, detections = somma); groups e' la
    mappa idx_originale -> idx_del_gruppo, utile per chi deve rimappare
    riferimenti esterni (es. found/false_positives di compare_final_state.py).
    """
    parent = {o["idx"]: o["idx"] for o in objects}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    by_category = {}
    for o in objects:
        by_category.setdefault(o.get("category"), []).append(o)

    for cat, group in by_category.items():
        if cat is None or len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                d = _dist(a.get("centroid"), b.get("centroid"))
                if d is not None and d <= max_dist_m:
                    union(a["idx"], b["idx"])
                elif d is None:
                    # Almeno uno dei due non ha centroide: il criterio di
                    # posizione non e' applicabile, si ricade sul solo
                    # embedding visuale.
                    sim = _cosine(a.get("clip_ft"), b.get("clip_ft"))
                    if sim is not None and sim >= min_clip_sim:
                        union(a["idx"], b["idx"])

    clusters = {}
    for o in objects:
        root = find(o["idx"])
        clusters.setdefault(root, []).append(o)

    merged_objects = []
    idx_to_group = {}
    for root, members in clusters.items():
        centroids = [m["centroid"] for m in members if m.get("centroid") is not None]
        if centroids:
            n = len(centroids)
            centroid = [sum(c[k] for c in centroids) / n for k in range(3)]
        else:
            centroid = None
        # Il membro con piu' detection e' quello con la vista piu' affidabile:
        # la sua category/clip_ft (non solo il centroide medio) rappresenta
        # meglio il gruppo di uno scelto a caso o del primo incontrato.
        best = max(members, key=lambda m: m.get("detections") or 0)
        merged_objects.append({
            "idx": root,
            "category": best.get("category"),
            "detections": sum(m.get("detections") or 0 for m in members),
            "centroid": centroid,
            "merged_from": sorted(m["idx"] for m in members),
        })
        for m in members:
            idx_to_group[m["idx"]] = root

    merged_objects.sort(key=lambda o: o["idx"])
    return merged_objects, idx_to_group


def build_deduped_graph(graph_stream_path: Path, max_dist_m: float,
                        min_clip_sim: float = MIN_CLIP_SIM_NO_CENTROID):
    frame_entry = load_last_frame(graph_stream_path)
    if frame_entry is None:
        return {"error": f"{graph_stream_path} e' vuoto o non esiste"}

    objects = frame_entry.get("objects", [])
    merged_objects, idx_to_group = dedup_objects(objects, max_dist_m, min_clip_sim)

    n_before, n_after = len(objects), len(merged_objects)
    fragmented = [o for o in merged_objects if len(o["merged_from"]) > 1]

    return {
        "frame": frame_entry.get("frame"),
        "num_objects_originale": n_before,
        "num_objects_deduplicato": n_after,
        "oggetti_fusi": len(fragmented),
        "max_dist_m": max_dist_m,
        "min_clip_sim_senza_centroide": min_clip_sim,
        "idx_originale_to_gruppo": idx_to_group,
        "objects": merged_objects,
        "nota": (
            f"Deduplica per post-processing (vedi docstring dello script): "
            f"{n_before} nodi nel grafo grezzo -> {n_after} dopo aver fuso "
            f"quelli con stessa category e (centroide entro {max_dist_m} m, "
            f"o se il centroide manca -- oggetto con una sola detection, mai "
            f"ancora renderizzato abbastanza -- similarita' CLIP visuale "
            f">= {min_clip_sim}). 'merged_from' su ogni oggetto elenca gli "
            f"idx originali confluiti nel gruppo. Non modifica "
            f"graph_stream.jsonl: e' una vista derivata, da rigenerare "
            f"quando il grafo cresce."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-stream", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-dist", type=float, default=0.3,
                    help="distanza massima (m) tra centroidi per fondere due "
                         "nodi della stessa category (default 0.3m)")
    ap.add_argument("--min-clip-sim", type=float, default=MIN_CLIP_SIM_NO_CENTROID,
                    help="similarita' CLIP minima per fondere due nodi della "
                         "stessa category quando il centroide manca per "
                         "almeno uno dei due (default 0.92)")
    args = ap.parse_args()

    result = build_deduped_graph(args.graph_stream, args.max_dist, args.min_clip_sim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"Scritto: {args.out}")
    if "error" in result:
        print(f"  ERRORE: {result['error']}")
        return
    print(f"  frame {result['frame']}: {result['num_objects_originale']} nodi "
          f"-> {result['num_objects_deduplicato']} dopo la deduplica "
          f"({result['oggetti_fusi']} gruppi fusi)")
    fragmented = [o for o in result["objects"] if len(o["merged_from"]) > 2]
    if fragmented:
        print("  frammentazioni piu' vistose:")
        for o in sorted(fragmented, key=lambda o: -len(o["merged_from"]))[:8]:
            print(f"    {o['category']:<20} idx {o['merged_from']} -> gruppo {o['idx']}")


if __name__ == "__main__":
    main()
