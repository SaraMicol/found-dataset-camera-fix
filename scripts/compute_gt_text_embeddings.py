#!/usr/bin/env python3
"""Calcola UNA VOLTA gli embedding testuali CLIP per le categorie dei
template Habitat usati nello script (es. "051_large_clamp" -> "large
clamp") e li salva su disco. Usa lo stesso identico modello/checkpoint
della pipeline (ViT-H-14, laion2b_s32b_b79k -- vedi
scripts/dynamic_gsg_real_ssim.py) cosi' l'embedding testuale vive nello
stesso spazio vettoriale del clip_ft VISUALE che la pipeline salva per
ogni oggetto osservato (log_graph_state, graph_stream.jsonl): il
confronto testo<->immagine via cosine similarity e' esattamente il modo
in cui CLIP e' stato addestrato a funzionare (non un trucco per parole).

Va eseguito una volta all'inizio (o quando cambia lo script), non ad ogni
refresh del watcher: caricare ViT-H-14 richiede diversi secondi/GB, farlo
ogni 3s nel loop di watch_gt_vs_graph.py sarebbe troppo lento.

Uso:
    /home/vodka/miniconda3/envs/dgsg/bin/python scripts/compute_gt_text_embeddings.py \
        --script /path/to/household_experiments_scene_824.json \
        --out experiments/FOUND/00824_live_0/gt_text_embeddings.json
"""

import argparse
import json
from pathlib import Path


def template_to_category(template: str) -> str:
    parts = template.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return " ".join(parts)


def graph_categories(graph_stream_path: Path):
    """Categorie DISTINTE presenti nell'ultimo frame di graph_stream.jsonl.

    Servono per il confronto testo<->testo in compare_final_state.py: la
    category del grafo (es. "rubiks cube") e quella del template GT (es.
    "puzzle cube") sono due stringhe diverse per lo stesso oggetto fisico,
    e solo incorporandole entrambe nello stesso spazio CLIP si puo'
    misurarne la similarita' invece di sperare in un match per parole.

    Legge solo l'ultima riga: il file e' grande (>100MB) e lo stato finale
    e' l'unico che interessa."""
    if graph_stream_path is None or not graph_stream_path.exists():
        return []
    last = None
    with open(graph_stream_path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return []
    entry = json.loads(last)
    cats = {o.get("category") for o in entry.get("objects", []) if o.get("category")}
    return sorted(cats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, type=Path)
    ap.add_argument("--graph-stream", type=Path, default=None,
                    help="graph_stream.jsonl: incorpora anche le categorie osservate "
                         "nell'ultimo frame, cosi' il confronto GT<->grafo e' testo<->testo "
                         "nello stesso spazio vettoriale")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import open_clip
    import torch

    data = json.loads(args.script.read_text())
    name_to_category = {}
    for step in data.get("steps", []):
        if step.get("action") == "spawn" and step.get("template"):
            name_to_category[step["name"]] = template_to_category(step["template"])

    print(f"Categorie da incorporare: {name_to_category}")

    observed = graph_categories(args.graph_stream)
    if observed:
        print(f"Categorie osservate nel grafo finale: {len(observed)}")

    print("Carico CLIP ViT-H-14 (stesso modello della pipeline)...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model = model.to("cuda") if torch.cuda.is_available() else model

    names = list(name_to_category.keys())
    categories = list(name_to_category.values())
    with torch.no_grad():
        tokens = tokenizer(categories)
        if torch.cuda.is_available():
            tokens = tokens.to("cuda")
        text_feats = model.encode_text(tokens)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.cpu().numpy()

    out = {
        name: {
            "category_text": name_to_category[name],
            "clip_text_ft": text_feats[i].tolist(),
        }
        for i, name in enumerate(names)
    }

    # Le categorie del grafo vanno in una sezione separata, indicizzate per
    # stringa e non per name: nel grafo non esiste il "name" dell'oggetto GT
    # (la pipeline non sa come si chiama), solo la categoria rilevata.
    graph_out = {}
    if observed:
        with torch.no_grad():
            tokens = tokenizer(observed)
            if torch.cuda.is_available():
                tokens = tokens.to("cuda")
            feats = model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy()
        graph_out = {cat: feats[i].tolist() for i, cat in enumerate(observed)}

    payload = {"gt": out, "graph_categories": graph_out}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"Scritto: {args.out} ({len(out)} embedding GT, "
          f"{len(graph_out)} categorie del grafo)")


if __name__ == "__main__":
    main()
