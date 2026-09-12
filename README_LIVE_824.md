# Scena dinamica 824 — come avviare tutto da terminale

Confronto tra la ground truth di una scena dinamica Habitat (spawn/move/remove
scriptati) e il grafo ricostruito da dynamic-gsg.

Tutto gira su `experiments/FOUND/00824_live_0/`, che **non è versionata**
(`.gitignore`): i file di output si rigenerano a ogni run.

---

## Avvio rapido

Una run completa, tutto incluso (simulatore + pipeline + monitor + camminata):

```bash
cd /home/vodka/dynamic-gsg
bash scripts/live_824.sh
```

In background, per non tenere occupato il terminale:

```bash
nohup bash scripts/live_824.sh > /tmp/live824.log 2>&1 &
disown
```

Lo script avvia **quattro processi in quest'ordine** (non è intercambiabile):

| # | processo | cosa fa |
|---|---|---|
| 1 | `run_sim_824.sh` (FOUND-Dataset) | nodo Habitat, pubblica `/camera/*` |
| 2 | `scripts/dynamic_gsg_real_ssim.py` | pipeline dgsg, si iscrive al bridge ROS |
| 3 | `scripts/watch_live_progress.py` | monitor: scrive `00824_live.json` ogni 5s |
| 4 | `run_record_824_live.sh` (FOUND-Dataset) | cammina nella scena ed esegue lo script |

Il passo 4 parte **solo dopo** che la pipeline è pronta: ogni frame pubblicato
senza consumatore aspetta l'ack fino a 60s prima di procedere con un warning.

---

## Seguire la run mentre gira

```bash
# progresso live (si aggiorna ogni 5s)
watch -n2 'python3 -c "
import json; d=json.load(open(\"experiments/FOUND/00824_live_0/00824_live.json\"))
print(\"frame\", d[\"meta\"][\"ultimo_frame_elaborato\"], \"| oggetti\", d[\"totale_oggetti_nel_grafo\"])
print(\"gt_progress  \", d[\"gt_progress\"][\"spawn\"])
print(\"confermati   \", d[\"progress_live\"][\"confirmed_in_graph\"][\"spawn\"])"'

# eventi realmente avvenuti nel simulatore
tail -f experiments/FOUND/00824_live_0/gt_events.jsonl

# log della pipeline (cercare OOM o crash)
tail -f "$(ls -t logs/pipeline824live_*.log | head -1)"

# quanti oggetti e quanti duplicati ci sono adesso
python3 -c "
import json; from collections import Counter
last=None
with open('experiments/FOUND/00824_live_0/graph_stream.jsonl') as f:
    for line in f: last=line
d=json.loads(last)
print('frame', d['frame'], '-> num_objects', d['num_objects'])
for cat,n in Counter(o['category'] for o in d['objects']).most_common(8):
    print(' %2dx %s' % (n,cat))
"
```

Fermare tutto:

```bash
pkill -9 -f "live_824.sh|dynamic_gsg_real_ssim|habitat_camera_objects_node"
pkill -9 -f "watch_live_progress|record_dynamic_sequence_live"
```

---

## I tre numeri in `00824_live.json`

Sono viste **diverse**, non ridondanti. Leggerle insieme è l'unico modo per
capire se un valore basso è colpa del metodo o della run.

| campo | domanda a cui risponde | quando è significativo |
|---|---|---|
| `gt_progress` | quanti eventi il **simulatore** ha davvero emesso finora | sempre — è un fatto, non un confronto |
| `progress_live` | di quegli eventi, quanti il grafo ha **già confermato** | sempre, si aggiorna man mano |
| `progress` | confronto contro lo **stato finale** atteso (rigioca tutto il piano) | solo a run completa |

Esempio di lettura: `gt_progress.spawn = 3/10` e
`progress_live.confirmed_in_graph.spawn = 1/10` significa "il simulatore ha
spawnato 3 oggetti, la pipeline ne ha ritrovato 1". Invece `progress.spawn =
0/10` a run appena iniziata **non** è un fallimento: confronta contro la fine.

Vale l'invariante **trovati ⊆ avvenuti**: un evento non ancora accaduto non può
essere stato trovato. In `spawn_eventi` i motivi possibili sono:

- `non ancora eseguito dal simulatore` → evento futuro, non è un fallimento
- `pianificato per essere rimosso` → l'esito si verifica in `remove_eventi`
- `N candidati entro 0.9m ma label incompatibile` → **questo** è un fallimento vero

---

## Script singoli

Utili per rianalizzare una run già fatta, senza rieseguirla.

```bash
PY=/home/vodka/miniconda3/envs/dgsg/bin/python
D=experiments/FOUND/00824_live_0
```

**Embedding testuali CLIP** (una volta per run; `--graph-stream` va passato
*dopo* che il grafo esiste, altrimenti `graph_categories` resta vuoto e ogni
similarità torna 0.00):

```bash
$PY scripts/compute_gt_text_embeddings.py \
  --script $D/household_experiments_scene_824_plan.json \
  --graph-stream $D/graph_stream.jsonl \
  --out $D/gt_text_embeddings.json
```

**Report finale** (`00824_summary.json`, `00824_log.txt`, `00824_mappa.json`):

```bash
$PY scripts/report_scene_results.py \
  --script $D/household_experiments_scene_824_plan.json \
  --graph-stream $D/graph_stream.jsonl \
  --gt-log $D/gt_events.jsonl \
  --text-embeddings $D/gt_text_embeddings.json \
  --scene 00824 --out-dir $D
```

**Confronto a stato finale** da solo:

```bash
$PY scripts/compare_final_state.py \
  --script $D/household_experiments_scene_824_plan.json \
  --graph-stream $D/graph_stream.jsonl \
  --gt-log $D/gt_events.jsonl \
  --text-embeddings $D/gt_text_embeddings.json \
  --out $D/live_comparison.json
```

**Deduplica dei nodi frammentati** (stessa categoria + centroidi vicini, o
similarità CLIP alta quando il centroide manca):

```bash
$PY scripts/dedup_graph_objects.py \
  --graph-stream $D/graph_stream.jsonl \
  --out $D/00824_graph_deduped.json \
  --max-dist 0.3
```

---

## File prodotti

| file | contenuto |
|---|---|
| `00824_live.json` | progresso live, riscritto ogni 5s |
| `00824_live_log.txt` | diagnostica leggibile: chi trovato, chi no, perché |
| `00824_mappa.json` | tutti gli oggetti del grafo con posizione Habitat |
| `00824_graph_deduped.json` | grafo dopo la deduplica dei nodi frammentati |
| `graph_stream.jsonl` | il grafo, una riga per frame (può superare i 100MB) |
| `gt_events.jsonl` | eventi realmente eseguiti dal simulatore |

Log di processo in `logs/`: `live824_*` (principale), `pipeline824live_*`,
`sim824live_*`, `record824live_*`, `watch824live_*`.

---

## Problemi noti e come riconoscerli

**CUDA out of memory.** Era la causa della run interrotta al frame 158.
Mitigato con `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (già in
`live_824.sh`), `map_every=2` e un `try/except` che salta il frame invece di
uccidere il processo. Per verificare:

```bash
grep -c "OutOfMemoryError" "$(ls -t logs/pipeline824live_*.log | head -1)"
```

**Oggetti duplicati (problema aperto).** Lo stesso oggetto fisico compare come
molti nodi (es. 19x `pillow`), quasi tutti con `detections: 1` — mai
riagganciati. La catena è: un oggetto nuovo non riceve gaussiane → si
rasterizza a 0 pixel → sparisce dal pool di confronto di
`compute_similarities_and_merge` → la detection successiva ne crea un altro.

Per diagnosticare, c'è una traccia opzionale (inerte se la variabile non è
impostata):

```bash
DGSG_DEBUG_RENDER=1 bash scripts/live_824.sh
grep "\[DBG\]" "$(ls -t logs/pipeline824live_*.log | head -1)" | head -20
```

Stampa per ogni oggetto non renderizzato: `n_gauss` (gaussiane assegnate),
`in_frustum`, opacità. `n_gauss=0` significa che l'oggetto non ha mai ricevuto
gaussiane — è il caso dominante e la pista ancora da chiudere.

`scripts/dedup_graph_objects.py` **non** risolve il problema: ripulisce il
report a posteriori, ma il grafo continua a riempirsi durante la run.

**Similarità label tutte a 0.00.** Vuol dire che `gt_text_embeddings.json` ha
`graph_categories` vuoto (calcolato prima che il grafo esistesse). Il report lo
segnala con `graph_categories_missing: true` e un avviso in cima a `nota`. Si
risolve rilanciando `compute_gt_text_embeddings.py` **con** `--graph-stream`.

---

## Dipendenze

- conda env `dgsg` → pipeline, script di analisi
- conda env `foundgraph` → nodo Habitat e recorder (FOUND-Dataset)
- FOUND-Dataset in `/home/vodka/Desktop/TiagoSara/FOUND-Dataset`
- GPU con ~15 GiB liberi (GroundingDINO + SAM + CLIP ViT-H-14 + training gaussiano)
