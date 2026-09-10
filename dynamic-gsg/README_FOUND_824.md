# Scena 824 — registrazione dinamica + costruzione del grafo

Riepilogo di setup e comandi per: registrare una sequenza RGB-D dinamica sulla
scena Habitat `00824-Dd4bFSTQ8gi` (dataset FOUND) e costruire/aggiornare lo
scene graph 3D con Dynamic-GSG su quella sequenza.

Il lavoro tocca due repository separati sulla stessa macchina:

- **`FOUND-Dataset`** (`/home/vodka/Desktop/TiagoSara/FOUND-Dataset`) — genera
  il dataset: simulatore Habitat + script che pilota una camera "umana" tra
  gli oggetti che cambiano.
- **`dynamic-gsg`** (`/home/vodka/dynamic-gsg`, questo repo) — legge il
  dataset e costruisce/aggiorna lo scene graph 3D con Gaussian Splatting.

Copia di lavoro delle modifiche a FOUND-Dataset, salvata anche su GitHub:
https://github.com/SaraMicol/found-dataset-camera-fix (privato).

---

## 1. Registrare il dataset (FOUND-Dataset)

Tre terminali, in ordine. Ognuno ha uno script che isola l'ambiente ROS
dell'env conda `foundgraph` da eventuali variabili di un `/ros2_humble`
sourceato nella shell chiamante (altrimenti: `ImportError` su `nav_msgs`,
difficile da diagnosticare — vedi sezione Troubleshooting).

```bash
cd /home/vodka/Desktop/TiagoSara/FOUND-Dataset

# Terminale 1 — simulatore Habitat, tienilo aperto per tutta la registrazione
./run_sim_824.sh

# Terminale 2 — finestra live su /camera/rgb (opzionale, solo per controllo visivo)
./run_live_view.sh
#   ./run_live_view.sh --depth      # affianca anche la depth colorizzata

# Terminale 3 — registrazione vera e propria
./run_record_824.sh
```

Aspetta che il terminale 1 stampi `Habitat ROS viewer ready` prima di avviare
il terminale 3: il recorder aspetta il primo frame su `/camera/rgb` e
`/camera/depth` e fallisce dopo 15s se il simulatore non sta pubblicando.

Output: `/home/vodka/dynamic-gsg/data/FOUND/00824_dynamic/` (`results/` con
`frameNNNNNN.jpg` + `depthNNNNNN.png`, `traj.txt` con le pose,
`dynamic_meta.json` con `num_frames` e `frame_begin_update`).

### Cosa fa la registrazione

Lo script scena (`scripts/generated/household_experiments_scene_824.json`)
descrive 23 cambiamenti: 10 spawn, 10 move, 3 remove su oggetti YCB (banana,
sugar box, bowl, cucchiaio, cubo di Rubik, ...). Per ognuno, la camera:

1. **Cammina** lungo la navmesh (non si teletrasporta) fino al punto di
   ripresa, a passi regolari — traiettoria continua, non scatti isolati.
2. Cattura qualche frame **"prima"** del cambiamento (l'oggetto ancora al suo
   posto, o il posto ancora vuoto per uno spawn).
3. Esegue davvero il comando (spawn/move/remove) via topic ROS sul nodo
   Habitat.
4. Cattura qualche frame **"dopo"**, dallo stesso punto di vista.

La quota della camera è fissa ad altezza occhi di una persona in piedi
(1.55 m assoluti da terra, non relativa all'oggetto), con inclinazione
naturale verso il basso — non a picco, non da terra verso l'alto.

`frame_begin_update` nel meta segna il primo frame in cui la scena inizia
davvero a cambiare (dopo tutti gli spawn iniziali): è il valore che va passato
alla pipeline del grafo per sapere da quando iniziare a controllare se
qualcosa è sparito/spostato.

Valori dell'ultima registrazione buona:

```json
{
  "num_frames": 1453,
  "frame_begin_update": 517
}
```

### Parametri utili di `run_record_824.sh`

```bash
./run_record_824.sh --step-m 0.35 --walk-pause-s 0.1        # camminata piu' veloce
./run_record_824.sh --settle-seconds 3 --frames-per-waypoint 20   # sosta piu' lunga
```

Default: `--step-m 0.20`, `--walk-pause-s 0.25`, `--settle-seconds 1.5`,
`--frames-per-waypoint 10`, `--remove-frames 6`.

Quota della camera regolabile senza toccare il codice (di norma non serve,
resta a 0 — la quota la decide interamente il recorder):

```bash
HABITAT_SENSOR_HEIGHT=0.2 ./run_sim_824.sh
```

---

## 2. Costruire/aggiornare il grafo (dynamic-gsg)

Il dataset deve essere già registrato per intero — la pipeline legge i frame
da disco, non gira in streaming durante la registrazione.

```bash
cd /home/vodka/dynamic-gsg
DISPLAY=:1 /home/vodka/miniconda3/envs/dgsg/bin/python \
    scripts/dynamic_gsg_real_ssim.py configs/found/dgsg_dynamic.py
```

Si apre una finestra OpenCV (`DynamicGSG - scena, oggetti, grafo`): a sinistra
la camera con le maschere degli oggetti rilevati, a destra il grafo che
cresce e si aggiorna — disegnata dalla pipeline stessa mentre processa i
frame, non precalcolata.

**Prima di lanciare**, se il dataset è stato registrato di nuovo, aggiorna in
[configs/found/dgsg_dynamic.py](configs/found/dgsg_dynamic.py) i due valori
con quelli scritti in `dynamic_meta.json`:

```python
data=dict(
    ...
    num_frames=1453,        # <- da dynamic_meta.json: num_frames
    frame_begin_update=517, # <- da dynamic_meta.json: frame_begin_update
    ...
),
```

### Come funziona (in breve)

- **Nodi** = oggetti 3D mappati (`MapObjectList` in
  [utils/slam_classes.py](utils/slam_classes.py)): nuvola di punti gaussiana,
  bbox, classe, embedding CLIP.
- **Archi** = relazioni spaziali tra coppie di oggetti (`MapEdgeMapping`,
  stesso file): `rel_type` + contatore di quante volte osservata.
- Ogni frame: rileva oggetti nell'immagine (GroundingDINO/SAM), li confronta
  con quelli già mappati (IoU + similarità CLIP), aggiorna un nodo esistente
  o ne crea uno nuovo
  ([utils/map_objects_utils_up_with_groupv3.py](utils/map_objects_utils_up_with_groupv3.py),
  `compute_similarities_and_merge`).
- Da `frame_begin_update` in poi, per ogni oggetto già mappato controlla se è
  ancora visibile dove si aspetta di vederlo; se sistematicamente non lo è,
  lo rimuove fisicamente dalla mappa 3D
  ([scripts/dynamic_gsg_real_ssim.py](scripts/dynamic_gsg_real_ssim.py),
  intorno a `check_update`/`objects_to_remove`).

Tempo stimato: circa 3-4 secondi/frame osservati sui primi frame → diverse
ore per 1453 frame. Il config ha `report_global_progress_every=500`.

### Dipendenze necessarie (env `dgsg`)

Modelli in `./models/` (già presenti su questa macchina):
`groundingdino_swint_ogc.pth`, `sam_l.pt`, `ram_plus_swin_large_14m.pth`,
`open_clip_pytorch_model.bin`, `yolov8l-world.pt`.

Ollama attivo su `localhost:11434` con il modello `gemma3:4b` (usato per le
descrizioni testuali degli oggetti):

```bash
ollama list   # deve comparire gemma3:4b
```

---

## Troubleshooting

**`ImportError: ... nav_msgs ... undefined symbol: nav_msgs__msg__goals__convert_from_py`**
Il terminale ha `/ros2_humble/install/setup.bash` sourceato insieme all'env
conda `foundgraph` (che ha il suo ROS 2 Humble completo, RoboStack):
Python carica la metà Python di `nav_msgs` da un prefisso e la libreria
nativa dall'altro. Soluzione: usare sempre `./run_sim_824.sh` /
`./run_record_824.sh` / `./run_live_view.sh`, mai chiamare direttamente
`python habitat_camera_objects_node.py` — i launcher ripartono da un
ambiente vuoto (`env -i`) e reintroducono solo le variabili necessarie.

**`cv2.error: ... window.cpp ... Rebuild the library with ... GTK ...`**
L'opencv dell'env `foundgraph` è `opencv-python-headless` (nessun backend
finestra). `live_view.py` usa tkinter invece di `cv2.imshow`, quindi questo
errore non dovrebbe più comparire — se compare, verificare di non aver
reinstallato un opencv diverso in quell'env.

**Il recorder resta fermo su "Nessun frame ricevuto"**
Il terminale 1 (`run_sim_824.sh`) non è ancora pronto o non è attivo.
Controllare che stampi `Habitat ROS viewer ready` prima di avviare il
recorder.

**Un recorder precedente è rimasto vivo in background**
Se interrotto con Ctrl+C a metà, un `record_dynamic_sequence.py` rimasto
vivo continua a pubblicare comandi al nodo Habitat, mescolandosi con un
nuovo test/run e sporcando la scena (oggetti duplicati) o sovrascrivendo
`dynamic_meta.json` a metà. Controllare prima di ripartire:

```bash
ps aux | grep record_dynamic_sequence
```

**Il nodo Habitat ha oggetti residui da un test precedente**
Riavviare il nodo (terminale 1) pulito prima di una registrazione vera,
altrimenti la scena avrà oggetti in più rispetto allo script.
