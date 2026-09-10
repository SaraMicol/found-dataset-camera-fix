# FOUND-Dataset — camera fix per la registrazione dinamica

Modifiche a [FOUND-Dataset](https://github.com/saraian/FOUND-Dataset) per la scena
`00824-Dd4bFSTQ8gi`: questi file vanno copiati nella root del repo FOUND-Dataset,
sostituendo gli originali.

## Cosa cambia

- **`habitat_camera_objects_node.py`** — `sensor_height` configurabile via
  `HABITAT_SENSOR_HEIGHT` (default `0.0`): la quota della camera ora la decide
  interamente `record_dynamic_sequence.py`, non piu' sommata due volte.
- **`record_dynamic_sequence.py`**:
  - **Bug fix**: la posizione di partenza della camera veniva letta dal TF ROS
    (coordinate Z-up) invece che restare `None` per il primo teletrasporto in
    coordinate Habitat (Y-up) — mescolava due sistemi di assi e produceva
    inquadrature quasi verticali per gran parte della sequenza.
  - **Quota camera a altezza umana** (`HUMAN_EYE_HEIGHT_M = 1.55`, assoluta,
    non relativa all'oggetto): inclinazione naturale invece che a picco (~63°)
    o da terra.
  - **Camminata piu' lenta** (`--step-m`, `--walk-pause-s`) e **sosta piu'
    lunga** davanti a ogni cambiamento (`--settle-seconds`,
    `--frames-per-waypoint`).
  - **Prima/dopo per ogni cambiamento**: spawn, move e remove catturano ora
    sia lo stato precedente sia quello nuovo, dallo stesso punto di vista —
    non solo il "dopo" come nella versione originale.
- **`live_view.py`** (nuovo) — finestra live su `/camera/rgb` in tkinter
  (l'opencv dell'env `foundgraph` e' headless, senza supporto GUI).
- **`run_sim_824.sh`, `run_record_824.sh`, `run_live_view.sh`** (nuovi) —
  launcher che isolano l'ambiente ROS dell'env conda `foundgraph` da eventuali
  variabili di un `/ros2_humble` sourceato nella shell (causa di un
  `ImportError` su `nav_msgs` altrimenti difficile da diagnosticare).

## Uso

```bash
cd FOUND-Dataset  # dopo aver copiato questi file dentro
./run_sim_824.sh        # terminale 1 — simulatore
./run_live_view.sh      # terminale 2 — opzionale, vista live
./run_record_824.sh     # terminale 3 — registrazione
```
