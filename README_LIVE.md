# Eseguire una scena e ottenere <numero_scena>_live.json

```bash
cd /home/vodka/dynamic-gsg

# scena 824
bash scripts/live_824.sh

# scena 829
bash scripts/live_829.sh
```

Uno script `live_<N>.sh` avvia tutto quello che serve (simulatore, pipeline,
camminata) **e già include** il monitor che scrive il file live — non serve
lanciare nient'altro a parte.

Il file compare in:

```
experiments/FOUND/00824_live_0/00824_live.json
experiments/FOUND/00829_live_0/00829_live.json
```

e si riscrive da solo ogni 5 secondi finché la run è in corso.

Per farla girare senza tenere il terminale aperto:

```bash
nohup bash scripts/live_824.sh > /tmp/live824.log 2>&1 &
disown
```
