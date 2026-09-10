"""Pianifica un percorso camminabile sulla navmesh Habitat.

Uso (env con habitat_sim, es. foundgraph):
    python tools/plan_walk_path.py \
        --script /home/vodka/Desktop/TiagoSara/FOUND-Dataset/scripts/generated/household_experiments_scene_824.json \
        --navmesh /home/vodka/concept-graphs/hm3d_eval/data/val/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.navmesh \
        --scene /home/vodka/concept-graphs/hm3d_eval/data/val/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb \
        --out tools/walk_path_00824.npz

I waypoint sono i capture_eye dello script FOUND (punti semantici: la camera
guardava davvero li'), collegati con shortest path sulla navmesh invece dei
teleport della registrazione. Output: posizioni camera + sguardi in frame
Habitat (Y-up), quota costante a EYE_ABOVE_FLOOR dal pavimento.
"""

import argparse
import os

import numpy as np


EYE_ABOVE_FLOOR = 1.6
STEP = 0.25


def load_nav(script_path, navmesh_path, scene_path):
    import habitat_sim

    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = scene_path
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(sc, [habitat_sim.AgentConfiguration()])
    )
    pf = sim.pathfinder
    pf.load_nav_mesh(navmesh_path)
    return pf


def chain_paths(pf, waypoints):
    import habitat_sim

    full = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        sp = habitat_sim.ShortestPath()
        sp.requested_start = a
        sp.requested_end = b
        if not pf.find_path(sp) or len(sp.points) < 2:
            continue
        pts = [np.array(p, dtype=np.float64) for p in sp.points]
        if full:
            pts = pts[1:]
        full.extend(pts)
    return full


def resample(polyline, step):
    polyline = np.array(polyline)
    segs = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    total = segs.sum()
    n = max(2, int(total / step) + 1)
    dists = np.concatenate([[0.0], np.cumsum(segs)])
    idx = np.linspace(0.0, total, n)
    return np.stack([np.interp(idx, dists, polyline[:, k]) for k in range(3)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--navmesh", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eye", type=float, default=EYE_ABOVE_FLOOR)
    ap.add_argument("--step", type=float, default=STEP)
    args = ap.parse_args()

    import json

    with open(args.script) as f:
        d = json.load(f)
    eyes = np.array(
        [s["capture_eye"] for s in d["steps"] if "capture_eye" in s],
        dtype=np.float64,
    )
    print(f"waypoint: {len(eyes)}")

    pf = load_nav(args.script, args.navmesh, args.scene)
    snapped = np.array([pf.snap_point(e) for e in eyes])
    print("max snap dist:", round(float(np.linalg.norm(snapped - eyes, axis=1).max()), 3))

    poly = chain_paths(pf, snapped)
    print(f"punti path: {len(poly)}")
    path = resample(poly, args.step)
    floor = np.array([pf.snap_point(p) for p in path])
    cam = floor.copy()
    cam[:, 1] = floor[:, 1] + args.eye

    fwd = np.zeros_like(cam)
    for i in range(len(cam)):
        j = min(i + 1, len(cam) - 1)
        k = max(i - 1, 0)
        v = cam[j] - cam[k]
        v[1] = 0.0
        n = np.linalg.norm(v)
        fwd[i] = v / n if n > 1e-6 else (fwd[i - 1] if i > 0 else np.array([0, 0, -1.0]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, hab_pos=cam, hab_fwd=fwd)
    print(f"scritto {args.out} ({len(cam)} pose)")


if __name__ == "__main__":
    main()
