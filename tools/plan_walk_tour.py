"""Tour camminato che copre tutte le stanze (navmesh Habitat).

Uso (env con habitat_sim, es. foundgraph):
    python tools/plan_walk_tour.py \
        --navmesh /home/vodka/concept-graphs/hm3d_eval/data/val/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.navmesh \
        --scene /home/vodka/concept-graphs/hm3d_eval/data/val/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb \
        --out tools/walk_tour_00824.npz --points 30

Campionamento copertura (farthest-point) sull'isola principale, giro
nearest-neighbour, shortest path tra tappe, quota 1.6m, sguardo lungo il
moto mai ribaltato. Output in frame Habitat come plan_walk_path.py.
"""

import argparse
import os

import numpy as np


EYE_ABOVE_FLOOR = 1.6
STEP = 0.25


def sample_coverage(pf, n_samples=4000, cell=0.5, n_points=30, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_samples):
        p = np.array(pf.get_random_navigable_point(), dtype=np.float64)
        pts.append(p)
    pts = np.array(pts)
    key = np.floor(pts / cell).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    pts = pts[np.sort(idx)]

    start = pts[len(pts) // 2]
    sp0 = __import__("habitat_sim").ShortestPath()
    good = []
    for p in pts:
        sp0.requested_start = start
        sp0.requested_end = p
        if pf.find_path(sp0) and len(sp0.points) >= 2:
            good.append(p)
    pts = np.array(good)
    print(f"punti navigabili isola principale: {len(pts)}")

    chosen = [pts[np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1))]]
    while len(chosen) < min(n_points, len(pts)):
        d = np.min(
            np.linalg.norm(pts[:, None, :] - np.array(chosen)[None, :, :], axis=2),
            axis=1,
        )
        chosen.append(pts[int(np.argmax(d))])
    return np.array(chosen)


def order_nearest(pts):
    order = [0]
    rest = list(range(1, len(pts)))
    while rest:
        last = pts[order[-1]]
        j = min(rest, key=lambda i: float(np.linalg.norm(pts[i] - last)))
        order.append(j)
        rest.remove(j)
    return pts[order]


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
    ap.add_argument("--navmesh", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=30)
    ap.add_argument("--eye", type=float, default=EYE_ABOVE_FLOOR)
    ap.add_argument("--step", type=float, default=STEP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import habitat_sim

    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = args.scene
    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(sc, [habitat_sim.AgentConfiguration()])
    )
    pf = sim.pathfinder
    pf.load_nav_mesh(args.navmesh)

    wp = sample_coverage(pf, n_points=args.points, seed=args.seed)
    wp = order_nearest(wp)
    poly = chain_paths(pf, wp)
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
    total = float(np.linalg.norm(np.diff(cam, axis=0), axis=1).sum())
    print(f"scritto {args.out} ({len(cam)} pose, {total:.1f} m)")


if __name__ == "__main__":
    main()
