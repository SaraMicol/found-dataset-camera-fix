"""Video dei momenti in cui cambiano gli oggetti, con inquadratura buona.

Uso (env dgsg):
    python3 viz_scripts/changes_walkthrough.py configs/found/dgsg_dynamic_fast.py \
        --out experiments/FOUND/00824_dynamic_fast_0/changes.mp4 --fps 2 --pitch 30

Legge graph_stream.jsonl, trova i frame dove compaiono oggetti nuovi,
riprende la stessa posizione camera ma livellata (solo yaw) e inclinata
di --pitch gradi verso il basso. Rendering dalla mappa salvata.
"""

import argparse
import json
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.common_utils import seed_everything
from utils.slam_external import build_rotation
from viz_scripts.walkthrough import render_frame, load_all


R_CHANGE = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
TRAJ_UP = np.array([0.0, 0.0, 1.0])
TRAJ_DOWN = -TRAJ_UP
SENSOR_UP = np.array([0.0, 1.5, 0.0])


def _shifted(w2c, up, meters):
    c = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    c[:3, 3] = c[:3, 3] - np.asarray(up, dtype=np.float64) * meters
    return np.linalg.inv(c)


def target_views(script_path, traj0_path):
    with open(script_path) as f:
        d = json.load(f)
    steps = [s for s in d["steps"] if "capture_eye" in s]
    c0 = np.loadtxt(traj0_path).reshape(-1, 4, 4)[0]
    c0_inv = np.linalg.inv(c0)
    views = []
    for s in steps:
        eye = np.asarray(s["capture_eye"], dtype=np.float64)
        tgt = np.asarray(s.get("target_surface_point", eye + [0, 0, -1.0]), dtype=np.float64)
        cam = eye + SENSOR_UP
        f_hab = tgt - cam
        f_hab = f_hab / (np.linalg.norm(f_hab) + 1e-12)
        f = R_CHANGE @ f_hab
        right = np.cross(TRAJ_DOWN, f)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            continue
        right = right / rn
        down = np.cross(f, right)
        down = down / (np.linalg.norm(down) + 1e-12)
        c2w = np.eye(4)
        c2w[:3, :3] = np.column_stack([right, down, f])
        c2w[:3, 3] = R_CHANGE @ cam
        views.append((s.get("target_category", "?"), np.linalg.inv(c0_inv @ c2w)))
    return views


def change_frames(graph_path, max_events=12):
    seen = set()
    events = []
    with open(graph_path) as f:
        for line in f:
            d = json.loads(line)
            ids = set(o["idx"] for o in d["objects"])
            new = sorted(ids - seen)
            if new:
                events.append((int(d["frame"]), new))
            seen |= ids
    events.sort(key=lambda e: -len(e[1]))
    return sorted(f for f, _ in events[:max_events])


def height_axis_and_up(cam_centers, points):
    axis = int(np.argmin(cam_centers.std(axis=0)))
    floor = np.percentile(points[:, axis], 1.0)
    up = np.zeros(3)
    up[axis] = 1.0 if float(np.median(cam_centers[:, axis])) > floor else -1.0
    return axis, up


def level_with_pitch(R_raw, tr_raw, up, pitch_deg):
    center = -(np.asarray(R_raw) .T @ np.asarray(tr_raw, dtype=np.float64))
    fwd = np.asarray(R_raw) @ np.array([0.0, 0.0, 1.0])
    fwd = fwd - up * np.dot(fwd, up)
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-8 else fwd
    p = np.radians(pitch_deg)
    f = fwd * np.cos(p) - up * np.sin(p)
    f = f / (np.linalg.norm(f) + 1e-12)
    down = -up
    right = np.cross(down, f)
    right = right / (np.linalg.norm(right) + 1e-12)
    w2c = np.eye(4)
    w2c[:3, :3] = np.column_stack([right, down, f])
    w2c[:3, 3] = -w2c[:3, :3] @ center
    return w2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--events", type=int, default=12)
    ap.add_argument("--targets", default=None,
                    help="Scene script JSON FOUND: inquadra i punti dove cambiano gli oggetti")
    ap.add_argument("--traj0", default="./data/FOUND/00824_dynamic/traj.txt")
    ap.add_argument("--lower", type=float, default=0.0)
    args = ap.parse_args()

    exp = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    seed_everything(seed=exp.config["seed"])
    viz = exp.config["viz"]
    results_dir = os.path.join(exp.config["workdir"], exp.config["run_name"])
    scene_path = exp.config.get("scene_path", os.path.join(results_dir, "params_with_idx.npz"))

    params, _, k, _, raw = load_all(scene_path)
    W, H = viz["viz_w"], viz["viz_h"]
    k = np.array(k, dtype=np.float64)

    d = dict(np.load(scene_path, allow_pickle=True))
    n = d["cam_unnorm_rots"].shape[-1]
    rots = torch.tensor(d["cam_unnorm_rots"]).float()
    trans = torch.tensor(d["cam_trans"]).float()
    centers = []
    Rs = []
    for t in range(n):
        r = F.normalize(rots[..., t].reshape(4), dim=0).cpu().numpy()
        R = build_rotation(torch.tensor(r).unsqueeze(0))[0].cpu().numpy()
        Rs.append(R)
        centers.append(np.array(trans[..., t].cpu().numpy()).reshape(-1))
    centers = np.array(centers)
    axis, up = height_axis_and_up(centers, np.array(d["means3D"]))
    print(f"asse alto: {axis}, frame totali: {n}")

    out = args.out or os.path.join(results_dir, "changes.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if args.targets:
        views = target_views(args.targets, args.traj0)
        print("viste oggetto:", [v[0] for v in views])
        if args.lower:
            floor = float(np.percentile(np.asarray(d["means3D"])[:, 1], 1.0))
            c0 = np.linalg.inv(np.asarray(views[0][1]))[:3, 3]
            up = np.array([0.0, 1.0 if c0[1] > floor else -1.0, 0.0])
            views = [(name, _shifted(w2c, up, args.lower)) for name, w2c in views]
        for name, w2c in tqdm(views, desc="targets"):
            frame = render_frame(params, np.array(w2c), k, W, H)
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    else:
        frames = change_frames(os.path.join(results_dir, "graph_stream.jsonl"), args.events)
        print("change frames:", frames)
        for t in tqdm(frames, desc="changes"):
            t = min(int(t), n - 1)
            w2c = level_with_pitch(Rs[t], centers[t], up, args.pitch)
            frame = render_frame(params, w2c, k, W, H)
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"scritto {out}")


if __name__ == "__main__":
    main()
