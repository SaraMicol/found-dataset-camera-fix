"""Render video su percorso camminabile da navmesh (frame Habitat -> mappa).

Uso (env dgsg):
    python3 viz_scripts/walkthrough_navmesh.py configs/found/dgsg_dynamic.py \
        --path tools/walk_path_00824.npz \
        --out experiments/FOUND/00824_dynamic_0/walk_nav.mp4 --fps 10

Lo sguardo e' costruito con la stessa ricetta del nodo Habitat
(right = fwd x up, mai sottosopra), espressa nel frame traj (Z-up,
camera ottica x-destra/y-giu'/z-avanti) e poi relativizzata come fa
GradSLAM (identita' al primo frame).
"""

import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.gradslam_datasets.geometryutils import relative_transformation
from viz_scripts.walkthrough import render_frame, load_all
from utils.common_utils import seed_everything


R_CHANGE = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
TRAJ_UP = np.array([0.0, 0.0, 1.0])
TRAJ_DOWN = -TRAJ_UP


def rotation_optical(fwd):
    f = np.asarray(fwd, dtype=np.float64)
    f = f - TRAJ_UP * np.dot(f, TRAJ_UP)
    n = np.linalg.norm(f)
    f = f / (n + 1e-12) if n > 1e-8 else np.array([1.0, 0.0, 0.0])
    right = np.cross(TRAJ_DOWN, f)
    right = right / (np.linalg.norm(right) + 1e-12)
    return np.column_stack([right, TRAJ_DOWN, f])


def shift_down(w2cs, scene_path, meters):
    if not meters:
        return w2cs
    d = dict(np.load(scene_path, allow_pickle=True))
    floor = float(np.percentile(np.asarray(d["means3D"])[:, 1], 1.0))
    c0 = np.linalg.inv(np.asarray(w2cs[0]))[:3, 3]
    up = np.array([0.0, 1.0 if c0[1] > floor else -1.0, 0.0])
    out = []
    for w in w2cs:
        c = np.linalg.inv(np.asarray(w, dtype=np.float64))
        c[:3, 3] = c[:3, 3] - up * meters
        out.append(np.linalg.inv(c))
    return out


def habitat_path_to_map_w2cs(hab_pos, hab_fwd, pitch_deg=0.0):
    pitch = np.radians(pitch_deg)
    c2ws = []
    for p, f in zip(hab_pos, hab_fwd):
        f = np.asarray(f, dtype=np.float64)
        f = f * np.cos(pitch) + np.array([0.0, -np.sin(pitch), 0.0])
        f = f / (np.linalg.norm(f) + 1e-12)
        c2w = np.eye(4)
        c2w[:3, :3] = rotation_optical(R_CHANGE @ f)
        c2w[:3, 3] = R_CHANGE @ np.asarray(p, dtype=np.float64)
        c2ws.append(c2w)
    poses = torch.tensor(np.stack(c2ws)).float()
    rel = relative_transformation(
        poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1), poses
    )
    return [torch.linalg.inv(p).numpy() for p in rel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--lower", type=float, default=0.0)
    args = ap.parse_args()

    exp = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    seed_everything(seed=exp.config["seed"])
    viz = exp.config["viz"]
    results_dir = os.path.join(exp.config["workdir"], exp.config["run_name"])
    scene_path = exp.config.get("scene_path", os.path.join(results_dir, "params_with_idx.npz"))

    params, _, k, _, _ = load_all(scene_path)
    W, H = viz["viz_w"], viz["viz_h"]
    k = np.array(k, dtype=np.float64)
    k[0, :] *= W / 640
    k[1, :] *= H / 480

    d = np.load(args.path)
    w2cs = habitat_path_to_map_w2cs(d["hab_pos"], d["hab_fwd"], args.pitch)
    w2cs = shift_down(w2cs, scene_path, args.lower)
    print(f"pose percorso: {len(w2cs)}")

    out = args.out or os.path.join(results_dir, "walk_nav.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for w2c in tqdm(w2cs, desc="walkthrough navmesh"):
        frame = render_frame(params, np.array(w2c), k, W, H)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"scritto {out}")


if __name__ == "__main__":
    main()
