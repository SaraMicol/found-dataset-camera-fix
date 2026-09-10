"""Traiettoria di visualizzazione a altezza uomo.

La traiettoria raw e' stop-and-go: pose duplicate, salti di metri, sguardo
che vaga (a terra, al soffitto, agli oggetti). Seguirla raw da' inquadrature
che saltano o si ribaltano.

Qui: asse alto rilevato dai dati (varianza minima delle posizioni),
teleport spezzati in segmenti (niente swoop attraverso i muri), quota
smussata ma conservata, sguardo orizzontale livellato (solo yaw, mai
sottosopra, yaw rate limitato).
"""

import numpy as np


def _moving_average(x, window):
    if window <= 1 or len(x) <= 2:
        return x
    w = max(1, int(window) // 2 * 2 + 1)
    pad = w // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(w) / w
    out = np.stack([np.convolve(xp[:, i], kernel, mode="valid") for i in range(x.shape[1])], axis=1)
    return out


def _detect_height_axis(positions):
    return int(np.argmin(positions.std(axis=0)))


def _rotation_level_y_up(yaw):
    s, c = np.sin(yaw), np.cos(yaw)
    R = np.array([[-c, 0.0, s],
                  [0.0, -1.0, 0.0],
                  [s, 0.0, c]])
    return R


def _rotation_level_z_up(yaw):
    s, c = np.sin(yaw), np.cos(yaw)
    R = np.array([[s, 0.0, c],
                  [-c, 0.0, s],
                  [0.0, -1.0, 0.0]])
    return R


def _yaw_from_forward(fwd, height_axis, prev_yaw):
    if height_axis == 1:
        hx, hz = fwd[0], fwd[2]
        if hx * hx + hz * hz < 1e-6:
            return prev_yaw
        return float(np.arctan2(hx, hz))
    hx, hy = fwd[0], fwd[1]
    if hx * hx + hy * hy < 1e-6:
        return prev_yaw
    return float(np.arctan2(hy, hx))


def _rotation_for_axis(yaw, height_axis):
    if height_axis == 1:
        return _rotation_level_y_up(yaw)
    return _rotation_level_z_up(yaw)


def _smooth_segment(positions, forwards, height_axis, smooth_window, max_yaw_rate,
                    seed_yaw=None):
    smooth_pos = _moving_average(positions, smooth_window)
    yaw = np.zeros(len(forwards))
    prev = _yaw_from_forward(forwards[0], height_axis,
                             seed_yaw if seed_yaw is not None else 0.0)
    for i, f in enumerate(forwards):
        prev = _yaw_from_forward(f, height_axis, prev)
        yaw[i] = prev
    yaw_sm = _moving_average(np.unwrap(yaw)[:, None], max(3, smooth_window // 2))[:, 0]
    if seed_yaw is not None and len(yaw_sm):
        shift = (seed_yaw - yaw_sm[0] + np.pi) % (2 * np.pi) - np.pi
        if abs(shift) > np.pi / 2:
            yaw_sm = yaw_sm + shift
    dyaw = (np.diff(yaw_sm) + np.pi) % (2 * np.pi) - np.pi
    dyaw = np.clip(dyaw, -max_yaw_rate, max_yaw_rate)
    yaw_sm[1:] = yaw_sm[0] + np.cumsum(dyaw)
    out = []
    for p, y in zip(smooth_pos, yaw_sm):
        c2w = np.eye(4)
        c2w[:3, :3] = _rotation_for_axis(float(y), height_axis)
        c2w[:3, 3] = p
        out.append(np.linalg.inv(c2w))
    return out, float(yaw_sm[-1])


def smooth_walk_trajectory(all_w2cs, smooth_window=15, eye_height=None,
                           level_camera=True, dedup_pos_thresh=0.02,
                           dedup_ang_thresh=0.005, max_yaw_rate=0.15,
                           teleport_thresh=1.0):
    del level_camera
    all_w2cs = [np.asarray(w, dtype=np.float64) for w in all_w2cs]
    if len(all_w2cs) == 0:
        return []

    positions, forwards = [], []
    for w2c in all_w2cs:
        c2w = np.linalg.inv(w2c)
        positions.append(c2w[:3, 3])
        forwards.append(c2w[:3, 2])
    positions = np.array(positions)
    forwards = np.array(forwards)

    height_axis = _detect_height_axis(positions)

    keep = [0]
    for i in range(1, len(positions)):
        dp = np.linalg.norm(positions[i] - positions[keep[-1]])
        cos_a = np.clip(np.dot(forwards[i], forwards[keep[-1]]), -1.0, 1.0)
        if dp > dedup_pos_thresh or cos_a < (1.0 - dedup_ang_thresh):
            keep.append(i)
    if keep[-1] != len(positions) - 1:
        keep.append(len(positions) - 1)
    positions = positions[keep]
    forwards = forwards[keep]

    cuts = [0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - positions[i - 1]) > teleport_thresh:
            cuts.append(i)
    cuts.append(len(positions))

    out = []
    last_yaw = None
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 2:
            for p in positions[a:b]:
                y = last_yaw if last_yaw is not None else _yaw_from_forward(
                    forwards[a], height_axis, 0.0)
                c2w = np.eye(4)
                c2w[:3, :3] = _rotation_for_axis(y, height_axis)
                c2w[:3, 3] = p
                out.append(np.linalg.inv(c2w))
                last_yaw = y
        else:
            seg, last_yaw = _smooth_segment(positions[a:b], forwards[a:b], height_axis,
                                            smooth_window, max_yaw_rate, seed_yaw=last_yaw)
            out.extend(seg)
    if eye_height is not None:
        for w in out:
            c = np.linalg.inv(w)
            c[height_axis, 3] = eye_height
            w[:] = np.linalg.inv(c)
    return out


def resample_trajectory(w2cs, num_frames):
    if len(w2cs) <= 1 or num_frames <= 1:
        return w2cs
    pos = np.array([np.linalg.inv(np.asarray(w))[:3, 3] for w in w2cs])
    height_axis = _detect_height_axis(pos)
    fwd = np.array([np.linalg.inv(np.asarray(w))[:3, 2] for w in w2cs])
    yaw = np.array([_yaw_from_forward(f, height_axis, 0.0) for f in fwd])
    yaw = np.unwrap(yaw)
    idx = np.linspace(0, len(w2cs) - 1, num_frames)
    pos_i = np.stack([np.interp(idx, np.arange(len(pos)), pos[:, k]) for k in range(3)], axis=1)
    yaw_i = np.interp(idx, np.arange(len(yaw)), yaw)
    out = []
    for p, y in zip(pos_i, yaw_i):
        c2w = np.eye(4)
        c2w[:3, :3] = _rotation_for_axis(float(y), height_axis)
        c2w[:3, 3] = p
        out.append(np.linalg.inv(c2w))
    return out
