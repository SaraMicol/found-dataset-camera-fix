#!/usr/bin/env python3

"""Registra una sequenza RGB-D + pose stile Replica CHE CONTIENE UN CAMBIAMENTO
DINAMICO REALE della scena, eseguendo uno script di esperimento FOUND
(scripts/generated/household_experiments_scene_*.json) mentre la camera lo
segue camminando lungo un percorso navigabile tra un oggetto e il successivo,
invece di teletrasportarsi: la sequenza risultante e' una traiettoria continua
attraverso la scena, non una serie di scatti isolati.

Esegue davvero le azioni spawn/move/remove pubblicandole sui topic ROS del
nodo, cosi' gli oggetti si spostano fisicamente nella scena tra un gruppo di
frame e il successivo. Il dataset ha quindi una prima parte "statica" (dopo
tutti gli spawn) e una seconda parte dove alcuni oggetti sono stati spostati
o rimossi: esattamente cio' che serve a Dynamic-GSG per esercitare
l'aggiornamento dinamico dello scene graph (whether_to_update=True,
frame_begin_update=<primo frame dopo gli spawn>).

Non modifica il rendering: usa gli stessi topic del runner originale
(script_runner.py) per pilotare gli oggetti, e /habitat/find_path +
/habitat/set_agent_pose (aggiunti ad hoc in habitat_camera_objects_node.py)
per muovere la camera con continuita' lungo la navmesh.

Uso:
    # terminale 1 (ambiente Habitat/ROS2), scena scelta:
    export HABITAT_SCENE=.../00824-Dd4bFSTQ8gi.basis.glb
    python3 habitat_camera_objects_node.py

    # terminale 2:
    python3 record_dynamic_sequence.py \
        --out /home/vodka/dynamic-gsg/data/FOUND/00824_dynamic \
        --script scripts/generated/household_experiments_scene_824.json \
        --frames-per-waypoint 4 --step-m 0.35
"""

import argparse
import json
import time
import uuid
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from PIL import Image as PILImage


def quat_to_rotmat(qx, qy, qz, qw):
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


# Quota assoluta degli occhi di una persona in piedi (m da terra). Non e'
# un offset sommato sopra l'oggetto: e' fissa, come lo sono davvero gli occhi
# di chi guarda un tavolo o un mobile, che siano bassi o alti. Il campo
# "capture_eye" dello script FOUND da' gia' la direzione/distanza corrette
# (X/Z) verso l'oggetto; qui si sostituisce solo la Y, cosi' l'inquadratura
# finale somiglia a quella di un umano che si affaccia sulla scena, con
# un'inclinazione moderata verso il basso invece che a picco (con
# sensor_height=1.5 sommato sopra un capture_eye gia' vicino all'oggetto la
# camera finiva a 2.2-2.9 m, inclinata di ~63 gradi: quasi vista da drone).
HUMAN_EYE_HEIGHT_M = 1.55


def _human_eye(capture_eye):
    """capture_eye dello script, con la quota sostituita da HUMAN_EYE_HEIGHT_M."""
    return [capture_eye[0], HUMAN_EYE_HEIGHT_M, capture_eye[2]]


class DynamicRecorderNode(Node):
    def __init__(self, out_dir: Path, walk_pause_s: float = 0.25):
        super().__init__("dynamic_sequence_recorder")
        # Pausa tra un passo e il successivo durante _walk_to: scandisce il
        # ritmo del cammino "in tempo reale", indipendentemente da step_m
        # (che invece scandisce quanti frame vengono salvati per metro).
        self.walk_pause_s = walk_pause_s
        self.out_dir = out_dir
        self.results_dir = out_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._latest_rgb = None
        self._latest_depth = None
        self.create_subscription(Image, "/camera/rgb", self._on_rgb, qos)
        self.create_subscription(Image, "/camera/depth", self._on_depth, qos)
        self.set_pose_pub = self.create_publisher(String, "/habitat/set_agent_pose", qos)

        self.find_path_pub = self.create_publisher(String, "/habitat/find_path", qos)
        self.create_subscription(
            String, "/habitat/find_path_result", self._on_find_path_result, qos
        )
        self._path_results = {}

        self.spawn_pub = self.create_publisher(String, "/habitat/spawn_object", qos)
        self.move_pub = self.create_publisher(String, "/habitat/set_object_position", qos)
        self.remove_pub = self.create_publisher(String, "/habitat/remove_object", qos)
        self.results_queue = Queue()
        self.create_subscription(
            String, "/habitat/object_command_result", self._on_object_result, qos
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.frame_idx = 0
        self.poses = []
        self.frame_begin_update = None
        self.object_ids = {}
        self.current_position = None  # ultima posizione nota della camera
        # Ultimo (capture_eye, position) noto di ogni oggetto: lo script da'
        # solo la posizione di DESTINAZIONE per move/remove, non quella di
        # origine. Tenendo qui il capture_eye dell'ultimo spawn/move su
        # quell'oggetto, prima di un nuovo move/remove si puo' tornare a
        # guardare da dove l'oggetto era prima che il comando lo cambiasse:
        # e' cosi' che si ottiene il frame "prima" oltre a quello "dopo",
        # dallo stesso punto di vista.
        self.last_capture = {}

    def _on_rgb(self, msg: Image):
        if msg.encoding != "rgb8":
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        self._latest_rgb = arr.reshape(msg.height, msg.width, 3)

    def _on_depth(self, msg: Image):
        if msg.encoding != "32FC1":
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
        self._latest_depth = arr.reshape(msg.height, msg.width).copy()

    def _on_object_result(self, msg: String):
        try:
            self.results_queue.put(json.loads(msg.data))
        except json.JSONDecodeError:
            pass

    def _on_find_path_result(self, msg: String):
        try:
            result = json.loads(msg.data)
            self._path_results[result.get("request_id")] = result
        except json.JSONDecodeError:
            pass

    def _wait_result(self, action: str, request_id: str, timeout: float = 8.0):
        deadline = time.monotonic() + timeout
        deferred = []
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                result = self.results_queue.get_nowait()
            except Empty:
                continue
            if result.get("action") == action and result.get("request_id") == request_id:
                for other in deferred:
                    self.results_queue.put(other)
                return result
            deferred.append(result)
        for other in deferred:
            self.results_queue.put(other)
        return {"success": False, "message": "timeout"}

    def _find_path(self, start, end, timeout: float = 5.0):
        request_id = uuid.uuid4().hex
        self.find_path_pub.publish(String(data=json.dumps({
            "start": list(start), "end": list(end), "request_id": request_id,
        })))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if request_id in self._path_results:
                return self._path_results.pop(request_id)
        return {"success": False}

    def _get_c2w(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "habitat_camera_optical", rclpy.time.Time()
            )
        except Exception as exc:
            self.get_logger().warn(f"TF non disponibile: {exc}")
            return None
        tr = t.transform.translation
        q = t.transform.rotation
        R = quat_to_rotmat(q.x, q.y, q.z, q.w)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = [tr.x, tr.y, tr.z]
        return c2w

    def _send_set_pose(self, eye, target):
        msg = String()
        msg.data = json.dumps({"eye": list(eye), "target": list(target)})
        self.set_pose_pub.publish(msg)

    def _save_frame(self):
        c2w = self._get_c2w()
        if c2w is None or self._latest_rgb is None or self._latest_depth is None:
            return False
        color_path = self.results_dir / f"frame{self.frame_idx:06d}.jpg"
        depth_path = self.results_dir / f"depth{self.frame_idx:06d}.png"
        PILImage.fromarray(self._latest_rgb.copy(), mode="RGB").save(color_path, quality=95)
        depth_scaled = np.clip(self._latest_depth.copy() * 6553.5, 0, 65535).astype(np.uint16)
        PILImage.fromarray(depth_scaled, mode="I;16").save(depth_path)
        self.poses.append(c2w)
        self.frame_idx += 1
        return True

    def _walk_to(self, target_eye, look_at, step_m: float, settle_seconds: float):
        """Cammina dalla posizione corrente a target_eye lungo la navmesh,
        catturando un frame ogni step_m metri. Continuita' della traiettoria
        invece del salto secco di un teletrasporto diretto."""
        target_eye = np.asarray(target_eye, dtype=np.float64)
        look_at = np.asarray(look_at, dtype=np.float64)

        start = self.current_position if self.current_position is not None else target_eye
        start_y, end_y = start[1], target_eye[1]

        if self.current_position is None:
            waypoints = [target_eye]
        else:
            path = self._find_path(self.current_position, target_eye)
            if path.get("success") and len(path.get("points", [])) >= 2:
                waypoints = [np.asarray(p) for p in path["points"]]
            else:
                # Nessun percorso navigabile: meglio un salto diretto che
                # bloccare la registrazione.
                self.get_logger().warn("find_path fallito, salto diretto")
                waypoints = [self.current_position, target_eye]

        # I punti della navmesh sono all'altezza del pavimento: la camera
        # deve restare alla sua quota, interpolata linearmente lungo il
        # percorso (proiezione X/Z presa dalla navmesh, Y dalla camera).
        total_len = sum(
            np.linalg.norm(np.asarray(b)[[0, 2]] - np.asarray(a)[[0, 2]])
            for a, b in zip(waypoints[:-1], waypoints[1:])
        ) or 1e-6
        walked = 0.0
        fixed_waypoints = []
        for i, wp in enumerate(waypoints):
            if i > 0:
                walked += np.linalg.norm(
                    np.asarray(wp)[[0, 2]] - np.asarray(waypoints[i - 1])[[0, 2]]
                )
            y = start_y + (end_y - start_y) * (walked / total_len)
            fixed_waypoints.append(np.array([wp[0], y, wp[2]]))
        waypoints = fixed_waypoints

        # Ricampiona il percorso spezzato a passi regolari di step_m.
        dense = [waypoints[0]]
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            seg = np.asarray(b) - np.asarray(a)
            length = np.linalg.norm(seg)
            n_steps = max(1, int(length / step_m))
            for i in range(1, n_steps + 1):
                dense.append(np.asarray(a) + seg * (i / n_steps))

        # Direzione orizzontale di marcia UNICA per tutto il segmento, presa
        # dal primo all'ultimo punto sul piano X/Z. Ricalcolarla ad ogni
        # passo dal punto successivo (come in una versione precedente) la
        # rende instabile appena due punti consecutivi sono quasi allineati
        # in verticale: una variazione infinitesima di rumore puo' far
        # ribaltare di 180 gradi la direzione "destra" della camera nel nodo
        # (vedi _handle_set_agent_pose), dando l'effetto di un frame con il
        # mondo capovolto rispetto al precedente.
        overall_horiz = np.array([
            dense[-1][0] - dense[0][0], 0.0, dense[-1][2] - dense[0][2],
        ])
        has_horiz_dir = np.linalg.norm(overall_horiz) > 1e-3

        for i, point in enumerate(dense):
            is_last = i == len(dense) - 1
            if is_last or not has_horiz_dir:
                facing = look_at
            else:
                facing = point + overall_horiz
            self._send_set_pose(point, facing)
            # Un frame ogni passo: e' cio' che rende la traiettoria continua
            # invece che un salto secco tra un waypoint e il successivo.
            # self.walk_pause_s scandisce il ritmo del cammino: piu' alto e'
            # piu' lenta la camminata "in tempo reale" (non solo nel numero
            # di frame, che dipende da step_m).
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(self.walk_pause_s)
            self._save_frame()

        self.current_position = target_eye

        deadline = time.monotonic() + settle_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _capture_here(self, frames: int):
        for _ in range(frames):
            rclpy.spin_once(self, timeout_sec=0.1)
            self._save_frame()
            time.sleep(0.1)

    def _wait_first_frame(self, timeout_s: float = 15.0):
        deadline = time.monotonic() + timeout_s
        while (self._latest_rgb is None or self._latest_depth is None) and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_rgb is None or self._latest_depth is None:
            raise RuntimeError("Nessun frame ricevuto: il nodo Habitat e' in esecuzione?")

    def _spawn(self, step, frames_per_waypoint, settle_seconds, step_m):
        name = step["name"]
        has_capture = bool(step.get("capture_eye") and step.get("position"))

        # PRIMA: arriva sul punto di ripresa e cattura la scena COSI' COM'E',
        # senza l'oggetto -- lo spawn e' l'unico cambiamento (a differenza di
        # move/remove) per cui "prima" non viene da uno step precedente sullo
        # stesso oggetto: e' semplicemente il posto vuoto, ripreso prima di
        # pubblicare il comando che lo riempie.
        if has_capture:
            self._walk_to(_human_eye(step["capture_eye"]), step["position"], step_m, settle_seconds)
            self._capture_here(frames_per_waypoint)

        request_id = uuid.uuid4().hex
        payload = {"template": step["template"], "request_id": request_id}
        if "position" in step:
            payload["position"] = step["position"]
        if step.get("visual_surface_validated"):
            payload["visual_surface_validated"] = True
        self.spawn_pub.publish(String(data=json.dumps(payload)))
        result = self._wait_result("spawn", request_id)
        if not result.get("success"):
            self.get_logger().warn(f"spawn fallito per {name}: {result}")
            return
        self.object_ids[name] = int(result["object_id"])

        # DOPO: camera gia' li' (nessun _walk_to: l'oggetto e' comparso
        # esattamente dove la si sta gia' guardando), basta aspettare che la
        # fisica si assesti e catturare.
        if has_capture:
            self._capture_here(frames_per_waypoint)
            # Ricordato per poter tornare qui prima del prossimo move/remove
            # su questo oggetto e catturare il suo stato "prima".
            self.last_capture[name] = (step["capture_eye"], step["position"])

    def _move(self, step, frames_per_waypoint, settle_seconds, step_m):
        name = step.get("object")
        object_id = self.object_ids.get(name)
        if object_id is None:
            self.get_logger().warn(f"move: oggetto sconosciuto {name}")
            return

        # PRIMA: torna a guardare l'oggetto dove si trova ancora (posizione
        # precedente, dallo spawn o dall'ultimo move), cosi' il dataset ha un
        # frame chiaro dello stato originale oltre a quello nuovo.
        prev = self.last_capture.get(name)
        if prev is not None:
            prev_eye, prev_position = prev
            self._walk_to(_human_eye(prev_eye), prev_position, step_m, settle_seconds)
            self._capture_here(frames_per_waypoint)

        # frame_begin_update va segnato qui, non prima: i frame "prima" appena
        # catturati mostrano ancora la scena statica (l'oggetto non si e'
        # mosso), quindi non fanno parte della fase dinamica. Il primo frame
        # dinamico e' quello immediatamente successivo al comando move.
        if self.frame_begin_update is None:
            self.frame_begin_update = self.frame_idx
            self.get_logger().info(
                f"--- fine fase statica: frame_begin_update={self.frame_begin_update} ---"
            )

        request_id = uuid.uuid4().hex
        payload = {"object_id": object_id, "position": step["position"], "request_id": request_id}
        if step.get("visual_surface_validated"):
            payload["visual_surface_validated"] = True
        self.move_pub.publish(String(data=json.dumps(payload)))
        result = self._wait_result("move", request_id)
        if not result.get("success"):
            self.get_logger().warn(f"move fallito per {name}: {result}")
            return
        # DOPO: stesso oggetto, nuova posizione.
        if step.get("capture_eye") and step.get("position"):
            self._walk_to(_human_eye(step["capture_eye"]), step["position"], step_m, settle_seconds)
            self._capture_here(frames_per_waypoint)
            self.last_capture[name] = (step["capture_eye"], step["position"])

    def _remove(self, step, frames_per_waypoint, settle_seconds, step_m):
        name = step.get("object")
        object_id = self.object_ids.get(name)
        if object_id is None:
            self.get_logger().warn(f"remove: oggetto sconosciuto {name}")
            return

        # PRIMA: torna a guardare l'oggetto mentre e' ancora li', usando
        # l'ultimo capture_eye noto (spawn o move precedente) -- lo script
        # non da' coordinate per remove, quindi senza questo non c'e' modo
        # di sapere dove guardare per il frame "prima" della rimozione.
        prev = self.last_capture.get(name)
        if prev is not None:
            prev_eye, prev_position = prev
            self._walk_to(_human_eye(prev_eye), prev_position, step_m, settle_seconds)
            self._capture_here(frames_per_waypoint)

        # Vedi il commento equivalente in _move(): se un remove fosse il
        # primo cambiamento della sequenza (non il caso nello script 824,
        # dove i move vengono sempre prima, ma non e' garantito in generale),
        # la fase dinamica comincia qui, non prima dei frame "prima".
        if self.frame_begin_update is None:
            self.frame_begin_update = self.frame_idx
            self.get_logger().info(
                f"--- fine fase statica: frame_begin_update={self.frame_begin_update} ---"
            )

        request_id = uuid.uuid4().hex
        self.remove_pub.publish(String(data=json.dumps({"object_id": object_id, "request_id": request_id})))
        result = self._wait_result("remove", request_id)
        if not result.get("success"):
            self.get_logger().warn(f"remove fallito per {name}: {result}")

    def run(self, script_path: Path, frames_per_waypoint: int, settle_seconds: float,
            remove_frames: int, step_m: float):
        self._wait_first_frame()
        # BUG CORRETTO: qui c'era self.current_position = _get_c2w()[:3, 3].
        # _get_c2w() legge il TF map->habitat_camera_optical, che e' in
        # coordinate ROS (Z-up, vedi habitat_pose_to_ros nel nodo), mentre
        # ovunque nel resto di questo file (_walk_to, _send_set_pose,
        # capture_eye/position dello script) le coordinate sono Habitat
        # (Y-up). Il primo _walk_to interpolava quindi tra un punto in un
        # sistema di assi e uno nell'altro: waypoint assurdi, con la camera
        # che finiva a guardare quasi dritta in su/giu' per gran parte della
        # sequenza (osservato: fwd_z_col~0.98 nella traj, traslazioni fuori
        # scala tipo [-6, -8.8, 2.4] in una scena dove gli oggetti stanno
        # entro pochi metri). current_position resta None: il primo
        # _walk_to prende cosi' il ramo "nessuna posizione nota" e
        # teletrasporta direttamente al primo capture_eye, in coordinate
        # Habitat pure, senza mescolare sistemi.
        data = json.loads(script_path.read_text())

        for step in data["steps"]:
            action = step.get("action")
            if action == "wait":
                continue
            elif action == "spawn":
                self.get_logger().info(f"spawn: {step.get('name')}")
                self._spawn(step, frames_per_waypoint, settle_seconds, step_m)
            elif action == "move":
                # frame_begin_update e' settato dentro _move(), dopo il
                # frame "prima" e prima del comando vero e proprio: vedi il
                # commento li'.
                self.get_logger().info(f"move: {step.get('object')}")
                self._move(step, frames_per_waypoint, settle_seconds, step_m)
            elif action == "remove":
                self.get_logger().info(f"remove: {step.get('object')}")
                self._remove(step, frames_per_waypoint, settle_seconds, step_m)
                if remove_frames > 0:
                    self._capture_here(remove_frames)

        self.write_traj()
        self.write_meta()

    def write_traj(self):
        traj_path = self.out_dir / "traj.txt"
        with open(traj_path, "w") as f:
            for c2w in self.poses:
                f.write(" ".join(f"{v:.8e}" for v in c2w.reshape(-1)) + "\n")
        self.get_logger().info(f"Scritte {len(self.poses)} pose in {traj_path}")

    def write_meta(self):
        meta_path = self.out_dir / "dynamic_meta.json"
        meta_path.write_text(json.dumps({
            "num_frames": self.frame_idx,
            "frame_begin_update": self.frame_begin_update,
        }, indent=2))
        self.get_logger().info(f"frame_begin_update={self.frame_begin_update}, num_frames={self.frame_idx}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--script", required=True)
    # Frame fermi salvati davanti a ogni cambiamento (spawn/move), dopo che
    # la camera e' arrivata al capture_eye dello step: e' la "sosta davanti
    # alla scena esatta". Alzato da 4 a 10 cosi' la sosta e' piu' lunga e
    # piu' facilmente visibile/utilizzabile nel dataset.
    parser.add_argument("--frames-per-waypoint", type=int, default=10)
    # Attesa "morta" (nessun frame salvato) subito dopo l'arrivo, prima di
    # iniziare a salvare i frame fermi: da' tempo alla fisica di assestarsi
    # cosi' i primi frame della sosta non hanno l'oggetto ancora in caduta.
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--remove-frames", type=int, default=6)
    parser.add_argument("--step-m", type=float, default=0.20,
                        help="distanza tra un frame e il successivo mentre si cammina tra i waypoint "
                             "(piu' piccolo = piu' frame di camminata = camminata piu' lenta)")
    # Pausa reale (secondi) tra un passo e il successivo durante il cammino:
    # a differenza di --step-m, questa rallenta il tempo di esecuzione senza
    # aumentare il numero di frame salvati.
    parser.add_argument("--walk-pause-s", type=float, default=0.25,
                        help="pausa in secondi tra un passo e il successivo mentre si cammina")
    args = parser.parse_args()

    rclpy.init()
    node = DynamicRecorderNode(Path(args.out), walk_pause_s=args.walk_pause_s)
    try:
        node.run(Path(args.script), args.frames_per_waypoint, args.settle_seconds,
                 args.remove_frames, args.step_m)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
