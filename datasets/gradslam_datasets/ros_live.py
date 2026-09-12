"""Dataset "a coda" che alimenta la pipeline dgsg con frame RGB-D+pose
ricevuti in tempo reale dal simulatore Habitat via ROS2, invece che da file
già scritti su disco.

Non eredita da GradSLAMDataset: quella classe base presuppone sempre un
elenco di file risolvibile con glob() al momento della costruzione
(get_filepaths()/load_poses() sono astratti e pensati per quello). Qui
serve solo la STESSA INTERFACCIA (__getitem__, __len__) che il resto della
pipeline usa per accedere ai dati -- duck typing, non ereditarietà.

Punti verificati sul chiamante (scripts/dynamic_gsg_real_ssim.py), che
determinano il contratto di questa classe:
  - dataset[0] viene letto DUE VOLTE all'avvio (riga 679, poi di nuovo
    dentro initialize_first_timestep riga 222) -- __getitem__ deve essere
    idempotente per indici già ricevuti, non consumare un frame nuovo ad
    ogni chiamata.
  - se load_checkpoint=True, dataset[time_idx] viene riletto per indici
    passati (righe 724-726) -- va tenuta una cache di TUTTI i frame
    ricevuti, non solo l'ultimo.
  - ogni elemento restituito deve essere (color, depth, intrinsics, pose):
    color (H,W,3) float raw 0-255 (la pipeline normalizza /255 da sola),
    depth (H,W,1) float in metri, intrinsics (4,4) omogenea con [:3,:3]=K,
    pose (4,4) c2w (la pipeline la inverte sempre con torch.linalg.inv).
  - use_gt_poses=False (config found) MA modify_real_gt_poses=True: la
    pipeline non usa la pose come stima finale (quella la rifinisce col
    tracking), ma la usa per INIZIALIZZARE la posa di ogni frame oltre al
    primo (scripts/dynamic_gsg_real_ssim.py:776-784) -- un placeholder
    fisso (es. identità) farebbe credere alla pipeline che la camera non
    si è mai mossa. Serve la posa reale (c2w), letta dal TF Habitat lato
    chi pilota la camera e inclusa nel messaggio di commit -- stesso
    formato già scritto in traj.txt da
    FOUND-Dataset/record_dynamic_sequence.py::write_traj().
"""

import threading
import time

import numpy as np
import torch


class RosLiveDataset:
    """Non è un torch.utils.data.Dataset: nessun DataLoader/Sampler la usa,
    la pipeline accede sempre per indice diretto (dataset[i]), quindi non
    serve ereditare da Dataset -- solo implementare __getitem__/__len__.
    """

    def __init__(
        self,
        config_dict,
        basedir=None,   # ignorato: nessun file da risolvere, tenuto solo
        sequence=None,  # per chiamata uniforme con get_dataset() (righe
                        # 75-103 di scripts/dynamic_gsg_real_ssim.py, che
                        # passa sempre (config_dict, basedir, sequence, **kwargs))
        rgb_topic: str = "/camera/rgb",
        depth_topic: str = "/camera/depth",
        camera_info_topic: str = "/camera/camera_info",
        commit_topic: str = "/dgsg/committed_frame",
        consumed_topic: str = "/dgsg/frame_consumed",
        device="cuda:0",
        dtype=torch.float,
        frame_timeout_s: float = 120.0,
        **kwargs,
    ):
        # kwargs assorbe start/end/stride/desired_height/desired_width/
        # relative_pose/ignore_bad/use_train_split ecc. che get_dataset()
        # passa sempre (stesso identico kwargs di tutte le altre classi
        # Dataset, get_dataset() non sa quale verrà istanziata) -- questo
        # bridge non fa subsampling né resize: riceve i frame già alla
        # risoluzione pubblicata dal nodo Habitat.
        #
        # num_frames NON è tra i kwargs di get_dataset() (verificato:
        # scripts/dynamic_gsg_real_ssim.py:620-632 non lo passa mai --
        # quel valore è letto separatamente da dataset_config["num_frames"]
        # DOPO aver costruito il dataset, riga 634). Va quindi letto da
        # config_dict stesso, coerente con come le altre classi ci leggono
        # camera_params/png_depth_scale: il config yaml del bridge deve
        # avere la chiave "num_frames" allo stesso livello di "dataset_name".
        self.name = config_dict.get("dataset_name", "ros_live")
        self.device = device
        self.dtype = dtype
        self._num_frames = int(config_dict.get("num_frames", 0))
        self._frame_timeout_s = frame_timeout_s

        # Cache di TUTTI i frame ricevuti: {index: (color, depth, intrinsics, pose)},
        # tensori già pronti nel formato che __getitem__ deve restituire.
        # Necessaria sia per l'idempotenza su dataset[0] (letto due volte)
        # sia per il ramo load_checkpoint che rilegge indici passati.
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._cache_event = threading.Condition(self._cache_lock)

        self._intrinsics = None  # (4,4) letta una volta dal primo CameraInfo

        self._node = None
        self._spin_thread = None
        self._consumed_pub = None
        self._init_ros(rgb_topic, depth_topic, camera_info_topic, commit_topic, consumed_topic)

    # ------------------------------------------------------------------
    # Setup ROS2: un nodo dedicato, spin su un thread separato cosi'
    # __getitem__ puo' bloccare in attesa senza fermare la ricezione dei
    # messaggi (che arrivano su callback, gestiti dallo spin).
    # ------------------------------------------------------------------
    def _init_ros(self, rgb_topic, depth_topic, camera_info_topic, commit_topic, consumed_topic):
        import json as _json

        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init(args=None)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        outer = self
        node = Node("dgsg_ros_live_dataset")

        # Ultimo colore/depth ricevuti dal simulatore: il nodo Habitat
        # pubblica RGB e depth su due topic separati, non sincronizzati a
        # livello di singolo messaggio (stesso pattern gia' usato in
        # FOUND-Dataset/record_dynamic_sequence.py) -- si accoppiano solo
        # quando arriva un "commit" esplicito per un indice.
        state = {"rgb": None, "depth": None}

        def on_rgb(msg: Image):
            if msg.encoding != "rgb8":
                return
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            state["rgb"] = arr.reshape(msg.height, msg.width, 3).copy()

        def on_depth(msg: Image):
            if msg.encoding != "32FC1":
                return
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
            state["depth"] = arr.reshape(msg.height, msg.width).copy()

        def on_camera_info(msg: CameraInfo):
            if outer._intrinsics is not None:
                return  # letti una sola volta, vedi docstring punto 7 del piano
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            intrinsics = np.eye(4, dtype=np.float64)
            intrinsics[:3, :3] = K
            outer._intrinsics = torch.from_numpy(intrinsics)

        def on_commit(msg: String):
            # Lo script di cammino pubblica {"index": N} quando un frame e'
            # pronto (rgb+depth gia' aggiornati in state). Qui si "congela"
            # come dataset[N]: da questo momento __getitem__(N) lo
            # restituisce sempre, anche se il simulatore continua a
            # camminare e state cambia sotto.
            try:
                payload = _json.loads(msg.data)
                index = int(payload["index"])
                pose_flat = payload["c2w"]  # 16 float, stesso formato di traj.txt
                if len(pose_flat) != 16:
                    raise ValueError("c2w deve avere 16 valori (matrice 4x4 appiattita)")
            except (ValueError, KeyError, TypeError) as exc:
                node.get_logger().warn(f"commit malformato ({exc}): {msg.data!r}")
                return
            if state["rgb"] is None or state["depth"] is None:
                node.get_logger().warn(f"commit per frame {index} ma nessun rgb/depth ricevuto ancora")
                return
            if outer._intrinsics is None:
                node.get_logger().warn(f"commit per frame {index} ma /camera/camera_info non ancora ricevuto")
                return

            color = torch.from_numpy(state["rgb"].astype(np.float32))  # (H,W,3), 0-255
            depth = torch.from_numpy(state["depth"].astype(np.float32)).unsqueeze(-1)  # (H,W,1), metri
            # Posa reale (c2w), non un placeholder: con
            # modify_real_gt_poses=True (config found, riga 63) la pipeline
            # inizializza la posa di OGNI frame successivo al primo
            # direttamente dalla GT prima di rifinarla col tracking
            # (scripts/dynamic_gsg_real_ssim.py:776-784) -- un'identita'
            # farebbe credere alla pipeline che la camera non si e' mai
            # mossa, rompendo silenziosamente la geometria. Stesso formato
            # (c2w, frame Habitat) gia' scritto in traj.txt da
            # FOUND-Dataset/record_dynamic_sequence.py::write_traj().
            pose = torch.tensor(pose_flat, dtype=torch.float32).reshape(4, 4)

            item = (
                color.to(outer.device).type(outer.dtype),
                depth.to(outer.device).type(outer.dtype),
                outer._intrinsics.to(outer.device).type(outer.dtype),
                pose.to(outer.device).type(outer.dtype),
            )
            with outer._cache_event:
                outer._cache[index] = item
                outer._cache_event.notify_all()

        node.create_subscription(Image, rgb_topic, on_rgb, qos)
        node.create_subscription(Image, depth_topic, on_depth, qos)
        node.create_subscription(CameraInfo, camera_info_topic, on_camera_info, qos)
        node.create_subscription(String, commit_topic, on_commit, qos)
        consumed_pub = node.create_publisher(String, consumed_topic, qos)

        self._node = node
        self._consumed_pub = consumed_pub
        self._json = _json
        self._String = String

        def spin():
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)

        self._spin_thread = threading.Thread(target=spin, daemon=True)
        self._spin_thread.start()

    # ------------------------------------------------------------------
    # Interfaccia usata dalla pipeline
    # ------------------------------------------------------------------
    def __len__(self):
        return self._num_frames

    def __getitem__(self, index):
        with self._cache_event:
            deadline = time.monotonic() + self._frame_timeout_s
            while index not in self._cache:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"RosLiveDataset: nessun frame per index={index} entro "
                        f"{self._frame_timeout_s}s. Lo script di cammino sta "
                        f"pubblicando su /dgsg/committed_frame?"
                    )
                self._cache_event.wait(timeout=min(remaining, 0.5))
            return self._cache[index]

    def notify_consumed(self, index: int):
        """Da chiamare dopo che time_idx==index e' stato processato per
        intero (tracking+mapping fatti): sblocca lo script di cammino, che
        aspetta questo ack prima di mandare il frame successivo (lock-step
        scelto esplicitamente dall'utente, vedi piano)."""
        msg = self._String()
        msg.data = self._json.dumps({"index": int(index)})
        self._consumed_pub.publish(msg)
