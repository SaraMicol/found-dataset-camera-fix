#!/usr/bin/env python3
"""Finestra live su /camera/rgb (+ /camera/depth) del nodo Habitat.

Serve a vedere in tempo reale l'inquadratura mentre record_dynamic_sequence.py
registra: su questa macchina rqt_image_view non e' installato.

GUI in tkinter, non OpenCV: il cv2 dell'env foundgraph e'
opencv-python-headless (build "GUI: NONE"), quindi cv2.imshow() non puo'
funzionare e nessun flag lo abilita. tkinter e' nella stdlib e PIL.ImageTk e'
gia' presente, cosi' non serve toccare l'env conda -- reinstallare un opencv
non-headless rischierebbe di rompere il linkaggio di habitat_sim o di ROS.

Uso (via launcher, che isola ROS):
    ./run_live_view.sh
    ./run_live_view.sh --depth      # affianca anche la depth colorizzata

'q' o ESC per chiudere.
"""

import argparse
import tkinter as tk

import numpy as np
import rclpy
from PIL import Image as PILImage
from PIL import ImageDraw, ImageTk
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def colorize_depth(depth):
    """Depth metrica -> RGB con una rampa tipo turbo, calcolata in numpy.

    Sostituisce cv2.applyColorMap: senza GUI OpenCV resta importabile, ma non
    vale la pena tenerne la dipendenza solo per una lookup table.
    """
    d = depth.astype(np.float32)
    finite = np.isfinite(d) & (d > 0)
    if finite.any():
        lo, hi = np.percentile(d[finite], [2, 98])
        d = np.clip((d - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    else:
        d = np.zeros_like(d)
    d[~finite] = 0.0
    # Rampa scuro -> blu -> ciano -> verde -> giallo -> rosso.
    stops = np.array([
        [48, 18, 59], [70, 134, 251], [27, 229, 181],
        [165, 254, 69], [252, 200, 47], [122, 4, 3],
    ], dtype=np.float32)
    pos = d * (len(stops) - 1)
    idx = np.clip(pos.astype(np.int32), 0, len(stops) - 2)
    frac = (pos - idx)[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    return rgb.astype(np.uint8)


class LiveView(Node):
    def __init__(self, show_depth: bool):
        super().__init__("habitat_live_view")
        self.show_depth = show_depth
        self.rgb = None
        self.depth = None
        self.n_rgb = 0
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Image, "/camera/rgb", self._on_rgb, qos)
        if show_depth:
            self.create_subscription(Image, "/camera/depth", self._on_depth, qos)

    def _on_rgb(self, msg: Image):
        if msg.encoding != "rgb8":
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        self.rgb = arr.reshape(msg.height, msg.width, 3)
        self.n_rgb += 1

    def _on_depth(self, msg: Image):
        if msg.encoding != "32FC1":
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
        self.depth = arr.reshape(msg.height, msg.width)

    def frame(self):
        """Ultimo frame come PIL.Image, o None se non e' ancora arrivato."""
        if self.rgb is None:
            return None
        rgb = self.rgb
        img = PILImage.fromarray(rgb.copy(), mode="RGB")
        if self.show_depth and self.depth is not None:
            dimg = PILImage.fromarray(colorize_depth(self.depth), mode="RGB")
            combined = PILImage.new("RGB", (img.width + dimg.width,
                                            max(img.height, dimg.height)))
            combined.paste(img, (0, 0))
            combined.paste(dimg, (img.width, 0))
            img = combined

        # Mirino al centro della sola RGB: rende evidente dove punta davvero
        # la camera, che e' esattamente cio' che si sta ritarando con
        # HABITAT_SENSOR_HEIGHT.
        draw = ImageDraw.Draw(img)
        cx, cy = rgb.shape[1] // 2, rgb.shape[0] // 2
        draw.line([(cx - 12, cy), (cx + 12, cy)], fill=(0, 255, 0), width=1)
        draw.line([(cx, cy - 12), (cx, cy + 12)], fill=(0, 255, 0), width=1)
        draw.text((8, 8), f"frame {self.n_rgb}", fill=(0, 255, 0))
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", action="store_true",
                    help="mostra anche la depth accanto alla RGB")
    args = ap.parse_args()

    rclpy.init()
    node = LiveView(args.depth)

    root = tk.Tk()
    root.title("habitat live (/camera/rgb)")
    label = tk.Label(root, text="in attesa di frame da /camera/rgb ...",
                     fg="#00ff00", bg="black", width=60, height=20)
    label.pack()
    state = {"running": True, "photo": None}

    def stop(_event=None):
        state["running"] = False

    root.bind("<Escape>", stop)
    root.bind("q", stop)
    root.protocol("WM_DELETE_WINDOW", stop)

    def tick():
        if not state["running"] or not rclpy.ok():
            root.quit()
            return
        rclpy.spin_once(node, timeout_sec=0.0)
        img = node.frame()
        if img is not None:
            # Il riferimento va tenuto vivo: tkinter non possiede la
            # PhotoImage e senza questo l'immagine sparisce subito.
            photo = ImageTk.PhotoImage(img)
            state["photo"] = photo
            label.configure(image=photo, text="", width=0, height=0)
        root.after(30, tick)

    root.after(0, tick)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
