"""Finestra unica che segue la pipeline mentre gira.

A sinistra la vista della camera con sopra le maschere degli oggetti
rilevati; a destra il grafo della scena, che cresce man mano che gli oggetti
vengono mappati e si aggiorna quando un cambiamento li rimuove.

La finestra non blocca la pipeline: viene ridisegnata a ogni frame e il
mapping prosegue.
"""

import cv2
import numpy as np

WINDOW = "DynamicGSG - scena, oggetti, grafo"

_PALETTE = np.array([
    [232, 68, 42], [58, 160, 122], [201, 154, 62], [86, 130, 214],
    [188, 92, 176], [70, 176, 190], [222, 118, 74], [126, 148, 84],
    [176, 74, 108], [96, 116, 200], [214, 164, 96], [64, 148, 152],
], dtype=np.uint8)

_BG = (22, 26, 32)
_FG = (228, 231, 235)
_MUTED = (138, 147, 160)
_ALERT = (60, 68, 232)

_removed_history = []


def _color(idx):
    return tuple(int(c) for c in _PALETTE[idx % len(_PALETTE)])


def _as_uint8_rgb(image):
    """Accetta tensori torch o array numpy, in scala 0-1 o 0-255."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:      # (3,H,W) -> (H,W,3)
        image = np.transpose(image, (1, 2, 0))
    image = np.nan_to_num(image)
    if image.dtype != np.uint8:
        if image.max() <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _iter_detections(detections, classes=None):
    """Normalizza i due formati: DetectionList (lista di dict) o dict di array.

    class_name (nel caso lista di dict) e' gia' risolto in
    process_this_frame_detection col vocabolario giusto -- i tag RAM del
    frame per GroundingDINO, non i 199 nomi fissi di ScanNet200 -- e va
    preferito. Il fallback su classes[cid] resta solo per il formato dict
    legacy (nessun class_name per-detection li') e per detection che ne
    fossero prive.
    """
    if not detections:
        return []
    items = []
    if isinstance(detections, dict):
        masks = detections.get("mask")
        class_ids = detections.get("class_id")
        names = detections.get("classes", classes)
        for i, mask in enumerate(masks if masks is not None else []):
            cid = int(class_ids[i]) if class_ids is not None else i
            items.append((mask, cid, str(names[cid]) if names is not None else None))
    else:
        for i, det in enumerate(detections):
            mask = det.get("mask")
            if mask is None:
                continue
            raw = det.get("class_id")
            cid = int(raw[0]) if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) else i
            name = det.get("class_name")
            if not name:
                name = str(classes[cid]) if classes is not None and cid < len(classes) else None
            items.append((mask, cid, name))
    return items


def _mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _draw_camera_panel(rgb, detections, height, classes=None):
    """Vista camera con maschere e riquadri degli oggetti rilevati."""
    canvas = cv2.cvtColor(_as_uint8_rgb(rgb), cv2.COLOR_RGB2BGR)
    items = _iter_detections(detections, classes)

    if items:
        overlay = canvas.copy()
        for mask, cid, _ in items:
            mask = np.asarray(mask.detach().cpu() if hasattr(mask, "detach") else mask)
            overlay[mask.astype(bool)] = _color(cid)
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

        for mask, cid, name in items:
            mask = np.asarray(mask.detach().cpu() if hasattr(mask, "detach") else mask)
            box = _mask_to_box(mask.astype(bool))
            if box is None:
                continue
            x1, y1, x2, y2 = box
            color = _color(cid)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            if name:
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(canvas, name, (x1 + 3, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    scale = height / canvas.shape[0]
    return cv2.resize(canvas, (int(canvas.shape[1] * scale), height))


def _object_label(obj, classes=None):
    """Nome leggibile: categoria gia' assegnata, altrimenti class_name gia'
    risolto dalla detection, altrimenti (solo fallback legacy) classe
    grezza indicizzata in 'classes', mai un idx nudo.

    Il salto diretto a classes[class_id] (senza passare da class_name) e'
    il bug osservato in produzione: class_id e' un indice nel vocabolario
    RAM/GroundingDINO DI QUEL FRAME (vedi detection_vocab in
    process_this_frame_detection, map_objects_utils_up_with_groupv3.py),
    mentre 'classes' qui e' quasi sempre obj_classes.get_classes_arr(),
    cioe' i 199 nomi fissi di ScanNet200 -- un vocabolario diverso da
    quello che class_id indicizza davvero. Un oggetto appena creato (senza
    ancora 'category') con class_id piccolo finiva quindi rietichettato
    con qualunque nome stia in quella posizione in ScanNet200 ("bag" e'
    alla posizione 3): non e' una detection, e' un indice letto nella
    lista sbagliata. class_name e' gia' risolto correttamente per-oggetto
    in process_this_frame_detection (vedi il suo stesso identico fix,
    commentato li') e va sempre preferito quando c'e'.
    """
    category = obj.get('category')
    if category:
        return str(category)
    class_name = obj.get('class_name')
    if class_name:
        return str(class_name)
    class_id = obj.get('class_id')
    if isinstance(class_id, (list, tuple, np.ndarray)) and len(class_id):
        class_id = class_id[0]
    if classes is not None and class_id is not None:
        try:
            return str(classes[int(class_id)])
        except (ValueError, IndexError, TypeError):
            pass
    return f"oggetto {obj['idx']}"


def _draw_graph_panel(objects, frame_idx, num_gaussians, removed, width, height, classes=None):
    """Grafo della scena: un nodo per oggetto mappato."""
    panel = np.full((height, width, 3), _BG, dtype=np.uint8)

    cv2.putText(panel, "SCENE GRAPH", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _FG, 1, cv2.LINE_AA)
    info = f"frame {frame_idx}   {len(objects)} oggetti   {num_gaussians:,} gaussiani"
    cv2.putText(panel, info, (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _MUTED, 1, cv2.LINE_AA)
    cv2.line(panel, (18, 68), (width - 18, 68), (54, 60, 70), 1)

    # Nodo radice: la scena. Ogni oggetto mappato le si collega.
    root = (width // 2, 100)
    cv2.circle(panel, root, 7, _FG, -1)
    cv2.putText(panel, "scene", (root[0] + 14, root[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _FG, 1, cv2.LINE_AA)

    top, row_h = 132, 26
    max_rows = max(1, (height - top - 60) // row_h)
    shown = objects[:max_rows]

    for i, obj in enumerate(shown):
        y = top + i * row_h
        color = _color(int(obj['idx']))
        cv2.line(panel, (root[0], root[1] + 8), (34, y - 4), (54, 60, 70), 1)
        cv2.circle(panel, (34, y - 4), 5, color, -1)

        label = _object_label(obj, classes)
        cv2.putText(panel, f"[{obj['idx']}] {label}", (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _FG, 1, cv2.LINE_AA)

        det = int(obj.get('num_detections', 0))
        cv2.putText(panel, f"{det} viste", (width - 90, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _MUTED, 1, cv2.LINE_AA)

    if len(objects) > max_rows:
        y = top + max_rows * row_h
        cv2.putText(panel, f"+ altri {len(objects) - max_rows}", (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _MUTED, 1, cv2.LINE_AA)

    if removed:
        _removed_history.append((frame_idx, list(removed)))
    if _removed_history:
        y = height - 46
        cv2.line(panel, (18, y - 18), (width - 18, y - 18), (54, 60, 70), 1)
        cv2.putText(panel, "CAMBIAMENTI NELLA SCENA", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _ALERT, 1, cv2.LINE_AA)
        last_frame, last_ids = _removed_history[-1]
        cv2.putText(panel, f"frame {last_frame}: rimossi {last_ids}", (18, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _FG, 1, cv2.LINE_AA)

    return panel


def show(rgb, detections, objects, frame_idx, num_gaussians, removed=None, classes=None):
    """Ridisegna la finestra. Non blocca: la pipeline prosegue."""
    try:
        height = 520
        left = _draw_camera_panel(rgb, detections, height, classes)
        right = _draw_graph_panel(objects, frame_idx, num_gaussians,
                                  removed, 430, height, classes)
        cv2.imshow(WINDOW, np.hstack([left, right]))
        cv2.waitKey(1)
    except Exception as exc:
        print(f"[live_viewer] {exc}")


def show_change(expected, observed, obj_idx, ssim_score):
    """Confronto mappa attesa / vista reale per un oggetto sotto verifica.

    Sotto la soglia SSIM l'oggetto viene giudicato non piu' al suo posto.
    """
    try:
        def prep(img):
            img = np.clip(np.nan_to_num(img), 0, 1)
            img = (img * 255).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        left, right = prep(expected), prep(observed)
        h = 240
        left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
        right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))

        # "atteso" e "osservato" arrivano con aspect ratio diverso (il primo
        # e' spesso un ritaglio stretto attorno al solo oggetto, il secondo
        # il frame intero della camera): scalarli solo alla stessa altezza,
        # come sopra, lascia ognuno alla propria larghezza naturale -- il
        # riquadro piu' stretto finisce addossato al bordo dell'hstack
        # invece che centrato sotto la sua etichetta, quindi il confronto
        # visivo (dov'e' l'oggetto in un pannello rispetto all'altro) si
        # rompe ogni volta che le due larghezze non coincidono per caso.
        # Si compone invece su una tela comune, ogni pannello centrato nella
        # sua meta': l'inquadratura resta centrata indipendentemente da
        # quanto sono state ridotte le immagini in ingresso.
        panel_w = max(left.shape[1], right.shape[1])
        canvas_w = panel_w * 2

        def _center_on(img, width):
            canvas = np.full((h, width, 3), _BG, dtype=np.uint8)
            x0 = (width - img.shape[1]) // 2
            canvas[:, x0:x0 + img.shape[1]] = img
            return canvas

        left = _center_on(left, panel_w)
        right = _center_on(right, panel_w)
        pair = np.hstack([left, right])
        header = np.full((44, pair.shape[1], 3), _BG, dtype=np.uint8)
        moved = ssim_score < 0.15
        cv2.putText(header, f"oggetto {obj_idx}   SSIM {ssim_score:.3f}", (12, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _FG, 1, cv2.LINE_AA)
        cv2.putText(header,
                    "spostato -> rimosso dal grafo" if moved else "confermato al suo posto",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    _ALERT if moved else _MUTED, 1, cv2.LINE_AA)

        labels = np.full((26, pair.shape[1], 3), _BG, dtype=np.uint8)
        cv2.putText(labels, "atteso dalla mappa", (12, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _MUTED, 1, cv2.LINE_AA)
        # panel_w, non left.shape[1]: dopo la centratura ogni pannello
        # occupa esattamente meta' della tela, a prescindere dalla
        # larghezza naturale delle due immagini in ingresso.
        cv2.putText(labels, "visto dalla camera", (panel_w + 12, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _MUTED, 1, cv2.LINE_AA)

        cv2.imshow("DynamicGSG - verifica cambiamento",
                   np.vstack([header, labels, pair]))
        cv2.waitKey(1)
    except Exception as exc:
        print(f"[live_viewer] {exc}")


def close():
    try:
        cv2.destroyWindow(WINDOW)
    except Exception:
        pass
