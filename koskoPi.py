import os
import sys
import signal
import argparse
import time
import random
import numpy as np
import cv2

from hailo_platform import (
    HEF, Device, VDevice, HailoStreamInterface,
    InferVStreams, ConfigureParams, InputVStreamParams,
    OutputVStreamParams, FormatType,
)
from picamera2 import Picamera2
import requests

from rim_detector import RimDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS = 10
FRAME_SIZE = 640

LABEL_MAP = {0: "ball"}

API_URL = "https://api.kosko.rs/api/logs"
CONFIG_FILE = "/boot/overlays/cnf.txt"
SCORES_FILE = "scores.txt"

# How often (in frames) to re-check rim detection once locked
RIM_RECHECK_INTERVAL = 1000
# How often (in frames) to push scores to the server
UPLOAD_INTERVAL = 90000

# Graceful-shutdown flag (set by SIGTERM / SIGINT)
_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Config / network helpers
# ---------------------------------------------------------------------------


def load_config(path):
    """Read key:value config from *path*.  Returns a dict."""
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                config[k.strip()] = v.strip()

    if "I" not in config:
        raise RuntimeError("Missing device ID (I) in cnf.txt")
    config.setdefault("P", "0")
    return config


def send_scores(filename, url, device_id):
    """POST current score to the API, reset local file on success."""
    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        p_value = "0"
        for line in lines:
            if line.startswith("P:"):
                p_value = line.split(":", 1)[1].strip()
                break

        payload = {
            "id": device_id,
            "logs": [{"id": device_id, "points": int(p_value)}],
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=10)

        if resp.status_code == 200:
            _write_scores(filename, device_id, 0, force_sync=True)
            return True

        print(f"Score upload failed: {resp.status_code} {resp.text}")
    except Exception as exc:
        print(f"Score upload error: {exc}")
    return False


def _write_scores(filename, device_id, count, force_sync=False):
    """Write I:/P: score file.  Only fsync when force_sync is True
    (e.g. after upload reset) — avoids blocking SD-card flushes on
    every single basket which can cost 10-50 ms each."""
    with open(filename, "w") as f:
        f.write(f"I:{device_id}\nP:{count}\n")
        f.flush()
        if force_sync:
            os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# BasketDetector
# ---------------------------------------------------------------------------

class BasketDetector:
    """Detects baskets from a top-down camera by tracking ball movement
    through the rim zone."""

    STRICTNESS_PRESETS = {
        "very_lenient": {
            "rim_zone_scale": 0.8,
            "min_confidence": 0.4,
            "area_change_threshold": 0.2,
            "min_positions": 2,
            "basket_interval": 1.0,
            "max_positions": 6,
            "min_velocity": 5,
            "max_velocity": 150,
        },
        "lenient": {
            "rim_zone_scale": 0.7,
            "min_confidence": 0.45,
            "area_change_threshold": 0.25,
            "min_positions": 2,
            "basket_interval": 1.0,
            "max_positions": 6,
            "min_velocity": 6,
            "max_velocity": 140,
        },
        "medium": {
            "rim_zone_scale": 0.6,
            "min_confidence": 0.5,
            "area_change_threshold": 0.3,
            "min_positions": 3,
            "basket_interval": 1.2,
            "max_positions": 8,
            "min_velocity": 8,
            "max_velocity": 130,
        },
    }

    def __init__(self, rim_center, rim_radius, device_id, strictness="very_lenient"):
        self.device_id = device_id
        self.rim_center_x, self.rim_center_y = rim_center
        self.rim_radius_x, self.rim_radius_y = rim_radius
        self.ball_trackers = {}
        self.last_basket_time = 0.0
        self.basket_count = 0
        self.ball_crossed_up = False
        self._set_strictness(strictness)

    def _set_strictness(self, level):
        if level not in self.STRICTNESS_PRESETS:
            raise ValueError(
                f"Unknown strictness '{level}'. "
                f"Choose from: {list(self.STRICTNESS_PRESETS)}"
            )
        self.settings = self.STRICTNESS_PRESETS[level]
        self.MAX_POSITIONS = self.settings["max_positions"]

    # ----- tracker management -----

    def _get_nearest_tracker(self, pos):
        min_dist = float("inf")
        nearest = None
        for tid, t in self.ball_trackers.items():
            if t["positions"]:
                last = t["positions"][-1][0]
                d = ((pos[0] - last[0]) ** 2 + (pos[1] - last[1]) ** 2) ** 0.5
                if d < min_dist:
                    min_dist = d
                    nearest = tid

        if min_dist < 140:
            return nearest

        new_id = max(self.ball_trackers.keys(), default=-1) + 1
        self.ball_trackers[new_id] = {
            "positions": [],
            "was_above_rim": False,
            "color": tuple(int(np.random.randint(50, 255)) for _ in range(3)),
        }
        return new_id

    # ----- geometry helpers -----

    def _in_rim_zone(self, pos):
        dx = (pos[0] - self.rim_center_x) / self.rim_radius_x
        dy = (pos[1] - self.rim_center_y) / self.rim_radius_y
        return (dx * dx + dy * dy) <= self.settings["rim_zone_scale"]

    @staticmethod
    def _ball_area(box):
        w = abs(box[3] - box[1])
        h = abs(box[2] - box[0])
        return w * h * 2.0

    def _velocity(self, positions):
        if len(positions) < 2:
            return 0.0
        p1, p2 = positions[-2][0], positions[-1][0]
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    def _valid_basket_movement(self, tracker):
        positions = tracker["positions"]
        if len(positions) < self.settings["min_positions"]:
            return False
        v = self._velocity(positions)
        if v < self.settings["min_velocity"] or v > self.settings["max_velocity"]:
            return False
        areas = [p[1] for p in positions]
        if areas[0] <= 0:
            return False
        change = (areas[-1] - areas[0]) / areas[0]
        return change <= -self.settings["area_change_threshold"]

    # ----- main per-frame entry point -----

    def detect_basket(self, frame, results, current_count, draw=True):
        now = time.time()
        active = set()
        h, w = frame.shape[:2]

        n = int(results["num_detections"][0])
        for i in range(n):
            cls = int(results["detection_classes"][0][i])
            conf = float(results["detection_scores"][0][i])
            if cls != 0 or conf < self.settings["min_confidence"]:
                continue

            box = results["detection_boxes"][0][i]
            area = self._ball_area(box)
            if area < 0.05 or area > 0.2:
                continue

            cx = int((box[1] + box[3]) * w / 2)
            cy = int((box[0] + box[2]) * h / 2)
            center = (cx, cy)

            tid = self._get_nearest_tracker(center)
            active.add(tid)
            trk = self.ball_trackers[tid]

            trk["positions"].append((center, area))
            if len(trk["positions"]) > self.MAX_POSITIONS:
                trk["positions"].pop(0)

            # Check if ball has been near rim centre (proxy for "above rim")
            if not trk["was_above_rim"]:
                dx = cx - self.rim_center_x
                dy = cy - self.rim_center_y
                if (dx * dx + dy * dy) ** 0.5 < self.rim_radius_x * 0.6:
                    trk["was_above_rim"] = True

            # Basket decision
            if (
                len(trk["positions"]) >= self.settings["min_positions"]
                and self._in_rim_zone(center)
                and self._valid_basket_movement(trk)
                and trk["was_above_rim"]
                and (now - self.last_basket_time) > self.settings["basket_interval"]
            ):
                current_count += 1
                self.last_basket_time = now
                trk["was_above_rim"] = False

                _write_scores(SCORES_FILE, self.device_id, current_count)

                if draw:
                    cv2.putText(
                        frame, "BASKET!",
                        (int(self.rim_center_x) - 50, int(self.rim_center_y) - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 4,
                    )

            # Visualisation (only when someone is watching)
            if draw:
                color = trk.get("color", (0, 255, 0))
                cv2.circle(frame, center, 5, color, -1)
                if len(trk["positions"]) >= 2:
                    pts = np.array(
                        [p[0] for p in trk["positions"]], dtype=np.int32
                    ).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], False, color, 2)

        # Remove stale trackers
        for tid in set(self.ball_trackers) - active:
            del self.ball_trackers[tid]

        return current_count, frame


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def extract_detections(inputs):
    boxes, scores, classes = [], [], []
    n = 0
    for cls_idx, det_array in enumerate(inputs):
        if len(det_array) == 0:
            continue
        for det in det_array:
            boxes.append(det[:4])
            scores.append(det[4])
            classes.append(cls_idx)
            n += 1
    return {
        "detection_boxes": [boxes],
        "detection_classes": [classes],
        "detection_scores": [scores],
        "num_detections": [n],
    }


def post_nms_infer(raw, output_name):
    return extract_detections(raw[output_name][0])


def draw_boxes(frame, results, used_colors):
    h, w = frame.shape[:2]
    for i in range(int(results["num_detections"][0])):
        cls = int(results["detection_classes"][0][i])
        conf = float(results["detection_scores"][0][i])
        if conf < 0.3:
            continue
        if cls not in used_colors:
            used_colors[cls] = tuple(random.randint(0, 255) for _ in range(3))

        box = results["detection_boxes"][0][i]
        p1 = (round(box[1] * w), round(box[0] * h))
        p2 = (round(box[3] * w), round(box[2] * h))
        c = used_colors[cls]
        cv2.rectangle(frame, p1, p2, c, 2)

        label = LABEL_MAP.get(cls, f"Class {cls}")
        cv2.putText(
            frame, f"{label}: {conf:.2f}",
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2,
        )
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Basketball basket detector — Hailo + PiCamera2"
    )
    p.add_argument("hef", help="Path to HEF model file")
    p.add_argument("output_video", nargs="?", default=None,
                   help="Optional output video path (mp4)")
    p.add_argument("--show", action="store_true",
                   help="Display live window (requires a connected monitor)")
    p.add_argument("--strictness", default="very_lenient",
                   choices=list(BasketDetector.STRICTNESS_PRESETS),
                   help="Basket detection strictness")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    global _shutdown

    config = load_config(CONFIG_FILE)
    device_id = config["I"]
    args = parse_args()

    if not os.path.exists(args.hef):
        raise FileNotFoundError(f"HEF file not found: {args.hef}")

    # When no monitor and no recording, skip all cv2 drawing calls
    need_display = args.show or args.output_video is not None

    # ---- Camera ----
    picam2 = Picamera2()
    cam_cfg = picam2.create_preview_configuration(
        main={"size": (1640, 1232), "format": "RGB888"}
    )
    picam2.configure(cam_cfg)
    picam2.start()

    # ---- Optional video writer ----
    video_writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.output_video, fourcc, FPS, (FRAME_SIZE, FRAME_SIZE)
        )
        if not video_writer.isOpened():
            print("Warning: video writer failed to open, disabling recording.")
            video_writer.release()
            video_writer = None

    # ---- Hailo setup ----
    devices = Device.scan()
    if not devices:
        raise RuntimeError("No Hailo devices found")

    hef = HEF(args.hef)
    output_info = hef.get_output_vstream_infos()
    output_name = output_info[0].name

    rim_detector = RimDetector(debug_mode=False)
    basket_count = 0
    rim_locked = False
    detector = None
    ellipse = None
    total_frames = 0
    frames_since_rim_check = 0
    used_colors = {}

    try:
        with VDevice(device_ids=devices) as target:
            cfg_params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe
            )
            net_group = target.configure(hef, cfg_params)[0]
            net_params = net_group.create_params()

            in_params = InputVStreamParams.make_from_network_group(
                net_group, quantized=False, format_type=FormatType.UINT8
            )
            out_params = OutputVStreamParams.make_from_network_group(
                net_group, quantized=False, format_type=FormatType.FLOAT32
            )

            input_name = hef.get_input_vstream_infos()[0].name

            with InferVStreams(net_group, in_params, out_params) as pipeline:
                pipeline.set_nms_iou_threshold(1.0)
                pipeline.set_nms_score_threshold(0.5)

                with net_group.activate(net_params):
                    while not _shutdown:
                        # --- Capture & pre-process ---
                        frame = picam2.capture_array()
                        # PiCamera2 delivers RGB; OpenCV (and the rim detector) expect BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

                        total_frames += 1
                        frames_since_rim_check += 1

                        # --- Always run inference ---
                        input_data = {
                            input_name: np.expand_dims(frame, axis=0).astype(np.uint8)
                        }
                        raw = pipeline.infer(input_data)
                        results = post_nms_infer(raw, output_name)
                        if need_display:
                            frame = draw_boxes(frame, results, used_colors)
                            cv2.putText(
                                frame, "Masinski fakultet u Nisu",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA,
                            )

                        # --- Rim detection (periodic or until locked) ---
                        if frames_since_rim_check >= RIM_RECHECK_INTERVAL or not rim_locked:
                            frames_since_rim_check = 0
                            det = rim_detector.detect_rim(frame)
                            if det is not None:
                                ellipse = det
                                if not rim_locked:
                                    (cx, cy), (maj, mn), _ = det
                                    rim_locked = True
                                    rim_center = (cx, cy)
                                    rim_radius = (maj / 2.0, mn / 2.0)

                        # --- Initialise basket detector once rim is locked ---
                        if rim_locked and detector is None:
                            print("Initialising BasketDetector …")
                            detector = BasketDetector(
                                rim_center, rim_radius,
                                device_id, args.strictness,
                            )

                        # --- Basket detection ---
                        if rim_locked and detector is not None:
                            if need_display:
                                rim_detector.draw_rim(frame, ellipse)
                            basket_count, frame = detector.detect_basket(
                                frame, results, basket_count,
                                draw=need_display,
                            )

                        # --- Periodic logging ---
                        if total_frames % 10000 == 0:
                            print(
                                f"Frames: {total_frames}  "
                                f"Baskets: {basket_count}"
                            )

                        # --- Periodic score upload ---
                        if total_frames % UPLOAD_INTERVAL == 0:
                            if send_scores(SCORES_FILE, API_URL, device_id):
                                basket_count = 0

                        # --- Optional recording ---
                        if video_writer is not None:
                            video_writer.write(frame)

                        # --- Optional live display ---
                        if args.show:
                            cv2.imshow("Basketball Detector", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

    except Exception as exc:
        print(f"Error: {exc}")
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"Shutdown after {total_frames} frames, {basket_count} baskets.")


if __name__ == "__main__":
    main()
