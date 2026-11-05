from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from feature_extract import apply_crop
from pipeline import PipelineConfig, SceneDetector
from run_video import DEFAULT_SOURCE, OUTPUT_ROOT, resolve_crop

PROBABILITY_ABBR = {
    "falling": "F",
    "fighting": "V",
    "running": "R",
    "normal": "N",
    "sitting": "S",
    "walking": "W",
    "kicking": "K",
    "wall_crossing": "X",
}

LABEL_EN = {
    "奔跑": "Running",
    "摔倒": "Falling",
    "打架": "Fighting",
    "坐": "Sitting",
    "走": "Walking",
    "踢": "Kicking",
    "攀爬/翻越": "Wall crossing",
    "场景": "Scene",
}


def translate_label(label: str) -> str:
    return LABEL_EN.get(label, label)

# 报警横幅亮多久（毫秒）
BANNER_MS = 4000

# 不同动作给不同的响声，不看屏幕也能分辨出是哪种报警。
# 每项是 (频率Hz, 持续毫秒)，频率为 0 表示静音停顿。
BEEP_PATTERNS = {
    "falling": ((1400, 130), (0, 70), (1400, 130), (0, 70), (1400, 200)),
    "fighting": ((950, 220), (0, 70), (950, 220)),
    "running": ((700, 280),),
    "wall_crossing": ((1200, 150), (0, 60), (1200, 220)),
}

HERE = os.path.dirname(os.path.abspath(__file__))
CLIMB_WEIGHTS = os.path.join(HERE, "runs_climb_hn", "best.pt")
CLIMB_THRESHOLD = 0.60
CLIMB_WINDOW_SECONDS = 2.0
CLIMB_STRIDE_SECONDS = 0.50
CLIMB_COOLDOWN_SECONDS = 5.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def play_alarm_sound(action: str) -> None:
    """在后台线程里发蜂鸣声。

    绝不能直接在 UI 线程里调用 winsound.Beep —— 它是阻塞的，
    响一次界面就卡住几百毫秒，报警越密集卡得越明显。
    丢到 daemon 线程里响，界面完全不受影响。
    """
    pattern = BEEP_PATTERNS.get(action, ((1000, 220),))

    def _play() -> None:
        try:
            import winsound

            for frequency, duration in pattern:
                if frequency <= 0:
                    time.sleep(duration / 1000.0)
                else:
                    winsound.Beep(int(frequency), int(duration))
        except Exception:
            # 没有声卡或者不是 Windows 时退化成系统的提示音
            try:
                QApplication.beep()
            except Exception:
                pass

    threading.Thread(target=_play, daemon=True).start()


def to_qimage(frame: np.ndarray) -> QImage:
    """把 OpenCV 的 BGR 画面转成 QImage。

    必须调用 .copy()：QImage 默认只是包了 numpy 的缓冲区指针，
    函数返回后原数组被回收，画面就会花屏或者崩溃。
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(
        rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
    )
    return image.copy()


class InferenceWorker(QThread):
    """独立线程里跑整条流水线。"""

    frame_ready = Signal(object, object)   # 画好的画面, 本次新触发的事件
    status = Signal(object)                # {frame, fps, events, total, live}
    climb_status = Signal(object)          # {name, confidence, alarm}
    climb_event = Signal(object)           # 场景级攀爬事件
    failed = Signal(str)
    done = Signal(str)

    def __init__(
        self,
        source: str,
        config: PipelineConfig,
        crop: str,
        limit: int = 0,
        climb_alarm: bool = True,
    ):
        super().__init__()
        self.source = source
        self.config = config
        self.crop = crop
        self.limit = limit
        self.climb_alarm = climb_alarm
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            detector = SceneDetector(self.config)
            climb_model = None
            climb_frames = 16
            climb_image_size = 224
            climb_history = deque(maxlen=3)
            last_climb_status = {"name": "normal", "confidence": 0.0, "alarm": False}
            last_climb_event = -1e9
            try:
                from train_climb import ClimbClassifier

                if os.path.exists(CLIMB_WEIGHTS):
                    blob = torch.load(
                        CLIMB_WEIGHTS, map_location="cpu", weights_only=False
                    )
                    climb_device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    climb_model = ClimbClassifier(
                        dropout=blob.get("dropout", 0.35)
                    ).to(climb_device)
                    climb_model.load_state_dict(blob["state_dict"])
                    climb_model.eval()
                    climb_frames = int(blob.get("frames", 16))
                    climb_image_size = int(blob.get("image_size", 224))
                    print(
                        f"Loading wall-crossing branch: "
                        f"{os.path.basename(CLIMB_WEIGHTS)} "
                        f"{climb_frames} frames / {climb_image_size}px"
                    )
                else:
                    self.climb_status.emit({"name": "unavailable"})
            except Exception as exc:  # noqa: BLE001
                print(f"Wall-crossing branch disabled: {exc}")
                climb_model = None
                self.climb_status.emit({"name": "unavailable"})
            capture = cv2.VideoCapture(self.source)
            if not capture.isOpened():
                self.failed.emit(f"Cannot open video: {self.source}")
                return
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = 1.0 / max(fps, 1.0)
            frame_index = 0
            event_count = 0
            started = time.time()
            climb_frames_buffer: deque = deque()
            climb_every = max(1, int(round(CLIMB_STRIDE_SECONDS * fps)))

            while self._running:
                ok, frame = capture.read()
                if not ok:
                    break
                if self.crop != "none":
                    frame = apply_crop(frame, self.crop)

                tick = time.time()
                annotated, events = detector.process(frame, frame_index, fps)
                if climb_model is not None:
                    climb_frames_buffer.append(frame.copy())
                    max_buffer = max(
                        climb_frames,
                        int(round(CLIMB_WINDOW_SECONDS * fps)),
                    )
                    while len(climb_frames_buffer) > max_buffer:
                        climb_frames_buffer.popleft()
                    if (
                        frame_index % climb_every == 0
                        and len(climb_frames_buffer) >= min(8, climb_frames)
                    ):
                        picks = np.linspace(
                            0, len(climb_frames_buffer) - 1, climb_frames
                        ).astype(int)
                        sampled = [
                            cv2.resize(
                                climb_frames_buffer[index],
                                (climb_image_size, climb_image_size),
                            )
                            for index in picks
                        ]
                        rgb = np.stack(
                            [
                                cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
                                for item in sampled
                            ]
                        ).astype(np.float32)
                        rgb = rgb / 255.0
                        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
                        climb_tensor = torch.from_numpy(
                            rgb.transpose(0, 3, 1, 2)
                        ).unsqueeze(0).float().to(climb_device)
                        with torch.no_grad():
                            with torch.amp.autocast(
                                "cuda", enabled=climb_device.type == "cuda"
                            ):
                                logits = climb_model(climb_tensor)
                            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        best = int(np.argmax(probabilities))
                        best_name = "wall_crossing" if best == 1 else "normal"
                        climb_history.append(best_name)
                        candidate = (
                            "wall_crossing"
                            if list(climb_history).count("wall_crossing")
                            > list(climb_history).count("normal")
                            else "normal"
                        )
                        stable_name = (
                            candidate
                            if len(climb_history) >= 2
                            else best_name
                        )
                        confidence = float(
                            probabilities[1 if stable_name == "wall_crossing" else 0]
                        )
                        last_climb_status = {
                            "name": stable_name,
                            "confidence": confidence,
                            "alarm": stable_name == "wall_crossing"
                            and confidence >= CLIMB_THRESHOLD,
                        }
                        self.climb_status.emit(last_climb_status)
                        timestamp = frame_index / fps
                        if (
                            last_climb_status["alarm"]
                            and self.climb_alarm
                            and timestamp - last_climb_event
                            >= CLIMB_COOLDOWN_SECONDS
                        ):
                            last_climb_event = timestamp
                            climb_event_data = {
                                "time_sec": round(timestamp, 2),
                                "action": "wall_crossing",
                                "action_zh": "Wall crossing",
                                "track_id": "Scene",
                                "confidence": round(confidence, 4),
                            }
                            event_count += 1
                            self.climb_event.emit(climb_event_data)

                    if last_climb_status.get("alarm"):
                        label = (
                            f"Wall crossing  {last_climb_status['confidence']:.2f}"
                        )
                        color = (0, 0, 255)
                    else:
                        label = (
                            f"Wall crossing: Normal  "
                            f"{last_climb_status.get('confidence', 0):.2f}"
                        )
                        color = (80, 160, 80)
                    cv2.rectangle(
                        annotated,
                        (0, annotated.shape[0] - 38),
                        (annotated.shape[1], annotated.shape[0]),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (10, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                event_count += len(events)
                self.frame_ready.emit(annotated, [e.to_dict() for e in events])

                # 每一帧都发实时判定快照，界面上的「此刻模型看到什么」才不会滞后
                elapsed = max(time.time() - started, 1e-6)
                self.status.emit(
                    {
                        "frame": frame_index,
                        "fps": frame_index / elapsed,
                        "events": event_count,
                        "total": total,
                        "live": detector.live_tracks,
                    }
                )
                frame_index += 1
                if self.limit and frame_index >= self.limit:
                    break

                # 按视频原始帧率节流，模拟实时播放；推理比播放慢时自然就不 sleep
                spare = interval - (time.time() - tick)
                if spare > 0:
                    time.sleep(spare)

            capture.release()
            elapsed = max(time.time() - started, 1e-6)
            self.done.emit(
                f"Processed {frame_index} frames at {frame_index/elapsed:.1f} fps, "
                f"{event_count} alerts"
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self, source: str, classifier: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Scene Detection - Six Actions + Wall Crossing")
        self.resize(1180, 720)

        self.source = source
        self.classifier_path = classifier
        self.worker: InferenceWorker | None = None
        self.paused = False
        self._climb_last_name = "normal"

        self._build_ui()
        self._load_source(source)

    # ------------------------------------------------------------------ 界面

    def _build_ui(self) -> None:
        self.banner = QLabel("")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setFixedHeight(38)
        self.banner.setStyleSheet(
            "background:#2b2b2b;color:#dddddd;font-size:15px;border-radius:4px;"
        )

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setStyleSheet("background:#111;color:#888;")

        self.live_label = QLabel("Live detection: (not started)")
        self.live_label.setWordWrap(True)
        self.live_label.setMinimumHeight(130)
        self.live_label.setStyleSheet(
            "background:#1e1e1e;color:#7fd67f;font-family:Consolas,monospace;"
            "font-size:12px;padding:6px;border-radius:4px;"
        )
        self.climb_label = QLabel("Wall crossing: waiting")
        self.climb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.climb_label.setMinimumHeight(42)
        self.climb_label.setStyleSheet(
            "background:#1e1e1e;color:#8fd18f;font-size:15px;"
            "font-weight:bold;border-radius:4px;padding:6px;"
        )

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Time (s)", "Track", "Action", "Confidence"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.open_button = QPushButton("Open Video…")
        self.open_button.clicked.connect(self._choose_file)
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self._toggle_run)
        self.start_button.setEnabled(False)

        self.fighting_check = QCheckBox("Fighting alarm")
        self.fighting_check.setChecked(False)
        self.fighting_check.setToolTip(
            "F1 0.93 on the six-class validation set, but crowded bus scenes "
            "can still cause false positives. Display-only unless enabled."
        )
        self.running_check = QCheckBox("Running alarm")
        self.running_check.setChecked(False)
        self.climb_check = QCheckBox("Wall crossing alarm")
        self.climb_check.setChecked(True)
        self.climb_check.setToolTip(
            "Hard-negative model, Test F1 0.964. Recommended for "
            "dedicated wall/perimeter cameras."
        )

        self.sound_check = QCheckBox("Alert sound")
        self.sound_check.setChecked(True)
        self.sound_check.setToolTip(
            "Falling: three high beeps; fighting: two beeps; "
            "running/wall crossing: one beep. Played in a background thread."
        )

        self.info_label = QLabel("Waiting")
        self.info_label.setStyleSheet("color:#666;")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Live model output (per frame)"))
        note = QLabel("Sitting / Walking / Kicking: display only")
        note.setStyleSheet("color:#888;font-size:12px;")
        right_layout.addWidget(note)
        right_layout.addWidget(self.climb_label)
        right_layout.addWidget(self.live_label)
        right_layout.addSpacing(8)
        right_layout.addWidget(QLabel("Alert events"))
        right_layout.addWidget(self.table, 1)
        right_layout.addWidget(self.fighting_check)
        right_layout.addWidget(self.running_check)
        right_layout.addWidget(self.climb_check)
        right_layout.addWidget(self.sound_check)
        legend = QLabel(
            "Box colors: Red=Falling  Orange=Sitting  Green=Walking  "
            "Purple=Kicking  Yellow=Running\n"
            "Bottom bar: Wall crossing (perimeter camera)"
        )
        legend.setStyleSheet("color:#888;font-size:12px;")
        legend.setWordWrap(True)
        right_layout.addWidget(legend)
        right_layout.addWidget(self.info_label)
        right.setFixedWidth(430)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.start_button)
        controls.addStretch(1)
        controls.addWidget(QLabel("File:"))
        self.path_label = QLabel("—")
        self.path_label.setStyleSheet("color:#666;")
        controls.addWidget(self.path_label)

        left = QVBoxLayout()
        left.addWidget(self.banner)
        left.addWidget(self.video_label, 1)

        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addWidget(right)

        root = QVBoxLayout()
        root.addLayout(body, 1)
        root.addLayout(controls)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.banner_timer = QTimer(self)
        self.banner_timer.setSingleShot(True)
        self.banner_timer.timeout.connect(self._clear_banner)

    def _load_source(self, path: str) -> None:
        self.source = path
        self.path_label.setText(os.path.basename(path) if path else "—")
        self.start_button.setEnabled(bool(path) and os.path.exists(path))
        self.info_label.setText(
            "Ready. Click Start Detection." if path else "Waiting for video"
        )
        if path and path.lower().endswith(".mp4") and "urfall" in path.lower():
            self.info_label.setText(
                "Ready (UR Fall: left depth half will be cropped automatically)"
            )

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            os.path.dirname(DEFAULT_SOURCE),
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self._load_source(path)

    # ------------------------------------------------------------------ 运行

    def _toggle_run(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
            self.start_button.setText("Start Detection")
            self.info_label.setText("Stopped")
            return

        config = PipelineConfig()
        if self.classifier_path:
            config.classifier_path = self.classifier_path
        config.alarm_fighting = self.fighting_check.isChecked()
        config.alarm_running = self.running_check.isChecked()
        crop = resolve_crop(self.source, "auto")

        self.table.setRowCount(0)
        self._climb_last_name = "normal"
        self.banner.setStyleSheet(
            "background:#2b2b2b;color:#dddddd;font-size:15px;border-radius:4px;"
        )
        self.banner.setText("Detecting...")

        self.worker = InferenceWorker(
            self.source,
            config,
            crop,
            climb_alarm=self.climb_check.isChecked(),
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status.connect(self._on_status)
        self.worker.climb_status.connect(self._on_climb_status)
        self.worker.climb_event.connect(self._on_climb_event)
        self.worker.failed.connect(self._on_failed)
        self.worker.done.connect(self._on_done)
        self.worker.start()
        self.start_button.setText("Stop")
        self.info_label.setText("Detecting...")

    def _on_frame(self, frame: np.ndarray, events: list) -> None:
        pixmap = QPixmap.fromImage(to_qimage(frame))
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        for event in events:
            action_text = translate_label(event["action_zh"])
            track_text = translate_label(str(event["track_id"]))
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(f"{event['time_sec']:.2f}"))
            self.table.setItem(row, 1, QTableWidgetItem(track_text))
            self.table.setItem(row, 2, QTableWidgetItem(action_text))
            self.table.setItem(row, 3, QTableWidgetItem(f"{event['confidence']:.3f}"))
            self.table.scrollToBottom()
            self._flash_banner(
                f"Alert: Track {event['track_id']}  {action_text}  "
                f"(confidence {event['confidence']:.2f})"
            )
            if self.sound_check.isChecked():
                play_alarm_sound(event["action"])

    def _on_climb_status(self, info: dict) -> None:
        name = info.get("name")
        previous = self._climb_last_name
        self._climb_last_name = name or "normal"
        if name == "unavailable":
            self.climb_label.setText("Wall crossing: model unavailable")
            self.climb_label.setStyleSheet(
                "background:#1e1e1e;color:#b0b0b0;font-size:15px;"
                "font-weight:bold;border-radius:4px;padding:6px;"
            )
            return
        if name == "wall_crossing":
            self.climb_label.setText(
                f"Wall crossing: detected  {info.get('confidence', 0):.2f}"
            )
            self.climb_label.setStyleSheet(
                "background:#8b1a1a;color:white;font-size:15px;"
                "font-weight:bold;border-radius:4px;padding:6px;"
            )
            if previous != "wall_crossing":
                self._flash_banner(
                    f"Scene alert: Wall crossing  "
                    f"(confidence {info.get('confidence', 0):.2f})"
                )
        else:
            self.climb_label.setText(
                f"Wall crossing: Normal  {info.get('confidence', 0):.2f}"
            )
            self.climb_label.setStyleSheet(
                "background:#1e1e1e;color:#8fd18f;font-size:15px;"
                "font-weight:bold;border-radius:4px;padding:6px;"
            )

    def _on_climb_event(self, event: dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(f"{event['time_sec']:.2f}"))
        self.table.setItem(row, 1, QTableWidgetItem(translate_label(str(event["track_id"]))))
        self.table.setItem(row, 2, QTableWidgetItem(translate_label(event["action_zh"])))
        self.table.setItem(row, 3, QTableWidgetItem(f"{event['confidence']:.3f}"))
        self.table.scrollToBottom()
        self._flash_banner(
            f"Alert: Scene  {translate_label(event['action_zh'])}  "
            f"(confidence {event['confidence']:.2f})"
        )
        if self.sound_check.isChecked():
            play_alarm_sound(event["action"])

    def _flash_banner(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.setStyleSheet(
            "background:#c62828;color:white;font-size:16px;font-weight:bold;"
            "border-radius:4px;"
        )
        self.banner_timer.start(BANNER_MS)

    def _clear_banner(self) -> None:
        self.banner.setText("")
        self.banner.setStyleSheet(
            "background:#2b2b2b;color:#dddddd;font-size:15px;border-radius:4px;"
        )

    def _on_status(self, info: dict) -> None:
        total = info["total"] or "?"
        self.info_label.setText(
            f"Frame {info['frame']}/{total}   {info['fps']:.1f} fps   "
            f"Total alerts: {info['events']}"
        )
        live = info.get("live") or []
        if not live:
            self.live_label.setText("Live detection: no people in frame")
            return
        rows = []
        for track in live[:5]:
            parts = " ".join(translate_label(label) for label in track["labels"])
            detail = " ".join(
                f"{PROBABILITY_ABBR.get(key, key[:1])}{value:.2f}"
                for key, value in track["probabilities"].items()
            )
            marker = "★ " if track["alarm"] else "  "
            rows.append(f"{marker}ID{track['track_id']:<3} {parts}")
            rows.append(f"    {detail}")
        if len(live) > 5:
            rows.append(f"  ... and {len(live)-5} more")
        self.live_label.setText("Live detection:\n" + "\n".join(rows))

    def _on_failed(self, message: str) -> None:
        self.start_button.setText("Start Detection")
        self.info_label.setText("Error")
        QMessageBox.critical(self, "Runtime Error", message)

    def _on_done(self, message: str) -> None:
        self.start_button.setText("Start Detection")
        self.info_label.setText(message)
        self.banner.setText("Detection finished")

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Scene Detection Qt UI")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Initial video")
    parser.add_argument(
        "--classifier",
        default=None,
        help="Default: runs_six_transfer/best.pt; legacy two-class weights are optional",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow(
        args.source if os.path.exists(args.source) else "",
        classifier=args.classifier,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
