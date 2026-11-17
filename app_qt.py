from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque


def _bootstrap_desktop_launch() -> None:
    """Run this file with the SC environment's pythonw, not the default Python."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        os.path.join(os.environ.get("TEMP", "."), "scene_detect_yolo"),
    )
    os.environ.setdefault(
        "TORCH_HOME",
        os.path.join(project_root, "scene_detect", ".torch_cache"),
    )

    target_python = r"C:\Users\13295\anaconda3\envs\SC\pythonw.exe"
    already_bootstrapped = os.environ.get("SCENE_DETECT_DESKTOP_BOOTSTRAPPED") == "1"
    if os.name == "nt" and os.path.isfile(target_python) and not already_bootstrapped:
        environment = os.environ.copy()
        environment["SCENE_DETECT_DESKTOP_BOOTSTRAPPED"] = "1"
        environment["SCENE_DETECT_AUTOSTART"] = "1"
        subprocess.Popen(
            [target_python, os.path.abspath(__file__), *sys.argv[1:]],
            cwd=project_root,
            env=environment,
            close_fds=True,
            creationflags=subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW,
        )
        raise SystemExit(0)


if __name__ == "__main__":
    _bootstrap_desktop_launch()

import cv2
import numpy as np
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

DEFAULT_VLM_MODEL = r"D:\Models\Qwen2.5-VL-3B-Instruct"
DEFAULT_QT_SOURCE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "scene_all_categories_no_gap.mp4",
    )
)
VLM_ACTIONS = {
    "running",
    "falling",
    "fighting",
    "sitting",
    "walking",
    "kicking",
    "waving",
    "nodding",
    "handshaking",
}

PROBABILITY_ABBR = {
    "falling": "F",
    "fighting": "V",
    "running": "R",
    "normal": "N",
    "sitting": "S",
    "walking": "W",
    "kicking": "K",
    "waving": "WV",
    "nodding": "ND",
    "handshaking": "HS",
}

LABEL_EN = {
    "奔跑": "Running",
    "摔倒": "Falling",
    "打架": "Fighting",
    "坐": "Sitting",
    "走": "Walking",
    "踢": "Kicking",
    "挥手": "Waving",
    "点头": "Nodding",
    "握手": "Handshaking",
}


def translate_label(label: str) -> str:
    return LABEL_EN.get(label, label)


def event_key(event: dict) -> str:
    return (
        f"{event.get('track_id')}:"
        f"{float(event.get('time_sec', 0)):.2f}:"
        f"{event.get('action', '')}"
    )


# 报警横幅亮多久（毫秒）
BANNER_MS = 4000

# 不同动作给不同的响声，不看屏幕也能分辨出是哪种报警。
# 每项是 (频率Hz, 持续毫秒)，频率为 0 表示静音停顿。
BEEP_PATTERNS = {
    "falling": ((1400, 130), (0, 70), (1400, 130), (0, 70), (1400, 200)),
    "fighting": ((950, 220), (0, 70), (950, 220)),
    "running": ((700, 280),),
}


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
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected BGR frame, got {frame.shape}")
    height, width, channels = frame.shape
    image = QImage(
        frame.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_BGR888,
    )
    return image.copy()


class InferenceWorker(QThread):
    """独立线程里跑整条流水线。"""

    frame_ready = Signal(object, object, object)
    status = Signal(object)                # {frame, fps, events, total, live}
    failed = Signal(str)
    done = Signal(str)

    def __init__(
        self,
        source: str,
        config: PipelineConfig,
        crop: str,
        limit: int = 0,
        vlm_confirm: bool = False,
    ):
        super().__init__()
        self.source = source
        self.config = config
        self.crop = crop
        self.limit = limit
        self.vlm_confirm = vlm_confirm
        self._running = True
        directory, filename = os.path.split(source)
        optical_name = filename.replace("_no_gap", "_optical")
        smooth_name = filename.replace("_no_gap", "_smooth")
        candidates = [
            os.path.join(directory, name)
            for name in (optical_name, smooth_name)
            if name != filename
        ]
        self.display_source = next(
            (path for path in candidates if os.path.isfile(path)),
            "",
        )

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            detector = SceneDetector(self.config)
            capture = cv2.VideoCapture(self.source)
            display_capture = (
                cv2.VideoCapture(self.display_source)
                if self.display_source
                else None
            )
            if not capture.isOpened():
                self.failed.emit(f"Cannot open video: {self.source}")
                return
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = 1.0 / max(fps, 1.0)
            frame_index = 0
            event_count = 0
            started = time.time()
            last_status_at = 0.0
            recent_frame_times = deque(maxlen=30)

            while self._running:
                ok, frame = capture.read()
                if not ok:
                    break
                if self.crop != "none":
                    frame = apply_crop(frame, self.crop)
                display_frame = None
                if display_capture is not None:
                    display_ok, candidate = display_capture.read()
                    if display_ok:
                        display_frame = (
                            apply_crop(candidate, self.crop)
                            if self.crop != "none"
                            else candidate
                        )

                tick = time.time()
                annotated, events = detector.process(
                    frame,
                    frame_index,
                    fps,
                    display_frame=display_frame,
                )
                event_count += len(events)
                event_dicts = []
                for event in events:
                    item = event.to_dict()
                    item["event_id"] = event_key(item)
                    event_dicts.append(item)

                candidates = {}
                if self.vlm_confirm and event_dicts:
                    indices = sorted(detector.frame_buffer)
                    if indices:
                        picks = np.linspace(
                            0,
                            len(indices) - 1,
                            min(8, len(indices)),
                        ).astype(int)
                        candidate_frames = [
                            detector.frame_buffer[indices[index]].copy()
                            for index in picks
                        ]
                        for item in event_dicts:
                            if item["action"] in VLM_ACTIONS:
                                candidates[item["event_id"]] = candidate_frames
                display = annotated
                display_max_width = 720
                if display.shape[1] > display_max_width:
                    scale = display_max_width / display.shape[1]
                    display = cv2.resize(
                        display,
                        (
                            display_max_width,
                            max(1, int(round(display.shape[0] * scale))),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                self.frame_ready.emit(
                    to_qimage(display),
                    event_dicts,
                    candidates,
                )

                # 每一帧都发实时判定快照，界面上的「此刻模型看到什么」才不会滞后
                elapsed = max(time.time() - started, 1e-6)
                now = time.time()
                recent_frame_times.append(now)
                if now - last_status_at >= 0.2 or event_dicts:
                    rolling_fps = (
                        (len(recent_frame_times) - 1)
                        / max(recent_frame_times[-1] - recent_frame_times[0], 1e-6)
                        if len(recent_frame_times) > 1
                        else 0.0
                    )
                    self.status.emit(
                        {
                            "frame": frame_index,
                            "fps": rolling_fps,
                            "events": event_count,
                            "total": total,
                            "live": detector.live_tracks,
                        }
                    )
                    last_status_at = now
                frame_index += 1
                if self.limit and frame_index >= self.limit:
                    break

                # 按视频原始帧率节流，模拟实时播放；推理比播放慢时自然就不 sleep
                spare = interval - (time.time() - tick)
                if spare > 0:
                    time.sleep(spare)

            capture.release()
            if display_capture is not None:
                display_capture.release()
            elapsed = max(time.time() - started, 1e-6)
            self.done.emit(
                f"Processed {frame_index} frames at {frame_index/elapsed:.1f} fps, "
                f"{event_count} alerts"
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


class VlmConfirmWorker(QThread):
    """Load the VLM once and confirm queued candidate events in order."""

    result_ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        model_path: str,
        gpu_memory_gib: float = 3.5,
        cpu_memory_gib: float = 5.0,
    ):
        super().__init__()
        self.model_path = model_path
        self.gpu_memory_gib = gpu_memory_gib
        self.cpu_memory_gib = cpu_memory_gib
        self.tasks: queue.Queue = queue.Queue(maxsize=3)
        self._running = True

    def submit(self, event: dict, frames: list[np.ndarray]) -> None:
        if not self._running or not frames:
            return
        task = {"event": event, "frames": frames}
        try:
            self.tasks.put_nowait(task)
        except queue.Full:
            try:
                self.tasks.get_nowait()
            except queue.Empty:
                pass
            try:
                self.tasks.put_nowait(task)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            self.status.emit("loading")
            from vlm_confirmer import VLMConfirmer

            confirmer = VLMConfirmer(
                self.model_path,
                load_in_4bit=True,
                gpu_memory_gib=self.gpu_memory_gib,
                cpu_memory_gib=self.cpu_memory_gib,
            )
            self.status.emit("ready")
            while self._running:
                try:
                    task = self.tasks.get(timeout=0.25)
                except queue.Empty:
                    continue
                event = task["event"]
                try:
                    result = confirmer.confirm(
                        task["frames"],
                        event["action"],
                        threshold=0.70,
                    )
                    result["event_id"] = event["event_id"]
                    result["event"] = event
                    self.result_ready.emit(result)
                except Exception as exc:  # noqa: BLE001
                    self.failed.emit(
                        f"VLM confirmation failed for track "
                        f"{event.get('track_id')}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            )


class MainWindow(QMainWindow):
    def __init__(
        self,
        source: str,
        classifier: str | None = None,
        vlm_enabled: bool = True,
        vlm_model: str = DEFAULT_VLM_MODEL,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Scene Detection - Nine Actions")
        self.resize(1180, 720)

        self.source = source
        self.classifier_path = classifier
        self.worker: InferenceWorker | None = None
        self.vlm_worker: VlmConfirmWorker | None = None
        self.vlm_model = vlm_model
        self.vlm_enabled_on_start = vlm_enabled
        self._vlm_pending: dict[str, tuple[int, dict]] = {}
        self._pending_image: QImage | None = None
        self.paused = False

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
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Time (s)", "Track", "Action", "YOLO", "VLM"]
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
        self.vlm_check = QCheckBox("VLM confirmation")
        self.vlm_check.setChecked(self.vlm_enabled_on_start)
        self.vlm_check.setToolTip(
            "Confirm falling/fighting candidates with Qwen2.5-VL. "
            "The model loads once in a background thread."
        )

        self.sound_check = QCheckBox("Alert sound")
        self.sound_check.setChecked(True)
        self.sound_check.setToolTip(
            "Falling: three high beeps; fighting: two beeps; "
            "running: one beep. Played in a background thread."
        )

        self.info_label = QLabel("Waiting")
        self.info_label.setStyleSheet("color:#666;")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Live model output (per frame)"))
        note = QLabel(
            "Sitting / Walking / Kicking / Waving / Nodding / Handshaking: "
            "display plus VLM confirmation"
        )
        note.setStyleSheet("color:#888;font-size:12px;")
        right_layout.addWidget(note)
        right_layout.addWidget(self.live_label)
        right_layout.addSpacing(8)
        right_layout.addWidget(QLabel("Alert events"))
        right_layout.addWidget(self.table, 1)
        right_layout.addWidget(self.fighting_check)
        right_layout.addWidget(self.running_check)
        right_layout.addWidget(self.vlm_check)
        right_layout.addWidget(self.sound_check)
        legend = QLabel(
            "Box colors: Red=Falling  Orange=Sitting  Green=Walking  "
            "Purple=Kicking  Yellow=Running  Cyan=Waving  "
            "Blue=Nodding  Orange-brown=Handshaking"
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
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(40)
        self.render_timer.timeout.connect(self._render_pending_frame)
        self.render_timer.start()

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
        config.imgsz = 640
        if self.classifier_path:
            config.classifier_path = self.classifier_path
        config.alarm_fighting = (
            self.fighting_check.isChecked()
            or self.vlm_check.isChecked()
        )
        config.alarm_running = self.running_check.isChecked()
        crop = resolve_crop(self.source, "auto")

        if self.vlm_check.isChecked() and (
            self.vlm_worker is None or not self.vlm_worker.isRunning()
        ):
            if not os.path.isdir(self.vlm_model):
                QMessageBox.warning(
                    self,
                    "VLM unavailable",
                    f"VLM model directory not found:\n{self.vlm_model}",
                )
                self.vlm_check.setChecked(False)
                config.alarm_fighting = self.fighting_check.isChecked()
            else:
                self.vlm_worker = VlmConfirmWorker(self.vlm_model)
                self.vlm_worker.result_ready.connect(self._on_vlm_result)
                self.vlm_worker.status.connect(self._on_vlm_status)
                self.vlm_worker.failed.connect(self._on_vlm_failed)
                self.vlm_worker.start()

        self.table.setRowCount(0)
        self._vlm_pending.clear()
        self._pending_image = None
        self.banner.setStyleSheet(
            "background:#2b2b2b;color:#dddddd;font-size:15px;border-radius:4px;"
        )
        self.banner.setText("Detecting...")

        self.worker = InferenceWorker(
            self.source,
            config,
            crop,
            vlm_confirm=self.vlm_check.isChecked(),
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status.connect(self._on_status)
        self.worker.failed.connect(self._on_failed)
        self.worker.done.connect(self._on_done)
        self.worker.start()
        self.start_button.setText("Stop")
        self.info_label.setText("Detecting...")

    def _on_frame(
        self,
        frame: object,
        events: list,
        candidates: dict,
    ) -> None:
        self._pending_image = (
            frame if isinstance(frame, QImage) else to_qimage(frame)
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
            should_confirm = (
                event.get("event_id") in candidates
                and self.vlm_worker is not None
                and self.vlm_worker.isRunning()
            )
            self.table.setItem(
                row,
                4,
                QTableWidgetItem("Pending" if should_confirm else "Not required"),
            )
            self.table.scrollToBottom()
            if should_confirm and self.vlm_worker is not None:
                self._vlm_pending[event["event_id"]] = (row, event)
                self.vlm_worker.submit(event, candidates[event["event_id"]])
                self._flash_banner(
                    f"Candidate: Track {event['track_id']} {action_text}; "
                    "VLM checking..."
                )
                continue
            self._flash_banner(
                f"Alert: Track {event['track_id']}  {action_text}  "
                f"(confidence {event['confidence']:.2f})"
            )
            if self._should_play_alarm(event["action"]):
                play_alarm_sound(event["action"])

    def _render_pending_frame(self) -> None:
        if self._pending_image is None:
            return
        image = self._pending_image
        self._pending_image = None
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def _on_vlm_status(self, status: str) -> None:
        if status == "loading":
            self.vlm_check.setText("VLM confirmation (loading...)")
        elif status == "ready":
            self.vlm_check.setText("VLM confirmation (ready)")
        else:
            self.vlm_check.setText("VLM confirmation")

    def _on_vlm_result(self, result: dict) -> None:
        event = result["event"]
        pending = self._vlm_pending.pop(result.get("event_id"), None)
        if pending is None:
            return
        row, _pending_event = pending

        accepted = bool(result.get("accepted"))
        prediction = str(result.get("action", "parse_error"))
        confidence = float(result.get("confidence", 0.0))
        status_text = (
            f"{prediction} {confidence:.2f} "
            f"({'confirmed' if accepted else 'rejected'})"
        )
        self.table.setItem(row, 4, QTableWidgetItem(status_text))
        action_text = translate_label(event["action_zh"])
        if accepted:
            self._flash_banner(
                f"Confirmed: Track {event['track_id']} {action_text}  "
                f"VLM {confidence:.2f}"
            )
            if self._should_play_alarm(event["action"]):
                play_alarm_sound(event["action"])
        else:
            self._flash_banner(
                f"YOLO candidate not confirmed: Track {event['track_id']} "
                f"{action_text} -> VLM {prediction} {confidence:.2f}"
            )

    def _on_vlm_failed(self, message: str) -> None:
        self.vlm_check.setText("VLM confirmation (unavailable)")
        pending = list(self._vlm_pending.values())
        self._vlm_pending.clear()
        for row, event in pending:
            self.table.setItem(row, 4, QTableWidgetItem("Unavailable"))
            if self._should_play_alarm(event["action"]):
                play_alarm_sound(event["action"])
        self.info_label.setText("VLM unavailable; using YOLO result")
        if self.worker is not None:
            self.worker.vlm_confirm = False
        print(message)

    def _should_play_alarm(self, action: str) -> bool:
        if not self.sound_check.isChecked():
            return False
        if action == "falling":
            return True
        if action == "fighting":
            return self.fighting_check.isChecked()
        if action == "running":
            return self.running_check.isChecked()
        return False

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
    default_source = (
        DEFAULT_QT_SOURCE if os.path.exists(DEFAULT_QT_SOURCE) else DEFAULT_SOURCE
    )
    parser.add_argument("--source", default=default_source, help="Initial video")
    parser.add_argument(
        "--classifier",
        default=None,
        help="Default: runs_six_transfer/best.pt; legacy two-class weights are optional",
    )
    parser.add_argument(
        "--vlm",
        dest="vlm_enabled",
        action="store_true",
        help="Enable Qwen2.5-VL confirmation (default)",
    )
    parser.add_argument(
        "--no-vlm",
        dest="vlm_enabled",
        action="store_false",
        help="Disable Qwen2.5-VL and run the lightweight real-time pipeline only",
    )
    parser.set_defaults(vlm_enabled=True)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    parser.add_argument(
        "--autostart",
        action="store_true",
        help="Start detection automatically after the window opens",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow(
        args.source if os.path.exists(args.source) else "",
        classifier=args.classifier,
        vlm_enabled=args.vlm_enabled,
        vlm_model=args.vlm_model,
    )
    window.show()
    if args.autostart or os.environ.get("SCENE_DETECT_AUTOSTART") == "1":
        QTimer.singleShot(600, window._toggle_run)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
