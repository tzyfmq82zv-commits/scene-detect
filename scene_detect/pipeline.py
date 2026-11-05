"""实时场景检测流水线：YOLO-Pose + 多目标跟踪 + 动作模型 + 奔跑规则 + 报警逻辑。

这是整个系统的核心，命令行脚本和 Qt 界面都调用它。

每一帧做四件事：
    1. YOLO-Pose 检出人，跟踪器给他们分配稳定的 ID；
    2. 每条轨迹维持一个滑动窗口的骨架序列；
    3. 每隔若干帧对每条够长的轨迹做一次分类——摔倒和打架走模型，
       奔跑走速度规则；
    4. 把逐次判定结果送进投票器，够票且过了冷却时间才真正触发报警。

为什么要有第 4 步：单次判定的抖动很大，直接按帧报警的话画面会疯狂闪。
投票加冷却能把「偶发一帧判错」和「持续几十帧的行为」分开。
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import torch

import cv2

from dataset import (
    LABEL_COLUMNS,
    LABEL_NAMES_EN,
    LABEL_NAMES_ZH,
    apply_normalization,
    normalization_params,
    resample_temporal,
)
from extract_appearance import crop_with_padding, to_batch
from model import build_model
from running_rule import RunningRule

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_DISPLAY_EN = {
    LABEL_NAMES_ZH[key]: LABEL_NAMES_EN[key] for key in LABEL_NAMES_EN
}

# 六类迁移版：跑/摔/打/坐/走/踢。摔倒和打架从旧生产权重迁移而来，
# 跨数据集泛化比 runs_dual2 好（GMDCSA 摔倒 F1 0.10 -> 0.58）。
SIX_CLASS_CLASSIFIER = os.path.join(HERE, "runs_six_transfer", "best.pt")
PRODUCTION_CLASSIFIER = os.path.join(HERE, "runs_dual2", "best.pt")
DEFAULT_CLASSIFIER = (
    SIX_CLASS_CLASSIFIER
    if os.path.exists(SIX_CLASS_CLASSIFIER)
    else PRODUCTION_CLASSIFIER
)

# 只有这两类触发报警。坐/走/踢只做状态显示，奔跑继续走速度规则。
ALARM_COLUMNS = ("falling", "fighting")

# 画面配色（BGR）
COLOR_NORMAL = (200, 200, 200)
COLOR_RUNNING = (0, 200, 255)
COLOR_FALLING = (0, 0, 255)
COLOR_FIGHTING = (255, 100, 0)
COLOR_ALARM = (0, 0, 255)
COLOR_SITTING = (0, 165, 255)
COLOR_WALKING = (0, 200, 0)
COLOR_KICKING = (200, 0, 200)


@dataclass
class PipelineConfig:
    """所有可调参数集中在这里，界面上想开放哪几个也方便。"""

    pose_model: str = "yolo11n-pose.pt"
    classifier_path: str = DEFAULT_CLASSIFIER
    tracker: str = "botsort.yaml"
    device: str = "cuda"
    imgsz: int = 640
    person_confidence: float = 0.3

    clip_length: int = 48
    classify_stride: int = 6       # 每隔多少帧对每条轨迹分类一次
    warmup_frames: int = 24        # 轨迹短于这个帧数不判定，避免把噪声当动作

    # 投票与冷却：最近 vote_window 次判定里有 vote_min 次为真才触发
    vote_window: int = 5
    vote_min: int = 3
    cooldown_seconds: float = 8.0

    # 奔跑规则
    running_threshold: float = 1.05

    # 外观支路每个片段采样几帧。必须和离线抽特征时的取值一致，
    # 否则模型看到的输入分布和训练时不同
    appearance_frames: int = 8

    # 摔倒的物理闸门。模型有时会在人还在正常走动时就输出很高的摔倒概率，
    # 这是数据集层面的捷径（UR Fall 的摔倒和日常活动是分场次录的，
    # 模型能部分认出「这是摔倒视频」）。而摔倒在物理上必然伴随
    # 重心快速下坠或身体由竖转横，所以要求两者之一成立才放行报警。
    # 实测这道闸门不起作用：提前触发时检测框的高宽比已经被判为"躺倒"，
    # 而下坠速度在朝镜头方向倒地时接近 0（框在变大而不是下移）。
    # 强行打开它会把 fall-03 这类真摔倒也一起拦掉，造成更严重的漏报，
    # 所以默认关闭。留着代码是为了以后有更好的物理线索时能接上。
    fall_requires_descent: bool = False
    fall_descent_threshold: float = 0.5    # 身体重心下坠速度，单位 身高/秒
    fall_lying_ratio: float = 1.3          # 检测框 高/宽 低于这个值算已躺倒

    # 打架六类迁移版验证 F1 0.93。默认仍只做画面提示，
    # 在界面勾选后才会触发声音和事件记录。
    alarm_fighting: bool = False
    alarm_running: bool = False

    # 同一帧里最多允许几个人被标成「打架」。
    # 公交车上五个人里往往只有两个在动手，但模型的打架分支几乎给所有人都
    # 打出高分——训练时把片段级的暴力标签盖到了片段里所有人的头上，
    # 所以它学成了「这段视频里的人都算打架」。模型本身分不出谁在动手
    # （实测运动量、两两距离都分不开），这道闸门只能把伤害面收窄到得分
    # 最高的前几个，避免把整车人一起指控。设为 0 表示不限制。
    fight_max_tracks: int = 2

    # 打架的两个硬前提：画面里至少要有两个人，而且这个人附近得真的有人。
    # 没有这两条时，单独一个人在奔跑或摔倒也会被判成打架——实测在
    # Weizmann 的单人奔跑片段上打架分支给出过 0.70 的分数。一个人身边
    # 没人，物理上不可能在打架，这是最可靠的一条约束。
    fight_min_people: int = 2
    fight_neighbor_distance: float = 2.5   # 单位：人体身高，按检测框高度换算

    max_track_history: int = 120   # 每条轨迹保留多少帧骨架


@dataclass
class TrackState:
    """一条轨迹的运行时状态。"""

    track_id: int
    keypoints: deque = field(default_factory=deque)
    boxes: deque = field(default_factory=deque)
    frames: deque = field(default_factory=deque)
    probabilities: dict = field(default_factory=lambda: {"falling": 0.0, "fighting": 0.0})
    running_speed: float = 0.0
    # 注意：这里的 deque 必须带长度上限，否则「最近 N 次判定」会退化成
    # 「历史上任意 N 次判定」，一旦攒够票就永远报下去，再也停不下来。
    # 上限在 _raise_alarms 里按配置创建，所以这里留空字典。
    votes: dict = field(default_factory=dict)
    last_alarm_frame: dict = field(default_factory=dict)
    last_frame_seen: int = 0
    # 物理闸门的中间量，便于在界面上解释为什么没报警
    fall_descent: float = 0.0
    fall_ratio: float = 0.0
    # 专用坐/走/踢/其他分类器的输出（四类 softmax）
    state_probs: dict = field(default_factory=dict)
    state_history: deque = field(default_factory=lambda: deque(maxlen=3))

    # 当前帧要显示的内容，由流水线每帧刷新
    labels: list = field(default_factory=list)
    alarm_labels: list = field(default_factory=list)


@dataclass
class Event:
    """一次报警事件。"""

    frame_index: int
    timestamp: float
    track_id: int
    action: str
    confidence: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_index,
            "time_sec": round(self.timestamp, 2),
            "track_id": self.track_id,
            "action": self.action,
            "action_zh": LABEL_NAMES_ZH.get(self.action, self.action),
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
        }


class SceneDetector:
    """把检测、跟踪、动作判定、报警串起来的实时流水线。"""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        from ultralytics import YOLO

        self.config = config or PipelineConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )

        self.pose = YOLO(self.config.pose_model)
        self.classifier, self.label_columns, self.thresholds = self._load_classifier()
        self.running_rule = RunningRule(threshold=self.config.running_threshold)
        self.state_bundle = None
        state_path = os.path.join(
            os.path.dirname(self.config.classifier_path), "state_classifier.joblib"
        )
        if os.path.exists(state_path):
            try:
                import joblib

                bundle = joblib.load(state_path)
                if list(bundle.get("source_columns", [])) == list(self.label_columns):
                    self.state_bundle = bundle
                    print(
                        "加载专用坐/走/踢状态分类器"
                        f"（验证宏 F1 {bundle.get('validation_macro_f1', 0):.3f}）"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"状态分类器加载失败，回退到六头输出：{exc}")

        # 双流模型需要外观输入，这里按需加载 ResNet18。
        # 用纯骨架模型时不加载，省掉一次前向的开销
        self.appearance_backbone = None
        if getattr(self.classifier, "appearance_hidden", None) is not None:
            self._load_appearance_backbone()

        # 最近若干帧的原图缓存。外观支路要从画面里裁人，必须留着原始像素。
        # 全分辨率的 48 帧在 1080p 下约 300 MB，单路视频可以接受；
        # 如果要同时跑很多路，得改成缓存降采样后的副本并同步缩放检测框
        self.frame_buffer: dict[int, np.ndarray] = {}
        self._frame_buffer_order: deque = deque()

        self.tracks: dict[int, TrackState] = {}
        self.events: list[Event] = []
        # 每条当前可见轨迹的实时判定结果，供界面逐帧刷新显示
        self.live_tracks: list[dict] = []

    def _load_appearance_backbone(self) -> None:
        """加载专门给打架判定用的 ResNet18 外观提取器。"""
        import torchvision

        print("加载外观支路 ResNet18……")
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        backbone.fc = torch.nn.Identity()
        backbone.eval().to(self.device)
        self.appearance_backbone = backbone

    def _load_classifier(self):
        """读模型权重，同时把它训练时用的标签列和最优阈值一起取出来。

        阈值一定要用训练时扫出来的那一组，固定 0.5 在类别不平衡的数据上
        会让召回率惨不忍睹。
        """
        path = self.config.classifier_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到模型文件 {path}\n"
                "先跑一次 python train.py 生成，或者用 --classifier 指定别的权重"
            )
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")

        columns = tuple(blob.get("label_columns", LABEL_COLUMNS))
        model = build_model(
            blob.get("model", "stgcn"),
            num_joints=17,
            in_channels=3,
            num_classes=len(columns),
        )
        state = blob["state_dict"]
        if not any(key.startswith("backbone.") for key in state):
            state = remap_legacy_state_dict(state)
        model.load_state_dict(state)
        model.to(self.device).eval()

        thresholds = {
            column: float(value)
            for column, value in zip(columns, blob.get("thresholds", []))
        }
        # 六类模型是在 32 帧上训练的；旧生产模型是 48 帧。
        # 推理窗口跟着权重走，避免训练/推理时间尺度不一致。
        self.config.clip_length = int(blob.get("clip_length", self.config.clip_length))
        return model, columns, thresholds

    # ------------------------------------------------------------------ 推理

    @torch.no_grad()
    def _classify(self, state: TrackState) -> dict[str, float]:
        """用最近一个窗口的骨架跑一次分类。"""
        keypoints = np.stack(list(state.keypoints)[-self.config.clip_length :]).astype(
            np.float32
        )
        boxes = np.stack(list(state.boxes)[-self.config.clip_length :]).astype(np.float32)
        frames = list(state.frames)[-self.config.clip_length :]

        neighbor, pair = self._neighbor_features(keypoints, boxes, frames, state.track_id)
        centers, scale = normalization_params(keypoints, boxes)

        primary = apply_normalization(keypoints, centers, scale)
        neighbor = apply_normalization(neighbor, centers, scale)
        pair = pair.copy()
        pair[:, 0] /= scale
        pair[:, 2] /= scale
        pair[pair[:, 3] < 0.5] = 0.0
        pair[:, 3] = (pair[:, 3] >= 0.5).astype(np.float32)

        length = self.config.clip_length
        primary = resample_temporal(primary, length)
        neighbor = resample_temporal(neighbor, length)
        pair = resample_temporal(pair, length)
        pair[:, 3] = (pair[:, 3] >= 0.5).astype(np.float32)

        tensor = torch.from_numpy(primary.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        neighbor_t = (
            torch.from_numpy(neighbor.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )
        pair_t = torch.from_numpy(pair.transpose(1, 0)).unsqueeze(0).to(self.device)
        appearance_t = self._appearance_features(state)
        state.state_probs = {}
        if self.state_bundle is not None and hasattr(
            self.classifier, "forward_with_features"
        ):
            logits, fused = self.classifier.forward_with_features(
                tensor, neighbor_t, pair_t, appearance_t
            )
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()[0]
            state_model = self.state_bundle["classifier"]
            state_values = state_model.predict_proba(
                fused.float().cpu().numpy()
            )[0]
            class_names = list(self.state_bundle["class_names"])
            state.state_probs = {
                name: float(value) for name, value in zip(class_names, state_values)
            }
            state.state_history.append(
                class_names[int(np.argmax(state_values))]
            )
            for index, name in enumerate(class_names[:3]):
                if name in self.label_columns:
                    probabilities[self.label_columns.index(name)] = state_values[index]
        else:
            logits = self.classifier(tensor, neighbor_t, pair_t, appearance_t)
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()[0]
        return {column: float(probabilities[i]) for i, column in enumerate(self.label_columns)}

    @torch.no_grad()
    def _appearance_features(self, state: TrackState):
        """裁出这个人最近若干帧的画面块，过 ResNet 得到外观特征。

        流程必须和离线抽取时逐字一致——按检测框外扩同样比例裁剪、缩放到
        224、用同一套 ImageNet 统计量归一化。离线在线只要有一处不同，
        模型看到的就是两种分布。
        """
        import torch

        needed = self.config.appearance_frames
        if self.appearance_backbone is None:
            return None

        frames = list(state.frames)[-self.config.clip_length :]
        boxes = list(state.boxes)[-self.config.clip_length :]
        if not frames:
            return torch.zeros(
                1, 512, needed, device=self.device, dtype=torch.float32
            )

        picks = np.linspace(0, len(frames) - 1, needed).astype(int)
        crops = []
        for pick in picks:
            frame = self.frame_buffer.get(frames[pick])
            if frame is None:
                continue
            crop = crop_with_padding(frame, boxes[pick])
            if crop is not None:
                crops.append(crop)
        if not crops:
            return torch.zeros(
                1, 512, needed, device=self.device, dtype=torch.float32
            )

        feature = self.appearance_backbone(to_batch(crops).to(self.device))
        feature = feature.cpu().numpy()
        # 帧数不足时用最后一帧补齐，和离线抽取的处理方式保持一致
        if feature.shape[0] < needed:
            pad = np.repeat(feature[-1:], needed - feature.shape[0], axis=0)
            feature = np.concatenate([feature, pad], axis=0)
        return torch.from_numpy(feature).transpose(0, 1).unsqueeze(0).to(self.device)

    def _neighbor_features(
        self, keypoints: np.ndarray, boxes: np.ndarray, frames: list[int], track_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """算出窗口内每一帧的最近邻人。单人场景会全部返回零。

        当前默认模型看不到邻人（stgcn 会忽略这两路输入），但这里仍然照常算，
        这样以后想换成交互版模型不用改流水线。
        """
        count = keypoints.shape[0]
        neighbor = np.zeros_like(keypoints)
        pair = np.zeros((count, 4), dtype=np.float32)

        # 建立 帧号 -> 其他轨迹在该帧的骨架 的查表
        frame_lookup: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
        for other_id, state in self.tracks.items():
            if other_id == track_id:
                continue
            for slot, frame_index in enumerate(state.frames):
                frame_lookup[frame_index].append(
                    (other_id, state.keypoints[slot], state.boxes[slot])
                )

        for slot, frame_index in enumerate(frames):
            candidates = frame_lookup.get(frame_index)
            if not candidates:
                continue
            own_box = boxes[slot]
            own_cx = (own_box[0] + own_box[2]) * 0.5
            own_cy = (own_box[1] + own_box[3]) * 0.5
            best = None
            for _other_id, other_kpts, other_box in candidates:
                cx = (other_box[0] + other_box[2]) * 0.5
                cy = (other_box[1] + other_box[3]) * 0.5
                distance = float(np.hypot(cx - own_cx, cy - own_cy))
                if best is None or distance < best[0]:
                    best = (distance, other_kpts, other_box)
            distance, other_kpts, other_box = best
            neighbor[slot] = other_kpts
            pair[slot, 0] = distance
            pair[slot, 1] = _bbox_iou(own_box, other_box)
            pair[slot, 3] = 1.0
            visible_own = keypoints[slot][:, 2] > 0.2
            visible_other = other_kpts[:, 2] > 0.2
            if visible_own.any() and visible_other.any():
                delta = (
                    keypoints[slot][visible_own][:, None, :2]
                    - other_kpts[visible_other][None, :, :2]
                )
                pair[slot, 2] = float(np.linalg.norm(delta, axis=2).min())
        return neighbor, pair

    # ------------------------------------------------------------------ 主循环

    def process(self, frame: np.ndarray, frame_index: int, fps: float) -> list[Event]:
        """处理一帧，返回 (画好标注的画面, 这一帧新触发的报警事件)。"""
        # 先把原图存进缓存：外观支路要从画面里裁出每个人，
        # 而骨架和检测框里没有像素信息，只能靠这里留底
        self.frame_buffer[frame_index] = frame
        self._frame_buffer_order.append(frame_index)
        while len(self._frame_buffer_order) > self.config.clip_length + 2:
            oldest = self._frame_buffer_order.popleft()
            self.frame_buffer.pop(oldest, None)

        results = self.pose.track(
            source=frame,
            persist=True,
            tracker=self.config.tracker,
            classes=[0],
            conf=self.config.person_confidence,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        result = results[0] if isinstance(results, list) else results
        new_events: list[Event] = []

        boxes = result.boxes
        keypoints = result.keypoints
        seen_ids: set[int] = set()

        if boxes is not None and boxes.id is not None and keypoints is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            kpts = keypoints.data.cpu().numpy()
            for person_index, track_id in enumerate(ids):
                if kpts.shape[1] != 17:
                    continue
                seen_ids.add(int(track_id))
                state = self.tracks.get(int(track_id))
                if state is None:
                    state = TrackState(track_id=int(track_id))
                    self.tracks[int(track_id)] = state
                state.keypoints.append(kpts[person_index])
                state.boxes.append(xyxy[person_index])
                state.frames.append(frame_index)
                state.last_frame_seen = frame_index
                while len(state.keypoints) > self.config.max_track_history:
                    state.keypoints.popleft()
                    state.boxes.popleft()
                    state.frames.popleft()

        annotated = frame.copy()
        # 分三遍走。不能边分类边决定，因为「谁在打架」是一个全画面范围的
        # 比较问题——必须等所有人的分数都算出来，才能挑出最像在动手的那几个
        active = [state for tid, state in self.tracks.items() if tid in seen_ids]

        # 第一遍：到期的轨迹做分类，并算出奔跑速度
        for state in active:
            if len(state.frames) < self.config.warmup_frames:
                state.labels = []
                state.alarm_labels = []
                continue
            if frame_index % self.config.classify_stride == 0:
                state.probabilities = self._classify(state)
                boxes = np.stack(
                    list(state.boxes)[-self.config.clip_length :]
                ).astype(np.float32)
                keypoints = np.stack(
                    list(state.keypoints)[-self.config.clip_length :]
                ).astype(np.float32)
                state.running_speed = self.running_rule.speed(boxes, keypoints, fps)

        # 第二遍：全画面比较，挑出允许被标成「打架」的轨迹
        allowed_fighting = self._fighting_candidates(active)

        # 第三遍：出标签、投票、报警、绘制
        for state in active:
            if len(state.frames) < self.config.warmup_frames:
                self._draw_track(annotated, state)
                continue
            self._decide(state, allowed_fighting)
            self._raise_alarms(state, frame_index, fps, new_events, allowed_fighting)
            self._draw_track(annotated, state)

        # 每帧刷新一次实时判定快照。界面拿它显示「模型此刻看到了什么」，
        # 和右侧的报警事件表区分开：那张表是历史，这个是当下
        self.live_tracks = [
            {
                "track_id": state.track_id,
            "labels": (
                [LABEL_DISPLAY_EN.get(label, label) for label in state.labels]
                if state.labels
                else ["Normal"]
            ),
                "speed": round(state.running_speed, 2),
                "probabilities": {
                    key: round(value, 3) for key, value in state.probabilities.items()
                },
                "alarm": [
                    LABEL_DISPLAY_EN.get(label, label)
                    for label in state.alarm_labels
                ],
            }
            for track_id, state in self.tracks.items()
            if track_id in seen_ids and len(state.frames) >= self.config.warmup_frames
        ]
        self._drop_stale(frame_index)
        return annotated, new_events

    def reset(self) -> None:
        """换到另一段视频时调用，把跟踪状态清干净。

        跟踪器内部也要清：它的全局运动补偿会留着上一段视频的灰度图，
        换到分辨率不同的视频时，光流金字塔尺寸对不上，会刷出一屏
        "GMC failed" 警告；而且上一段的轨迹 ID 会残留到新视频的开头。
        单段视频内不需要调用，只在切换输入源时调一次。
        """
        self.tracks.clear()
        self.live_tracks = []
        self.frame_buffer.clear()
        self._frame_buffer_order.clear()
        try:
            self.pose.predictor.trackers = []
        except AttributeError:
            pass

    def _fighting_candidates(self, states: list[TrackState]) -> set[int]:
        """挑出这一帧允许标成「打架」的轨迹：得分最高的前几个，且要过阈值。

        这只是把伤害面收窄——模型本身分不出谁真的在动手（实测运动量、
        两两距离在暴力和非暴力片段上分布完全重合）。设成 0 就退回不限制。
        """
        if self.config.fight_max_tracks <= 0:
            return {state.track_id for state in states}
        # 硬前提一：画面里至少要有两个人，否则谈不上打架
        if len(states) < self.config.fight_min_people:
            return set()
        threshold = self.thresholds.get("fighting", 0.5)
        scored = []
        for state in states:
            score = state.probabilities.get("fighting", 0.0)
            if score < threshold:
                continue
            # 硬前提二：这个人附近得真的有人，孤零零一个不算
            if not self._has_close_neighbor(state, states):
                continue
            scored.append((score, state.track_id))
        scored.sort(reverse=True)
        return {track_id for _, track_id in scored[: self.config.fight_max_tracks]}

    def _has_close_neighbor(
        self, state: TrackState, states: list[TrackState]
    ) -> bool:
        """判断这个人附近有没有别人。

        距离用检测框高度归一化，抵消远近带来的尺度差异——否则远处的人
        怎么量都算「离得远」，近处的人怎么量都算「挨着」。
        """
        if not state.boxes:
            return False
        box = state.boxes[-1]
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        own_height = max(box[3] - box[1], 1.0)
        for other in states:
            if other.track_id == state.track_id or not other.boxes:
                continue
            other_box = other.boxes[-1]
            other_x = (other_box[0] + other_box[2]) * 0.5
            other_y = (other_box[1] + other_box[3]) * 0.5
            other_height = max(other_box[3] - other_box[1], 1.0)
            scale = (own_height + other_height) * 0.5
            distance = float(np.hypot(other_x - center_x, other_y - center_y)) / scale
            if distance <= self.config.fight_neighbor_distance:
                return True
        return False

    def _decide(self, state: TrackState, allowed_fighting: set[int]) -> None:
        """按每类的阈值给这条轨迹打当前标签。"""
        labels: list[str] = []
        state_columns = (
            set(self.state_bundle["class_names"][:3])
            if self.state_bundle is not None
            else set()
        )
        for column in self.label_columns:
            # 六类模型虽然也有 running 头，但它的验证 F1 只有 0.39，
            # 奔跑继续用速度规则，避免把这条支路的噪声显示出来。
            if column == "running":
                continue
            if column in state_columns:
                # 坐/走/踢由专用四分类统一裁决，后面只追加一个 argmax，
                # 不再让三个独立 sigmoid 同时亮。
                continue
            if column == "fighting" and state.track_id not in allowed_fighting:
                # 没被挑中说明同画面里有人比你更像在动手，这一帧就不标你
                continue
            threshold = self.thresholds.get(column, 0.5)
            if state.probabilities.get(column, 0.0) >= threshold:
                labels.append(LABEL_NAMES_ZH.get(column, column))
        is_running = state.running_speed >= self.config.running_threshold and (
            "摔倒" not in labels
        )
        if self.state_bundle is not None and state.state_probs:
            class_names = list(self.state_bundle["class_names"])
            best_index = int(np.argmax([state.state_probs[name] for name in class_names]))
            best_name = class_names[best_index]
            if state.state_history:
                # 取最近 2–3 次判定的多数，抑制坐/踢之间的单帧闪烁。
                history = list(state.state_history)
                candidate = max(set(history), key=history.count)
                if history.count(candidate) >= 2:
                    best_name = candidate
            threshold = float(self.state_bundle.get("confidence_threshold", 0.45))
            if (
                best_name in state_columns
                and state.state_probs[best_name] >= threshold
                and not is_running
                and "摔倒" not in labels
                and "打架" not in labels
            ):
                labels.append(LABEL_NAMES_ZH.get(best_name, best_name))
        if is_running:
            labels.append(LABEL_NAMES_ZH["running"])
        state.labels = labels

    def _raise_alarms(
        self,
        state: TrackState,
        frame_index: int,
        fps: float,
        new_events: list[Event],
        allowed_fighting: set[int],
    ) -> None:
        """投票加冷却，够票才真正报警。"""
        config = self.config
        checks: list[tuple[str, bool, float, bool]] = []
        for column in self.label_columns:
            if column not in ALARM_COLUMNS:
                # 坐/走/踢以及模型自带的 running 头都只做状态显示，
                # 不进入投票队列，否则界面会被它们刷满「报警」。
                continue
            if column == "fighting" and state.track_id not in allowed_fighting:
                fired = False
            else:
                threshold = self.thresholds.get(column, 0.5)
                fired = state.probabilities.get(column, 0.0) >= threshold
            enabled = column == "falling" or config.alarm_fighting
            if column == "falling" and config.fall_requires_descent:
                # 物理闸门：光模型说是摔倒不算，还得真的在下坠或者在躺着
                confirmed, descent, ratio = self.fall_motion_confirmed(state, fps)
                if not confirmed:
                    state.fall_descent = descent
                    state.fall_ratio = ratio
                    fired = False
            checks.append((column, fired, state.probabilities.get(column, 0.0), enabled))
        checks.append(
            (
                "running",
                state.running_speed >= config.running_threshold,
                state.running_speed,
                config.alarm_running,
            )
        )

        state.alarm_labels = []
        for column, fired, score, enabled in checks:
            votes = state.votes.get(column)
            if votes is None:
                votes = deque(maxlen=config.vote_window)
                state.votes[column] = votes
            votes.append(1 if fired else 0)
            if len(votes) < config.vote_window:
                continue
            if sum(votes) < config.vote_min:
                continue
            last = state.last_alarm_frame.get(column)
            cooldown_frames = config.cooldown_seconds * max(fps, 1.0)
            if last is not None and frame_index - last < cooldown_frames:
                state.alarm_labels.append(LABEL_NAMES_ZH.get(column, column))
                continue
            if not enabled:
                state.alarm_labels.append(LABEL_NAMES_ZH.get(column, column))
                continue
            state.last_alarm_frame[column] = frame_index
            state.alarm_labels.append(LABEL_NAMES_ZH.get(column, column))
            new_events.append(
                Event(
                    frame_index=frame_index,
                    timestamp=frame_index / max(fps, 1.0),
                    track_id=state.track_id,
                    action=column,
                    confidence=float(score),
                    detail=f"命中 {sum(votes)}/{len(votes)} 次判定",
                )
            )

    def _drop_stale(self, frame_index: int) -> None:
        """长时间没出现的轨迹直接丢掉，避免状态无限增长。"""
        limit = int(self.config.cooldown_seconds * 3)
        for track_id in [
            tid
            for tid, state in self.tracks.items()
            if frame_index - state.last_frame_seen > limit
        ]:
            self.tracks.pop(track_id, None)

    def fall_motion_confirmed(
        self, state: TrackState, fps: float
    ) -> tuple[bool, float, float]:
        """用物理线索确认这真的是摔倒。

        返回 (是否确认, 重心下坠速度, 检测框高宽比)。

        摔倒在物理上一定会留下两个痕迹：身体重心快速下坠，或者身体由竖直
        转成横躺。人在正常走动时这两个量都不会触发，所以拿它们当闸门，
        能把「模型凭场景猜出来的提前报警」挡掉，而且不损失真摔倒的召回
        ——真摔倒时这两个量必然出现。
        """
        boxes = np.stack(list(state.boxes)[-self.config.clip_length :]).astype(np.float32)
        if boxes.shape[0] < 8:
            return False, 0.0, 0.0
        heights = boxes[:, 3] - boxes[:, 1]
        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
        valid = heights > 1
        if not valid.any():
            return False, 0.0, 0.0
        scale = float(np.median(heights[valid]))
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        duration = boxes.shape[0] / max(fps, 1.0)
        descent = float((centers_y[-1] - centers_y[0]) / scale / duration)
        recent_ratio = float(np.median(heights[-8:] / widths[-8:]))
        confirmed = (
            descent >= self.config.fall_descent_threshold
            or recent_ratio < self.config.fall_lying_ratio
        )
        return confirmed, descent, recent_ratio

    # ------------------------------------------------------------------ 绘制

    def _track_color(self, state: TrackState) -> tuple[int, int, int]:
        if "摔倒" in state.alarm_labels:
            return COLOR_FALLING
        if "打架" in state.alarm_labels:
            return COLOR_FIGHTING
        if "奔跑" in state.labels:
            return COLOR_RUNNING
        if "坐" in state.labels:
            return COLOR_SITTING
        if "走" in state.labels:
            return COLOR_WALKING
        if "踢" in state.labels:
            return COLOR_KICKING
        return COLOR_NORMAL

    def _display_text(self, state: TrackState) -> list[str]:
        parts = ["Normal"]
        if state.labels:
            parts = [LABEL_DISPLAY_EN.get(label, label) for label in state.labels]
        text = [f"ID {state.track_id}  " + " ".join(parts)]
        if state.running_speed:
            text.append(f"速度 {state.running_speed:.2f} 身高/秒")
        return text

    def _draw_track(self, canvas: np.ndarray, state: TrackState) -> None:
        """在画面上画框和标签。报警时加粗并在上方打一条红色提示。"""
        if not state.boxes:
            return
        box = state.boxes[-1]
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        color = self._track_color(state)
        alarming = bool(state.alarm_labels)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if alarming else 2)

        text = f"ID{state.track_id}"
        if state.labels:
            text += " " + " ".join(
                LABEL_DISPLAY_EN.get(label, label) for label in state.labels
            )
        else:
            text += " Normal"
        if state.running_speed:
            text += f"  {state.running_speed:.2f}h/s"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
        cv2.putText(
            canvas,
            text,
            (x1 + 4, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        if alarming:
            banner = "ALERT: " + " ".join(
                LABEL_DISPLAY_EN.get(label, label) for label in state.alarm_labels
            )
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), COLOR_ALARM, -1)
            cv2.putText(
                canvas,
                banner,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-6 else 0.0


def remap_legacy_state_dict(state_dict: dict) -> dict:
    """给旧权重补上 backbone. 前缀。

    model.py 后来被拆成 STGCNBackbone + head，为了做主目标和邻人共享权重。
    拆分之前训练出来的权重键名是 input_bn.* 和 blocks.*，少一层前缀，
    直接 load_state_dict 会报一堆 missing / unexpected。这里补上前缀，
    之前训好的模型就能继续用，不必重训。
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(("input_bn.", "blocks.")):
            remapped[f"backbone.{key}"] = value
        else:
            remapped[key] = value
    return remapped
