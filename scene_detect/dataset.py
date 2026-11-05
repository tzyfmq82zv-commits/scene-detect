"""训练数据加载：把提取好的骨架片段读成张量，并做归一化和增强。

数据来源是 feature_extract.py 产出的 .npz 片段，配合一个标注 CSV。
六个 0/1 标签列：running、falling、fighting、sitting、walking、kicking。
允许一行里出现多个 1（比如边跑边打）；六列全 0 就是「正常」。
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

# 水平镜像时左右关节要互换，否则镜像增强会造出解剖学上错误的人体
LEFT_RIGHT_PERMUTATION = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    dtype=np.int64,
)

# 头部关节在摔倒时经常被遮挡，置信度很低，归一化时不能参与，否则会把整个骨架拉偏
HEAD_JOINTS = (0, 1, 2, 3, 4)
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

LABEL_COLUMNS = (
    "running",
    "falling",
    "fighting",
    "sitting",
    "walking",
    "kicking",
)
LABEL_NAMES_ZH = {
    "running": "奔跑",
    "falling": "摔倒",
    "fighting": "打架",
    "sitting": "坐",
    "walking": "走",
    "kicking": "踢",
}
LABEL_NAMES_EN = {
    "running": "Running",
    "falling": "Falling",
    "fighting": "Fighting",
    "sitting": "Sitting",
    "walking": "Walking",
    "kicking": "Kicking",
}


@dataclass
class Clip:
    path: str
    labels: np.ndarray
    group: str


def read_label_csv(
    csv_path: str,
    root: str | None = None,
    columns: tuple[str, ...] = LABEL_COLUMNS,
) -> list[Clip]:
    """读取标注 CSV，返回所有片段。group 取文件名里双下划线之前的部分。

    columns 决定读哪几列当标签。奔跑改用速度规则之后，训练时只传
    ("falling", "fighting")，把 8 个正样本的奔跑列排除掉——那一列的
    类别权重会高达几百倍，训起来会把共享特征一起带偏。
    """
    clips: list[Clip] = []
    csv.field_size_limit(10**7)
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"clip", *columns} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"标注文件缺少列：{sorted(missing)}")
        for row in reader:
            clip_path = (row["clip"] or "").strip()
            if not clip_path:
                continue
            if root and not os.path.isabs(clip_path):
                clip_path = os.path.join(root, clip_path)
            labels = np.array(
                [float(row[col]) for col in columns], dtype=np.float32
            )
            stem = os.path.splitext(os.path.basename(clip_path))[0]
            group = stem.split("__")[0] if "__" in stem else stem
            clips.append(Clip(path=clip_path, labels=labels, group=group))
    return clips


def split_by_group(
    clips: list[Clip],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Clip], list[Clip]]:
    """按视频（而不是按片段）划分训练集和验证集。

    这一点必须严格：同一段视频切出来的相邻片段高度相似，
    如果随机按片段划分，验证集里的片段几乎都能在训练集里找到近邻，
    指标会虚高十几个点，上线后就原形毕露。
    """
    groups: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        groups[clip.group].append(clip)

    names = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)

    target_val = len(clips) * val_ratio
    val_names: list[str] = []
    running = 0
    for name in names:
        if running >= target_val:
            break
        val_names.append(name)
        running += len(groups[name])

    val_set = set(val_names)
    train = [clip for clip in clips if clip.group not in val_set]
    val = [clip for clip in clips if clip.group in val_set]
    return train, val


def split_by_group_balanced(
    clips: list[Clip],
    columns: tuple[str, ...] = LABEL_COLUMNS,
    val_ratio: float = 0.2,
    seed: int = 42,
    trials: int = 400,
) -> tuple[list[Clip], list[Clip]]:
    """按组划分，同时尽量让每个类别在验证集里占到 val_ratio。

    只按片段总数抽组时，GMDCSA 这种「4 个受试者、每人几百个片段」的数据
    会让验证集被某个受试者整个包场：摔倒占了验证集一半，踢腿只剩几十条。
    这里尝试多次随机分组，挑各类别比例最接近目标的一次。
    """
    groups: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        groups[clip.group].append(clip)
    if not groups:
        return [], []

    total = len(clips)
    target = total * val_ratio
    positives = {
        column: sum(float(clip.labels[index]) for clip in clips)
        for index, column in enumerate(columns)
    }
    names = sorted(groups)
    best_score = float("inf")
    best_val: list[str] = []

    for trial in range(max(1, trials)):
        rng = np.random.default_rng(seed + trial)
        shuffled = list(names)
        rng.shuffle(shuffled)
        selected: list[str] = []
        size = 0
        for name in shuffled:
            if size >= target:
                break
            selected.append(name)
            size += len(groups[name])
        chosen = set(selected)
        val_clips = [clip for clip in clips if clip.group in chosen]
        score = ((size - target) / max(target, 1)) ** 2
        for index, column in enumerate(columns):
            if positives[column] <= 0:
                continue
            count = sum(float(clip.labels[index]) for clip in val_clips)
            fraction = count / positives[column]
            score += ((fraction - val_ratio) / 0.25) ** 2
            if fraction < val_ratio * 0.5:
                score += 4.0
        if score < best_score:
            best_score = score
            best_val = selected

    chosen = set(best_val)
    train = [clip for clip in clips if clip.group not in chosen]
    val = [clip for clip in clips if clip.group in chosen]
    return train, val


def normalization_params(
    keypoints: np.ndarray, bbox: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """从主目标算出归一化参数，返回 (每帧中心点 (T,2), 每帧尺度 (T,))。

    单独抽出来是为了让邻居共用同一套参数：只有把两个人都用**同一个**原点和
    尺度换算，他们之间的相对位置才有意义，否则「两人相距多远」会跟着
    各自的检测框抖动而乱变。
    """
    frames = keypoints.shape[0]
    xs = keypoints[..., 0]
    ys = keypoints[..., 1]
    conf = keypoints[..., 2]

    # 中心点优先用双髋中点，髋部不可信时退回检测框中心
    hip_ok = (conf[:, [LEFT_HIP, RIGHT_HIP]] > 0.2).all(axis=1)
    centers = np.zeros((frames, 2), dtype=np.float32)
    centers[hip_ok, 0] = 0.5 * (xs[hip_ok, LEFT_HIP] + xs[hip_ok, RIGHT_HIP])
    centers[hip_ok, 1] = 0.5 * (ys[hip_ok, LEFT_HIP] + ys[hip_ok, RIGHT_HIP])

    has_bbox = bbox is not None and np.isfinite(bbox).all()
    if has_bbox:
        bbox_center = np.stack(
            [(bbox[:, 0] + bbox[:, 2]) * 0.5, (bbox[:, 1] + bbox[:, 3]) * 0.5], axis=1
        )
        centers[~hip_ok] = bbox_center[~hip_ok]
    for index in np.where(~hip_ok)[0]:
        if not has_bbox:
            valid = conf[index] > 0.2
            if valid.any():
                centers[index, 0] = xs[index][valid].mean()
                centers[index, 1] = ys[index][valid].mean()

    # 尺度优先用肩宽，肩部不可信时用检测框高度
    scale = np.ones(frames, dtype=np.float32)
    shoulder_ok = (conf[:, [LEFT_SHOULDER, RIGHT_SHOULDER]] > 0.2).all(axis=1)
    shoulder_width = np.linalg.norm(
        keypoints[:, LEFT_SHOULDER, :2] - keypoints[:, RIGHT_SHOULDER, :2], axis=1
    )
    scale[shoulder_ok] = shoulder_width[shoulder_ok]
    if has_bbox:
        height = bbox[:, 3] - bbox[:, 1]
        fallback = (~shoulder_ok) & (height > 1)
        scale[fallback] = height[fallback] * 0.4
    scale = np.where(scale > 1e-3, scale, 1.0).astype(np.float32)
    return centers, scale


def apply_normalization(
    keypoints: np.ndarray, centers: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """按给定的中心和尺度换算骨架坐标，置信度原样保留。"""
    out = keypoints.astype(np.float32).copy()
    out[..., 0] = (keypoints[..., 0] - centers[:, 0:1]) / scale[:, None]
    out[..., 1] = (keypoints[..., 1] - centers[:, 1:2]) / scale[:, None]
    return out


def normalize_clip(keypoints: np.ndarray, bbox: np.ndarray | None) -> np.ndarray:
    """把骨架平移到以髋部为原点、并按人体尺度缩放。

    不做这一步，模型学到的其实是「人在画面什么位置、离镜头多远」，
    而不是「人在做什么动作」，换机位或换段视频就完全失效。
    """
    centers, scale = normalization_params(keypoints, bbox)
    return apply_normalization(keypoints, centers, scale)


def resample_temporal(array: np.ndarray, length: int) -> np.ndarray:
    """把时间轴线性重采样到固定长度，兼容长短不一的片段。"""
    current = array.shape[0]
    if current == length:
        return array
    if current < 2:
        return np.repeat(array, length, axis=0)
    source = np.linspace(0.0, 1.0, current)
    target = np.linspace(0.0, 1.0, length)
    flat = array.reshape(current, -1)
    resampled = np.stack(
        [np.interp(target, source, flat[:, col]) for col in range(flat.shape[1])],
        axis=1,
    )
    return resampled.reshape(length, *array.shape[1:]).astype(array.dtype)


class SkeletonDataset(Dataset):
    """读取 .npz 骨架片段，输出 (C, T, V) 张量和 (3,) 标签。"""

    def __init__(
        self,
        clips: list[Clip],
        clip_length: int = 48,
        augment: bool = False,
        noise_std: float = 0.01,
        rotation_degrees: float = 10.0,
        scale_jitter: float = 0.10,
        temporal_crop: float = 0.85,
    ) -> None:
        self.clips = clips
        self.clip_length = clip_length
        self.augment = augment
        self.noise_std = noise_std
        self.rotation_degrees = rotation_degrees
        self.scale_jitter = scale_jitter
        self.temporal_crop = temporal_crop
        self.appearance = self._load_appearance(clips)
        self.appearance_frames = 8

    @staticmethod
    def _load_appearance(clips: list[Clip]) -> dict[str, np.ndarray]:
        """把各个数据集的外观特征一次性读进内存。

        不能在 __getitem__ 里按需读：那样每个样本都要把整个 npz 解压一遍，
        几千个片段乘几十个 epoch 会慢到没法训练。全部载入也就 30 MB 左右。
        """
        table: dict[str, np.ndarray] = {}
        for folder in sorted({os.path.dirname(clip.path) for clip in clips}):
            path = os.path.join(folder, "appearance.npz")
            if not os.path.exists(path):
                continue
            with np.load(path) as data:
                for key in data.files:
                    table[key] = data[key]
        return table

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (主目标骨架, 最近邻骨架, 逐帧关系量, 外观特征, 标签)。

        老格式的 .npz（没有邻人字段）会自动补零，所以只有公交车那批数据
        需要重新抽一遍，摔倒和奔跑的数据不用动。
        """
        clip = self.clips[index]
        with np.load(clip.path) as data:
            keypoints = data["keypoints"].astype(np.float32)
            bbox = data["bbox"].astype(np.float32) if "bbox" in data else None
            if "neighbor_keypoints" in data:
                neighbor = data["neighbor_keypoints"].astype(np.float32)
                pair = data["pair"].astype(np.float32)
            else:
                neighbor = np.zeros_like(keypoints)
                pair = np.zeros((keypoints.shape[0], 4), dtype=np.float32)

        stem = os.path.splitext(os.path.basename(clip.path))[0]
        appearance_raw = self.appearance.get(stem)
        if appearance_raw is None:
            appearance = np.zeros(
                (self.appearance_frames, 512), dtype=np.float32
            )
        else:
            appearance = appearance_raw.astype(np.float32)
            if appearance.shape[0] != self.appearance_frames:
                appearance = resample_temporal(appearance, self.appearance_frames)

        if self.augment:
            keypoints, bbox, neighbor, pair = self._augment(
                keypoints, bbox, neighbor, pair, allow_geometry=self.has_appearance
            )

        # 主目标和邻居共用同一套归一化参数，二者之间的相对位置才有意义
        centers, scale = normalization_params(keypoints, bbox)
        primary = apply_normalization(keypoints, centers, scale)
        neighbor_norm = apply_normalization(neighbor, centers, scale)

        # 像素距离换算成「人体身高」单位，和坐标同一量纲
        pair = pair.copy()
        pair[:, 0] /= scale
        pair[:, 2] /= scale
        # 没有邻人的帧把坐标清零，避免残留的绝对坐标被当成真实关节位置
        absent = pair[:, 3] < 0.5
        neighbor_norm[absent] = 0.0

        primary = resample_temporal(primary, self.clip_length)
        neighbor_norm = resample_temporal(neighbor_norm, self.clip_length)
        pair = resample_temporal(pair, self.clip_length)
        # 重采样会把「有没有邻人」插值成小数，这里重新二值化
        pair[:, 3] = (pair[:, 3] >= 0.5).astype(np.float32)

        return (
            torch.from_numpy(primary.transpose(2, 0, 1)).contiguous(),
            torch.from_numpy(neighbor_norm.transpose(2, 0, 1)).contiguous(),
            torch.from_numpy(pair.transpose(1, 0)).contiguous(),
            torch.from_numpy(appearance.transpose(1, 0)).contiguous(),
            torch.from_numpy(clip.labels.copy()),
        )

    @property
    def has_appearance(self) -> bool:
        return bool(self.appearance)

    def _augment(
        self,
        keypoints: np.ndarray,
        bbox: np.ndarray | None,
        neighbor: np.ndarray,
        pair: np.ndarray,
        allow_geometry: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        """几何变换必须同时作用在主目标和邻居身上。

        如果只翻转主目标而邻居不动，两者的相对位置就全错了——
        模型会学到「邻人永远在左边」这种假规律。所以这里镜像、旋转、缩放
        都只掷一次骰子，然后应用到所有输入上。

        allow_geometry 为假时跳过全部几何变换：外观特征是从裁剪出来的
        画面块上提前算好的 embedding，骨架翻转过之后外观没法跟着翻，
        两条支路会对不上。这种时候只保留关节点噪声和遮挡这两类不影响
        几何位置的增强。时序裁剪同理——它改变了时间窗，而外观特征是按
        整段固定的 8 帧算的，改了就对不齐。
        """
        rng = np.random.default_rng()
        frames = keypoints.shape[0]
        out = keypoints.copy()
        out_neighbor = neighbor.copy()
        out_pair = pair.copy()

        apply_geometry = allow_geometry

        # 水平镜像：左右关节一起换，x 坐标取负。
        # 距离、交并比、最近关节距在镜像和旋转下都不变，所以 pair 不用改
        if apply_geometry and rng.random() < 0.5:
            out = out[:, LEFT_RIGHT_PERMUTATION, :]
            out[..., 0] *= -1.0
            out_neighbor = out_neighbor[:, LEFT_RIGHT_PERMUTATION, :]
            out_neighbor[..., 0] *= -1.0
            if bbox is not None:
                bbox = bbox.copy()
                width = bbox[:, 2].copy()
                bbox[:, 0], bbox[:, 2] = -width, -bbox[:, 0]

        # 轻微平面旋转，模拟摄像头安装角度差异
        if apply_geometry and self.rotation_degrees > 0:
            angle = np.deg2rad(rng.uniform(-self.rotation_degrees, self.rotation_degrees))
            cos, sin = np.cos(angle), np.sin(angle)
            for target in (out, out_neighbor):
                x, y = target[..., 0].copy(), target[..., 1].copy()
                target[..., 0] = cos * x - sin * y
                target[..., 1] = sin * x + cos * y

        # 尺度抖动。坐标和像素距离要一起缩放，否则「两人相距几个身高」
        # 会和「人体多大」脱节，等于人为制造了矛盾的输入
        if apply_geometry and self.scale_jitter > 0:
            factor = 1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter)
            out[..., :2] *= factor
            out_neighbor[..., :2] *= factor
            out_pair[:, 0] *= factor
            out_pair[:, 2] *= factor

        # 关节噪声：两个人体分别加，他们本来就不是同一个人
        if self.noise_std > 0:
            out[..., :2] += rng.normal(0.0, self.noise_std, size=out[..., :2].shape)
            out_neighbor[..., :2] += rng.normal(
                0.0, self.noise_std, size=out_neighbor[..., :2].shape
            )

        # 随机遮挡少量关节，强迫模型不依赖单一关节
        if rng.random() < 0.3:
            hidden = rng.choice(out.shape[1], size=int(rng.integers(1, 4)), replace=False)
            out[:, hidden, 2] *= 0.0
            out_neighbor[:, hidden, 2] *= 0.0

        # 时间裁剪，模拟动作快慢差异
        if apply_geometry and self.temporal_crop < 1.0 and frames > 8:
            keep = max(int(frames * self.temporal_crop), 4)
            start = int(rng.integers(0, frames - keep + 1))
            out = out[start:start + keep]
            out_neighbor = out_neighbor[start:start + keep]
            out_pair = out_pair[start:start + keep]
            if bbox is not None:
                bbox = bbox[start:start + keep]

        return out, bbox, out_neighbor, out_pair


def compute_pos_weight(
    clips: list[Clip], num_labels: int = len(LABEL_COLUMNS)
) -> torch.Tensor:
    """按「负样本数 / 正样本数」给每个类别算损失权重。

    三个类的正样本比例差得很远（摔倒片段可能只占几个百分点），
    不加权的话模型会直接躺平全判 0，准确率还很高，但召回率是 0。
    """
    if not clips:
        return torch.ones(num_labels)
    labels = np.stack([clip.labels for clip in clips])
    positives = labels.sum(axis=0)
    weights = (labels.shape[0] - positives) / np.maximum(positives, 1.0)
    return torch.from_numpy(np.clip(weights, 1.0, 50.0).astype(np.float32))


def summarize(clips: list[Clip], columns: tuple[str, ...] = LABEL_COLUMNS) -> str:
    """打印数据集分布，训练前先看一眼心里有数。"""
    if not clips:
        return "（空）"
    labels = np.stack([clip.labels for clip in clips])
    total = labels.shape[0]
    lines = [f"共 {total} 个片段，{len({c.group for c in clips})} 段视频"]
    for index, column in enumerate(columns):
        count = int(labels[:, index].sum())
        lines.append(
            f"  {LABEL_NAMES_ZH[column]:<4} {count:>6}  ({count / total * 100:5.1f}%)"
        )
    normal = int((labels.sum(axis=1) == 0).sum())
    lines.append(f"  正常   {normal:>6}  ({normal / total * 100:5.1f}%)")
    return "\n".join(lines)
