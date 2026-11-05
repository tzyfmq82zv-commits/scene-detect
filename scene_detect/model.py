"""骨架动作识别模型：ST-GCN，多标签输出（奔跑 / 摔倒 / 打架）。

为什么用多标签而不是四分类：
  1. 一个人可以同时是「奔跑」和「打架」，四分类会强迫模型二选一；
  2. 三个头各自的判定阈值可以独立调，方便在召回率和误报之间做取舍；
  3. 「正常」不需要单独建模——三个头都不触发就是正常。

输入张量：(N, C, T, V)
    N 批大小, C=3 通道(x 坐标, y 坐标, 置信度), T 帧数, V=17 个 COCO 关节
输出张量：(N, 3) logits，依次对应 [奔跑, 摔倒, 打架]
    训练时直接喂给 BCEWithLogitsLoss；
    推理时先 sigmoid 再和各自阈值比较。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# COCO-17 关键点顺序（ultralytics YOLO-Pose 输出顺序）
COCO_KEYPOINTS = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# 骨架连接关系，直接决定空间图卷积里信息怎么在关节之间流动
COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # 头部
    (3, 5), (4, 6), (5, 6),                    # 肩颈
    (5, 7), (7, 9), (6, 8), (8, 10),           # 双臂
    (5, 11), (6, 12), (11, 12),                # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),    # 双腿
]

# 肢体分组，用来给「肢体交叠程度」这类交互特征取关节子集（打架判定会用到）
LIMB_GROUPS = {
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
    "torso": [5, 6, 11, 12],
    "head": [0, 1, 2, 3, 4],
}


def build_adjacency(num_joints: int = 17, edges=None) -> np.ndarray:
    """构造归一化的骨架邻接矩阵（含自环），形状 (V, V)。

    归一化方式和 ST-GCN 论文一致：D^-1/2 * (A + I) * D^-1/2，
    这样度数高的关节（比如肩膀）不会在网络里产生过大的激活值。
    """
    if edges is None:
        edges = COCO_SKELETON_EDGES
    adjacency = np.eye(num_joints, dtype=np.float32)
    for src, dst in edges:
        if src < num_joints and dst < num_joints:
            adjacency[src, dst] = 1.0
            adjacency[dst, src] = 1.0
    degree = adjacency.sum(axis=1)
    # 先把度为 0 的位置替换成 1 再取幂，否则这些位置会留下未初始化的内存
    safe_degree = np.where(degree > 0, degree, 1.0)
    degree_inv_sqrt = np.power(safe_degree, -0.5)
    degree_inv_sqrt[degree == 0] = 0.0
    normalized = adjacency * degree_inv_sqrt[:, None] * degree_inv_sqrt[None, :]
    return normalized.astype(np.float32)


class SpatialGraphConv(nn.Module):
    """空间图卷积：先在关节之间按骨架拓扑聚合信息，再用 1x1 卷积做通道变换。

    除了固定的骨架邻接矩阵，还加了一个可学习的残差矩阵 B（2s-AGCN 的做法）。
    固定拓扑只能表达「哪些关节物理相连」，而像「打架」这种动作，
    真正有判别力的关系可能是「左手腕和对方肩膀」，需要网络自己学出来。
    """

    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("adjacency", torch.from_numpy(adjacency))
        self.learnable_adjacency = nn.Parameter(torch.zeros_like(self.adjacency))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        topology = self.adjacency + self.learnable_adjacency
        aggregated = torch.einsum("nctv,vw->nctw", x, topology)
        return self.conv(aggregated)


class STGCNBlock(nn.Module):
    """一个时空块：空间图卷积（关节之间）+ 时间卷积（帧之间）+ 残差连接。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        temporal_kernel: int = 9,
        temporal_stride: int = 1,
        residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, adjacency)
        padding = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(temporal_stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual: nn.Module = nn.Identity()
        elif in_channels == out_channels and temporal_stride == 1:
            self.residual = nn.Identity()
        else:
            # 通道数或时间长度变了，残差支路也要跟着变换
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(temporal_stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        # 第一层输入 3 通道、输出 64 通道，两边维度对不上，不能做残差相加。
        # 这里单独记一个开关：residual=False 时必须彻底跳过相加，
        # 而不是退化成 Identity——Identity 会原样返回 3 通道的输入，
        # 和 64 通道的输出相加就会报维度不匹配。
        self.use_residual = residual
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn(self.gcn(x))
        if self.use_residual:
            out = out + self.residual(x)
        return self.relu(out)


class STGCNBackbone(nn.Module):
    """骨架时空特征提取器，输出与帧数无关的定长向量。

    单独抽出来是为了让主目标和「邻居」共用同一套权重——两者的输入都是
    17 个关节的骨架序列，用同一个提取器既能省参数，又能保证两者被映射到
    同一个特征空间里，后续拼接才有意义。
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 3,
        channels: tuple[int, ...] = (64, 64, 64, 128, 128, 128, 256, 256, 256),
        temporal_strides: tuple[int, ...] = (1, 1, 1, 2, 1, 1, 2, 1, 1),
        dropout: float = 0.3,
        edges=None,
    ) -> None:
        super().__init__()
        if len(channels) != len(temporal_strides):
            raise ValueError("channels 和 temporal_strides 长度必须一致")

        adjacency = build_adjacency(num_joints, edges)
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.out_channels = channels[-1]

        # 输入的 x/y 坐标尺度差异很大，先在(通道×关节)维度上做一次批归一化
        self.input_bn = nn.BatchNorm1d(in_channels * num_joints)

        blocks: list[nn.Module] = []
        previous = in_channels
        for index, (out_channels, stride) in enumerate(zip(channels, temporal_strides)):
            blocks.append(
                STGCNBlock(
                    previous,
                    out_channels,
                    adjacency,
                    temporal_stride=stride,
                    residual=index > 0,  # 第一层输入通道数不同，不加残差
                    dropout=dropout if index in (4, 7) else 0.0,
                )
            )
            previous = out_channels
        self.blocks = nn.ModuleList(blocks)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        # 可学习邻接矩阵从零起步，训练初期等价于纯骨架拓扑，比较稳
        for module in self.modules():
            if isinstance(module, SpatialGraphConv):
                nn.init.constant_(module.learnable_adjacency, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        batch, channels, frames, joints = x.shape
        out = x.permute(0, 1, 3, 2).reshape(batch, channels * joints, frames)
        out = self.input_bn(out)
        out = out.reshape(batch, channels, joints, frames).permute(0, 1, 3, 2).contiguous()

        for block in self.blocks:
            out = block(out)

        # 对时间和关节同时做平均池化，得到与帧数无关的定长表示
        return out.mean(dim=(2, 3))


class STGCN(nn.Module):
    """只看单人的骨架时空网络，用作基线。

    neighbor 和 pair 两个参数是为了和交互版模型保持同样的调用签名，
    这里直接忽略——本模型看不到旁边的人。
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 3,
        num_classes: int = 3,
        channels: tuple[int, ...] = (64, 64, 64, 128, 128, 128, 256, 256, 256),
        temporal_strides: tuple[int, ...] = (1, 1, 1, 2, 1, 1, 2, 1, 1),
        dropout: float = 0.3,
        edges=None,
    ) -> None:
        super().__init__()
        self.backbone = STGCNBackbone(
            num_joints=num_joints,
            in_channels=in_channels,
            channels=channels,
            temporal_strides=temporal_strides,
            dropout=dropout,
            edges=edges,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_channels, num_classes),
        )

    def forward(self, x, neighbor=None, pair=None, appearance=None) -> torch.Tensor:
        return self.head(self.backbone(x))


class STGCNInteraction(nn.Module):
    """带交互建模的骨架网络：主目标 + 最近邻人 + 两人之间的关系量。

    这是针对「打架」的核心改动。纯单人骨架抓不到打架，因为打架是两个人之间
    的关系：同样的挥手动作，打人和打招呼在单人体特征上几乎一样，
    区别在于另一个人是否在附近、是否在同步晃动、肢体有没有交叠。

    三路输入：
        x        主目标骨架          (N, 3, T, 17)
        neighbor 最近邻人骨架        (N, 3, T, 17)  与主目标同一坐标系
        pair     逐帧关系量          (N, 4, T)      距离 / 框交并比 / 最近关节距 / 是否存在邻人

    主目标和邻人共用同一个 backbone。没有邻人的帧会把邻人特征乘零，
    所以单人场景（摔倒、奔跑）不会因此变差——这一路输入直接退化成常数。
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 3,
        num_classes: int = 2,
        pair_channels: int = 4,
        channels: tuple[int, ...] = (64, 64, 64, 128, 128, 128, 256, 256, 256),
        temporal_strides: tuple[int, ...] = (1, 1, 1, 2, 1, 1, 2, 1, 1),
        dropout: float = 0.3,
        edges=None,
    ) -> None:
        super().__init__()
        self.backbone = STGCNBackbone(
            num_joints=num_joints,
            in_channels=in_channels,
            channels=channels,
            temporal_strides=temporal_strides,
            dropout=dropout,
            edges=edges,
        )
        self.pair_encoder = nn.Sequential(
            nn.Conv1d(pair_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        fused = self.backbone.out_channels * 2 + 64
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, neighbor=None, pair=None, appearance=None) -> torch.Tensor:
        primary = self.backbone(x)
        if neighbor is None or pair is None:
            # 没给交互输入就当单人模型用，便于拿同一份权重做消融
            zeros = torch.zeros_like(primary)
            fused = torch.cat([primary, zeros, torch.zeros(
                primary.size(0), 64, device=primary.device, dtype=primary.dtype
            )], dim=1)
            return self.head(fused)

        # pair 的第 4 个通道是「这一帧有没有邻人」，用来把邻人特征门控掉
        presence = pair[:, 3:4, :].mean(dim=2)          # (N, 1)
        neighbor_feature = self.backbone(neighbor) * presence
        pair_feature = self.pair_encoder(pair)          # (N, 64)
        fused = torch.cat([primary, neighbor_feature, pair_feature], dim=1)
        return self.head(fused)


class DualStream(nn.Module):
    """双流模型：骨架看动作，RGB 外观看画面。

    为什么需要第二条支路：骨架模态在打架这件事上已经被否定了——运动量、
    两两距离、最近关节距在暴力和非暴力片段上的分布完全重合，分类 F1 只有
    0.57。而同样一批片段，只用 RGB 外观的 AUROC 就有 0.81（按时间块划分
    后仍有 0.81，排除了相邻片段泄漏）。所以打架得靠外观，摔倒和奔跑骨架够用。

    两条支路各自出特征再拼接，而不是把外观塞进骨架网络：
    两者的时间尺度和数据结构完全不同，混在一起训容易一边压倒另一边。

    输入：
        x           骨架       (N, 3, T, 17)
        appearance  外观特征   (N, 512, K)   K 是采样帧数（默认 8）
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 3,
        num_classes: int = 2,
        appearance_dim: int = 512,
        appearance_hidden: int = 128,
        channels: tuple[int, ...] = (64, 64, 64, 128, 128, 128, 256, 256, 256),
        temporal_strides: tuple[int, ...] = (1, 1, 1, 2, 1, 1, 2, 1, 1),
        dropout: float = 0.3,
        edges=None,
    ) -> None:
        super().__init__()
        self.appearance_hidden = appearance_hidden
        self.skeleton_backbone = STGCNBackbone(
            num_joints=num_joints,
            in_channels=in_channels,
            channels=channels,
            temporal_strides=temporal_strides,
            dropout=dropout,
            edges=edges,
        )
        # 8 帧的 512 维特征，在时间轴上做卷积把整段时间的变化压成一个向量
        self.appearance_encoder = nn.Sequential(
            nn.Conv1d(appearance_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, appearance_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(appearance_hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        fused = self.skeleton_backbone.out_channels + appearance_hidden
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def fuse_features(self, x, appearance=None) -> torch.Tensor:
        skeleton_feature = self.skeleton_backbone(x)
        if appearance is None or appearance.numel() == 0:
            # 没有外观输入时退化成人只看骨架，方便做消融对比
            appearance_feature = torch.zeros(
                skeleton_feature.size(0),
                self.appearance_hidden,
                device=skeleton_feature.device,
                dtype=skeleton_feature.dtype,
            )
        else:
            appearance_feature = self.appearance_encoder(appearance)
        return torch.cat([skeleton_feature, appearance_feature], dim=1)

    def forward(self, x, neighbor=None, pair=None, appearance=None) -> torch.Tensor:
        return self.head(self.fuse_features(x, appearance))

    def forward_with_features(
        self, x, neighbor=None, pair=None, appearance=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """同时返回 logits 和融合特征，供专用状态分类器复用同一次前向。"""
        fused = self.fuse_features(x, appearance)
        return self.head(fused), fused


class TemporalCNN(nn.Module):
    """轻量基线：不做图卷积，把每帧的关节坐标拉平后用一维卷积扫时间轴。

    结构比 ST-GCN 简单得多，训练也快。作用有两个：
    一是当基线，如果 ST-GCN 打不过它，说明我们的骨架邻接设计没带来收益；
    二是当备份方案，数据量还小的时候它往往更稳。
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 3,
        num_classes: int = 3,
        hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm1d(num_joints * in_channels)
        self.network = nn.Sequential(
            nn.Conv1d(num_joints * in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, neighbor=None, pair=None) -> torch.Tensor:
        # 和另外两个模型保持同样的调用签名；本基线看不到邻人，直接忽略后两个输入
        batch, channels, frames, joints = x.shape
        # BatchNorm1d 对三维输入是沿第 1 维归一化的，
        # 所以先把特征维搬到第 1 位，(N, T, C*V) -> (N, C*V, T)
        out = x.permute(0, 2, 1, 3).reshape(batch, frames, channels * joints)
        out = out.permute(0, 2, 1)
        out = self.input_bn(out)
        return self.head(self.network(out))


def build_model(name: str = "stgcn", **kwargs) -> nn.Module:
    """按名字建模型，训练脚本通过 --model 参数调用。"""
    name = name.lower()
    if name == "stgcn":
        return STGCN(**kwargs)
    if name in {"stgcn_interaction", "interaction", "stgcn2"}:
        return STGCNInteraction(**kwargs)
    if name in {"dual", "dualstream", "dual_stream"}:
        return DualStream(**kwargs)
    if name in {"tcn", "temporalcnn", "baseline"}:
        return TemporalCNN(**kwargs)
    raise ValueError(f"未知模型：{name}（可选 dual / stgcn_interaction / stgcn / tcn）")


if __name__ == "__main__":
    # 自检：打印参数量和一次前向的形状，确认张量维度对得上
    batch, frames, joints = 8, 48, 17
    skeleton = torch.randn(batch, 3, frames, joints)
    neighbor = torch.randn(batch, 3, frames, joints)
    pair = torch.rand(batch, 4, frames)
    for model_name in ("stgcn_interaction", "stgcn", "tcn"):
        model = build_model(model_name, num_classes=2)
        output = model(skeleton, neighbor, pair)
        params = sum(p.numel() for p in model.parameters())
        print(
            f"{model_name:18s} 参数量 {params/1e6:.2f}M  "
            f"输入 {tuple(skeleton.shape)} 输出 {tuple(output.shape)}"
        )

    # 邻人通道全零（单人场景）时不应该报错，也不该产生 NaN
    model = build_model("stgcn_interaction", num_classes=2).eval()
    with torch.no_grad():
        out = model(skeleton, torch.zeros_like(neighbor), torch.zeros_like(pair))
    print(f"邻人为空时输出 {tuple(out.shape)}，含 NaN = {bool(torch.isnan(out).any())}")
