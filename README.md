# Scene Detection

面向视频的实时人体动作检测项目。当前主线是九动作识别，并提供可选的
Qwen2.5-VL 事件复核；仓库同时保留一个独立的 `Normal / Wall crossing`
攀爬二分类支路。

> 模型权重、训练数据、测试视频和推理输出不在仓库中。请按本文说明在本地准备。

## 功能

- YOLO11n-Pose 同时检测人体框和 17 个 COCO 关键点
- BoT-SORT 多人跟踪
- ST-GCN 骨架分支
- ResNet18 + 时间卷积外观分支
- 九动作 softmax 主分类器
- 奔跑、走路、踢腿专用二级分类器
- 动作稳定、投票、冷却和物理运动约束
- PySide6 Qt 实时界面
- 可选 Qwen2.5-VL 事件复核
- 命令行标注视频和 JSON 事件输出
- 独立的攀爬二分类训练、评估和推理工具

## 九动作类别

```text
running       奔跑
falling       摔倒
fighting      打架
sitting       坐
walking       走
kicking       踢腿
waving        挥手
nodding       点头
handshaking   握手
```

没有动作事件时，界面显示为 `Normal`。默认九分类模型没有独立
`normal` 输出头；实验性十类模型仅用于后续研究。

## 整体架构

```text
视频帧
  |
  +-> YOLO11n-Pose：人体框 + 17 关键点
  |       |
  |       +-> BoT-SORT：稳定 Track ID
  |
  +-> 每条轨迹最近 32 帧
          |
          +-> 骨架归一化 -> ST-GCN -------------> 256 维
          |
          +-> 8 帧人体裁剪 -> ResNet18 -> TCN --> 128 维
                                                    |
                                              拼接 384 维
                                                    |
                                        九分类 softmax
                                                    |
                              +---------------------+
                              |
                 奔跑/走路/踢腿专用分类器
                              |
                  动作稳定、投票、冷却、运动约束
                              |
                   Qt 界面 / JSON 事件 / 报警声音
```

| 模块 | 模型 | 说明 |
|---|---|---|
| 人体检测 | YOLO11n-Pose | 1 个 person 类，17 个关键点 |
| 跟踪 | BoT-SORT | 默认 `botsort.yaml` |
| 骨架特征 | ST-GCN | 9 个时空图卷积块，输出 256 维 |
| 外观特征 | ImageNet ResNet18 | 每帧 512 维，默认 8 帧 |
| 外观时序编码 | Conv1d | 512 -> 256 -> 128 |
| 主分类器 | DualStream | 九动作 softmax |
| 专用分类器 | DualStream | 奔跑、走路、踢腿 |
| VLM | Qwen2.5-VL-3B-Instruct | 可选，4-bit 推理 |
| 攀爬支路 | ResNet18 + 时间注意力池化 | Normal / Wall crossing |

主分类器约 313 万参数：

```text
骨架输入：(batch, 3, 32, 17)
外观输入：(batch, 512, 8)
```

## 实时推理流程

1. YOLO11n-Pose 检测人和关键点。
2. BoT-SORT 分配轨迹 ID。
3. 每条轨迹缓存关键点和检测框。
4. 每 6 帧执行一次动作分类。
5. 主分类器输出九类概率。
6. 奔跑、走路、踢腿发生歧义时调用专用分类器。
7. 动作需连续一致才允许切换。
8. 低优先级动作执行跨类冷却。
9. 投票和物理运动条件共同决定事件。
10. Qt 显示标签、事件、报警和声音。

## 仓库结构

```text
.
├── README.md
├── nine_holdout_sources.txt
├── run_qt_test.ps1
├── test_nine_full.ps1
├── train_nine_full.ps1
├── train_locomotion.ps1
├── train_nine_normal.ps1
├── train_kick_normal.ps1
└── scene_detect
    ├── app_qt.py
    ├── pipeline.py
    ├── model.py
    ├── dataset.py
    ├── train.py
    ├── run_video.py
    ├── running_rule.py
    ├── vlm_confirmer.py
    ├── train_climb.py
    ├── run_climb.py
    ├── evaluate_climb.py
    └── adapters
        ├── prepare_hmdb_videos.py
        ├── build_yolo17_nine_manifest.py
        ├── extract_new.py
        ├── extract_appearance_new.py
        ├── make_nine_action_test_video.py
        ├── evaluate_timeline.py
        ├── vlm_check.py
        ├── vlm_batch_validate.py
        └── ...
```

## 环境

项目主要在 Windows + NVIDIA GPU 环境运行。已验证：

- Python 3.11
- PyTorch 2.11 + CUDA 12.8
- TorchVision 0.26
- Ultralytics 8.x
- OpenCV 5.x
- PySide6
- Transformers、Accelerate、bitsandbytes、qwen-vl-utils

参考安装：

```powershell
conda create -n SC python=3.11 -y
conda activate SC

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python PySide6 numpy pandas
pip install transformers accelerate bitsandbytes qwen-vl-utils sentencepiece pillow
```

50 系显卡必须使用 CUDA 12.8 或更新版本的 PyTorch。

## 本地文件准备

仓库不包含模型和数据。

### 人体检测权重

```text
scene_detect/yolo11n-pose.pt
```

### 九动作权重

默认主模型：

```text
scene_detect/runs_nine_full_softmax/best.pt
```

专用分类器：

```text
scene_detect/runs_locomotion_softmax/best.pt
```

实验模型：

```text
scene_detect/runs_nine_normal_softmax/best.pt
scene_detect/runs_kick_normal_softmax/best.pt
```

### VLM

本地模型目录示例：

```text
D:\Models\Qwen2.5-VL-3B-Instruct
```

目录必须包含完整权重、tokenizer 和 `config.json`。

### 九动作数据

主要来源：

- HMDB51：run、fall_floor、punch、sit、walk、kick、kick_ball、wave、shake_hands
- NTHU-DDD：点头
- 测试数据：GMDCSA24 摔倒片段

默认特征目录：

```text
D:\SceneDetection\features\yolo17_nine_full
```

含正常类的实验目录：

```text
D:\SceneDetection\features\yolo17_nine_normal_full
```

训练脚本中的 `D:\SceneDetection`、`D:\HMDB51` 是本机默认路径，
迁移到其他机器时必须修改。

## Qt 界面

启动：

```powershell
python scene_detect\app_qt.py
```

当前默认会：

- 加载九动作模型
- 加载奔跑/走路/踢腿分类器
- 启用 VLM 二次确认
- 如果默认测试视频存在，自动加载并开始

关闭 VLM、只运行轻量实时检测：

```powershell
python scene_detect\app_qt.py --no-vlm
```

指定视频和分类器：

```powershell
python scene_detect\app_qt.py `
  --source "D:\path\video.mp4" `
  --classifier ".\scene_detect\runs_nine_full_softmax\best.pt"
```

Windows 下可以直接双击 `app_qt.py`。入口会尝试使用当前机器 `SC`
环境的 `pythonw.exe`；迁移到其他机器时需要修改入口中的 Python 路径。

### Qt 性能策略

- 推理在独立 QThread 中执行
- 界面最多每 40 ms 重绘一次，只显示最新帧
- QImage 转换和显示缩放尽量在后台线程完成
- 实时概率面板限制刷新频率
- 检测使用 640 像素输入
- 如果原视频名以 `_no_gap` 结尾，Qt 会优先尝试同名 `_optical`
  或 `_smooth` 视频作为显示源；检测仍读取原始视频

## 命令行推理

```powershell
python scene_detect\run_video.py `
  --source "D:\path\video.mp4" `
  --output ".\out.mp4" `
  --events ".\events.json" `
  --imgsz 640 `
  --temporal-confirmations 3
```

默认输出目录：

```text
D:\SceneDetection\output
```

## 九动作训练

### 全量主模型

```powershell
.\train_nine_full.ps1 -Epochs 40 -BatchSize 16
```

该脚本依次执行：

1. HMDB51 视频转换
2. 九动作清单生成
3. YOLO17 骨架特征提取
4. ResNet18 外观特征提取
5. 九分类 softmax 训练

固定测试来源会通过以下文件排除：

```text
nine_holdout_sources.txt
```

### 奔/走/踢专用分类器

```powershell
.\train_locomotion.ps1 -Epochs 30 -BatchSize 16
```

### 正常类实验

十类 `九动作 + normal`：

```powershell
.\train_nine_normal.ps1
```

`kicking / normal` 二分类：

```powershell
.\train_kick_normal.ps1
```

这两个模型目前不作为默认主模型。

`scene_detect/train.py` 支持：

- `--single-label`
- 交叉熵 softmax
- 类别频率加权
- `--group-column video_uid`
- `--init-from`
- `--freeze-epochs`
- 每类阈值扫描

## 测试与验收

先准备测试视频，再运行：

```powershell
.\test_nine_full.ps1 -TemporalConfirmations 3
```

输出：

```text
*_annotated.mp4
*_events.json
*_summary.json
```

`evaluate_timeline.py` 统计：

- 九动作覆盖
- 正确和错误动作事件
- 正常段误报
- 过渡段误报

最近一次本机固定测试集结果：

```text
动作覆盖：9/9
正常段误报：0
过渡段误报：0
错误动作事件：3
自动推理速度：约 42 fps
```

这只是固定合成测试集结果，不代表现场部署精度。

## VLM 二次确认

相关文件：

```text
scene_detect/vlm_confirmer.py
scene_detect/adapters/vlm_check.py
scene_detect/adapters/vlm_batch_validate.py
scene_detect/adapters/vlm_architecture.py
```

单段视频：

```powershell
python scene_detect\adapters\vlm_check.py `
  --source "D:\path\video.mp4" `
  --start 10 `
  --duration 5 `
  --frames 4 `
  --load-in-4bit
```

批量验证：

```powershell
python scene_detect\adapters\vlm_batch_validate.py `
  --manifest ".\vlm_validation_six\manifest.csv" `
  --load-in-4bit `
  --output ".\vlm_validation_six\results"
```

注意事项：

- Qwen2.5-VL 不是实时主分类器
- 8GB 显存建议使用 NF4 4-bit
- 不建议同时让 VLM、YOLO 和动作模型占满显存
- 当前 VLM 验证样本较少，旧版结果不足以证明九类可靠性
- VLM 更适合离线复核、伪标签筛选和低置信度候选确认

## 攀爬二分类支路

类别：

```text
normal
wall_crossing
```

流程：

```powershell
python scene_detect\adapters\prepare_climb.py
python scene_detect\adapters\cache_climb_frames.py
python scene_detect\train_climb.py
python scene_detect\evaluate_climb.py
python scene_detect\run_climb.py --source "D:\path\video.mp4"
```

难负样本挖掘：

```powershell
python scene_detect\adapters\mine_climb_hard_negatives.py
```

模型输出在 `scene_detect/runs_climb`，默认不提交。

## 已知限制

- 九动作训练主要来自公开数据集，真实场景仍存在域偏移
- 奔跑和走路仍可能混淆
- 打架依赖多人关系和外观线索
- 小目标、远景人物和低帧率来源会降低效果
- 摔倒检测可能受场景偏置影响
- VLM 推理较慢，不适合逐帧实时运行
- Windows Smart App Control、AppLocker 或企业代码完整性策略可能阻止
  `torch_cuda.dll` 等未签名依赖


端到端验收。

