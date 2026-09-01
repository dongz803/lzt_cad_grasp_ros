# FoundPose 基线接入与数据选择

## 已确定的路线

本仓库独立探索学习辅助路线，传统方法继续保留在
`/home/lzt/lzt_ros2_ws/src/lzt_mujoco_ros`。第一阶段使用 T-LESS 建立
FoundPose 标准基线；第二阶段使用 ROBI 检验反光工业件、真实深度噪声和跨数据集
泛化。铜柱只是 MuJoCo 系统联调对象，不再作为论文方法成立的前提。

FoundPose 的输出是物体相对相机的 6D 位姿 `T_camera_object`，不是机械臂末端抓取
位姿。项目中的完整变换链为：

```text
RGB/深度/实例掩膜 + CAD
        │
        ▼
FoundPose: T_camera_object（可保留 Top-K 候选）
        │  相机外参
        ▼
T_world_object
        │  CAD 坐标系抓取模板 T_object_grasp
        ▼
T_world_grasp
        │  取料、放料、IK、碰撞和不确定度联合筛选
        ▼
最终上下料轨迹
```

因此，FoundPose 是位姿感知前端；论文的抓取创新应落在“对称性/不确定性感知的
CAD 抓取模板生成与取放联合排序”，而不是把 FoundPose 的物体位姿称为抓取位姿。

## 为什么先选 T-LESS

T-LESS 是 BOP 原生格式的无纹理工业物体数据集，可直接对齐 FoundPose 和 BOP
Toolkit 的流程，适合先回答“官方基线能否复现、接口是否正确”。第一阶段只下载：

- T-LESS base；
- T-LESS models；
- T-LESS BOP19 test subset；
- BOP 2023 Task 4 的 CNOS-FastSAM 默认检测/分割结果。

不下载 PBR 训练集，也不进行 FoundPose 专项训练。上述三个 T-LESS 压缩包约
0.82 GB；检测结果另计。先运行标准测试子集，避免一开始下载全量测试与训练数据。

推荐目录：

```text
/home/lzt/lzt_ros2_ws/datasets/bop/
├── tless/
│   ├── camera_primesense.json
│   ├── models/
│   ├── models_eval/
│   └── test_primesense/
└── detections/
    └── cnos-fastsam/
        └── cnos-fastsam_tless_test.json
```

数据不放进 Git 仓库。实际目录结构以 BOP 压缩包内容和 FoundPose 配置为准，解压后
先用 BOP Toolkit 校验，再生成 T-LESS 专用的模板、对象表征和推理配置。

## ROBI 的定位

ROBI 更接近项目最终应用：包含 7 种反光工业物料、双相机数据和真实 6D 标注，
但不是 FoundPose 开箱即用的 BOP 目录。因此放在第二阶段，先编写 ROBI→BOP
适配器，再做泛化实验。首批只取两个对象：

- `Chrome Screw`：反光、细长，适合暴露 RGB-D 和姿态估计问题；
- `Gear`：对称性明显，适合验证对称等价位姿和抓取排序。

适配完成并通过单元测试后，再扩展到全部 7 类，避免把“数据格式问题”和“算法
效果问题”混在同一次实验中。

## 环境隔离

官方 CUDA 环境使用独立 Conda 环境（Python 3.9、PyTorch 2.3、CUDA 11.7），
ROS 2 Humble 包继续使用工作空间自己的 Python 环境。二者第一版通过 BOP JSON
和图像文件交换，不在同一进程中导入，从而规避 Python、PyTorch 和 ROS ABI 冲突。

RTX 3060 的保守默认配置为 DINOv2 ViT-S、单图批量 1。FoundPose 本身无需训练；
如果后续训练语义/实例分割或学习式抓取评分器，可以再使用显存更大的电脑或云服务器。
若本机是 12 GB 桌面版，可以在完成 ViT-S 基线后再测试更大的骨干网络；若是 6 GB
笔记本版，则继续以 ViT-S 和离线特征缓存为主。

## 官方流程映射

FoundPose 官方开源流程依次为：

```bash
python scripts/gen_templates.py --opts-path <tless-template-config.json>
python scripts/gen_repre.py --opts-path <tless-representation-config.json>
python scripts/infer.py --opts-path <tless-inference-config.json>
python scripts/prepare_bop_submission.py
```

正式执行前必须完成：官方源码及递归子模块下载、独立 Conda 环境创建、`BOP_PATH`
设置、CNOS 默认掩膜就位，以及 T-LESS 配置生成。公开仓库不含论文中的
featuremetric refinement，因此实验中要明确称为“FoundPose 开源粗位姿基线”。

## 下一步验收顺序

1. 下载并记录 FoundPose 源码提交号，创建 `foundpose_gpu` 环境；
2. 下载 T-LESS 最小集合与 CNOS 默认掩膜，并通过目录完整性检查；
3. 在少量 T-LESS 图像上跑通模板生成、对象表征和推理；
4. 运行完整 BOP19 test subset，记录 BOP AR、单帧耗时和显存峰值；
5. 将 BOP 输出转换为本包的 `PoseHypothesis`，保留置信度和对称等价候选；
6. 接入 CAD 抓取模板，开始取料—放料约束下的抓取创新实验；
7. 编写 ROBI→BOP 适配器，用 Chrome Screw 和 Gear 做跨数据集验证。

## 官方来源与许可

- FoundPose 官方仓库：<https://github.com/facebookresearch/foundpose>
- BOP 数据集：<https://bop.felk.cvut.cz/datasets/>
- ROBI 数据集：<https://www.trailab.utias.utoronto.ca/robi>
- ROBI Toolkit：<https://github.com/junyang224/ROBI_Toolkit>

FoundPose 使用 CC BY-NC 4.0；ROBI 使用 CC BY-NC-SA 4.0。用于论文研究没有路线
冲突，但未来若面向商业部署，需要重新审查代码、数据和模型许可。
