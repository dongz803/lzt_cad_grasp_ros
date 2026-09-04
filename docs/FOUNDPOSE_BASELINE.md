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

当前实际目录：

```text
/home/lzt/lzt_ros2_ws/datasets/bop/
├── tless/
│   ├── camera_primesense.json
│   ├── models_cad/
│   ├── models_eval/
│   ├── models_reconst/
│   └── test_primesense/
└── detections/
    └── cnos-fastsam/
        └── cnos-fastsam_tless-test.json
```

数据不放进 Git 仓库。上述命名与 FoundPose 当前代码及 BOP 压缩包保持一致。

## 已跑通的小样本

2026-09-01 已在本机完成一个端到端 smoke test：T-LESS 对象 1、场景 20、图像 3。
该样本使用 CNOS-FastSAM 默认实例掩膜和 DINOv2 ViT-S 特征。为快速验证安装，模板
配置只采样 9 个视点和 4 个平面内旋转，共生成 36 个模板；PCA 使用 128 维，FAISS
使用 256 个视觉词。这些参数只证明流程连通，不代表论文实验或 FoundPose 标准精度。

隔离环境与数据位置：

```text
/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu
/home/lzt/lzt_ros2_ws/datasets/bop
/home/lzt/lzt_ros2_ws/datasets/bop_smoke
/home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros/third_party/foundpose
```

可复现命令：

```bash
cd /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros

/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python \
  scripts/prepare_tless_smoke.py \
  --source /home/lzt/lzt_ros2_ws/datasets/bop \
  --destination /home/lzt/lzt_ros2_ws/datasets/bop_smoke

export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop_smoke
export BOP_OUTPUT_PATH=/home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros/outputs/foundpose_smoke
export PYOPENGL_PLATFORM=egl
export FOUNDPOSE_PYTHON=/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python

$FOUNDPOSE_PYTHON scripts/run_foundpose.py gen_templates \
  --opts-path config/foundpose_smoke/tless_templates.json
$FOUNDPOSE_PYTHON scripts/run_foundpose.py gen_repre \
  --opts-path config/foundpose_smoke/tless_repre.json
$FOUNDPOSE_PYTHON scripts/run_foundpose.py infer \
  --opts-path config/foundpose_smoke/tless_infer.json
```

首次连通性验证得到 195 组查询—模板特征对应并保存最佳 1 个粗位姿；随后扩展为
真实 Top-5 PnP 粗位姿，5 个候选不再由对称等价旋转伪造。Top-5 单图任务的墙钟
时间为 7.81 秒、增量峰值显存 828 MiB（包含初始化）。最佳候选 MSSD 为
34.314 mm、MSPD 为 32.732 px。由于这里只用了 36 个稀疏模板，数值不能替代完整
BOP 评测。

输出文件位于：

```text
outputs/foundpose_smoke/inference/tless_smoke_v1/1/estimated-poses.json
outputs/foundpose_smoke/inference/tless_smoke_v1/1/pose-hypotheses-top5.json
```

## 标准 BOP19 状态

标准配置位于 `config/foundpose_bop19/`。正式基线保持 FoundPose 上游的
`best_coarse`，Top-K 使用单独的 `tless_infer_topk.json`，避免研究接口改变官方
基线的提交排名。全量数据检查结果为 30 个 CAD、20 个场景、1000 张目标图像、
4904 条目标项和 6423 个目标实例，缺失文件为 0。

截至 2026-09-04，对象 1–5 均已完成标准密度模板、对象表示、完整测试推理、
可视化和官方 BOP19 评价，共输出 1248 个 6D 位姿。标准模板完成 19/30；按当前
实验安排，对象 20–30 的模板生成暂停，已有中间文件保留。完整对象链路完成
5/30，最终基线尚未冻结。

物体 2–4 的本次结果如下：

- 对象 2：134216 个 256 维对象特征，表示耗时 54.32 秒；推理输出 203/441 个
  目标实例，耗时 119.80 秒；中位 MSSD/MSPD 为 27.17 mm/17.34 px；单类官方
  AR 为 0.2242（VSD 0.1259、MSSD 0.1474、MSPD 0.3993）。
- 对象 3：139374 个 256 维对象特征，表示耗时 54.46 秒；推理输出 304/384 个
  目标实例，耗时 192.02 秒；中位 MSSD/MSPD 为 17.29 mm/17.28 px；单类官方
  AR 为 0.4694（VSD 0.3267、MSSD 0.3714、MSPD 0.7102）。
- 对象 4：115986 个 256 维对象特征，表示耗时 46.97 秒；推理输出 270/594 个
  目标实例，耗时 83.64 秒；中位 MSSD/MSPD 为 31.88 mm/19.54 px；单类官方
  AR 为 0.2075（VSD 0.1220、MSSD 0.1466、MSPD 0.3539）。

这里的单类官方 AR 从 BOP Toolkit 各阈值分数文件的 `obj_recalls` 提取并平均。
单对象提交的顶层 `scores_bop19.json` 会把其余 29 类计为零，因此只用于确认完整
评估链路，不能直接引用为单类分数或最终 30 类论文基线。

本机桌面环境占用约 2.3 GiB 显存时，FoundPose 默认 GPU FAISS 聚类会因额外申请
1 GiB 临时空间而 OOM。对象 2–4 改为 DINOv2/PCA 保持 GPU、相同 2048 类 FAISS
K-means 改在 CPU 执行；算法配置不变，表示阶段 GPU 总峰值为 4495–4681 MiB。
该兼容修改保存在 `third_party/patches/foundpose_cpu_kmeans.patch` 并纳入最终冻结
清单。

批处理入口 `scripts/run_bop19_stage.py` 按对象保存配置和 profile，重跑时自动跳过
完整对象。最终只有在 30 个对象的三个阶段、官方五项 score 字段和结果 CSV 均
齐全时，`scripts/freeze_foundpose_baseline.py` 才允许生成 `FROZEN_BASELINE.json`。

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

1. [x] 下载并记录 FoundPose 源码，创建 `foundpose_gpu` 环境；
2. [x] 下载 T-LESS 与 CNOS 默认掩膜，并通过完整性检查；
3. [x] 跑通小样本及对象 1–5 的标准模板、对象表示、完整目标推理和官方评价；
4. [ ] 完成 30 个对象，记录全量 BOP AR、耗时和显存并冻结基线；
5. [x] 输出真实 Top-K PnP 候选、对称类别和 SE(3) 不确定度；
6. [x] 实现 CAD 配置校验和解析式平行夹爪候选生成；
7. [x] 将不确定度接入取料—放料约束联合评分；
8. [x] 编写 ROBI→BOP 适配器，完成 Chrome Screw 真实反光数据小样本验证；
9. [ ] 将 ROBI 扩展到 Gear 的真实场景，验证离散对称性与抓取排序。

## 官方来源与许可

- FoundPose 官方仓库：<https://github.com/facebookresearch/foundpose>
- BOP 数据集：<https://bop.felk.cvut.cz/datasets/>
- ROBI 数据集：<https://www.trailab.utias.utoronto.ca/robi>
- ROBI Toolkit：<https://github.com/junyang224/ROBI_Toolkit>

FoundPose 使用 CC BY-NC 4.0；ROBI 使用 CC BY-NC-SA 4.0。用于论文研究没有路线
冲突，但未来若面向商业部署，需要重新审查代码、数据和模型许可。
