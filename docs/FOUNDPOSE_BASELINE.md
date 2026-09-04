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

截至 2026-09-04，对象 1–10 均已完成标准密度模板、对象表示、完整测试推理、
可视化和官方 BOP19 评价，共输出 1896 个 6D 位姿。标准模板完成 19/30；按当前
实验安排，对象 20–30 的模板生成暂停，已有中间文件保留。完整对象链路完成
10/30，最终基线尚未冻结。

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

对象 5–10 的完整测试结果如下。覆盖率是有效位姿数/目标实例数；AR 是从 BOP
Toolkit 各阈值文件的单类 `obj_recalls` 正确汇总所得。

| 对象 | 有效位姿/目标 | 覆盖率 | 中位 MSSD | 中位 MSPD | 单类官方 AR |
|---|---:|---:|---:|---:|---:|
| 5 | 176/190 | 92.6% | 18.24 mm | 14.89 px | 0.6529 |
| 6 | 79/98 | 80.6% | 24.69 mm | 20.65 px | 0.4369 |
| 7 | 168/250 | 67.2% | 32.22 mm | 19.44 px | 0.3853 |
| 8 | 48/150 | 32.0% | 468.97 mm | 137.93 px | 0.0384 |
| 9 | 226/246 | 91.9% | 30.67 mm | 19.50 px | 0.5386 |
| 10 | 127/143 | 88.8% | 16.75 mm | 15.55 px | 0.5741 |

对象 8 虽然存在 MSSD 2.90 mm、MSPD 4.03 px 的成功个例，但整体覆盖率和 AR
显著低于其余对象，并出现极端错误 PnP 位姿。该对象应保留为后续检测失败、模板
匹配歧义和粗位姿异常值处理的重点案例，不能用最佳图片代表整体效果。

### 对象 2–4 的标准 Top-K 结果

2026-09-04 使用独立版本 `bop19_standard_topk_v1` 完成对象 2–4 的完整测试推理，
复用 `bop19_standard_v1` 对象表示，不重新生成模板或训练网络。官方 Best-1 输出
保留在 `outputs/foundpose_bop19/`，Top-K 单独保存到
`outputs/foundpose_bop19_topk/`，运行前后 Best-1 文件 SHA-256 一致。

- 对象 2：203 组、1015 个候选，耗时 386.86 秒；
- 对象 3：304 组、1520 个候选，耗时 689.64 秒；
- 对象 4：270 组、1350 个候选，耗时 170.33 秒；
- 合计 777 组、3885 个候选，每组均为 5 个不同的 PnP 位姿，旋转矩阵和数值
  格式检查通过；总计时 1246.83 秒，整卡峰值显存 4065 MiB。

原始结果位于各对象目录的 `estimated-poses.json`；对称归一化、米制平移、按
置信度排序、原始 `confidence`、组内归一化 `weight` 和加权 6×6 SE(3) 协方差
位于 `pose-hypotheses-top5.json`。每组 5 个权重之和为 1。T-LESS
对象 2–4 都按 CAD 绕局部 Z 轴的连续对称性以 1° 分辨率归一化。

本次审计还表明，真实 Top-K 不能直接等同于五个可执行抓取输入：对象 2/3/4
分别有 10/32/38 个负深度候选，涉及 10/27/30 个实例，其中按 FoundPose 分数
排第一的负深度解分别有 1/8/5 个；对象 4 另有一个约 17.8 km 深度的极端 PnP
离群解。这些候选的矩阵形式合法，但物理上不可用。下一步必须先进行正深度、
机器人工作空间、重投影或轮廓一致性、置信度和协方差联合门控，再传播到抓取规划。

Top-K 独立版本首次运行还暴露了 FoundPose 源码使用 `version` 而不是
`repre_version` 查找对象表示的问题。兼容修正已保存为
`third_party/patches/foundpose_repre_version.patch`，并纳入源码拉取和基线冻结输入；
它允许多个推理策略安全复用同一对象表示，不改变官方 Best-1 算法配置。

### 对象 2–4 的 CAD 抓取候选

下游 CAD 生成器已经从“PCA 中心固定姿态”扩展为可审计的平行夹爪候选。PLY
加载器读取顶点、法向和三角面；生成器估计均匀密度体积质心，在主轴框架中围绕
质心采样 60 个接触位置与接近方向，并依次执行：

1. 两侧接触点法向、法向互反性和接触连线一致性检查；
2. 5–85 mm 夹爪开度及开度余量检查；
3. 抓取中点到 CAD 质心不超过 20 mm 的检查；
4. 接触点到局部接触面凸包边界不少于 1 mm 的检查；
5. 使用项目 Robotiq 2F-85 指垫 8×22×37.5 mm 包络与 CAD 三角面进行碰撞检查。

对象 2/3/4 最终分别保留 12/4/12 个互异、无手指碰撞候选。对象 2 有 4 个框架
因手指碰撞被拒绝；对象 3 有 14 个因边缘距离、42 个因手指碰撞被拒绝；对象 4
有 4 个因质心偏差、18 个因手指碰撞被拒绝，另去除 14 个重复框架。保留候选的
开度范围分别为 43.43–64.77、65.42–65.65、41.50–77.11 mm；最小接触边缘距离
分别为 8.67、3.41、1.28 mm。

每个候选 JSON 都保存接触点、接触法向、开度与开度余量、质心偏差、边缘距离、
碰撞结果和解析质量。当前质量权重为对向接触 0.40、开度余量 0.15、质心 0.25、
边缘距离 0.20；这是可解释基线而不是最终学习权重。对象 2 和 4 的首组 Top-K
联合鲁棒分数接近零，原因是上游位姿离群与分散，而不是 CAD 候选不满足几何条件。

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
3. [x] 跑通小样本及对象 1–10 的标准模板、对象表示、完整目标推理和官方评价；
4. [ ] 完成 30 个对象，记录全量 BOP AR、耗时和显存并冻结基线；
5. [x] 对对象 2–4 输出真实 Top-K PnP 候选、对称类别和 SE(3) 不确定度；
6. [ ] 完成 Top-K 物理可行性和几何一致性门控，再扩展到对象 5–10；
7. [x] 完成对象 2–4 的接触、开度、质心、边缘和手指碰撞 CAD 候选生成；
8. [x] 将不确定度接入取料—放料约束联合评分；
9. [x] 编写 ROBI→BOP 适配器，完成 Chrome Screw 真实反光数据小样本验证；
10. [ ] 将 ROBI 扩展到 Gear 的真实场景，验证离散对称性与抓取排序。

## 官方来源与许可

- FoundPose 官方仓库：<https://github.com/facebookresearch/foundpose>
- BOP 数据集：<https://bop.felk.cvut.cz/datasets/>
- ROBI 数据集：<https://www.trailab.utias.utoronto.ca/robi>
- ROBI Toolkit：<https://github.com/junyang224/ROBI_Toolkit>

FoundPose 使用 CC BY-NC 4.0；ROBI 使用 CC BY-NC-SA 4.0。用于论文研究没有路线
冲突，但未来若面向商业部署，需要重新审查代码、数据和模型许可。
