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

因此，FoundPose 是位姿感知前端；论文方法的创新边界落在“校准多假设位姿感知与模式精修”、
“任务等价性对称消歧与风险取放决策”以及“不确定度驱动的柔顺放置控制”，而不是把
FoundPose 的物体位姿称为抓取位姿，也不把常规 CAD 抓取候选生成本身包装为创新。

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
  scripts/datasets/prepare_tless_smoke.py \
  --source /home/lzt/lzt_ros2_ws/datasets/bop \
  --destination /home/lzt/lzt_ros2_ws/datasets/bop_smoke

export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop_smoke
export BOP_OUTPUT_PATH=/home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros/outputs/foundpose_smoke
export PYOPENGL_PLATFORM=egl
export FOUNDPOSE_PYTHON=/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python

$FOUNDPOSE_PYTHON scripts/pipeline/run_foundpose.py gen_templates \
  --opts-path lzt_perception/config/foundpose_smoke/tless_templates.json
$FOUNDPOSE_PYTHON scripts/pipeline/run_foundpose.py gen_repre \
  --opts-path lzt_perception/config/foundpose_smoke/tless_repre.json
$FOUNDPOSE_PYTHON scripts/pipeline/run_foundpose.py infer \
  --opts-path lzt_perception/config/foundpose_smoke/tless_infer.json
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

标准配置位于 `lzt_perception/config/foundpose_bop19/`。正式基线保持 FoundPose 上游的
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

### GT/CNOS 统一掩膜评价

2026-09-06 新增 `scripts/evaluation/evaluate_mask_sources.py`，使用同一份 BOP19
`test_targets_bop19.json` 比较可见 GT mask、CNOS 和后续改进分割。所有检测源
统一按分数保留 Top-`inst_count`，CNOS 默认执行与 FoundPose 相同的 3×3
开运算，并使用按分数排序的一对一 IoU 匹配。输出包括整体及逐对象 IoU、像素
精确率/召回率、边界 F1、中心误差、面积比、IoU 0.50/0.75 检测指标和逐实例 CSV。

对象 1–10 共 2012 个目标组、3239 个目标实例。GT 上限的各项重叠指标均为 1；
CNOS 有 2565 个实际送入 FoundPose 预算的预测，缺少 674 个目标槽位，目标平均
IoU 为 0.4586、平均边界 F1 为 0.3667，IoU 0.50/0.75 召回率分别为
0.5403/0.4853。对象 8 的目标平均 IoU 仅 0.0843，IoU 0.50 召回率仅 0.0733，
说明其低位姿覆盖率首先包含显著的检测/分割问题，不能全部归因于 PnP。

当前报告位于：

```text
outputs/mask_evaluation/tless_objects1_10_gt_cnos/summary.json
outputs/mask_evaluation/tless_objects1_10_gt_cnos/matches.csv
```

可选 `--pose-root NAME=PATH` 会以同一目标分母汇总对应 FoundPose 的覆盖率与
初步 MSSD/MSPD。GT 位姿推理使用隔离配置
`lzt_perception/config/foundpose_bop19/tless_infer_gt.json`，但本次只建立配置和评价入口，尚未
运行对象 1–10 的 GT-mask FoundPose 推理，因此不能把 GT/CNOS 的位姿差异称为
已完成实验。

### RGB-D mask 修正结果

2026-09-07 新增 `lzt_perception/lzt_perception/mask_refinement.py` 和
`scripts/perception/refine_bop_masks.py`。算法不读取 GT：先从 CNOS mask 腐蚀核心估计鲁棒
深度区间，再用深度和 Lab 颜色一致性形成候选修正；只有候选改动占原 mask
10%–25%、向外增长不超过 8% 时才采用，否则保留原 mask。对象 2–4 共处理
2299 个检测，其中 227 个采用修正、2071 个因保护条件保持不变、1 个因有效深度
不足保持不变。

在同一 BOP19 目标和预测预算下，CNOS 与修正结果为：

| 来源 | 目标平均 IoU | 边界 F1 | Recall@0.50 | Recall@0.75 |
|---|---:|---:|---:|---:|
| CNOS | 0.42118 | 0.37466 | 0.50951 | 0.45455 |
| RGB-D 修正 | 0.42165 | 0.37549 | 0.50951 | 0.45666 |

提升幅度较小，说明该模块是防止明显深度污染的保守后处理，不应宣称替代 CNOS
或学习式分割网络。结果位于 `outputs/mask_refinement/tless_objects2_4_guarded/`
和 `outputs/mask_evaluation/tless_objects2_4_rgbd_guarded_compare/`。改进 mask 尚未
重新运行 FoundPose，因此当前只证明 mask 指标改善，不能声称已经提高 6D 位姿 AR。

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
离群解。这些候选的矩阵形式合法，但物理上不可用。

2026-09-21 起，`lzt_perception/lzt_perception/pose_gating.py` 和
`scripts/perception/gate_foundpose_topk.py` 改为“极少数硬门控＋连续软特征”：非有限位姿、
非正深度/CAD 位于相机后方、超出已配置工作空间以及缺少必要观测 mask 才会拒绝；
PnP 内点与分数、CAD 投影轮廓 IoU、双向轮廓重投影误差、观测/渲染深度中位误差
和深度内点率只产生软告警与连续质量分。硬有效候选以原始 PnP 分数乘连续几何
质量重排序，再进行 CAD 对称归并和 SE(3) 协方差估计。过程不读取 GT 误差。

下表是 2026-09-07 旧版“全部阈值均硬拒绝”的历史基线，不能当作新版软评分结果；
新版必须在独立输出目录完整重跑并用官方 BOP 指标复评后才能声称性能提升。

对象 2–4 的完整结果为：

| 对象 | 原始组/保留组 | 原始候选/保留候选 | Top-1 MSSD 中位数 (mm) | 门控后 (mm) | Top-1 MSPD 原始/门控后 (px) |
|---:|---:|---:|---:|---:|---:|
| 2 | 203/171 | 1015/442 | 27.35 | 11.81 | 18.39/18.87 |
| 3 | 304/281 | 1520/878 | 19.43 | 10.14 | 19.19/16.15 |
| 4 | 270/220 | 1350/629 | 31.83 | 13.56 | 20.79/18.32 |
| 合计 | 777/672 | 3885/1949 | 25.19 | 11.38 | 19.60/17.24 |

80 个负深度候选全部被拒绝，对象 4 的 17767.8 m 极端解被工作空间、PnP、轮廓
和深度规则联合拒绝。总体保留组覆盖率为 86.5%，表明这是以拒绝部分不确定样本
换取更低错误的安全门控。T-LESS 图像没有机器人—相机外参，当前只启用相机工作
空间；机器人工作空间代码接口已实现，必须在标定后才能启用。CPU CAD 投影采用
点渲染近似，后续需与精确三角网格渲染对照。

上述 86.5% 是“原始已有位姿的 777 组”内的条件覆盖率，不能作为端到端覆盖率。
2026-09-07 新增 `lzt_benchmark/lzt_benchmark/topk_evaluation.py` 和
`scripts/evaluation/evaluate_topk_coverage.py`，固定使用对象 2–4 的 1419 个 BOP19 目标实例
作为分母。缺失检测、PnP 失败和门控后全拒绝均计为失败；每个检测只选一个 Top-1，
每个图像/对象最多输出 `inst_count` 个预测，再交给官方 BOP Toolkit 做一对一匹配。
2026-09-21 又增加固定分母 Recall@1/3/5 和六阶段互斥失败瀑布；完整区分缺失检测
与检测后无位姿时必须传入 `--detections`。瀑布的检测阶段采用场景/图像/类别槽位
计数，并非 mask-to-GT IoU 匹配，因此最终精度仍以官方 BOP 评价为准。

公平评价结果如下：

| 方法 | 输出位姿/目标 | 目标覆盖率 | 官方 MSSD AR | 官方 MSPD AR | 官方 VSD AR | 三指标 AR |
|---|---:|---:|---:|---:|---:|---:|
| 原始 Top-K 分数 Top-1 | 777/1419 | 0.5476 | 0.2095 | 0.4855 | 0.1793 | 0.2914 |
| RGB-D 几何门控 Top-1 | 672/1419 | 0.4736 | 0.3367 | 0.4531 | 0.2876 | 0.3592 |
| 对称模式 RGB-D 精修 Top-1 | 777/1419 | 0.5476 | 0.4246 | 0.4936 | 0.3753 | 0.4312 |

门控使三指标宏平均 AR 绝对提高 0.0677（相对约 23.2%），MSSD 和 VSD 明显提高，
但以少输出 105 个位姿、目标覆盖率下降 7.40 个百分点为代价，且 MSPD 绝对下降
0.0323。因此结论应表述为“门控提高了三维表面和可见表面可靠性，但当前二维投影
排序仍需改进”，不能只引用保留样本的中位误差。对象 2、3、4 的三指标 AR 分别从
0.2237/0.4461/0.2045 提升到 0.2699/0.5520/0.2556。

后续 mode-aware 方法使用对称感知模式聚类、模式中心 RGB-D 精修、修正传播和
重新校准，恢复到与原始方法相同的 777/1419 输出和 54.76% 覆盖率，同时将综合
AR 提高到 0.4312。这是对象 2–4 子集结果，不能与 FoundPose 论文的完整
T-LESS 30 类 AR 直接宣称优劣。结果位于
`outputs/innovation1_pose_modes/tless_objects2_4_v1/evaluation/summary.json`。

同样保留 672 个位姿（47.36%目标覆盖率）时，记录内快速评价的条件 MSSD AR
由原始排序的 0.3996 提升到门控排序的 0.6750；MSPD AR 由 0.6641 小幅提升到
0.6763。该结果说明收益不完全来自拒绝更多样本，但记录内误差没有官方一对一匹配，
论文主结果仍以上表为准。完整报告、风险—覆盖率曲线和官方逐阈值文件位于：

```text
outputs/topk_evaluation/tless_objects2_4_v1/summary.json
outputs/topk_evaluation/tless_objects2_4_v1/risk_coverage.csv
outputs/topk_evaluation/tless_objects2_4_v1/risk_coverage.svg
outputs/topk_evaluation/tless_objects2_4_v1/official_eval/
```

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

批处理入口 `scripts/pipeline/run_bop19_stage.py` 按对象保存配置和 profile，重跑时自动跳过
完整对象。最终只有在 30 个对象的三个阶段、官方五项 score 字段和结果 CSV 均
齐全时，`scripts/evaluation/freeze_foundpose_baseline.py` 才允许生成 `FROZEN_BASELINE.json`。

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
6. [x] 完成对象 2–4 的 Top-K 物理和 RGB-D 几何门控；待扩展到对象 5–10；
7. [x] 完成对象 2–4 的接触、开度、质心、边缘和手指碰撞 CAD 候选生成；
8. [x] 将不确定度接入取料—放料约束联合评分；
9. [x] 编写 ROBI→BOP 适配器，完成 Chrome Screw 真实反光数据小样本验证；
10. [ ] 将 ROBI 扩展到 Gear 的真实场景，验证离散对称性与抓取排序。
11. [x] 完成对象 2–4 的对称模式精修和官方 BOP19 评价；
12. [ ] 完成 T-LESS 1–30 同配置全量评价并将模式级协方差接入 CVaR 抓取。

## 官方来源与许可

- FoundPose 官方仓库：<https://github.com/facebookresearch/foundpose>
- BOP 数据集：<https://bop.felk.cvut.cz/datasets/>
- ROBI 数据集：<https://www.trailab.utias.utoronto.ca/robi>
- ROBI Toolkit：<https://github.com/junyang224/ROBI_Toolkit>

FoundPose 使用 CC BY-NC 4.0；ROBI 使用 CC BY-NC-SA 4.0。用于论文研究没有路线
冲突，但未来若面向商业部署，需要重新审查代码、数据和模型许可。
