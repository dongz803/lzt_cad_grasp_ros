# lzt_cad_grasp_ros

本仓库用于独立探索“CAD 驱动、学习辅助的工业物料 6D 位姿估计与鲁棒上下料”路线，
并按机械臂算法栈组织为多个 ROS 2 包。
原包 `lzt_mujoco_ros` 保留为传统方法基线，两者不互相导入算法代码。

## 研究边界

- 输入为任务指定的工业物料 CAD、RGB-D 观测和目标 ROI；
- 位姿模块输出对称性感知的 Top-K 位姿假设、置信度和可选协方差；
- 抓取模块使用 CAD 坐标系候选，并联合检查取料、放料、IK 和碰撞约束；
- 铜柱只作为首个系统验证对象，不在接口中固化颜色、尺寸、主轴或粗细端规则；
- 传统 HSV、RANSAC、PCA 路线继续保留在 `lzt_mujoco_ros` 中作为对照。

## 机械臂算法栈

运行链路固定为：

```text
RGB-D → perception → grasp_planning → kinematics
      → motion_planning → trajectory → control → MuJoCo/真实机械臂
```

- `lzt_perception` 与 `lzt_grasp_planning` 保存论文研究算法；
- `lzt_kinematics` 只定义正逆运动学契约和结果，当前 MuJoCo Jacobian IK 实现在
  `lzt_mujoco`，以后可以替换为 MoveIt 或实机模型；
- `lzt_motion_planning` 输出无时间戳的无碰撞几何路径；
- `lzt_trajectory` 将路径转换为带时间、速度和加速度的轨迹；
- `lzt_control` 消费轨迹并定义关节、笛卡尔、夹爪和力控接口；
- `lzt_mujoco` 是执行后端，不作为通用算法层；
- `lzt_cad_grasp_ros` 仅保留旧 ROS 可执行程序的兼容入口。

详细边界、输入输出和当前实现程度见
[`docs/架构说明.md`](docs/架构说明.md)。

## 当前状态

当前已跑通 FoundPose + T-LESS 小样本，并完成标准 BOP19 对象 1–10 的完整链路；
这仍是阶段基线，不是完整论文方法：

- 已定义 RGB-D 观测、CAD 对象、位姿假设、抓取候选和取放任务的数据契约；
- 已建立可替换的位姿估计器与任务抓取规划器接口；
- 已复制原包的 MuJoCo 机器人资产和三个基线场景；
- 已实现物料 YAML 校验和 CAD 平行夹爪候选生成；生成器显式检查两侧接触法向、
  夹爪开度、均匀密度质心偏差、接触边缘距离和 Robotiq 2F-85 手指—CAD 碰撞；
- FoundPose 已安装到隔离环境，并已在 RTX 3060 上完成 T-LESS 粗 6D 位姿推理；
- 已扩展真实 Top-5 PnP 粗位姿，并输出对称归一化的加权 SE(3) 协方差；对象
  1–30 已完成 Top-5 推理，共形成 4024 组、20120 个真实候选；
- 已将 Top-K SE(3) 协方差按抓取点力臂传播到取料和放料容差，联合 Top-K
  置信度、IK/碰撞回调、解析抓取质量和运动距离输出鲁棒排序；
- T-LESS BOP19 的 30 个 CAD、20 个场景、1000 张目标图像和 6423 个目标实例已校验完整；
- 标准对象 1–30 均已完成 798 模板、对象表示、完整目标推理和官方
  VSD/MSSD/MSPD 流程，共输出 4024 个 6D 位姿，官方综合 AR 为 0.32545；
- 已建立 GT/CNOS 统一掩膜评价入口，按同一 BOP19 目标、Top-`inst_count`
  预测预算和一对一匹配输出 IoU、边界 F1、检测召回与逐实例记录，并可在同一
  报告中汇总 FoundPose MSSD/MSPD；
- 已实现带保守回退的 RGB-D mask 修正：只在深度不一致区域达到 10%–25%、
  RGB 引导增长不超过 8% 时采用修正，否则保留 CNOS 原 mask；对象 2–4 的平均
  IoU、边界 F1 和 IoU≥0.75 召回率分别由 0.42118/0.37466/0.45455 提升到
  0.42165/0.37549/0.45666；
- 已完成对象 2–4 的 Top-K 正深度、相机工作空间、PnP 内点/分数、CAD 重投影
  轮廓和深度一致性门控，并使用几何质量重加权后重新计算对称性感知协方差；
  3885 个候选保留 1949 个，777 组中 672 组保留安全候选，门控 Top-1 中位
  MSSD/MSPD 由 25.19 mm/19.60 px 降到 11.38 mm/17.24 px；固定全部 1419 个
  BOP 目标实例并采用官方一对一匹配后，目标覆盖率由 54.76% 降到 47.36%，
  对象 2–4 宏平均 BOP AR 由 0.2914 提升到 0.3592，其中 MSSD/VSD 提升，
  MSPD 从 0.4855 降到 0.4531；
- 已增加对称感知位姿模式、模式内 6×6 协方差和模式中心 RGB-D 精修；对象 2–4
  开发集的 10 mm/5° 设置达到 0.4332/0.4981/0.3844/0.4386 的官方
  MSSD/MSPD/VSD/综合 AR，与逐候选精修 0.4383 的综合 AR 持平；
- 对象 5–10 已使用冻结的 14 维可靠性模型和 10 mm/5° 参数完成独立 CAD 留出评价：
  FoundPose 原始、门控可靠性、逐候选精修和模式精修的综合 AR 分别为
  0.4404、0.4977、0.5423 和 0.5418；该结果不包含对象 5–10 的预测协方差标定；
- 已接通 `T_base_object = T_base_camera × T_camera_object`、MuJoCo 仿真精确
  外参、UR5e Jacobian 阻尼最小二乘 IK 和逐段路径接触检查，排序结果会输出基座系
  抓取矩阵、六关节解和碰撞对；实机仍需单独完成手眼标定；
- 已将 T-LESS 物体 4 的同一 CAD 动态注入 MuJoCo，接通固定相机 RGB-D/GT mask、
  FoundPose Top-K、RGB-D 门控与局部修正、CAD 抓取、UR5e IK、路径碰撞检查和
  夹爪动力学；GT 位姿闭环平面误差 3.70 mm，未修正 FoundPose 闭环为 6.49 mm
  （未过 5 mm 阈值），修正后预测位姿闭环三维放置误差 2.63 mm、对称轴误差
  0.33°，无禁止接触并通过验收；
- T-LESS 用于第一阶段标准复现；ROBI→BOP 转换、RealSense 深度到 RGB
  重投影、GT 掩膜生成和 FoundPose 数据集注册已实现，Chrome Screw 的5张真实
  RealSense图像、190个反光实例已完成端到端6D位姿验证；
- 已用对象 2–4 冻结的 14 维可靠性模型和 10 mm/5° 聚类阈值完成 T-LESS
  对象 1–30 全链路评价；原始 Top-5、门控可靠性、逐候选精修和模式精修的官方
  综合 AR 分别为 0.32234、0.38700、0.43706 和 0.43671；对象 1 和 5–30 未用于重新调参。

## 目录

```text
lzt_cad_grasp_ros/
├── lzt_common/                   # 公共契约、SE(3)、配置和资源路径
├── lzt_interfaces/               # ROS 消息、服务和动作
├── lzt_description/              # UR5e/夹爪/相机/CAD/场景资源
├── lzt_perception/               # 分割、Top-K 位姿、门控、精修和可靠性
├── lzt_grasp_planning/           # CAD 抓取生成和风险感知排序
├── lzt_kinematics/               # 运动学接口与通用结果
├── lzt_motion_planning/          # 几何路径和碰撞规划接口
├── lzt_trajectory/               # 插值和时间参数化
├── lzt_control/                  # 关节、笛卡尔、夹爪和力控接口
├── lzt_mujoco/                   # MuJoCo 模型、IK、碰撞和执行适配
├── lzt_bringup/                  # ROS 节点、launch 与系统配置
├── lzt_benchmark/                # BOP/ROBI 评价与论文实验
├── lzt_system_tests/             # 跨包边界和端到端测试
├── lzt_examples/                 # 最小使用示例
├── lzt_cad_grasp_ros/            # 旧 ROS 包兼容入口
├── tests/                         # 原有算法回归测试
├── third_party/                  # 外部方法说明；源码本身不纳入 Git
└── outputs/                      # 实验结果，不纳入版本控制
```

## FoundPose 数据与输出目录说明

FoundPose 的数据位于工作空间级目录 `/home/lzt/lzt_ros2_ws/datasets/`，不放入
Git 仓库。当前保留三个目录，各自用途不同：

```text
/home/lzt/lzt_ros2_ws/datasets/
├── bop/                         # 解压后、可直接运行的正式 BOP 数据
├── bop_smoke/                   # 单图小样本使用的轻量目录
└── downloads/                   # 最初下载的 ZIP 和原始 JSON
```

- `bop/` 是 FoundPose 实际使用的数据根目录，当前包含 T-LESS 测试图像、CAD
  模型和 CNOS-FastSAM 检测结果，约占 1017 MB。运行完整数据或建立小样本都依赖
  该目录，不能随意删除。
- `bop_smoke/` 用于限制 smoke test 只处理 T-LESS 对象 1、场景 20、图像 3。
  它实际只占约 36 KB，大部分内容是指向 `bop/` 的软链接，可以通过
  `scripts/datasets/prepare_tless_smoke.py` 重新生成。
- `downloads/` 保存三个 T-LESS 下载压缩包和原始 CNOS JSON，约占 891 MB。
  这些内容已经解压并整理到 `bop/`，当前作为原始下载备份保留，不参与推理。

FoundPose 的小样本输出位于本仓库的 `outputs/foundpose_smoke/`，标准 BOP19 输出
位于 `outputs/foundpose_bop19/`。内部目录是按“处理阶段—数据集—版本—对象编号”
组织的，不是重复结果：

```text
outputs/foundpose_smoke/
├── templates/
│   └── smoke_v1/tless/1/        # 对象 1 的多视角 RGB、深度和掩膜模板
├── object_repre/
│   └── tless/smoke_v1/1/        # DINOv2、PCA、FAISS 对象特征表示
└── inference/
    └── tless_smoke_v1/1/        # 最终位姿数值和可视化结果
```

日常查看结果只需要关注以下两个文件：

```text
outputs/foundpose_smoke/inference/tless_smoke_v1/1/estimated-poses.json
outputs/foundpose_smoke/inference/tless_smoke_v1/1/20_3_1_0_0.jpg
```

其中 `estimated-poses.json` 保存物料相对相机的 6D 位姿、评分、耗时、MSSD 和
MSPD；`20_3_1_0_0.jpg` 展示输入掩膜、候选模板、特征对应以及预测位姿与真值
位姿的投影对比。`templates/` 和 `object_repre/` 是可复用的中间缓存，保留它们后
再次对同一 CAD 推理时不需要重新渲染模板和提取全部对象特征。

当前暂不移动、重命名或删除上述目录。后续若确认需要简化，可在不改变 FoundPose
内部结构的前提下增加“最新 JSON/图片”的快捷入口。

标准对象 1–10 的日常查看入口为：

```text
outputs/foundpose_bop19/inference/tless_bop19_standard_v1/<对象号>/estimated-poses.json
outputs/foundpose_bop19/visualizations/object<对象号>_best_pose_overlay.png
outputs/foundpose_bop19/visualizations/object<对象号>_result_overview.png
outputs/foundpose_bop19/official_eval_object<对象号>/
```

前三项依次是数值位姿、最佳样本叠加图和误差分布总览；最后一项是官方 BOP19
评估的逐阈值匹配文件与分数。单对象提交的顶层 `scores_bop19.json` 仍按 T-LESS
全部 30 类平均，不能直接当作该物体自身的 AR；单类结果需要从逐阈值文件的
`obj_recalls` 提取。

对象 2–4 的完整 Top-K 研究结果保存在独立输出树，不会覆盖上述官方 Best-1：

```text
outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/<对象号>/estimated-poses.json
outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/<对象号>/pose-hypotheses-top5.json
```

前者是 FoundPose 原始 PnP 候选，后者按置信度排序、将平移统一为米，显式保存
原始 `confidence` 和组内归一化 `weight`，并根据 CAD 连续对称性归一化后附加
6×6 SE(3) 协方差。对象 2/3/4 分别包含 203/304/270
个实例，每个实例均有 5 个不同候选。原始 Top-K 中已观察到负深度和极端平移
离群解，因此这些候选目前用于异常值分析和鲁棒抓取研究，不能未经门控直接执行。

### GT/CNOS 统一评价

统一评价脚本始终把可见 GT mask 作为理想上限；CNOS 默认复现 FoundPose 的
每目标 Top-`inst_count` 选择和 3×3 开运算。对象 1–10 的当前基线命令为：

```bash
python3 scripts/evaluation/evaluate_mask_sources.py \
  --object-ids 1 2 3 4 5 6 7 8 9 10 \
  --pose-root cnos=outputs/foundpose_bop19/inference/tless_bop19_standard_v1 \
  --output-dir outputs/mask_evaluation/tless_objects1_10_gt_cnos
```

结果写入 `summary.json` 和 `matches.csv`。前者保存整体、逐对象和可选下游位姿
指标，后者保存每个目标槽位的预测排名、分数、匹配 GT、IoU、像素精确率/召回率、
边界 F1、中心误差和面积比。新增改进分割可通过重复的
`--detections improved=<BOP检测JSON>` 接入，不需要修改评价代码。

GT-mask FoundPose 使用独立配置 `lzt_perception/config/foundpose_bop19/tless_infer_gt.json`，
结果版本为 `bop19_standard_gt_v1`，不会覆盖 CNOS 基线。该推理尚未批量运行。

### RGB-D mask 修正与 Top-K 异常门控

mask 修正不读取 GT，只利用 CNOS mask 内部的鲁棒深度区间、邻域 RGB Lab
一致性和连通区域。保守保护条件防止后处理破坏原本正确的 mask：

```bash
python3 scripts/perception/refine_bop_masks.py \
  --object-ids 2 3 4 \
  --output-dir outputs/mask_refinement/tless_objects2_4_guarded

python3 scripts/evaluation/evaluate_mask_sources.py \
  --object-ids 2 3 4 \
  --detections cnos=/home/lzt/lzt_ros2_ws/datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json \
  --detections rgbd=outputs/mask_refinement/tless_objects2_4_guarded/detections_rgbd_refined.json \
  --source-opening cnos=3 --source-opening rgbd=3 \
  --output-dir outputs/mask_evaluation/tless_objects2_4_rgbd_guarded_compare
```

随后对原始 FoundPose Top-K 做候选级可靠性处理。当前版本只把非有限位姿、
非正深度/CAD 位于相机后方、超出相机或机器人工作空间、缺少必要观测 mask
作为硬拒绝；PnP、轮廓和深度阈值仅生成软告警与连续质量特征，不再一票否决。
以下命令分别运行对象 2–4：

```bash
export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop
for obj in 2 3 4; do
  python3 scripts/perception/gate_foundpose_topk.py \
    --input outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/$obj/estimated-poses.json \
    --object-config lzt_description/config/objects/tless_$(printf '%06d' "$obj").yaml \
    --detections outputs/mask_refinement/tless_objects2_4_guarded/detections_rgbd_refined.json \
    --output-dir outputs/pose_gating/tless_$(printf '%06d' "$obj")
done
```

每个对象输出 `summary.json`、逐候选 `audit.json/candidates.csv`、门控后的
`gated-poses.json` 和重新归并/估计协方差的 `pose-hypotheses-gated.json`。
审计结果分别保存 `hard_rejection_reasons`、`soft_quality_flags`、`soft_pass` 和
各连续质量分量。运行过程不使用 MSSD/MSPD；这两个真值指标仅在结束后写入 `evaluation_only`
用于论文对照。当前 T-LESS 离线图像没有机器人外参，因此相机工作空间门控已启用，
机器人工作空间接口已实现但未启用；传入 `--robot-camera-transform` 和机器人边界
后才能进行有物理意义的机器人工作空间检查。CAD 投影采用确定性 CPU 点渲染，
适合候选筛选，但后续仍应与精确三角网格渲染做消融。

### 覆盖率公平的 Top-K 评价

门控允许拒绝全部候选，因此不能只比较“保留下来的位姿”中位误差。
`scripts/evaluation/evaluate_topk_coverage.py` 固定使用 BOP19 `test_targets_bop19.json`
中的目标实例数作为分母；缺失检测、PnP 失败和门控后全拒绝都计为失败。同时输出
完整工作点、相同覆盖率比较、风险—覆盖率曲线、仅用于分析的 Top-K GT oracle，
以及供官方 BOP Toolkit 做一对一 GT 匹配的结果 CSV：

```bash
PY=/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python
$PY scripts/evaluation/evaluate_topk_coverage.py \
  --method raw=outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/2/estimated-poses.json \
  --method raw=outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/3/estimated-poses.json \
  --method raw=outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/4/estimated-poses.json \
  --method gated=outputs/pose_gating/tless_000002/gated-poses.json \
  --method gated=outputs/pose_gating/tless_000003/gated-poses.json \
  --method gated=outputs/pose_gating/tless_000004/gated-poses.json \
  --object-id 2 --object-id 3 --object-id 4 \
  --detections /home/lzt/lzt_ros2_ws/datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json \
  --output-dir outputs/topk_evaluation/tless_objects2_4_v1 \
  --run-official --official-workers 1
```

无图形环境时，官方 VSD 渲染需要 EGL，例如在命令前添加
`PYOPENGL_PLATFORM=egl`。输出目录中的 `summary.json` 是完整机器可读报告，
其中 `recall_at_k` 默认给出固定分母的 Recall@1/3/5；`failure_waterfall`
把全部目标互斥归入缺失检测、检测后无位姿、硬门控全拒绝、Top-K 无正确候选、
有正确候选但 Top-1 选错、Top-1 正确六个阶段，并逐对象报告。检测阶段按
场景/图像/类别的输出槽位计数并受 `inst_count` 限制，不等同于 mask 与 GT 的
IoU 一对一匹配；最终论文指标仍应使用官方 BOP 匹配。
`risk_coverage.csv` 和 `risk_coverage.svg` 分别用于统计和画图，`bop_results/`
与 `official_eval/` 保存官方协议的输入和结果。快速的记录内 MSSD/MSPD 只用于
调试；论文主结果应引用 `official_bop19`，并同时报告目标覆盖率。

## 标准 BOP19 与抓取接口

完整 T-LESS 运行采用官方 FoundPose 密度：57 个视点、14 个平面内旋转，即每个
CAD 798 张模板。每个对象独立保存配置、输出和资源 profile，已完成对象会自动跳过：

```bash
cd /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros
export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop
PY=/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python

python3 scripts/datasets/validate_tless_bop19.py
$PY scripts/pipeline/run_bop19_stage.py gen_templates --workers 2
$PY scripts/pipeline/run_bop19_stage.py gen_repre --workers 1
$PY scripts/pipeline/run_bop19_stage.py infer --workers 2

python3 scripts/evaluation/export_foundpose_bop.py \
  --output outputs/foundpose_bop19/bop_results/foundpose-bop19-standard_tless-test.csv
$PY scripts/evaluation/evaluate_bop19_official.py \
  --result-csv outputs/foundpose_bop19/bop_results/foundpose-bop19-standard_tless-test.csv
```

`gen_repre` 在 6 GiB GPU 上被强制为单进程，并采用 GPU 提取 DINOv2/PCA 特征、
CPU 执行同一 FAISS K-means 的混合模式，避免 FAISS 默认 1 GiB 临时显存导致
OOM；模板虽允许最多三路，但 RTX 3060 实测采用两路更稳妥，推理最多两路。显存
profile 测量的是整张 GPU 的总占用，并同时记录运行前基线、峰值和峰值增量。

将 FoundPose 结果转换为项目 Top-K 接口，以及从 CAD 生成物体坐标系抓取候选：

```bash
export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop
python3 scripts/perception/convert_foundpose_topk.py \
  --input outputs/foundpose_smoke/inference/tless_smoke_v1/1/estimated-poses.json \
  --object-config lzt_description/config/objects/tless_000001.yaml \
  --output outputs/foundpose_smoke/inference/tless_smoke_v1/1/pose-hypotheses-top5.json
python3 scripts/grasp/generate_cad_grasps.py \
  --object-config lzt_description/config/objects/tless_000001.yaml \
  --output outputs/cad_grasps/tless_000001.json
python3 scripts/grasp/rank_task_grasps.py \
  --pose-hypotheses outputs/foundpose_smoke/inference/tless_smoke_v1/1/pose-hypotheses-top5.json \
  --grasp-candidates outputs/cad_grasps/tless_000001.json \
  --object-config lzt_description/config/objects/tless_000001.yaml \
  --output outputs/task_grasps/tless_000001_scene20_image3.json
```

正式 BOP 基线使用上游一致的 `best_coarse`；Top-K 研究接口使用独立配置
`lzt_perception/config/foundpose_bop19/tless_infer_topk.json`，避免多个同实例假设改变官方基线排名。
对象 2–4 的完整 Top-K 可使用以下命令断点续跑：

```bash
PYOPENGL_PLATFORM=egl $PY scripts/pipeline/run_bop19_stage.py infer \
  --config lzt_perception/config/foundpose_bop19/tless_infer_topk.json \
  --output-root outputs/foundpose_bop19_topk \
  --object-start 2 --object-end 4 --workers 1
```

对象 2–4 的 CAD 抓取候选和 Top-K 联合排序结果分别位于：

```text
outputs/cad_grasps/tless_00000<对象号>.json
outputs/task_grasps/tless_00000<对象号>_topk_set0.json
```

自动生成器围绕 CAD 均匀密度质心采样 60 个 PCA 抓取框架，使用真实三角面和
2F-85 指垫碰撞盒过滤，再按对向接触 0.40、开度余量 0.15、质心 0.25、边缘
距离 0.20 的解析权重排序。对象 2/3/4 最终分别保留 12/4/12 个候选；数量不足时
不会用碰撞候选补足。该权重是可解释的工程基线，后续仍需通过消融确定论文配置。

## 场景对应关系

- `lzt_description/scenes/baseline_vision_single.xml`：原单铜柱 RGB-D 场景；
- `lzt_description/scenes/baseline_vision_ten.xml`：原十铜柱无遮挡场景；
- `lzt_description/scenes/exploration_workcell.xml`：原完整上下料工作台场景。
- `lzt_description/scenes/industrial_cad_single.xml`：工业 CAD 动态注入场景；不再把铜柱固定写在 XML，
  由执行脚本按物料 YAML 选择 CAD。
- `lzt_description/scenes/industrial_cad_grasp_focus.xml`：抓取优先场景；保留相同机器人、相机、
  工作台和上料区，但把 6 个铜柱圆形点位改为一个通用工业物料矩形下料区。

这些文件是新路线的初始环境副本。后续修改新场景不会影响传统基线包。

## 工业 CAD 端到端执行基准

先用仿真真值位姿验证“位姿之后”的整条执行链，避免把视觉误差与机械执行错误混在
一起。默认对象是 T-LESS 4，CAD、抓取候选和动态碰撞体来自同一个物料配置：

```bash
cd /home/lzt/lzt_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to lzt_cad_grasp_ros lzt_system_tests
source install/setup.bash
export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop

# 只采集 RGB-D/掩膜并审核 IK 与碰撞路径，不执行运动
MUJOCO_GL=egl python3 src/lzt_cad_grasp_ros/scripts/pipeline/run_industrial_cad_pick_place.py \
  --check-only

# 无窗口完整执行并保存结果
MUJOCO_GL=egl python3 src/lzt_cad_grasp_ros/scripts/pipeline/run_industrial_cad_pick_place.py

# 打开 MuJoCo 窗口观察完整过程
python3 src/lzt_cad_grasp_ros/scripts/pipeline/run_industrial_cad_pick_place.py --viewer
```

未指定 `--output` 时，结果按对象、输入类型、执行模式、运行编号和随机种子保存，
例如：

```text
outputs/industrial_cad_pick_place/tless_000004/
├── gt_check/<run_id>_seed0/
├── gt_dynamic/<run_id>_seed0/
├── predicted_check/<run_id>_seed0/
└── predicted_dynamic/<run_id>_seed0/
```

已有非空运行目录绝不会被覆盖。每次运行额外保存 `run_manifest.json` 和
`config_snapshot/`，记录命令、随机种子、Git 提交/状态、环境、输入副本及 SHA256。
主要结果文件包括：

- `initial_rgb.png`、`initial_depth.png`、`initial_gt_mask.png`：同一固定相机观测；
- `observation.json`：内参和 `T_camera_object`；
- `plan.json`：所选 CAD 抓取、7 个关节路点、IK 与碰撞审计；
- `00_home.png` 至 `08_retreated.png`、`pick_place.gif`：执行过程；
- `result.json`：抬升、放置误差、稳定性、禁止接触和最终成功判据。

若当前阶段只研究抓取误差，可使用独立入口运行抓取优先场景：

```bash
cd /home/lzt/lzt_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# 只审核观测、抓取候选、IK 和碰撞路径
MUJOCO_GL=egl python3 \
  src/lzt_cad_grasp_ros/scripts/pipeline/run_industrial_cad_grasp_focus.py --check-only

# 执行完整抓放并保存过程图片和 GIF
MUJOCO_GL=egl python3 \
  src/lzt_cad_grasp_ros/scripts/pipeline/run_industrial_cad_grasp_focus.py
```

该入口仍将物料送至绿色区域中心，但验收不再使用精确放置位置和姿态容差。
成功条件为：物料被有效抬升、最终完整落在矩形区域内、稳定且无禁止碰撞。
`result.json` 单独记录 `grasp_target_position_error_m`、
`grasp_target_angle_error_deg` 和 `grasp_success`；原来的放置误差只作为诊断值，
并通过 `placement_pose_error_used_for_acceptance=false` 明确标记为不参与判定。
抓取优先结果写入带 `_grasp_focus` 后缀的独立目录，不会与原精确放置结果混用。

以上默认命令使用 `mujoco_ground_truth` 位姿，作用是冻结一个可回归的机械执行
基线。完整预测闭环可用一条命令运行：

```bash
cd /home/lzt/lzt_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/lzt_cad_grasp_ros/scripts/pipeline/run_tless_obj4_pipeline.py \
  --run-id system_baseline_v1 --seed 0
```

该命令依次运行“采集 → FoundPose Top-K → RGB-D 门控 → CAD/深度位姿修正 →
抓放”，结果隔离到
`outputs/system_baselines/tless_000004/system_baseline_v1_seed0/`。顶层
`pipeline_manifest.json` 记录各阶段命令、耗时、返回码和方法边界，
`pipeline_result.json` 汇总最终验收。当前单图导出使用 GT mask 隔离分割误差，
但 FoundPose、RGB-D 门控和修正均不读取 GT 位姿；GT 位姿仅用于仿真结束后的
误差审计。

如需逐阶段调试，下列命令与一键脚本等价：

```bash
# 1. 将同一仿真观测导出为单图 T-LESS BOP 输入
MUJOCO_GL=egl python3 src/lzt_cad_grasp_ros/scripts/pipeline/run_industrial_cad_pick_place.py \
  --image-width 720 --image-height 540 \
  --run-id capture_debug --seed 0 \
  --export-bop-root /home/lzt/lzt_ros2_ws/datasets/bop_mujoco_obj4 --check-only

# 2. 用已有对象 4 表示运行 FoundPose Top-K
cd /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros
FPY=/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python
BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop_mujoco_obj4 \
BOP_OUTPUT_PATH=$PWD/outputs/foundpose_bop19 PYOPENGL_PLATFORM=egl \
  $FPY scripts/pipeline/run_foundpose.py infer \
  --opts-path lzt_perception/config/foundpose_bop19/tless_mujoco_obj4_infer.json

# 3. 使用轮廓和深度门控 Top-K，再做不使用 GT 位姿的局部修正
cd /home/lzt/lzt_ros2_ws
export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop
python3 src/lzt_cad_grasp_ros/scripts/perception/gate_foundpose_topk.py \
  --input src/lzt_cad_grasp_ros/outputs/foundpose_bop19/inference/tless_mujoco_obj4_gtmask_v1/4/estimated-poses.json \
  --object-config src/lzt_cad_grasp_ros/lzt_description/config/objects/tless_000004.yaml \
  --detections datasets/bop_mujoco_obj4/gt-mask-detections.json \
  --dataset-root datasets/bop_mujoco_obj4/tless --mask-opening-kernel 1 \
  --output-dir src/lzt_cad_grasp_ros/outputs/pose_gating/tless_000004_mujoco_gtmask
python3 src/lzt_cad_grasp_ros/scripts/perception/refine_foundpose_poses.py \
  --input src/lzt_cad_grasp_ros/outputs/pose_gating/tless_000004_mujoco_gtmask/gated-poses.json \
  --object-config src/lzt_cad_grasp_ros/lzt_description/config/objects/tless_000004.yaml \
  --detections datasets/bop_mujoco_obj4/gt-mask-detections.json \
  --dataset-root datasets/bop_mujoco_obj4/tless --reweight-score \
  --output-dir src/lzt_cad_grasp_ros/outputs/pose_refinement/tless_000004_mujoco_gtmask
python3 src/lzt_cad_grasp_ros/scripts/perception/convert_foundpose_topk.py \
  --input src/lzt_cad_grasp_ros/outputs/pose_refinement/tless_000004_mujoco_gtmask/refined-poses.json \
  --object-config src/lzt_cad_grasp_ros/lzt_description/config/objects/tless_000004.yaml \
  --output src/lzt_cad_grasp_ros/outputs/pose_refinement/tless_000004_mujoco_gtmask/pose-hypotheses-refined.json

# 4. 用预测位姿实际执行抓取、搬运和放置
MUJOCO_GL=egl python3 src/lzt_cad_grasp_ros/scripts/pipeline/run_industrial_cad_pick_place.py \
  --image-width 720 --image-height 540 \
  --pose-hypotheses src/lzt_cad_grasp_ros/outputs/pose_refinement/tless_000004_mujoco_gtmask/pose-hypotheses-refined.json \
  --run-id predicted_debug --seed 0
```

第一次同源单图结果中，原始置信度最高候选 MSSD 为 41.84 mm；RGB-D 门控拒绝
2/5 个候选并把 11.93 mm 候选排到首位。直接执行门控结果完成抓放但放置误差
6.49 mm，未过 5 mm 阈值；局部修正后首选预测平移误差降至 3.00 mm，动态闭环
最终三维放置误差为 2.63 mm。该单次结果证明接口闭环可用，尚不能代表统计提升。
固定种子后的版本化运行可能得到不同 Top-K 代表姿态并暴露抓放失败，因此基线判据
以保存完整成功/失败样本为准，不把单次成功写成算法性能结论。完整定义见
`docs/SYSTEM_BASELINE_V1.md`。

## 创新点一：面向对称工业物料的校准多假设 6D 位姿感知与模式级精修

当前创新点一已形成感知原型：先用物理硬门控删除负深度、极端工作空间及明显
RGB-D 不一致候选，再从 PnP、CAD 轮廓、深度残差和组内一致性提取 14 维无量纲
特征，用 L2 逻辑回归输出候选正确概率；随后按 CAD 对称先验将 Top-K 聚类为多个非等价
位姿模式，对模式中心执行 RGB-D/CAD 精修、修正传播、重审核和重校准。其目标输出是
“多个非等价物体位姿模式 + 模式概率 + 模式内 SE(3) 不确定度”，而不是抓取位姿。

下游已实现创新点二的基础原型：将校准概率、Top-K 位姿、CAD 抓取质量、IK/碰撞和
SE(3) 不确定度用于期望成功和 CVaR 尾部风险排序，并支持主动拒绝。V3 一键流水线
已支持 `--mode-aware` 和冻结预测协方差模型，最终 `pose-modes-refined.json` 可直接作为
执行输入，使每个模式的总协方差进入抓取风险计算；显式任务等价性消歧仍未完成。
CAD 抓取候选生成、IK 和碰撞检查作为基础能力，不单独宣称创新。

冻结模型位于：

```text
lzt_perception/config/reliability/tless_objects2_4_logistic_v3.json
```

该模型仅使用 T-LESS 对象 2–4 的 MSSD 作为离线训练标签；运行时不读取真值。
留一 CAD 交叉验证共使用 1949 个硬门控后候选，ROC AUC 为 0.9009、Brier 分数
为 0.0873、ECE 为 0.0327。BOP 共含 1419 个目标实例，其中 FoundPose 形成
777 个原始候选组、硬门控后覆盖 672 组（47.36%）。手工 RGB-D 分数与学习概率的
Top-1 正确数分别为 151 和 205，即固定目标分母成功率从 10.64% 提高到 14.45%；该
结果只证明对象 2–4 的离线候选排序提升，不代表跨数据域或动态抓取成功率提升。

重新训练和评估：

```bash
cd /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros
python3 scripts/perception/train_pose_reliability.py \
  --audits outputs/pose_gating/tless_000002/audit.json \
           outputs/pose_gating/tless_000003/audit.json \
           outputs/pose_gating/tless_000004/audit.json \
  --output-dir outputs/pose_reliability/my_run
```

运行一次兼容性端到端流程（脚本中的 `innovation1` 为历史命名）：

```bash
MUJOCO_GL=egl python3 scripts/pipeline/run_tless_obj4_pipeline.py \
  --run-id innovation1_demo --seed 0 --grasp-focus \
  --reliability-model lzt_perception/config/reliability/tless_objects2_4_logistic_v3.json \
  --mode-aware \
  --predictive-covariance-model lzt_perception/config/uncertainty/tless_objects2_4_predictive_v3.json \
  --grasp-ranking cvar --cvar-tail-probability 0.25 --cvar-risk-weight 0.5
```

不传 `--mode-aware` 时仍保留原逐候选精修兼容路径。模式路径会同时保存普通候选
`pose-hypotheses-refined.json`用于审计，并将最终模式文件作为实际执行输入。

运行现有风险决策原型的严格配对消融（脚本名保留历史 `innovation1` 标识；基线和方法分支
使用相同随机条件，并复用同一份图像和 FoundPose 原始 Top-K）：

```bash
MUJOCO_GL=egl python3 scripts/evaluation/run_innovation1_grasp_ablation.py \
  --trials 20 --seed 300 \
  --output-dir outputs/grasp_ablation/innovation1_paired_20_v1
```

当前 3 组严格配对预实验中，两种方法均为 3/3 抓取成功，成功率差为 0；样本量
过小且对象 4 的开放抓取场景存在天花板效应。T-LESS 模型在这 3 个理想 MuJoCo
样本上的位姿集可靠性均值仅 0.405，而抓取实际均成功，说明概率存在跨域欠校准。
因此正式论文实验还必须加入 MuJoCo/真实工作站校准集、失败与拒绝样本、非对称及
离散对称 CAD，并按固定分母报告成功率、覆盖率、Wilson 区间与配对统计。

## 构建与检查

```bash
cd /home/lzt/lzt_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to lzt_cad_grasp_ros lzt_system_tests
source install/setup.bash
colcon test --packages-select lzt_cad_grasp_ros lzt_system_tests
colcon test-result --verbose
ros2 run lzt_examples show_robot_stack
ros2 run lzt_cad_grasp_ros cad_grasp_exploration_node
```

启动节点当前只检查配置和 MuJoCo 场景是否可加载，并发布探索路线状态；它不会调用传统位姿算法。

## 后续实现顺序

1. 用独立 MuJoCo/真实工作站校准集修正 T-LESS 概率的跨域偏差，冻结主动拒绝阈值；
2. 完成 mode-aware 一键路径的多种子配对实验，核对模式级协方差、空假设和CVaR
   在所有运行清单中的可追溯性；
3. 用对象 8–10 冻结集验证 RGB-D mask 修正，另行运行改进 mask 下的 FoundPose；
4. 整理已完成的 T-LESS 1–30 Best-1/原始 Top-K/门控/逐候选精修/模式精修主表、
   逐对象结果和失败瀑布；
5. 将严格配对实验扩展到多 CAD 和每种方法至少 50 次，形成置信区间与失败分布；
6. 实机手眼标定后替换仿真外参，验证真实工作空间、IK 和碰撞检查；
7. 在已完成 T-LESS 1–30 同配置全量评价的基础上，扩展 ROBI 与多工业 CAD 跨域实验。

具体选择与操作边界见 `docs/FOUNDPOSE_BASELINE.md`，新增位姿修正与机器人约束链
见 `docs/POSE_REFINEMENT_AND_ROBOT_CONSTRAINTS.md`。
