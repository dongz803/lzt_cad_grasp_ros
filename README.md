# lzt_cad_grasp_ros

本包用于独立探索“CAD 驱动、学习辅助的工业物料 6D 位姿估计与鲁棒上下料”路线。
原包 `lzt_mujoco_ros` 保留为传统方法基线，两者不互相导入算法代码。

## 研究边界

- 输入为任务指定的工业物料 CAD、RGB-D 观测和目标 ROI；
- 位姿模块输出对称性感知的 Top-K 位姿假设、置信度和可选协方差；
- 抓取模块使用 CAD 坐标系候选，并联合检查取料、放料、IK 和碰撞约束；
- 铜柱只作为首个系统验证对象，不在接口中固化颜色、尺寸、主轴或粗细端规则；
- 传统 HSV、RANSAC、PCA 路线继续保留在 `lzt_mujoco_ros` 中作为对照。

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
  2–4 已完成完整测试，共形成 777 组、3885 个真实候选；
- 已将 Top-K SE(3) 协方差按抓取点力臂传播到取料和放料容差，联合 Top-K
  置信度、IK/碰撞回调、解析抓取质量和运动距离输出鲁棒排序；
- T-LESS BOP19 的 30 个 CAD、20 个场景、1000 张目标图像和 6423 个目标实例已校验完整；
- 标准对象 1–10 均已完成 798 模板、对象表示、完整目标推理、可视化和官方
  VSD/MSSD/MSPD 流程，共输出 1896 个 6D 位姿；
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
- 已增加 CAD 轮廓与 RGB-D 深度反馈的局部 SE(3) 位姿修正，并以独立脚本保存
  修正前后指标、审计 JSON 和轮廓预览；对象 2 的单图 3 候选 MSSD 均降低，但
  MSPD 有小幅波动，因此尚不把 smoke test 当作全数据集提升结论；
- 已接通 `T_base_object = T_base_camera × T_camera_object`、MuJoCo 仿真精确
  外参、UR5e Jacobian 阻尼最小二乘 IK 和逐段路径接触检查，排序结果会输出基座系
  抓取矩阵、六关节解和碰撞对；实机仍需单独完成手眼标定；
- T-LESS 用于第一阶段标准复现；ROBI→BOP 转换、RealSense 深度到 RGB
  重投影、GT 掩膜生成和 FoundPose 数据集注册已实现，Chrome Screw 的5张真实
  RealSense图像、190个反光实例已完成端到端6D位姿验证；
- 标准模板已完成对象 1–19；对象 20–30 的模板生成按当前实验安排暂停，已有中间
  文件保留，最终 30 类基线尚未冻结。

## 目录

```text
lzt_cad_grasp_ros/
├── assets/                       # 独立复制的 UR5e、Tracer、2F-85 资源
├── config/
│   ├── exploration.yaml          # ROS 2 探索节点配置
│   ├── foundpose_baseline.yaml   # FoundPose、GPU 和数据阶段配置
│   └── objects/                  # 每种 CAD 的独立配置
├── docs/                         # 新路线的方法与环境说明
├── launch/
├── scenes/                       # 从传统包复制的仿真环境基线
├── src/
│   ├── lzt_cad_grasp_core/       # 与 ROS 无关的方法和数据契约
│   └── lzt_cad_grasp_ros/        # ROS 2 适配层
├── tests/
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
  `scripts/prepare_tless_smoke.py` 重新生成。
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
python3 scripts/evaluate_mask_sources.py \
  --object-ids 1 2 3 4 5 6 7 8 9 10 \
  --pose-root cnos=outputs/foundpose_bop19/inference/tless_bop19_standard_v1 \
  --output-dir outputs/mask_evaluation/tless_objects1_10_gt_cnos
```

结果写入 `summary.json` 和 `matches.csv`。前者保存整体、逐对象和可选下游位姿
指标，后者保存每个目标槽位的预测排名、分数、匹配 GT、IoU、像素精确率/召回率、
边界 F1、中心误差和面积比。新增改进分割可通过重复的
`--detections improved=<BOP检测JSON>` 接入，不需要修改评价代码。

GT-mask FoundPose 使用独立配置 `config/foundpose_bop19/tless_infer_gt.json`，
结果版本为 `bop19_standard_gt_v1`，不会覆盖 CNOS 基线。该推理尚未批量运行。

### RGB-D mask 修正与 Top-K 异常门控

mask 修正不读取 GT，只利用 CNOS mask 内部的鲁棒深度区间、邻域 RGB Lab
一致性和连通区域。保守保护条件防止后处理破坏原本正确的 mask：

```bash
python3 scripts/refine_bop_masks.py \
  --object-ids 2 3 4 \
  --output-dir outputs/mask_refinement/tless_objects2_4_guarded

python3 scripts/evaluate_mask_sources.py \
  --object-ids 2 3 4 \
  --detections cnos=/home/lzt/lzt_ros2_ws/datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json \
  --detections rgbd=outputs/mask_refinement/tless_objects2_4_guarded/detections_rgbd_refined.json \
  --source-opening cnos=3 --source-opening rgbd=3 \
  --output-dir outputs/mask_evaluation/tless_objects2_4_rgbd_guarded_compare
```

随后对原始 FoundPose Top-K 做候选级门控。以下命令分别运行对象 2–4：

```bash
export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop
for obj in 2 3 4; do
  python3 scripts/gate_foundpose_topk.py \
    --input outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/$obj/estimated-poses.json \
    --object-config config/objects/tless_$(printf '%06d' "$obj").yaml \
    --detections outputs/mask_refinement/tless_objects2_4_guarded/detections_rgbd_refined.json \
    --output-dir outputs/pose_gating/tless_$(printf '%06d' "$obj")
done
```

每个对象输出 `summary.json`、逐候选 `audit.json/candidates.csv`、门控后的
`gated-poses.json` 和重新归并/估计协方差的 `pose-hypotheses-gated.json`。
门控过程不使用 MSSD/MSPD；这两个真值指标仅在结束后写入 `evaluation_only`
用于论文对照。当前 T-LESS 离线图像没有机器人外参，因此相机工作空间门控已启用，
机器人工作空间接口已实现但未启用；传入 `--robot-camera-transform` 和机器人边界
后才能进行有物理意义的机器人工作空间检查。CAD 投影采用确定性 CPU 点渲染，
适合候选筛选，但后续仍应与精确三角网格渲染做消融。

### 覆盖率公平的 Top-K 评价

门控允许拒绝全部候选，因此不能只比较“保留下来的位姿”中位误差。
`scripts/evaluate_topk_coverage.py` 固定使用 BOP19 `test_targets_bop19.json`
中的目标实例数作为分母；缺失检测、PnP 失败和门控后全拒绝都计为失败。同时输出
完整工作点、相同覆盖率比较、风险—覆盖率曲线、仅用于分析的 Top-K GT oracle，
以及供官方 BOP Toolkit 做一对一 GT 匹配的结果 CSV：

```bash
PY=/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python
$PY scripts/evaluate_topk_coverage.py \
  --method raw=outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/2/estimated-poses.json \
  --method raw=outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/3/estimated-poses.json \
  --method raw=outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1/4/estimated-poses.json \
  --method gated=outputs/pose_gating/tless_000002/gated-poses.json \
  --method gated=outputs/pose_gating/tless_000003/gated-poses.json \
  --method gated=outputs/pose_gating/tless_000004/gated-poses.json \
  --object-id 2 --object-id 3 --object-id 4 \
  --output-dir outputs/topk_evaluation/tless_objects2_4_v1 \
  --run-official --official-workers 1
```

无图形环境时，官方 VSD 渲染需要 EGL，例如在命令前添加
`PYOPENGL_PLATFORM=egl`。输出目录中的 `summary.json` 是完整机器可读报告，
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

python3 scripts/validate_tless_bop19.py
$PY scripts/run_bop19_stage.py gen_templates --workers 2
$PY scripts/run_bop19_stage.py gen_repre --workers 1
$PY scripts/run_bop19_stage.py infer --workers 2

python3 scripts/export_foundpose_bop.py \
  --output outputs/foundpose_bop19/bop_results/foundpose-bop19-standard_tless-test.csv
$PY scripts/evaluate_bop19_official.py \
  --result-csv outputs/foundpose_bop19/bop_results/foundpose-bop19-standard_tless-test.csv
```

`gen_repre` 在 6 GiB GPU 上被强制为单进程，并采用 GPU 提取 DINOv2/PCA 特征、
CPU 执行同一 FAISS K-means 的混合模式，避免 FAISS 默认 1 GiB 临时显存导致
OOM；模板虽允许最多三路，但 RTX 3060 实测采用两路更稳妥，推理最多两路。显存
profile 测量的是整张 GPU 的总占用，并同时记录运行前基线、峰值和峰值增量。

将 FoundPose 结果转换为项目 Top-K 接口，以及从 CAD 生成物体坐标系抓取候选：

```bash
export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop
python3 scripts/convert_foundpose_topk.py \
  --input outputs/foundpose_smoke/inference/tless_smoke_v1/1/estimated-poses.json \
  --object-config config/objects/tless_000001.yaml \
  --output outputs/foundpose_smoke/inference/tless_smoke_v1/1/pose-hypotheses-top5.json
python3 scripts/generate_cad_grasps.py \
  --object-config config/objects/tless_000001.yaml \
  --output outputs/cad_grasps/tless_000001.json
python3 scripts/rank_task_grasps.py \
  --pose-hypotheses outputs/foundpose_smoke/inference/tless_smoke_v1/1/pose-hypotheses-top5.json \
  --grasp-candidates outputs/cad_grasps/tless_000001.json \
  --object-config config/objects/tless_000001.yaml \
  --output outputs/task_grasps/tless_000001_scene20_image3.json
```

正式 BOP 基线使用上游一致的 `best_coarse`；Top-K 研究接口使用独立配置
`config/foundpose_bop19/tless_infer_topk.json`，避免多个同实例假设改变官方基线排名。
对象 2–4 的完整 Top-K 可使用以下命令断点续跑：

```bash
PYOPENGL_PLATFORM=egl $PY scripts/run_bop19_stage.py infer \
  --config config/foundpose_bop19/tless_infer_topk.json \
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

- `scenes/baseline_vision_single.xml`：原单铜柱 RGB-D 场景；
- `scenes/baseline_vision_ten.xml`：原十铜柱无遮挡场景；
- `scenes/exploration_workcell.xml`：原完整上下料工作台场景。

这些文件是新路线的初始环境副本。后续修改新场景不会影响传统基线包。

## 构建与检查

```bash
cd /home/lzt/lzt_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select lzt_cad_grasp_ros
source install/setup.bash
colcon test --packages-select lzt_cad_grasp_ros
colcon test-result --verbose
ros2 run lzt_cad_grasp_ros cad_grasp_exploration_node
```

启动节点当前只检查配置和 MuJoCo 场景是否可加载，并发布探索路线状态；它不会调用传统位姿算法。

## 后续实现顺序

1. 对对象 2–4 全量运行 CAD/RGB-D 位姿修正，并用固定分母与官方 BOP 指标确认
   MSSD 改善是否能保持，同时处理 MSPD 权衡；
2. 用对象 8–10 冻结集验证 RGB-D mask 修正，另行运行改进 mask 下的 FoundPose；
3. 将通过审计的 Top-K 门控扩展到对象 5–10，并形成 Best-1/原始 Top-K/门控
   Top-K 对照；
4. 将目标工业 CAD 作为碰撞体导入 MuJoCo，并执行预测位姿驱动的完整上下料；
5. 实机手眼标定后替换仿真外参，验证真实工作空间、IK 和碰撞检查；
6. 扩展 ROBI 与多工业 CAD 实验，再决定何时恢复对象 20–30 的标准模板任务。

具体选择与操作边界见 `docs/FOUNDPOSE_BASELINE.md`，新增位姿修正与机器人约束链
见 `docs/POSE_REFINEMENT_AND_ROBOT_CONSTRAINTS.md`。
