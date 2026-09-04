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

当前已跑通 FoundPose + T-LESS 小样本，并完成标准 BOP19 对象 1–5 的完整链路；
这仍是阶段基线，不是完整论文方法：

- 已定义 RGB-D 观测、CAD 对象、位姿假设、抓取候选和取放任务的数据契约；
- 已建立可替换的位姿估计器与任务抓取规划器接口；
- 已复制原包的 MuJoCo 机器人资产和三个基线场景；
- 已实现物料 YAML 校验和 CAD 平行夹爪候选生成，真实 T-LESS CAD 已通过实跑；
- FoundPose 已安装到隔离环境，并已在 RTX 3060 上完成 T-LESS 粗 6D 位姿推理；
- 已扩展真实 Top-5 PnP 粗位姿，并输出对称归一化的加权 SE(3) 协方差；
- 已将 Top-K SE(3) 协方差按抓取点力臂传播到取料和放料容差，联合 Top-K
  置信度、IK/碰撞回调、解析抓取质量和运动距离输出鲁棒排序；
- T-LESS BOP19 的 30 个 CAD、20 个场景、1000 张目标图像和 6423 个目标实例已校验完整；
- 标准对象 1–5 均已完成 798 模板、对象表示、完整目标推理、可视化和官方
  VSD/MSSD/MSPD 流程，共输出 1248 个 6D 位姿；
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

标准对象 1–5 的日常查看入口为：

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

1. 当前保持模板生成暂停；恢复后完成余下 T-LESS 标准模板、表示和完整测试推理；
2. 生成全量官方 BOP 指标并写入不可变基线清单；
3. 扩展 ROBI 的标准密度模板和真实实例分割对照实验；
4. 在训练未见工业 CAD 和 MuJoCo 完整上下料中做消融实验。

具体选择与操作边界见 `docs/FOUNDPOSE_BASELINE.md`。
