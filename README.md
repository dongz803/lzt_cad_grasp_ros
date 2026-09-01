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

当前已跑通 FoundPose + T-LESS 的第一阶段小样本基线，但尚未宣称实现论文方法：

- 已定义 RGB-D 观测、CAD 对象、位姿假设、抓取候选和取放任务的数据契约；
- 已建立可替换的位姿估计器与任务抓取规划器接口；
- 已复制原包的 MuJoCo 机器人资产和三个基线场景；
- 已提供铜柱对象配置示例，但后续物料必须通过新配置接入；
- FoundPose 已安装到隔离环境，并已在 RTX 3060 上完成 T-LESS 单图粗 6D 位姿推理；
- T-LESS 用于第一阶段标准复现，ROBI 用于第二阶段反光工业件验证；
- 工业数据集适配器、CAD 抓取生成和最终评分方法仍待实现。

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

FoundPose 的运行输出位于本仓库的 `outputs/foundpose_smoke/`。内部目录是按“处理
阶段—数据集—版本—对象编号”组织的，不是三份重复结果：

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

1. 将小样本配置扩展为 T-LESS BOP19 标准密度模板和完整测试子集；
2. 记录完整 BOP 指标、运行时间和显存，并冻结 FoundPose 基线；
3. 将输出统一为 Top-K 对称位姿假设和不确定度接口；
4. 实现 CAD 抓取候选生成与物料配置校验；
5. 将位姿假设传播到取料—放料联合鲁棒评分；
6. 增加 ROBI→BOP 适配器并做反光工业件验证；
7. 在训练未见工业 CAD 和 MuJoCo 完整上下料中做消融实验。

具体选择与操作边界见 `docs/FOUNDPOSE_BASELINE.md`。
