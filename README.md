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

当前已确定 FoundPose + T-LESS 的第一阶段基线，但尚未宣称实现论文方法：

- 已定义 RGB-D 观测、CAD 对象、位姿假设、抓取候选和取放任务的数据契约；
- 已建立可替换的位姿估计器与任务抓取规划器接口；
- 已复制原包的 MuJoCo 机器人资产和三个基线场景；
- 已提供铜柱对象配置示例，但后续物料必须通过新配置接入；
- FoundPose 作为训练自由的粗 6D 物体位姿前端，尚未完成本机环境安装和推理；
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

1. 安装 FoundPose 独立环境并下载 T-LESS 最小数据集合；
2. 跑通 FoundPose 开源粗位姿基线和 BOP 评测；
3. 将输出统一为 Top-K 对称位姿假设和不确定度接口；
4. 实现 CAD 抓取候选生成与物料配置校验；
5. 将位姿假设传播到取料—放料联合鲁棒评分；
6. 增加 ROBI→BOP 适配器并做反光工业件验证；
7. 在训练未见工业 CAD 和 MuJoCo 完整上下料中做消融实验。

具体选择与操作边界见 `docs/FOUNDPOSE_BASELINE.md`。
