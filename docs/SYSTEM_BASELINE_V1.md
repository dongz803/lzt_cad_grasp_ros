# 工业 CAD 上下料系统基线 v1

## 基线目的

本版本冻结 T-LESS 对象 4 的单图系统链路，用于后续算法修改前后的可复现对照。
它是工程与系统版本基线，不是“算法已经达到最佳性能”的结论。

完整数据流为：MuJoCo RGB-D/GT mask 采集 → FoundPose Top-5 → 物理与 RGB-D
几何门控 → CAD 轮廓/深度局部 SE(3) 修正 → CAD 抓取候选与 Top-K 联合排序 →
UR5e IK/碰撞检查 → Robotiq 2F-85 动态抓放与验收。

## 一键运行

```bash
cd /home/lzt/lzt_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/lzt_cad_grasp_ros/scripts/run_tless_obj4_pipeline.py \
  --run-id system_baseline_v1 --seed 0
```

`run-id` 只能包含字母、数字、点、横线和下划线。相同输出目录非空时程序直接拒绝
运行，避免覆盖历史数据。更换 `run-id` 或 `--seed` 即生成新的独立试验。

## 运行目录契约

每次运行保存在：

```text
outputs/system_baselines/tless_000004/<run_id>_seed<seed>/
├── config_snapshot/       # 物料、抓取候选与 FoundPose 配置副本
├── capture/               # RGB-D、mask、BOP 导出及采集清单
├── bop/                   # 本次运行专属单图 BOP 数据
├── foundpose/             # 本次复制归档的原始 Top-K
├── gating/                # 门控结果与逐候选审计
├── refinement/            # 修正结果、预览和项目 Top-K 接口
├── execution/             # 规划、阶段图、GIF 与抓放结果
├── pipeline_manifest.json # Git、种子、阶段命令、耗时和返回码
└── pipeline_result.json   # 最终汇总
```

单独运行抓放脚本时也按照 `gt/predicted × check/dynamic` 分目录，并保存
`run_manifest.json`、输入快照和 SHA256。GT、预测、`check-only` 与动态执行不会
写入同一个默认结果文件。

## 方法边界

- 当前使用 MuJoCo GT mask，因此尚未评价真实分割误差。
- FoundPose、门控和修正不读取 GT 位姿；GT 位姿只在执行后计算仿真误差。
- 当前只冻结对象 4、固定初始位姿和固定相机，不能代表跨 CAD 泛化。
- 成功和失败均是有效实验数据；失败运行仍保存完整清单与 `result.json`。

## 2026-09-11 验证事实

- 新输出隔离、非空目录拒绝覆盖和运行编号校验测试通过。
- Python 全套测试为 53 项通过；ROS 2 构建成功，colcon 汇总 75 项测试，
  0 错误、0 失败、0 跳过。
- 一键命令实际完成全部阶段，并生成各阶段清单与结果。
- 固定种子运行中，FoundPose 生成 5 个候选，门控保留 2 个，修正接受 2 个；
  选择位姿相对仿真真值的位置误差约 2.12 mm、对称感知角度误差约 2.27°。
- 动态执行完成抓取和约 138.49 mm 抬升，但放置后物料倾倒，最终位置误差约
  43.70 mm，未通过 5 mm 放置阈值。该结果暴露出“位姿指标较小但任务可能失败”
  的系统难点，应作为下一阶段任务感知重评分和对称姿态处理的基线失败样本。
- 先前未冻结随机状态的单次运行曾以 2.63 mm 放置误差通过，不能用它代替固定
  配置的统计实验。

## 升级为论文实验基线的条件

在本系统版本上继续补充：多随机初始位姿重复、至少三种工业 CAD、GT/CNOS mask
对照，以及 Best-1、Top-K、门控、修正和任务约束重评分消融。所有方法必须使用
相同目标实例分母，并同时报告覆盖率、位姿指标和完整抓放成功率。
