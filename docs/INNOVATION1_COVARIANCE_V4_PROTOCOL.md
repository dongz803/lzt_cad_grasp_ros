# 创新点一：预测协方差 V4 固定协议

## 1. 目的

本协议将预测位姿协方差的覆盖率从离线报告指标闭环到以下环节：

1. 训练集拟合基础协方差；
2. 校准集选择收缩系数、协方差放大系数和经验马氏距离阈值；
3. 测试集只进行最终覆盖率与锐度评价；
4. 在线加载器只允许使用通过验收且适用于当前对象的模型。

GT 只用于离线选择正确模式和构造对称感知 SE(3) 残差，不进入在线路径。
椭球外残差必须计为未覆盖，不能根据覆盖结果删除样本。

## 2. 固定数据划分

对象按可观测子空间分层。连续轴对称对象去除不可观测的轴向旋转，使用秩 5
投影；其余对象使用完整秩 6 子空间。

| 分层 | 训练对象 | 校准对象 | 测试对象 |
|---|---|---|---|
| 连续轴对称、秩 5 | 1、2、3、4、16、17、30 | 14、24 | 13、15 |
| 完整 SE(3)、秩 6 | 5、6、8、10、12、20、22、23、25、26、29 | 7、9、21、28 | 11、18、19、27 |

测试对象不参与协方差拟合、超参数选择、尺度校准或经验阈值计算。

## 3. 候选模型与选择规则

- 收缩系数搜索：`0、0.05、0.1、0.2、0.5`；
- 审计两种语义：`predictive-only` 和
  `predictive + candidate dispersion`；
- 正式部署语义固定为后者，与在线总协方差定义一致；
- 使用校准集选择满足 90% 覆盖率下限的最小放大尺度；
- 合格候选优先选择平均总协方差 log-pseudodeterminant 更小者，再比较最大标准差；
- 使用有限样本 split-conformal 上分位数保存 50%、68.27%、90%、95%
  的经验马氏距离阈值；
- 测试集只评价一次，不用于再次选择模型。

## 4. 验收和在线准入

固定验收条件为：

- 校准集 90% 覆盖率不低于 90%；
- 测试集整体 90% 覆盖率不低于 85%；
- 单对象 90% 覆盖率不低于 80%；
- 单对象正确模式测试样本不少于 30；
- 最大平移标准差不超过 10 mm；
- 最大可观测旋转标准差不超过 15°。

Schema v2 模型保存审批对象、拒绝对象、尺度、经验阈值、协方差语义、数据划分和
验收结果。默认在线行为是 fail closed：旧 Schema v1、未审批对象、模型整体未通过
或在线语义不一致时拒绝加载。研究调试覆盖参数必须显式指定。

## 5. 可复现命令

```bash
python3 scripts/analysis/calibrate_pose_mode_uncertainty.py \
  --protocol-config lzt_perception/config/uncertainty/tless_objects1_30_covariance_v4_protocol.json \
  --output-dir outputs/uncertainty/tless_objects1_30_covariance_v4_frozen_v1 \
  --dataset-root /home/lzt/lzt_ros2_ws/datasets/bop/tless \
  --detections /home/lzt/lzt_ros2_ws/datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json \
  --continuous-symmetry-samples 360 \
  --correct-mode-normalized-surface-threshold 0.1 \
  --minimum-gt-match-iou 0.1
```

该命令只读取已有精修模式并进行离线统计，不重新运行 FoundPose、RGB-D/CAD 精修
或官方 BOP 评价。

## 6. 证据边界

对象 1–30 的模式和聚合结果在冻结本协议前已经存在，因此这次属于“前瞻冻结的
协方差协议评价”，不能表述成从未看过数据的外部盲测。当前模型还以“存在 GT
关联正确模式”为条件，不等价于整个感知系统的成功率；正确模式存在率另行使用 GT
离线统计，当前尚不能在线预测。
