# 创新点一候选排序消融执行协议

更新日期：2026-10-15

## 1. 目的

验证冻结的 14 维 V3 可靠性概率是否优于单一几何线索，而不是把硬门控、排序和
RGB-D/CAD 精修混为同一个增益来源。本轮不运行 ITODD，不重新训练可靠性模型，也不
评价模式精修或预测协方差。

## 2. 固定输入

- 数据集：T-LESS BOP19，对象 1–30；
- 固定分母：6423 个目标实例；
- 固定候选：20120 个原始 FoundPose Top-5 候选、4024 个候选组；
- 硬门控后候选：19652 个，候选组仍为 4024 个；
- 检测输入：同一份 CNOS-FastSAM 结果；
- 可靠性模型：`tless_objects2_4_logistic_v3.json`；
- 正确候选离线定义：`MSSD_n < 0.1`；
- GT 只用于记录内诊断和官方 BOP19 评价，不参与在线打分。

所有 `gate_*` 方法使用完全相同的硬门控存活候选，只替换排序分数。

## 3. 对比方法

| 方法 | 排序分数 |
|---|---|
| `raw` | 原始 FoundPose 分数，无硬门控 |
| `gate_original` | 硬门控后仍用 FoundPose 原始分数 |
| `gate_reliability_v3` | 冻结 14 维 V3 可靠性概率 |
| `gate_pnp_score` | PnP 内点比例 |
| `gate_pnp_inliers` | PnP 内点数量 |
| `gate_iou` | CAD 投影与观测 Mask IoU |
| `gate_depth` | 负的直径归一化中位深度误差 |

当前 FoundPose `original_score` 就是保存的 PnP 内点比例，且本次数据中 PnP 内点数量
没有改变组内 Top-1，因此三个 PnP 变体会产生相同 BOP CSV。报告必须把它们视为同一
组结果，不能宣称为三个独立算法。

## 4. 评价指标

1. 官方 BOP19 一对一匹配：MSSD、MSPD、VSD 和综合 AR；
2. 候选级 AUC；
3. 固定 6423 分母的 `MSSD_n < 0.1` Top-1 成功率；
4. 开发对象 2–4、历史留出对象 5–10、跨对象集合 1 和 5–30 的逐对象宏平均；
5. 在同一 4024 组候选上，以精确双侧 McNemar 检验比较 Top-1 正确性。

官方总体 AR 与逐对象子集宏平均使用不同聚合方式，必须分栏报告，不能互换。

## 5. 可复现实验入口

```bash
PYOPENGL_PLATFORM=egl \
/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python \
scripts/evaluation/run_candidate_ranking_ablation.py \
  --output-dir outputs/innovation1_ranking_ablation/tless_objects1_30_v3_frozen_v3 \
  --run-official \
  --official-workers 1 \
  --official-cache raw=outputs/innovation1_v3_full/tless_objects1_30_fixed_10mm_5deg_v1/evaluation_official_fixed_10mm_5deg/official_eval/raw/raw_tless-test \
  --official-cache gate_reliability_v3=outputs/innovation1_v3_full/tless_objects1_30_fixed_10mm_5deg_v1/evaluation_official_fixed_10mm_5deg/official_eval/gated/gated_tless-test
```

缓存复用不是按方法名信任旧结果：脚本会核对旧、新 BOP CSV 的 SHA-256；只有内容完全
相同才复用。其余哈希相同的方法只执行一次官方评价并记录来源。

## 6. 证据边界

- 对象 2–4 参与了 V3 可靠性拟合，不是独立测试；
- 对象 5–10 是历史冻结留出，1 和 5–30 是跨对象集合，但仍属于同一 T-LESS 数据域；
- 本消融只证明候选排序模块的作用，不证明模式精修、协方差或抓取成功率；
- 本协议随实现和运行清单一起冻结，但不是实验前公开注册协议。
