# 创新点一：问题诊断与修法清单

> 对象：T-LESS 2/3/4 实测（[pose-modes-initial.json](../outputs/innovation1_pose_modes/tless_objects2_4_v1/obj4/refinement/pose-modes-initial.json)、[logistic_v2.json](../lzt_perception/config/reliability/tless_objects2_4_logistic_v2.json)）
> 范围：感知层（`lzt_perception`），不含决策与控制
> 状态：三个问题均已用数据确认，修法待实施

## 摘要

| 编号 | 问题 | 严重度 | 核心修法 | 影响文件 |
|---|---|---|---|---|
| P1 | 16 维特征中 2 维是死特征（恒为常数） | 中 | 硬门控降级为软标志 + 先验下限 | `pose_gating.py`、`pose_reliability.py` |
| P2 | 单成员模态零协方差 + 高权重（双重过度自信） | **高** | 协方差加聚类阈值先验 | `pose_modes.py` |
| P3 | 聚类阈值无依据且可刷分 | **高** | 数据驱动定阈值 + 用官方 AR 消融 + 解耦 | `pose_modes.py`、新增分析脚本 |

P2 与 P3 的解耦方案是**同一处改动**（见 3.3），建议一起实施。

---

## 问题 P1：16 维特征中有 2 维是死特征

### 1.1 问题说明

**现象**

可靠性模型的 16 维特征中，`positive_model_fraction` 与 `projected_model_fraction` 在全部 1949 个训练样本上恒等于 1.0，逻辑回归学到的系数精确为 0。

**证据**（取自 `tless_objects2_4_logistic_v2.json`）

```
feature                          coef       mean      scale   状态
pnp_score                      0.5647     0.2400     0.0631   ok
log1p_pnp_inliers              0.0197     3.7578     0.2731   ok
positive_model_fraction        0.0000     1.0000     1.0000   ❌ 常数
projected_model_fraction       0.0000     1.0000     1.0000   ❌ 常数
silhouette_iou                 0.0838     0.7133     0.1672   ok
...
有效特征数: 14 / 16
```

**根因**（代码级）

这是一处**逻辑循环**：

```python
# pose_gating.py:25  —— 硬门控阈值
minimum_positive_model_fraction: float = 0.95

# pose_gating.py:395-397 —— 用于硬拒绝
positive_fraction = float(projection["positive_model_fraction"])
if positive_fraction < settings.minimum_positive_model_fraction:
    hard_reasons.append("model_behind_camera")      # ← 候选被拒绝

# pose_reliability.py:113-114 —— 同一个量又当特征
_unit(metrics.get("positive_model_fraction")),
_unit(metrics.get("projected_model_fraction")),
```

被拒绝的候选不会进入 `train_pose_reliability.py` 的样本集，因此**幸存者上这两个量必然全部 ≥0.95**，在 T-LESS 数据上实际全部取 1.0 → 方差为零 → 标准化后为零 → 系数学不到任何东西。

**影响**

1. 论文若声称"16 维无量纲特征"，实际只有 14 维在工作，属于可被评审直接质疑的事实性偏差。
2. 更本质的是方法学问题：**门控与特征共享变量**，导致硬门控一旦收紧，相关特征就在幸存者上失去区分度。这是一个会反复出现的结构性缺陷，不只是这两个特征的问题。

### 1.2 修法

推荐**方案 B'**：保留维度，把门控拆成"硬下限 + 软参考点"，让特征在中段有区分度。

**改动 1：`PoseGateConfig` 增加一条硬下限**

```python
# pose_gating.py，PoseGateConfig（约 :25 附近）
minimum_positive_model_fraction: float = 0.95      # 保留：软标志参考点
hard_minimum_positive_model_fraction: float = 0.5  # 新增：硬门控下限
```

`__post_init__` 中增加校验：

```python
if not 0.0 <= self.hard_minimum_positive_model_fraction \
        <= self.minimum_positive_model_fraction <= 1.0:
    raise ValueError(
        "hard_minimum_positive_model_fraction must not exceed "
        "minimum_positive_model_fraction"
    )
```

**改动 2：硬门控改用下限，中段降级为软标志**

```python
# pose_gating.py:395-397，原：
positive_fraction = float(projection["positive_model_fraction"])
if positive_fraction < settings.minimum_positive_model_fraction:
    hard_reasons.append("model_behind_camera")

# 改为：
positive_fraction = float(projection["positive_model_fraction"])
if positive_fraction < settings.hard_minimum_positive_model_fraction:
    hard_reasons.append("model_behind_camera")           # 极端情况仍硬拒绝
elif positive_fraction < settings.minimum_positive_model_fraction:
    soft_flags.append("low_positive_model_fraction")     # 中段转软标志
```

**设计理由**：`positive_model_fraction` 低于 0.5 意味着超过一半的 CAD 点跑到相机背后，这确实物理上不可能；而 0.5–0.95 之间是"位姿可疑但可能正确"，正好是特征该发挥作用的区间。

**备选方案 A（更简单，但信息量为零）**

直接从 `FEATURE_NAMES` 删除这 2 维，诚实报 14 维：

```python
# pose_reliability.py:18-35
FEATURE_NAMES = (
    "pnp_score",
    "log1p_pnp_inliers",
    # "positive_model_fraction",    ← 删除
    # "projected_model_fraction",   ← 删除
    "silhouette_iou",
    ...
)
```

成本：所有已冻结模型失效（`PoseReliabilityModel.from_dict` 会因 `feature_names` 不匹配而 `raise`，这是期望的 fail-fast 行为），需要重训。

**方案对比**

| | 方案 A（删维） | 方案 B'（拆分门控） |
|---|---|---|
| 改动量 | 小 | 中 |
| 特征数 | 14 | 16（全部有效） |
| 是否保留信息 | ❌ 丢弃 | ✅ 恢复区分度 |
| 需要重训 | 是 | 是 |
| 风险 | 无 | 硬门控放松，需验证安全性 |

### 1.3 验证方式

**测试 1：特征非退化断言**（新增 `tests/test_reliability_feature_variance.py`）

```python
def test_all_reliability_features_vary_across_training_samples(audit_paths):
    """训练集上每一维特征的标准差都必须非零。"""
    features = build_feature_matrix(audit_paths)
    scale = features.std(axis=0)
    degenerate = [
        name for name, value in zip(FEATURE_NAMES, scale) if value < 1e-6
    ]
    assert not degenerate, f"degenerate features: {degenerate}"
```

**测试 2：软标志行为**（扩展现有 `tests/test_pose_gating.py`）

```python
def test_mid_range_positive_fraction_becomes_soft_flag_not_rejection():
    decision = evaluate_pose_candidate(record_with_positive_fraction(0.7), ...)
    assert decision.accepted is True
    assert "low_positive_model_fraction" in decision.soft_quality_flags
    assert "model_behind_camera" not in decision.rejection_reasons


def test_extreme_positive_fraction_is_still_rejected():
    decision = evaluate_pose_candidate(record_with_positive_fraction(0.2), ...)
    assert decision.accepted is False
    assert "model_behind_camera" in decision.rejection_reasons
```

**测试 3：指标回归**

```bash
python3 scripts/perception/train_pose_reliability.py \
  --audits outputs/pose_gating/tless_000002/audit.json \
           outputs/pose_gating/tless_000003/audit.json \
           outputs/pose_gating/tless_000004/audit.json \
  --output-dir outputs/pose_reliability/tless_objects2_4_v4
```

验收标准：

| 指标 | 当前（v2） | 要求 |
|---|---|---|
| 有效特征数 | 14 / 16 | **16 / 16** |
| ROC AUC | 0.9009 | 不低于 0.895 |
| ECE | 0.0327 | 不高于 0.040 |
| 硬门控拒绝率变化 | — | 需记录并说明 |

**测试 4：安全性回归**

因为硬门控被放松（0.95 → 0.5），必须确认没有引入物理上不可行的候选：

```bash
python3 scripts/analysis/analyze_symmetry_covariance.py \
  --pose-hypotheses <新的门控输出> \
  --raw-poses <新的门控输出>
# 检查：正深度比例、相机工作空间违规数应为 0
```

---

## 问题 P2：单成员模态零协方差 + 高权重

### 2.1 问题说明

**现象**

单成员模态（`len(member_hypothesis_ids) == 1`）的 `covariance_se3` 是**精确的零矩阵**，但其中相当一部分持有很高的模式权重。

**证据**（T-LESS 2/3/4 全部 1983 个初始模态）

```
单成员模态 n=1073   多成员模态 n=910

单成员模态权重分布:
   p50  = 0.0000
   p75  = 0.2000
   p90  = 0.9774      ← 注意这里
   p95  = 1.0000
   p100 = 1.0000
   >0.5 的占比: 12.6%
   >0.8 的占比: 11.7%

★ 单成员模态(零协方差)持有 >0.8 权重的模态数: 126
```

**最坏个案**（`obj4/scene=5/img=20/inst=0`）

```
模式 1  weight = 0.000000  成员 = [4]  协方差对角线 = 全 0
        ↑ 该候选 score 为 0.0000
```

**根因**（代码级）

```python
# pose_modes.py:182-229  _mode_statistics
residuals = []
for transform in aligned:
    translation = transform[:3, 3] - mean[:3, 3]
    rotation = _rotation_vector(mean[:3, :3].T @ transform[:3, :3])
    residuals.append(np.concatenate((translation, rotation)))
residual_array = np.asarray(residuals, dtype=np.float64)
covariance = (
    residual_array * local_weights[:, None]
).T @ residual_array                                  # ← :221-223
```

当 `len(members) == 1` 时：
- `aligned` 只有一个元素
- `mean`（加权平均）就等于这个元素本身
- 所以 `residual = 0`，`covariance = 0 × 0ᵀ = 0`

协方差矩阵精确为零，**不是数值噪声，是数学必然**。

**影响**

下游 `task_grasp.py::_maximum_std(covariance) = 0`，于是：

```
_tolerance_score(std=0, tol=10°, sigma=2) = exp(-0.5 × 0²) = 1.0
```

**系统判定这些位姿"零误差"。**

而这 126 个模态的性质恰恰相反 —— 它们是**孤立候选**：与其他所有候选的距离都超过了聚类阈值（15° / 30mm），所以单独成组。这**正是 PnP 孤立高分错误解（对称歧义解）的典型特征**，本该最受怀疑，却被赋予了最高可信度。

**这是双重过度自信**：

> "我非常确定是这种摆法"（权重 > 0.8） **+** "这个位姿完全准确"（零协方差）

### 2.2 修法

**核心思路**：单成员模态**无法从数据估计散布**，应改用**有物理意义的保守先验**，而不是报零。

**改动：`_mode_statistics` 增加先验下限**

```python
# pose_modes.py:182-229
def _mode_statistics(
    members: list[PoseHypothesis],
    global_weights: dict[int, float],
    cad_object: CADObject,
    continuous_symmetry_samples: int,
    *,
    prior_translation_std_m: float = 0.0,        # 新增
    prior_rotation_std_deg: float = 0.0,         # 新增
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[float, str]]]:
    ...
    covariance = (
        residual_array * local_weights[:, None]
    ).T @ residual_array

    # 单成员模态没有可估计的散布；用聚类分辨率作为保守先验，
    # 而不是报告精确为零的协方差。
    if len(members) == 1 and (prior_translation_std_m > 0.0
                              or prior_rotation_std_deg > 0.0):
        prior = np.zeros((6, 6), dtype=np.float64)
        prior[:3, :3] = np.eye(3) * prior_translation_std_m ** 2
        prior[3:, 3:] = np.eye(3) * np.radians(prior_rotation_std_deg) ** 2
        covariance = covariance + prior

    covariance = 0.5 * (covariance + covariance.T)
    ...
```

**调用处传入聚类阈值作为先验**（`pose_modes.py:286` 附近）

```python
mean, covariance, metadata = _mode_statistics(
    cluster,
    global_weights,
    cad_object,
    settings.continuous_symmetry_samples,
    prior_translation_std_m=settings.maximum_translation_distance_m,
    prior_rotation_std_deg=settings.maximum_rotation_distance_deg,
)
```

**为什么用聚类阈值当先验？**

因为"这个候选与所有其他候选的距离都超过阈值，所以它独立成模态" —— **阈值本身就是我们对"多远算不同姿态"的定义**。既然无法从单个样本估计散布，那么用"刚好没被合并的最小距离"作为散布的量级，是**自洽且保守**的：15° 就是当前聚类器能分辨的姿态差异下限。

**为什么只在单成员模态加先验？**

多成员模态有 ≥2 个样本，其协方差是真实估计出来的；单成员模态没有样本，协方差恒为零。只在后者加先验，避免污染真实估计。

**辅助改动：`PoseMode` 暴露成员数**

```python
# pose_modes.py，PoseMode 类内
@property
def is_singleton(self) -> bool:
    """单成员模态的协方差来自先验而非样本估计。"""
    return len(self.member_hypothesis_ids) == 1

@property
def member_count(self) -> int:
    return len(self.member_hypothesis_ids)
```

这样决策层（创新点二闭环）可以直接判断"这个模式的不确定性是估计的还是先验的"，从而决定要不要额外打折。

**影响预期**：加了先验后，「期望角度得分」会**下降**（因为不再虚高）。这是**正确的下降** —— 需要配合 P3 的验证确认它更接近真实误差。

### 2.3 验证方式

**测试 1：单成员模态协方差非零**（新增 `tests/test_pose_modes_prior.py`）

```python
def test_singleton_mode_covariance_uses_prior():
    mixture = cluster_pose_hypotheses(
        pose_set_with_one_isolated_candidate,
        cad_object,
        PoseModeClusteringConfig(
            maximum_rotation_distance_deg=15.0,
            maximum_translation_distance_m=0.03,
        ),
    )
    singleton = next(m for m in mixture.modes if m.is_singleton)
    rotation_std_deg = np.degrees(_maximum_std(singleton.covariance_se3[3:, 3:]))
    translation_std_m = _maximum_std(singleton.covariance_se3[:3, :3])
    assert np.isclose(rotation_std_deg, 15.0, atol=1e-6)
    assert np.isclose(translation_std_m, 0.03, atol=1e-9)


def test_multi_member_mode_covariance_is_unaffected_by_prior():
    """先验只作用于单成员模态，多成员模态的协方差必须与加先验前一致。"""
    ...
```

**测试 2：数值回归**（重跑分析脚本）

```bash
python3 scripts/analysis/analyze_pose_modes.py \
  --pose-hypotheses outputs/pose_gating/tless_000004/pose-hypotheses-gated.json \
  --output outputs/symmetry_analysis/after_prior.json
```

验收标准：

| 指标 | 修改前 | 修改后（预期） |
|---|---|---|
| `mode_rotation_std_deg_singleton` 中位 | **9.99e-15** | **15.0** |
| `mode_rotation_std_deg_multi_member` 中位 | 3.47 | 3.47（不变） |
| `mixture_expected` 角度得分中位 | 0.8333 | 下降（预期 0.6–0.75） |

**测试 3：先验是否更接近真实（关键验证）**

这是唯一能证明"先验是对的"而不是"先验让数字变难看"的实验。利用 `gated-poses.json` 里的 GT 误差 `mssd`：

```python
# 伪代码
for mode in mixture.modes:
    predicted_std = _maximum_std(mode.covariance_se3[3:, 3:])   # 声称的不确定度
    actual_error  = 该模式代表位姿对 GT 的旋转误差             # 真实误差
# 比较：predicted_std 与 actual_error 的相关性 / 覆盖率
```

**验收标准**：加了先验后，"声称的 1σ"应该更接近"实际误差分布"，即

```
P(actual_error < k · predicted_std) ≈ 理论值（k=1 时约 68%）
```

修改前这个比例会**远低于** 68%（因为单成员模态声称 0）；修改后应显著改善。

建议把这个做成 `scripts/analysis/calibrate_mode_uncertainty.py`，输出可靠性曲线（reliability diagram）。

---

## 问题 P3：聚类阈值无依据且可刷分

### 3.1 问题说明

**现象**

`maximum_rotation_distance_deg = 15.0` 是硬编码值，没有依据；而"模式内不确定度"随阈值单调变化，意味着相关指标可以被超参数直接刷高。

**证据**（60 组抽样扫描）

| 阈值 | 平均模态数 | 单成员模态占比 | 模态内旋转 std | **期望角度得分** |
|---|---:|---:|---:|---:|
| 5° | 2.51 | **71.4%** | **0.000°** | **0.9893** |
| 10° | 1.81 | 46.9% | 0.758° | 0.8796 |
| **15°（当前）** | 1.58 | 39.3% | 1.705° | 0.8336 |
| 25° | 1.45 | 33.8% | 2.515° | 0.8004 |

**读法**：阈值调小 → 模态更碎 → 单成员更多 → 协方差趋零 → "不确定性得分"趋近满分。

在 5° 时，71.4% 的模态是单成员，模态内旋转 std **精确为 0.000°**，角度得分 **0.9893**。

**根因**

1. 直接原因：`pose_modes.py:46` 硬编码默认值，无消融依据。
2. 结构性原因：**「模式内不确定度」与「聚类阈值」耦合**。
   ```
   阈值 → 模态粒度 → 成员数 → 残差散布 → 协方差
   ```
   协方差不是独立于超参数的估计，而是阈值的函数。

**影响**

- 论文核心主张之一是"模式内 SE(3) 不确定度"，但这个量可以被超参数操纵，评审必然质疑。
- 你自己在 [EXPERIMENT_SUMMARY.md](../outputs/innovation1_pose_modes/tless_objects2_4_v1/EXPERIMENT_SUMMARY.md) 第 60 行已列为待办："下一轮必须增加：聚类阈值消融"。

### 3.2 修法

**修法 3a：用数据驱动方式定阈值**

阈值的物理含义是"多远还算同一个姿态"。应从上界估计：[gated-poses.json](../outputs/pose_gating/tless_000004/gated-poses.json) 中有 GT 误差 `mssd_n`，可以用**已知正确**的候选对来统计。

新增 `scripts/analysis/calibrate_mode_threshold.py`：

```python
def suggest_rotation_threshold(
    audit_paths: list[Path],
    *,
    correct_field: str = "mssd_n",
    correct_threshold: float = 0.1,
    percentile: float = 95.0,
) -> dict[str, float]:
    """从已知正确的候选对估计「同一姿态」的旋转距离上界。

    对每个实例组，取该组内所有满足 mssd_n < correct_threshold 的候选对，
    计算对称归一化后的旋转距离；返回指定分位数作为建议阈值。
    """
```

输出：

```json
{
  "pair_count": 1234,
  "rotation_distance_deg": {"p50": 2.1, "p90": 5.8, "p95": 7.3, "p99": 12.4},
  "suggested_rotation_threshold_deg": 7.3,
  "suggested_translation_threshold_m": 0.012
}
```

**修法 3b：用官方 BOP AR 做阈值消融（而不是用角度得分）**

新增 `scripts/evaluation/run_mode_threshold_ablation.py`：

```python
def run_threshold_ablation(
    thresholds_deg: Sequence[float] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
    audit_inputs: Mapping[int, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """对每个阈值跑完整的「聚类→精修→重评分→重聚类」，用官方 BOP19
    评估选择工作点。

    关键：选择依据必须是 official_bop19 的 AR，不能是模式内角度得分
    （后者可被阈值刷高，见 P3 问题说明）。
    """
```

输出表格：

| 阈值 | 平均模态数 | 单成员占比 | 模式内 std | **官方 AR** |
|---|---:|---:|---:|---:|
| 5° | … | … | … | … |
| 15° | … | … | … | … |

**选点原则**：取官方 AR 的**平台区中心**，而非峰值（避免过拟合到某个阈值）。

**修法 3c：解耦「不确定度」与「阈值」（与 P2 同源）**

这是根本修法，且**与 P2 的改动是同一处**：给协方差加先验下限后，

- 阈值只决定"分几组"（结构性作用）
- 协方差由"数据估计 + 固定先验"决定（度量性作用）

**解耦验证**：重跑阈值扫描，若"模态内 std"不再随阈值单调变化（趋于平稳），说明解耦成功。

### 3.3 与 P2 的关系

```
P2 修法（加先验）  ──┬──→ 解决：单成员模态零协方差
                     └──→ 同时解决：P3 的"阈值可刷分"
                          因为协方差不再由阈值唯一决定
```

所以 **P2 和 P3 应作为一个改动一起实施**，P3 的额外工作只是"补消融实验"。

### 3.4 验证方式

**测试 1：阈值消融完整性**

```bash
python3 scripts/evaluation/run_mode_threshold_ablation.py \
  --thresholds 5 10 15 20 25 30 \
  --audit obj2=outputs/.../obj2/gating/audit.json \
  --audit obj3=outputs/.../obj3/gating/audit.json \
  --audit obj4=outputs/.../obj4/gating/audit.json \
  --output-dir outputs/innovation1_pose_modes/threshold_ablation_v1
```

验收标准：
- 6 个阈值均有完整官方 BOP19 结果
- 报告中同时列出 AR 与模式内 std，用于说明二者不相关（解耦证据）

**测试 2：解耦断言**（新增 `tests/test_mode_threshold_decoupling.py`）

```python
def test_mode_uncertainty_is_insensitive_to_threshold_after_prior():
    """加先验后，单成员模态的不确定度由先验决定，不随阈值坍缩到零。"""
    stds = []
    for threshold in (5.0, 15.0, 30.0):
        mixture = cluster(hypotheses, cad_object, config(threshold))
        singletons = [m for m in mixture.modes if m.is_singleton]
        stds.append(median_rot_std_deg(singletons))
    assert min(stds) > 5.0, "单成员模态不确定度仍随阈值坍缩"
```

**测试 3：阈值建议的可复现性**

```bash
python3 scripts/analysis/calibrate_mode_threshold.py \
  --audits outputs/pose_gating/tless_00000{2,3,4}/audit.json \
  --output outputs/symmetry_analysis/mode_threshold_calibration.json
```

要求：确定性输出（同输入同结果），且建议阈值与消融最优区间一致（若不一致需解释）。

---

## 实施顺序与依赖

```
P2 + P3c（同一处改动：加先验下限）
   │
   ├─→ P3a 阈值标定（独立，可并行）
   │
   └─→ P1（独立，但需重训模型）
            │
            └─→ P3b 阈值消融（需要稳定的门控+特征，故放最后）
                      │
                      └─→ 全部完成后重跑端到端：
                          scripts/evaluation/run_innovation1_pose_ablation.py
```

**建议批次**：

| 批次 | 内容 | 理由 |
|---|---|---|
| 第 1 批 | P2 + P3c（加先验） | 单一改动，收益最大，风险最低 |
| 第 2 批 | P1（门控拆分 + 重训） | 独立，但需要重训和指标回归 |
| 第 3 批 | P3a + P3b（标定 + 消融） | 依赖前两批的稳定实现 |

## 全局验收标准

完成三批后，必须同时满足：

| 项 | 标准 |
|---|---|
| 特征有效性 | 16 / 16 维非退化 |
| 单成员模态协方差 | 不再为零，等于聚类先验 |
| 协方差—阈值解耦 | 模态内 std 不随阈值单调坍缩 |
| 官方 BOP AR | 不低于当前 0.43118（允许 ±0.01） |
| 覆盖率 | 不低于当前 0.54757 |
| 不确定性可靠性 | 声称 1σ 的实际覆盖率显著优于修改前 |
| 测试 | `pytest tests/ -q` 全绿（当前 85 passed） |

## 风险提示

1. **P1 的硬门控放松（0.95 → 0.5）会引入更多候选**，必须确认没有物理上不可行的解进入训练集，否则会污染可靠性标签。
2. **P2 的先验是"保守估计"，不是真实估计**。论文中必须明确区分"从样本估计的协方差"和"来自聚类分辨率先验的协方差"，不能混为一谈。建议在 `PoseMode` 上用 `is_singleton` 显式标注。
3. **P3b 的阈值消融有选择性偏差风险**：如果在同一批数据上选阈值又报告结果，会被质疑过拟合。正确做法是在对象 2–4 上选阈值，在**独立 CAD**（如 T-LESS 5–10 或 ROBI）上验证。
