# 创新点一：T-LESS 对象 1–30 全量运行手册

更新日期：2026-10-14  
适用分支：`develop/innovation1-v3`  
固定开发参数：对象 2–4 选择的 10 mm/5° 模式聚类阈值  
冻结可靠性模型：`tless_objects2_4_logistic_v3.json`

本手册定义运行顺序、输入输出和停止点。2026-10-14 已按手册完成对象 1–30
的固定配置运行和官方 BOP19 评价。再现或续跑前仍必须检查 Git 状态和实际输出；
不要删除、覆盖或回退现有 V3 修改。

## 1. 当前实际状态

| 阶段 | 已完成 | 待完成 |
|---|---|---|
| T-LESS BOP19 数据与 CNOS 检测 | 对象 1–30 | 无 |
| FoundPose 798 张/CAD 模板 | 对象 1–30 | 无 |
| FoundPose `repre.pth` | 对象 1–30 | 无 |
| FoundPose 标准 Best-1 | 对象 1–30 | 无 |
| FoundPose Top-5 | 对象 1–30 | 无 |
| 项目对象 YAML | 对象 1–30 | 无 |
| 创新链路正式评价 | 对象 1–30 统一固定配置汇总 | 无 |

现有 Top-5、对象 2–4 开发集和对象 5–10 留出集都要保留。后续运行使用新目录，
FoundPose 阶段依靠完成检查跳过已有对象，不使用 `--force`。

本次成功运行的主要输出为：

- 标准 FoundPose Best-1：4024 条位姿，官方综合 AR 0.32545；
- Top-5：4024 组、20120 个原始候选；物理硬门控后保留 19652 个候选，未丢失任何候选组；
- 逐候选精修：4024 组、19652 个候选，19427 个接受更新；
- 10 mm/5° 模式精修：18345 个初始模式，17344 个精修后模式；
- BOP 固定分母为 6423 个目标实例，4024 个有位姿输出的组对应 62.65% 预测覆盖率。

## 2. 实验边界

1. 对象 2–4 仍是开发集；10 mm/5° 和 14 维可靠性模型已经冻结。
2. 对象 5–10 已经完成一次独立留出评价，不得根据其结果重新训练可靠性模型或选择阈值。
3. 对象 11–30 只用于扩大固定配置评价范围，也不得用于重新选择上述参数。
4. 对象 2–4 的预测协方差模型只允许加载到对象 2、3、4。对象 1、5–30 运行模式精修时
   不得传入该模型；它们只能输出模式内离散协方差，不能宣称预测协方差已完成标定。
5. T-LESS 离线位姿评价不要求生成抓取候选；某个对象在 85 mm 夹爪下没有抓取候选，
   不阻塞位姿感知实验。

## 3. 长任务开始前的固定变量

以下示例使用全新目录 `tless_objects1_30_fixed_10mm_5deg_v1`。如果该目录已经存在，
请更换名称，不要删除旧目录。

```bash
cd /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros

git status --short --branch
git rev-parse --short HEAD

export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop
FPY=/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python
DATASET_ROOT=/home/lzt/lzt_ros2_ws/datasets/bop/tless
DETECTIONS=/home/lzt/lzt_ros2_ws/datasets/bop/detections/cnos-fastsam/cnos-fastsam_tless-test.json
TARGETS=/home/lzt/lzt_ros2_ws/datasets/bop/tless/test_targets_bop19.json
RELIABILITY_MODEL=/home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros/lzt_perception/config/reliability/tless_objects2_4_logistic_v3.json
PREDICTIVE_MODEL=/home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros/lzt_perception/config/uncertainty/tless_objects2_4_predictive_v3.json
TOPK_BASE=outputs/foundpose_bop19_topk/inference/tless_bop19_standard_topk_v1
RUN_ROOT=outputs/innovation1_v3_full/tless_objects1_30_fixed_10mm_5deg_v1

python3 scripts/datasets/validate_tless_bop19.py
test -x "$FPY"
test -f "$DETECTIONS"
test -f "$TARGETS"
test -f "$RELIABILITY_MODEL"
test -f "$PREDICTIVE_MODEL"

if [ -e "$RUN_ROOT" ]; then
  echo "将按完整结果文件检查并续跑已有目录：$RUN_ROOT"
else
  echo "将创建新的全量实验目录：$RUN_ROOT"
fi
```

## 4. 对象 11–30 项目配置（已完成）

2026-10-12 已使用 `scripts/datasets/generate_tless_object_configs.py` 从
`datasets/bop/tless/models_cad/models_info.json` 补齐对象 11–30，并与现有对象 1–10
执行同源一致性检查。生成规则如下，不能复制某一个已有对象：

- `symmetries_continuous` 转为 `symmetry.type: continuous_axis`，轴保持在物体坐标系，
  BOP 的毫米偏移转换为米；
- `symmetries_discrete` 转为 `symmetry.type: discrete`，4×4 变换中的平移从毫米转换为米；
- 两类字段都不存在时写为 `symmetry.type: none`；
- CAD 路径必须指向对应的 `models_cad/obj_XXXXXX.ply`；
- 真实夹爪最大开口仍保持 0.085 m，不得为了生成抓取候选而修改。

对象 1–30 的最终分类为：连续轴对称 11 个、离散对称 16 个、无显式对称 3 个；
其中对象 27 保留 `models_info.json` 中的 3 个离散变换。30 个配置均已通过
`load_object_config`、CAD 文件存在性和变换数值一致性检查。

生成工具默认只检查；只有显式传入 `--write-missing` 才会写缺失文件，并且不会覆盖
任何已有配置：

```bash
python3 scripts/datasets/generate_tless_object_configs.py \
  --object-start 1 --object-end 30
```

补齐后执行只读检查：

```bash
for OBJ_ID in $(seq 1 30)
do
  OBJ_PAD=$(printf '%06d' "$OBJ_ID")
  test -f "lzt_description/config/objects/tless_${OBJ_PAD}.yaml" || exit 1
  test -f "$BOP_PATH/tless/models_cad/obj_${OBJ_PAD}.ply" || exit 1
done

python3 - <<'PY'
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
from _bootstrap import activate_project_packages
activate_project_packages(root)
from lzt_common.object_config import load_object_config

for object_id in range(1, 31):
    path = root / "lzt_description/config/objects" / f"tless_{object_id:06d}.yaml"
    config = load_object_config(path)
    assert config.cad_object.object_id == f"tless_{object_id:06d}"
print("30 object configs: OK")
PY
```

## 5. FoundPose 资源和 Top-5

### 5.1 补模板：对象 20–30

状态：已于 2026-10-13 完成。对象 20–30 的 `metadata.json` 均含 798 条记录，
`rgb`、`depth`、`mask` 均各有 798 张 `template_0000.png`–`template_0797.png`，
逐对象 profile 的 `exit_code` 均为 0。生成过程中对象 28–30 曾因运行会话中断留下
不完整输出，随后仅重跑对象 28–30 并通过同一完整性检查；对象 20–27 未重复覆盖。

预计输出到：
`outputs/foundpose_bop19/templates/bop19_standard_v1/tless/<对象号>/`。

```bash
PYOPENGL_PLATFORM=egl "$FPY" scripts/pipeline/run_bop19_stage.py gen_templates \
  --output-root outputs/foundpose_bop19 \
  --object-start 20 --object-end 30 \
  --workers 2
```

停止并检查：`outputs/foundpose_bop19/profiles/gen_templates_summary.json` 中请求范围全部完成，
每个对象的 metadata、RGB、depth、mask 数量均为 798。

### 5.2 补表示：对象 11–30

状态：已于 2026-10-13 完成，20 个 `repre.pth` 均存在且非空，逐对象 profile 均为成功。

该阶段在 6 GiB GPU 上必须保持单进程。预计输出到：
`outputs/foundpose_bop19/object_repre/tless/bop19_standard_v1/<对象号>/repre.pth`。

```bash
"$FPY" scripts/pipeline/run_bop19_stage.py gen_repre \
  --output-root outputs/foundpose_bop19 \
  --object-start 11 --object-end 30 \
  --workers 1
```

停止并检查全部 20 个 `repre.pth` 存在且非空，再进入推理。

### 5.3 补标准 FoundPose Best-1：对象 11–30

状态：已于 2026-10-13 完成对象 1–30 统一官方评价。标准 Best-1 官方
MSSD/MSPD/VSD/综合 AR 分别为 0.26408/0.46986/0.24241/0.32545。

这一步建立完整的 FoundPose 官方基线；后面的创新链路仍使用独立 Top-5 配置。
命令会自动跳过已经完成的对象 1–10。

```bash
PYOPENGL_PLATFORM=egl "$FPY" scripts/pipeline/run_bop19_stage.py infer \
  --config lzt_perception/config/foundpose_bop19/tless_infer.json \
  --output-root outputs/foundpose_bop19 \
  --object-start 1 --object-end 30 \
  --workers 1
```

全部对象完成后再单独导出和评价，避免与创新链路的 Top-5 原始对照混为一谈：

```bash
STANDARD_CSV=outputs/foundpose_bop19/bop_results/foundpose-bop19-standard-1to30_tless-test.csv
STANDARD_EVAL=outputs/foundpose_bop19/official_eval_1to30_v1

if [ -e "$STANDARD_CSV" ] || [ -e "$STANDARD_EVAL" ]; then
  echo "标准FoundPose 1–30评价输出已存在，为防止覆盖已停止"
  exit 1
fi

python3 scripts/evaluation/export_foundpose_bop.py \
  --input-root outputs/foundpose_bop19/inference/tless_bop19_standard_v1 \
  --object-start 1 --object-end 30 \
  --output "$STANDARD_CSV"

PYOPENGL_PLATFORM=egl "$FPY" scripts/evaluation/evaluate_bop19_official.py \
  --result-csv "$STANDARD_CSV" \
  --eval-root "$STANDARD_EVAL" \
  --num-workers 1
```

### 5.4 补 Top-5：对象 1、11–30

状态：已于 2026-10-13 完成对象 1–30，每组均含 5 个候选。对象 21 首次遇到
PyTorch 瞬时计时器异常，仅重试对象 21–30 后成功，未修改已有输出。

下面命令请求 1–30，但会自动跳过已有的对象 2–10；不要添加 `--force`。

```bash
test -L outputs/foundpose_bop19_topk/object_repre || exit 1

PYOPENGL_PLATFORM=egl "$FPY" scripts/pipeline/run_bop19_stage.py infer \
  --config lzt_perception/config/foundpose_bop19/tless_infer_topk.json \
  --output-root outputs/foundpose_bop19_topk \
  --object-start 1 --object-end 30 \
  --workers 1
```

停止并检查：`outputs/foundpose_bop19_topk/profiles/infer_summary.json` 应列出 1–30 全部完成。

### 5.5 转换为项目 Top-5 格式

已有转换结果自动跳过；若发现只有部分文件的目录，先停止检查，不要覆盖。

```bash
for OBJ_ID in $(seq 1 30)
do
  OBJ_PAD=$(printf '%06d' "$OBJ_ID")
  OBJ_DIR="$TOPK_BASE/$OBJ_ID"
  OUTPUT="$OBJ_DIR/pose-hypotheses-top5.json"

  if [ -f "$OUTPUT" ]; then
    echo "[$OBJ_ID] converted Top-5 exists; skipping"
    continue
  fi

  python3 scripts/perception/convert_foundpose_topk.py \
    --input "$OBJ_DIR/estimated-poses.json" \
    --object-config "lzt_description/config/objects/tless_${OBJ_PAD}.yaml" \
    --output "$OUTPUT" \
    --top-k 5 || exit 1
done
```

## 6. 新目录中运行完整创新链路

从本节开始统一重跑对象 1–30，形成单一、可审计的实验目录。每个循环只跳过具有完整
结果文件的对象；遇到不完整目录时停止，避免覆盖半成品。

状态：已于 2026-10-14 完成对象 1–30 的物理门控与 14 维可靠性、逐候选精修、
对称模式聚类、模式中心精修和精修后重校准。全部 4024 组运行时均未使用 GT。

### 6.1 物理门控和 14 维可靠性

```bash
for OBJ_ID in $(seq 1 30)
do
  OBJ_PAD=$(printf '%06d' "$OBJ_ID")
  GATE_DIR="$RUN_ROOT/gating/tless_${OBJ_PAD}"

  if [ -f "$GATE_DIR/gated-poses.json" ] && \
     [ -f "$GATE_DIR/audit.json" ] && \
     [ -f "$GATE_DIR/summary.json" ]; then
    echo "[$OBJ_ID] gating complete; skipping"
    continue
  fi
  if [ -e "$GATE_DIR" ]; then
    echo "[$OBJ_ID] incomplete gating directory; stopped: $GATE_DIR"
    exit 1
  fi

  python3 scripts/perception/gate_foundpose_topk.py \
    --input "$TOPK_BASE/$OBJ_ID/estimated-poses.json" \
    --object-config "lzt_description/config/objects/tless_${OBJ_PAD}.yaml" \
    --detections "$DETECTIONS" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$GATE_DIR" \
    --top-k 5 \
    --render-point-count 12000 \
    --reliability-model "$RELIABILITY_MODEL" || exit 1
done
```

### 6.2 逐候选精修强基线

```bash
for OBJ_ID in $(seq 1 30)
do
  OBJ_PAD=$(printf '%06d' "$OBJ_ID")
  GATE_DIR="$RUN_ROOT/gating/tless_${OBJ_PAD}"
  OUTPUT_DIR="$RUN_ROOT/per_candidate/obj${OBJ_ID}"

  if [ -f "$OUTPUT_DIR/refined-poses.json" ] && [ -f "$OUTPUT_DIR/summary.json" ]; then
    echo "[$OBJ_ID] per-candidate refinement complete; skipping"
    continue
  fi
  if [ -e "$OUTPUT_DIR" ]; then
    echo "[$OBJ_ID] incomplete per-candidate directory; stopped: $OUTPUT_DIR"
    exit 1
  fi

  python3 scripts/perception/refine_foundpose_poses.py \
    --input "$GATE_DIR/gated-poses.json" \
    --object-config "lzt_description/config/objects/tless_${OBJ_PAD}.yaml" \
    --detections "$DETECTIONS" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --top-k 5 \
    --render-point-count 5000 \
    --preview-limit 0 \
    --progress-every 50 \
    --reliability-model "$RELIABILITY_MODEL" \
    --gate-audit "$GATE_DIR/audit.json" || exit 1
done
```

### 6.3 10 mm/5° 对称模式中心精修

对象 2–4 才加载现有预测协方差；其余对象明确不加载，避免跨对象静默误用。

```bash
for OBJ_ID in $(seq 1 30)
do
  OBJ_PAD=$(printf '%06d' "$OBJ_ID")
  GATE_DIR="$RUN_ROOT/gating/tless_${OBJ_PAD}"
  OUTPUT_DIR="$RUN_ROOT/mode_aware_10mm_5deg/obj${OBJ_ID}"

  if [ -f "$OUTPUT_DIR/refined-poses.json" ] && \
     [ -f "$OUTPUT_DIR/pose-modes-refined.json" ] && \
     [ -f "$OUTPUT_DIR/summary.json" ]; then
    echo "[$OBJ_ID] mode-aware refinement complete; skipping"
    continue
  fi
  if [ -e "$OUTPUT_DIR" ]; then
    echo "[$OBJ_ID] incomplete mode-aware directory; stopped: $OUTPUT_DIR"
    exit 1
  fi

  COVARIANCE_ARGS=()
  if [ "$OBJ_ID" -ge 2 ] && [ "$OBJ_ID" -le 4 ]; then
    COVARIANCE_ARGS=(--predictive-covariance-model "$PREDICTIVE_MODEL")
  fi

  python3 scripts/perception/refine_foundpose_poses.py \
    --input "$GATE_DIR/gated-poses.json" \
    --object-config "lzt_description/config/objects/tless_${OBJ_PAD}.yaml" \
    --detections "$DETECTIONS" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --top-k 5 \
    --render-point-count 5000 \
    --preview-limit 0 \
    --progress-every 50 \
    --reliability-model "$RELIABILITY_MODEL" \
    --gate-audit "$GATE_DIR/audit.json" \
    --mode-aware \
    --mode-translation-threshold-m 0.01 \
    --mode-rotation-threshold-deg 5 \
    --mode-continuous-symmetry-samples 360 \
    "${COVARIANCE_ARGS[@]}" || exit 1
done
```

## 7. 统一对象 1–30 官方 BOP19 评价

评价目录必须是新的。缺失检测、PnP 失败和门控后全拒绝继续计入固定目标分母。
状态：已于 2026-10-14 完成。官方 VSD 需要可用的 EGL/GPU 设备；受限沙箱中可能在
VisPy EGL 初始化时失败，本次在具有 EGL 设备访问权限的环境中成功完成。

官方评价由发起脚本用 `sys.executable` 作为子进程拉起，因此**发起本次运行的
解释器本身必须能 `import imageio, vispy`**。项目内应使用
`.envs/foundpose_gpu/bin/python`（runbook 中的 `$FPY`）；若用系统
`/usr/bin/python3` 启动，评价阶段会报 `ModuleNotFoundError: No module named
'imageio'`。相关入口（`evaluate_bop19_official.py`、`evaluate_topk_coverage.py`、
`run_innovation1_pose_ablation.py`、`run_mode_threshold_sweep.py`）现在会在开工前
检查该依赖并直接给出可执行的换解释器提示，不会等到精修全部跑完才失败。

```bash
EVAL_OFFICIAL="$RUN_ROOT/evaluation_official_fixed_10mm_5deg"

if [ -e "$EVAL_OFFICIAL" ]; then
  echo "正式评价目录已存在，为防止覆盖已停止：$EVAL_OFFICIAL"
  exit 1
fi

EVAL_ARGS=()
for OBJ_ID in $(seq 1 30)
do
  OBJ_PAD=$(printf '%06d' "$OBJ_ID")
  EVAL_ARGS+=(--method "raw=$TOPK_BASE/$OBJ_ID/estimated-poses.json")
  EVAL_ARGS+=(--method "gated=$RUN_ROOT/gating/tless_${OBJ_PAD}/gated-poses.json")
  EVAL_ARGS+=(--method "per_candidate=$RUN_ROOT/per_candidate/obj${OBJ_ID}/refined-poses.json")
  EVAL_ARGS+=(--method "mode_aware=$RUN_ROOT/mode_aware_10mm_5deg/obj${OBJ_ID}/refined-poses.json")
  EVAL_ARGS+=(--object-id "$OBJ_ID")
done

PYOPENGL_PLATFORM=egl "$FPY" scripts/evaluation/evaluate_topk_coverage.py \
  "${EVAL_ARGS[@]}" \
  --targets "$TARGETS" \
  --detections "$DETECTIONS" \
  --raw-method raw \
  --success-field mssd_n \
  --success-threshold 0.1 \
  --output-dir "$EVAL_OFFICIAL" \
  --run-official \
  --official-workers 1
```

## 8. 最终完成检查

实际官方 BOP19 结果（对象 1–30）：

| 方法 | MSSD AR | MSPD AR | VSD AR | 综合 AR |
|---|---:|---:|---:|---:|
| 原始 FoundPose Top-5 选择 | 0.25932 | 0.46955 | 0.23816 | 0.32234 |
| 物理门控 + 14 维可靠性 | 0.36740 | 0.45334 | 0.34025 | 0.38700 |
| 逐候选 RGB-D/CAD 精修 | **0.43705** | **0.47511** | **0.39901** | **0.43706** |
| 10 mm/5° 对称模式精修 | 0.43645 | 0.47486 | 0.39882 | 0.43671 |

表中数值来自各方法 `scores_bop19.json` 的官方整体指标。`summary.json` 还保存对象
1–30 逐对象等权宏平均，两种平均口径不应混用。精修输出主动删除了输入中陈旧的
GT 误差字段，因此其快速记录内 MSSD/MSPD 值为 0 只表示“不复用陈旧误差”，精度结论必须
读取上表的官方重算结果。

不能只看命令是否退出。至少检查：

```bash
test -f "$EVAL_OFFICIAL/summary.json"
jq '.methods | with_entries(.value |= {
  target_instances,
  selected_predictions,
  coverage,
  official_bop19: .official_bop19.macro_average
})' "$EVAL_OFFICIAL/summary.json"

pgrep -af 'run_bop19_stage|gate_foundpose_topk|refine_foundpose_poses|evaluate_topk_coverage' || true
```

结果整理时必须同时报告：

- 1–30 固定分母目标数和预测覆盖率；
- raw、gated、per-candidate、mode-aware 的 MSSD/MSPD/VSD/综合 AR；
- 每对象结果，尤其保留失败对象；
- 可靠性模型只在对象 2–4 训练；
- 10 mm/5° 只在对象 2–4 选择；
- 预测协方差只对对象 2–4 有效，对象 1、5–30 未完成标定；
- 1–30 结果是固定模型的扩展评价，不能反过来用于调参后仍称为独立测试。

## 9. 推荐停止点

长任务不要一次无检查地串到底，按以下阶段停止：

1. 对象 11–30 YAML 补齐并通过加载检查；
2. 对象 20–30 模板全部 798 张；
3. 对象 11–30 `repre.pth` 全部存在；
4. 对象 11–30 标准 Best-1 和官方基线评价完成；
5. 对象 1、11–30 Top-5 推理和格式转换完成；
6. 对象 1–30 门控完成；
7. 对象 1–30 逐候选精修完成；
8. 对象 1–30 模式中心精修完成；
9. 最后单独运行官方评价并核对 `summary.json`。

当前已完成全部 9 个停止点并逐阶段验收。最终输出位于
`outputs/innovation1_v3_full/tless_objects1_30_fixed_10mm_5deg_v1/`；当前没有后台实验进程。
