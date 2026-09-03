# ROBI→BOP 适配与反光工业件验证

## 验证范围

ROBI 官方数据包含 7 类反光工业物料、63 个料箱场景、RealSense 与 Ensenso
多视角数据以及逐实例 6D 真值。本项目先验证 `Chrome_screw`，格式与推理链路稳定后
再加入具有 12 重离散对称性的 `Gear`。

官方来源：

- 数据集主页：<https://www.trailab.utias.utoronto.ca/robi>
- 官方 Toolkit：<https://github.com/junyang224/ROBI_Toolkit>

## 坐标与数据映射

官方 `cam_pose` 表示相机在世界坐标系中的位姿 `T_world_camera`，
`GT_world2obj.json` 中矩阵表示 `T_world_object`。BOP 所需物体相对相机位姿为：

```text
T_camera_object = inverse(T_world_camera) × T_world_object
```

RealSense RGB 和深度相机拥有不同内参及各自位姿。适配器不会把原深度图直接伪装为
RGB 对齐深度，而是先用深度内参反投影，再通过两个相机的世界位姿变换，最后投影到
RGB 相机并执行 z-buffer。输出深度单位为毫米，BOP `depth_scale=1.0`。

对象编号固定为：Zigzag=1、Chrome Screw=2、Gear=3、Eye Bolt=4、
Tube Fitting=5、DIN Connector=6、D-Sub Connector=7。转换同时保存
`robi_scene_map.json` 和每场景的 `robi_image_map.json`，保证BOP编号能追溯回原始数据。

## 使用顺序

```bash
cd /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros
PY=/home/lzt/lzt_ros2_ws/.envs/foundpose_gpu/bin/python

$PY scripts/prepare_robi_bop_models.py --objects Chrome_screw Gear

python3 scripts/convert_robi_to_bop.py \
  --objects Chrome_screw \
  --scene-ids 4 \
  --maximum-views-per-scene 5

PYOPENGL_PLATFORM=egl $PY scripts/render_robi_bop_masks.py
```

最后一步从真实CAD和6D真值渲染 `mask`、`mask_visib` 与
`scene_gt_info.json`，并生成名为 `cnos-fastsam_robi-test.json` 的GT掩膜检测文件。
该文件仅用于隔离并验证反光外观条件下的FoundPose位姿前端，不能宣称为真实检测
结果；后续应单独接入CNOS或实例分割网络评价检测召回率。

## 当前证据

- 真实官方 Chrome Screw 与 Gear STL 已转换为毫米PLY；直径分别为
  29.475 mm 和 37.191 mm。
- 两个真实CAD均通过物料配置校验，并各生成12个平行夹爪候选。
- RGB/深度相机标定、深度重投影、世界到相机位姿变换、场景/图像编号映射已通过
  自动测试和真实数据验证；官方相机位姿文件的 JSON Lines 格式也有回归测试。
- Chrome Screw Scene 4 的5张RealSense图像已转换，共190个实例；完整/可见掩膜、
  GT-mask检测和6D真值结构检查全部通过，注册深度平均有效像素比例为79.84%。
- FoundPose真实图像链路已跑通：36张稀疏模板耗时58.31 s、表示生成5.06 s、
  190个实例推理36.41 s，输出覆盖率100%。中位MSSD/MSPD为39.75 mm/107.83 px，
  最优实例为7.61 mm/23.43 px。该结果只证明适配与推理可运行；模板过稀，不能作为
  最终精度结论，也不包含真实检测网络的误差。
- 定量报告与可视化分别保存在 `outputs/foundpose_robi/robi_validation.json` 和
  `outputs/foundpose_robi/visualizations/`。
