# CAD 位姿修正与机器人约束链

## 作用与边界

本模块在现有 FoundPose、Top-K 门控和 CAD 抓取排序之上增加三个独立环节，不修改
FoundPose 主体网络，也不改变已有目录职责：

```text
FoundPose/门控位姿 T_camera_object
        │
        ▼
CAD 轮廓与 RGB-D 深度反馈修正
        │  T_base_object = T_base_camera × T_camera_object
        ▼
机器人基坐标系物体位姿
        │  T_base_grasp = T_base_object × T_object_grasp
        ▼
UR5e 数值 IK + 关节限位 + MuJoCo 路径碰撞检查
```

对应代码为：

- `src/lzt_cad_grasp_core/pose_refinement.py`：CAD/RGB-D 位姿修正核心；
- `scripts/refine_foundpose_poses.py`：批量读取 FoundPose 记录、RGB-D 和 mask；
- `src/lzt_cad_grasp_core/robot_frames.py`：坐标系契约与外参加载；
- `scripts/transform_pose_to_robot.py`：相机系物体位姿转换为机器人基坐标系；
- `src/lzt_cad_grasp_core/robot_constraints.py`：UR5e Jacobian IK 和路径碰撞检查；
- `scripts/rank_task_grasps.py`：把外参、IK 和碰撞结果接入原有 Top-K 抓取排序。

## CAD 轮廓与深度反馈修正

修正器把 CAD 点采样投影到图像，计算轮廓 IoU、轮廓距离、深度中位绝对误差和
深度内点率，并加入相对初始位姿的正则项。它在有限 SE(3) 邻域内做由粗到细的
确定性坐标搜索，默认限制平移不超过 25 mm、旋转不超过 5°。GT 位姿和 MSSD/MSPD
不参与优化，只能在优化完成后用于评价。

对象 2 的单图、3 候选 smoke test 中，修正目标函数均下降，MSSD 从
7.20/13.59/15.00 mm 降到 4.35/3.61/2.69 mm；MSPD 从
3.71/3.27/1.56 px 变为 4.87/4.86/2.28 px。该结果证明局部三维修正有效，也说明
二维投影指标仍有权衡；在对象 2–4 全量官方评价完成前，不能声称整体 BOP AR 已提升。

复现单图检查：

```bash
cd /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros
python3 scripts/refine_foundpose_poses.py \
  --input outputs/pose_gating/tless_000002/gated-poses.json \
  --object-config config/objects/tless_000002.yaml \
  --detections outputs/mask_refinement/tless_objects2_4_guarded/detections_rgbd_refined.json \
  --output-dir outputs/pose_refinement/tless_000002_smoke \
  --top-k 3 --limit-groups 1 --render-point-count 5000 --preview-limit 3
```

主要输出为 `refined-poses.json`、`audit.json`、`summary.json` 和 `previews/`。
预览中蓝线是观测 mask，红线是初始 CAD 轮廓，绿线是修正后轮廓。

## 相机坐标系到机器人基坐标系

系统统一使用列向量和 OpenCV 光学坐标约定：相机 X 向右、Y 向下、Z 向前。
转换公式固定为：

```text
T_base_object = T_base_camera_cv @ T_camera_object
```

仿真场景的精确外参保存在
`config/calibration/mujoco_vision_camera_to_ur5e_base.json`，自动测试会将该文件与
实时加载的 MuJoCo 场景矩阵逐元素比较。转换命令为：

```bash
python3 scripts/transform_pose_to_robot.py \
  --input <与该相机对应的修正位姿.json> \
  --calibration <该相机的机器人外参.json> \
  --output outputs/robot_poses/result.json
```

重要限制：T-LESS 图像并不是 MuJoCo 相机拍摄的，不能把 T-LESS 位姿与上述仿真
外参组合后用于真实执行。实机必须先通过手眼标定得到实际
`T_base_camera_cv`，并保持时间同步、单位和相机光学坐标约定一致。

## UR5e IK 与碰撞检查

IK 使用 MuJoCo 中实际 UR5e 六关节链和夹爪 `pinch` site 的位置/旋转 Jacobian，
通过阻尼最小二乘迭代求解，并逐步限制关节范围。碰撞检查对当前位姿、预抓取、
抓取、搬运到放置预备位姿和最终放置位姿之间的关节路径插值，读取每个采样点的
MuJoCo 接触对；结果包含可达性、残差、关节余量、六关节角和碰撞对。

接入原抓取排序的命令示例：

```bash
export BOP_PATH=/home/lzt/lzt_ros2_ws/datasets/bop
python3 scripts/rank_task_grasps.py \
  --pose-hypotheses <与仿真相机对应的Top-K位姿.json> \
  --grasp-candidates <CAD抓取候选.json> \
  --object-config config/objects/tless_000002.yaml \
  --goal-transform-json <机器人基坐标系放料目标.json> \
  --robot-camera-calibration config/calibration/mujoco_vision_camera_to_ur5e_base.json \
  --use-mujoco-constraints --path-samples 25 \
  --output outputs/task_grasps/robot_checked.json
```

输出中的 `transform_base_grasp_pick_best`、`transform_base_grasp_place`、
`pick_joint_positions_rad`、`place_joint_positions_rad` 和 `collision_pairs` 可直接用于
审计。当前场景中的 `target` 是占位物体，路径检查默认忽略它与机器人的接触；CAD
候选生成器已经单独检查夹指与目标 CAD 的碰撞，但正式执行前仍需把目标 CAD 作为
场景碰撞体接入，才能覆盖机械臂其他连杆与目标物体的碰撞。

## 当前验证

- Python 测试共 49 项通过；
- 仿真相机到 UR5e 基座的矩阵复合误差小于 `1e-12`；
- UR5e site 上移 30 mm 的 IK 在 3 次迭代内收敛，位置残差约 `2.22e-5 m`；
- 端到端 smoke test 返回 1 个可行候选、真实六关节解和空碰撞对；
- 人工构造的机器人—墙体穿透会被路径碰撞测试检出。

上述验证完成了“位姿修正—基坐标变换—IK/碰撞约束”的软件链路；尚未完成的部分
是对象 2–4 全量修正评价、目标 CAD 动态导入场景、完整抓取动力学执行和实机手眼
标定。
