# 外部方法源码

FoundPose 官方源码计划放在 `third_party/foundpose/`，该目录不纳入本仓库版本控制：

```bash
git clone --filter=blob:none --recurse-submodules \
  https://github.com/facebookresearch/foundpose.git \
  /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros/third_party/foundpose
git -C /home/lzt/lzt_ros2_ws/src/lzt_cad_grasp_ros/third_party/foundpose \
  rev-parse HEAD
```

克隆完成后必须把实际提交号记录进实验记录。ROS 包不直接修改第三方源码；T-LESS
配置和结果保存在本仓库自己的 `config/`、`outputs/` 中。
