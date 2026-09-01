# Tracer MuJoCo model

This model is derived from the local ROS package at
`/home/lzt/robot_driver_ws/src/tracer/tracer_ros/tracer_description`.

The original Xacro applies `rpy="1.57 0 0"` to a Z-up chassis mesh. That
rotation is intentionally removed here because it makes the base stand on its
side in MuJoCo. The main wheels keep hinge joints; the chassis is fixed for the
first vision/manipulation milestone. Four castors and the UR5e mounting hardware
are currently visual-only.

The converted wheel meshes also use different native axes from their URDF
joint axes. A visual-only 90-degree X rotation is applied to both drive-wheel
meshes and all four castor-wheel meshes, so the tyres stand vertically without
changing the physical hinge axes.
