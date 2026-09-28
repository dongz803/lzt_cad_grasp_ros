# Object assets

将具有真实尺寸的铜料 CAD/mesh 放在此目录。

推荐同时保存：

- `visual.obj` 或 `visual.ply`：高精度渲染模型；
- `collision.stl`：简化、封闭、适合碰撞计算的模型；
- `object.json`：单位、物体坐标系、对称轴和真实尺寸说明。

MuJoCo 工程内部统一使用米。若外部位姿算法要求毫米，应只在适配器边界转换，不要改变物理场景的单位。

