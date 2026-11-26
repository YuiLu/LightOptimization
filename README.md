# 参考图驱动的全场景 3D 灯光生成

Unity 插件用**场景自身渲染的最终 RGB**优化可编辑的 3D 灯具。每组候选灯具都由 Unity URP 和场景原材质渲染；插件以图像误差为反馈，不需要 Shader 梯度、irradiance 目标或外部渲染器。

- **在 Unity 开始使用：** `Lighting MVP > Optimize Lighting (Unity RGB)`。选择相机和实时 Point / Spot Lights，捕获目标，手动或按种子扰动，拟合并复拍。优化与回放不改写场景；显式 Apply 可撤销。
- 通过 `Lighting MVP > Debug` 或 RGB Lighting Optimizer 顶部的 **Debug** 按钮打开调试面板，查看直接入射光和深度，或拖动时间轴回放最近一次 RGB 优化。
- **实验：** [E0–E3：黑盒可行性](docs/blackbox/README.md) · [E4–E6：LM、初始化与 Spot](docs/blackbox-next/README.md) · [E7–E8：原生响应与三灯冷启动](docs/blackbox-structure/README.md) · [E9–E12：联合强度与七灯场景](docs/blackbox-scale/README.md) · [E13–E16：Classroom 失败诊断](docs/blackbox-diagnosis/README.md) · [Unity 安装、使用和批处理](Light/Assets/LightingMvp/README.md)。
- E20 的对照、原始查询和复现命令见 [单灯信号与点光源 MSM](docs/blackbox-e20/README.md) 和 [MBD、直接辐照度与可见性拓扑](docs/blackbox-e20-mbd/README.md)。
- E21 的梯度校准、108 组恢复试次和复跑命令见 [完整灯位差分、联合恢复与阴影代理](docs/blackbox-e21/README.md)。

| 路径 | 内容 |
|---|---|
| `Light/` | Unity 插件源代码及项目文本配置（Unity 2022.3 / URP 14） |
| `mvp/` | 独立的 Mitsuba / PyTorch 研究，以及黑盒结果报告与可选 ZeroGrads 对照脚本；不是插件依赖 |
| `docs/blackbox/`、`docs/blackbox-next/`、`docs/blackbox-structure/`、`docs/blackbox-scale/`、`docs/blackbox-diagnosis/` | 黑盒实验与可复查的数据 |
| `papers/`, `plan.md` | 调研与技术路线 |
| `results/`, `tmp/` | 本地生成结果与临时目录，不入库 |

Git 只包含 Unity 插件源码与配置文本，不包含场景、模型、贴图、材质或 `.meta`。可将 `Light/Assets/LightingMvp` 放入已有的线性 HDR URP 项目使用；Cornell 和 StarRail 批处理复现需要对应的本地项目资产。
