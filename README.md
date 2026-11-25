# 参考图驱动的全场景 3D 灯光生成

Unity 插件用**场景自身渲染的最终 RGB**优化可编辑的 3D 灯具。每组候选灯具都由 Unity URP 和场景原材质渲染；插件以图像误差为反馈，不需要 Shader 梯度、irradiance 目标或外部渲染器。

- **在 Unity 开始使用：** `Lighting MVP > Optimize Lighting (Unity RGB)`。选择相机和实时 Point / Spot Lights，捕获目标，手动或按种子扰动，拟合并复拍。优化与回放不改写场景；显式 Apply 可撤销。
- **调试工具：** `Lighting MVP > Debug > Native Irradiance Preview` 用于观察直接入射光和深度；`Legacy field and replay` 可查看历史 Cornell 光场与回放。二者不进入 RGB loss。
- **实验：** [E0–E3：黑盒可行性](docs/blackbox/README.md) · [E4–E6：LM、初始化与 Spot](docs/blackbox-next/README.md) · [Unity 安装、使用和批处理](Light/Assets/LightingMvp/README.md)。

| 路径 | 内容 |
|---|---|
| `Light/` | Unity 插件源代码及项目文本配置（Unity 2022.3 / URP 14） |
| `mvp/` | 独立的 Mitsuba / PyTorch 研究，以及黑盒结果报告与可选 ZeroGrads 对照脚本；不是插件依赖 |
| `docs/blackbox/`、`docs/blackbox-next/` | 黑盒实验与可复查的数据 |
| `papers/`, `plan.md` | 调研与技术路线 |
| `results/`, `tmp/` | 本地生成结果与临时目录，不入库 |

Git 只包含 Unity 插件源码与配置文本，不包含场景、模型、贴图、材质或 `.meta`。可将 `Light/Assets/LightingMvp` 放入已有的线性 HDR URP 项目使用；Cornell 和 StarRail 批处理复现需要对应的本地项目资产。
