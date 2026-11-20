# 参考图驱动的全场景 3D 灯光生成

本仓库使用**单 Repo、两条开发线路**：研究代码与应用代码共享技术目标、实验结论和文档，但保持实现及发布节奏独立。

## 目录边界

| 目录 | 职责 | 当前定位 |
|---|---|---|
| `mvp/` | Mitsuba / PyTorch 优化器验证、解析光照、梯度与遮挡实验 | 研究代码 |
| `Light/` | Unity 场景适配、Compute Shader、C# 优化器与 Lighting Debug | 应用开发 |
| `papers/` | 论文与调研材料 | 文档 |
| `third_party/` | 外部代码与资产，不直接纳入主仓库 | 外部依赖 |
| `results/` | 实验报告、图像、CSV、优化历史 | 生成结果，不入库 |
| `tmp/` | 临时文件 | 不入库 |
| `plan.md` | 技术路线与阶段目标 | 共享设计文档 |

## 技术边界

两条线路采用相同的分层思想：

```text
Lighting Optimization Core
        +
Lighting Field Backend
        +
Scene Adapter
```

Mitsuba / PyTorch 用于快速验证目标函数、参数化、梯度与优化策略；Unity 用于场景接入、世界空间表面光照计算、可编辑灯具和后续系统应用开发。整个技术路线不要求 URP 或最终渲染管线可微。

## Unity 入口

Unity 工程位于 `Light/`。主要菜单：

- `Lighting MVP > Run Cornell Box Experiments`
- `Lighting MVP > Debug Window`

详细说明见 `Light/Assets/LightingMvp/README.md`。

## 提交策略

保持一个 Git 仓库，提交按变更目的拆分，而不是按目录机械混合：

1. `research: validate lighting optimizer with Mitsuba` — Python 优化器、实验及研究结论。
2. `feat: add Unity lighting optimization debug tools` — Unity 运行时、编辑器工具、Shader 和实验场景。
3. `chore: document repository structure` — 根目录文档与忽略规则（可并入任一提交）。

后续同时修改两条线路时，如果共享了参数定义、实验报告格式或技术约定，可以使用一个跨目录提交；否则分别提交，保持历史易读、易回滚。

不要提交 `Library`、临时验证项目、生成图片、CSV 和优化历史。需要长期保留的实验基准应提炼为小型、明确版本的测试数据，再放入专门的 fixtures 目录。

## 本次源码入库范围

Unity 只提交 `LightingMvp` 下的 C#、Compute Shader、Shader 和说明文档，以及 `Packages` / `ProjectSettings` 文本配置。场景、模型、贴图、材质、`.unity`、`.asset`、`.meta` 和其他 Unity 资产均不提交。

