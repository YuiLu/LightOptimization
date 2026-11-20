# 参考图驱动的全场景 3D 灯光生成
## 技术路线、优化器验证与工程平台选型

**技术方案讨论稿 · 2026-11-20**

---

## 1. 项目目标与核心判断

### 1.1 目标定义

输入：
- 一个已有的 3D 场景，包括几何、法线、材质。
- 一张或多张描述目标照明风格的参考图。

输出：
- 一组场景级、可编辑的 3D 灯具配置。
- 灯具具有明确的世界空间位置、朝向、颜色、强度、尺寸和类型。
- 灯具能够在不同相机视角下保持一致，而不是针对某个 Viewport 单独生成。
- 最终效果尽可能接近参考图所表达的照明意图。

这不是严格意义上的原始灯光恢复问题，因为参考图通常来自另一个场景，原始灯具 GT 不存在。

更准确的定义是：

**Reference-conditioned physically realizable 3D lighting synthesis。**

给定参考照明风格和目标 3D 场景，寻找一组紧凑、物理可实现、跨视角一致的 3D 灯具，使其产生尽可能接近目标照明效果的渲染结果。

### 1.2 核心问题

整个问题可以分解为：

1. Reference Image → Target Lighting Appearance
2. Target Lighting Representation → Initial 3D Light Rig
3. Initial Light Rig → Optimized 3D Light Rig
4. Multi-view Validation → Global Lighting Consistency

### 1.3 当前技术判断

不建议一开始就开发端到端灯具预测网络，也不建议一开始就开发完整 Agent 系统。

目前最值得优先验证的是优化器本身。

原因是：无论后续使用 RGBX-Next、图像生成模型、稀疏初始化还是多视角融合，最终都需要一个能够将目标照明场转化为合理 3D 灯具的优化器。

如果这一环节不成立，前面的工作就没有意义。

因此，建议将技术开发顺序调整为：

**先验证 Optimizer → 再验证 Target-to-Rig → 最后验证 Reference-to-Rig。**

---

## 2. 两类核心优化路线

目前最值得比较的是 LightOpt 和 VIALT。

它们都在优化 3D 灯具，但优化对象和物理模型并不完全相同。

### 2.1 路线 A：LightOpt-style Surface Irradiance Optimization

参考论文：

[LightOpt: Lights Optimization for Real-time Rendering, SIGGRAPH 2026](https://mirage-c.github.io/LightOpt/)

核心思想：

在场景表面采样一组点，计算原始灯光配置产生的 Irradiance Field，将其作为优化目标，再调整灯具参数，使新的灯光配置尽可能复现该照明场。

目标函数可以简化为：

$$
\mathcal L_{\mathrm{irr}}=
\sum_j w_j
\left\|
\hat E(x_j;\mathcal L)-E^*(x_j)
\right\|^2
$$

其中：

- $x_j$：场景表面采样点。
- $w_j$：表面面积或其他置信度权重。
- $E^*$：目标 Irradiance。
- $\hat E$：当前灯具配置产生的 Irradiance。
- $\mathcal L$：当前灯具集合。

LightOpt 使用 Unity 风格的 Point Light / Spot Light 照明模型：

$$
E_i(x)=
V_i(x)c_iI_iD_i(x)A_i(\omega_i)
\max(0,n\cdot\omega_i)
$$

它采用解析的距离衰减和角度衰减，并通过梯度优化灯具的位置、颜色、强度和方向。

此外还设计了：

- Overlap Loss：减少过度重叠的灯光影响区域。
- Attraction Loss：避免出现没有灯具覆盖的欠照明区域。
- Light Removal：删除不重要的灯。
- Light Merging：合并照明影响相似的灯。
- Light Addition：根据欠照明区域添加新灯。

**对本项目的价值：**

它的优化目标恰好是我们希望从 RGBX 等模型获得的表面照明场，因此非常容易建立接口。

例如：

$$
E^*_{\mathrm{model}}
\rightarrow
\mathcal L_{\mathrm{irr}}
\rightarrow
\mathcal L_{\mathrm{optimized}}
$$

不需要先构建完整的可微路径追踪器。

但必须注意论文的适用范围：

1. 原始任务是已有灯光集合的压缩和简化，而不是完全从零生成灯具。
2. 主要针对直接照明，作者认为多次反弹 GI 通常由预烘焙处理。
3. 论文中的可微照明模型假设可见性 $V_i=1$，不处理完整的动态遮挡和阴影变化。
4. 主要支持 Point Light 和 Spot Light，没有直接实现 Area Light。
5. 不显式优化视角相关的高光。

因此，LightOpt 是一个很好的初始灯具优化框架，但不能直接等同于完整的物理逆向光照系统。

**重要判断：LightOpt 的核心优化方法可以迁移到本项目，而且不要求使用 Unity。**

不过其官方项目目前没有提供可直接复用的公开实现，建议依据论文公式自行实现，而不是尝试移植 Unity 内部代码。

### 2.2 路线 B：VIALT — View-Independent Adjoint Light Tracing

参考论文：

[View-Independent Adjoint Light Tracing for Lighting Design Optimization, TOG 2024 / SIGGRAPH 2024](https://www.cg.tuwien.ac.at/research/publications/2024/lipp-2024-val/)

代码：

[AdjointLightTracing-backup](https://github.com/david-hahn/AdjointLightTracing-backup)

核心思想：

不通过相机图像定义优化目标，而是在 3D 场景表面构建一个可更新的辐射场表示。

论文采用：

- Mesh Vertex 作为空间离散节点。
- HSH（Hemispherical Harmonics）作为方向表达。
- Light Tracing 计算正向照明。
- Adjoint Light Tracing 计算优化梯度。

这里需要明确一个重要区别：

**VIALT 优化的不是简单的 Diffuse Irradiance，而是表面出射 Radiance Field。**

其目标可以写成：

$$
\mathcal L_{\mathrm{rad}}=
\sum_{k,l,m}
A_k\alpha_k\alpha_l
\left(
L_{klm}-L^*_{klm}
\right)^2
$$

其中 $L_{klm}$ 是每个顶点上的出射辐射场 HSH 系数。

对于纯 Diffuse 材质，可以使用 HSH 的零阶表达，即只保存 RGB：

$$
L_o(x)=\frac{\rho(x)}{\pi}E(x)
$$

因此，后续如果想把 RGBX 输出的 Irradiance 作为 VIALT 的优化目标，需要根据 Albedo 转换为 Diffuse Outgoing Radiance，而不是直接将 Irradiance Buffer 当作 Radiance Buffer。

VIALT 的优势：

1. 不依赖固定相机。
2. 可以优化整个场景的表面照明。
3. 支持间接光传输。
4. 能够处理不同类型灯具的位置、强度和方向优化。
5. 使用专门推导的 Adjoint Gradient，避免简单自动微分在 Light Tracing 中产生的系统性错误。

论文实验使用了两次间接反弹，并展示了 GD、Adam、L-BFGS 等优化器。

但也有一个需要注意的局限：

VIALT 没有显式处理由遮挡变化引起的移动硬阴影不连续性。其梯度主要处理连续部分，不能假设在所有复杂遮挡场景中都能得到无偏梯度。

**对本项目的价值：**

VIALT 更适合验证最终的全场景、View-independent 物理灯光优化能力。

但它比 LightOpt 复杂得多，而且它的目标表示与 RGBX 的 Irradiance 并不相同。

### 2.3 两种方法如何选择？

| 对比 | LightOpt-style | VIALT |
|---|---|---|
| 优化目标 | Surface Irradiance | Outgoing Radiance |
| 相机依赖 | 不依赖相机 | 不依赖相机 |
| 直接光 | 支持 | 支持 |
| 间接光 | 原论文不考虑 | 支持 |
| 方向相关表达 | 不显式处理 | HSH |
| 梯度方法 | 解析灯光模型 | Adjoint Light Tracing |
| 初始灯具 | 已有灯具集合 | 已有灯具集合 |
| 开发复杂度 | 低 | 高 |
| 适合作为第一版 | 是 | 适合作为物理优化器验证 |

我的判断：

- 如果最终目标主要是让 RGBX 预测的 Direct Lighting 对应到 Point/Spot Lights，LightOpt-style 很可能已经足够。
- 如果需要处理多次反弹、复杂几何、全场景辐射场和更完整的物理光传输，VIALT 更有优势。

---

## 3. 第一阶段：先验证优化器，而不是先开发完整系统

这是整个项目的第一个技术里程碑。

### 3.1 MVP 目标

暂时不考虑：

- 参考图。
- 生图模型。
- RGBX。
- 多视角。
- 灯具数量自动预测。
- 灯具自动增加、删除和合并。
- UE 集成。
- 用户绘制 Radiance Target。

只验证一件事情：

**给定一组 GT 灯具和它们产生的 GT 照明场，将灯具参数扰动后，优化器能否将它们恢复到一个照明效果接近 GT 的配置？**

这是一个标准的 Inverse Lighting Recovery Test。

### 3.2 推荐代码基础：VIALT Backup Repository

我建议首先尝试：

[https://github.com/david-hahn/AdjointLightTracing-backup](https://github.com/david-hahn/AdjointLightTracing-backup)

原因不是已经确定最终系统要使用 VIALT，而是这个仓库已经具备：

- Light Tracing 正向模拟。
- Adjoint Gradient。
- Light Parameter Optimization。
- L-BFGS、GD、Adam、CMA-ES 等优化器。
- Python Bindings。
- Radiance Target Buffer。
- 优化参数历史记录。

更重要的是，它已经有非常适合我们做 MVP 的接口：

```python
scene.lightsToParameterVector()
scene.parameterVectorToLights(...)
ialt.forward(...)
ialt.backward()
ialt.currentRadianceAsTarget(...)
ialt.optimize(...)
```

这些接口可以直接支持 GT 灯具恢复测试。

### 3.3 如何构造 GT 恢复实验？

实验流程：

**Step 1：准备简单场景**

建议先使用一个简单的室内场景：

- 一个地面。
- 几面墙。
- 一个简单物体。
- 纯 Diffuse 材质。
- 一盏 Point Light 或 Spot Light。

先不考虑复杂材质、透明物体和多种灯具混合。

**Step 2：生成 GT 灯具参数**

例如：

$$
\mathcal L_{\mathrm{GT}}=
\{p_{\mathrm{GT}},I_{\mathrm{GT}},c_{\mathrm{GT}},d_{\mathrm{GT}}\}
$$

**Step 3：生成 GT Radiance Field**

通过 VIALT 的正向模拟生成当前 Radiance Buffer，然后将它复制为 Target：

```python
gt_params = scene.lightsToParameterVector()

ialt.forward(gt_params)

ialt.currentRadianceAsTarget(
    clearWeights=True
)
```

这里不需要复现用户绘制 Radiance Field 的功能。

直接用已知 GT 灯具生成 Target 即可。

**Step 4：扰动灯具参数**

例如：

$$
p_{\mathrm{init}}=p_{\mathrm{GT}}+\Delta p
$$

$$
I_{\mathrm{init}}=I_{\mathrm{GT}}(1+\epsilon)
$$

第一轮只扰动位置，不同时扰动所有参数。

第二轮再加入强度。

第三轮加入方向和颜色。

**Step 5：优化**

```python
scene.parameterVectorToLights(init_params)

history, final_loss = ialt.optimize(
    ialt.Optimizers.LBFGS,
    stepSize=1.0,
    maxIterations=100
)
```

具体参数需要结合仓库中的优化器实现调整，不应直接假设默认步长适用于所有场景。

**Step 6：评估**

至少记录：

- GT Radiance Field 与优化结果的 L2 Error。
- GT 灯具位置误差。
- 强度误差。
- 每次迭代的 Objective。
- 优化时间。
- 不同初始扰动下的收敛情况。

注意：灯光逆问题可能存在多个等价解，因此 Field Error 是主要指标，参数误差是辅助指标。

### 3.4 实验顺序

| 实验 | 灯具 | 优化参数 | 目的 |
|---|---|---|---|
| Test 1 | 1 Point Light | Position | 验证位置梯度 |
| Test 2 | 1 Point Light | Position + Intensity | 验证距离与强度耦合 |
| Test 3 | 1 Spot Light | Position + Rotation | 验证方向优化 |
| Test 4 | 2 Point Lights | Position + Intensity | 验证多灯干扰 |
| Test 5 | 2–3 Lights | 多参数 | 验证局部最优与鲁棒性 |

先完成 Test 1、Test 2。

如果 Test 1 就无法稳定收敛，不要继续增加场景复杂度。

### 3.5 必须增加 Gradient Check

除了直接看优化结果，还要检查梯度本身是否正确。

对某个灯具参数 $p$，比较：

$$
g_{\mathrm{analytic}}
=
\frac{\partial \mathcal L}{\partial p}
$$

与：

$$
g_{\mathrm{FD}}=
\frac{\mathcal L(p+h)-\mathcal L(p-h)}{2h}
$$

这一步能够区分：

- 梯度计算错误。
- 优化器步长不合适。
- 目标函数本身不适合。
- 初始位置过远，导致局部最优。

VIALT 仓库本身已经提供梯度可视化和有限差分对比相关功能，可以利用它们进行验证。

### 3.6 第一阶段的通过条件

满足以下条件后，才进入下一阶段：

1. 简单场景中，解析/伴随梯度与有限差分结果基本一致。
2. 单灯位置优化可以稳定降低目标误差。
3. 位置和强度联合优化不会频繁发散。
4. 多组合理扰动下能够得到接近 GT 的照明场。
5. 能够明确观察到优化失败时的原因。

不要求一开始就恢复任意复杂场景中的真实灯具数量。

---

## 4. VIALT 仓库的实际限制

这个仓库值得使用，但需要控制投入。

### 优点

- 有真实的 View-independent Adjoint Light Tracing 实现。
- 使用 Vulkan Ray Tracing。
- 具备独立的 Renderer。
- 有 C++ 和 Python 接口。
- 已实现多种优化器。
- 已经具备 Radiance Buffer 和 Target Buffer 的管理逻辑。

### 风险

1. 仓库被作者标注为 Backup，且不再积极维护。
2. 它使用 Tamashii 自己的渲染框架，并不是 Mitsuba 插件。
3. 依赖 Vulkan Ray Tracing、C++20 和相关第三方库。
4. 当前仓库的 `runPredefinedTestCase` 是空实现，不能期待直接运行完整论文测试。
5. Python 侧的 `setTarget` 主要提供统一颜色设置，不是完整的任意 Per-Vertex Target 导入接口。

不过，GT 恢复实验不需要解决第 5 点，因为可以使用 `currentRadianceAsTarget()` 自动生成目标。

如果后续需要导入 RGBX 的 Irradiance Field，则需要额外增加 Per-Vertex Array Target 的接口。

### 开发建议

不要一开始重构这个仓库。

先原样构建，跑通现有 Python Bindings 和一个单灯恢复测试。

如果构建成本过高，或者在合理的短期尝试后无法运行，再考虑下面的轻量替代方案：

- 自己建立一个 Python + PyTorch 的小型优化模块。
- 实现 LightOpt 的 Point/Spot Light 照明公式。
- 使用 Adam 或 L-BFGS。
- 只针对固定的 Mesh Surface Samples 优化。

不建议为了复现 VIALT，先开发自己的完整 Renderer。

---

## 5. LightOpt-style 优化是否可以直接用于本项目？

可以，而且很可能是更容易落地的一条路线。

假设已经通过 RGBX 获得了目标照明场：

$$
E^*=\{E^*(x_1),...,E^*(x_N)\}
$$

那么可以直接在世界空间表面采样点上优化：

$$
\min_{\mathcal L}
\sum_{j=1}^{N}
A_j
\left\|
\sum_{i=1}^{M}E_i(x_j;\theta_i)
-
E^*(x_j)
\right\|^2
$$

其中 $A_j$ 为表面采样面积。

这并不需要路径追踪。

### 5.1 轻量实现

第一版只实现：

- Point Light。
- Spot Light。
- Position。
- Intensity。
- Color。
- Range。
- Spot Cone。

利用解析光照模型计算每盏灯对表面采样点的贡献。

梯度可以通过：

- PyTorch Autograd。
- 手写解析梯度。
- Eigen + L-BFGS。

实现。

### 5.2 不要一开始就复现 LightOpt 全部 Loss

第一版只保留：

$$
\mathcal L_{\mathrm{field}}
=
\sum_j A_j
\left\|
\hat E(x_j)-E^*(x_j)
\right\|^2
$$

先不要加入：

- Overlap Loss。
- Attraction Loss。
- Light Merging。
- Light Removal。
- Light Addition。

因为这些机制主要用于灯具数量调整和灯具集合重构，并不是验证基础参数优化是否成立的必要条件。

### 5.3 后续如何初始化没有灯具的情况？

当基础参数优化通过后，再加入 Light Addition。

可以从欠照明区域提取候选灯具：

1. 找到目标与当前照明差异较大的表面点。
2. 聚类欠照明点。
3. 计算每个聚类的平均位置、法线和目标颜色。
4. 在聚类附近生成 Point / Spot Light 候选。
5. 用局部搜索确定灯具位置和方向。
6. 加入当前灯具集合。
7. 再进行连续优化。

这与 LightOpt 的 Light Addition 思想一致。

需要强调：这不是简单地把每个欠照明点放一盏灯，而是利用空间聚类得到少量有意义的候选灯具。

### 5.4 Irradiance 目标的一个重要问题

RGBX-Next 的输出中有两个值得关注的照明表示：

- $E$：Diffuse Irradiance，包含多次反弹照明。
- $d$：Albedo-free Direct Lighting，采用简化的 diffuse + glossy 模型，不包含多次反弹照明。

这两个量不能混用。

LightOpt 原论文主要优化 Direct Lighting，而且假设可见性为 1。

因此：

- 对于 LightOpt-style 直接灯具拟合，$d$ 是值得优先测试的输入。
- $E$ 更适合表达整体漫反射照明。
- 如果用 $E$ 直接拟合局部灯具，可能将 GI 错误解释为额外 Point Lights。

此外，$d$ 并不严格等同于纯 Diffuse Irradiance，因为其中包含简化的 Glossy 分量。第一版可以在纯 Diffuse 合成数据上先建立直接光模型，再逐步扩展。

对于完整 $E$，可以将低频环境照明或 GI Residual 作为独立项：

$$
E^*(x)\approx
E_{\mathrm{env}}(x)
+
\sum_i E_i(x;\theta_i)
$$

避免所有间接光都被错误地拟合为局部灯具。

---

## 6. 第二阶段：Target Lighting Field → Initial 3D Light Rig

只有当优化器通过第一阶段后，才开始研究自动初始化。

### 6.1 目标表示

建议保留三种数据：

| 表示 | 作用 |
|---|---|
| Target RGB | 最终视觉效果 |
| Diffuse Irradiance $E$ | 整体漫反射照明 |
| Direct Lighting $d$ | 局部直接光照明线索 |

不要只保留一张 RGB 目标图，也不要只保留 Irradiance。

### 6.2 Surface Sample Construction

在当前场景中构建固定的世界空间表面采样点：

$$
\mathcal S=\{(x_j,n_j,A_j)\}
$$

采样建议：

- 优先使用 Poisson Disk Sampling。
- 保留世界空间位置。
- 保留世界空间法线。
- 保留表面面积权重。
- 记录材质和 Albedo。
- 排除不需要参与灯光优化的特殊材质区域。

### 6.3 Initial Light Rig

初期不要训练 DETR 直接预测灯具集合。

建议先采用过完备候选：

1. 在场景自由空间生成候选灯具位置。
2. 根据墙面、天花板、目标照明分布生成辅助候选。
3. 排除实体内部位置。
4. 生成 Point / Spot Light 候选。
5. 计算每个候选对 Surface Samples 的照明响应。
6. 通过稀疏优化选择少量候选。

例如：

$$
\min_{w\geq 0}
\left\|Aw-E^*\right\|_2^2+
\lambda\|w\|_1
$$

其中 $A$ 是候选灯具的照明响应矩阵。

第一版只优化候选灯具的强度，之后再连续优化位置、方向和尺寸。

### 6.4 这一阶段的验证

比较：

- Random Initialization。
- Grid Initialization。
- Sparse Initialization。
- Learned Initialization（后续再加入）。

重点比较：

- 初始照明误差。
- 后续优化收敛速度。
- 最终灯具数量。
- 最终照明误差。
- 不同场景下的稳定性。

---

## 7. 第三阶段：Reference Image → Target Lighting

前两阶段验证完成之后，再引入生成模型。

### 7.1 Reference-conditioned Target Generation

输入：

- 当前场景 G-buffer。
- 当前场景 Albedo。
- 参考图。

输出：

$$
I_t=F(G_S,A_S,I_{\mathrm{ref}})
$$

目标是生成一张适配当前场景几何和材质的目标打光图。

不要求模型直接输出灯具。

### 7.2 Lighting Extraction

对生成图进行 Intrinsic / Lighting Decomposition：

$$
(E^*,d^*)=F_{\mathrm{inverse}}(I_t,G_S)
$$

第一版使用单视角即可。

优先比较 RGB↔X 和 RGBX-Next 的预测质量。

### 7.3 World-space Projection

通过 Depth、Normal、相机内外参，将图像空间照明数据反投影到世界空间。

将每个像素对应到当前场景的 Mesh Surface Sample。

得到：

$$
E^*_{\mathrm{world}}(x_j)
$$

并记录：

- 预测置信度。
- 可见性。
- 材质。
- 对应 Surface Sample ID。

对于边界、遮挡和低置信度区域，降低优化权重。

这一阶段不需要先做多视角，先验证单视角 Target-to-Rig 是否成立。

---

## 8. 第四阶段：Multi-view Global Lighting

当单视角已经可以生成合理的 3D 灯具后，再引入多视角。

生成多个目标视角：

$$
I_t^{(1)},...,I_t^{(V)}
$$

反投影到同一组 3D Surface Samples。

同一个世界空间表面点，在不同视角下应尽可能具有一致的 Diffuse Irradiance：

$$
E^{(1)}(x)\approx E^{(2)}(x)
$$

然后优化共享的灯具集合：

$$
\mathcal L^*=
\arg\min_{\mathcal L}
\sum_{v=1}^{V}
\mathcal L_{\mathrm{view}}^{(v)}(\mathcal L)
$$

这里每个视角都使用同一组 3D 灯具参数。

留出一个没有参与优化的视角，作为最终泛化验证。

不要一开始就开发完整 Multi-view Generation。可以先用人工指定的三个相机位置测试。

---

## 9. 渲染平台与代码基础

### 9.1 推荐的职责分配

| 平台 | 推荐用途 |
|---|---|
| VIALT Backup | 第一阶段 View-independent Optimizer 验证 |
| PyTorch | LightOpt-style 解析优化原型 |
| Mitsuba 3 | 可微渲染研究、离线参考和对照实验 |
| Unreal Engine | 最终场景接入、灯具生成、真实渲染验证 |

不建议把 NRP 作为整个项目的核心代码基础，因为它的核心设定是固定相机、固定材质的场景级代理。

### 9.2 Mitsuba 验证后能否迁移到 UE？

可以。

但要区分：

**可以迁移的是优化算法和参数化方式，不是直接把 Mitsuba 的 Renderer 代码复制进 UE。**

建议将系统分成三层：

#### Layer A：Lighting Optimization Core

负责：

- 灯具参数。
- 参数约束。
- Objective Function。
- Optimizer。
- 参数更新。
- 优化历史。

这一层尽量不依赖具体 Renderer。

#### Layer B：Lighting Field Backend

提供：

```text
EvaluateLightingField(lightRig, surfaceSamples)
EvaluateGradient(lightRig, surfaceSamples)
```

不同平台可以有不同实现：

- Mitsuba Backend。
- VIALT Backend。
- LightOpt Analytic Backend。
- Unreal Backend。

#### Layer C：Scene Adapter

负责：

- 场景几何导入。
- 坐标转换。
- Surface Samples。
- 材质信息。
- 灯具参数导入导出。

这样可以先在研究平台验证，再把相同的 Objective 和参数系统接入 UE。

### 9.3 UE 中如何获得 Irradiance？

这里要区分几种情况。

**A. G-buffer**

UE 的 G-buffer 可以提供：

- Base Color。
- Normal。
- Roughness。
- Metallic。
- Depth 等。

但它本身不是 Irradiance Buffer。

SceneColor 也不能直接等同于 Irradiance，因为其中可能混合了：

- Direct Diffuse。
- Indirect Diffuse。
- Direct Specular。
- Indirect Specular。
- 曝光和后处理影响。

因此不能简单读取 SceneColor 再除以 Albedo，就认为获得了准确 Irradiance。

**B. Lightmap**

UE 的静态 Lightmap 数据可以通过引擎内部的 Lightmap Interaction 和纹理资源访问。

例如：

`FLightMapInteraction`

它提供 Lightmap Texture、Coefficient Scale/Add 等信息。

但这只适合已有静态烘焙数据的场景，不能直接代表任意动态灯光产生的完整 Irradiance Field。

**C. Lumen Surface Cache**

Lumen 会为场景表面建立 Surface Cache，并计算 Direct / Indirect Lighting。

理论上可以研究从引擎内部 Surface Cache 中提取数据。

但它不是一个可以直接通过普通 G-buffer API 获取的稳定、统一 Irradiance Texture。

需要结合 UE 源码修改 Renderer 或增加专门的 RDG Pass。

不建议把这项工作放在第一阶段。

**D. Path Tracer**

UE Path Tracer 支持输出不同 Lighting Components：

- Diffuse。
- Indirect Diffuse。
- Specular。
- Indirect Specular。

可以通过 Movie Render Queue 做离线对照和 GT 验证。

如果结合 Depth、Normal 和相机参数，可以把可见像素重新映射到世界空间表面。

但它仍然是相机可见区域的输出，不是完整的 Camera-free Surface Irradiance Field。

**E. 自定义 Surface Probe / Ray Tracing Pass**

如果希望获得完整世界空间照明场，可以在 UE 中建立 Surface Samples：

$$
\mathcal S=\{(x_j,n_j)\}
$$

对每个点进行直接照明计算，必要时通过 Ray Tracing 计算可见性和间接照明。

这是最终实现 View-independent Lighting Field 的一种可行方式。

### 9.4 我更推荐 UE 第一版怎么做？

如果使用 LightOpt-style 直接照明优化，其实不需要先从 UE 提取 Irradiance。

可以直接：

1. 在 UE 场景中生成 Surface Samples。
2. 从外部导入模型预测的目标 Irradiance / Direct Lighting。
3. 在 UE 中根据候选灯具参数计算每个 Surface Sample 的照明贡献。
4. 计算目标场和当前场的误差。
5. 更新灯具参数。

即：

$$
E^*_{\mathrm{external}}
\quad\text{vs.}\quad
E_{\mathrm{UE}}(\mathcal L)
$$

这样就不需要先实现复杂的 UE Irradiance Extraction。

初期可以假设可见性为 1，后续再增加 Ray Tracing Visibility。

### 9.5 LightOpt 能否从 Unity 迁移到 UE？

可以，而且并不需要迁移 Unity 本身。

LightOpt 的核心优化方法是：

- Surface Sampling。
- Light Attenuation Model。
- Irradiance Error。
- Gradient-based Parameter Optimization。
- Light Set Restructuring。

这些都可以在 UE C++ 和 GPU Compute Shader 中重新实现。

但需要注意 Unity 与 UE 的灯光模型存在差异。

特别是：

- Light Intensity Units。
- Distance Attenuation。
- Light Range。
- Spot Cone Attenuation。
- Source Radius。
- Shadow Visibility。

因此不能直接照搬 Unity 的参数数值。

正确做法是：

**使用 UE 实际灯具的光照模型重新实现 LightOpt-style 优化器。**

这样优化结果才能与 UE 最终渲染保持一致。

---

## 10. 最终推荐的开发里程碑

整个项目不应该一次性启动，而应该有明确的阶段门槛。

| 阶段 | 任务 | 通过后再做什么 |
|---|---|---|
| M0 | VIALT 单灯 GT Recovery Test | 确认优化器能力 |
| M1 | LightOpt-style Irradiance Fitting | 比较直接照明优化路线 |
| M2 | Target Field → 3D Surface Samples | 验证目标表示接口 |
| M3 | Sparse Light Initialization | 验证从目标场生成初始灯具 |
| M4 | Reference → Generated Target | 接入生成模型 |
| M5 | Single-view to Multi-view | 验证全场景一致性 |
| M6 | UE Backend | 接入实际生产引擎 |

### M0：当前唯一应该启动的任务

建议首先只做：

**VIALT GT Light Recovery Test。**

它不需要任何 AI 模型，也不需要 UE 集成。

如果成功，就说明一个完整的 View-independent 3D 灯具优化器至少在简单场景中具备可行性。

如果失败，则根据梯度检查和收敛情况判断问题：

- VIALT 实现或环境问题。
- Adjoint Gradient 问题。
- Light Parameterization 问题。
- Objective Function 问题。
- 局部最优问题。

不要直接将失败归因于整个 3D Lighting Optimization 方向。

如果 VIALT 的构建或使用成本过高，则转为 M1 的轻量 PyTorch 解析优化器，而不是立即尝试在 UE 中实现完整可微渲染。

---

## 11. 重要参考论文与代码

### A. 优化器与 3D 灯光配置

**1. LightOpt: Lights Optimization for Real-time Rendering**

SIGGRAPH 2026

https://mirage-c.github.io/LightOpt/

价值：
- Irradiance Field Optimization。
- Overlap / Attraction Loss。
- Light Addition / Removal / Merging。
- 适合 Direct Lighting 的灯具优化和集合重构。

建议：重点阅读 Method 部分，作为 LightOpt-style 优化器的实现依据。

**2. View-Independent Adjoint Light Tracing for Lighting Design Optimization**

TOG 2024 / SIGGRAPH 2024

https://www.cg.tuwien.ac.at/research/publications/2024/lipp-2024-val/

代码：

https://github.com/david-hahn/AdjointLightTracing-backup

价值：
- View-independent Lighting Optimization。
- HSH Radiance Field。
- Adjoint Light Tracing。
- GPU Gradient。
- 多次间接光传输。

建议：作为 M0 的首选代码参考。

**3. Neural Render Proxies for Interactive and Differentiable Lighting**

EGSR 2026

https://studios.disneyresearch.com/2026/07/01/neural-render-proxies-for-interactive-and-differentiable-lighting/

价值：
- Light-agnostic Path Cache。
- Scene-specific Neural Proxy。
- Differentiable Image-space Lighting。
- 约 30–60 Hz 交互式 Relighting。

局限：
- 固定相机和材质。
- 更适合离线场景、影视渲染提效。
- 不适合直接作为全场景 View-independent 优化器。

代码参考：
- https://github.com/bgyss/nrp
- https://github.com/bgyss/Blender-NRP

注意：上述是第三方实现，不是 Disney 官方代码。

### B. 目标照明表示与生成

**4. RGB↔X: Image Decomposition and Synthesis Using Material- and Lighting-aware Diffusion Models**

SIGGRAPH 2024

https://research.adobe.com/publication/rgb%E2%86%94x-image-decomposition-and-synthesis-using-material-and-lighting-aware-diffusion-models/

价值：
- RGB → Intrinsic / Lighting。
- X → RGB。
- 将材质和光照作为显式条件。
- 可作为目标照明分解和生成式目标渲染的基础。

**5. RGBX-Next: Towards Realistic Generative Rendering from G-Buffers**

arXiv 2026

https://arxiv.org/abs/2608.13929

价值：
- 扩展 RGB↔X 到图像、视频和流式生成。
- RGB → G-buffer。
- G-buffer → RGB。
- 包含 Diffuse Irradiance $E$ 和 Albedo-free Direct Lighting $d$。
- Direct Lighting 不包含 Multi-bounce，适合研究直接灯光控制。

建议：重点研究 $E$ 和 $d$ 的物理定义及其在灯具拟合中的作用。论文提出公开模型，但应以实际 release 情况为准。

### C. 初始灯具估计

**6. Deep Parametric Indoor Lighting Estimation**

ICCV 2019

https://lvsn.github.io/deepparametric/

价值：
- 单图预测参数化 3D 灯具。
- 可作为 Learned Initialization Baseline。
- 适合比较 Sparse Initialization 是否比直接网络回归更有效。

### D. 可微渲染与工程平台

**7. Mitsuba 3**

https://github.com/mitsuba-renderer/mitsuba3

https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/gradient_based_opt.html

价值：
- PRB / Radiative Backpropagation。
- Scene Parameter Optimization。
- 研究级可微渲染。
- 可作为离线 Forward / Inverse Rendering 对照平台。

**8. Unreal Engine Lumen Technical Details**

https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine

价值：
- Surface Cache。
- Direct / Indirect Lighting。
- World Space Radiance Cache。
- 研究 UE 中世界空间照明数据的获取方式。

**9. Unreal Engine Path Tracer**

https://dev.epicgames.com/documentation/en-us/unreal-engine/path-tracer-in-unreal-engine

价值：
- 离线高质量渲染。
- Lighting Components 分离。
- GT / 参考照明验证。

---

## 12. 最终建议

我不建议现在就确定整个系统一定采用 VIALT，也不建议直接认定 LightOpt 已经足够。

我建议当前采取一个明确的决策顺序：

**第一优先级：验证 VIALT 的 GT Light Recovery。**

利用现有代码完成单灯和双灯的参数恢复，确认 View-independent Adjoint Optimizer 是否能够稳定工作。

**第二优先级：验证 LightOpt-style Irradiance Optimization。**

它与 RGBX 的输出接口更直接，而且更容易迁移到 UE。

比较两种方法在相同合成场景下的：

- 照明场误差。
- 收敛速度。
- 灯具参数恢复能力。
- 遮挡情况下的表现。
- 多灯情况下的稳定性。

**第三优先级：再接入 RGBX-Next。**

只有当目标照明场可以被稳定优化之后，才值得投入生成式目标图和 Irradiance Extraction。

**第四优先级：最后做 UE 集成。**

不要把 Mitsuba、VIALT 或 LightOpt 绑定为最终生产平台。将优化核心与 Renderer Backend 分离，就可以在研究平台验证后迁移到 UE。

最终可能形成的混合架构是：

```text
Reference Image
       |
       v
Generative Target Rendering
       |
       v
RGBX / Lighting Decomposition
       |
       +------ Diffuse Irradiance
       |
       +------ Direct Lighting
       |
       v
World-space Surface Lighting Field
       |
       v
LightOpt-style Sparse Initialization
       |
       v
3D Light Rig
       |
       v
VIALT-style Physical Refinement
       |
       v
Unreal Engine Validation
       |
       v
Final Global Lighting Rig
```

但这只是通过前述实验后的候选架构，不是当前就要完整开发的任务。

**当前真正需要投入的开发任务只有一个：把 VIALT 的单灯 GT 恢复实验跑通，并确认它的梯度与优化能力。**

这一步的结果将决定后续究竟应该以 LightOpt-style 解析照明优化为主，还是需要 VIALT 这种完整的 Adjoint Light Tracing 优化器。