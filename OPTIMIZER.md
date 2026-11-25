# 当前灯光优化算法原理

更新时间：2026-11-23
对应实现：`mvp/lighting.py`、`mvp/radiosity.py`、`mvp/radiosity_experiments.py`

---

## 1. 一句话概括

当前系统是一个**确定性的、世界空间表面照明场优化器**：

> 把场景表面切成许多小三角形，计算每盏灯的直接照明，再通过固定的表面传输矩阵计算多次反弹，最后调整灯具的位置、强度和色温，使总照明场接近 GT。

它不是图像生成，也不是用 Path Tracing 反向传播。Path Tracing 只用于优化后的独立渲染验证。

当前阶段固定灯数和灯具类型，不负责候选灯生成、自动初始化或灯具数量预测。

---

## 2. 场景表面怎么变成优化目标

### 2.1 表面离散：Radiosity patch，不是屏幕片元

将场景几何划分为 $N$ 个非重叠的表面 patches：

$$
\mathcal{S}=\{(x_j,\,n_j,\,A_j,\,\rho_j)\}_{j=1}^{N}
$$

- $x_j$：patch 中心位置（世界空间）
- $n_j$：接收面法线
- $A_j$：patch 面积，同时是优化时的权重
- $\rho_j$：Diffuse albedo，RGB 三通道分别保存

**这里的 patch 不是屏幕上的 fragment。** 两者的区别：

| | Radiosity patch | 屏幕 fragment |
|---|---|---|
| 空间 | 世界空间 | 屏幕像素空间 |
| 作用 | 承担面积和能量交换 | 着色输出的最小单元 |
| 与相机关系 | 无关（view-independent） | 绑定当前视角 |
| 数量来源 | 几何细分 | 渲染分辨率 |

### 2.2 三角形怎么划分

使用递归四等分：连接原三角形三条边的中点，得到 4 个小三角形；再对每个小三角形做一次，得到 16 个。

```
原三角形（34 个）
    ↓ 连接三条边中点
4 个子三角形
    ↓ 再做一次
16 个子三角形
    ↓
34 × 16 = 544 patches
```

每个 patch 都有真实顶点、世界空间中心、法线、面积和 albedo。它们互不重叠，面积之和等于原三角形面积。

细分的目的不是让画面更清晰，而是让 Radiosity 的能量计算更准确——特别是近场 form factor 的积分精度。

---

## 3. 直接光：每盏灯怎么照亮一个 patch

### 3.1 单盏灯的贡献

第 $\ell$ 盏灯对第 $j$ 个 patch 的直接 irradiance：

$$
E_{\ell,j}
=
V_{\ell,j}\,
c_\ell\, I_\ell
\cdot
\underbrace{\frac{a_\ell}{a_\ell+\|p_\ell-x_j\|^2}}_{\text{距离衰减}}
\cdot
\underbrace{A_\ell(\omega_{\ell,j})}_{\text{spot 角度项，point 为 1}}
\cdot
\underbrace{\max\bigl(0,\;n_j\cdot\omega_{\ell,j}\bigr)}_{\text{表面朝向}}
$$

- $p_\ell$：灯的世界空间位置
- $I_\ell$：灯的强度
- $c_\ell$：RGB 色度，也可由 CCT 色温计算
- $a_\ell$：attenuation smoothness
- $V_{\ell,j}$：灯到 patch 是否被遮挡，由 Mitsuba BVH 计算

实际实现还包含 range window，使灯光超出 range 后平滑衰减到零。

### 3.2 多灯求和

每个 patch 的直接 irradiance 是全部灯具贡献之和：

$$
E_j^{\text{direct}}=\sum_{\ell=1}^{M}E_{\ell,j}
$$

输出形状统一为 `[N,3]`。多灯优化不是分别拟合三张图，而是让**同一个总照明场**接近 GT。

### 3.3 硬遮挡：什么叫"可见性固定"和"跨阴影边界"

每次 forward 都根据当前灯的位置重新计算 $V_{\ell,j}$，但可见性 mask 是 detached 的——它参与前向计算，但不参与求导。

**"可见性固定"** 更准确的说法是：在某个参数邻域内，可见性拓扑不变。假设灯的位置是 $p$，检查附近两个点 $p-h$ 和 $p+h$：

$$
V(p-h)=V(p)=V(p+h)
$$

即扰动前后，每盏灯照到哪些 patch、被哪些物体挡住，完全没有变化。例如灯向左移 1 mm，方块仍遮住同样的地板；向右移 1 mm 也一样。这时距离、方向余弦等连续项正常变化，autograd 梯度与有限差分梯度一致。

**"跨阴影边界"** 是指灯继续移动后，某条"灯 → patch"的光线从没有碰到遮挡物变成碰到（或反之）：

$$
V_{\ell,j}:\;0\leftrightarrow1
$$

它不要求灯穿过墙面。灯可能只是水平移动了一点，但阴影边缘扫过了一块地板。边界就是这些"光线刚好擦过遮挡物轮廓"的位置。

当前 mask 是硬 0/1，因此跨边界时照明跳变，普通 autograd 不会自动给出正确的边界梯度。这就是当前方法在跨阴影初始化时可能提前停止、进入退化状态或收敛到其他局部解的根本原因。

---

## 4. 从直接光到 Diffuse Radiosity

直接 irradiance 先转换成直接出射 radiosity：

$$
D_j=\frac{\rho_j}{\pi}\,E_j^{\text{direct}}
$$

**为什么不能直接把 irradiance 当 radiosity？**

Irradiance 描述"来到表面的光"，radiosity 描述"表面反射出去的光"。对于纯 Diffuse 材质，两者通过 $\rho/\pi$ 联系。

### 当前这一步是写死的 Lambertian 假设

公式 $D=\rho\,E/\pi$ 成立的前提是表面向所有方向均匀反射，即理想 Diffuse BRDF 为 $f_r=\rho/\pi$，出射 radiance 与观察方向无关。

### 非 Diffuse 材质不能这样直接代入

一般材质必须回到完整渲染方程：

$$
L_o(x,\omega_o)
=
L_e(x,\omega_o)
+
\int_{\Omega}
f_r(x,\omega_i,\omega_o)\,
L_i(x,\omega_i)\,
(n\cdot\omega_i)\,
d\omega_i
$$

这时出射光与观察方向 $\omega_o$ 有关，高光随视角变化，不能只在每个 patch 存一个 RGB radiosity，也不能只用 albedo 乘一个固定矩阵。

### 游戏项目里的自定义材质更不能自动映射

游戏材质可能包含 metallic/roughness、clearcoat、anisotropy、sheen、subsurface scattering、自定义节点和 shader 分支、纹理法线贴图、非物理 artistic shading 等。仅知道 baseColor 并不能推出正确的 $D_j$，更不能推出通用 GI。

| 材质情况 | 可行方式 |
|---|---|
| 纯 Diffuse | 当前固定 Radiosity 矩阵 |
| Diffuse + 少量附加 lobe | Diffuse GI 用 Radiosity，其他 lobe 单独做方向相关计算 |
| 一般 glossy | HSH/方向 bin 的传输算子，或 VIALT/Mitsuba 等方法 |
| 任意自定义 shader | 必须提供 shader 的 BRDF 求值器，否则无法建立可靠的物理优化 |
| 非物理卡通 / artist shading | 可能只能自定义目标场和近似优化，不能声称物理 GI |

如果要支持真正的自定义游戏材质，自然的接口不是再传一个 baseColor，而是：

```
EvaluateBRDF(position, normal, ωi, ωo, materialParameters)
```

然后构建方向相关的 $L(x,\omega_o)$ 和方向相关的传输算子。

**当前 Radiosity 只适合固定 Diffuse 的 MVP，不是通用游戏材质方案。**

---

## 5. 固定矩阵计算 GI

### 5.1 表面之间的传输

用 form-factor 矩阵 $F$ 描述 patch 之间的几何传输和遮挡关系。直接出射 radiosity 经过反复反射后形成总 radiosity：

$$
B=D+\operatorname{diag}(\rho)\,F\,B
$$

整理得到：

$$
\boxed{
B=\bigl(I-\operatorname{diag}(\rho)\,F\bigr)^{-1}D
}
$$

- $D$：直接光
- $F$：固定几何下 patch 之间的传输
- $B$：包含多次反弹后的总 radiosity
- $\operatorname{diag}(\rho)$：接收 patch 的 albedo

RGB 三通道分别求解（详见附录 A 和 B 关于迭代解与直接解的关系）：

$$
B_c=\bigl(I-\operatorname{diag}(\rho_c)\,F\bigr)^{-1}D_c,
\qquad c\in\{R,G,B\}
$$

### 5.2 为什么可以重复使用 LU？

几何和材质不变，因此以下矩阵固定：

$$
A_c=I-\operatorname{diag}(\rho_c)\,F
$$

只需预分解一次，每次灯具变化时只更新直接光 $D_c$，再求解 $A_c\,B_c=D_c$。这使优化过程不需要重新计算 GI 的几何结构，也不需要 PT 采样。原理见附录 B。

---

## 6. 优化目标与梯度

### 6.1 目标函数

当前使用面积加权场误差：

$$
\mathcal{L}(\theta)
=
\frac{\displaystyle\sum_j A_j\,\|B_j(\theta)-B_j^{\text{GT}}\|^2}
{\displaystyle 3\sum_j A_j}
$$

实际优化时还会按 GT 场尺度归一化，主要评估指标为 field NRMSE：

$$
\operatorname{NRMSE}
=
\sqrt{
\frac{\displaystyle\sum_j A_j\,\|B_j-B_j^{\text{GT}}\|^2}
{\displaystyle\sum_j A_j\,\|B_j^{\text{GT}}\|^2}
}
$$

**场误差是主指标，灯具参数误差是辅助指标。**

### 6.2 梯度链路

$$
\theta
\;\longrightarrow\;
E^{\text{direct}}(\theta)
\;\longrightarrow\;
D(\theta)
\;\longrightarrow\;
B(\theta)
\;\longrightarrow\;
\mathcal{L}
$$

PyTorch 对解析直接光公式和固定线性系统执行自动微分。固定矩阵时，通过线性系统的反向传播等价于：

$$
\frac{\partial\mathcal{L}}{\partial D_c}
=
\bigl(I-\operatorname{diag}(\rho_c)\,F\bigr)^{-\!T}
\frac{\partial\mathcal{L}}{\partial B_c}
$$

因此：

- 不需要 PT 梯度
- 不需要对 form-factor 矩阵求导
- 直接光参数仍然可以连续优化
- 阴影边界的硬 mask 仍是单独的限制

---

## 7. 为什么要用 Adam、L-BFGS 和混合策略？

这不是理论上的必需组合，而是实验上有效的 warm start 策略。

**Adam** 更像"稳妥地开车接近目的地"：每个参数有自适应步长，对位置、强度、CCT 尺度不同的问题比较友好，初期通常稳定，但接近极值时可能振荡、收敛慢。

**L-BFGS** 更像"到附近后精确停车"：利用历史梯度近似曲率，对当前这种确定性、平滑区域占主导的目标很有效，通常比 Adam 少很多次迭代。配合 strong-Wolfe 线搜索可以快速精修，但遇到阴影跳变时可能提前停止。

为什么需要组合？因为当前目标同时有多灯耦合、阴影不连续、不同参数尺度、多解和排列对称性、局部最优。实验里出现了某些 seed Adam 成功而 L-BFGS 提前停止、另一些 seed 相反的情况。

所以采用：

```
Adam 先进入较好的区域
        ↓
L-BFGS 再做局部精确收敛
```

这不是"越多越好"或"保证正确"，而是用 Adam 的鲁棒性换初始化，再用 L-BFGS 的局部效率。原始 Adam、L-BFGS 结果仍然保留，混合策略没有被证明普遍最优。

---

## 8. 当前灯具参数

| 参数 | 当前状态与含义 |
|---|---|
| 灯位置 $p_\ell$ | 已优化，每灯 3 个自由度 |
| 强度 $I_\ell$ | 已优化，使用 log-intensity 保证正值 |
| RGB color | API 支持 linear；也支持 normalized RGB，避免负色和整体尺度简并 |
| CCT 色温 | 已优化，使用 log-Kelvin，经近似 CCT→RGB 得到色度 |
| Spot direction | API 支持；当前 point-light 实验未启用 |
| Range | 固定为 100 |
| Attenuation smoothness | 固定为 $10^{-4}$ |
| Spot inner/outer cone | 固定，仅 spot 有意义 |
| 灯数与灯类型 | 当前固定；三灯实验为 3 个 point lights |
| 遮挡 mask | 每次 forward 重算，但不参与求导 |
| Form factor / albedo | 固定，仅对 direct RHS 求梯度 |

### 8.1 CCT 与 RGB 不是一回事

当前 CCT 模式使用 Tanner–Helland 显示 RGB 近似：

$$
\operatorname{CCT}(\text{Kelvin})\;\longrightarrow\;(R,G,B)
$$

随后归一化色度。它是显示颜色近似，**不是完整的光谱 CIE 模型**。

log-Kelvin 参数化保证色温为正。normalized RGB 则通过 softmax 让色度为正且总和固定，降低颜色与强度之间的整体尺度简并。

---

## 9. 与原版 LightOpt 的区别

核心直接光公式很接近，但完整算法已经明显不同。

| 方面 | 原版 LightOpt | 当前实现 |
|---|---|---|
| 主要任务 | 减少已有灯具数量，同时保持外观 | 已知 GT 灯数，恢复/合成灯具参数 |
| 直接光模型 | Point/Spot 解析衰减 | 沿用 LightOpt-style 公式 |
| 目标场 | Surface irradiance | Direct + 多次反弹 Diffuse Radiosity |
| 可见性 | 原论文假设 $V=1$ | BVH 硬遮挡，每次重算但 detached |
| GI | 原论文不考虑 | 固定 form-factor 矩阵，有限反弹或完整 LU |
| 灯具数量调整 | Removal / Merging / Addition | 完全没有，固定灯数 |
| Overlap loss | 有 | 没有 |
| Attraction loss | 有 | 没有 |
| 表面离散 | Poisson surface sampling | 为 Radiosity 能量守恒而构造的真实三角形剖分 |
| 颜色参数 | RGB 等常规参数 | RGB、normalized RGB、log-CCT |
| 参数评价 | 主要关注外观和运行成本 | Field NRMSE + 排列对齐后的参数误差 |
| 优化器 | 连续优化结合离散灯集重构 | Adam / L-BFGS / Adam→L-BFGS，无离散重构 |
| 生产指标 | Light 数量、ALO、shading cost | 目前没有这些指标 |

一句话概括当前算法的定位：

> **LightOpt-style 直接光项 + BVH 硬遮挡 + 固定 Diffuse Radiosity + 数值优化器**

它不是官方 LightOpt 的完整复刻，也不是 VIALT。原版的灯具集合重构、数量优化和运行成本目标，目前都还没有进入实现范围。

---

## 10. 多灯的排列对称性

多盏灯交换列表顺序，不会改变总照明场：

$$
\sum_{\ell=1}^{M}E_{\ell,j}
=
\sum_{\ell=1}^{M}E_{\pi(\ell),j}
$$

因此多灯参数误差需要经过排列匹配。三灯时穷举 $3!=6$ 种对应关系，选位置误差总和最小的那个：

$$
\pi^{*}
=
\arg\min_{\pi}
\sum_{\ell=1}^{M}
\bigl\|p_{\pi(\ell)}-p_\ell^{\text{GT}}\bigr\|^2
$$

报告同时保留：

- 按标签的位置误差
- 排列对齐后的位置误差
- `parameter_permutation`：GT 灯到结果灯的对应关系

**场拟合成功不等于每盏灯的标签唯一确定。**

---

## 11. 当前验证方式

1. 直接光与多灯逐灯求和测试
2. 遮挡、epsilon 和自遮挡测试
3. Form-factor 面积互易、近场积分和线性残差检查
4. 有限反弹与完整 LU 的一致性检查
5. float64 中心有限差分，且只在 mask 固定的平滑点判断梯度
6. 固定 GT 灯数和既有扰动下的 GT recovery
7. GT/result Radiosity 场图与独立 PT 图

PT 不参与优化目标或 backward。

---

## 12. 当前边界

当前结果不能外推为以下能力：

- 任意跨阴影初始化下的全局恢复
- 正确的移动阴影边界梯度
- 非 Diffuse、纹理、rough conductor 或 dielectric 材质
- 独立 PT 绝对物理精度已经被完全证明
- 候选灯生成、自动灯具初始化或数量预测
- 完整 VIALT / adjoint light tracing

下一步仍应优先验证更多 patch 深度、独立 PT probe 和直接阴影边界处理；候选初始化属于后续 Target-to-Rig 阶段。

---

## 13. 相关结果

- 单灯 Radiosity：`results/round3/report.md`
- 三灯位置、强度与 CCT：`results/round4/report.md`
- 三灯原始数据：`results/round4/raw.json`
- 三灯 GT/result：`results/round4/gt_result/`

---

## 附录 A：Neumann 展开——把间接光按反弹次数展开

令 $K=\operatorname{diag}(\rho)\,F$，GI 方程为：

$$
B=D+K\,B
$$

不断把 $B=D+K\,B$ 代入自己：

$$
\begin{aligned}
B
&=D+K\,B\\
&=D+K\,(D+K\,B)\\
&=D+K\,D+K^{2}B\\
&=D+K\,D+K^{2}D+K^{3}D+\cdots
\end{aligned}
$$

所以：

$$
B=\sum_{k=0}^{\infty}K^{k}\,D
$$

每一项有明确的物理意义：

| 项 | 含义 |
|---|---|
| $D$ | 直接光 |
| $K\,D$ | 一次反弹 |
| $K^{2}D$ | 两次反弹 |
| $K^{3}D$ | 三次反弹 |

这就是 Neumann 展开。它只在矩阵谱半径满足 $\operatorname{sr}(K)<1$ 时保证收敛——物理上对应"每次反弹都有能量损失"。

实验中已用 0/1/2/4/8/16 次反弹与完整解对照，验证收敛行为。

---

## 附录 B：LU 分解——直接解线性方程

把 GI 方程写成线性方程组：

$$
(I-K)\,B=D
$$

令 $A=I-K$，LU 分解把 $A$ 分解为下三角矩阵 $L$ 和上三角矩阵 $U$。实际数值计算通常还带主元置换：

$$
P\,A=L\,U
$$

求解 $A\,B=D$ 时，先解 $L\,y=P\,D$（前向替换），再解 $U\,B=y$（后向替换），而不是显式求逆矩阵。

**LU 和 Neumann 的关系：**

- Neumann：一次次反弹，逐渐接近答案（迭代法）
- LU：直接求出离散线性方程的完整答案（直接法）

LU 的好处：

1. 固定几何/材质时只分解一次
2. 每次灯变化，只需换右侧的 $D$
3. 没有 Monte Carlo 噪声
4. PyTorch 可以通过这个求解过程反向传播

**LU 不负责解决阴影不连续，也不负责解决非凸优化。它只负责 GI 的线性求解。**
