# E20 · 独立灯光信号与点光源 MSM

基线是 `blackbox@60285c5` 的 [UrpNativeComplexRoom](../blackbox-e19/README.md)。使用 Unity 2022.3.62f3c1、URP 14.0.12、D3D11、RTX 4070 Ti SUPER、线性 HDR 和原生 RGB 重拍。训练视角 0–2 为 448×256。五个复拍视角为 896×512，视角 3–4 留出。原生 LM 维持 E19 上限，每臂 2,000 次三视角查询。通过要求每盏参与拟合的灯同时满足 **位置误差 ≤5 cm、相对强度误差 ≤5%**。

## E20-A · 关闭阴影后的失败机制

复用 E19 的 131、733 号种子，用同一启动器增加 157、181。seed 181 的 0.45 倍插值违反 E19 的可行域检查，按 E19 原逻辑回退到 Far。同一实际起点的重复臂保留在原始数据中，从 Near 成功率中扣除。

| Shadow Off 对照 | 实际 Near | Far |
|---|---:|---:|
| A1 · 全图，单灯 4D | 19/21 | 27/28 |
| Oracle · 只开当前灯且环境光归零，单灯 4D | 18/21 | 26/28 |
| A2 · 全图，两灯 8D | 5/6 | 6/8 |
| A3 · 全图，七灯 28D | 1/3 | 0/4 |

- **联合问题随起点和灯组合变化。** seed 733 的 Near 在完整画面单灯对照中 7/7 通过，两个双灯组合也通过。七灯训练 NRMSE 从 88.96% 降到 64.38%，同视角高分辨率 69.24%，留出 83.30%，最大位置误差 19.14 cm、强度误差 28.12%。seed 157 的 Near 中，L5 和 L7 的单灯对照分别通过，联合后最大位置和强度误差分别为 10.12 cm 和 16.43%。
- **隔离单灯以后仍会失败。** seed 131 Near 的 L6 在 A1 和 oracle 中分别停在 8.61 cm 和 8.57 cm，停止原因均为 `no_decreasing_step`。seed 733 的 L3 在 A1 通过，oracle 停在 10.60 cm。每视角按该目标能量归一化，切换到 oracle 会改变视角的相对权重。两组分开解读。
- **局部方向随尺度变化。** 三训练视角合成的 2 cm / 0.05 log 强度 Jacobian 在 GT 和所有起点都达到 28/28 有效秩。GT、seed 131 Near、733 Near 和 157 Near 的条件数分别是 99、2,369、229 和 460。733 Near 的 L3-Y 从 2 cm 到 5 cm，列方向余弦只有 0.058，范数变为 17.41 倍。沿同起点负梯度，归一化步长 0.01 使原生平方误差下降 4.7%，0.02 时实际增加 72.4%，而一阶模型预测下降 12.8%。满秩支持局部响应存在，步长扫描直接暴露了线性模型的适用范围。
- **少量可见表面像素主导误差。** seed 131 Near 的视角 1 中，Top 1% 像素占平方误差 99.99%，最高 RGB 误差幅值约 4,769。其峰值邻近的固定表面采样点离初始 L5 约 6 cm，屏幕距离 2.24 px。该距离对应邻近采样点。seed 157 Near 的七灯最终训练 NRMSE 为 64.65%，同视角高分辨率复拍为 1,026.61%，表明低分辨率训练还会低估热点。

双灯组合来自单灯通过、同组联合失败的记录，再按列间最大绝对相关选择强相关 L5–L7（0.542）和同级的弱相关 L2–L7（0.00075）。两个数是所用起点与尺度下的最大值。所有臂保留参数真值、起点、终点、逐次原生 RGB 查询以及 LM 预测和实际变化。局部诊断录入五个独立视角和三训练视角合成矩阵，奇异值有效秩使用最大奇异值的 `1e-5` 阈值。

[维度与恢复率](recovery-by-dimension.png) · [逐视角奇异值谱](singular-spectrum.png) · [列相关矩阵](jacobian-correlation.png) · [负梯度真实损失扫描](gradient-direction-scans.png)

## E20-direct · 无阴影直接照明代理

在每个视角固定采 7,168 个可见表面点，保存位置、法线、已知 URP Lit 的漫反射颜色以及对应 GT 单灯线性 RGB。目标 RGB 扣除同视角全关灯残量。候选由点光源的 Lambert 漫反射、距离平方和 URP Range 衰减计算。模型假设这里的材质为不透明 URP Lit，保留近距离钳位，省略镜面项。GT 单灯图像仅用于合成 oracle。B2 用代理给出局部候选，按三训练视角的原生 RGB 真实损失决定是否接受。[Unity URP 点光衰减源码](https://github.com/Unity-Technologies/Graphics/blob/master/Packages/com.unity.render-pipelines.universal/ShaderLibrary/RealtimeLights.hlsl)

| 方法 | Near 通过 | Far 通过 | Near 位置误差中位数 | Far 位置误差中位数 |
|---|---:|---:|---:|---:|
| B0 · 原生 RGB，复用对应的 A1 | 13/14 | 14/14 | <0.01 cm | <0.01 cm |
| B1 · 直接照明代理 | 4/14 | 4/14 | 4.27 cm | 4.27 cm |
| B2 · 代理提案，原生 RGB 验收 | 4/14 | 3/14 | 4.26 cm | 4.29 cm |

- **代理在个别失败案例提供了有效的位置方向。** seed 131 Near 的 L6 从 B0 的 8.61 cm 降到 B2 的 0.65 cm，高分辨率同视角 NRMSE 从 49.37% 降到 1.25%，留出从 4.10% 降到 0.77%。强度误差上升至 7.45%，所以该臂仍未通过。seed 131 Far 的同一盏灯，B0 恢复到测量精度，B2 停在 49.30 cm。
- **已知材质下仍有模型偏差。** 直接将 GT 参数代入代理，七盏灯的训练特征 NRMSE 仍有 5.00%–49.99%。以 2 cm 与 0.05 log 强度估计的代理和原生梯度余弦中位数为 0.75。seed 131 Near 的 L5 在三组差分尺度下得到 −0.41、0.12、−0.84。B2 的接受步骤均降低原生训练损失，但 28 臂中有 20 臂因原生 RGB 持续拒绝代理提案而停止。正向模型误差和提案方向的尺度敏感性共同限制了这版代理的适用范围。

[B0/B1/B2 参数误差](direct-proxy-recovery.png) · [差分尺度与梯度一致性](direct-gradient-agreement.png)

## E20-MSM · Unity 点光源软可见性

首灯按 GT 中单独切换该灯阴影带来的全图 RGB 变化选择，L1 最大（训练 NRMSE 3.06%）。自定义 Shadow Pass 渲染六面径向深度四阶矩，固定相机表面采样点，跨面做两步高斯滤波，再在 HLSL 中计算 MSM 可见性。主配置为每面 128×128、实际 4× MSAA 和 5 像素核，使用 `R32G32B32A32_SFloat`。近远平面为 0.20 m 和 7 m，moment bias 为 0.001，depth bias 为 1.2 cm，leak bias 为 0.06。测试含 64/128/256 分辨率、实际 1×/4×/8× MSAA、3/5/9 像素核和 VSM 对照。采用硬件光栅化与有限差分估计灯位梯度。[Peters 与 Klein 的 MSM](https://momentsingraphics.de/I3D2015.html) · [Unity SRP 的矩阴影参考实现](https://github.com/Unity-Technologies/ScriptableRenderPipelineData)

| L1 起点 | 初始位置误差 | MSM XYZ 终点 | 阴影特征训练 NRMSE | 特征留出 NRMSE | 原生 On 正向同时优于中心和反向 | MSM 4D 原生训练 NRMSE |
|---|---:|---:|---:|---:|---:|---:|
| 131 Near | 23.83 cm | 0.04 cm | 44.49% → 0.45% | 0.53% | 4/5 | 41.13% → 2.90% |
| 131 Far | 52.95 cm | 0.12 cm | 56.90% → 0.57% | 0.71% | 5/5 | 45.97% → 1.64% |
| 733 Near | 27.88 cm | 0.37 cm | 51.18% → 1.81% | 2.37% | 5/5 | 38.73% → 2.23% |
| 733 Far | 61.95 cm | 82.49 cm | 65.79% → 63.61% | 56.13% | 0/5 | 43.40% → 43.40% |

- **Level 1 固定 GT 强度，只拟合 XYZ。** 每臂执行 175–180 次可见性查询，每次渲染六面。seed 733 Far 的代理损失降低，位置却进一步偏离 GT，原生训练 NRMSE 从 44.63% 升到 45.37%。
- **Level 2 保留扰动后的强度，从起点沿代理方向扫 0.5、1、2、4、8 cm。** Shadow On 和 Off 各起点的正向支持数相同。733 Far 在两种原生目标中的五个中心差分方向导数均为正，代理方向与指向 GT 的余弦为 −0.75。随后在 XYZ 提案处用原生 RGB 估计强度修正，再联合试探。前三个起点有原生 RGB 改善，位置和强度同时通过 2/4，733 Far 的所有 4D 提案都被拒绝。4D 验收额外用了每臂 9 次原生查询和 6 次可见性查询。L1 的纯原生 Shadow On 对照 M0 以 124–299 次原生查询做到 4/4 通过，训练 NRMSE 逼近零。
- **可见性和梯度诊断。** 主配置 GT 在五视角的可见性相对原生单灯 RGB 阴影因子的加权 RMSE 为 0.37%–2.17%。采样到的跨面最大可见性差为 0.00048，八组设置记录到的退化矩与 NaN/Inf 均为 0。128 分辨率下，步长 0.25–5 cm 的相邻梯度最小余弦从 1× MSAA 的 0.57 提高到主配置的 0.90。733 Far 在 2 cm 步长处，各设置相对主配置的梯度余弦最低约为 0.974。梯度稳定时仍观察到目标方向冲突，现有结果不足以把该失败归因于边缘覆盖跳变。

本轮将进入七灯代理的门槛设为每个 Near 起点的位置均恢复至 5 cm 内，且每个 Near/Far 起点均在至少 3 个步长上通过原生 Shadow On 方向检查。两个 Near 均恢复，方向检查通过 3/4 个起点，实验停在首灯。28D Hybrid 与外部光栅化分别受单灯方向检查和边缘覆盖证据约束。可行域沿用 E19，灯位监督来自直接照明和可见性观测。

[原生方向扫描](moment-native-direction.png) · [原生方向导数](moment-native-derivative.png) · [差分稳定性](moment-gradient-stability.png) · [六面矩纹理](moment-cube.png)

## 数据与复现

三阶段实际执行 204 个实验臂，共 152,093 次原生相机渲染。汇总表有 232 行，其中 28 行是复用的 B0。A、Direct、MSM 分别用时 1,468.7、327.4、72.1 秒。A 期间曾与代理短跑共享 GPU，墙钟供复现参考。查询归档含配置、逐次查询、逐轴差分探针、特征和原生 RGB 复测指标、Jacobian 谱与 Gram 矩阵。汇总 CSV 的 `dx_cm`、`dy_cm`、`dz_cm` 按活动灯顺序记录有符号位置误差。

[实验臂汇总 CSV](trials.csv) · [CSV/JSON 查询与配置归档](query-data.zip) · [GT、初始值和最优值对照](features-and-native-rgb.png) · [逐视角原生 RGB 误差图](native-rgb-error-by-view.png)

复跑后，`--output` 下的 `e20_a/captures`、`e20_direct/captures`、`e20_msm/captures` 保存五视角原生线性 RGB EXR、可视化 PNG 和误差图。Direct 与 MSM 另保存固定表面特征以及 MSM 原始和滤波后的六面 float32 矩纹理。优化器用分阶段查询预算核算原生相机和 shadow face 渲染。对一次查询未触发的目标，查询表保留空值。

从仓库根目录在 PowerShell 运行下列命令。将 `$unity` 指向 Unity 2022.3.62f3c1 的可执行文件，将 `$settings` 指向 E19 对应的线性 HDR URP 项目设置。报告脚本需要 NumPy、Matplotlib 和 Pillow。

```powershell
$unity = 'C:\Program Files\Unity\Hub\Editor\2022.3.62f3c1\Editor\Unity.exe'
$settings = 'C:\Work\E19Project\Assets\Settings'

python mvp/blackbox_e20.py run `
  --unity $unity `
  --settings $settings `
  --project results/e20-repro/project `
  --output results/e20-repro

python mvp/blackbox_e20.py report `
  --output results/e20-repro `
  --report-dir results/e20-repro/report
```
