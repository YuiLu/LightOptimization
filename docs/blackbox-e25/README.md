# E25 · 原生 RGB 恢复、观测消融与 CPU 等价性

E25 使用 E24 冻结的场景、灯阵与相机，在 **Unity 2022.3.62f3c1 / URP 14.0.12 / D3D11 / GTX 1060 6GB** 的隔离批处理工程中执行。优化仍由 Unity 原生 RGB 验收；活动灯联合更新 XYZ + log(I)，主对照的 Strict Collider、LampScaledDense 和信任域设定保持不变。C3/C4 直接对锚定 RGB 的 Signed Log 值做差分，不使用链式映射近似。

参数通过标准：每盏活动灯的位置误差 ≤5 cm、相对强度误差 ≤5%，终态合法。图像质量单列；Raw、Signed Log 与固定 Tone-Exp+sRGB 都按视角真值能量归一化。`1560` 检查点只取预算线前最后一个已接受状态，未追加模型探针。

## 实验动机与判读

- **A｜排除阴影后，L6 为什么还失败？** E24 同起点的三视角 A0 终态为 **63.10 cm / 38.00%**，而同组 L1 是成功的对照。L6 的 4D Jacobian 在初态、失败点、GT 及 5/20/50 mm 三种步长下均为满秩；初态的下降梯度与 GT 方向余弦在 −0.075 到 −0.003 之间。失败点首次合法 LM 提案按 ¼、½、1 倍重拍两次，三种步长均未取得真实下降。起点→GT 的两条 41 点路径各含 7 个非法采样，失败点→GT 各含 3 个；合法损失也有回升。数据支持**局部下降方向与恢复方向错位**，不把直线扫描当成四维局部极小值的证明。
- **A｜分离观测与映射。** 全灯 Off 与逐灯贡献叠加的相对差低于 `9e-8`；代表状态的 Raw 差分相对不一致低于 `1.9e-5`。逐灯去背景后，失败点 V0 的 Signed Log 梯度余弦变为 **−0.668**，Raw 梯度余弦仍近 1。A1 改善至 **12.70 cm / 1.06%**，没有通过。固定 4096 个表面点的 D0 到 **3.17 cm / 5.56%**、消耗 1572 次解析评估和 1 次 GT 表面捕获，未过强度线；A2-visible 未启动。
- **A｜视角数量与求解路径。** 同一起点独立用 V0、V1、V2 均通过；V3 复现原失败，V5 为 **40.23 cm / 42.23%**。在共同的 4 次完整模型构建边界，V2 仍是唯一通过的单视角，V3/V5 也均失败。放宽到相同的相机图上限后，V5 在两张独立 Hard 留出图的 Raw NRMSE 从 V3 的 **43.77% / 5.29%** 降到 **35.35% / 4.98%**；等模型次数时该优势消失。多视角未达到预设的七灯扩展门槛。这组结果区分了「单视角已有足够观测」与「联合目标给出可用搜索路径」。
- **B｜联合失败是否由一盏无法恢复的灯造成？** 七盏灯各自都恢复（其他灯固定 GT），共 **1425** 张优化相机图。预先固定的归一化 Gram 块范数排序从 **L6+L7：`8.08e-7`** 到 **L5+L7：`0.1646`**；低端、高端两组 8D 分别用 **309 / 564** 张图恢复。E24 同起点 28D 用 **4620** 张图仍到 **112.38 cm / 194.12%**，误差集中在 L2/L6。共有 37/46 个候选经历全局严格可行性回退；L2/L6 的起点列范数最大/最小比都超过 3，因此增加一项逐参数对角阻尼对照，保留跨灯 Gram 项与全局接受规则。该对照用 4617 张图，终态 **111.45 cm / 196.45%**，仍有 39/48 个候选触发回退；这次尺度调整没有解决联合失败。单灯和双灯均为 Oracle 对照，不能当作联合恢复率。
- **C｜硬阴影对照与低响应量级。** Complex 三个独立起点 C3/C4 各 **3/3**，H0 **0/3**；Classroom 两个起点三种 Hard 方法均失败。dir1301 关阴影全灯可恢复，dir1601 关阴影依然失败。dir1301 追加同一 Hard Signed Log 的完整原生 FD（H1），留出 Raw NRMSE 从其他 Hard 臂的约 66–70% 降至 **42.29%**，但参数仍为 **8.86 cm / 17.82%**，未过线。Classroom 的失败不能统一归因于阴影，也不能由 Complex 的结果外推。
- **C / E｜掩码与条件启动。** 重测历史 Classroom L6/V2：低响应像素占 **96.52%**，却只占该灯该视角 C4 映射 XYZ 列平方能量的 **`1.45e-6`**；弱区梯度范数 `3.73e-11`，全量 `2.52e-5`。弱区与全量梯度范数的比不是可加贡献百分比。所测状态未达软回退的弱区代理误差和复拍联合门槛。dir1301 的 5/50 mm Hard 提案与 20 mm 方向余弦为 **0.860 / 0.803**，未过预先固定的 `<0.8` 不稳定线；dir1601 有不稳定方向，但 Off 不通过。没有同起点满足全部 Moment 门槛，未运行 Moment 代理。

- **D｜瓶颈是映射，不是 28×28 线性求解。** CPU 缓存版的 39 项固定输入全部通过；5 对完整复验的终态、停止原因、实际相机图数及 Hard 留出 Raw NRMSE 完全相同。seed131 热身后三次墙钟中位数 **106.01→86.91 秒（−18.0%）**，未到 20% 门槛。缓存后映射/差分仍占 **62.9%**，Gram/梯度约 **6.6%**；因此测试融合 GPU 列，未启动 GPU Gram 归约。GPU 固定输入列、Gram、梯度和方向通过 **42/42** 项，最大相对 L2 为 `1.36e-5`，上传和同步读回计入后固定输入中位速度比是 **18.31×**。

- **D｜GPU 端到端。** seed131 三次复验墙钟中位数 **106.01→49.09 秒（−53.7%）**；seed157 和 Classroom dir1301 单次分别降 **41.5% / 44.3%**。保留全部参考成功，失败样本的参数与 Hard 留出误差无实质变化。GPU 浮点路径的接受轨迹不完全相同：例如 seed157 优化图从 3288 增为 4209，仍低于相同的 4725 上限。GPU 版是 E25 实验选项，未改插件默认方法。

## C · 独立起点

| 起点 | 方法 | 1560 前通过 | 最终通过 | 最大位置 cm | 最大强度误差 | 实际图/4725 | Hard 留出 Raw / Log / Display | 停止 |
|---|---|---|---|---:|---:|---:|---:|---|
| dir911_20cm | H0 | 否 | 否 | 20.09 | 5.49% | 4662 | 10.30% / 6.75% / 2.47% | `budget` |
| dir911_20cm | C3 | 是 | 是 | 0.00 | 0.00% | 3756 | 0.00% / 0.00% / 0.00% | `near_zero_residual` |
| dir911_20cm | C4 | 否 | 是 | 0.02 | 0.02% | 4665 | 0.04% / 0.14% / 0.09% | `budget` |
| dir1201_30cm | H0 | 否 | 否 | 29.80 | 49.98% | 4689 | 210.51% / 40.37% / 17.66% | `budget` |
| dir1201_30cm | C3 | 是 | 是 | 0.00 | 0.00% | 3039 | 0.00% / 0.00% / 0.00% | `near_zero_residual` |
| dir1201_30cm | C4 | 是 | 是 | 0.01 | 0.06% | 4722 | 0.05% / 0.14% / 0.11% | `budget` |
| seed509_complex_far | H0 | 否 | 否 | 55.12 | 73.13% | 4647 | 39.29% / 30.21% / 17.63% | `budget` |
| seed509_complex_far | C3 | 否 | 是 | 0.00 | 0.00% | 4632 | 0.00% / 0.00% / 0.00% | `budget` |
| seed509_complex_far | C4 | 否 | 是 | 0.00 | 0.00% | 4635 | 0.00% / 0.00% / 0.00% | `budget` |
| dir1301_10cm | H0 | 否 | 否 | 10.86 | 34.76% | 3726 | 70.10% / 26.37% / 14.29% | `no_decreasing_step` |
| dir1301_10cm | C3 | 否 | 否 | 8.05 | 14.44% | 3159 | 66.31% / 20.08% / 10.84% | `no_decreasing_step` |
| dir1301_10cm | C4 | 否 | 否 | 10.03 | 5.69% | 4005 | 67.19% / 25.94% / 13.68% | `no_decreasing_step` |
| dir1301_10cm | A-log | 是 | 是 | 0.00 | 0.00% | 2568 | 0.00% / 0.00% / 0.00% | `near_zero_residual` |
| dir1601_10cm | H0 | 否 | 否 | 10.43 | 6.82% | 4617 | 68.10% / 30.17% / 16.57% | `budget` |
| dir1601_10cm | C3 | 否 | 否 | 10.19 | 5.24% | 1794 | 68.03% / 30.63% / 16.74% | `no_decreasing_step` |
| dir1601_10cm | C4 | 否 | 否 | 10.23 | 4.95% | 2241 | 68.06% / 30.69% / 16.81% | `no_decreasing_step` |
| dir1601_10cm | A-log | 否 | 否 | 9.14 | 5.07% | 1863 | 67.16% / 29.36% / 16.17% | `no_decreasing_step` |
| dir1301_10cm | H1 | 否 | 否 | 8.86 | 17.82% | 4701 | 42.29% / 16.65% / 8.76% | `budget` |

留出指标为高分辨率视角 3–4 的均方根。A-log 的图像列仍以真实 Hard 渲染复拍，参数列为 Off Oracle 结果。

## A · Classroom L6 观测消融

| 臂 | 最大位置 cm | 最大强度误差 | 实际优化相机图 | 通过 | 停止 |
|---|---:|---:|---:|---|---|
| seed509_classroom_near_oracle_L6_A1-olat | 12.70 | 1.06% | 561 | 否 | `no_decreasing_step` |
| seed509_classroom_near_oracle_L6_A2-D0 | 3.17 | 5.56% | 0 | 否 | `budget` |
| seed509_classroom_near_oracle_L6_single_v0_A-log | 0.00 | 0.00% | 223 | 是 | `no_decreasing_step` |
| seed509_classroom_near_oracle_L6_single_v1_A-log | 0.00 | 0.00% | 100 | 是 | `near_zero_residual` |
| seed509_classroom_near_oracle_L6_single_v2_A-log | 0.03 | 0.00% | 37 | 是 | `near_zero_residual` |
| seed509_classroom_near_oracle_L6_V3_A-log | 63.10 | 38.00% | 1071 | 否 | `no_decreasing_step` |
| seed509_classroom_near_oracle_L6_V5_A-log | 40.23 | 42.23% | 2900 | 否 | `no_decreasing_step` |

## B · Complex Far 单灯与联合

| 臂 | 最大位置 cm | 最大强度误差 | 实际优化相机图 | 通过 | 停止 |
|---|---:|---:|---:|---|---|
| seed509_complex_far_oracle_L1_A-log | 0.00 | 0.00% | 165 | 是 | `near_zero_residual` |
| seed509_complex_far_oracle_L3_A-log | 0.00 | 0.00% | 165 | 是 | `near_zero_residual` |
| seed509_complex_far_oracle_L4_A-log | 0.00 | 0.00% | 165 | 是 | `near_zero_residual` |
| seed509_complex_far_oracle_L5_A-log | 0.00 | 0.00% | 273 | 是 | `near_zero_residual` |
| seed509_complex_far_oracle_L7_A-log | 0.00 | 0.00% | 165 | 是 | `near_zero_residual` |
| seed509_complex_far_lowest_L6+L7_A-log | 0.00 | 0.00% | 309 | 是 | `near_zero_residual` |
| seed509_complex_far_highest_L5+L7_A-log | 0.00 | 0.00% | 564 | 是 | `near_zero_residual` |
| seed509_complex_far_A-log-diag | 111.45 | 196.45% | 4617 | 否 | `budget` |

## D · 固定输入与端到端复验

固定输入通过 39/39 项（L2：`max(1e-12, 1e-6 × 参考范数)`）。真实锚点/探针由 D 固定输入阶段重新捕获；另测有符号极值及一、二阶单边模板。

| 起点 | 方法 | 重复 | 墙钟中位数（秒） | 范围（秒） |
|---|---|---:|---:|---:|
| d-replay-1301-classroom | C4 | 1 | 77.50 | 77.50–77.50 |
| d-replay-1301-classroom | C4-cache | 1 | 64.91 | 64.91–64.91 |
| d-replay-131-complex | C4 | 3 | 106.01 | 100.29–106.70 |
| d-replay-131-complex | C4-cache | 3 | 86.91 | 86.77–88.76 |
| d-replay-157-complex | C4 | 1 | 72.04 | 72.04–72.04 |
| d-replay-157-complex | C4-cache | 1 | 66.86 | 66.86–66.86 |

相机提交/同步读回与 CPU 阶段分开记录。未取得 GPU 独占时间时标记 `unavailable`。

完整恢复复验共 5 对，参考成功保留 4/4；最大归一化终态差 0。

GPU 固定输入对照通过 42/42 项。含上传、提交和同步读回的列构建，相对 CPU 缓存的中位速度比为 18.31×。此处仅测固定输入，不把微基准速度比当作恢复端到端收益。

GPU 恢复对照完成 5 对。参考成功丢失 0 个；相同预算下实际渲染数和终态差异逐臂见 `gpu-end-to-end.csv`。

| 起点 | GPU 重复 | GPU 墙钟中位数（秒） | CPU 参考中位数（秒） | 相对降幅 |
|---|---:|---:|---:|---:|
| d-gpu-recovery-1301-classroom | 1 | 43.16 | 77.50 | 44.3% |
| d-gpu-recovery-131-complex | 3 | 49.09 | 106.01 | 53.7% |
| d-gpu-recovery-157-complex | 1 | 42.18 | 72.04 | 41.5% |

复验仍用 CPU 的 mask/P95、完整 Gram 和 LM 求解；GPU 只融合 XYZ 列。实际捕获与求解器之间仍有传输。

## 代表图

- Complex 20 cm：[起点](figures/c-complex-911_dir911_20cm_C3_initial_v0.png) · [GT](figures/c-complex-911_gt_on_gt_v0.png) · [H0](figures/c-complex-911_dir911_20cm_H0_best_v0.png) · [C3](figures/c-complex-911_dir911_20cm_C3_best_v0.png) · [C4](figures/c-complex-911_dir911_20cm_C4_best_v0.png)。
- Classroom L6：[Off 失败点的 RGB 残差热点](figures/a-classroom_L6_hotspots_raw_error_v2.png) · [历史 Hard 失败点弱响应区](figures/c-audit-classroom_mask_e24_seed509_classroom_near_log_failed_v2_weak.png) · [同一状态的 Hard 位置响应](figures/c-audit-classroom_mask_e24_seed509_classroom_near_log_failed_v2_hard_position_response.png)。

## 可复现记录

`stage-summary.json` 保存实际完成的阶段、资产及运行版本、条件门槛。`raw-data.zip` 保存每阶段完整数值表、源文件哈希与 GT/初态/终态快照；仓库只保留代表图，不包含 Unity 缓存、全程画面或 D 的浮点帧缓冲。D 固定输入的哈希、坐标和 stencil 在 `fixed-inputs.csv`。

在与 E24 资产清单匹配的独立批处理工程中运行（不要指定主 Unity 工程）：

```powershell
$unity = "E:\UnityEngine\2022.3.62f3c1\Editor\Unity.exe"
$project = "E:\Light\tmp\e24-work\project"
.venv\Scripts\python.exe -B mvp/blackbox_e25.py --unity $unity --project $project --output tmp/e25 --sync --pipeline --resume
.venv\Scripts\python.exe -B mvp/blackbox_e25_report.py --roots tmp/e25 --output docs/blackbox-e25
```

本次分批使用 `tmp/e25-recovery`、`tmp/e25-diagnostics`、`tmp/e25-followups`、`tmp/e25-gpu-replay`，原始证据按实际阶段独立保存；报告生成器支持将四个目录一起传给 `--roots`。GPU 开关只组装 XYZ 列，不改变渲染、强度列、接受规则或最终验收。
