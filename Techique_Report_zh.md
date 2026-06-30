# 技术报告

本文档记录氩 Franck-Hertz 教学实验数据有效多通道响应分析工作流的实现细节。内容与当前源码树和整理后的 source-data package 同步。该分析将测得的电流-电压曲线分解为有效响应通道；本文档不把这些通道解释为逐条原子能级谱线指认。

## 1. 仓库范围与源码映射

公开复现目标是当前仓库。运行结果写入 `output/`，并由 git 忽略。已提交的 `source_data_package/` 是面向手稿核验的精选归档，由当前重算工作流派生。

主要实现文件如下：

- `run.py`：命令行入口。
- `src/sublevel_detect/cli.py`：CLI 合约和工作流分发。
- `src/sublevel_detect/paths.py`：仓库相对路径默认值；正式输出根目录为 `output/`。
- `src/sublevel_detect/main_pipeline.py`：主 fullscan 配置、forward prior 构造和后评估分发。
- `src/sublevel_detect/model.py`：数据读取、物理核、损失函数、优化器、训练循环、指标、超参搜索和 K-neutral selector。
- `src/sublevel_detect/ablation_pipeline.py`：selector-only 消融和 no-forward-anchor-gap 消融。
- `src/sublevel_detect/robustness_pipeline.py`：selector 权重扰动和 leave-one-retarding-voltage-out 稳健性流程。
- `src/sublevel_detect/sensitivity_pipeline.py`：修复后的 prior-strength 扫描、残差 bootstrap、噪声扰动、peak-window-radius 扰动和 K=4 不确定度汇总。
- `scripts/build_source_data_package.py`：精选数据包构建、哈希生成、validation 和路径清理。

## 2. 输入数据与输出合约

默认输入为 `data/argon/FHdata.xlsx`。读取器支持 Excel 和 CSV，并自动解析曲线编号、拒止电压 `Vr`、加速电压 `Va` 和电流列的常见列名。当常规 Excel 引擎不可用时，`.xlsx` 文件会通过 workbook XML 后备读取。

正式输出根目录为 `output/`。主流程写出：

- `output/main/fullscan/decision.json`
- `output/main/fullscan/model_selection_table.csv`
- `output/main/fullscan/scan_summary.csv`
- `output/main/k_selected_full/prediction_points.csv`
- `output/main/paper_summary.json`

selected-model 逐点预测表固定字段为：

```text
curve_id,Vr,Va,observed,predicted,residual
```

该表是 residual bootstrap 的强制输入。若表不存在，sensitivity 分析会直接失败。

## 3. 完整工作流

正式复算命令为：

```powershell
python run.py --mode fullscan --output output
python run.py --mode fullscan --output output --ablation --robustness
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

工作流阶段如下：

1. 读取并归一化输入曲线。
2. 从观测振荡结构估计 forward evidence。
3. 编译 soft forward prior，并应用到 production 配置。
4. 按需运行 successive-halving 超参优化。
5. 在配置的 K 候选范围内训练模型。
6. 保留最佳 K=1 运行和 selected K 运行。
7. 导出 selected model 的逐点预测和残差。
8. 写出模型选择、结构、物理响应、退化诊断和论文摘要输出。
9. 在请求时运行消融、稳健性和敏感性流程。
10. 构建精选 source-data package 并生成 validation report。

## 4. 配置默认值

production fullscan 使用：

| 参数 | 数值 |
| --- | ---: |
| `epochs` | 3500 |
| `scan_seeds` | `0,1,2` |
| `level_scan_min`, `level_scan_max` | 1, 8 |
| `optimizer` | `muon_hybrid` |
| hyperopt 后 `lr` | 0.0025 |
| `weight_decay` | 0.0001 |
| `grad_clip` | 5.0 |
| `early_stop_min_epochs` | 300 |
| `early_stop_warmup` | 300 |
| `early_stop_patience` | 45 |
| `peak_window_radius` | 1.5 |
| `kernel_mode` | `k_neutral` |
| `cluster_tolerance_eV` | 0.02 |
| `cpu_workers` | 4 |
| `dispatch_strategy` | CPU 上为 `cpu_4` |

`config_used.json` 中记录的 production forward prior 为：

| 参数 | 数值 |
| --- | ---: |
| `forward_main_spacing` | 11.5 |
| `forward_spacing_std` | 0.4310839052125854 |
| `forward_confidence` | 0.9475202202349895 |
| `forward_anchor_step` | 0.25 |
| `w_prior_anchor` | 0.0005790080880939959 |
| `w_prior_gap` | 0.0003895040440469979 |

## 5. 物理核

模型类为 `PoissonRateFHCoreMultiLevel`。对于加速电压 `Va` 和拒止电压 `Vr` 下的一个采样点，核心计算为：

```text
E_collision = clamp(Va, min=0)
E_collect = Va - vr_scale * Vr
drive = softplus(E_collision - v_emit)
envelope = amp * drive^power / norm + offset + slope * E_collision
collector_transmission = collector_floor + (1 - collector_floor) * sigmoid((E_collect - collector_threshold) / collector_width)
```

高电压和后段响应门控为：

```text
late_gate = sigmoid((E_collision - late_onset) / late_width)
high_vr_gate = sigmoid((Vr - 6.0) / 2.0)
vr_norm = clamp((Vr - 5.0) / 5.0, -1.5, 1.5)
contrast_scale = exp(clamp(vr_contrast * vr_norm + late_contrast * late_gate + vr_late_contrast * vr_norm * late_gate, -0.65, 0.65))
```

高能衰减因子为：

```text
high_energy_gate = sigmoid((E_collision - high_energy_loss_onset) / high_energy_loss_width)
high_energy_loss = 1 - high_energy_loss_strength * high_energy_gate * (0.35 + 0.65 * high_vr_gate)
```

对于 K 个有效通道，能量为 `E_i`，softmax 权重为 `w_i`。振荡项为：

```text
phase_i = (E_collision + phase) / E_i
residual_i = phase_i - round(phase_i)
dip_i = exp(-0.5 * (residual_i * E_i / width)^2)
weighted_dip = sum_i w_i * dip_i
decay = exp(-damping * E_collision)
modulation = clamp(1 - osc_amp * contrast_scale * weighted_dip * decay, 0.03, 1.15)
prediction = clamp(envelope * collector_transmission * high_energy_loss * modulation, min=0)
```

逐曲线 nuisance 参数在 neutral core 之后施加 gain、bias 和加速电压偏移。能量参数通过构造保证边界：第一通道能量映射到 9.0 至 14.5 V 区间；后续能量由正间隔产生，并由 minimum-gap 正则项约束。

## 6. 损失函数

对每条曲线，训练损失包含原始电流误差、一阶导数误差、二阶导数误差、peak-window 加权电流误差、平滑项、后段 bias、后段振幅比例、高拒止电压 valley-depth 和正则项。总损失为：

```text
L = w_raw * L_raw
  + w_d1 * L_d1
  + w_d2 * L_d2
  + w_peak_window * L_peak
  + w_smooth * L_smooth
  + w_vr_late_bias * L_late_bias
  + w_vr_amplitude_ratio * L_late_ratio
  + w_high_vr_valley_depth * L_high_vr_valley
  + L_reg
```

production 权重为：

| 项 | 权重 |
| --- | ---: |
| `w_raw` | 1.0 |
| `w_d1` | 0.04 |
| `w_d2` | 0.02 |
| `w_peak_window` | 0.08 |
| hyperopt 后 `w_smooth` | 0.005 |
| `w_vr_late_bias` | 0.04 |
| `w_vr_amplitude_ratio` | 0.04 |
| `w_high_vr_valley_depth` | 0.06 |
| `w_reg` | 0.0001 |
| `level_weight_entropy` | 0.001 |

正则项包含参数 L2 惩罚、通道权重熵惩罚、minimum-gap 惩罚和 forward-anchor 惩罚。forward prior 启用时，anchor grid 从 production main spacing 开始；未启用时，fallback 为 11.55 V 起点和 0.25 V 步长。

early-stopping monitor 比训练损失更严格：

```text
L_monitor = L_total
          + 0.25 * L_d1
          + 0.10 * L_d2
          + 0.25 * L_peak
          + 0.20 * L_late_bias
          + 0.10 * L_late_ratio
          + 0.20 * L_high_vr_valley
```

## 7. 优化器与训练

默认优化器为 `muon_hybrid`。

本地 Muon 分支对向量或矩阵参数维护 momentum buffer，并执行：

```text
buffer_t = momentum * buffer_{t-1} + gradient
denom = sqrt(mean(buffer_t^2))
theta_t = theta_{t-1} - lr * buffer_t / max(denom, 1e-8)
```

weight decay 会先加入梯度，再进入 momentum 更新。标量参数和不适合 Muon 分支的参数使用 AdamW fallback。若设置 `optimizer="adamw"`，所有可训练参数均使用 AdamW。

训练循环支持 fullscan 模式下的 checkpoint resume。只有当 model schema、请求 epoch、early-stop 下限和配置 hash 全部匹配时，scorecard 才被认为可复用。CPU fullscan 使用 `cpu_workers=4` 的进程级并行。

## 8. 超参优化

fullscan 默认启用超参优化；传入 `--exclude hpopt` 可跳过。方法为 successive halving：

| 阶段 | Epochs | Levels | Seeds | 保留候选数 |
| --- | ---: | --- | --- | ---: |
| Stage 1 | 80 | 1,2,4,6,8 | 0 | 4 |
| Stage 2 | 240 | 2,4,6,8 | 0,1 | 2 |
| Stage 3 | 300 | 1,2,3,4,5,6,7,8 | 0,1 | 1 |

候选更新集包含 baseline、structure guard、low-smooth peak、更强二阶导数、平衡一阶/二阶导数、low-smooth high-peak、拒止电压响应 guard、更强 late-bias guard、更强 prior anchor、更强 prior gap、更低学习率加更长 patience、以及带 guard 的更高学习率。当前 production 选中的候选为 `lr_higher_guarded`，对应 `lr=0.0025` 与 `w_smooth=0.005`。

hyperopt score 为：

```text
score = rmse_mean
      + 0.10 * structure_score
      + 0.08 * vr_physical_response_score
      + 0.03 * d1_rmse_mean
      + 0.03 * d2_rmse_mean
      + 0.35 if flatline_guard_pass is false
```

## 9. K-neutral 模型选择

selector 扫描 K=1 至 K=8，并采用 rank aggregation，而不是单一指标。默认 rank 权重为：

| 组件 | 权重 |
| --- | ---: |
| fit RMSE | 1.0 |
| summary/CV RMSE | 1.0 |
| 一阶导数 | 1.0 |
| 二阶导数 | 1.0 |
| structure score | 1.25 |
| physical-response score | 1.0 |
| BIC | 1.0 |
| AIC | 0.5 |
| degeneracy penalty | 1.0 |

composite score 是加权 rank 之和，并加入 top-heavy 或 low-weight 通道退化、能量聚簇和 flatline failure 惩罚。selected K 为 composite score 最小的 K；若分数相同，选择较小 K。

当前 production 结果为：

| 指标 | 数值 |
| --- | --- |
| selected K | 4 |
| best K by fit | 4 |
| best K by first derivative | 8 |
| best K by second derivative | 8 |
| best K by structure | 8 |
| best K by physical response | 4 |
| best K by BIC | 4 |
| best K by AIC | 4 |

这意味着 K=4 是 production prior 和 selector workflow 下选择的有效响应模型。K=8 仍是导数和结构指标暴露出的明确边界情形。

## 10. Production K=4 参数

保留的 production K=4 anchors 为：

| 通道 | Energy V | Weight |
| ---: | ---: | ---: |
| 1 | 11.500251770019531 | 0.3593961000442505 |
| 2 | 11.739255905151367 | 0.3961048126220703 |
| 3 | 12.59369945526123 | 0.08622404932975769 |
| 4 | 13.96471881866455 | 0.15827499330043793 |

这些通道是从教学实验曲线推断得到的有效响应通道，不是 term-resolved spectroscopic assignment。

## 11. 消融试验设计

消融流程分为两类。

selector-only 消融不重训，只重新计算决策：

- 移除 weight-degeneracy penalty；
- 移除 energy-cluster degeneracy penalty；
- 使用 metric/BIC/AIC-only selector 变体。

no-forward-anchor-gap 消融会在关闭 forward anchor 和 gap prior 后重新训练扫描。当前证据中，selector-only 消融保持 K=4；no-forward-anchor-gap 重训选择 K=7。这说明 K=4 是 production physical prior 和 audit workflow 下的条件性结果。

## 12. 稳健性试验设计

稳健性流程包含两部分。

selector-weight perturbation 在固定 rank-weight 网格上重新计算决策，不重训。当前 package 记录 23 个 selector-weight perturbation 情景全部选择 K=4。

leave-one-retarding-voltage-out 每次排除一条拒止电压曲线，并复用主基线超参配置重训，不在每折重新运行 hyperopt。当前汇总中 5 折里有 3 折选择 K=4，其余折暴露 K=1 和 K=6 边界结果。

## 13. 敏感性与扰动设计

sensitivity 工作流复用已有 main fullscan 作为基线。

prior-strength 扫描使用：

```text
0, 0.25, 0.5, 1, 2, 4
```

其中 `1x` 严格定义为 `main/fullscan/config_used.json` 中记录的 production prior 权重，不会从已应用 prior 的配置上再次编译。当前 selected-K 结果为：

| Prior factor | Selected K |
| ---: | ---: |
| 0 | 8 |
| 0.25 | 4 |
| 0.5 | 4 |
| 1 | 4 |
| 2 | 8 |
| 4 | 8 |

uncertainty 扰动包含：

- seed-dependent 初始化扰动；
- 从 selected-model residuals 抽样的 residual bootstrap；
- 基于原始电流局部残差尺度的曲线级噪声扰动；
- peak-window-radius 扰动。

residual bootstrap 是强制口径：它按曲线从 `main/k_selected_full/prediction_points.csv` 采样；若该表缺失则失败。它不再 fallback 到中心化原始电流。package validation 中 bootstrap noise standard deviation 为 0.07944132053812393 uA，selected-model residual standard deviation 为 0.08032634434767792 uA，绝对差为 0.000885023809553992 uA。

K=4 不确定度分成两类：

- `conditional_k4_all_fits`：扰动条件下所有拟合得到的 K=4 通道；这是 stress-test drift 表。
- `production_anchor_matched_k4`：将扰动 K=4 通道按总绝对能量差最小原则匹配回四个 production anchors；这是 production-anchor matched uncertainty 表。

## 14. Source-Data Package

精选 package 包含：

- 29 个 manuscript-facing CSV/JSON source tables；
- 71 个 figure assets；
- 16 个保留的 K=1 和 selected K=4 run-record files；
- 193 个精选 output-result files；
- `FILE_INDEX.csv`、`SHA256SUMS.txt`、`validation_report.json` 和 `source_data_package_manifest.md`。

当前 validation status 为 `pass`。package builder 在哈希和 manifest 写出前，会清理文本载荷中的本地绝对路径。

从已有正式输出根重建 package：

```powershell
$env:SUBLEVEL_OUTPUT='output'
python scripts/build_source_data_package.py
```

如果 manuscript visualization 的 source-data 和 figure 目录位于仓库外部，应显式指定：

```powershell
$env:SUBLEVEL_SOURCE_TABLES='<path-to-source-data>'
$env:SUBLEVEL_FIGURES='<path-to-figures>'
python scripts/build_source_data_package.py
```

## 15. 验证命令

仓库级检查命令为：

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest -q
python scripts/build_source_data_package.py
```

package 级一致性检查保存在 `source_data_package/validation_report.json`，覆盖：

- source-table 数量为 29；
- figure-asset 数量为 71；
- prior-strength `1x` 选择 K=4；
- production anchor energies 与保留 K=4 anchors 一致；
- residual-bootstrap noise scale 与 selected-model residual scale 的差异小于 0.01 uA；
- 所有 package 文件都有 SHA256 哈希。
