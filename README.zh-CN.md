# SubLevel Detect 中文说明

[English README](README.md)

这是用于论文复现的 Frank-Hertz 氩有效多通道响应分析源码项目。

本仓库在常规复现层面采用“代码优先”策略：仓库包含源码、输入数据表、测试与说明文档；新的运行输出仍写入 `output/`，不作为普通源码变更提交。为便于手稿核验，当前版本单独提交了整理后的 `source_data_package/`，其中包含当前论文草稿使用的图像资产、manuscript-facing CSV/JSON 表、保留的 K=1/K=4 运行记录和校验清单。

## 技术报告

实现细节记录在：

- `Techique_Report.md`：英文技术报告。
- `Techique_Report_zh.md`：中文技术报告。

报告覆盖完整工作流、损失函数设计、物理核解析式、优化器分流、超参搜索、K-neutral selector、forward-prior 逻辑、扰动结构、消融试验、稳健性试验和 source-data package validation。

## 项目复现内容

代码拟合并评估多能级 Frank-Hertz 氩模型。正式复现流程包含两条平行基线：

- 主基线：构建正向证据、可选超参调节、候选 K 扫描训练、自动后评估。
- 消融基线：selector-only 消融，以及关闭 forward anchor gap 的 retrain，用于检验最终选择对正向锚点的依赖程度。
- 稳健性基线：selector 权重扰动，以及固定主基线超参后的 leave-one-retarding-voltage-out 重训。
- 敏感性补充实验：forward-anchor prior-strength 扫描，以及 seed jitter、残差 bootstrap、噪声扰动和峰谷窗口半径扰动下的两类 K=4 不确定度汇总。`conditional_k4_all_fits` 是所有 K=4 条件拟合的 stress-test drift；`production_anchor_matched_k4` 将扰动后的 K=4 通道匹配回 production K=4 四个锚定通道。

物理响应审核只保留两项 caveat：late-bias 与 high-retarding-voltage valley-depth。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 如果你是 AI agent

请按以下流程快速、稳定地复现实验，不要猜测目录结构：

1. 先阅读 `README.md`、`README.zh-CN.md`、`docs/reproduction.md` 和 `docs/outputs.md`。
2. 确认 `data/argon/FHdata.xlsx` 存在。做审核复算时先归档旧运行输出，再将正式运行写入 `output/`。
3. 运行功能级 smoke check：

```powershell
python run.py --mode smoke --exclude hpopt --ablation
```

4. smoke 输出只用于确认程序能跑通，不能作为论文结论；正式分析前请删除或忽略 smoke 输出。
5. 将论文主基线写入复算目录：

```powershell
python run.py --mode fullscan --output output
```

6. 运行包含消融和稳健性证据的完整流程：

```powershell
python run.py --mode fullscan --output output --ablation --robustness
```

7. 基于复算主基线运行 forward-prior 和 uncertainty sensitivity：

```powershell
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

8. `output/` 现在就是正式默认输出根；做新旧核对时不要混用已归档旧证据和当前 `output/` 证据。

9. 若需要更快但不含超参调节的复现，可运行：

```powershell
python run.py --mode fullscan --exclude hpopt --ablation
```

10. 结论应只从所选输出根下的 `main/fullscan/decision.json`、`main/fullscan/model_selection_table.csv`、`main/paper_summary.json`、`ablation/ablation_summary.csv`、`robustness/robustness_summary.json` 和 `sensitivity/` CSV/JSON 文件汇总。

11. 写不确定度结论时，production K=4 候选区间使用 `sensitivity/uncertainty/channel_uncertainty_anchor_matched.csv`。`channel_uncertainty_conditional_k4.csv` 及兼容别名 `channel_uncertainty_summary.csv` 只能解释为条件性 stress-test drift，不能当作 production 能级置信区间。

## 审核复算命令

```powershell
python run.py --mode fullscan --output output
python run.py --mode fullscan --output output --ablation --robustness
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

smoke 命令只用于功能检查。除非测试本身需要保留输出，否则 smoke 输出目录用完后应删除。

## 试验设计

输入数据：

- 默认文件：`data/argon/FHdata.xlsx`
- 预期内容：Frank-Hertz 氩实验的电流-电压曲线，包含加速电压、拒止电压、曲线编号和测量电流列。
- 读取方式：代码会解析常见列名；常规 Excel 依赖不可用时，也包含 `.xlsx` 后备读取逻辑。

主基线流程：

- `python run.py --mode fullscan` 会先从实验曲线的振荡结构中构建 forward evidence。
- fullscan 默认开启超参调节；可用 `--exclude hpopt` 跳过。
- 程序在配置的 K 范围内扫描候选能级数。
- 每个候选 K 会在 `output/main/fullscan/` 下写出 per-level metrics、checkpoint、scorecard、scan table 与 selector diagnostics。
- 自动后评估会在 `output/main/` 下写出论文汇总文件。

消融流程：

- `--ablation` 会先运行主基线，再运行消融分析。
- selector 消融会在移除不同 selector 组件后重新计算选择结果。
- no-forward-anchor-gap 条件会关闭 forward anchor priors 后重新训练。
- 消融输出写入 `output/ablation/`。

稳健性流程：

- `--robustness` 在主基线已经生成 sweep table 之后运行。
- selector 扰动在固定 rank-weight 网格下重新计算选择结果，不重新训练。
- leave-one-Vr-out 每次排除一条阻滞电压曲线，复用主基线超参配置重训 K 候选，不在每折重新运行 hyperopt。
- 稳健性输出写入 `output/robustness/`。

敏感性补充实验：

- `--sensitivity` 在已有主 fullscan 时复用主基线；与 `--exclude hpopt` 配合时不重新运行超参调节。
- prior-strength 扫描使用 `0`、`0.25`、`0.5`、`1`、`2`、`4` 六档 forward-anchor 强度，并在每档扫描 K=1..8。`1x` 严格定义为 `main/fullscan/config_used.json` 中记录的实际 production prior 权重，其他倍率直接缩放这些记录权重。
- 主流程会输出 `main/k_selected_full/prediction_points.csv`，字段为 `curve_id,Vr,Va,observed,predicted,residual`。residual bootstrap 只从这些 selected-model 残差采样；若该表不存在，sensitivity 会直接失败并提示先运行主流程。
- uncertainty 扫描包含 seed-dependent 初始化扰动、残差 bootstrap、曲线级噪声扰动和 peak-window-radius 扰动，同时保留 K=1..8 scorecard。
- `channel_uncertainty_conditional_k4.csv` 汇总所有 K=4 条件拟合，是 stress-test drift 表。
- `channel_uncertainty_anchor_matched.csv` 将扰动 K=4 拟合匹配回 production K=4 四个锚定通道，是 production K=4 不确定度表述的候选依据。
- 敏感性输出写入 `output/sensitivity/`。

设备和路径控制：

```powershell
python run.py --mode fullscan --input data/argon/FHdata.xlsx --output output --device cpu
```

`--device` 支持 `cpu`、`cuda` 或 `auto`。项目默认使用 CPU 调度；`auto` 也按 CPU 处理，只有显式传入 `--device cuda` 时才会使用 CUDA。

## 数据分析方式

主结论文件：

- `output/main/fullscan/decision.json`：最终选择的 K 与决策诊断。
- `output/main/fullscan/model_selection_table.csv`：用于模型选择的候选 K 评分表。
- `output/main/fullscan/scan_summary.csv`：按候选 K 汇总拟合、保留的 seed-summary diagnostic、结构和物理响应指标。
- `output/main/k_selected_full/prediction_points.csv`：selected model 的逐点预测和残差，供 residual-bootstrap sensitivity 使用。
- `output/main/paper_summary.json`：面向论文写作的紧凑结论摘要。

结构分析：

- `structure_metrics.csv` 汇总 peak-valley 保真度和 flatline guard。
- `peak_valley_segments.csv`、`curve_structure_summary.csv` 与 `class_structure_summary.csv` 提供分段、曲线和类别级诊断。

物理响应分析：

- `vr_physical_response.csv` 包含逐曲线物理响应诊断。
- `vr_physical_response.json` 保存物理响应汇总和分组结果。
- 主文中报告的物理 caveat 只使用 late-bias 与 high-retarding-voltage valley-depth。

消融分析：

- `output/ablation/ablation_summary.csv` 比较主基线、no-forward-anchor-gap retrain 和 selector-only 变体。
- `output/ablation/selector_ablation_decision.json` 保存各消融组的详细 selector 决策。
- `output/ablation/ablation_report.md` 是可直接阅读的消融报告。

稳健性分析：

- `output/robustness/selector_weight_perturbation.csv` 报告每个 rank-weight 扰动情景及其 selected K。
- `output/robustness/selector_weight_perturbation_summary.csv` 汇总扰动情景下的 selected-K 分布。
- `output/robustness/leave_one_vr_out_summary.csv` 报告每个留出阻滞电压曲线对应的 selected K 和关键指标。
- `output/robustness/robustness_summary.json` 是面向论文写作的紧凑稳健性摘要。

论文写作时应引用生成的 JSON/CSV 表，而不是中间 checkpoint。已提交的 `source_data_package/` 是当前手稿对应的整理归档；该目录之外新生成的 checkpoint 和运行输出仍由 `.gitignore` 排除。

## Source Data Package

`source_data_package/` 目录保存当前手稿的 source-data package：

- `manuscript_source_tables/`：29 个 CSV/JSON 文件，用于支撑手稿图、模型选择表、物理响应审核、消融讨论、稳健性检查和修复后的 sensitivity/uncertainty 检查。
- `figures/main/` 与 `figures/supplementary/`：71 个生成图像文件，包括 R/GGPlot2 sensitivity 的 PNG/PDF/SVG/TIFF 导出。
- `run_records/k1_full/` 与 `run_records/k_selected_full/`：16 个 K=1 和 selected K=4 的保留运行记录文件，包括 metrics、parameters、prediction points、logs、scorecards、status files 和 checkpoints。
- `output_results/`：从 `output/` 精选出的 decision table、summary、prediction residual points、robustness 输出和 sensitivity 输出，用于支撑手稿结论。
- `FILE_INDEX.csv`、`SHA256SUMS.txt`、`validation_report.json` 与 `source_data_package_manifest.md`：记录包内路径、源路径、文件大小、SHA256 校验值、校验结果和可读清单。

该数据包可从已有正式输出根重建：

```powershell
$env:SUBLEVEL_OUTPUT='output'
python scripts/build_source_data_package.py
```

若 manuscript visualization 的 source table 或 figure 目录位于仓库外部，请通过 `SUBLEVEL_SOURCE_TABLES` 和 `SUBLEVEL_FIGURES` 显式指定。

当前数据包包含 309 个文件。校验已通过 29 个源表、71 个图件、`prior_strength=1x selected_k=4`、四个 production anchor energy，以及 residual-bootstrap scale check。

## 常用命令

smoke check：

```powershell
python run.py --mode smoke --exclude hpopt
python run.py --mode smoke --exclude hpopt --ablation
```

完整主基线：

```powershell
python run.py --mode fullscan
```

跳过超参调节：

```powershell
python run.py --mode fullscan --exclude hpopt
```

主基线加消融：

```powershell
python run.py --mode fullscan --ablation
```

主基线加消融与稳健性分析：

```powershell
python run.py --mode fullscan --ablation --robustness
```

forward-prior 与通道不确定度补充实验：

```powershell
python run.py --mode fullscan --exclude hpopt --sensitivity --device cpu
```

## 输出文件

主基线证据：

- `output/main/fullscan/decision.json`
- `output/main/fullscan/model_selection_table.csv`
- `output/main/fullscan/scan_summary.csv`
- `output/main/fullscan/vr_physical_response.csv`
- `output/main/paper_summary.json`
- `output/main/paper_summary.md`

消融基线证据：

- `output/ablation/ablation_summary.csv`
- `output/ablation/ablation_report.md`
- `output/ablation/selector_ablation_decision.json`

稳健性基线证据：

- `output/robustness/robustness_summary.json`
- `output/robustness/selector_weight_perturbation.csv`
- `output/robustness/selector_weight_perturbation_summary.csv`
- `output/robustness/leave_one_vr_out_summary.csv`

敏感性补充实验证据：

- `output/sensitivity/prior_strength/prior_strength_selection.csv`
- `output/sensitivity/prior_strength/prior_strength_channel_drift.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_samples.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_conditional_k4.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_anchor_matched.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_summary.csv`
- `output/sensitivity/uncertainty/uncertainty_selection_summary.csv`
- `output/sensitivity/sensitivity_summary.json`

## 测试

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest
```

smoke 测试会在仓库内创建临时输出目录，并在测试结束前删除。

