# SubLevel Detect 中文说明

[English README](README.md)

这是用于论文复现的 Frank-Hertz 氩原子亚能级检测源码项目。

本仓库在常规复现层面采用“代码优先”策略：仓库包含源码、输入数据表、测试与说明文档；新的运行输出仍写入 `outputs/`，不作为普通源码变更提交。为便于手稿核验，当前版本单独提交了整理后的 `source_data_package/`，其中包含当前论文草稿使用的图像资产、manuscript-facing CSV/JSON 表、保留的 K=1/K=4 运行记录和校验清单。

## 项目复现内容

代码拟合并评估多能级 Frank-Hertz 氩模型。正式复现流程包含两条平行基线：

- 主基线：构建正向证据、可选超参调节、候选 K 扫描训练、自动后评估。
- 消融基线：selector-only 消融，以及关闭 forward anchor gap 的 retrain，用于检验最终选择对正向锚点的依赖程度。
- 稳健性基线：selector 权重扰动，以及固定主基线超参后的 leave-one-retarding-voltage-out 重训。

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
2. 确认 `data/argon/FHdata.xlsx` 存在，并确认没有复用旧的 `outputs/` 目录。
3. 运行功能级 smoke check：

```powershell
python run.py --mode smoke --exclude hpopt --ablation
```

4. smoke 输出只用于确认程序能跑通，不能作为论文结论；正式分析前请删除或忽略 smoke 输出。
5. 运行论文主基线：

```powershell
python run.py --mode fullscan
```

6. 运行包含消融证据的完整流程：

```powershell
python run.py --mode fullscan --ablation
```

7. 运行包含消融和稳健性证据的完整流程：

```powershell
python run.py --mode fullscan --ablation --robustness
```

8. 若需要更快但不含超参调节的复现，可运行：

```powershell
python run.py --mode fullscan --exclude hpopt --ablation
```

9. 论文结论应只从 `outputs/main/fullscan/decision.json`、`outputs/main/fullscan/model_selection_table.csv`、`outputs/main/paper_summary.json`、`outputs/ablation/ablation_summary.csv` 和 `outputs/robustness/robustness_summary.json` 汇总。

## 试验设计

输入数据：

- 默认文件：`data/argon/FHdata.xlsx`
- 预期内容：Frank-Hertz 氩实验的电流-电压曲线，包含加速电压、拒止电压、曲线编号和测量电流列。
- 读取方式：代码会解析常见列名；常规 Excel 依赖不可用时，也包含 `.xlsx` 后备读取逻辑。

主基线流程：

- `python run.py --mode fullscan` 会先从实验曲线的振荡结构中构建 forward evidence。
- fullscan 默认开启超参调节；可用 `--exclude hpopt` 跳过。
- 程序在配置的 K 范围内扫描候选能级数。
- 每个候选 K 会在 `outputs/main/fullscan/` 下写出 per-level metrics、checkpoint、scorecard、scan table 与 selector diagnostics。
- 自动后评估会在 `outputs/main/` 下写出论文汇总文件。

消融流程：

- `--ablation` 会先运行主基线，再运行消融分析。
- selector 消融会在移除不同 selector 组件后重新计算选择结果。
- no-forward-anchor-gap 条件会关闭 forward anchor priors 后重新训练。
- 消融输出写入 `outputs/ablation/`。

稳健性流程：

- `--robustness` 在主基线已经生成 sweep table 之后运行。
- selector 扰动在固定 rank-weight 网格下重新计算选择结果，不重新训练。
- leave-one-Vr-out 每次排除一条阻滞电压曲线，复用主基线超参配置重训 K 候选，不在每折重新运行 hyperopt。
- 稳健性输出写入 `outputs/robustness/`。

设备和路径控制：

```powershell
python run.py --mode fullscan --input data/argon/FHdata.xlsx --output outputs --device cpu
```

`--device` 支持 `cpu`、`cuda` 或 `auto`。项目默认使用 CPU 调度；`auto` 也按 CPU 处理，只有显式传入 `--device cuda` 时才会使用 CUDA。

## 数据分析方式

主结论文件：

- `outputs/main/fullscan/decision.json`：最终选择的 K 与决策诊断。
- `outputs/main/fullscan/model_selection_table.csv`：用于模型选择的候选 K 评分表。
- `outputs/main/fullscan/scan_summary.csv`：按候选 K 汇总拟合、保留的 seed-summary diagnostic、结构和物理响应指标。
- `outputs/main/paper_summary.json`：面向论文写作的紧凑结论摘要。

结构分析：

- `structure_metrics.csv` 汇总 peak-valley 保真度和 flatline guard。
- `peak_valley_segments.csv`、`curve_structure_summary.csv` 与 `class_structure_summary.csv` 提供分段、曲线和类别级诊断。

物理响应分析：

- `vr_physical_response.csv` 包含逐曲线物理响应诊断。
- `vr_physical_response.json` 保存物理响应汇总和分组结果。
- 主文中报告的物理 caveat 只使用 late-bias 与 high-retarding-voltage valley-depth。

消融分析：

- `outputs/ablation/ablation_summary.csv` 比较主基线、no-forward-anchor-gap retrain 和 selector-only 变体。
- `outputs/ablation/selector_ablation_decision.json` 保存各消融组的详细 selector 决策。
- `outputs/ablation/ablation_report.md` 是可直接阅读的消融报告。

稳健性分析：

- `outputs/robustness/selector_weight_perturbation.csv` 报告每个 rank-weight 扰动情景及其 selected K。
- `outputs/robustness/selector_weight_perturbation_summary.csv` 汇总扰动情景下的 selected-K 分布。
- `outputs/robustness/leave_one_vr_out_summary.csv` 报告每个留出阻滞电压曲线对应的 selected K 和关键指标。
- `outputs/robustness/robustness_summary.json` 是面向论文写作的紧凑稳健性摘要。

论文写作时应引用生成的 JSON/CSV 表，而不是中间 checkpoint。已提交的 `source_data_package/` 是当前手稿对应的整理归档；该目录之外新生成的 checkpoint 和运行输出仍由 `.gitignore` 排除。

## Source Data Package

`source_data_package/` 目录保存当前手稿的 source-data package：

- `manuscript_source_tables/`：19 个 CSV/JSON 文件，用于支撑手稿图、模型选择表、物理响应审核、消融讨论和稳健性检查。
- `figures/main/` 与 `figures/supplementary/`：55 个生成图像文件，格式包括 PDF/PNG/SVG 或 PDF/PNG。
- `run_records/k1_full/` 与 `run_records/k_selected_full/`：14 个 K=1 和 selected K=4 的保留运行记录文件，包括 metrics、parameters、logs、scorecards、status files 和 checkpoints。
- `FILE_INDEX.csv`、`SHA256SUMS.txt` 与 `validation_report.json`：记录包内路径、原始源路径、文件大小、SHA256 校验值和校验结果。

该数据包由 `ESSAY/ajp_argon_sublevel_manuscript/source_data_package_manifest.md` 生成；88 个 manifest payload 文件的大小和 SHA256 均与清单一致，19 个 manuscript-facing CSV/JSON 表均已通过解析检查。

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

## 输出文件

主基线证据：

- `outputs/main/fullscan/decision.json`
- `outputs/main/fullscan/model_selection_table.csv`
- `outputs/main/fullscan/scan_summary.csv`
- `outputs/main/fullscan/vr_physical_response.csv`
- `outputs/main/paper_summary.json`
- `outputs/main/paper_summary.md`

消融基线证据：

- `outputs/ablation/ablation_summary.csv`
- `outputs/ablation/ablation_report.md`
- `outputs/ablation/selector_ablation_decision.json`

稳健性基线证据：

- `outputs/robustness/robustness_summary.json`
- `outputs/robustness/selector_weight_perturbation.csv`
- `outputs/robustness/selector_weight_perturbation_summary.csv`
- `outputs/robustness/leave_one_vr_out_summary.csv`

## 测试

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest
```

smoke 测试会在仓库内创建临时输出目录，并在测试结束前删除。
