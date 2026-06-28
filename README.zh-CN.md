# SubLevel Detect 中文说明

[English README](README.md)

这是用于论文复现的 Frank-Hertz 氩原子亚能级检测源码项目。

本仓库采用“代码优先”策略：仓库只包含源码、输入数据表、测试与说明文档；训练 checkpoint 和生成的分析产物不提交。运行流程后，所有结果会重新生成到 `outputs/`。

## 项目复现内容

代码拟合并评估多能级 Frank-Hertz 氩模型。正式复现流程包含两条平行基线：

- 主基线：构建正向证据、可选超参调节、候选 K 扫描训练、自动后评估。
- 消融基线：selector-only 消融，以及关闭 forward anchor gap 的 retrain，用于检验最终选择对正向锚点的依赖程度。

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

7. 若需要更快但不含超参调节的复现，可运行：

```powershell
python run.py --mode fullscan --exclude hpopt --ablation
```

8. 论文结论应只从 `outputs/main/fullscan/decision.json`、`outputs/main/fullscan/model_selection_table.csv`、`outputs/main/paper_summary.json` 和 `outputs/ablation/ablation_summary.csv` 汇总。

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

设备和路径控制：

```powershell
python run.py --mode fullscan --input data/argon/FHdata.xlsx --output outputs --device cpu
```

`--device` 支持 `cpu`、`cuda` 或 `auto`。

## 数据分析方式

主结论文件：

- `outputs/main/fullscan/decision.json`：最终选择的 K 与决策诊断。
- `outputs/main/fullscan/model_selection_table.csv`：用于模型选择的候选 K 评分表。
- `outputs/main/fullscan/scan_summary.csv`：按候选 K 汇总拟合、交叉验证、结构和物理响应指标。
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

论文写作时应引用生成的 JSON/CSV 表，而不是中间 checkpoint。checkpoint 是运行时产物，已通过 `.gitignore` 排除。

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

## 测试

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest
```

smoke 测试会在仓库内创建临时输出目录，并在测试结束前删除。
