# Financial Performance Prediction：通用 AI 编程代理全流程执行手册

> **用途**：将本文件复制到项目根目录 `financial-performance-prediction/`。使用 Codex 时建议命名为 `AGENTS.md`；使用其他 AI 编程代理时可命名为 `MASTER_WORKFLOW.md`，并要求代理完整阅读后执行。  
> **目标**：由一个 AI 编程代理独立完成实现、运行、审计、修订、自主优化和交付检查，最终生成满足老师要求的 Jupyter Notebook、分析报告和 `submission.csv`。  
> **原则**：追求完整、严谨、可复现和格式统一；不得伪造实验结果，不得以排行榜分数替代交叉验证，不得手工修改报告中的数值。  
> **默认运行环境**：使用本机已有 Conda 环境 `QuantEnv`。禁止默认创建 `.venv`，禁止误用 `base` 环境。  
> **重要说明**：没有任何工作流可以承诺“绝对完美”，但本文把“完美”拆成可验证的硬性验收门槛。只有所有门槛通过后，才允许交付。

---

## 使用方式

推荐将本文件直接复制到项目根目录。

```text
financial-performance-prediction/
├── AGENTS.md
├── train.csv
├── test.csv
├── sample_submission.csv
├── data_dictionary.txt
└── 说明.docx
```

使用 Codex 时，文件名优先采用 `AGENTS.md`。使用其他 AI 编程代理时，也可以保留 `AGENTS.md`，或者命名为 `MASTER_WORKFLOW.md` 后在首条提示中明确要求完整阅读。本文末尾提供一条平台无关的主提示词。

---

## 0. 一次性总指令：先读完再开始

你是该项目的执行代理。项目根目录为：

```text
financial-performance-prediction/
```

项目根目录中已经存在以下 5 个原始文件：

```text
train.csv
test.csv
sample_submission.csv
data_dictionary.txt
说明.docx
```

不要覆盖、移动或修改原始文件。先建立版本控制、冻结输入文件摘要、读取任务说明和数据字典，再开始任何分析。

最终必须交付：

```text
deliverables/
├── financial_performance_prediction_final.ipynb
├── financial_performance_prediction_report.docx
├── submission.csv
└── README_delivery.md
```

同时保留可复现工程：

```text
src/
scripts/
configs/
tests/
results/
figures/
notebooks/
docs/
```

老师的明确要求是：最终提交 Jupyter Notebook、分析报告和 `submission.csv`；Notebook 图表结果应与报告一致，内容包含数据探索、分析过程、机器学习训练过程与结果，以及交叉验证。执行时必须围绕这些要求组织内容。

---

# 1. 任务理解与硬约束

## 1.1 任务定义

这是一个 **9 个目标变量分别预测的监督回归任务**。每一行代表一家上市公司。已知过去 10 个季度 `Q1` 至 `Q10` 的财务指标和公司元数据，需要预测最新季度 `Q0` 的 9 项财务指标。

时间顺序：

```text
Q10 → Q9 → ... → Q2 → Q1 → Q0
较早                         最新季度
```

目标变量：

```text
Q0_TOTAL_ASSETS
Q0_TOTAL_LIABILITIES
Q0_TOTAL_STOCKHOLDERS_EQUITY
Q0_GROSS_PROFIT
Q0_COST_OF_REVENUES
Q0_REVENUES
Q0_OPERATING_INCOME
Q0_OPERATING_EXPENSES
Q0_EBITDA
```

评价指标：

```text
先分别计算 9 个目标的 R²，再取算术平均值。
```

`sample_submission.csv` 的列顺序必须原样保留。不得按照自己的理解重新排序。

## 1.2 已知数据规模

在正式运行前必须再次由脚本验证，预期为：

```text
train.csv              1624 行 × 212 列
test.csv                406 行 × 203 列
sample_submission.csv   406 行 × 10 列
```

测试集特征可拆为：

```text
1 个 Id
31 个公司元数据字段
1 个 Q0_fiscal_year_end
10 个季度 × 17 个历史字段
= 203 列
```

训练集比测试集多 9 个 Q0 目标变量：

```text
203 + 9 = 212 列
```

## 1.3 不可违反的建模约束

必须遵守：

1. `Id` 只能作为记录标识，不得作为模型特征。
2. 9 个目标必须分别记录 R²，最终再求平均。
3. 每个实验必须保存 OOF（out-of-fold）预测和分折结果。
4. 交叉验证不得让完全重复的历史特征记录同时出现在训练折和验证折。
5. 滑动窗口扩增时，同一家公司的全部窗口必须属于同一折。
6. 不得将 `Q0_` 目标列混入特征。
7. `sample_submission.csv` 只允许复制列顺序和 Id，不允许从中获取预测信息。
8. 任何后处理、融合、权重选择和超参数选择都必须依据 OOF 结果，不得依据测试集或排行榜反复试错。
9. 报告中的表格、图表和数值必须由脚本读取 `results/` 与 `figures/` 自动生成，不得手工抄写。
10. 原始文件保持只读；全部派生文件写入工程目录的子文件夹。

---

# 2. 网上现有项目：如何组织和借鉴

以下项目与当前任务直接相关。只借鉴思路、结构和公开基线，不复制整份代码。需要在报告中说明：公开 Notebook 用于建立基准和启发实验设计，最终结果来自本项目独立复现、交叉验证和迭代。

## 2.1 官方入口

| 类型 | 项目 | 链接 | 用途 |
|---|---|---|---|
| 官方比赛页 | Financial Performance Prediction | https://www.kaggle.com/competitions/financial-performance-prediction | 核对任务定义、文件结构和评价方式 |
| 官方数据说明 | 当前项目中的 `data_dictionary.txt` 与 `说明.docx` | 本地文件 | 核对季度映射、字段定义、老师交付要求 |

## 2.2 直接相关的 Kaggle Notebook

| 类型 | 项目 | 链接 | 可借鉴点 | 不可直接照搬的原因 |
|---|---|---|---|---|
| 时间序列基线 | `use only test data, time series, top 10 score` | https://www.kaggle.com/code/georgiikirsanov/use-only-test-data-time-series-top-10-score | 强调短序列外推；提醒部分指标只靠历史序列也可取得较好效果 | 只使用测试历史数据的路线难以完整展示监督学习训练与交叉验证，不适合作为课程报告主体 |
| 缺失值 + XGBoost 基线 | `Quick missing data imputation and model` | https://www.kaggle.com/code/ryazanoff/quick-missing-data-imputation-and-model | 快速建立可运行模型；参考缺失值处理、逐目标训练和提交文件生成 | 需要补充更严格的交叉验证、特征工程、重复样本处理和报告叙事 |
| 常规模型示例 | `Financial data prediction` | https://www.kaggle.com/code/emish8/financial-data-prediction | 核对字段、代码组织和 submission 生成 | 仅作为结构参考；不得替代独立实验 |

## 2.3 官方技术文档

| 主题 | 链接 | 本项目中的用途 |
|---|---|---|
| `r2_score` | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html | 明确 R² 最佳值为 1，也可能为负数；实现逐目标和平均 R² |
| `GroupKFold` | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html | 防止重复样本或同公司窗口跨折泄漏 |
| `HistGradientBoostingRegressor` | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html | 建立可处理 NaN 的 scikit-learn 树模型基线 |
| CatBoost 分类变量 | https://catboost.ai/docs/en/features/categorical-features | CatBoost 可原生处理分类特征；不要预先 One-Hot 后再喂给 CatBoost |
| CatBoost 数值与 NaN | https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic | 数值变量可以包含 NaN；分类变量需统一转字符串并填补 `"Missing"` |
| LightGBM 缺失值 | https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html | 可选对照模型；原生处理 NaN |
| XGBoost 分类数据 | https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html | 可选对照模型；需要正确设置 pandas category 或编码 |
| nbconvert 执行 Notebook | https://nbconvert.readthedocs.io/en/v7.13.0/execute_api.html | 最终清空并从头执行 Notebook，确保可复现 |

## 2.4 对网上项目的结论

最终路线不能只选“传统机器学习”或“时间序列”其中之一。最合理的是：

```text
EDA 与数据审计
→ 泄漏安全的交叉验证
→ 最近季度 / 季节性 / 趋势基线
→ 财务特征工程
→ CatBoost 为主的逐目标模型
→ 直接预测与残差预测对比
→ OOF 驱动的逐目标融合
→ 会计一致性后处理实验
→ 最终全量训练
→ submission、Notebook 和 Word 报告统一生成
```

---

# 3. 单代理执行模式：实现、审计与修订闭环

## 3.1 核心原则

本项目默认只使用 **一个 AI 编程代理**。该代理必须在不同阶段切换角色，而不是完成代码后立即宣布结束。

统一执行顺序：

```text
实现阶段
→ 运行阶段
→ 自我审计阶段
→ 修订阶段
→ 重新运行阶段
→ 交付验收阶段
```

每一个阶段都必须留下可检查的文件、日志或测试结果。禁止以“看起来已经完成”替代实际运行。

## 3.2 角色切换

同一个 AI 在执行过程中依次承担以下角色：

| 角色 | 主要职责 | 输出 |
|---|---|---|
| 项目经理 | 读取要求，拆分阶段，维护状态 | `docs/AGENT_STATE.md` |
| 数据分析师 | 审计数据、生成 EDA、检查财务关系 | `results/data_audit.json`、`figures/` |
| 机器学习工程师 | 编写特征、模型、交叉验证与融合代码 | `src/`、`scripts/`、`configs/` |
| 实验研究员 | 每轮提出一个假设，运行实验并记录结果 | `results/experiment_log.csv` |
| 代码审查员 | 检查数据泄漏、重复代码、异常处理和可复现性 | `docs/ISSUES.md` |
| 学术写作助手 | 根据统一结果表生成 Notebook 与报告 | `notebooks/`、`deliverables/` |
| 交付验收员 | 运行测试、重跑 Notebook、检查 Word 和 CSV | `results/delivery_validation.json` |

角色可以由同一个 AI 完成，但必须按阶段执行。进入下一阶段前，要先完成当前阶段的验收条件。

## 3.3 自我审计规则

每次准备宣布某一阶段完成前，必须先进行一次“反方审查”：

```text
假设当前实现存在错误。
逐项寻找：
1. 数据泄漏；
2. 指标计算错误；
3. 训练与验证预处理不一致；
4. 测试集信息被用于选模；
5. 重复样本跨折；
6. Notebook 与报告数值不一致；
7. submission 列顺序错误；
8. 图表或报告格式问题；
9. 代码无法从空环境复现；
10. 结论超出实验支持范围。
```

发现问题后写入：

```text
docs/ISSUES.md
```

并按以下级别分类：

```text
BLOCKER：不修复就禁止交付
MAJOR：显著影响质量，应优先修复
MINOR：格式、表达或可维护性优化
```

## 3.4 单代理状态文件

始终维护：

```text
docs/AGENT_STATE.md
docs/ISSUES.md
docs/DECISIONS.md
results/experiment_log.csv
configs/best_config.json
```

`docs/AGENT_STATE.md` 至少记录：

```text
当前阶段
最近一次成功命令
当前最佳 OOF 平均 R²
当前最佳实验编号
尚未解决的 BLOCKER
下一步唯一动作
最近一次 Git commit
```

## 3.5 Codex 与其他 AI 的兼容方式

本文件内容本身保持平台中立，不依赖任何特定供应商。

- 使用 Codex 时，将本文件保存为项目根目录的 `AGENTS.md`。Codex 会在工作前读取该文件。
- 使用其他 AI 编程代理时，将本文件保存为 `MASTER_WORKFLOW.md`，并在首次提示中要求代理完整阅读后执行。
- 若代理支持子代理、并行任务或计划模式，可以使用，但不是完成任务的必要条件。
- 无论使用什么 AI，都必须遵守本文件中的测试、审计和交付门槛。

# 4. 本地 Skill 使用策略

项目开始时，不要假设 Skill 一定安装在固定目录。先搜索 `SKILL.md`，再阅读所需技能的说明。

推荐命令：

```bash
find ~ -path '*/skills/*/SKILL.md' -o -path '*/skill/*/SKILL.md' 2>/dev/null | sort
```

如果存在 `find-skills`，优先调用它定位技能。只使用已安装技能；缺失技能要记录并采用普通脚本替代。

## 4.1 必用技能

| 阶段 | Skill | 用法 |
|---|---|---|
| 数据理解与图表 | `data-analysis` | 读取数据、EDA、可视化、统计审计 |
| 财务逻辑 | `quantitative-research` | 检查财务比率、序列特征和建模设计 |
| 工程质量 | `code-review-and-quality` | 审查模块化、异常处理、重复代码和可复现性 |
| 单元测试 | `python-testing-patterns` | 为 schema、特征、metric、submission 编写 pytest |
| 报告写作 | `academic-writing`、`paper-writing` | 组织报告结构、方法说明和结果讨论 |
| 审稿 | `paper-audit`、`paper-verification`、`peer-review` | 检查方法严谨性、数值一致性、结论证据 |
| 逻辑审查 | `critical-analysis` | 识别数据泄漏、过度推断和不充分论证 |
| Word 生成 | `word-docx` | 生成并检查 `.docx` 报告 |
| Markdown 转换 | `markdown-converter` | 将中间报告 Markdown 转换或辅助构建 Word |
| PDF 预览检查 | `pdf` | 将报告转 PDF 后检查分页、图表和空白页 |
| 闭环迭代 | `diagnose`、`evaluate`、`iterate` | 发现问题、打分、修订、复核 |

## 4.2 可选技能

| Skill | 适用条件 |
|---|---|
| `financial-data-collector` | 仅在做 SEC / yfinance 外部拓展实验时使用；主模型不依赖外部数据 |
| `research-manager` | 迭代较多时维护实验队列 |
| `research-synthesis`、`synthesize` | 合并多个实验发现并撰写结论 |
| `humanizer-zh-academic`、`humanizer-zh` | 只用于提升中文自然度和减少空话，不得用于规避检测或伪装作者身份 |
| `brokerage-report` | 仅用于改善财务分析叙事，不要把课程报告写成券商研报 |

---

# 5. 推荐工程目录

初始化以下结构：

```text
financial-performance-prediction/
├── train.csv
├── test.csv
├── sample_submission.csv
├── data_dictionary.txt
├── 说明.docx
├── MASTER_WORKFLOW.md
├── README.md
├── environment-quantenv-history.yml
├── environment-quantenv-lock.yml
├── conda-explicit-spec.txt
├── requirements-pip-lock.txt
├── .gitignore
├── configs/
│   ├── baseline.yaml
│   ├── catboost_direct.yaml
│   ├── catboost_residual.yaml
│   └── best_config.json
├── docs/
│   ├── AGENT_STATE.md
│   ├── ISSUES.md
│   ├── DECISIONS.md
│   ├── REPORT_DRAFT.md
│   └── sources.md
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── io_utils.py
│   ├── validation.py
│   ├── cv.py
│   ├── feature_engineering.py
│   ├── baselines.py
│   ├── models.py
│   ├── blending.py
│   ├── accounting_checks.py
│   ├── plots.py
│   ├── reporting.py
│   └── pipeline.py
├── scripts/
│   ├── bootstrap.py
│   ├── audit_data.py
│   ├── run_baselines.py
│   ├── run_experiment.py
│   ├── optimize.py
│   ├── train_final.py
│   ├── build_notebook.py
│   ├── build_report.py
│   ├── validate_delivery.py
│   └── package_delivery.py
├── tests/
│   ├── test_schema.py
│   ├── test_metric.py
│   ├── test_features.py
│   ├── test_cv_no_leakage.py
│   ├── test_submission.py
│   └── test_report_assets.py
├── notebooks/
│   ├── 00_scratch.ipynb
│   └── financial_performance_prediction_final.ipynb
├── results/
│   ├── input_manifest.json
│   ├── environment_audit.json
│   ├── data_audit.json
│   ├── experiment_log.csv
│   ├── cv_scores/
│   ├── oof/
│   ├── predictions/
│   ├── models/
│   └── tables/
├── figures/
└── deliverables/
```

`.gitignore` 至少包含：

```text
.venv/
.conda/
__pycache__/
.ipynb_checkpoints/
*.pyc
results/models/
*.log
```

原始 CSV、说明文档和最终交付物不要忽略。

---

# 6. 环境初始化：固定使用本机 Conda 环境 `QuantEnv`

## 6.1 环境策略

本项目默认使用用户本机已有的 Conda 环境：

```text
QuantEnv
```

禁止默认创建 `.venv`，禁止误用 Conda 的 `base` 环境。执行代理应先审计 `QuantEnv`，再安装必要的缺失依赖。

为什么这样设计：

- 用户已经长期使用 `QuantEnv`；
- 避免重复创建环境；
- 保留用户熟悉的量化研究工具链；
- 通过导出快照和依赖锁文件，仍然保证项目可复现。

同时要注意：`QuantEnv` 可能服务于其他量化项目。因此，在安装或升级依赖前必须先导出环境快照。不得无条件执行大规模升级，不得运行 `pip install --upgrade` 批量升级全部包。

## 6.2 首次检查

在项目根目录执行：

```bash
conda env list
conda run -n QuantEnv python --version
conda run -n QuantEnv python -c "import sys; print(sys.executable)"
conda run -n QuantEnv python -m pip --version
```

其中，`conda env list` 用于确认环境存在；`conda run -n QuantEnv ...` 用于确保命令确实在指定环境中运行。

执行代理还应生成：

```text
results/environment_audit.json
```

至少记录：

```text
conda_environment_name
python_executable
python_version
platform
conda_version
pip_version
required_packages
installed_versions
missing_packages
timestamp
```

## 6.3 依赖检查脚本

建立：

```text
scripts/check_environment.py
```

必需包：

```text
pandas
numpy
scipy
matplotlib
scikit-learn
catboost
xgboost
lightgbm
optuna
joblib
pyyaml
jupyter
nbconvert
nbformat
ipykernel
python-docx
openpyxl
pytest
pytest-cov
```

脚本只检查，不擅自升级。若缺少依赖，再最小化安装：

```bash
conda run -n QuantEnv python -m pip install \
    pandas numpy scipy matplotlib scikit-learn \
    catboost xgboost lightgbm optuna joblib pyyaml \
    jupyter nbconvert nbformat ipykernel \
    python-docx openpyxl pytest pytest-cov
```

如果环境中已有包，不要批量升级。若出现依赖冲突，停止并记录到：

```text
docs/ISSUES.md
```

## 6.4 冻结环境快照

首次安装依赖前先导出一次，安装后再导出一次。至少保存：

```bash
# 便于跨平台理解核心依赖
conda env export -n QuantEnv --from-history > environment-quantenv-history.yml

# 完整 Conda 环境快照
conda env export -n QuantEnv > environment-quantenv-lock.yml

# 当前平台可精确复现的显式包清单
conda list -n QuantEnv --explicit > conda-explicit-spec.txt

# 记录 pip 安装包
conda run -n QuantEnv python -m pip freeze > requirements-pip-lock.txt
```

若 Conda 版本支持新版导出命令，也可额外执行：

```bash
conda export -n QuantEnv --format=environment-yaml > environment-quantenv-export.yml
```

## 6.5 Jupyter 内核

为避免 Notebook 误用其他 Python，在 `QuantEnv` 中注册内核：

```bash
conda run -n QuantEnv python -m ipykernel install \
  --user \
  --name QuantEnv \
  --display-name "Python (QuantEnv)"
```

最终 Notebook 的 kernelspec 必须指向：

```text
Python (QuantEnv)
```

## 6.6 推荐运行方式

交互式终端中可以先执行：

```bash
conda activate QuantEnv
```

但为了减少 Codex、脚本或终端会话切换导致的误用环境，自动化命令优先使用：

```bash
conda run -n QuantEnv <command>
```

例如：

```bash
conda run -n QuantEnv python scripts/bootstrap.py
conda run -n QuantEnv python -m pytest -q
```

## 6.7 可选隔离方案

如果用户不希望修改长期使用的 `QuantEnv`，可以克隆环境：

```bash
conda create -n QuantEnv-fpp --clone QuantEnv
```

然后将本文件中的：

```text
QuantEnv
```

统一替换为：

```text
QuantEnv-fpp
```

默认仍使用 `QuantEnv`，除非用户主动选择隔离方案。

## 6.8 初始化 Git

```bash
git init
git add .
git commit -m "chore: freeze raw inputs and QuantEnv snapshot"
```

---


# 7. 第一阶段：输入审计与数据冻结

## 7.1 生成输入清单

`scripts/bootstrap.py` 必须：

1. 计算 5 个原始文件的 SHA256；
2. 记录文件大小、修改时间；
3. 读取 CSV 行列数；
4. 读取 `说明.docx` 和 `data_dictionary.txt`；
5. 输出 `results/input_manifest.json`；
6. 若原始文件被修改，立即停止后续任务。

## 7.2 Schema 检查

`scripts/audit_data.py` 必须验证：

```text
train.shape == (1624, 212)
test.shape == (406, 203)
sample_submission.shape == (406, 10)
```

目标变量必须精确为 9 个。提交列必须精确复制 `sample_submission.columns`。

必须输出：

```text
results/data_audit.json
results/tables/schema_summary.csv
results/tables/missing_rate_by_column.csv
results/tables/missing_rate_by_quarter.csv
results/tables/dtype_summary.csv
results/tables/category_summary.csv
results/tables/duplicate_summary.csv
results/tables/target_summary.csv
```

## 7.3 数据审计问题清单

至少检查：

- 特征整体缺失率；
- 各列缺失率；
- Q1 至 Q10 每季度平均缺失率；
- 分类变量类别数量；
- 训练集和测试集类别差异；
- 极端值和负值；
- 重复记录；
- train / test 的历史特征完全匹配记录；
- 目标变量分布；
- 会计恒等式偏差；
- 元数据与目标的相关性；
- `Id` 是否唯一；
- 是否存在 `inf`、`-inf`；
- 是否存在常量列和近常量列。

## 7.4 重复样本分组

对模型特征生成稳定哈希：

```python
feature_signature = pd.util.hash_pandas_object(
    X.sort_index(axis=1),
    index=False
).astype(str)
```

实际实现要考虑 NaN、字符串列和列顺序，确保同样的特征记录得到同样的组编号。

将完全相同的特征记录放入同一个 `group_id`，用于 `GroupKFold`。

---

# 8. 第二阶段：EDA 与图表

所有图表必须由 `src/plots.py` 统一生成，保存在 `figures/`。Notebook 和 Word 报告只能引用这些文件，不得各画一套。

推荐图表：

| 编号 | 图表 | 文件名 | 用途 |
|---|---|---|---|
| 图 1 | sector 样本数量柱状图 | `fig01_sector_distribution.png` | 展示公司所属板块分布 |
| 图 2 | industry Top 20 样本数量 | `fig02_industry_top20.png` | 展示行业集中度 |
| 图 3 | 缺失率最高的 20 个字段 | `fig03_missing_top20.png` | 展示数据质量 |
| 图 4 | Q1 至 Q10 平均缺失率折线图 | `fig04_missing_by_quarter.png` | 展示越早季度缺失更多的趋势 |
| 图 5 | 9 个目标的 signed-log 分布 | `fig05_target_distributions.png` | 展示偏态和极端值 |
| 图 6 | 历史滞后值与 Q0 目标相关性热力图 | `fig06_lag_correlation_heatmap.png` | 判断 Q1、Q4 等季度的重要性 |
| 图 7 | 会计恒等式偏差分布 | `fig07_accounting_identity_error.png` | 检查数据一致性 |
| 图 8 | 模型平均 R² 对比 | `fig08_model_comparison.png` | 展示模型改进 |
| 图 9 | 各目标 R² 热力图 | `fig09_target_score_heatmap.png` | 识别难预测指标 |
| 图 10 | OOF 真实值与预测值散点图 | `fig10_oof_scatter.png` | 检查拟合 |
| 图 11 | 特征重要性 Top 30 | `fig11_feature_importance.png` | 解释主模型 |
| 图 12 | 残差分布 | `fig12_residual_distribution.png` | 检查误差结构 |

要求：

- 图表标题、坐标轴和图例使用中文或中英结合；
- 统一字体、字号和 DPI；
- 报告中每张图都有编号和简短分析；
- Notebook 和报告引用完全相同的图；
- 图表不能只展示，不解释；
- 若某图信息重复，可删减，但报告中至少保留 8 张关键图。

---

# 9. 第三阶段：先做基线，不要直接调参

实现 `src/baselines.py`。

## 9.1 B0：均值预测

用于确认 metric 实现无误：

```text
对每个目标始终预测训练折均值。
```

验证集 R² 应接近 0，允许因折间分布差异略有变化。

## 9.2 B1：最近季度复制

```text
Q0_TARGET ≈ Q1_TARGET
```

如 `Q1` 缺失，则按 `Q2 → Q3 → ... → Q10` 回退；仍缺失时使用训练折中位数。

## 9.3 B2：季节性复制

```text
Q0_TARGET ≈ Q4_TARGET
```

因为 `Q4` 大致对应上一年度相近季度。缺失时按临近历史值回退。

## 9.4 B3：短趋势外推

使用最近 4 个可用季度对时间索引拟合简单线性趋势，预测下一期。只在每个样本自身的历史值上操作，不使用验证集标签。

## 9.5 B4：逐目标基线融合

针对每个目标，使用 OOF 结果在以下组合中搜索最优权重：

```text
B1
B2
B3
B1 + B2
B1 + B3
B1 + B2 + B3
```

权重网格：

```text
0.0, 0.1, 0.2, ..., 1.0
```

要求权重和为 1。权重选择必须只基于 OOF。

输出：

```text
results/tables/baseline_scores.csv
results/oof/baseline_oof.csv
configs/baseline_blend_weights.json
```

---

# 10. 第四阶段：特征工程

在 `src/feature_engineering.py` 中分组实现，每一组都要有开关，以便做消融实验。

## 10.1 原始历史特征

保留：

```text
Q1 至 Q10 的全部财务指标
Q0_fiscal_year_end
Q1 至 Q10_fiscal_year_end
```

## 10.2 元数据特征

保留测试集中存在的公司元数据：

```text
industry
sector
fullTimeEmployees
auditRisk
boardRisk
compensationRisk
shareHolderRightsRisk
overallRisk
trailingPE
forwardPE
floatShares
sharesOutstanding
trailingEps
forwardEps
targetHighPrice
targetLowPrice
targetMeanPrice
targetMedianPrice
recommendationMean
recommendationKey
numberOfAnalystOpinions
totalCash
totalCashPerShare
ebitda
totalDebt
totalRevenue
revenuePerShare
freeCashflow
operatingCashflow
revenueGrowth
financialCurrency
```

注意：

- `ebitda` 元数据与目标 `Q0_EBITDA` 不是同一列；
- 元数据来自最新状态，允许用于主模型，因为 train 和 test 都提供；
- 但必须做消融实验，展示元数据带来的提升；
- 滑动窗口伪样本中不得使用 Q0 元数据预测历史季度，否则产生未来信息泄漏。

## 10.3 时间序列统计

对每一个历史财务指标构造：

```text
last_value
mean_last_2
mean_last_4
mean_last_8
median_last_4
std_last_4
std_last_8
min_last_4
max_last_4
slope_last_4
slope_last_8
non_missing_count
missing_count
```

## 10.4 环比、同比和变化量

构造：

```text
Q1 - Q2
Q2 - Q3
Q1 - Q5
Q1 / Q2 - 1
Q2 / Q3 - 1
Q1 / Q5 - 1
```

安全除法：

```python
def safe_ratio(a, b, eps=1e-9):
    return a / np.where(np.abs(b) < eps, np.nan, b)
```

对增长率做合理截断，例如按训练折的 0.5% 和 99.5% 分位数缩尾。所有阈值只能在训练折拟合，再应用到验证折。

## 10.5 财务比率

优先构造：

```text
资产负债率 = TOTAL_LIABILITIES / TOTAL_ASSETS
权益比率 = TOTAL_STOCKHOLDERS_EQUITY / TOTAL_ASSETS
流动比率 = TOTAL_CURRENT_ASSETS / TOTAL_CURRENT_LIABILITIES
非流动资产占比 = TOTAL_NONCURRENT_ASSETS / TOTAL_ASSETS
毛利率 = GROSS_PROFIT / REVENUES
营业利润率 = OPERATING_INCOME / REVENUES
EBITDA 利润率 = EBITDA / REVENUES
成本率 = COST_OF_REVENUES / REVENUES
费用率 = OPERATING_EXPENSES / REVENUES
```

对 Q1、Q2、Q4、Q5 构造重点比率，再构造最近期变化。

## 10.6 缺失值特征

构造：

```text
row_missing_count
row_missing_ratio
history_missing_count
metadata_missing_count
quarter_available_count
is_missing_<important_feature>
```

## 10.7 有符号对数特征

财务指标可能有负数，不要对全部变量直接 `log1p(x)`。使用：

```python
signed_log1p = np.sign(x) * np.log1p(np.abs(x))
```

为高偏态数值变量增加 signed-log 版本，同时保留原始值。

---

# 11. 第五阶段：交叉验证设计

## 11.1 主交叉验证

使用：

```python
GroupKFold(n_splits=5)
```

分组依据：

```text
完全相同模型特征的稳定哈希 group_id
```

目的：避免重复记录在训练折和验证折之间泄漏。

保存：

```text
results/cv_scores/<experiment_id>.csv
results/oof/<experiment_id>.csv
```

每个实验必须记录：

```text
experiment_id
timestamp
model_name
feature_set
target_strategy
fold
target
r2
mean_r2
std_r2
runtime_seconds
seed
git_commit
notes
```

## 11.2 可选的重复 KFold 稳定性检查

主模型确定后，再以多个随机种子做稳定性检查。不要在报告中只给最好的一次。

## 11.3 滑动窗口增强的特殊规则

只有完成主路线后才尝试滑动窗口扩增。

可以构造：

```text
Q10...Q2 → 预测 Q1
Q9...Q1  → 预测 Q0
```

但必须满足：

1. 同一家公司的全部窗口属于同一个 group；
2. 预测历史季度的伪样本只能使用当时及之前可见的历史财务特征；
3. 不得使用 Q0 元数据预测 Q1、Q2 等历史季度；
4. 先单独验证是否提升，再决定是否进入最终模型；
5. 若实现复杂或增益不稳定，保留为拓展实验，不进入交付主模型。

---

# 12. 第六阶段：模型路线

## 12.1 M1：Ridge 基线

用途：

- 建立线性机器学习基线；
- 验证人工构造特征是否有效；
- 为报告提供可解释对照。

预处理：

```text
数值：中位数填充 + RobustScaler
分类：Missing 填补 + OneHotEncoder(handle_unknown="ignore")
```

## 12.2 M2：HistGradientBoostingRegressor

用途：

- 建立纯 scikit-learn 非线性树模型；
- 利用其原生 NaN 处理能力；
- 与 CatBoost 对比。

分类特征可采用 One-Hot 或目标安全的 Ordinal 编码。编码器只能在训练折拟合。

## 12.3 M3：CatBoost 直接预测

这是主力候选模型。

策略：

```text
X → 直接预测 Q0_TARGET
```

分类特征：

```text
industry
sector
recommendationKey
financialCurrency
```

处理方式：

```text
分类缺失值统一转换为字符串 "Missing"
数值 NaN 保留给 CatBoost
```

建议初始参数：

```python
{
    "loss_function": "RMSE",
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 8.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False
}
```

每折使用 early stopping。记录 best iteration。

## 12.4 M4：CatBoost 残差预测

这是重点优化。

对每个目标定义：

```text
residual = Q0_TARGET - baseline_prediction
```

优先使用 B4 的逐目标融合基线作为 baseline。模型预测 residual：

```text
final_prediction = baseline_prediction + predicted_residual
```

好处：

- 模型只需学习偏离历史规律的部分；
- 降低公司规模差异带来的难度；
- 更容易与时间序列基线融合。

## 12.5 M5：LightGBM / XGBoost 对照

只作为可选对照，不要无限扩模型数量。

建议原则：

- CatBoost 为主；
- LightGBM 或 XGBoost 二选一作为额外树模型；
- 若 OOF 无明显提升，不进入最终融合；
- 报告重点放在方法逻辑，不要堆砌算法名称。

## 12.6 目标变换实验

可以对每个目标分别测试：

```text
原始值预测
signed_log1p 目标预测后反变换
残差预测
```

接受标准：OOF 平均 R² 或目标 R² 有稳定提升。不得凭直觉选择。

---

# 13. 第七阶段：消融实验

至少完成以下实验表：

| 编号 | 方法 | 特征 | 目的 |
|---|---|---|---|
| B0 | 均值预测 | 无 | 检查 metric |
| B1 | Q1 复制 | 最近一期 | 最近季度基线 |
| B2 | Q4 复制 | 上年同期近似 | 季节性基线 |
| B3 | 线性趋势 | 最近若干季度 | 时间序列基线 |
| B4 | 基线融合 | B1+B2+B3 | 更稳健的规则基线 |
| M1 | Ridge | 原始历史特征 | 线性模型 |
| M2 | HistGradientBoosting | 原始历史特征 | 非线性树基线 |
| M3a | CatBoost | 仅历史特征 | 主模型历史版 |
| M3b | CatBoost | 历史 + 行业分类 | 行业增益 |
| M3c | CatBoost | 历史 + 全部元数据 | 元数据增益 |
| M3d | CatBoost | M3c + 人工特征 | 特征工程增益 |
| M4 | CatBoost 残差模型 | 最佳特征 | 残差建模增益 |
| M5 | 可选 LightGBM / XGBoost | 最佳特征 | 模型差异 |
| M6 | OOF 融合 | B4 + M3d + M4 + 可选 M5 | 最终模型 |
| M7 | 会计一致性调整 | M6 输出 | 可选后处理 |

每个实验必须输出：

```text
9 个目标的 R²
平均 R²
分折标准差
运行时间
配置文件
OOF 文件
```

---

# 14. 第八阶段：OOF 融合与会计一致性

## 14.1 逐目标融合

不要使用统一权重。对每个目标分别在 OOF 上搜索：

```text
baseline
CatBoost direct
CatBoost residual
可选 LightGBM / XGBoost
```

可先用 0.05 步长粗搜，再在最佳附近用 0.01 步长细搜。

目标函数：

```text
最大化该目标 OOF R²
```

保存：

```text
configs/blend_weights.json
results/tables/blend_scores.csv
```

## 14.2 会计恒等式检查

检查：

```text
TOTAL_ASSETS ≈ TOTAL_LIABILITIES + TOTAL_STOCKHOLDERS_EQUITY
GROSS_PROFIT ≈ REVENUES - COST_OF_REVENUES
OPERATING_INCOME ≈ GROSS_PROFIT - OPERATING_EXPENSES
```

只允许把恒等式用于：

- EDA；
- 误差诊断；
- 生成替代预测；
- OOF 驱动的加权融合。

禁止不经验证直接强制修改所有预测。

示例：

```text
gross_profit_final
= alpha × model_gross_profit
+ (1 - alpha) × (pred_revenues - pred_cost_of_revenues)
```

`alpha` 只能由 OOF 搜索得到。

---

# 15. 第九阶段：AI 自主优化闭环

实现 `scripts/optimize.py`，让 AI 可以有边界地迭代，而不是无休止尝试。

## 15.1 每轮只提出一个可检验假设

示例：

```text
H01：加入 Q1/Q5 同比特征可以提高收入类指标。
H02：股东权益更适合 residual 建模。
H03：加入 missing_count 可以提高早期季度缺失较多样本的稳定性。
H04：CatBoost depth=5 比 depth=8 更能抑制小样本过拟合。
```

禁止一轮同时改 10 项内容，否则无法判断提升来自哪里。

## 15.2 每轮固定流程

```text
1. 读取 AGENT_STATE.md 和 best_config.json
2. 提出唯一假设
3. 新建配置文件
4. 运行 pytest
5. 运行 5 折 GroupKFold
6. 保存 OOF、逐折分数、运行时间
7. 追加 experiment_log.csv
8. 与当前最佳实验比较
9. 接受或拒绝
10. 更新 DECISIONS.md
11. Git commit
```

## 15.3 接受标准

建议使用：

```text
平均 OOF R² 提升 >= 0.002
或
某个困难目标 R² 提升 >= 0.01 且总体不下降超过 0.001
```

同时要求：

```text
没有新增数据泄漏
没有测试失败
没有关键目标灾难性下降
运行时间可接受
分折标准差没有明显恶化
```

## 15.4 停止条件

满足任一条件即停止自动实验：

```text
连续 3 轮没有有效提升
累计完成 12 轮优化
当前最佳方案已经稳定
剩余改进需要大量外部数据或高风险复杂实现
```

停止后转入报告和交付阶段。

## 15.5 实验日志格式

`results/experiment_log.csv` 至少包含：

```text
experiment_id
parent_experiment_id
hypothesis
model
feature_groups
target_strategy
cv_mean_r2
cv_std_r2
target_scores_json
runtime_seconds
accepted
reason
git_commit
timestamp
```

---

# 16. Notebook 组织结构

最终 Notebook：

```text
notebooks/financial_performance_prediction_final.ipynb
```

应当是一份可以从头执行的课程项目成品，不是开发草稿。

必须包含：

## 16.1 封面 Markdown

```text
上市公司财务指标预测：基于历史季度数据与机器学习模型的实证分析
姓名：
学号：
课程：
日期：
```

不要编造姓名、学号和课程信息。留空等待用户补充。

## 16.2 章节顺序

```text
1. 项目背景与任务定义
2. 数据读取与字段结构
3. 数据质量审计
4. 探索性数据分析
5. 评价指标与交叉验证设计
6. 基线模型
7. 特征工程
8. 机器学习模型
9. 消融实验与结果比较
10. 最终模型融合
11. 预测结果合理性检查
12. 生成 submission.csv
13. 结论与不足
```

## 16.3 Notebook 代码风格

要求：

- 路径全部相对项目根目录；
- 设置随机种子；
- 复杂逻辑优先调用 `src/` 模块；
- 每个章节有简短说明；
- 关键表格直接显示；
- 图表引用 `figures/` 中统一生成的文件；
- 不展示过长日志；
- 不保留失败实验的杂乱输出；
- 最终运行无报错；
- 输出 submission 校验结果。

## 16.4 最终执行

先备份，再清空输出并从头执行：

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/financial_performance_prediction_final.ipynb \
  --output financial_performance_prediction_final.ipynb \
  --output-dir deliverables \
  --ExecutePreprocessor.timeout=1800
```

执行失败即禁止交付。

---

# 17. Word 报告组织结构

最终报告：

```text
deliverables/financial_performance_prediction_report.docx
```

中间稿：

```text
docs/REPORT_DRAFT.md
```

报告必须由脚本从结果表和统一图表生成，确保与 Notebook 一致。

## 17.1 推荐标题

```text
基于历史季度财务数据的上市公司财务指标预测
——机器学习模型、残差建模与融合方法的比较
```

## 17.2 报告结构

```text
封面
摘要
关键词
1. 任务背景与研究目标
2. 数据来源与字段说明
3. 数据质量审计与探索性分析
4. 方法设计
   4.1 评价指标
   4.2 交叉验证与防泄漏设计
   4.3 基线模型
   4.4 特征工程
   4.5 机器学习模型
   4.6 残差建模与融合
5. 实验结果
   5.1 基线结果
   5.2 模型对比
   5.3 消融实验
   5.4 各目标预测难度分析
   5.5 会计一致性检验
6. 最终模型与 submission 生成
7. 结论、不足与改进方向
参考资料
附录：字段、参数与交付检查
```

## 17.3 报告内容要求

必须做到：

- 任务定义准确；
- 9 个目标完整列出；
- 明确平均 R² 的计算方式；
- 说明为什么使用 GroupKFold；
- 展示缺失值、异常值、重复值和偏态分析；
- 展示基线，不只展示最佳模型；
- 解释元数据消融实验；
- 解释残差模型为什么合理；
- 说明会计恒等式只在 OOF 验证后用于后处理；
- 说明滑动窗口拓展的泄漏风险；
- 给出最终 submission 的 schema 验证；
- 结论只依据实验结果，不夸大。

## 17.4 Word 排版标准

若老师没有额外模板，使用稳妥的中文课程报告格式：

```text
纸张：A4
页边距：上下左右约 2.54 cm
正文：宋体，小四
英文与数字：Times New Roman
正文行距：1.5 倍
一级标题：黑体，三号或小三，加粗
二级标题：黑体，四号，加粗
三级标题：黑体，小四，加粗
图题：图下方居中，格式“图 1  ……”
表题：表上方居中，格式“表 1  ……”
页码：页脚居中
表格：优先三线表
引用：统一格式
```

若学校有模板，以模板为准。

## 17.5 语言质量

使用 `academic-writing`、`paper-writing` 和审稿技能进行修改。可以用 `humanizer-zh-academic` 改善自然表达，但必须遵守：

```text
不伪造个人经历
不伪造手工操作过程
不伪造数值
不为了规避检测而刻意改写
不使用空泛口号
不重复“首先、其次、最后”式模板句
```

---

# 18. submission.csv 生成与验证

必须直接复制模板：

```python
submission = sample_submission.copy()
for target in sample_submission.columns[1:]:
    submission[target] = final_predictions[target]
submission.to_csv("deliverables/submission.csv", index=False)
```

不得自行重新排列列。

`validate_delivery.py` 必须验证：

```python
assert submission.shape == sample_submission.shape
assert submission.columns.tolist() == sample_submission.columns.tolist()
assert submission["Id"].tolist() == sample_submission["Id"].tolist()
assert submission["Id"].is_unique
assert submission.iloc[:, 1:].notna().all().all()
assert np.isfinite(submission.iloc[:, 1:].to_numpy()).all()
```

还要检查：

```text
没有多余索引列
没有字符串预测
没有空行
没有科学计数法导致的解析问题
文件编码可正常读取
```

---

# 19. 测试要求

至少实现以下 pytest：

## 19.1 `test_schema.py`

检查：

- 原始文件存在；
- train、test、sample_submission 行列数；
- 目标列准确；
- test 不包含目标；
- submission 模板列顺序准确。

## 19.2 `test_metric.py`

检查：

- 完美预测 R² 为 1；
- 均值预测接近 0；
- 较差预测可以为负；
- 9 个目标的平均 R² 为逐目标算术平均。

## 19.3 `test_features.py`

检查：

- 特征工程不输出 `inf`；
- 训练和测试生成列一致；
- 不含 `Id`；
- 不含 Q0 目标；
- safe ratio 正确；
- signed log 可逆。

## 19.4 `test_cv_no_leakage.py`

检查：

- 同一个 `group_id` 不跨训练折和验证折；
- 滑动窗口样本按公司分组；
- 训练折拟合的缩尾阈值、填补器和编码器不读取验证折。

## 19.5 `test_submission.py`

检查：

- 行列数；
- 列顺序；
- Id；
- 非空；
- 数值有限；
- 可重新读取。

## 19.6 `test_report_assets.py`

检查：

- 报告引用的图全部存在；
- 表格文件存在；
- Notebook 与报告使用相同结果表；
- 最佳实验编号一致。

执行：

```bash
pytest -q
```

测试失败即禁止交付。

---

# 20. 报告与 Notebook 一致性机制

建立唯一事实来源：

```text
results/tables/
figures/
configs/best_config.json
```

规则：

1. Notebook 中的指标表读取 `results/tables/`；
2. Word 报告中的指标表读取同一文件；
3. Notebook 与报告中的图读取 `figures/`；
4. 最佳参数读取 `configs/best_config.json`；
5. 报告中不得手工写死实验数值；
6. `validate_delivery.py` 对比报告生成清单和 Notebook 生成清单。

保存：

```text
results/report_manifest.json
results/notebook_manifest.json
```

两个 manifest 的：

```text
best_experiment_id
figure_files
table_files
submission_sha256
```

必须一致。

---

# 21. 最终验收门槛

只有全部满足，才可以宣布完成。

## 21.1 数据与代码

- [ ] 原始文件 SHA256 已冻结；
- [ ] `pytest -q` 全部通过；
- [ ] 无数据泄漏；
- [ ] Notebook 可从头执行；
- [ ] 所有随机种子固定；
- [ ] 使用相对路径；
- [ ] `environment-quantenv-history.yml`、`environment-quantenv-lock.yml`、`conda-explicit-spec.txt` 和 `requirements-pip-lock.txt` 存在；
- [ ] 所有命令确认运行于 `QuantEnv`，而不是 `base` 或系统 Python；
- [ ] Git 工作区干净；
- [ ] 最佳实验可通过配置文件复现。

## 21.2 模型与实验

- [ ] 至少完成 B0、B1、B2、B3、B4；
- [ ] 至少完成 Ridge、HistGradientBoosting、CatBoost direct、CatBoost residual；
- [ ] 完成元数据消融；
- [ ] 保存全部 OOF；
- [ ] 展示 9 个目标 R² 和平均 R²；
- [ ] 展示分折标准差；
- [ ] 最终融合权重来自 OOF；
- [ ] 会计一致性后处理经过 OOF 验证；
- [ ] 对困难目标单独分析；
- [ ] 记录失败实验，不只记录成功实验。

## 21.3 Notebook

- [ ] 内容覆盖 EDA、分析过程、训练过程、结果和交叉验证；
- [ ] 章节完整；
- [ ] 代码简洁；
- [ ] 输出无错误；
- [ ] 图表清晰；
- [ ] 表格不溢出；
- [ ] 最后展示 submission 校验通过。

## 21.4 Word 报告

- [ ] 与 Notebook 数值和图表一致；
- [ ] 结构完整；
- [ ] 排版统一；
- [ ] 图表编号连续；
- [ ] 表格编号连续；
- [ ] 无空白页；
- [ ] 无跨页错乱；
- [ ] 无大段模板化废话；
- [ ] 无伪造主张；
- [ ] 结论对应实验结果；
- [ ] 已转 PDF 预览并检查。

## 21.5 submission.csv

- [ ] 与模板行列数一致；
- [ ] 列顺序完全一致；
- [ ] Id 顺序完全一致；
- [ ] 无 NaN；
- [ ] 无 inf；
- [ ] 无多余索引；
- [ ] 可重新读取；
- [ ] 已计算 SHA256。

---

# 22. 执行命令总览

执行代理应尽量把流程自动化为以下命令。默认全部使用本机 Conda 环境 `QuantEnv`：

```bash
# 0. 环境审计
conda env list
conda run -n QuantEnv python scripts/check_environment.py

# 1. 初始化
conda run -n QuantEnv python scripts/bootstrap.py

# 2. 数据审计与 EDA
conda run -n QuantEnv python scripts/audit_data.py

# 3. 测试
conda run -n QuantEnv python -m pytest -q

# 4. 基线
conda run -n QuantEnv python scripts/run_baselines.py

# 5. 执行单个实验
conda run -n QuantEnv python scripts/run_experiment.py \
  --config configs/catboost_direct.yaml

# 6. 有边界的自主优化
conda run -n QuantEnv python scripts/optimize.py \
  --max-rounds 12 \
  --patience 3

# 7. 最终训练
conda run -n QuantEnv python scripts/train_final.py \
  --config configs/best_config.json

# 8. 构建 Notebook
conda run -n QuantEnv python scripts/build_notebook.py

# 9. 从头执行 Notebook
conda run -n QuantEnv jupyter nbconvert \
  --to notebook \
  --execute notebooks/financial_performance_prediction_final.ipynb \
  --output financial_performance_prediction_final.ipynb \
  --output-dir deliverables \
  --ExecutePreprocessor.timeout=1800

# 10. 构建 Word 报告
conda run -n QuantEnv python scripts/build_report.py

# 11. 验收
conda run -n QuantEnv python scripts/validate_delivery.py

# 12. 打包
conda run -n QuantEnv python scripts/package_delivery.py
```

---


# 23. 通用 AI 编程代理主提示词

将以下提示词复制给 Codex 或任何能够读取项目文件、修改代码并运行命令的 AI 编程代理。使用 Codex 时，仍建议把整份本文保存为项目根目录的 `AGENTS.md`。

```text
请在当前 financial-performance-prediction 项目中严格执行 AGENTS.md；如果项目中没有 AGENTS.md，则读取 MASTER_WORKFLOW.md。不要跳过任何章节。

你是本项目唯一的执行代理，需要依次承担项目经理、数据分析师、机器学习工程师、实验研究员、代码审查员、学术写作助手和交付验收员的角色。你必须真实运行代码、保存结果、主动审计并修订，不能只给出建议或伪造完成状态。

项目根目录中已有 5 个原始文件：
- train.csv
- test.csv
- sample_submission.csv
- data_dictionary.txt
- 说明.docx

总目标：
生成符合老师要求的完整成品：
1. deliverables/financial_performance_prediction_final.ipynb
2. deliverables/financial_performance_prediction_report.docx
3. deliverables/submission.csv
4. deliverables/README_delivery.md

必须遵守：
- 不修改、移动或覆盖 5 个原始文件；
- 固定使用本机已有 Conda 环境 QuantEnv；不得默认创建 .venv，不得误用 base；
- 所有自动化命令优先使用 conda run -n QuantEnv；
- 安装依赖前先导出 QuantEnv 快照，不得无条件批量升级环境；
- 先冻结原始文件 SHA256，再开始分析；
- 不把 Id 作为特征；
- 不允许任何 Q0 目标列进入特征；
- 9 个目标分别计算 R² 后再取平均；
- 不允许重复特征样本跨训练折和验证折；
- 滑动窗口扩增时不允许使用未来元数据；
- 所有填补器、编码器、缩尾阈值只能在训练折拟合；
- 不允许使用测试集结果、排行榜分数或手工试探选择模型；
- 所有融合权重和后处理参数只能根据 OOF 预测选择；
- 报告和 Notebook 必须读取同一份 results/tables 与 figures；
- 不得手工写死报告中的实验数值；
- 不得伪造运行结果；
- 失败实验也要记录；
- 只有全部验收门槛通过后才能宣布完成。

本地 Skill 使用：
1. 先搜索本地所有 SKILL.md；
2. 阅读可用技能说明；
3. 优先使用已安装的 data-analysis、quantitative-research、code-review-and-quality、python-testing-patterns、academic-writing、paper-writing、paper-audit、paper-verification、critical-analysis、word-docx、pdf、diagnose、evaluate、iterate；
4. 某技能不存在时，记录缺失并使用普通脚本替代，不要停滞。

按阶段执行：

阶段 A：QuantEnv 审计、初始化与数据冻结
- 确认 QuantEnv 存在并使用 scripts/check_environment.py 审计依赖；
- 导出 environment-quantenv-history.yml、environment-quantenv-lock.yml、conda-explicit-spec.txt 和 requirements-pip-lock.txt；
- 建立工程目录、Git、依赖锁定；
- 编写 scripts/bootstrap.py；
- 保存 results/input_manifest.json；
- 校验原始文件 SHA256。

阶段 B：数据审计与 EDA
- 编写 scripts/audit_data.py；
- 完成 schema、缺失值、类别、异常值、重复值、会计恒等式、相关性和 train/test 匹配检查；
- 生成统一图表到 figures/；
- 生成统一表格到 results/tables/。

阶段 C：测试框架
- 完成 pytest；
- 至少覆盖 schema、metric、feature、CV 防泄漏、submission、报告资源；
- 测试失败不得进入最终交付。

阶段 D：基线
- 实现 B0 均值预测；
- 实现 B1 Q1 最近季度复制；
- 实现 B2 Q4 季节性复制；
- 实现 B3 短趋势外推；
- 实现 B4 OOF 驱动的逐目标基线融合；
- 保存 OOF 和逐折 R²。

阶段 E：特征工程
- 原始历史值；
- 元数据；
- 环比、同比、差值；
- 滚动均值、标准差、斜率、极值；
- 财务比率；
- 缺失指示；
- signed_log1p 特征；
- 所有操作保证训练和测试列一致。

阶段 F：模型
- Ridge；
- HistGradientBoostingRegressor；
- CatBoost direct；
- CatBoost residual；
- 可选 LightGBM 或 XGBoost 对照；
- 逐目标训练和记录。

阶段 G：自主优化
- 每轮只提出一个可检验假设；
- 运行 5 折 GroupKFold；
- 保存配置、OOF、逐折分数、时间、Git commit；
- 追加 results/experiment_log.csv；
- 根据规则接受或拒绝；
- 连续 3 轮无提升或达到 12 轮时停止；
- 不允许无限调参。

阶段 H：融合与后处理
- 对每个目标分别搜索 OOF 融合权重；
- 检查资产负债恒等式、毛利润关系和营业利润关系；
- 仅在 OOF 验证提升时启用会计一致性后处理。

阶段 I：Notebook 与报告
- 构建 final Notebook；
- 构建中文 Word 报告；
- Notebook 和报告使用同一套 results/tables 与 figures；
- 报告结构、图表编号、表格编号、字体和分页满足 AGENTS.md；
- 报告结论只依据真实实验。

阶段 J：交付验收
- 运行 pytest -q；
- 使用 nbconvert 从头执行 Notebook；
- 检查 Word 报告排版，可转 PDF 预览；
- 校验 submission.csv 行数、列数、列顺序、Id、NaN、inf 和索引；
- 生成 deliverables/README_delivery.md；
- 更新 docs/AGENT_STATE.md；
- 输出最终 PASS / FAIL。

自我审计要求：
在阶段 B、G、I、J 完成后，切换到严格审计模式。假设自己的实现有问题，主动寻找 BLOCKER、MAJOR 和 MINOR 问题，写入 docs/ISSUES.md。修复后重新运行测试和对应流程。禁止通过删除测试、放宽标准或手工改交付物来通过验收。

开始时只执行：
1. 读取要求；
2. 搜索并读取 Skill；
3. 确认 QuantEnv 并运行环境审计；
4. 导出 QuantEnv 快照；
5. 初始化目录；
6. 运行 bootstrap；
7. 运行 audit；
8. 汇报第一阶段真实结果。
不要一上来调模型。
```

---

# 24. 通用 AI 自我审计提示词

当主流程基本完成后，将以下提示词再次发给同一个 AI。它必须暂时停止扩展模型，转入独立审计模式。

```text
现在停止新增模型，切换到严格自我审计模式。假设当前项目存在隐藏错误，不要直接相信先前的完成声明。

请读取：
- AGENTS.md 或 MASTER_WORKFLOW.md
- docs/AGENT_STATE.md
- docs/DECISIONS.md
- docs/ISSUES.md
- results/experiment_log.csv
- configs/best_config.json
- src/
- scripts/
- tests/
- notebooks/
- deliverables/

请实际运行检查，并重点审计：
1. 5 个原始文件是否被修改；
2. schema、目标列和 submission 模板是否正确；
3. 是否错误使用 Id；
4. 是否将 Q0 目标或未来数据加入特征；
5. 重复样本是否跨折；
6. 滑动窗口是否使用未来元数据；
7. 填补、编码和缩尾是否只在训练折拟合；
8. 9 个目标是否分别计算 R² 后再平均；
9. OOF、融合权重和后处理是否真实来自交叉验证；
10. 是否用测试集或排行榜选择模型；
11. Notebook 是否能从头执行；
12. Notebook 与 Word 报告的图表、数值和最佳实验编号是否一致；
13. Word 是否存在空白页、错位、图表失真、表格溢出和格式不统一；
14. submission.csv 是否完全匹配模板且无 NaN / inf；
15. 报告结论是否全部有真实结果支持；
16. 是否记录失败实验；
17. 是否满足老师要求：Notebook 图表结果与报告一致，覆盖 EDA、分析过程、机器学习训练过程与结果、交叉验证。

将问题写入 docs/ISSUES.md，分为：
- BLOCKER
- MAJOR
- MINOR

每条问题包含：
- 文件与位置
- 问题描述
- 影响
- 可执行修复建议
- 验证方法

然后优先修复全部 BLOCKER，再修复 MAJOR。每修复一组问题：
1. 运行对应测试；
2. 更新 docs/DECISIONS.md；
3. 更新 docs/AGENT_STATE.md；
4. 创建 Git commit；
5. 重新运行 pytest、Notebook 执行和 validate_delivery.py。

禁止通过删除检查、跳过测试、降低阈值或手工修改交付物来“修复”问题。
最后给出 PASS / FAIL。
```

---

# 25. 通用 AI 中断后继续执行提示词

如果 AI 中途停止、会话中断或上下文不足，使用：

```text
请继续执行 financial-performance-prediction 项目。

先读取：
- AGENTS.md 或 MASTER_WORKFLOW.md
- docs/AGENT_STATE.md
- docs/ISSUES.md
- docs/DECISIONS.md
- results/experiment_log.csv
- configs/best_config.json

不要重复已经完成且通过验收的工作。根据 docs/AGENT_STATE.md 中的“下一步唯一动作”继续执行。任何新修改都必须运行对应测试、更新状态文件并提交 Git commit。不得跳过自我审计和最终验收门槛。
```

---

# 26. 最终交付 README 内容

`deliverables/README_delivery.md` 应包含：

```text
项目名称
原始数据文件列表及 SHA256
交付文件列表及 SHA256
Conda 环境名称（QuantEnv）
Python 可执行文件路径
Python 版本
依赖版本
最终实验编号
交叉验证策略
OOF 平均 R²
9 个目标的 OOF R²
最终模型概述
submission 校验结果
Notebook 从头执行命令
报告构建命令
已知限制
```

# 27. 参考资料

## 27.1 任务与公开基线

- Kaggle 官方比赛页：<https://www.kaggle.com/competitions/financial-performance-prediction>
- Kaggle Notebook：`Quick missing data imputation and model`  
  <https://www.kaggle.com/code/ryazanoff/quick-missing-data-imputation-and-model>
- Kaggle Notebook：`Financial data prediction`  
  <https://www.kaggle.com/code/emish8/financial-data-prediction>

## 27.2 AI 代理与工程执行

- Codex `AGENTS.md` 官方指南：<https://developers.openai.com/codex/guides/agents-md>
- Codex CLI 官方说明：<https://developers.openai.com/codex/cli>
- Codex Agent Skills 官方说明：<https://developers.openai.com/codex/skills>
- 开放格式 `AGENTS.md`：<https://agents.md/>
- Agent Skills 开放格式说明：<https://agentskills.io/home>

## 27.3 机器学习与 Notebook 执行

- scikit-learn `r2_score`：<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>
- scikit-learn `GroupKFold`：<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html>
- scikit-learn `HistGradientBoostingRegressor`：<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html>
- CatBoost 分类变量：<https://catboost.ai/docs/en/features/categorical-features>
- CatBoost 缺失值处理：<https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing>
- nbconvert 执行 Notebook：<https://nbconvert.readthedocs.io/en/latest/execute_api.html>
- Conda 环境管理：<https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html>
- Conda `env list`：<https://docs.conda.io/projects/conda/en/stable/commands/env/list.html>
- Conda `run`：<https://docs.conda.io/projects/conda/en/stable/commands/run.html>
- Conda `env export`：<https://docs.conda.io/projects/conda/en/stable/commands/env/export.html>
- Conda `list --explicit`：<https://docs.conda.io/projects/conda/en/stable/commands/list.html>

---

# 28. 最终路线摘要

本项目的最终推荐路线是：

```text
确认并冻结 QuantEnv 环境
→ 读取与冻结原始数据
→ 数据质量审计
→ 统一 EDA 与图表
→ GroupKFold 防重复泄漏
→ Q1、Q4 与趋势基线
→ 逐目标基线融合
→ 原始历史特征
→ 环比、同比、滚动统计、财务比率和缺失指示特征
→ Ridge 与 HistGradientBoosting 对照
→ CatBoost direct 主模型
→ CatBoost residual 重点优化
→ 元数据消融
→ OOF 逐目标融合
→ OOF 验证后的会计一致性调整
→ 可选滑动窗口拓展
→ 全量训练
→ sample_submission 原样填充
→ Notebook、Word 报告和 submission 统一验收
```

这条路线兼顾：

```text
老师要求
可解释性
建模效果
数据泄漏控制
AI 自主迭代
报告质量
交付可复现性
```

执行时不要追求无限模型堆叠。优先保证：每个结论都由真实实验支持，每张图都有解释，每个数值都能追溯，每个交付物都能复现。
