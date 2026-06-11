# Financial Performance Prediction

基于历史季度财务序列的九目标回归预测项目。通过公司过去 10 个季度（Q1–Q10）的财务报表指标及元数据，预测最近一个季度（Q0）的 9 个核心财务变量。

## 预测目标

| 目标变量 | 中文名称 | 类别 |
|---|---|---|
| `Q0_TOTAL_ASSETS` | 总资产 | 资产负债表 |
| `Q0_TOTAL_LIABILITIES` | 总负债 | 资产负债表 |
| `Q0_TOTAL_STOCKHOLDERS_EQUITY` | 股东权益 | 资产负债表 |
| `Q0_GROSS_PROFIT` | 毛利润 | 利润表 |
| `Q0_COST_OF_REVENUES` | 营业成本 | 利润表 |
| `Q0_REVENUES` | 营业收入 | 利润表 |
| `Q0_OPERATING_INCOME` | 营业利润 | 利润表 |
| `Q0_OPERATING_EXPENSES` | 营业费用 | 利润表 |
| `Q0_EBITDA` | EBITDA | 综合 |

## 数据

- **训练集**：`train.csv`（1,624 家公司 × 212 列）
- **测试集**：`test.csv`（406 家公司 × 203 列）
- **特征**：Q1–Q10 每季度 16 个财务指标 + 元数据（行业、板块、风险评分、分析师预测等）
- **详细字段说明**：见 `data_dictionary.txt`

## 方法

- **规则基线**：均值预测（B0）、最近季度复制（B1）、季节性复制（B2）、短趋势外推（B3）、规则融合（B4）
- **机器学习模型**：Ridge、HistGradientBoostingRegressor、CatBoost（直接预测 / 残差建模）
- **融合策略**：逐目标加权融合（M6）
- **后处理**：会计恒等式一致性校验（总资产 = 总负债 + 股东权益）
- **交叉验证**：nested GroupKFold，分组基于 Q1–Q10 原始历史特征哈希，避免重复历史记录跨折泄漏

## 项目结构

```
├── train.csv                          # 训练数据
├── test.csv                           # 测试数据
├── sample_submission.csv              # 提交模板
├── data_dictionary.txt                # 数据字典
├── src/                               # 核心模块
│   ├── baselines.py                   # 规则基线模型
│   ├── feature_engineering.py         # 特征工程
│   ├── models.py                      # 机器学习模型
│   ├── blending.py                    # 融合策略
│   ├── accounting_checks.py           # 会计恒等式校验
│   ├── cv.py                          # 交叉验证
│   ├── metrics.py                     # 评价指标
│   ├── validation.py                  # 验证工具
│   ├── plots.py                       # 可视化
│   ├── constants.py                   # 常量定义
│   └── io_utils.py                    # 数据读写工具
├── scripts/                           # 辅助脚本
├── configs/                           # 配置文件
├── tests/                             # 测试
├── deliverables/                      # 最终交付物
│   ├── financial_performance_prediction_final.ipynb
│   ├── financial_performance_prediction_report.docx
│   └── submission.csv
├── results/                           # 实验结果输出
├── figures/                           # 图表输出
└── notebooks/                         # 开发阶段 Notebook
```

## 环境

- **Conda 环境**：`QuantEnv`
- **Python 版本**：3.10.20
- **依赖管理**：`requirements-pip-lock.txt` / `environment-quantenv-lock.yml`

```bash
# 创建环境
conda env create -f environment-quantenv-lock.yml

# 或通过 pip 安装
pip install -r requirements-pip-lock.txt
```

## 独立运行

最终 Notebook 不依赖 `src/`、`scripts/` 等工程目录，可在仅包含数据的干净目录中独立执行：

```bash
conda run -n QuantEnv jupyter nbconvert \
  --to notebook \
  --execute deliverables/financial_performance_prediction_final.ipynb \
  --output financial_performance_prediction_final_executed.ipynb \
  --output-dir standalone_check \
  --ExecutePreprocessor.timeout=7200
```

## 评价指标

采用 R²（决定系数）评估各目标的预测精度，最终以 9 个目标 R² 的算术平均值作为综合得分。

## 已知限制

- 最终 Notebook 为可读性和运行时间保留最终参数，不包含本地工程中的大规模调参和失败实验全过程
- 会计后处理仅覆盖资产负债表恒等式，利润表内部勾稽关系未做强制约束
