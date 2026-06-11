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

### 模型层级

| 模型 | 类型 | 说明 |
|---|---|---|
| B0 | 均值基线 | 训练集各目标均值 |
| B1 | 规则基线 | 最近季度（Q1）直接复制 |
| B2 | 规则基线 | 同季度季节性复制 |
| B3 | 规则基线 | 近两季度差值外推 |
| B4 | 规则融合 | B1/B2/B3 逐目标加权融合 |
| M1 | 线性模型 | Ridge 回归 |
| M2 | 梯度提升 | HistGradientBoostingRegressor |
| M3a–M3d | CatBoost 直接预测 | 逐步增加特征集（历史 → 行业/元数据/工程特征） |
| M4 | CatBoost 残差建模 | 以 B4 为基线，对残差建模 |

最终选定 **M4（CatBoost 残差建模，历史 + 元数据 + 工程特征）** 作为提交模型。M6 逐目标融合仅保留为对照实验。

### 验证协议

- **交叉验证**：nested GroupKFold（外层 5 折 / 内层 4 折）
- **分组依据**：仅基于 Q1–Q10 原始历史季度特征哈希，排除元数据和工程特征，确保完全重复的历史记录不跨折
- **nested OOF**：外层折的验证集仅用于最终泛化估计，所有超参数选择（B4 权重、会计后处理方案）均在内层完成
- **最终测试集预测**：使用全量训练集内部 OOF 校准权重，在全量训练集上重新训练后对 `test.csv` 预测

## nested OOF 结果

最终模型 M4 的 nested GroupKFold 各目标 R²：

| 目标 | nested OOF R² |
|---|---|
| Q0_TOTAL_ASSETS | 0.5865 |
| Q0_TOTAL_LIABILITIES | 0.8756 |
| Q0_TOTAL_STOCKHOLDERS_EQUITY | −0.1496 |
| Q0_GROSS_PROFIT | 0.9778 |
| Q0_COST_OF_REVENUES | 0.9485 |
| Q0_REVENUES | 0.9486 |
| Q0_OPERATING_INCOME | 0.7315 |
| Q0_OPERATING_EXPENSES | 0.9754 |
| Q0_EBITDA | 0.9443 |
| **9 目标平均** | **0.7598** |

> 会计一致性后处理在 inner fold 评估中未稳定提升平均 R²，最终选择 `none`。

## 项目结构

```
├── train.csv                            # 训练数据
├── test.csv                             # 测试数据
├── sample_submission.csv                # 提交模板（仅作格式参考）
├── data_dictionary.txt                  # 数据字典
├── requirements-pip-lock.txt            # pip 依赖锁定
├── environment-quantenv-lock.yml        # Conda 环境锁定
├── conda-explicit-spec.txt              # Conda 显式包列表
├── src/                                 # 核心模块
│   ├── baselines.py                     # 规则基线（B0–B4）
│   ├── feature_engineering.py           # 特征工程
│   ├── models.py                        # 机器学习模型
│   ├── blending.py                      # 逐目标融合（M6）
│   ├── accounting_checks.py             # 会计恒等式校验
│   ├── cv.py                            # GroupKFold 分组哈希
│   ├── metrics.py                       # 评价指标
│   ├── validation.py                    # 交付校验
│   ├── plots.py                         # 可视化
│   ├── constants.py                     # 常量（目标、元数据列等）
│   └── io_utils.py                      # 数据读写
├── scripts/                             # 辅助脚本（数据审计、自举等）
├── configs/                             # 模型配置文件
├── deliverables/                        # 最终交付物
│   ├── financial_performance_prediction_final.ipynb
│   ├── financial_performance_prediction_report.docx
│   ├── financial_performance_prediction_report.pdf
│   └── submission.csv
├── results/                             # 实验结果（模型权重、OOF 预测等）
├── figures/                             # 图表输出
└── notebooks/                           # 开发阶段 Notebook
```

## 环境

- **Conda 环境**：`QuantEnv`
- **Python 版本**：3.10.20

```bash
# 通过 Conda 创建环境
conda env create -f environment-quantenv-lock.yml

# 或通过 pip 安装
pip install -r requirements-pip-lock.txt
```

## 独立运行

最终 Notebook 不依赖 `src/`、`scripts/` 等工程目录，可在仅包含原始数据的干净目录中独立执行：

```bash
conda run -n QuantEnv jupyter nbconvert \
  --to notebook \
  --execute deliverables/financial_performance_prediction_final.ipynb \
  --output financial_performance_prediction_final_executed.ipynb \
  --output-dir standalone_check \
  --ExecutePreprocessor.timeout=7200
```

## 已知限制

- 最终 Notebook 为可读性和运行时间保留最终参数，不包含本地工程中的大规模调参和失败实验全过程
- 会计后处理仅覆盖资产负债表恒等式（总资产 = 总负债 + 股东权益），利润表内部勾稽关系未做强制约束
- `Q0_TOTAL_STOCKHOLDERS_EQUITY` 的 nested OOF R² 为负值，该目标的预测难度较高，是主要改进方向
