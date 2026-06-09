# ISSUES

## MAJOR-001: QuantEnv 缺少后续建模与测试依赖

- 状态: open
- 发现时间: 2026-06-09
- 证据: `results/environment_audit.json`
- 缺失包: `catboost`, `xgboost`, `lightgbm`, `optuna`, `jupyter`, `pytest`, `pytest-cov`
- 影响: 第一阶段审计不受影响；进入 CatBoost 主模型、可选 GBDT 对照、Optuna 调参、pytest 验收或 Notebook 执行前必须补齐最小必要依赖。
- 处理原则: 按用户要求，第一阶段不批量安装或升级环境；后续只做最小安装。

## MAJOR-002: 完全重复特征记录必须用于分组交叉验证

- 状态: open
- 发现时间: 2026-06-09
- 证据: `results/tables/duplicate_summary.csv`
- 结果: 训练集 common-feature 重复行 16 行、2 个重复组；测试集 common-feature 重复行 2 行、1 个重复组；train/test 共有 2 个 common-feature 哈希，涉及训练行 18 行、测试行 4 行。
- 影响: 后续交叉验证若用普通 KFold，会让完全重复历史特征跨折，导致泄漏式高估。
- 处理原则: 后续所有 OOF 实验必须使用稳定 feature hash 的 `GroupKFold`。

## MINOR-001: 会计恒等式误差存在重尾极端值

- 状态: open
- 发现时间: 2026-06-09
- 证据: `results/tables/accounting_identity_summary.csv`, `figures/fig07_accounting_identity_error.png`
- 结果: 历史季度 `assets - liabilities - equity` 的相对误差中位数多为 0，但均值受极端值明显影响。
- 影响: 后续若做会计一致性后处理，不能直接依赖均值类指标，应使用稳健分位数和 OOF 验证。

## MAJOR-003: 数值列存在正无穷值

- 状态: open
- 发现时间: 2026-06-09
- 证据: `results/data_audit.json`, `results/tables/numeric_extreme_summary.csv`
- 结果: 训练集发现 13 个 `+inf`，测试集发现 4 个 `+inf`，未发现 `-inf`。
- 影响: 直接进入缩放器、线性模型或部分树模型会产生异常；相关性审计已先将 `inf` 视为缺失。
- 处理原则: 后续特征工程入口统一将 `±inf` 转为 `NaN`，再按折内拟合的预处理策略处理。
