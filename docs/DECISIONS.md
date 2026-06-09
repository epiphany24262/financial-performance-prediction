# DECISIONS

## 2026-06-09: 固定使用 QuantEnv

- 自动化命令均使用 `conda run -n QuantEnv ...`。
- 未创建 `.venv`，未使用 `base` 或系统 Python。

## 2026-06-09: 第一阶段不安装依赖

- 环境审计发现若干后续依赖缺失。
- 按用户要求，本阶段只记录缺失项，不批量安装或升级。

## 2026-06-09: 输入文件只读冻结

- 五个原始文件的 SHA256 已写入 `results/input_manifest.json`。
- 后续运行 `scripts/bootstrap.py` 时如发现 SHA256 变化，将停止执行。

## 2026-06-09: 提交列顺序以 sample_submission 为准

- `sample_submission.csv` 列顺序已验证。
- 后续 `submission.csv` 必须复制该列顺序，不能使用目标列表顺序重排。

## 2026-06-09: 重复记录用 feature hash 分组

- 数据审计发现 train/test 内部与跨集合存在完全相同 common-feature 记录。
- 后续 CV 以稳定 common-feature hash 作为 `GroupKFold` 的 group。

