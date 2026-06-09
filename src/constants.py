from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILES = [
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "data_dictionary.txt",
    "说明.docx",
]

ID_COLUMN = "Id"

TARGET_COLUMNS = [
    "Q0_TOTAL_ASSETS",
    "Q0_TOTAL_LIABILITIES",
    "Q0_TOTAL_STOCKHOLDERS_EQUITY",
    "Q0_GROSS_PROFIT",
    "Q0_COST_OF_REVENUES",
    "Q0_REVENUES",
    "Q0_OPERATING_INCOME",
    "Q0_OPERATING_EXPENSES",
    "Q0_EBITDA",
]

SUBMISSION_COLUMNS = [
    "Id",
    "Q0_REVENUES",
    "Q0_COST_OF_REVENUES",
    "Q0_GROSS_PROFIT",
    "Q0_OPERATING_EXPENSES",
    "Q0_EBITDA",
    "Q0_OPERATING_INCOME",
    "Q0_TOTAL_ASSETS",
    "Q0_TOTAL_LIABILITIES",
    "Q0_TOTAL_STOCKHOLDERS_EQUITY",
]

METADATA_COLUMNS = [
    "industry",
    "sector",
    "fullTimeEmployees",
    "auditRisk",
    "boardRisk",
    "compensationRisk",
    "shareHolderRightsRisk",
    "overallRisk",
    "trailingPE",
    "forwardPE",
    "floatShares",
    "sharesOutstanding",
    "trailingEps",
    "forwardEps",
    "targetHighPrice",
    "targetLowPrice",
    "targetMeanPrice",
    "targetMedianPrice",
    "recommendationMean",
    "recommendationKey",
    "numberOfAnalystOpinions",
    "totalCash",
    "totalCashPerShare",
    "ebitda",
    "totalDebt",
    "totalRevenue",
    "revenuePerShare",
    "freeCashflow",
    "operatingCashflow",
    "revenueGrowth",
    "financialCurrency",
]

HISTORICAL_METRICS = [
    "TOTAL_ASSETS",
    "TOTAL_CURRENT_ASSETS",
    "TOTAL_NONCURRENT_ASSETS",
    "TOTAL_LIABILITIES",
    "TOTAL_CURRENT_LIABILITIES",
    "TOTAL_NONCURRENT_LIABILITIES",
    "TOTAL_LIABILITIES_AND_EQUITY",
    "TOTAL_STOCKHOLDERS_EQUITY",
    "NET_INCOME",
    "GROSS_PROFIT",
    "COST_OF_REVENUES",
    "REVENUES",
    "OPERATING_INCOME",
    "OPERATING_EXPENSES",
    "EBITDA",
    "DEPRECIATION_AND_AMORTIZATION",
]

HISTORICAL_QUARTERS = [f"Q{i}" for i in range(1, 11)]

EXPECTED_TRAIN_SHAPE = (1624, 212)
EXPECTED_TEST_SHAPE = (406, 203)
EXPECTED_SAMPLE_SUBMISSION_SHAPE = (406, 10)

