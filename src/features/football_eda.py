"""
⚽ Football 데이터 EDA (CSV 기반)
data/raw 내 CSV들을 자동 탐색하여 스키마 요약과 로우카디널리티(유니크 ≤ 10) 컬럼을 보고합니다.
각 블록은 주피터 셀에 복사해 사용 가능합니다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# 셀 1: 경로 설정 및 CSV 자동 탐색
def _default_project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists():
        return cwd
    return Path("/Users/jina/Documents/GitHub/SKN18-2nd-4Team").resolve()


PROJECT_ROOT: Path = _default_project_root()
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def discover_csvs() -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in sorted(DATA_RAW_DIR.glob("*.csv")):
        name = p.stem
        mapping[name] = p
    return mapping


# 셀 1-1: 발견된 CSV 목록 출력용 (주피터 즉시 출력)
def print_discovered_csvs() -> Dict[str, Path]:
    mapping = discover_csvs()
    print("발견된 CSV 파일:")
    for name, path in mapping.items():
        print(f"- {name}: {path}")
    if not mapping:
        print("data/raw 에 CSV가 없습니다.")
    return mapping


# 셀 2: 유틸 함수들
def load_csv(csv_path: Path, nrows: int | None = None, low_memory: bool = True) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path, nrows=nrows, low_memory=low_memory)


def try_parse_datetimes(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    target = df if inplace else df.copy()
    datetime_like_markers = ("date", "_at", "time")
    for col in target.columns:
        lower = str(col).lower()
        if any(marker in lower for marker in datetime_like_markers):
            with pd.option_context("mode.chained_assignment", None):
                try:
                    target[col] = pd.to_datetime(target[col], errors="ignore")
                except Exception:
                    pass
    return target


def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def summarize_schema(df: pd.DataFrame, max_sample_values: int = 5) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total_rows = len(df)
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        missing_rate = float(1 - (non_null / total_rows)) if total_rows > 0 else np.nan
        nunique = int(s.nunique(dropna=True))
        dtype_str = str(s.dtype)
        uniques = s.dropna().unique()[:max_sample_values]
        sample_values = ", ".join(map(lambda x: str(x)[:80], uniques))
        rows.append(
            {
                "column": col,
                "dtype": dtype_str,
                "non_null": non_null,
                "missing_rate": round(missing_rate, 6),
                "nunique": nunique,
                "sample_values": sample_values,
            }
        )
    return pd.DataFrame(rows).sort_values(by=["missing_rate", "column"]).reset_index(drop=True)


# 셀 3: 테이블 분석 및 전체 루프
def analyze_table(name: str, csv_path: Path, sample_rows: int = 5, nrows: int | None = None):
    print(f"\n===== [{name}] =====")
    print(f"Path: {csv_path}")
    df = load_csv(csv_path, nrows=nrows)
    print(f"Rows: {len(df):,}, Cols: {df.shape[1]}, Memory: {memory_usage_mb(df):.2f} MB (before dt parsing)")
    df = try_parse_datetimes(df, inplace=False)
    print(f"Memory: {memory_usage_mb(df):.2f} MB (after dt parsing)\n")
    print("Head:")
    print(df.head(sample_rows))
    schema = summarize_schema(df)
    print("\nSchema summary (top by missing rate):")
    print(schema.head(30))
    return df, schema


def run_all(nrows: int | None = None):
    csvs = discover_csvs()
    results: Dict[str, Dict[str, object]] = {}
    for name, csv_path in csvs.items():
        try:
            df, schema = analyze_table(name=name, csv_path=csv_path, nrows=nrows)
            results[name] = {"df": df, "schema": schema}
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
    return results


# 셀 3-1: 특정 테이블 빠른 개요 출력 (행/열/메모리/헤드)
def print_table_overview(name: str, nrows: int | None = None, head_rows: int = 5):
    csvs = discover_csvs()
    if name not in csvs:
        print(f"테이블을 찾을 수 없습니다: {name}")
        return None
    path = csvs[name]
    df = load_csv(path, nrows=nrows)
    print(f"\n=== Overview: {name} ===")
    print(f"Path: {path}")
    print(f"Rows: {len(df):,}, Cols: {df.shape[1]}, Memory: {memory_usage_mb(df):.2f} MB")
    print("Head:")
    print(df.head(head_rows))
    return df


# 셀 3-2: 전체 테이블 스키마만 출력
def print_all_schemas(nrows: int | None = None, max_show: int = 30):
    for name, path in discover_csvs().items():
        try:
            df = load_csv(path, nrows=nrows)
            df = try_parse_datetimes(df, inplace=False)
            schema = summarize_schema(df)
            print(f"\n=== Schema: {name} ===")
            print(schema.head(max_show))
        except Exception as e:
            print(f"Error on {name}: {e}")


# 셀 3-3: 전체 테이블 헤드만 출력
def print_all_heads(head_rows: int = 5, nrows: int | None = None):
    for name, path in discover_csvs().items():
        try:
            df = load_csv(path, nrows=nrows)
            print(f"\n=== Head: {name} ===")
            print(df.head(head_rows))
        except Exception as e:
            print(f"Error on {name}: {e}")


# 셀 4: 로우카디널리티(유니크 ≤ 10) 리포트
def low_cardinality_report(max_unique: int = 10, nrows: int | None = None) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for name, csv_path in discover_csvs().items():
        try:
            df = load_csv(csv_path, nrows=nrows)
        except FileNotFoundError:
            continue
        df = try_parse_datetimes(df, inplace=False)
        for col in df.columns:
            s = df[col]
            nunique = int(s.nunique(dropna=True))
            if nunique <= max_unique:
                values = list(pd.Series(s.dropna().unique()).astype("string"))
                try:
                    values_sorted = sorted(values, key=lambda x: (x is None, str(x)))
                except Exception:
                    values_sorted = values
                values_preview = ", ".join([str(v) for v in values_sorted[:max_unique]])
                records.append({
                    "table": name,
                    "column": col,
                    "dtype": str(s.dtype),
                    "nunique": nunique,
                    "values": values_preview,
                })
    report = pd.DataFrame.from_records(records).sort_values(by=["table", "nunique", "column"]).reset_index(drop=True)
    return report


def print_low_cardinality_report(max_unique: int = 10, nrows: int | None = None) -> pd.DataFrame:
    report = low_cardinality_report(max_unique=max_unique, nrows=nrows)
    tables = report["table"].unique().tolist() if not report.empty else []
    for t in tables:
        subset = report[report["table"] == t]
        if subset.empty:
            continue
        print(f"\n===== [low-cardinality] {t} =====")
        for _, row in subset.iterrows():
            print(f"{row['column']} (dtype={row['dtype']}): nunique={row['nunique']} -> {row['values']}")
    if report.empty:
        print("No columns with low cardinality found under the given threshold.")
    return report


if __name__ == "__main__":
    NROWS = None  # 예: 1_000_000 등으로 샘플링
    print("\n=== 기본 스키마 점검 ===")
    _ = run_all(nrows=NROWS)

    print("\n=== 유니크 ≤ 10 컬럼 리포트 ===")
    report_df = print_low_cardinality_report(max_unique=10, nrows=NROWS)
    out_path = DATA_PROCESSED_DIR / "low_cardinality_report.csv"
    if not report_df.empty:
        report_df.to_csv(out_path, index=False)
        print(f"\n저장됨: {out_path}")


# =============================
# 주피터 즉시 실행용 복사 셀
# =============================

# 셀 A: 발견된 CSV 파일 목록
# (이 블록을 노트북 셀에 그대로 붙여넣고 Shift+Enter)
# mapping = discover_csvs()
# print("발견된 CSV 파일:")
# for name, path in mapping.items():
#     print(f"- {name}: {path}")
# if not mapping:
#     print("data/raw 에 CSV가 없습니다.")


# 셀 B: 특정 테이블 빠른 개요 (TABLE 이름만 바꿔 실행)
# TABLE = "appearances"  # 예시: 분석할 테이블 이름
# NROWS = None            # 대용량이면 예: 1_000_000
# csvs = discover_csvs()
# if TABLE not in csvs:
#     print(f"테이블을 찾을 수 없습니다: {TABLE}")
# else:
#     path = csvs[TABLE]
#     df = load_csv(path, nrows=NROWS)
#     df = try_parse_datetimes(df, inplace=False)
#     print(f"\n=== Overview: {TABLE} ===")
#     print(f"Path: {path}")
#     print(f"Rows: {len(df):,}, Cols: {df.shape[1]}, Memory: {memory_usage_mb(df):.2f} MB")
#     print("Head:")
#     print(df.head(5))
#     schema = summarize_schema(df)
#     print("\nSchema (top by missing rate):")
#     print(schema.head(30))


# 셀 C: 전체 테이블 스키마 요약 출력
# NROWS = None  # 샘플링 필요 시 숫자 지정
# for name, path in discover_csvs().items():
#     try:
#         df = load_csv(path, nrows=NROWS)
#         df = try_parse_datetimes(df, inplace=False)
#         schema = summarize_schema(df)
#         print(f"\n=== Schema: {name} ===")
#         print(schema.head(30))
#     except Exception as e:
#         print(f"Error on {name}: {e}")


# 셀 D: 유니크 ≤ 10 컬럼 리포트 (표시 + 저장 옵션)
# MAX_UNIQUE = 10
# NROWS = None
# report_df = low_cardinality_report(max_unique=MAX_UNIQUE, nrows=NROWS)
# if report_df.empty:
#     print("No columns with low cardinality found under the given threshold.")
# else:
#     for t in report_df["table"].unique().tolist():
#         subset = report_df[report_df["table"] == t]
#         if subset.empty:
#             continue
#         print(f"\n===== [low-cardinality] {t} =====")
#         for _, row in subset.iterrows():
#             print(f"{row['column']} (dtype={row['dtype']}): nunique={row['nunique']} -> {row['values']}")
#     # 저장
#     out_path = DATA_PROCESSED_DIR / "low_cardinality_report.csv"
#     report_df.to_csv(out_path, index=False)
#     print(f"\n저장됨: {out_path}")

