
import pandas as pd

def basic_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_per_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
    }

def print_profile(profile: dict):
    print("Rows:", profile["rows"], "Columns:", profile["columns"])
    print("Missing values:")
    for k, v in profile["missing_per_column"].items():
        print(f"{k}: {v}")
    print("Duplicate rows:", profile["duplicate_rows"])
