"""
data_processor.py
Loads, cleans and aggregates NOAA GHCND Daily Summaries
to monthly level. Data is loaded automatically from the data/ folder.

Special cases handled:
  - Copenhagen: TMAX has ~7000 nulls in 2000-2020
  - Oslo TAVG: 68% nulls -> recalculated from TMAX+TMIN
  - Stations with different date ranges
  - Values in tenths of °C (standard GHCND format)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------------------
# Edit here if you change a station
CITY_FILES = {
    "Helsinki":   "Helsinki.csv",
    "Copenhagen": "Copenhagen.csv",
    "Stockholm":  "Stockholm.csv",
    "Reykjavik":  "Reykjavik.csv",
    "Oslo":       "Oslo.csv",
}

DATA_DIR = Path(__file__).parent / "data"

# -------------------------------------------------------------------------------
CITIES_META = {
    "Helsinki":   {"country": "Finland",  "lat": 60.17, "lon":  24.94},
    "Stockholm":  {"country": "Sweden",   "lat": 59.33, "lon":  18.07},
    "Copenhagen": {"country": "Denmark",  "lat": 55.68, "lon":  12.57},
    "Oslo":       {"country": "Norway",   "lat": 59.91, "lon":  10.75},
    "Reykjavik":  {"country": "Iceland",  "lat": 64.13, "lon": -21.93},
}

MONTH_NAMES_EN = {
    1:"January", 2:"February", 3:"March",     4:"April",
    5:"May",     6:"June",     7:"July",       8:"August",
    9:"September",10:"October",11:"November", 12:"December",
}
MONTH_SHORT = {k: v[:3] for k, v in MONTH_NAMES_EN.items()}

MIN_DAYS_PER_MONTH = 10


# -------------------------------------------------------------------------------

def _detect_and_convert(df: pd.DataFrame, cols: list) -> dict:
    """
    Detect if values are in tenths of °C by evaluating all
    temperature columns together (TMIN alone in Nordic winter can give a false negative).
    """
    numeric  = {c: pd.to_numeric(df[c], errors="coerce") for c in cols}
    combined = pd.concat(list(numeric.values())).dropna()
    in_tenths = (combined.abs().mean() > 50) if len(combined) > 0 else False
    return {c: (s / 10.0 if in_tenths else s) for c, s in numeric.items()}


# -------------------------------------------------------------------------------

def load_csv(filepath: str | Path, city_name: str) -> pd.DataFrame:
    """
    Load a NOAA CSV file. Returns a daily DataFrame with:
      DATE, CITY, TMAX_C, TMIN_C, TAVG_C (en °C reales)

    Rules:
      - Station TAVG is ignored if >50% null and recalculated from TMAX+TMIN
      - TAVG is only calculated where BOTH TMAX and TMIN are available
      - Rows with no temperature data at all are dropped
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip().str.upper().str.replace('"', '')

    if "DATE" not in df.columns:
        raise ValueError(f"No DATE column in: {filepath}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    raw_temp_cols = [c for c in ["TMAX", "TMIN", "TAVG"] if c in df.columns]
    if not raw_temp_cols:
        avail = [c for c in df.columns if "ATTRIB" not in c]
        raise ValueError(
            f"No temperature columns (TMAX/TMIN/TAVG) in '{Path(filepath).name}'. "
            f"Available: {avail}"
        )

    converted = _detect_and_convert(df, raw_temp_cols)

    result = pd.DataFrame({"DATE": df["DATE"], "CITY": city_name})

    if "TMAX" in converted:
        result["TMAX_C"] = converted["TMAX"]
    if "TMIN" in converted:
        result["TMIN_C"] = converted["TMIN"]

    # Use station TAVG only if it has sufficient coverage (>50% non-null)
    if "TAVG" in converted:
        tavg = converted["TAVG"]
        if tavg.isna().mean() <= 0.50:
            result["TAVG_C"] = tavg

    # Calculate TAVG only where BOTH TMAX and TMIN are available
    if "TAVG_C" not in result.columns:
        if "TMAX_C" in result.columns and "TMIN_C" in result.columns:
            both = result["TMAX_C"].notna() & result["TMIN_C"].notna()
            result["TAVG_C"] = np.where(
                both,
                (result["TMAX_C"] + result["TMIN_C"]) / 2.0,
                np.nan,
            )

    temp_cols = [c for c in ["TMAX_C", "TMIN_C", "TAVG_C"] if c in result.columns]
    result = result.dropna(subset=temp_cols, how="all").reset_index(drop=True)

    if result.empty:
        raise ValueError(f"No valid temperature rows in '{Path(filepath).name}'.")

    return result


# -------------------------------------------------------------------------------

def aggregate_monthly(df_daily: pd.DataFrame,
                      min_days: int = MIN_DAYS_PER_MONTH) -> pd.DataFrame:
    """
    Aggregate daily data to monthly level.
    Output columns: CITY, YEAR, MONTH, PERIOD,
                    TAVG_mean, TMAX_mean, TMIN_mean, TMAX_abs, TMIN_abs,
                    DATA_DAYS, COVERAGE (%), LOW_COVERAGE (bool)
    """
    df = df_daily.copy()
    df["YEAR"]          = df["DATE"].dt.year
    df["MONTH"]         = df["DATE"].dt.month
    df["PERIOD"]        = df["DATE"].dt.to_period("M").astype(str)
    df["DAYS_IN_MONTH"] = df["DATE"].dt.days_in_month

    named_aggs = {
        "DATA_DAYS":     pd.NamedAgg(column="DATE",          aggfunc="count"),
        "DAYS_IN_MONTH": pd.NamedAgg(column="DAYS_IN_MONTH", aggfunc="first"),
    }
    if "TAVG_C" in df.columns:
        named_aggs["TAVG_mean"] = pd.NamedAgg(column="TAVG_C", aggfunc="mean")
    if "TMAX_C" in df.columns:
        named_aggs["TMAX_mean"] = pd.NamedAgg(column="TMAX_C", aggfunc="mean")
        named_aggs["TMAX_abs"]  = pd.NamedAgg(column="TMAX_C", aggfunc="max")
    if "TMIN_C" in df.columns:
        named_aggs["TMIN_mean"] = pd.NamedAgg(column="TMIN_C", aggfunc="mean")
        named_aggs["TMIN_abs"]  = pd.NamedAgg(column="TMIN_C", aggfunc="min")

    monthly = (
        df.groupby(["CITY", "YEAR", "MONTH", "PERIOD"])
        .agg(**named_aggs)
        .reset_index()
    )

    monthly["COVERAGE"]     = (monthly["DATA_DAYS"] / monthly["DAYS_IN_MONTH"] * 100).clip(0, 100).round(1)
    monthly["LOW_COVERAGE"] = monthly["DATA_DAYS"] < min_days
    monthly = monthly.drop(columns=["DAYS_IN_MONTH"])

    temp_cols = [c for c in monthly.columns
                 if c not in {"CITY","YEAR","MONTH","PERIOD","DATA_DAYS","COVERAGE","LOW_COVERAGE"}]
    monthly[temp_cols] = monthly[temp_cols].round(2)

    return monthly.sort_values(["CITY","YEAR","MONTH"]).reset_index(drop=True)


# ------------------------------------------------------------------------------

def load_all_cities(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all CSV files defined in CITY_FILES from data_dir.
    Returns (df_daily, df_monthly).
    Raises FileNotFoundError if any file is missing.
    """
    daily_frames, monthly_frames = [], []
    missing = []

    for city, filename in CITY_FILES.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(f"{city} ({filename})")
            continue
        daily   = load_csv(path, city)
        monthly = aggregate_monthly(daily)
        daily_frames.append(daily)
        monthly_frames.append(monthly)

    if missing:
        raise FileNotFoundError(
            f"Missing CSV files in '{data_dir}':\n" + "\n".join(f"  - {m}" for m in missing)
        )

    return (
        pd.concat(daily_frames,   ignore_index=True),
        pd.concat(monthly_frames, ignore_index=True),
    )


# -------------------------------------------------------------------------------

def coverage_summary(df_monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, g in df_monthly.groupby("CITY"):
        rows.append({
            "City":             city,
            "First month":      g["PERIOD"].min(),
            "Last month":       g["PERIOD"].max(),
            "Total months":     len(g),
            "Low cov. months":  int(g["LOW_COVERAGE"].sum()),
            "Mean coverage %":  round(g["COVERAGE"].mean(), 1),
        })
    return pd.DataFrame(rows).set_index("City")