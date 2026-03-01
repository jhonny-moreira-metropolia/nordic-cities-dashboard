"""
forecaster.py
Monthly temperature forecasting using Facebook Prophet.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class ForecastResult:
    """Container for the forecast result of a single city."""
    def __init__(self, city: str, temp_var: str):
        self.city        = city
        self.temp_var    = temp_var
        self.historical  = pd.DataFrame()
        self.forecast    = pd.DataFrame()
        self.future_only = pd.DataFrame()
        self.model       = None
        self.error       = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.future_only.empty


def forecast_city(
    df_monthly: pd.DataFrame,
    city: str,
    temp_var: str = "TAVG_mean",
    periods: int = 36,
    min_months: int = 24,
    cutoff_date: pd.Timestamp | None = None,
) -> ForecastResult:
    """
    Train Prophet on monthly data for one city and project `periods` months ahead.

    Parameters:
        df_monthly   Monthly DataFrame (output of aggregate_monthly)
        city         City name
        temp_var     Column to forecast (TAVG_mean, TMAX_mean, TMIN_mean)
        periods      Months to project (default: 36)
        min_months   Minimum months of data required to train (default: 24)
        cutoff_date  If set, training data is trimmed to this date so all cities
                     share the same forecast start point.
    """
    result = ForecastResult(city, temp_var)

    try:
        from prophet import Prophet
    except ImportError:
        result.error = "Prophet is not installed. Run: pip install prophet"
        return result

    # Filter to city, valid variable and months with good coverage
    cdf = df_monthly[
        (df_monthly["CITY"] == city) &
        (df_monthly[temp_var].notna()) &
        (~df_monthly["LOW_COVERAGE"])
    ].copy()

    if len(cdf) < min_months:
        result.error = (
            f"Only {len(cdf)} months of valid data "
            f"(minimum required: {min_months})."
        )
        return result

    # Prophet format: ds (date) and y (value) columns
    df_prophet = pd.DataFrame({
        "ds": pd.to_datetime(cdf["PERIOD"] + "-01"),
        "y":  cdf[temp_var].values,
    }).sort_values("ds").reset_index(drop=True)

    # Apply cutoff so all cities forecast from the same start date
    if cutoff_date is not None:
        df_prophet = df_prophet[df_prophet["ds"] <= cutoff_date]
        if len(df_prophet) < min_months:
            result.error = (
                f"Only {len(df_prophet)} months before cutoff "
                f"(minimum required: {min_months})."
            )
            return result

    result.historical = df_prophet.copy()

    # Model configuration
    # - yearly_seasonality: captures the summer/winter cycle (key for climate)
    # - seasonality_mode additive: appropriate for temperature data
    # - interval_width 0.95: 95% confidence band
    # - changepoint_prior_scale 0.05: moderate trend regularization
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_prophet)

    result.model = model

    # Prediction: freq="MS" = Month Start, aligned with monthly data
    future      = model.make_future_dataframe(periods=periods, freq="MS")
    forecast_df = model.predict(future)

    result.forecast = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result.forecast.columns = ["DATE", "FORECAST", "LOWER", "UPPER"]

    last_real           = df_prophet["ds"].max()
    result.future_only  = result.forecast[result.forecast["DATE"] > last_real].copy()

    return result


def forecast_all_cities(
    df_monthly: pd.DataFrame,
    cities: list,
    temp_var: str = "TAVG_mean",
    periods: int = 36,
) -> dict:
    """
    Run forecast_city for each city using all available data.
    No cutoff is applied - each city trains on its full history.
    """
    return {
        city: forecast_city(df_monthly, city, temp_var, periods)
        for city in cities
    }

    # Determine the common cutoff: earliest last data point among all cities
    last_dates = []
    for city in cities:
        cdf = df_monthly[
            (df_monthly["CITY"] == city) &
            (df_monthly[temp_var].notna()) &
            (~df_monthly["LOW_COVERAGE"])
        ]
        if not cdf.empty:
            last_dates.append(pd.to_datetime(cdf["PERIOD"].max() + "-01"))

    cutoff = min(last_dates) if last_dates else None

    return {
        city: forecast_city(df_monthly, city, temp_var, periods, cutoff_date=cutoff)
        for city in cities
    }


def forecast_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert a dict of ForecastResult objects to a combined DataFrame with future-only rows."""
    frames = []
    for city, res in results.items():
        if res.ok:
            df = res.future_only.copy()
            df["CITY"] = city
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)