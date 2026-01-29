from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import gridstatus

# Ridge (safe + cached)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Logging / Config
# -----------------------------
log = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

TZ = pytz.timezone("US/Pacific")

# CAISO fetch cache (fast refresh)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 min
CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "256"))
FETCH_CACHE = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)

# Model cache (don't retrain every request)
MODEL_TTL_SECONDS = int(os.getenv("MODEL_TTL_SECONDS", "3600"))  # 60 min (your spec)
MODEL_CACHE = TTLCache(maxsize=32, ttl=MODEL_TTL_SECONDS)

# Training window
TRAINING_DAYS = int(os.getenv("TRAINING_DAYS", "30"))  # your spec

# Cutoff: <=30h use "forecast if available", >30h predict (your spec)
FORECAST_CUTOFF_HOURS = float(os.getenv("FORECAST_CUTOFF_HOURS", "30"))

# LMP config (your spec)
LMP_NODE = os.getenv("LMP_NODE", "TH_NP15_GEN-APND")
LMP_MARKET_DA = os.getenv("LMP_MARKET_DA", "DAY_AHEAD_HOURLY")
LMP_MARKET_RT = os.getenv("LMP_MARKET_RT", "REAL_TIME_5_MIN")

# Ridge hyperparams (safe defaults; can tune later)
RIDGE_ALPHA = float(os.getenv("RIDGE_ALPHA", "2.0"))

RENEWABLE_KEYS = {
    "Solar",
    "Wind",
    "Small Hydro",
    "Large Hydro",
    "Geothermal",
    "Biomass",
    "Biogas",
}

caiso = gridstatus.CAISO()

app = FastAPI(title="CAISO Train Windows (Green + Cheap)")
app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------------
# Cache helpers
# -----------------------------
def _cache_key(prefix: str, **kwargs: Any) -> str:
    items = sorted((k, str(v)) for k, v in kwargs.items())
    return prefix + "|" + "|".join(f"{k}={v}" for k, v in items)


def _cached(cache: TTLCache, prefix: str, **kwargs: Any) -> Optional[Any]:
    return cache.get(_cache_key(prefix, **kwargs))


def _set_cache(cache: TTLCache, prefix: str, value: Any, **kwargs: Any) -> Any:
    cache[_cache_key(prefix, **kwargs)] = value
    return value


def _safe_fetch(fn, prefix: str, **kwargs: Any) -> pd.DataFrame:
    cached = _cached(FETCH_CACHE, prefix, **kwargs)
    if cached is not None:
        return cached

    try:
        df = fn(**kwargs)
    except Exception as e:
        log.exception("Upstream fetch failed: %s %s", prefix, kwargs)
        raise HTTPException(status_code=502, detail=f"Upstream CAISO fetch failed ({prefix}): {e}") from e

    if df is None:
        df = pd.DataFrame()

    return _set_cache(FETCH_CACHE, prefix, df, **kwargs)


# -----------------------------
# Time helpers
# -----------------------------
def now_pacific() -> datetime:
    return datetime.now(tz=TZ)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(TZ).isoformat()


def parse_time_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    time_candidates = [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "interval start",
        "Interval Start",
        "interval_start",
        "start",
        "Start",
    ]
    time_col = None
    for c in df.columns:
        if c in time_candidates or c.lower() in {t.lower() for t in time_candidates}:
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]

    out = df.copy().rename(columns={time_col: "time"})
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"])

    if getattr(out["time"].dt, "tz", None) is None:
        out["time"] = out["time"].dt.tz_localize(TZ)
    else:
        out["time"] = out["time"].dt.tz_convert(TZ)

    return out


def first_numeric_col(df: pd.DataFrame, exclude: set[str]) -> Optional[str]:
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


# -----------------------------
# Fetchers
# -----------------------------
def get_load_history(start: datetime, end: datetime) -> pd.DataFrame:
    return _safe_fetch(caiso.get_load, "load_history", date=start, end=end)


def get_load_forecast(end: datetime) -> pd.DataFrame:
    candidates = ["get_load_forecast", "get_demand_forecast"]
    for name in candidates:
        if hasattr(caiso, name):
            fn = getattr(caiso, name)
            return _safe_fetch(fn, f"{name}", date=now_pacific(), end=end)
    return pd.DataFrame()


def get_fuel_mix_history(start: datetime, end: datetime) -> pd.DataFrame:
    return _safe_fetch(caiso.get_fuel_mix, "fuel_mix_history", date=start, end=end)


def get_fuel_mix_forecast(end: datetime) -> pd.DataFrame:
    candidates = ["get_fuel_mix_forecast", "get_renewables_forecast", "get_solar_and_wind_forecast"]
    for name in candidates:
        if hasattr(caiso, name):
            fn = getattr(caiso, name)
            return _safe_fetch(fn, f"{name}", date=now_pacific(), end=end)
    return pd.DataFrame()


def get_lmp_series(end: datetime) -> Tuple[pd.DataFrame, bool]:
    if not hasattr(caiso, "get_lmp"):
        return pd.DataFrame(), False

    start = now_pacific() - timedelta(hours=2)

    frames: List[pd.DataFrame] = []
    available = False

    # DA
    try:
        df_da = _safe_fetch(
            caiso.get_lmp,
            "lmp_da",
            date=start,
            end=end,
            market=LMP_MARKET_DA,
            locations=[LMP_NODE],
        )
        if df_da is not None and not df_da.empty:
            frames.append(df_da)
            available = True
    except Exception:
        log.info("DA LMP not available for this adapter/config")

    # RT
    try:
        df_rt = _safe_fetch(
            caiso.get_lmp,
            "lmp_rt",
            date=start,
            end=end,
            market=LMP_MARKET_RT,
            locations=[LMP_NODE],
        )
        if df_rt is not None and not df_rt.empty:
            frames.append(df_rt)
            available = True
    except Exception:
        log.info("RT LMP not available for this adapter/config")

    if not frames:
        return pd.DataFrame(), False

    df = pd.concat(frames, ignore_index=True)
    df = parse_time_col(df)

    exclude = {"time", "Location", "location", "Market", "market", "LMP Type", "lmp_type"}
    price_col = None
    for c in df.columns:
        if c in exclude:
            continue
        if c.lower() in {"lmp", "lmp ($/mwh)", "lmp_usd_per_mwh"}:
            price_col = c
            break
    if price_col is None:
        price_col = first_numeric_col(df, exclude=exclude)
    if price_col is None:
        return pd.DataFrame(), False

    out = df[["time", price_col]].rename(columns={price_col: "lmp_usd_per_mwh"})
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return out, available


# -----------------------------
# Renewables from fuel mix
# -----------------------------
def compute_renewables_mw_from_fuel_mix(fuel_mix_df: pd.DataFrame) -> pd.DataFrame:
    if fuel_mix_df is None or fuel_mix_df.empty:
        return pd.DataFrame(columns=["time", "renewables_mw"])

    df = parse_time_col(fuel_mix_df)

    value_cols = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        return pd.DataFrame(columns=["time", "renewables_mw"])

    def is_renewable(col: str) -> bool:
        return any(k.lower() == col.lower() for k in RENEWABLE_KEYS)

    ren_cols = [c for c in value_cols if is_renewable(c)]
    if not ren_cols:
        return pd.DataFrame(columns=["time", "renewables_mw"])

    out = pd.DataFrame({"time": df["time"]})
    out["renewables_mw"] = df[ren_cols].sum(axis=1)
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return out


# -----------------------------
# Ridge feature engineering (hourly)
# -----------------------------
def _hourly_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Returns hourly mean series with columns [time, value] on an hourly grid.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "value"])

    df = parse_time_col(df)
    if value_col not in df.columns:
        return pd.DataFrame(columns=["time", "value"])

    tmp = df[["time", value_col]].rename(columns={value_col: "value"}).copy()
    tmp["time"] = tmp["time"].dt.floor("H")
    tmp = tmp.groupby("time")["value"].mean().reset_index()
    tmp = tmp.sort_values("time")
    return tmp


def _time_features(times: pd.Series) -> pd.DataFrame:
    """
    Cyclical time features from timestamp.
    """
    # times is tz-aware
    t = times.dt.tz_convert(TZ)
    hour = t.dt.hour.astype(int)
    dow = t.dt.dayofweek.astype(int)

    # cyclical encoding
    hour_rad = 2 * np.pi * hour / 24.0
    dow_rad = 2 * np.pi * dow / 7.0

    return pd.DataFrame(
        {
            "hour_sin": np.sin(hour_rad),
            "hour_cos": np.cos(hour_rad),
            "dow_sin": np.sin(dow_rad),
            "dow_cos": np.cos(dow_rad),
        },
        index=times.index,
    )


def _lag_features(values: pd.Series) -> pd.DataFrame:
    """
    Lag and rolling stats features from a numeric series (hourly).
    IMPORTANT: this is computed on a *complete* time-aligned series.
    """
    v = values.astype(float)
    return pd.DataFrame(
        {
            "lag1": v.shift(1),
            "lag24": v.shift(24),
            "roll6": v.rolling(6).mean(),
            "roll24": v.rolling(24).mean(),
        },
        index=values.index,
    )


@dataclass
class RidgeForecaster:
    pipeline: Pipeline

    def predict_next(self, X_row: np.ndarray) -> float:
        return float(self.pipeline.predict(X_row.reshape(1, -1))[0])


def _fit_ridge_model(training_hourly: pd.DataFrame) -> Optional[RidgeForecaster]:
    """
    Train ridge regression on hourly series with time+lag features.
    Returns None if not enough data.
    """
    if training_hourly is None or training_hourly.empty:
        return None

    df = training_hourly.copy().reset_index(drop=True)
    df = df.sort_values("time")

    # Build features
    tf = _time_features(df["time"])
    lf = _lag_features(df["value"])
    X = pd.concat([tf, lf], axis=1)

    y = df["value"].astype(float)

    # Drop rows with missing lags/rolls (first ~24 hours)
    ok = ~X.isna().any(axis=1) & ~y.isna()
    X = X.loc[ok]
    y = y.loc[ok]

    if len(X) < 200:  # safety: need enough hourly points
        return None

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=42)),
        ]
    )
    pipe.fit(X.values, y.values)
    return RidgeForecaster(pipeline=pipe)


def _get_or_train_model(target: str, training_hourly: pd.DataFrame) -> Optional[RidgeForecaster]:
    """
    Cached model: retrains at most every MODEL_TTL_SECONDS.
    """
    key = _cache_key("ridge_model", target=target, days=TRAINING_DAYS, alpha=RIDGE_ALPHA)
    cached = MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        model = _fit_ridge_model(training_hourly)
        if model is None:
            return None
        MODEL_CACHE[key] = model
        return model
    except Exception:
        log.exception("Ridge training failed for %s", target)
        return None


def _predict_future_hourly(
    history_hourly: pd.DataFrame,
    future_times: pd.DatetimeIndex,
    model: Optional[RidgeForecaster],
) -> pd.Series:
    """
    Recursive hourly prediction using ridge (so lags use predicted values).
    Fallback: hour-of-week median if ridge unavailable.
    """
    # Build a full timeline including history and future
    hist = history_hourly.copy().sort_values("time")
    hist = hist.dropna(subset=["time"])
    hist = hist.set_index("time")

    # Ensure hourly frequency on history (reindex + interpolate small gaps)
    if not hist.empty:
        full_hist_index = pd.date_range(hist.index.min().floor("H"), hist.index.max().floor("H"), freq="1H", tz=TZ)
        hist = hist.reindex(full_hist_index)
        # Don't get fancy: fill small gaps via time interpolation, remaining via median
        hist["value"] = hist["value"].interpolate(method="time", limit=6)
        hist["value"] = hist["value"].fillna(hist["value"].median())

    # Fallback medians for hour-of-week
    def hour_of_week(dt: pd.Timestamp) -> int:
        return int(dt.dayofweek) * 24 + int(dt.hour)

    how_medians: Dict[int, float] = {}
    fallback = float("nan")
    if not hist.empty:
        how = [hour_of_week(t) for t in hist.index]
        tmp = pd.DataFrame({"how": how, "v": hist["value"].values})
        how_medians = tmp.groupby("how")["v"].median().to_dict()
        fallback = float(tmp["v"].median())

    # Start series with history values
    combined_index = hist.index.union(future_times)
    combined_index = combined_index.sort_values()
    series = pd.Series(index=combined_index, dtype=float)

    if not hist.empty:
        series.loc[hist.index] = hist["value"].astype(float).values

    # Predict forward one hour at a time
    for t in future_times:
        if pd.notna(series.get(t)):
            continue

        # Build features using current series (lags/rolls)
        # Need previous hours present; if not, fallback
        # Create a small window up to t
        # We'll compute lags on a dataframe for simplicity
        idx = series.index
        # Make sure t exists
        if t not in idx:
            continue

        # Construct a mini frame
        # (rolling needs enough points; if not, fallback)
        loc = idx.get_loc(t)
        if isinstance(loc, slice) or isinstance(loc, np.ndarray):
            # shouldn't happen
            loc = int(loc[0])

        # If we don't have 24h of back context, fallback
        if loc < 25:
            pred = how_medians.get(hour_of_week(t), fallback)
            series.loc[t] = pred
            continue

        # Build feature row
        time_feat = _time_features(pd.Series([t], dtype="datetime64[ns, US/Pacific]")).iloc[0]

        # compute lags/rolls from series
        lag1 = series.iloc[loc - 1]
        lag24 = series.iloc[loc - 24]
        roll6 = series.iloc[loc - 6 : loc].mean()
        roll24 = series.iloc[loc - 24 : loc].mean()

        # If missing for any reason, fallback
        if any(pd.isna(x) for x in [lag1, lag24, roll6, roll24]):
            pred = how_medians.get(hour_of_week(t), fallback)
            series.loc[t] = pred
            continue

        X_row = np.array([time_feat["hour_sin"], time_feat["hour_cos"], time_feat["dow_sin"], time_feat["dow_cos"], lag1, lag24, roll6, roll24], dtype=float)

        if model is None:
            pred = how_medians.get(hour_of_week(t), fallback)
        else:
            try:
                pred = model.predict_next(X_row)
            except Exception:
                pred = how_medians.get(hour_of_week(t), fallback)

        series.loc[t] = pred

    return series.loc[future_times]


# -----------------------------
# Scoring + windows
# -----------------------------
def normalize_01(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    if x.dropna().empty:
        return pd.Series([float("nan")] * len(x), index=x.index)

    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-9:
        return pd.Series([0.5] * len(x), index=x.index)
    return (x - lo) / (hi - lo)


def duration_to_minutes(s: str) -> int:
    if not s:
        return 0
    s = s.strip().lower()
    if s.endswith("min"):
        return int(float(s[:-3]))
    raise ValueError(f"Unsupported duration: {s}")


def build_windows(
    series_df: pd.DataFrame,
    score_col: str,
    score_quantile: float,
    min_minutes: int,
    max_minutes: Optional[int],
) -> List[Dict[str, Any]]:
    df = series_df.copy().sort_values("time").reset_index(drop=True)
    s = df[score_col].astype(float)
    thresh = float(s.quantile(score_quantile)) if s.dropna().size else float("nan")
    good = s >= thresh

    windows: List[Dict[str, Any]] = []
    i = 0
    while i < len(df):
        if not bool(good.iloc[i]):
            i += 1
            continue
        j = i
        while j < len(df) and bool(good.iloc[j]):
            j += 1

        start = df.loc[i, "time"]
        end = df.loc[j - 1, "time"] + timedelta(hours=1)
        dur_min = int((end - start).total_seconds() / 60)

        if dur_min >= min_minutes and (max_minutes is None or dur_min <= max_minutes):
            block = df.iloc[i:j]
            avg_score = float(block[score_col].mean(skipna=True)) if block[score_col].dropna().size else None
            avg_net = float(block["net_load_forecast_mw"].mean(skipna=True)) if block["net_load_forecast_mw"].dropna().size else None
            windows.append(
                {
                    "start": to_iso(start),
                    "end": to_iso(end),
                    "duration_minutes": dur_min,
                    "avg_net_load": avg_net,
                    "avg_score": avg_score,
                }
            )

        i = j

    return windows


def intersect_windows(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def parse_iso(x: str) -> datetime:
        dt = pd.to_datetime(x)
        if dt.tzinfo is None:
            return TZ.localize(dt.to_pydatetime())
        return dt.tz_convert(TZ).to_pydatetime()

    for wa in a:
        sa = parse_iso(wa["start"])
        ea = parse_iso(wa["end"])
        for wb in b:
            sb = parse_iso(wb["start"])
            eb = parse_iso(wb["end"])
            s = max(sa, sb)
            e = min(ea, eb)
            if e > s:
                dur_min = int((e - s).total_seconds() / 60)
                out.append(
                    {
                        "start": to_iso(s),
                        "end": to_iso(e),
                        "duration_minutes": dur_min,
                        "avg_net_load": None,
                        "avg_score": None,
                    }
                )
    return sorted(out, key=lambda w: w["start"])


# -----------------------------
# Build series (keeps your /api/next24 schema)
# -----------------------------
def build_series(hours: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    now = now_pacific()
    horizon_end = now + timedelta(hours=hours)
    cutoff_time = now + timedelta(hours=FORECAST_CUTOFF_HOURS)

    # hourly grid starting next hour
    start_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    times = pd.date_range(start=start_hour, end=horizon_end, freq="1H", tz=TZ, inclusive="left")
    base = pd.DataFrame({"time": times})

    # --- Forecast (best effort) ---
    lf_raw = get_load_forecast(end=horizon_end)
    lf_raw = parse_time_col(lf_raw)
    load_col = None
    if not lf_raw.empty:
        for c in lf_raw.columns:
            if c.lower() in {"load", "load (mw)", "mw", "demand", "demand (mw)"} and c != "time":
                load_col = c
                break
        if load_col is None:
            load_col = first_numeric_col(lf_raw, exclude={"time"})
    load_forecast = pd.DataFrame(columns=["time", "load_forecast_mw"])
    if not lf_raw.empty and load_col:
        load_forecast = _hourly_series(lf_raw, load_col).rename(columns={"value": "load_forecast_mw"})

    fm_f = get_fuel_mix_forecast(end=horizon_end)
    ren_forecast = compute_renewables_mw_from_fuel_mix(fm_f)
    if not ren_forecast.empty:
        ren_forecast = _hourly_series(ren_forecast, "renewables_mw").rename(columns={"value": "renewables_forecast_mw"})
    else:
        ren_forecast = pd.DataFrame(columns=["time", "renewables_forecast_mw"])

    # Merge forecasts
    out = base.merge(load_forecast, on="time", how="left")
    out = out.merge(ren_forecast, on="time", how="left")

    # --- Training data (history) ---
    train_end = now
    train_start = train_end - timedelta(days=TRAINING_DAYS)

    # Load history hourly
    lh = get_load_history(train_start, train_end)
    lh = parse_time_col(lh)
    hist_load_col = None
    if not lh.empty:
        for c in lh.columns:
            if c.lower() in {"load", "load (mw)", "mw", "demand", "demand (mw)"} and c != "time":
                hist_load_col = c
                break
        if hist_load_col is None:
            hist_load_col = first_numeric_col(lh, exclude={"time"})
    load_hist_hourly = _hourly_series(lh, hist_load_col) if (hist_load_col and not lh.empty) else pd.DataFrame(columns=["time", "value"])

    # Renewables history hourly
    fm_h = get_fuel_mix_history(train_start, train_end)
    ren_h = compute_renewables_mw_from_fuel_mix(fm_h)
    ren_hist_hourly = _hourly_series(ren_h, "renewables_mw") if not ren_h.empty else pd.DataFrame(columns=["time", "value"])

    # --- Ridge models (cached) ---
    load_model = _get_or_train_model("load", load_hist_hourly)
    ren_model = _get_or_train_model("renewables", ren_hist_hourly)

    # --- Predict beyond cutoff (and fill missing within cutoff) ---
    out["is_predicted"] = out["time"] > cutoff_time

    # Future times for prediction (strictly > cutoff)
    future_times = pd.DatetimeIndex(out.loc[out["time"] > cutoff_time, "time"].values).tz_convert(TZ)
    if len(future_times) > 0:
        load_future = _predict_future_hourly(load_hist_hourly, future_times, load_model)
        ren_future = _predict_future_hourly(ren_hist_hourly, future_times, ren_model)

        out.loc[out["time"] > cutoff_time, "load_forecast_mw"] = load_future.values
        out.loc[out["time"] > cutoff_time, "renewables_forecast_mw"] = ren_future.values

    # If forecast missing inside cutoff, backfill with ridge predictions too (but not marked predicted)
    inside = out["time"] <= cutoff_time
    missing_load = inside & out["load_forecast_mw"].isna()
    missing_ren = inside & out["renewables_forecast_mw"].isna()

    if missing_load.any():
        # predict those timestamps using recursive routine, but we only need those times
        tmiss = pd.DatetimeIndex(out.loc[missing_load, "time"].values).tz_convert(TZ)
        pred = _predict_future_hourly(load_hist_hourly, tmiss, load_model)
        out.loc[missing_load, "load_forecast_mw"] = pred.values

    if missing_ren.any():
        tmiss = pd.DatetimeIndex(out.loc[missing_ren, "time"].values).tz_convert(TZ)
        pred = _predict_future_hourly(ren_hist_hourly, tmiss, ren_model)
        out.loc[missing_ren, "renewables_forecast_mw"] = pred.values

    # Net load
    out["net_load_forecast_mw"] = out["load_forecast_mw"] - out["renewables_forecast_mw"]

    # Green score: renewables share (normalized)
    share = (out["renewables_forecast_mw"] / out["load_forecast_mw"]).clip(lower=0, upper=1)
    out["green_score"] = normalize_01(share)

    # LMP (market-only) - fill hourly and cut off at cutoff, never predicted
    out["lmp_usd_per_mwh"] = float("nan")
    lmp_df, lmp_available = get_lmp_series(end=horizon_end)
    if lmp_available and not lmp_df.empty:
        lmp_df = lmp_df.copy()
        lmp_df["time"] = lmp_df["time"].dt.floor("H")
        lmp_hourly = lmp_df.groupby("time")["lmp_usd_per_mwh"].mean().reset_index()
        out = out.merge(lmp_hourly, on="time", how="left", suffixes=("", "_y"))
        if "lmp_usd_per_mwh_y" in out.columns:
            out["lmp_usd_per_mwh"] = out["lmp_usd_per_mwh_y"]
            out = out.drop(columns=["lmp_usd_per_mwh_y"])

    # hard cutoff for LMP
    out.loc[out["time"] > cutoff_time, "lmp_usd_per_mwh"] = float("nan")

    # Cheap score (only where LMP exists)
    if out["lmp_usd_per_mwh"].dropna().empty:
        out["cheap_score"] = float("nan")
    else:
        out["cheap_score"] = 1.0 - normalize_01(out["lmp_usd_per_mwh"])

    meta = {
        "horizon_start": to_iso(start_hour),
        "horizon_end": to_iso(horizon_end),
        "horizon_hours": hours,
        "forecast_cutoff_hours": FORECAST_CUTOFF_HOURS,
        "forecast_cutoff_time": to_iso(cutoff_time),
        "lmp_available": bool(lmp_available and not out["lmp_usd_per_mwh"].dropna().empty),
        "ridge_cached_ttl_seconds": MODEL_TTL_SECONDS,
        "ridge_alpha": RIDGE_ALPHA,
        "training_days": TRAINING_DAYS,
    }

    return out, meta


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "timezone": str(TZ),
        "fetch_cache_ttl_seconds": CACHE_TTL_SECONDS,
        "fetch_cache_size": len(FETCH_CACHE),
        "model_cache_ttl_seconds": MODEL_TTL_SECONDS,
        "model_cache_size": len(MODEL_CACHE),
        "training_days": TRAINING_DAYS,
        "forecast_cutoff_hours": FORECAST_CUTOFF_HOURS,
        "lmp_node": LMP_NODE,
        "now_pacific": to_iso(now_pacific()),
    }


@app.get("/api/next24")
def api_next24(
    hours: int = Query(24, ge=1, le=72),
    score_quantile: float = Query(0.80, ge=0.50, le=0.95),
    min_duration: str = Query("60min"),
    max_duration: str = Query("", description="Optional, e.g. 180min"),
    green_weight: float = Query(0.5, ge=0.0, le=1.0),
    cheap_weight: float = Query(0.5, ge=0.0, le=1.0),
) -> JSONResponse:
    series_df, meta = build_series(hours=hours)

    min_minutes = duration_to_minutes(min_duration)
    max_minutes = duration_to_minutes(max_duration) if max_duration else None

    gw = float(green_weight)
    cw = float(cheap_weight)
    denom = gw + cw
    if denom <= 1e-9:
        gw, cw, denom = 1.0, 0.0, 1.0

    gs = series_df["green_score"].astype(float)
    cs = series_df["cheap_score"].astype(float)
    cs_for_combo = cs.fillna(0.5)
    series_df["combined_score"] = (gw * gs + cw * cs_for_combo) / denom

    windows_green = build_windows(series_df, "green_score", score_quantile, min_minutes, max_minutes)
    windows_cheap = build_windows(series_df, "cheap_score", score_quantile, min_minutes, max_minutes) if meta["lmp_available"] else []
    windows_both = intersect_windows(windows_green, windows_cheap) if windows_green and windows_cheap else []

    windows_default = windows_both
    if not windows_default:
        windows_default = build_windows(series_df, "combined_score", score_quantile, min_minutes, max_minutes)

    records: List[Dict[str, Any]] = []
    for _, r in series_df.iterrows():
        records.append(
            {
                "t_start": to_iso(r["time"].to_pydatetime()),
                "load_forecast_mw": None if pd.isna(r["load_forecast_mw"]) else float(r["load_forecast_mw"]),
                "renewables_forecast_mw": None if pd.isna(r["renewables_forecast_mw"]) else float(r["renewables_forecast_mw"]),
                "net_load_forecast_mw": None if pd.isna(r["net_load_forecast_mw"]) else float(r["net_load_forecast_mw"]),
                "green_score": None if pd.isna(r["green_score"]) else float(r["green_score"]),
                "cheap_score": None if pd.isna(r["cheap_score"]) else float(r["cheap_score"]),
                "combined_score": None if pd.isna(r["combined_score"]) else float(r["combined_score"]),
                "lmp_usd_per_mwh": None if pd.isna(r["lmp_usd_per_mwh"]) else float(r["lmp_usd_per_mwh"]),
                "is_predicted": bool(r["is_predicted"]),
            }
        )

    meta_out = {
        **meta,
        "score_quantile": float(score_quantile),
        "min_duration": min_duration,
        "max_duration": max_duration if max_duration else None,
        "green_weight": float(gw),
        "cheap_weight": float(cw),
    }

    return JSONResponse(
        {
            "meta": meta_out,
            "series": records,
            "windows": windows_default,
            "windows_green": windows_green,
            "windows_cheap": windows_cheap,
            "windows_both": windows_both,
        }
    )
