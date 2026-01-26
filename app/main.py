from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import pytz
from cachetools import TTLCache
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import gridstatus


# -----------------------------
# Config
# -----------------------------
TZ = pytz.timezone("US/Pacific")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))  # 10 min default
CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "256"))

# In-memory TTL cache (fine for single-instance Render services)
CACHE = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)

caiso = gridstatus.CAISO()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------------
# Cache helpers
# -----------------------------
def _cache_key(prefix: str, **kwargs) -> str:
    items = sorted((k, str(v)) for k, v in kwargs.items())
    return prefix + "|" + "|".join([f"{k}={v}" for k, v in items])


def cached_get(prefix: str, fetch_fn, **kwargs):
    key = _cache_key(prefix, **kwargs)
    if key in CACHE:
        return CACHE[key]
    val = fetch_fn(**kwargs)
    CACHE[key] = val
    return val


# -----------------------------
# Data fetching via gridstatus (CAISO)
# -----------------------------
def fetch_day_ahead_load_forecast(start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    """Hourly day-ahead load forecast (CA ISO-TAC only)."""
    def _fetch(date, end):
        df = caiso.get_load_forecast_day_ahead(date, end=end)
        df = df[df["TAC Area Name"] == "CA ISO-TAC"].copy()
        return df

    df = cached_get("load_forecast_dam", _fetch, date=start_day, end=end_day)
    df = df.copy()
    df["Interval Start"] = pd.to_datetime(df["Interval Start"])
    df["Interval End"] = pd.to_datetime(df["Interval End"])
    return df[["Interval Start", "Interval End", "Load Forecast"]].sort_values("Interval Start")


def fetch_day_ahead_solar_wind_forecast(start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    """Hourly day-ahead solar+wind forecast (Location == 'CAISO' totals)."""
    def _fetch(date, end):
        return caiso.get_solar_and_wind_forecast_dam(date, end=end)

    df = cached_get("solar_wind_forecast_dam", _fetch, date=start_day, end=end_day)
    df = df.copy()
    df["Interval Start"] = pd.to_datetime(df["Interval Start"])
    df["Interval End"] = pd.to_datetime(df["Interval End"])

    df = df[df["Location"] == "CAISO"].copy()

    # Some versions expose "Solar MW" and "Wind MW"
    if "Solar MW" not in df.columns:
        df["Solar MW"] = pd.NA
    if "Wind MW" not in df.columns:
        df["Wind MW"] = pd.NA

    df["Renewables Forecast"] = pd.to_numeric(df["Solar MW"], errors="coerce").fillna(0) + pd.to_numeric(
        df["Wind MW"], errors="coerce"
    ).fillna(0)

    keep = ["Interval Start", "Interval End", "Solar MW", "Wind MW", "Renewables Forecast"]
    return df[keep].sort_values("Interval Start")


# -----------------------------
# Scoring + windows
# -----------------------------
def minmax_normalize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def contiguous_windows(
    df: pd.DataFrame,
    mask: pd.Series,
    min_duration: str = "60min",
) -> List[Dict[str, Any]]:
    """Build contiguous windows from an hourly mask."""
    tmp = df.copy().sort_values("Interval Start").reset_index(drop=True)
    m = mask.fillna(False).reset_index(drop=True)
    if tmp.empty:
        return []

    windows: List[Dict[str, Any]] = []
    start_idx = None

    for i, is_ok in enumerate(m):
        if is_ok and start_idx is None:
            start_idx = i
        if (not is_ok or i == len(m) - 1) and start_idx is not None:
            end_idx = i if is_ok and i == len(m) - 1 else i - 1
            start = tmp.loc[start_idx, "Interval Start"]
            end = tmp.loc[end_idx, "Interval End"]
            duration = end - start
            if duration >= pd.Timedelta(min_duration):
                chunk = tmp.loc[start_idx:end_idx]
                windows.append(
                    {
                        "start": pd.Timestamp(start).isoformat(),
                        "end": pd.Timestamp(end).isoformat(),
                        "duration_minutes": int(duration.total_seconds() // 60),
                        "avg_net_load": float(pd.to_numeric(chunk["Net Load Forecast"], errors="coerce").mean()),
                        "avg_green_score": float(pd.to_numeric(chunk["Green Score"], errors="coerce").mean()),
                    }
                )
            start_idx = None

    return windows


def build_next24_dataset(
    score_quantile: float = 0.80,
    min_duration: str = "60min",
) -> Dict[str, Any]:
    """Next-24h hourly series with recommended GREEN training windows."""
    now = pd.Timestamp.now(tz=TZ)
    horizon_start = now.floor("H")
    horizon_end = horizon_start + pd.Timedelta(hours=24)

    # Fetch enough day-ahead data covering today + tomorrow
    start_day = horizon_start.normalize()
    end_day = horizon_end.normalize() + pd.Timedelta(days=1)

    load_fc = fetch_day_ahead_load_forecast(start_day, end_day)
    ren_fc = fetch_day_ahead_solar_wind_forecast(start_day, end_day)

    df = load_fc.merge(ren_fc, on=["Interval Start", "Interval End"], how="left")

    # Net load proxy
    df["Net Load Forecast"] = pd.to_numeric(df["Load Forecast"], errors="coerce") - pd.to_numeric(
        df["Renewables Forecast"], errors="coerce"
    )

    # Slice to next 24h
    df = df[(df["Interval Start"] >= horizon_start) & (df["Interval Start"] < horizon_end)].copy()
    df = df.sort_values("Interval Start").reset_index(drop=True)

    # Normalize net load (lower is greener) -> green score higher is better
    netload_norm = minmax_normalize(df["Net Load Forecast"])
    df["Green Score"] = 1.0 - netload_norm

    threshold = float(df["Green Score"].quantile(score_quantile)) if not df.empty else None
    mask = df["Green Score"] >= threshold if threshold is not None else pd.Series([], dtype=bool)

    windows = contiguous_windows(df, mask, min_duration=min_duration)

    # Serialize series for plotting
    series = []
    for _, r in df.iterrows():
        series.append(
            {
                "t_start": pd.Timestamp(r["Interval Start"]).isoformat(),
                "t_end": pd.Timestamp(r["Interval End"]).isoformat(),
                "load_forecast_mw": float(r["Load Forecast"]) if pd.notna(r["Load Forecast"]) else None,
                "solar_mw": float(r["Solar MW"]) if pd.notna(r.get("Solar MW")) else None,
                "wind_mw": float(r["Wind MW"]) if pd.notna(r.get("Wind MW")) else None,
                "renewables_forecast_mw": float(r["Renewables Forecast"]) if pd.notna(r["Renewables Forecast"]) else None,
                "net_load_forecast_mw": float(r["Net Load Forecast"]) if pd.notna(r["Net Load Forecast"]) else None,
                "green_score": float(r["Green Score"]) if pd.notna(r["Green Score"]) else None,
            }
        )

    return {
        "meta": {
            "timezone": "US/Pacific",
            "generated_at": now.isoformat(),
            "horizon_start": horizon_start.isoformat(),
            "horizon_end": horizon_end.isoformat(),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "strategy": "green",
            "score_quantile": score_quantile,
            "min_duration": min_duration,
            "threshold_used": "Green Score",
            "threshold_value": threshold,
        },
        "series": series,
        "windows": windows,
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/api/next24")
def api_next24(
    score_quantile: float = Query(0.80, ge=0.50, le=0.95),
    min_duration: str = Query("60min"),
):
    """Returns next-24h hourly series + recommended GREEN training windows."""
    return build_next24_dataset(score_quantile=score_quantile, min_duration=min_duration)
