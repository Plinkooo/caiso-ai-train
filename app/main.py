from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import gridstatus

log = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------
TZ = pytz.timezone("US/Pacific")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "256"))

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
# Small helpers
# -----------------------------
def minmax_normalize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def safe_quantile(s: pd.Series, q: float) -> Optional[float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.quantile(q))


def pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def find_lmp_column(df: pd.DataFrame) -> Optional[str]:
    direct = pick_first_existing_column(df, ["LMP", "LMP ($/MWh)", "LMP $/MWh", "lmp", "price"])
    if direct:
        return direct
    for c in df.columns:
        if "lmp" in c.lower():
            return c
    return None


def ensure_hourly_intervals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Interval Start" in df.columns and "Interval End" in df.columns:
        df["Interval Start"] = pd.to_datetime(df["Interval Start"])
        df["Interval End"] = pd.to_datetime(df["Interval End"])
        return df

    if "Time" in df.columns:
        df["Interval Start"] = pd.to_datetime(df["Time"])
        df["Interval End"] = df["Interval Start"] + pd.Timedelta(hours=1)
        return df

    return pd.DataFrame(columns=["Interval Start", "Interval End"])


# -----------------------------
# Data fetching via gridstatus (CAISO)
# -----------------------------
def fetch_day_ahead_load_forecast(start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    def _fetch(date, end):
        fn = getattr(caiso, "get_load_forecast_day_ahead", None)
        if callable(fn):
            df = fn(date, end=end)
        else:
            fn2 = getattr(caiso, "get_load_forecast", None)
            if not callable(fn2):
                raise RuntimeError("gridstatus.CAISO() has no load forecast method")
            df = fn2(date, end=end)

        if "TAC Area Name" in df.columns:
            df = df[df["TAC Area Name"] == "CA ISO-TAC"].copy()

        return df

    df = cached_get("load_forecast_dam", _fetch, date=start_day, end=end_day).copy()

    if df.empty:
        return pd.DataFrame(columns=["Interval Start", "Interval End", "Load Forecast"])

    df["Interval Start"] = pd.to_datetime(df["Interval Start"])
    df["Interval End"] = pd.to_datetime(df["Interval End"])

    if "Load Forecast" not in df.columns:
        cand = pick_first_existing_column(df, ["Load", "Forecast", "Load (MW)", "MW"])
        df["Load Forecast"] = pd.to_numeric(df[cand], errors="coerce") if cand else pd.NA

    return df[["Interval Start", "Interval End", "Load Forecast"]].sort_values("Interval Start")


def fetch_day_ahead_solar_wind_forecast(start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    def _fetch(date, end):
        fn = getattr(caiso, "get_solar_and_wind_forecast_dam", None)
        if callable(fn):
            return fn(date, end=end)

        fn = getattr(caiso, "get_renewables_forecast_dam", None)
        if callable(fn):
            return fn(date, end=end)

        return pd.DataFrame()

    df = cached_get("renewables_dam", _fetch, date=start_day, end=end_day).copy()

    if df.empty:
        hours = pd.date_range(start_day, end_day, freq="H", tz=TZ)
        out = pd.DataFrame(
            {
                "Interval Start": hours[:-1],
                "Interval End": hours[1:],
                "Solar MW": 0.0,
                "Wind MW": 0.0,
                "Renewables Forecast": 0.0,
            }
        )
        return out.sort_values("Interval Start")

    df["Interval Start"] = pd.to_datetime(df["Interval Start"])
    df["Interval End"] = pd.to_datetime(df["Interval End"])

    if "Location" in df.columns:
        df = df[df["Location"] == "CAISO"].copy()

    if "Solar MW" not in df.columns:
        df["Solar MW"] = pd.NA
    if "Wind MW" not in df.columns:
        df["Wind MW"] = pd.NA

    df["Renewables Forecast"] = (
        pd.to_numeric(df["Solar MW"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Wind MW"], errors="coerce").fillna(0)
    )

    keep = ["Interval Start", "Interval End", "Solar MW", "Wind MW", "Renewables Forecast"]
    return df[keep].sort_values("Interval Start")


def fetch_day_ahead_lmp(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    locations: Optional[List[str]] = None,
) -> pd.DataFrame:
    if locations is None:
        locations = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]

    def _fetch(date, end, locations_key: str):
        fn = getattr(caiso, "get_lmp", None)
        if not callable(fn):
            return pd.DataFrame()
        return fn(
            date=date,
            end=end,
            market="DAY_AHEAD_HOURLY",
            locations=locations,
        )

    df = cached_get(
        "lmp_dah",
        _fetch,
        date=start_day,
        end=end_day,
        locations_key=",".join(locations),
    )
    if df is None:
        df = pd.DataFrame()

    df = df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Interval Start", "Interval End", "lmp_usd_per_mwh"])

    df = ensure_hourly_intervals(df)
    if df.empty:
        return pd.DataFrame(columns=["Interval Start", "Interval End", "lmp_usd_per_mwh"])

    lmp_col = find_lmp_column(df)
    if not lmp_col:
        return pd.DataFrame(columns=["Interval Start", "Interval End", "lmp_usd_per_mwh"])

    df["lmp_usd_per_mwh"] = pd.to_numeric(df[lmp_col], errors="coerce")

    out = (
        df.groupby(["Interval Start", "Interval End"], as_index=False)["lmp_usd_per_mwh"]
        .mean()
        .sort_values("Interval Start")
    )
    return out


# -----------------------------
# Windows
# -----------------------------
def contiguous_windows(
    df: pd.DataFrame,
    mask: pd.Series,
    min_duration: str = "60min",
    max_duration: Optional[str] = None,
    score_col: str = "Green Score",
    netload_col: str = "Net Load Forecast",
) -> List[Dict[str, Any]]:
    tmp = df.copy().sort_values("Interval Start").reset_index(drop=True)
    m = mask.fillna(False).reset_index(drop=True)
    if tmp.empty:
        return []

    min_td = pd.Timedelta(min_duration)
    max_td = pd.Timedelta(max_duration) if max_duration else None

    windows: List[Dict[str, Any]] = []
    start_idx = None

    def _add_window(i0: int, i1: int):
        start = tmp.loc[i0, "Interval Start"]
        end = tmp.loc[i1, "Interval End"]
        duration = end - start
        if duration < min_td:
            return
        chunk = tmp.loc[i0:i1]
        windows.append(
            {
                "start": pd.Timestamp(start).isoformat(),
                "end": pd.Timestamp(end).isoformat(),
                "duration_minutes": int(duration.total_seconds() // 60),
                "avg_net_load": float(pd.to_numeric(chunk[netload_col], errors="coerce").mean())
                if netload_col in chunk.columns
                else None,
                "avg_score": float(pd.to_numeric(chunk[score_col], errors="coerce").mean())
                if score_col in chunk.columns
                else None,
            }
        )

    for i, is_ok in enumerate(m):
        if is_ok and start_idx is None:
            start_idx = i

        if (not is_ok or i == len(m) - 1) and start_idx is not None:
            end_idx = i if (is_ok and i == len(m) - 1) else i - 1

            if not max_td:
                _add_window(start_idx, end_idx)
                start_idx = None
                continue

            chunk_start = start_idx
            while chunk_start <= end_idx:
                chunk_end = chunk_start
                while chunk_end < end_idx:
                    start = tmp.loc[chunk_start, "Interval Start"]
                    next_end = tmp.loc[chunk_end + 1, "Interval End"]
                    if (next_end - start) <= max_td:
                        chunk_end += 1
                    else:
                        break
                _add_window(chunk_start, chunk_end)
                chunk_start = chunk_end + 1

            start_idx = None

    return windows


# -----------------------------
# Dataset build: green + cheap + intersection
# -----------------------------
def build_forecast_dataset(
    hours: int = 24,
    score_quantile: float = 0.80,
    min_duration: str = "60min",
    max_duration: Optional[str] = None,
    green_weight: float = 0.5,
    cheap_weight: float = 0.5,
) -> Dict[str, Any]:
    if hours not in (24, 48, 72):
        raise ValueError("hours must be 24, 48, or 72")

    now = pd.Timestamp.now(tz=TZ)
    horizon_start = now.floor("H")
    horizon_end = horizon_start + pd.Timedelta(hours=hours)

    start_day = horizon_start.normalize()
    end_day = horizon_end.normalize() + pd.Timedelta(days=1)

    load_fc = fetch_day_ahead_load_forecast(start_day, end_day)
    ren_fc = fetch_day_ahead_solar_wind_forecast(start_day, end_day)
    lmp_df = fetch_day_ahead_lmp(start_day, end_day)

    df = load_fc.merge(ren_fc, on=["Interval Start", "Interval End"], how="left")
    df = df.merge(lmp_df, on=["Interval Start", "Interval End"], how="left")

    df["Net Load Forecast"] = pd.to_numeric(df["Load Forecast"], errors="coerce") - pd.to_numeric(
        df["Renewables Forecast"], errors="coerce"
    )

    df = df[(df["Interval Start"] >= horizon_start) & (df["Interval Start"] < horizon_end)].copy()
    df = df.sort_values("Interval Start").reset_index(drop=True)

    df["Green Score"] = 1.0 - minmax_normalize(df["Net Load Forecast"])

    lmp_available = "lmp_usd_per_mwh" in df.columns and df["lmp_usd_per_mwh"].notna().any()
    if lmp_available:
        df["Cheap Score"] = 1.0 - minmax_normalize(df["lmp_usd_per_mwh"])
    else:
        df["Cheap Score"] = pd.NA

    gw = float(green_weight)
    cw = float(cheap_weight)
    if gw < 0:
        gw = 0.0
    if cw < 0:
        cw = 0.0
    if gw == 0 and cw == 0:
        gw = 1.0
        cw = 0.0
    wsum = gw + cw

    if lmp_available:
        df["Combined Score"] = (gw * df["Green Score"] + cw * df["Cheap Score"]) / wsum
    else:
        df["Combined Score"] = df["Green Score"]

    green_thr = safe_quantile(df["Green Score"], score_quantile)
    cheap_thr = safe_quantile(df["Cheap Score"], score_quantile) if lmp_available else None

    green_mask = (df["Green Score"] >= green_thr) if green_thr is not None else pd.Series([False] * len(df))
    cheap_mask = (df["Cheap Score"] >= cheap_thr) if cheap_thr is not None else pd.Series([False] * len(df))
    both_mask = (
        (green_mask & cheap_mask) if (green_thr is not None and cheap_thr is not None) else pd.Series([False] * len(df))
    )

    windows_green = contiguous_windows(df, green_mask, min_duration=min_duration, max_duration=max_duration, score_col="Green Score")
    windows_cheap = contiguous_windows(df, cheap_mask, min_duration=min_duration, max_duration=max_duration, score_col="Cheap Score")
    windows_both = contiguous_windows(df, both_mask, min_duration=min_duration, max_duration=max_duration, score_col="Combined Score")

    series: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        series.append(
            {
                "t_start": pd.Timestamp(r["Interval Start"]).isoformat(),
                "t_end": pd.Timestamp(r["Interval End"]).isoformat(),
                "load_forecast_mw": float(r["Load Forecast"]) if pd.notna(r.get("Load Forecast")) else None,
                "solar_mw": float(r["Solar MW"]) if pd.notna(r.get("Solar MW")) else None,
                "wind_mw": float(r["Wind MW"]) if pd.notna(r.get("Wind MW")) else None,
                "renewables_forecast_mw": float(r["Renewables Forecast"]) if pd.notna(r.get("Renewables Forecast")) else None,
                "net_load_forecast_mw": float(r["Net Load Forecast"]) if pd.notna(r.get("Net Load Forecast")) else None,
                "lmp_usd_per_mwh": float(r["lmp_usd_per_mwh"]) if pd.notna(r.get("lmp_usd_per_mwh")) else None,
                "green_score": float(r["Green Score"]) if pd.notna(r.get("Green Score")) else None,
                "cheap_score": float(r["Cheap Score"]) if pd.notna(r.get("Cheap Score")) else None,
                "combined_score": float(r["Combined Score"]) if pd.notna(r.get("Combined Score")) else None,
            }
        )

    return {
        "meta": {
            "timezone": "US/Pacific",
            "generated_at": now.isoformat(),
            "horizon_hours": hours,
            "horizon_start": horizon_start.isoformat(),
            "horizon_end": horizon_end.isoformat(),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "score_quantile": score_quantile,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "weights": {"green": gw, "cheap": cw},
            "lmp_available": bool(lmp_available),
            "thresholds": {"green": green_thr, "cheap": cheap_thr},
            "window_types": ["green", "cheap", "both_intersection"],
        },
        "series": series,
        "windows_green": windows_green,
        "windows_cheap": windows_cheap,
        "windows_both": windows_both,
        "windows": windows_both,
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/api/next24")
def api_next24(
    hours: int = Query(24, ge=24, le=72),
    score_quantile: float = Query(0.80, ge=0.50, le=0.95),
    min_duration: str = Query("60min"),
    max_duration: Optional[str] = Query(None),
    green_weight: float = Query(0.5, ge=0.0, le=1.0),
    cheap_weight: float = Query(0.5, ge=0.0, le=1.0),
):
    # constrain to 24/48/72 only
    if hours not in (24, 48, 72):
        raise HTTPException(status_code=400, detail="hours must be 24, 48, or 72")

    try:
        return build_forecast_dataset(
            hours=hours,
            score_quantile=score_quantile,
            min_duration=min_duration,
            max_duration=max_duration,
            green_weight=green_weight,
            cheap_weight=cheap_weight,
        )
    except Exception as e:
        log.exception("api_next24 crashed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug_methods")
def debug_methods():
    return {
        "file": __file__,
        "python": os.getenv("PYTHON_VERSION", "unknown"),
        "has_load_forecast_day_ahead": callable(getattr(caiso, "get_load_forecast_day_ahead", None)),
        "has_load_forecast": callable(getattr(caiso, "get_load_forecast", None)),
        "has_solar_wind_dam": callable(getattr(caiso, "get_solar_and_wind_forecast_dam", None)),
        "has_renewables_dam": callable(getattr(caiso, "get_renewables_forecast_dam", None)),
        "lmp_methods_present": [name for name in ["get_lmp"] if callable(getattr(caiso, name, None))],
    }
