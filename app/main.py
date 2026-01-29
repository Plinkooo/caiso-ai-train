from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import gridstatus

# -----------------------------
# Logging / Config
# -----------------------------
log = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

TZ = pytz.timezone("US/Pacific")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 min default
CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "256"))
CACHE = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)

# How far into the future we treat data as "forecast" (not predicted)
FORECAST_CUTOFF_HOURS = float(os.getenv("FORECAST_CUTOFF_HOURS", "30"))

# Training window for predictions (load + renewables)
TRAINING_DAYS = int(os.getenv("TRAINING_DAYS", "30"))

# LMP configuration (node/market vary by gridstatus version; we try best-effort)
LMP_NODE = os.getenv("LMP_NODE", "TH_NP15_GEN-APND")
LMP_MARKET_DA = os.getenv("LMP_MARKET_DA", "DAY_AHEAD_HOURLY")
LMP_MARKET_RT = os.getenv("LMP_MARKET_RT", "REAL_TIME_5_MIN")

# Renewables categories used when computing renewables MW from fuel mix
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


def _cached(prefix: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    return CACHE.get(_cache_key(prefix, **kwargs))


def _set_cache(prefix: str, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    CACHE[_cache_key(prefix, **kwargs)] = df
    return df


def _safe_fetch(fn, prefix: str, **kwargs: Any) -> pd.DataFrame:
    cached_df = _cached(prefix, **kwargs)
    if cached_df is not None:
        return cached_df

    try:
        df = fn(**kwargs)
    except Exception as e:
        log.exception("Upstream fetch failed: %s %s", prefix, kwargs)
        raise HTTPException(status_code=502, detail=f"Upstream CAISO fetch failed ({prefix}): {e}") from e

    if df is None:
        df = pd.DataFrame()

    return _set_cache(prefix, df, **kwargs)


# -----------------------------
# Time helpers
# -----------------------------
def now_pacific() -> datetime:
    return datetime.now(tz=TZ)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(TZ).isoformat()


def parse_time_col(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Normalize a time column to 'time' and ensure tz-aware in US/Pacific.
    """
    if df is None or df.empty:
        return df, "time"

    time_candidates = [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "interval start",
        "Interval Start",
        "interval_start",
        "Interval Start Time",
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

    # Localize/convert
    if getattr(out["time"].dt, "tz", None) is None:
        out["time"] = out["time"].dt.tz_localize(TZ)
    else:
        out["time"] = out["time"].dt.tz_convert(TZ)

    out = out.dropna(subset=["time"])
    return out, "time"


def first_numeric_col(df: pd.DataFrame, exclude: set[str]) -> Optional[str]:
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


# -----------------------------
# Forecast fetchers (best-effort, version-tolerant)
# -----------------------------
def get_load_history(start: datetime, end: datetime) -> pd.DataFrame:
    return _safe_fetch(caiso.get_load, "load_history", date=start, end=end)


def get_load_forecast(end: datetime) -> pd.DataFrame:
    """
    Try common method names across gridstatus versions.
    If unavailable, return empty and we will rely on prediction.
    """
    candidates = ["get_load_forecast", "get_demand_forecast"]
    for name in candidates:
        if hasattr(caiso, name):
            fn = getattr(caiso, name)
            return _safe_fetch(fn, f"{name}", date=now_pacific(), end=end)
    return pd.DataFrame()


def get_fuel_mix_history(start: datetime, end: datetime) -> pd.DataFrame:
    return _safe_fetch(caiso.get_fuel_mix, "fuel_mix_history", date=start, end=end)


def get_fuel_mix_forecast(end: datetime) -> pd.DataFrame:
    """
    Fuel mix forecasts aren't always available in gridstatus versions.
    If unavailable, return empty and we will predict renewables from history.
    """
    candidates = ["get_fuel_mix_forecast", "get_renewables_forecast", "get_solar_and_wind_forecast"]
    for name in candidates:
        if hasattr(caiso, name):
            fn = getattr(caiso, name)
            return _safe_fetch(fn, f"{name}", date=now_pacific(), end=end)
    return pd.DataFrame()


def get_lmp_series(end: datetime) -> Tuple[pd.DataFrame, bool]:
    """
    Fetch LMP as far as CAISO/adapter supports (DA and/or RT).
    We will cut it off at FORECAST_CUTOFF_HOURS in the response.
    Returns (df, available).
    """
    if not hasattr(caiso, "get_lmp"):
        return pd.DataFrame(), False

    start = now_pacific() - timedelta(hours=2)  # slight backfill

    frames: List[pd.DataFrame] = []
    available = False

    # Day-ahead (forward-ish)
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
        # don't hard fail the whole endpoint because DA call failed
        log.info("DA LMP not available for this adapter/config")

    # Real-time (optional)
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

    # Normalize time column and pick price column
    df, _ = parse_time_col(df)

    # Common price column names vary; take first numeric after excluding time/location/market
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
    return out, True


# -----------------------------
# Renewables from fuel mix
# -----------------------------
def compute_renewables_mw_from_fuel_mix(fuel_mix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Output columns: time, renewables_mw
    """
    if fuel_mix_df is None or fuel_mix_df.empty:
        return pd.DataFrame(columns=["time", "renewables_mw"])

    df, _ = parse_time_col(fuel_mix_df)

    # Find numeric cols
    value_cols = [c for c in df.columns if c != "time" and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        return pd.DataFrame(columns=["time", "renewables_mw"])

    def is_renewable_col(col: str) -> bool:
        for k in RENEWABLE_KEYS:
            if k.lower() == col.lower():
                return True
        return False

    ren_cols = [c for c in value_cols if is_renewable_col(c)]
    if not ren_cols:
        # if adapter doesn't label fuel mix that way, we can't compute ren MW
        return pd.DataFrame(columns=["time", "renewables_mw"])

    out = pd.DataFrame({"time": df["time"]})
    out["renewables_mw"] = df[ren_cols].sum(axis=1)
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return out


# -----------------------------
# Simple prediction model (median by hour-of-week)
# -----------------------------
@dataclass
class HourOfWeekMedianModel:
    medians: Dict[int, float]
    fallback: float

    @staticmethod
    def fit(ts: pd.Series, values: pd.Series) -> "HourOfWeekMedianModel":
        # hour-of-week: Monday 0:00 = 0 ... Sunday 23:00 = 167
        how = (ts.dt.dayofweek * 24 + ts.dt.hour).astype(int)
        df = pd.DataFrame({"how": how, "v": values})
        df = df[pd.notna(df["v"])]

        if df.empty:
            return HourOfWeekMedianModel(medians={}, fallback=float("nan"))

        med = df.groupby("how")["v"].median().to_dict()
        fallback = float(df["v"].median())
        return HourOfWeekMedianModel(medians=med, fallback=fallback)

    def predict(self, ts: pd.Series) -> pd.Series:
        how = (ts.dt.dayofweek * 24 + ts.dt.hour).astype(int)
        preds = how.map(lambda k: self.medians.get(int(k), self.fallback))
        return preds.astype(float)


def build_training_series_load(now: datetime) -> pd.DataFrame:
    end = now
    start = end - timedelta(days=TRAINING_DAYS)
    df = get_load_history(start, end)
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "value"])

    df, _ = parse_time_col(df)
    # pick load column
    exclude = {"time"}
    load_col = None
    for c in df.columns:
        if c.lower() in {"load", "load (mw)", "mw", "demand", "demand (mw)"} and c not in exclude:
            load_col = c
            break
    if load_col is None:
        load_col = first_numeric_col(df, exclude=exclude)
    if load_col is None:
        return pd.DataFrame(columns=["time", "value"])

    out = df[["time", load_col]].rename(columns={load_col: "value"})
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    return out


def build_training_series_renewables(now: datetime) -> pd.DataFrame:
    end = now
    start = end - timedelta(days=TRAINING_DAYS)
    fm = get_fuel_mix_history(start, end)
    ren = compute_renewables_mw_from_fuel_mix(fm)
    if ren.empty:
        return pd.DataFrame(columns=["time", "value"])
    out = ren.rename(columns={"renewables_mw": "value"})
    return out


# -----------------------------
# Window scoring + building
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
    # e.g., "30min", "60min", "120min"
    if not s:
        return 0
    s = s.strip().lower()
    if s.endswith("min"):
        return int(float(s[:-3]))
    if s.endswith("h") or s.endswith("hr") or s.endswith("hrs"):
        # not used by your UI right now
        num = "".join(ch for ch in s if ch.isdigit() or ch == ".")
        return int(float(num) * 60)
    raise ValueError(f"Unsupported duration: {s}")


def build_windows(
    series_df: pd.DataFrame,
    score_col: str,
    score_quantile: float,
    min_minutes: int,
    max_minutes: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Finds contiguous blocks where score >= quantile threshold.
    Assumes series_df is hourly, sorted by time.
    """
    df = series_df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    s = df[score_col].astype(float)
    thresh = float(s.quantile(score_quantile)) if s.dropna().size else float("nan")

    good = s >= thresh
    windows = []

    # each row is 1 hour
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
    """
    Simple interval intersection between two window lists.
    """
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
    # de-dup / sort
    out = sorted(out, key=lambda w: w["start"])
    return out


# -----------------------------
# Core builder: /api/next24 schema (keeps your UI working)
# -----------------------------
def build_series(hours: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build an HOURLY time series for horizon 'hours':
      - Up to cutoff: uses actual forecast (if available), else predicted
      - Beyond cutoff: predicted
    LMP: filled only up to cutoff (or whatever is available), never predicted.
    """
    now = now_pacific()
    horizon_end = now + timedelta(hours=hours)
    cutoff_time = now + timedelta(hours=FORECAST_CUTOFF_HOURS)

    # Timeline: hourly points starting from the next hour boundary
    start_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    times = pd.date_range(start=start_hour, end=horizon_end, freq="1H", tz=TZ, inclusive="left")
    base = pd.DataFrame({"time": times})

    # --- Load: forecast best-effort ---
    lf = get_load_forecast(end=horizon_end)
    lf, _ = parse_time_col(lf)
    load_col = None
    if lf is not None and not lf.empty:
        # pick load column
        for c in lf.columns:
            if c.lower() in {"load", "load (mw)", "mw", "demand", "demand (mw)"} and c != "time":
                load_col = c
                break
        if load_col is None:
            load_col = first_numeric_col(lf, exclude={"time"})
    load_forecast = pd.DataFrame(columns=["time", "load_forecast_mw"])
    if lf is not None and not lf.empty and load_col:
        load_forecast = lf[["time", load_col]].rename(columns={load_col: "load_forecast_mw"})
        load_forecast = load_forecast.sort_values("time").drop_duplicates(subset=["time"], keep="last")

    # --- Renewables: from fuel mix forecast (if available), else history-prediction only ---
    fm_f = get_fuel_mix_forecast(end=horizon_end)
    ren_forecast = compute_renewables_mw_from_fuel_mix(fm_f)
    if not ren_forecast.empty:
        ren_forecast = ren_forecast.rename(columns={"renewables_mw": "renewables_forecast_mw"})

    # --- LMP: fetch and cut off later ---
    lmp_df, lmp_available = get_lmp_series(end=horizon_end)

    # --- Fit models from history for prediction ---
    train_load = build_training_series_load(now)
    train_ren = build_training_series_renewables(now)

    load_model = HourOfWeekMedianModel.fit(train_load["time"], train_load["value"]) if not train_load.empty else HourOfWeekMedianModel({}, float("nan"))
    ren_model = HourOfWeekMedianModel.fit(train_ren["time"], train_ren["value"]) if not train_ren.empty else HourOfWeekMedianModel({}, float("nan"))

    # --- Merge forecasts onto timeline (nearest hour) ---
    out = base.copy()

    def merge_hourly(out_df: pd.DataFrame, src: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if src is None or src.empty:
            out_df[value_col] = float("nan")
            return out_df
        src2 = src.copy()
        src2["time_hr"] = src2["time"].dt.floor("H")
        src2 = src2.groupby("time_hr")[value_col].mean().reset_index().rename(columns={"time_hr": "time"})
        out_df = out_df.merge(src2, on="time", how="left")
        return out_df

    out = merge_hourly(out, load_forecast, "load_forecast_mw")
    out = merge_hourly(out, ren_forecast, "renewables_forecast_mw")

    # --- Predict beyond cutoff (and also fill missing within cutoff) ---
    out["is_predicted"] = out["time"] > cutoff_time

    # predicted values
    load_pred = load_model.predict(out["time"].dt.tz_convert(TZ).dt.tz_localize(None))
    ren_pred = ren_model.predict(out["time"].dt.tz_convert(TZ).dt.tz_localize(None))

    # apply rules:
    # - if time > cutoff -> predicted
    # - else keep forecast if present, else predicted (but still mark is_predicted False)
    out.loc[out["time"] > cutoff_time, "load_forecast_mw"] = load_pred[out["time"] > cutoff_time].values
    out.loc[out["time"] > cutoff_time, "renewables_forecast_mw"] = ren_pred[out["time"] > cutoff_time].values

    out.loc[out["time"] <= cutoff_time, "load_forecast_mw"] = out.loc[out["time"] <= cutoff_time, "load_forecast_mw"].fillna(load_pred[out["time"] <= cutoff_time].values)
    out.loc[out["time"] <= cutoff_time, "renewables_forecast_mw"] = out.loc[out["time"] <= cutoff_time, "renewables_forecast_mw"].fillna(ren_pred[out["time"] <= cutoff_time].values)

    # net load
    out["net_load_forecast_mw"] = out["load_forecast_mw"] - out["renewables_forecast_mw"]

    # green score: higher renewables share
    # (we’ll compute share as renewables/load, clipped to [0,1], and normalize)
    share = out["renewables_forecast_mw"] / out["load_forecast_mw"]
    share = share.clip(lower=0, upper=1)
    out["green_score"] = normalize_01(share)

    # cheap score: lower LMP is better; we will only fill up to cutoff, never predicted
    out["lmp_usd_per_mwh"] = float("nan")
    if lmp_available and lmp_df is not None and not lmp_df.empty:
        lmp_df = lmp_df.copy()
        lmp_df["time"] = lmp_df["time"].dt.floor("H")
        lmp_hourly = lmp_df.groupby("time")["lmp_usd_per_mwh"].mean().reset_index()
        out = out.merge(lmp_hourly, on="time", how="left", suffixes=("", "_y"))
        if "lmp_usd_per_mwh_y" in out.columns:
            out["lmp_usd_per_mwh"] = out["lmp_usd_per_mwh_y"]
            out = out.drop(columns=["lmp_usd_per_mwh_y"])

    # cut LMP at cutoff
    out.loc[out["time"] > cutoff_time, "lmp_usd_per_mwh"] = float("nan")

    # cheap score: invert normalized price (only where LMP exists)
    if out["lmp_usd_per_mwh"].dropna().empty:
        out["cheap_score"] = float("nan")
    else:
        price_norm = normalize_01(out["lmp_usd_per_mwh"])
        out["cheap_score"] = 1.0 - price_norm

    meta = {
        "horizon_start": to_iso(start_hour),
        "horizon_end": to_iso(horizon_end),
        "horizon_hours": hours,
        "forecast_cutoff_hours": FORECAST_CUTOFF_HOURS,
        "forecast_cutoff_time": to_iso(cutoff_time),
        "lmp_available": bool(lmp_available and not out["lmp_usd_per_mwh"].dropna().empty),
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
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "cache_size": len(CACHE),
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
    """
    Keeps your UI schema:
      - series[].t_start
      - load_forecast_mw / renewables_forecast_mw / net_load_forecast_mw
      - green_score / cheap_score / combined_score
      - lmp_usd_per_mwh (cut off at cutoff, never predicted)
      - windows + windows_green + windows_cheap + windows_both
    """
    series_df, meta = build_series(hours=hours)

    min_minutes = duration_to_minutes(min_duration)
    max_minutes = duration_to_minutes(max_duration) if max_duration else None

    # combined score (only uses cheap where available; if cheap missing, combine will follow green_weight only)
    gw = float(green_weight)
    cw = float(cheap_weight)
    denom = gw + cw
    if denom <= 1e-9:
        gw, cw, denom = 1.0, 0.0, 1.0

    gs = series_df["green_score"].astype(float)
    cs = series_df["cheap_score"].astype(float)

    # If cheap is missing, treat it as NaN and weight will effectively reduce; fill NaN with 0.5 neutral for combination
    cs_for_combo = cs.fillna(0.5)

    series_df["combined_score"] = (gw * gs + cw * cs_for_combo) / denom

    # Windows from scores
    windows_green = build_windows(series_df, "green_score", score_quantile, min_minutes, max_minutes)
    windows_cheap = build_windows(series_df, "cheap_score", score_quantile, min_minutes, max_minutes) if meta["lmp_available"] else []
    windows_both = intersect_windows(windows_green, windows_cheap) if windows_green and windows_cheap else []

    # Default windows = both if available else combined score blocks
    windows_default = windows_both
    if not windows_default:
        windows_default = build_windows(series_df, "combined_score", score_quantile, min_minutes, max_minutes)

    # Render series records matching your current UI field names
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

    # Meta used by your UI
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
