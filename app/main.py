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
    # Try common names first
    direct = pick_first_existing_column(df, ["LMP", "LMP ($/MWh)", "LMP $/MWh", "LMP_MWH", "lmp"])
    if direct:
        return direct
    # Fallback: any column containing "lmp"
    for c in df.columns:
        if "lmp" in c.lower():
            return c
    return None


# -----------------------------
# Data fetching via gridstatus (CAISO)
# -----------------------------
def fetch_day_ahead_load_forecast(start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    """Hourly day-ahead load forecast (best-effort across gridstatus versions)."""

    def _fetch(date, end):
        # Prefer explicit day-ahead method if present, otherwise generic
        fn = getattr(caiso, "get_load_forecast_day_ahead", None)
        if callable(fn):
            df = fn(date, end=end)
        else:
            fn2 = getattr(caiso, "get_load_forecast", None)
            if not callable(fn2):
                raise RuntimeError("gridstatus.CAISO() has no load forecast method")
            df = fn2(date, end=end)

        # Some versions include TAC area; keep ISO-TAC if available
        if "TAC Area Name" in df.columns:
            df = df[df["TAC Area Name"] == "CA ISO-TAC"].copy()

        return df

    df = cached_get("load_forecast_dam", _fetch, date=start_day, end=end_day).copy()

    if df.empty:
        return pd.DataFrame(columns=["Interval Start", "Interval End", "Load Forecast"])

    df["Interval Start"] = pd.to_datetime(df["Interval Start"])
    df["Interval End"] = pd.to_datetime(df["Interval End"])

    # Column name in gridstatus is usually "Load Forecast"
    if "Load Forecast" not in df.columns:
        # best-effort fallback if column naming differs
        cand = pick_first_existing_column(df, ["Load", "Forecast", "Load (MW)", "MW"])
        if cand:
            df["Load Forecast"] = pd.to_numeric(df[cand], errors="coerce")
        else:
            df["Load Forecast"] = pd.NA

    return df[["Interval Start", "Interval End", "Load Forecast"]].sort_values("Interval Start")


def fetch_day_ahead_solar_wind_forecast(start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    """Hourly day-ahead solar+wind forecast (Location == 'CAISO' totals).
    Compatible across gridstatus versions. Falls back to zeros if DAM renewables isn't available.
    """

    def _fetch(date, end):
        fn = getattr(caiso, "get_solar_and_wind_forecast_dam", None)
        if callable(fn):
            return fn(date, end=end)

        fn = getattr(caiso, "get_renewables_forecast_dam", None)
        if callable(fn):
            return fn(date, end=end)

        # No supported method in this installed gridstatus
        return pd.DataFrame()

    df = cached_get("solar_wind_forecast_dam", _fetch, date=start_day, end=end_day).copy()

    # If no renewables method exists, return zeros so endpoint still works
    if df.empty:
        hours = pd.date_range(start_day, end_day, freq="H", tz=TZ)
        out = pd.DataFrame(
            {
                "Interval Start": hours[:-1],
                "Interval End": hours[1:],
                "Solar MW": 0.0,
                "Wind MW": 0.0,
            }
        )
        out["Renewables Forecast"] = 0.0
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
    market: str = "DAM",
    location: Optional[str] = None,
) -> pd.DataFrame:
    """Best-effort day-ahead LMP fetch across gridstatus versions.
    If not available, returns empty df with Interval Start/End + LMP column.
    """

    def _fetch(date, end, market, location):
        # Try a few likely method names across gridstatus versions
        method_names = [
            "get_lmp_day_ahead",
            "get_lmp",
            "get_locational_marginal_prices",
            "get_lmp_data",
        ]

        last_err = None
        for name in method_names:
            fn = getattr(caiso, name, None)
            if not callable(fn):
                continue
            try:
                # Try common calling patterns
                kwargs = {"end": end}
                # Some methods accept market=...
                kwargs["market"] = market

                # Some accept locations=... or location=...
                if location:
                    # prefer locations list if supported
                    try:
                        return fn(date, locations=[location], **kwargs)  # type: ignore[arg-type]
                    except TypeError:
                        try:
                            return fn(date, location=location, **kwargs)  # type: ignore[arg-type]
                        except TypeError:
                            return fn(date, **kwargs)  # type: ignore[arg-type]
                else:
                    return fn(date, **kwargs)  # type: ignore[arg-type]
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue

        if last_err:
            log.warning("LMP fetch unavailable via gridstatus methods: %s", last_err)
        return pd.DataFrame()

    df = cached_get("lmp_dam", _fetch, date=start_day, end=end_day, market=market, location=location).copy()

    if df.empty:
        return pd.DataFrame(columns=["Interval Start", "Interval End", "LMP"])

    # Normalize timestamps
    if "Interval Start" in df.columns:
        df["Interval Start"] = pd.to_datetime(df["Interval Start"])
    if "Interval End" in df.columns:
        df["Interval End"] = pd.to_datetime(df["Interval End"])

    # Ensure LMP column exists
    lmp_col = find_lmp_column(df)
    if lmp_col and lmp_col != "LMP":
        df["LMP"] = pd.to_numeric(df[lmp_col], errors="coerce")
    elif "LMP" in df.columns:
        df["LMP"] = pd.to_numeric(df["LMP"], errors="coerce")
    else:
        df["LMP"] = pd.NA

    keep = [c for c in ["Interval Start", "Interval End", "LMP"] if c in df.columns]
    out = df[keep].copy()

    # If the gridstatus method returned multiple nodes, try to filter to a single location if possible
    if location and "Location" in df.columns:
        tmp = df[df["Location"] == location].copy()
        if not tmp.empty:
            tmp_keep = [c for c in ["Interval Start", "Interval End", "LMP"] if c in tmp.columns]
            out = tmp[tmp_keep].copy()

    return out.sort_values("Interval Start")


# -----------------------------
# Windows
# -----------------------------
def contiguous_windows(
    df: pd.DataFrame,
    mask: pd.Series,
    min_duration: str = "60min",
    score_col: str = "Green Score",
    netload_col: str = "Net Load Forecast",
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
                        "avg_net_load": float(pd.to_numeric(chunk.get(netload_col, pd.Series([])), errors="coerce").mean())
                        if netload_col in chunk.columns
                        else None,
                        "avg_score": float(pd.to_numeric(chunk.get(score_col, pd.Series([])), errors="coerce").mean())
                        if score_col in chunk.columns
                        else None,
                    }
                )
            start_idx = None

    return windows


# -----------------------------
# Dataset build: green + cheap + intersection
# -----------------------------
def build_next24_dataset(
    score_quantile: float = 0.80,
    min_duration: str = "60min",
    green_weight: float = 0.5,
    cheap_weight: float = 0.5,
    lmp_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Next-24h hourly series with recommended windows:
    - Green windows (low net load proxy)
    - Cheap windows (low LMP proxy, best-effort)
    - Both/Intersection windows (green AND cheap)
    """

    now = pd.Timestamp.now(tz=TZ)
    horizon_start = now.floor("H")
    horizon_end = horizon_start + pd.Timedelta(hours=24)

    # Fetch enough day-ahead data covering today + tomorrow
    start_day = horizon_start.normalize()
    end_day = horizon_end.normalize() + pd.Timedelta(days=1)

    load_fc = fetch_day_ahead_load_forecast(start_day, end_day)
    ren_fc = fetch_day_ahead_solar_wind_forecast(start_day, end_day)
    lmp_df = fetch_day_ahead_lmp(start_day, end_day, market="DAM", location=lmp_location)

    # Merge
    df = load_fc.merge(ren_fc, on=["Interval Start", "Interval End"], how="left")
    if not lmp_df.empty and all(c in lmp_df.columns for c in ["Interval Start", "Interval End"]):
        df = df.merge(lmp_df, on=["Interval Start", "Interval End"], how="left")
    else:
        df["LMP"] = pd.NA

    # Net load proxy
    df["Net Load Forecast"] = pd.to_numeric(df["Load Forecast"], errors="coerce") - pd.to_numeric(
        df["Renewables Forecast"], errors="coerce"
    )

    # Slice to next 24h
    df = df[(df["Interval Start"] >= horizon_start) & (df["Interval Start"] < horizon_end)].copy()
    df = df.sort_values("Interval Start").reset_index(drop=True)

    # Scores
    # Green: lower net load is better
    df["Green Score"] = 1.0 - minmax_normalize(df["Net Load Forecast"])

    # Cheap: lower LMP is better (if available)
    lmp_available = df["LMP"].notna().any()
    if lmp_available:
        df["Cheap Score"] = 1.0 - minmax_normalize(df["LMP"])
    else:
        # If no LMP, keep it as NA; combined will fall back to Green only
        df["Cheap Score"] = pd.NA

    # Combined: prioritize both cheap+green
    # If Cheap Score missing, fallback to Green-only
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

    combined = []
    for _, r in df.iterrows():
        g = r.get("Green Score")
        c = r.get("Cheap Score")
        g_ok = pd.notna(g)
        c_ok = pd.notna(c)

        if g_ok and c_ok:
            combined.append((gw * float(g) + cw * float(c)) / wsum)
        elif g_ok:
            combined.append(float(g))
        else:
            combined.append(None)

    df["Combined Score"] = combined

    # Thresholds and windows
    green_thr = safe_quantile(df["Green Score"], score_quantile)
    cheap_thr = safe_quantile(df["Cheap Score"], score_quantile) if lmp_available else None

    green_mask = (df["Green Score"] >= green_thr) if green_thr is not None else pd.Series([False] * len(df))
    cheap_mask = (df["Cheap Score"] >= cheap_thr) if cheap_thr is not None else pd.Series([False] * len(df))
    both_mask = green_mask & cheap_mask if (green_thr is not None and cheap_thr is not None) else pd.Series([False] * len(df))

    windows_green = contiguous_windows(df, green_mask, min_duration=min_duration, score_col="Green Score")
    windows_cheap = contiguous_windows(df, cheap_mask, min_duration=min_duration, score_col="Cheap Score")
    windows_both = contiguous_windows(df, both_mask, min_duration=min_duration, score_col="Combined Score")

    # Serialize series for plotting
    series: List[Dict[str, Any]] = []
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
                "lmp_usd_per_mwh": float(r["LMP"]) if pd.notna(r.get("LMP")) else None,
                "green_score": float(r["Green Score"]) if pd.notna(r["Green Score"]) else None,
                "cheap_score": float(r["Cheap Score"]) if pd.notna(r.get("Cheap Score")) else None,
                "combined_score": float(r["Combined Score"]) if r.get("Combined Score") is not None else None,
            }
        )

    return {
        "meta": {
            "timezone": "US/Pacific",
            "generated_at": now.isoformat(),
            "horizon_start": horizon_start.isoformat(),
            "horizon_end": horizon_end.isoformat(),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
            "strategy": "green+cheap",
            "score_quantile": score_quantile,
            "min_duration": min_duration,
            "weights": {"green": gw, "cheap": cw},
            "lmp_available": bool(lmp_available),
            "lmp_location": lmp_location,
            "thresholds": {"green": green_thr, "cheap": cheap_thr},
            "window_types": ["green", "cheap", "both_intersection"],
        },
        "series": series,
        # New fields (what you asked for)
        "windows_green": windows_green,
        "windows_cheap": windows_cheap,
        "windows_both": windows_both,
        # Backward-compatible alias: "windows" = intersection (agree) windows
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
    score_quantile: float = Query(0.80, ge=0.50, le=0.95),
    min_duration: str = Query("60min"),
    green_weight: float = Query(0.5, ge=0.0, le=1.0),
    cheap_weight: float = Query(0.5, ge=0.0, le=1.0),
    lmp_location: Optional[str] = Query(None),
):
    """Returns next-24h hourly series + windows:
    - windows_green
    - windows_cheap
    - windows_both (intersection)
    Also includes legacy "windows" == windows_both.
    """
    try:
        return build_next24_dataset(
            score_quantile=score_quantile,
            min_duration=min_duration,
            green_weight=green_weight,
            cheap_weight=cheap_weight,
            lmp_location=lmp_location,
        )
    except Exception as e:
        log.exception("api_next24 crashed")
        # Return JSON error so frontend doesn't choke on non-JSON
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug_methods")
def debug_methods():
    """Helps diagnose which gridstatus methods exist in production."""
    return {
        "file": __file__,
        "python": os.getenv("PYTHON_VERSION", "unknown"),
        "has_load_forecast_day_ahead": callable(getattr(caiso, "get_load_forecast_day_ahead", None)),
        "has_load_forecast": callable(getattr(caiso, "get_load_forecast", None)),
        "has_solar_wind_dam": callable(getattr(caiso, "get_solar_and_wind_forecast_dam", None)),
        "has_renewables_dam": callable(getattr(caiso, "get_renewables_forecast_dam", None)),
        "lmp_methods_present": [
            name
            for name in ["get_lmp_day_ahead", "get_lmp", "get_locational_marginal_prices", "get_lmp_data"]
            if callable(getattr(caiso, name, None))
        ],
    }
