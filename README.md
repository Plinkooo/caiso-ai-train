# CAISO AI Training Windows (Green)

A small FastAPI + static frontend app that uses `gridstatus` to recommend
"good times to train AI" on the CAISO grid using day-ahead forecasts.

**Green strategy (default):**
- Net load forecast = Load forecast - (Solar+Wind forecast)
- "Good to train" windows are the top quantile of **Green Score**
  (i.e., the lowest net load periods).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Deploy on Render

1. Push this repo to GitHub
2. In Render: New -> Blueprint -> select repo (Render will read `render.yaml`)
3. Deploy

## Caching

Environment variables:
- CACHE_TTL_SECONDS (default 600)
- CACHE_MAXSIZE (default 256)

Note: This is an in-memory cache (per instance). For multi-instance scaling,
use an external cache (e.g., Redis).
