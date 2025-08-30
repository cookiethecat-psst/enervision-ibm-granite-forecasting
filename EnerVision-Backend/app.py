# app.py — EnerVision Backend (SQLite + IsolationForest + Daily-Profile Forecast + Advisor + Chat)

import logging, os, sqlite3, math, json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from uuid import UUID

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

# ----------------------------- Config -----------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "enervision.db")
FORECAST_HORIZON_MIN = 24 * 60
PROFILE_LOOKBACK_DAYS = 7
MIN_POINTS_FOR_PROFILE = 24 * 60 * 2  # at least 2 days of minute data

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enervision")

app = FastAPI(title="EnerVision Backend")

# CORS defaults for local dev; override with CORS_ORIGINS env if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8501",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Middleware -----------------------------

@app.middleware("http")
async def add_logger(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"INTERNAL_ERROR: {str(e)}"})

# ----------------------------- Models -----------------------------

class AnomalyPoint(BaseModel):
    ts: datetime
    value: float
    score: float
    reason: Optional[str] = None
    ts_ist: Optional[str] = None

class AnomalyResponse(BaseModel):
    building_id: UUID
    lookback_hours: int
    window_start: datetime
    window_end: datetime
    count: int
    anomalies: List[AnomalyPoint]

class ForecastPoint(BaseModel):
    ts: datetime
    yhat: float
    yhat_lower: float
    yhat_upper: float

class ForecastResponse(BaseModel):
    building_id: UUID
    horizon_minutes: int
    start: datetime
    end: datetime
    points: List[ForecastPoint]
    peak_time: Optional[datetime] = None
    peak_value: Optional[float] = None
    method: str

class AdvisorResponse(BaseModel):
    building_id: UUID
    as_of: datetime
    peak_hour_utc: Optional[int] = None
    peak_forecast_kW: Optional[float] = None
    recommendations: List[str]

# ----------------------- SQLite repo helpers ----------------------

def _to_iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _parse_iso_z(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def _to_iso_ist(dt: datetime) -> str:
    return dt.astimezone(IST).replace(microsecond=0).isoformat()

def _conn():
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"Database not found at {DB_PATH}. Run seed_db.py first.")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

async def get_timeseries(building_id: str, start: datetime, end: datetime) -> List[dict]:
    start_s = _to_iso_z(start)
    end_s   = _to_iso_z(end)
    con = _conn()
    try:
        cur = con.execute(
            """
            SELECT ts, value
            FROM meter_readings
            WHERE building_id = ?
              AND ts >= ? AND ts <= ?
            ORDER BY ts ASC
            """,
            (str(building_id), start_s, end_s),
        )
        rows = cur.fetchall()
        return [{"ts": _parse_iso_z(r["ts"]), "value": float(r["value"])} for r in rows]
    finally:
        con.close()

async def get_history_days(building_id: str, days: int) -> List[dict]:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    start = now - timedelta(days=days)
    return await get_timeseries(str(building_id), start, now)

# ------------------------ Anomaly detection -----------------------

async def detect_anomalies_for_window(building_id: str, start: datetime, end: datetime) -> List[dict]:
    rows = await get_timeseries(building_id, start, end)
    if not rows or len(rows) < 30:
        return []

    rows = sorted(rows, key=lambda r: r["ts"])
    ts = [r["ts"] for r in rows]
    vals = np.array([float(r["value"]) for r in rows], dtype=float).reshape(-1, 1)

    # Isolation Forest
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
    iso.fit(vals)
    preds = iso.predict(vals)          # -1 = anomaly
    scores = -iso.score_samples(vals)  # higher = more anomalous

    idxs = [i for i, p in enumerate(preds) if p == -1]
    if not idxs:
        return []

    anom_scores = [float(scores[i]) for i in idxs]
    cutoff = float(np.percentile(anom_scores, 50))  # keep stronger half
    strong = [i for i in idxs if float(scores[i]) >= cutoff]

    merged: List[dict] = []
    strong.sort()
    for i in strong:
        cand = {
            "ts": ts[i],
            "value": float(vals[i][0]),
            "score": float(scores[i]),
            "reason": "IsolationForest",
        }
        # merge if within 3 minutes
        if merged and (cand["ts"] - merged[-1]["ts"]).total_seconds() <= 180:
            if cand["score"] > merged[-1]["score"]:
                merged[-1] = cand
        else:
            merged.append(cand)

    return merged

# ----------------------- Forecast (daily profile) -----------------

def _minute_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute

def _build_daily_profile(rows: List[dict]) -> Tuple[np.ndarray, float]:
    if not rows:
        return np.full(1440, np.nan), float("nan")

    buckets: List[List[float]] = [[] for _ in range(1440)]
    for r in rows:
        buckets[_minute_of_day(r["ts"])].append(float(r["value"]))

    profile = np.zeros(1440, dtype=float)
    for i in range(1440):
        profile[i] = float(np.mean(buckets[i])) if buckets[i] else np.nan

    # forward/backward fill
    last = np.nan
    for i in range(1440):
        if not math.isnan(profile[i]): 
            last = profile[i]
        elif not math.isnan(last): 
            profile[i] = last
    last = np.nan
    for i in range(1439, -1, -1):
        if not math.isnan(profile[i]): 
            last = profile[i]
        elif not math.isnan(last): 
            profile[i] = last
    if np.isnan(profile).any():
        global_mean = float(np.nanmean(profile))
        profile = np.where(np.isnan(profile), global_mean, profile)

    residuals = [float(r["value"]) - profile[_minute_of_day(r["ts"])] for r in rows]
    resid_std = float(np.std(residuals)) if residuals else 0.0
    resid_std = max(resid_std, 1.0)

    # light smoothing
    k = 3
    smoothed = np.copy(profile)
    for i in range(1440):
        lo = max(0, i - k); hi = min(1440, i + k + 1)
        smoothed[i] = float(np.mean(profile[lo:hi]))
    return smoothed, resid_std

async def make_forecast(building_id: str, horizon_min: int = FORECAST_HORIZON_MIN) -> Dict:
    hist = await get_history_days(building_id, PROFILE_LOOKBACK_DAYS)
    now = datetime.now(UTC).replace(second=0, microsecond=0)

    # fallback if insufficient history
    if not hist or len(hist) < MIN_POINTS_FOR_PROFILE:
        lookback = await get_history_days(building_id, 1)
        if not lookback:
            return {
                "points": [], "start": now, "end": now + timedelta(minutes=horizon_min),
                "method": "fallback_no_data", "peak_time": None, "peak_value": None,
            }
        arr = np.array([r["value"] for r in lookback], dtype=float)
        base, std = float(np.mean(arr)), float(np.std(arr))
        pts = [{
            "ts": now + timedelta(minutes=m),
            "yhat": base,
            "yhat_lower": max(0.0, base - 1.96 * std),
            "yhat_upper": base + 1.96 * std
        } for m in range(1, horizon_min + 1)]
        peak_idx = int(np.argmax([p["yhat"] for p in pts])) if pts else None
        return {
            "points": pts, "start": now, "end": now + timedelta(minutes=horizon_min),
            "method": "flat_mean_24h",
            "peak_time": pts[peak_idx]["ts"] if peak_idx is not None else None,
            "peak_value": pts[peak_idx]["yhat"] if peak_idx is not None else None,
        }

    profile, resid_std = _build_daily_profile(hist)
    points = []
    for m in range(1, horizon_min + 1):
        ts = now + timedelta(minutes=m)
        y = float(profile[_minute_of_day(ts)])
        points.append({
            "ts": ts,
            "yhat": y,
            "yhat_lower": max(0.0, y - 1.96 * resid_std),
            "yhat_upper": y + 1.96 * resid_std,
        })
    peak_idx = int(np.argmax([p["yhat"] for p in points])) if points else None
    return {
        "points": points, "start": now, "end": now + timedelta(minutes=horizon_min),
        "method": f"daily_profile_{PROFILE_LOOKBACK_DAYS}d",
        "peak_time": points[peak_idx]["ts"] if peak_idx is not None else None,
        "peak_value": points[peak_idx]["yhat"] if peak_idx is not None else None,
    }

# --------------------------- Advisor ------------------------------

def _adv_shift_windows(peak_hour: int) -> List[str]:
    tips: List[str] = []
    off_peak1 = (peak_hour + 10) % 24
    tips.append(f"Schedule heavy appliances (laundry, dishwasher) around {off_peak1:02d}:00–{(off_peak1+1)%24:02d}:00 UTC to avoid peak.")
    tips.append(f"Pre-cool spaces before {peak_hour:02d}:00 UTC and raise AC setpoint by 1–2°C during peak.")
    tips.append("Stagger elevator/industrial motor starts; avoid concurrent high inrush during peak hour.")
    tips.append("If you have battery/solar: charge earlier and discharge through the forecasted peak window.")
    return tips

async def make_advice(building_id: str) -> AdvisorResponse:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    fc = await make_forecast(building_id, FORECAST_HORIZON_MIN)
    pts = fc["points"]
    if not pts:
        return AdvisorResponse(
            building_id=UUID(building_id),
            as_of=now,
            recommendations=["Not enough data to advise. Ensure meter is streaming and try later."],
        )
    peak_time, peak_val = fc["peak_time"], fc["peak_value"]
    peak_hour = peak_time.hour if peak_time else None
    recs = _adv_shift_windows(peak_hour) if peak_hour is not None else ["No clear peak; keep loads flat through the day."]
    if peak_hour is not None:
        peak = next(p for p in pts if p["ts"] == peak_time)
        band = peak["yhat_upper"] - peak["yhat_lower"]
        if band / max(peak["yhat"], 1.0) > 0.6:
            recs.append("Peak timing uncertainty is high; spread flexible loads over a wider 2–3 hour window.")
        else:
            recs.append("Peak timing is fairly confident; concentrate shifting just 1 hour around the peak.")
    return AdvisorResponse(
        building_id=UUID(building_id),
        as_of=now,
        peak_hour_utc=peak_hour,
        peak_forecast_kW=float(round(peak_val, 2)) if peak_val is not None else None,
        recommendations=recs,
    )

# ----------------------------- Routes -----------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/health")
async def health():
    return {"status": "ok", "time": _to_iso_z(datetime.now(UTC))}

@app.get("/anomalies", response_model=AnomalyResponse)
async def anomalies(building_id: UUID, lookback_hours: int = Query(24, ge=1, le=24*14)):
    now = datetime.now(UTC)
    start = now - timedelta(hours=lookback_hours)
    try:
        anns = await detect_anomalies_for_window(str(building_id), start, now)
        anns = [{**a, "ts_ist": _to_iso_ist(a["ts"])} for a in anns]
        anns = [AnomalyPoint(**a) for a in anns]
        return AnomalyResponse(
            building_id=building_id,
            lookback_hours=lookback_hours,
            window_start=start,
            window_end=now,
            count=len(anns),
            anomalies=anns,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ANOMALY_ERROR: {e}")

@app.get("/anomalies/summary")
async def anomalies_summary(
    building_id: UUID,
    lookback_hours: int = Query(24, ge=1, le=24*14),
    top: int = 5
):
    now = datetime.now(UTC)
    start = now - timedelta(hours=lookback_hours)
    anns = await detect_anomalies_for_window(str(building_id), start, now)

    by_hour_utc: Dict[str, int] = defaultdict(int)
    by_hour_ist: Dict[str, int] = defaultdict(int)

    top_list = sorted(anns, key=lambda a: a["score"], reverse=True)[:max(1, top)]
    top_list = [
        {
            "ts": _to_iso_z(a["ts"]),
            "ts_ist": _to_iso_ist(a["ts"]),
            "value": a["value"],
            "score": a["score"],
            "reason": a.get("reason", "IsolationForest"),
        } for a in top_list
    ]

    for a in anns:
        h_utc = a["ts"].replace(minute=0, second=0, microsecond=0)
        h_ist = a["ts"].astimezone(IST).replace(minute=0, second=0, microsecond=0)
        by_hour_utc[_to_iso_z(h_utc)] += 1
        by_hour_ist[_to_iso_ist(h_ist)] += 1

    if anns:
        scores = np.array([x["score"] for x in anns], dtype=float)
        t_med = float(np.percentile(scores, 50))
        t_hi  = float(np.percentile(scores, 80))
        sev = {
            "low": int((scores < t_med).sum()),
            "medium": int(((scores >= t_med) & (scores < t_hi)).sum()),
            "high": int((scores >= t_hi).sum()),
        }
    else:
        sev = {"low": 0, "medium": 0, "high": 0}

    return {
        "building_id": str(building_id),
        "window_start": _to_iso_z(start),
        "window_end": _to_iso_z(now),
        "count": len(anns),
        "severity": sev,
        "by_hour_utc": dict(by_hour_utc),
        "by_hour_ist": dict(by_hour_ist),
        "top": top_list,
    }

@app.get("/forecast", response_model=ForecastResponse)
async def forecast(
    building_id: UUID,
    minutes: int = Query(FORECAST_HORIZON_MIN, ge=30, le=7*24*60),
    hours: Optional[int] = None
):
    try:
        if hours is not None:
            minutes = max(30, min(hours * 60, 7 * 24 * 60))
        fc = await make_forecast(str(building_id), minutes)
        points = [ForecastPoint(**p) for p in fc["points"]]
        return ForecastResponse(
            building_id=building_id,
            horizon_minutes=minutes,
            start=fc["start"],
            end=fc["end"],
            points=points,
            peak_time=fc["peak_time"],
            peak_value=fc["peak_value"],
            method=fc["method"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FORECAST_ERROR: {e}")

@app.get("/advisor", response_model=AdvisorResponse)
async def advisor(building_id: UUID):
    try:
        return await make_advice(str(building_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ADVISOR_ERROR: {e}")

# ----------------------- Chat (grounded + fallback) ----------------

def _summarize_for_chat(forecast: Dict, anomalies: List[dict], advice: AdvisorResponse) -> str:
    peak_time = forecast.get("peak_time")
    peak_val = forecast.get("peak_value")
    peak_str_utc = _to_iso_z(peak_time) if peak_time else "N/A"
    peak_str_ist = _to_iso_ist(peak_time) if peak_time else "N/A"
    anom_count = len(anomalies)
    lines = [
        f"Peak forecast: {round(peak_val,2) if peak_val else 'N/A'} kW at {peak_str_utc} (UTC) / {peak_str_ist} (IST).",
        f"Anomalies last 24h: {anom_count}.",
        "Top advice:",
    ]
    for r in advice.recommendations[:3]:
        lines.append(f"- {r}")
    return "\n".join(lines)

def _general_fallback(q: str) -> str:
    m = (q or "").lower()
    if "solar" in m:
        return "Solar energy is electricity or heat generated from sunlight. PV panels make DC power; an inverter supplies AC for home use."
    if "ev" in m or "vehicle" in m:
        return "An electric vehicle runs on a battery-powered motor. Charge at home or via public chargers; running cost per km is usually lower than petrol/diesel."
    if "ac" in m or "air conditioner" in m:
        return "An air conditioner removes heat from indoors. Higher ISEER/EER means better efficiency. Keep filters clean and set 24–26°C to cut bills."
    return "I can help with solar, AC efficiency, EV charging, and ways to reduce your bill."

@app.post("/api/chat")
async def chat(req: Dict):
    # Shape your UI expects: { ok: true, reply: "...", used: {...} }
    message = (req or {}).get("message", "")
    building_id = (req or {}).get("building_id")

    used: Dict[str, object] = {}

    # If no building_id, treat as general question
    if not building_id:
        # Try Gemini if available; else fallback
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(
                    "Answer in plain text, concise and practical.\n\nUser: " + str(message)
                )
                text = getattr(resp, "text", "") or _general_fallback(message)
                return {"ok": True, "reply": text, "used": {"llm": "gemini"}}
            except Exception as e:
                used["llm_error"] = str(e)
        return {"ok": True, "reply": _general_fallback(message), "used": used}

    # With building_id: pull context and optionally call Gemini
    fc = await make_forecast(str(building_id), FORECAST_HORIZON_MIN)
    used["forecast"] = {"points": len(fc.get("points", [])), "method": fc.get("method")}
    now = datetime.now(UTC)
    anns = await detect_anomalies_for_window(str(building_id), now - timedelta(hours=24), now)
    used["anomalies"] = {"count": len(anns)}
    adv = await make_advice(str(building_id))
    used["advisor"] = {"tips": len(adv.recommendations)}

    answer = _summarize_for_chat(fc, anns, adv)

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            context_json = json.dumps({
                "question": message,
                "forecast": {
                    "peak_time_utc": _to_iso_z(fc["peak_time"]) if fc.get("peak_time") else None,
                    "peak_time_ist": _to_iso_ist(fc["peak_time"]) if fc.get("peak_time") else None,
                    "peak_value": fc.get("peak_value"),
                    "method": fc.get("method"),
                },
                "anomalies": [{"ts": _to_iso_z(a["ts"]), "ts_ist": _to_iso_ist(a["ts"]),
                               "value": a["value"], "score": a["score"]} for a in anns[:50]],
                "advisor": adv.dict(),
            })
            prompt = (
                "You are EnerVision's assistant. Answer using the provided JSON. "
                "Include peak load/time (UTC+IST), anomaly count, and 2–3 concrete actions.\n"
                f"DATA:\n{context_json}\n\nQUESTION:\n{message}\n"
            )
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", "") or answer
            return {"ok": True, "reply": text, "used": used | {"llm": "gemini"}}
        except Exception as e:
            used["llm_error"] = str(e)

    return {"ok": True, "reply": answer, "used": used}

# -------------------------- Local run -----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
