# app_influx.py — EnerVision Backend (InfluxDB v2 + Redis + Gemini Chat)
# ----------------------------------------------------------------------
# - Robust startup/shutdown; clients on app.state
# - Dependencies via request.app.state (no Depends leaks)
# - Clean plain-text replies (no markdown **), no error spam
# - Sanitizes bucket/measurement/field (removes stray quotes)
# - General Qs use Gemini directly; data Qs use building context
# - Endpoints: /health, /forecast, /anomalies, /advisor, POST /api/chat
# ----------------------------------------------------------------------

import os, json, logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient
import redis.asyncio as redis

# Optional: Gemini (graceful fallback if missing)
_GOOGLE_AVAILABLE = False
try:
    import google.generativeai as genai
    _GOOGLE_AVAILABLE = True
except Exception:
    _GOOGLE_AVAILABLE = False

# -------------------- Config --------------------
INFLUX_URL   = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-token")
INFLUX_ORG   = os.getenv("INFLUX_ORG", "my-org")
REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# -------------------- App/Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("enervision-influx")

app = FastAPI(title="EnerVision Influx Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Lifecycle --------------------
@app.on_event("startup")
async def startup():
    # Influx
    app.state.influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    app.state.query_api = app.state.influx.query_api()
    # Redis
    app.state.redis = await redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    log.info("Redis connected")
    log.info("Influx client ready")

    # Gemini
    if _GOOGLE_AVAILABLE and GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            app.state.gemini = genai.GenerativeModel(GEMINI_MODEL)
            log.info("Gemini client ready")
        except Exception as e:
            app.state.gemini = None
            log.warning(f"Gemini init failed (fallback): {e}")
    else:
        app.state.gemini = None
        log.info("GOOGLE_API_KEY not set or SDK unavailable; chat will use fallback.")

@app.on_event("shutdown")
async def shutdown():
    try:
        await app.state.redis.close()
    except Exception:
        pass
    try:
        app.state.influx.close()
    except Exception:
        pass

# -------------------- Dependencies --------------------
def get_query_api(request: Request):
    return request.app.state.query_api

def get_redis(request: Request):
    return request.app.state.redis

def get_gemini(request: Request):
    return request.app.state.gemini

# -------------------- Helpers --------------------
def _clean_str(x: Optional[str], default: str = "") -> str:
    if not x:
        return default
    # remove stray quotes/whitespace: handles \"power\" from UI/env
    return str(x).strip().strip('"').strip("'").strip()

def _make_flux(bucket: str, hours: int, measurement: str, field: str) -> str:
    return f'''
from(bucket: "{bucket}")
  |> range(start: -{hours}h)
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => r["_field"] == "{field}")
  |> aggregateWindow(every: 15m, fn: mean, createEmpty: false)
  |> yield(name: "mean")
'''.strip()

async def read_series(q, *, bucket: str, hours: int, measurement: str, field: str) -> List[Dict[str, Any]]:
    flux = _make_flux(bucket=bucket, hours=hours, measurement=measurement, field=field)
    try:
        tables = q.query(flux, org=INFLUX_ORG)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Influx query failed: {e}")

    points: List[Dict[str, Any]] = []
    for t in tables:
        for rec in t.records:
            points.append({"t": rec.get_time().isoformat(), "v": rec.get_value()})
    points.sort(key=lambda x: x["t"])
    return points

def zscore_anomalies(values: List[float], z: float = 3.0) -> List[int]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr)) or 1.0
    idx = np.where(np.abs((arr - mu) / sd) >= z)[0]
    return idx.tolist()

async def build_context(*, q, r, bucket: str, hours: int, measurement: str, field: str) -> Dict[str, Any]:
    key = f"ctx:{bucket}:{hours}:{measurement}:{field}"
    cached = await r.get(key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    series = await read_series(q, bucket=bucket, hours=hours, measurement=measurement, field=field)
    vals = [p["v"] for p in series]
    times = [p["t"] for p in series]

    avg = float(np.mean(vals)) if vals else 0.0
    peak = float(np.max(vals)) if vals else 0.0
    p95 = float(np.percentile(vals, 95)) if vals else 0.0

    an_idx = zscore_anomalies(vals, z=3.0)
    anomalies = [{"t": times[i], "v": vals[i]} for i in an_idx]

    ctx = {
        "summary": {"points": len(series), "avg": avg, "peak": peak, "p95": p95},
        "series": series,
        "anomalies": anomalies,
        "advice": [
            "Shift large appliance use (laundry, EVs) after 22:00.",
            "Increase AC temperature by 1–2°C during 18:00–22:00.",
            "Power down high-standby devices when not in use."
        ],
    }
    await r.setex(key, 60, json.dumps(ctx))
    return ctx

def _fallback_ai_reply(user_msg: str, ctx: Dict[str, Any]) -> str:
    s = ctx.get("summary", {})
    avg  = float(s.get("avg") or 0.0)
    peak = float(s.get("peak") or 0.0)
    p95  = float(s.get("p95") or 0.0)
    adv  = ctx.get("advice", [])
    tips = ("\n- " + "\n- ".join(adv[:4])) if adv else ""
    return (
        "Energy consumption status\n"
        f"Average load: {avg:.2f} kW\n"
        f"Peak load: {peak:.2f} kW (95th percentile ≈ {p95:.2f} kW)\n"
        "Recommended actions:" + tips + "\n\n"
        "You can ask: What should I move out of 18–22h today?"
    )

def _needs_context(user_msg: str) -> bool:
    """True if the question is about the user's own usage/forecast/costs."""
    msg = (user_msg or "").lower()
    keys = [
        "my ", "our ", "today", "tonight", "tomorrow",
        "bill", "cost", "save", "reduce", "peak", "off-peak",
        "ac", "hvac", "ev", "laundry", "shift", "forecast",
        "load", "anomaly", "kwh", "kw", "between", "18", "22"
    ]
    return any(k in msg for k in keys)

# -------------------- Routes --------------------
@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/forecast")
async def forecast(
    bucket: str,
    hours: int = 24,
    measurement: str = "power",
    field: str = "value",
    q = Depends(get_query_api),
):
    bucket = _clean_str(bucket, bucket)
    measurement = _clean_str(measurement, "power")
    field = _clean_str(field, "value")
    series = await read_series(q, bucket=bucket, hours=hours, measurement=measurement, field=field)
    return {"ok": True, "data": series}

@app.get("/anomalies")
async def anomalies(
    bucket: str,
    hours: int = 24,
    z: float = 3.0,
    measurement: str = "power",
    field: str = "value",
    q = Depends(get_query_api),
):
    bucket = _clean_str(bucket, bucket)
    measurement = _clean_str(measurement, "power")
    field = _clean_str(field, "value")
    series = await read_series(q, bucket=bucket, hours=hours, measurement=measurement, field=field)
    values = [p["v"] for p in series]
    times  = [p["t"] for p in series]
    idx = zscore_anomalies(values, z=z)
    anoms = [{"t": times[i], "v": values[i]} for i in idx]
    return {"ok": True, "count": len(anoms), "data": anoms}

@app.post("/advisor")
async def advisor(
    body: Dict[str, Any],
    q = Depends(get_query_api),
):
    bucket = _clean_str((body or {}).get("bucket"), "")
    if not bucket:
        raise HTTPException(status_code=400, detail="Missing 'bucket'")
    hours = int((body or {}).get("hours", 24))
    measurement = _clean_str((body or {}).get("measurement"), "power")
    field       = _clean_str((body or {}).get("field"), "value")

    series = await read_series(q, bucket=bucket, hours=hours, measurement=measurement, field=field)
    values = [p["v"] for p in series]
    if not values:
        return {"ok": True, "advice": ["No recent data found to generate advice."]}

    avg = float(np.mean(values))
    peak = float(np.max(values))
    msg = [
        f"Average load in the last {hours}h is about {avg:.2f} kW; peak reached {peak:.2f} kW.",
        "Shift deferrable loads after 22:00.",
        "Increase AC setpoint by 1–2°C during 18:00–22:00.",
        "If you have solar+battery, discharge during peak and charge after midnight."
    ]
    return {"ok": True, "advice": msg}

@app.post("/api/chat")
async def chat(
    body: Dict[str, Any],
    q = Depends(get_query_api),
    r = Depends(get_redis),
    g = Depends(get_gemini),
):
    user_msg   = _clean_str((body or {}).get("message"), "Give me a brief status.")
    bucket     = _clean_str((body or {}).get("bucket"), _clean_str(os.getenv("DEFAULT_BUCKET"), "enervision"))
    hours      = int((body or {}).get("hours", 24))
    measurement = _clean_str((body or {}).get("measurement"), "power")
    field       = _clean_str((body or {}).get("field"), "value")

    # General questions: answer directly (no context, no notes)
    if not _needs_context(user_msg):
        if g is not None:
            try:
                prompt = (
                    "You are EnerVision, an accurate and concise energy expert. "
                    "Answer in plain text without markdown. "
                    "If it is a definition or general concept, give a crisp, helpful explanation.\n\n"
                    f"User: {user_msg}\n"
                )
                resp = g.generate_content(prompt)
                text = getattr(resp, "text", None) or (
                    resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else ""
                )
                if not text:
                    text = "I’m here to help with energy questions. Could you rephrase that?"
                return {"ok": True, "reply": text, "model": GEMINI_MODEL}
            except Exception as e:
                log.warning(f"Gemini general reply failed, using fallback: {e}")
                basics = {
                    "solar": "Solar energy is electricity or heat produced from sunlight using photovoltaic panels or solar thermal systems.",
                    "ev": "An electric vehicle runs on electricity stored in a battery and is charged from the grid or solar.",
                    "ac": "Air conditioners move heat from indoors to outdoors; higher ISEER/EER means better efficiency."
                }
                key = next((k for k in basics if k in user_msg.lower()), None)
                return {"ok": True, "reply": basics.get(key, "Sorry, I couldn’t fetch a full answer just now."), "model": "fallback"}
        else:
            return {"ok": True, "reply": "Solar energy is produced from sunlight using photovoltaic or thermal systems.", "model": "fallback"}

    # Data-related questions: build context; never crash the user
    note = ""
    try:
        ctx = await build_context(q=q, r=r, bucket=bucket, hours=hours, measurement=measurement, field=field)
    except Exception as e:
        msg = str(e).lower()
        if "could not find bucket" in msg:
            note = f"Live data unavailable: bucket not found ({bucket})."
        elif "401" in msg or "unauthorized" in msg:
            note = "Live data unavailable: Influx token not authorized."
        elif "org" in msg and "not found" in msg:
            note = "Live data unavailable: Influx org not found."
        else:
            note = "Live data unavailable."
        ctx = {
            "summary": {"points": 0, "avg": 0.0, "peak": 0.0, "p95": 0.0},
            "series": [],
            "anomalies": [],
            "advice": [
                "Shift large appliance use (laundry, EVs) after 22:00.",
                "Increase AC temperature by 1–2°C during 18:00–22:00.",
                "Power down high-standby devices when not in use."
            ]
        }

    def with_note(text: str) -> str:
        return f"{text}\n\n{note}" if note else text

    if g is not None:
        try:
            prompt = (
                "You are EnerVision, an energy advisor for Indian buildings. "
                "Use the JSON context (series/anomalies/summary) to answer the user's question about their usage. "
                "Be concise, plain text, and action-oriented.\n\n"
                f"User: {user_msg}\n\n"
                f"Context JSON:\n{json.dumps(ctx)[:200000]}\n"
            )
            resp = app.state.gemini.generate_content(prompt)
            text = getattr(resp, "text", None) or (
                resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else ""
            )
            if not text:
                text = _fallback_ai_reply(user_msg, ctx)
            return {"ok": True, "reply": with_note(text), "model": GEMINI_MODEL}
        except Exception as e:
            log.warning(f"Gemini context reply failed, using fallback: {e}")
            return {"ok": True, "reply": with_note(_fallback_ai_reply(user_msg, ctx)), "model": "fallback"}
    else:
        return {"ok": True, "reply": with_note(_fallback_ai_reply(user_msg, ctx)), "model": "fallback"}
