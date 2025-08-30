# main.py — EnerVision Backend (Gemini 1.5 Flash + InfluxDB integration)
import os
import json
import statistics
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from influxdb_client import InfluxDBClient

# ------------------ ENV & CONFIG ------------------
load_dotenv()

# Gemini config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment.")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Backend base
SELF_BASE = os.getenv("SELF_BASE", "http://127.0.0.1:8000")

# InfluxDB config
INFLUX_URL    = "http://127.0.0.1:8087"
INFLUX_TOKEN  = "dev-token"
INFLUX_ORG    = "enervision"
INFLUX_BUCKET = "enervision"
MEASUREMENT   = "building_load"

# API prefix
API_PREFIX = os.getenv("API_PREFIX", "/api")

# CORS
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app = FastAPI(title="EnerVision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ ROOT & HEALTH ------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "EnerVision Backend",
        "hint": "Use /health, /forecast, /anomalies, /advisor, /api/chat",
        "self_base": SELF_BASE,
    }

@app.get("/health")
@app.get(f"{API_PREFIX}/health")
def health():
    return {"ok": True, "time_utc": datetime.now(timezone.utc).isoformat()}

# ------------------ REQ MODEL ------------------
class ChatIn(BaseModel):
    prompt: str
    role: str = "resident"              # "admin" | "resident"
    flat_id: Optional[str] = None
    building_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    tariff_inr_per_kwh: Optional[float] = 7.0
    peak_window: Optional[str] = "18:00–22:00"

# ------------------ InfluxDB-backed Endpoints ------------------
@app.get(f"{API_PREFIX}/forecast")
@app.get("/forecast")
def get_forecast():
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> filter(fn: (r) => r._field == "kW")
      |> keep(columns: ["_time", "_value"])
    '''
    tables = query_api.query(query=query, org=INFLUX_ORG)

    data = []
    for table in tables:
        for record in table.records:
            data.append({
                "time": record["_time"].isoformat(),
                "value": record["_value"]
            })
    client.close()
    return {"forecast": data}

@app.get(f"{API_PREFIX}/anomalies")
@app.get("/anomalies")
def get_anomalies():
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> filter(fn: (r) => r._field == "kW")
      |> keep(columns: ["_time", "_value"])
    '''
    tables = query_api.query(query=query, org=INFLUX_ORG)
    records = []
    for table in tables:
        for record in table.records:
            records.append({
                "time": record["_time"].isoformat(),
                "value": float(record["_value"])
            })
    client.close()

    anomalies = []
    for r in records:
        v = r["value"]
        if v < 1.8 or v > 2.8:  # simple band rule
            anomalies.append({
                "time": r["time"],
                "value": v,
                "reason": "Load outside expected band (1.8–2.8 kW)"
            })

    return {"anomalies": anomalies[:5]}

@app.get(f"{API_PREFIX}/advisor")
@app.get("/advisor")
def get_advisor():
    forecast = get_forecast().get("forecast", [])
    anomalies = get_anomalies().get("anomalies", [])

    prompt = f"""
You are EnerVision’s AI advisor for Indian apartments.
Here is today's forecast load data and detected anomalies.

Forecast (recent points):
{json.dumps(forecast[:15], indent=2)}

Anomalies detected:
{json.dumps(anomalies, indent=2)}

Instructions:
- Provide 3–5 practical, India-specific energy-saving tips.
- If anomalies exist, mention them explicitly with timestamp and value.
- Suggest corrective action for each anomaly (e.g., "Check AC thermostat at 17:43 (3.3 kW)").
- Keep advice concise, friendly, and actionable.
"""
    resp = make_model().generate_content(prompt)
    text = getattr(resp, "text", None) or "No advice generated."
    tips = [line.strip("-• ") for line in text.splitlines() if line.strip()]
    return {"tips": tips[:5]} if tips else {"tips": ["No advice available right now."]}

# ------------------ Context Helpers ------------------
def _compute_peak_info(forecast_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    peak_time, peak_val = None, None
    for pt in forecast_list or []:
        v = float(pt.get("value")) if isinstance(pt.get("value"), (int, float)) else None
        if v is None:
            continue
        if peak_val is None or v > peak_val:
            peak_val = v
            peak_time = pt.get("time")
    return {"peak_time": peak_time, "peak_value": peak_val}

def build_runtime_context(user: ChatIn) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    fc = get_forecast()
    an = get_anomalies()
    adv = get_advisor()

    peak_info = _compute_peak_info(fc.get("forecast", []))
    ctx: Dict[str, Any] = {
        "timestamp_utc": now_iso,
        "role": user.role,
        "building_id": user.building_id,
        "flat_id": user.flat_id,
        "tariff_inr_per_kwh": user.tariff_inr_per_kwh,
        "peak_window": user.peak_window,
        "forecast": fc.get("forecast", []),
        "anomalies": an.get("anomalies", []),
        "advisor": adv.get("tips", []),
        "peak_info": peak_info,
    }
    return ctx

# ------------------ Gemini ------------------
SYSTEM_PROMPT = """
You are EnerVision's AI advisor for Indian apartments.
Use forecast, anomalies, and advisor context to give answers.
First line = TL;DR, then 4–6 bullets with numbers (kWh, ₹, %).
If anomalies exist, mention their times/values and give corrective actions.
"""

def make_model():
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"temperature": 0.6, "top_p": 0.9, "max_output_tokens": 900},
    )

# ------------------ Chat ------------------
@app.post(f"{API_PREFIX}/chat")
@app.post("/api/chat")
def api_chat(body: ChatIn):
    ctx = build_runtime_context(body)

    # compute daily kWh + estimated bill
    daily_kwh = None
    try:
        vals = [pt["value"] for pt in ctx.get("forecast", []) if isinstance(pt.get("value"), (int, float))]
        if vals:
            daily_kwh = round(sum(vals), 2)
    except Exception:
        pass

    computed = {
        "daily_kwh_sum": daily_kwh,
        "est_monthly_bill_inr": round(daily_kwh * 30 * float(body.tariff_inr_per_kwh)) if daily_kwh else None,
    }

    payload_json = json.dumps({"context": ctx, "computed_features": computed}, ensure_ascii=False)

    user_prompt = f"""
User role: {body.role}
User question: {body.prompt}

Today's context (forecast, anomalies, advisor tips, computed features):
{payload_json}

Answering rules:
- If user asks about "anomalies":
    • List anomalies with timestamp & value.
    • Suggest specific corrective action for each.
- If user asks about "bill" or "cost":
    • Use est_monthly_bill_inr from computed features.
    • Show calculation clearly.
- Otherwise:
    • Blend forecast, anomalies, and advisor tips.
    • Start with TL;DR line, then 4–6 bullets.
    • Be concise, India-specific, and practical.
"""
    resp = make_model().generate_content(user_prompt)
    text = getattr(resp, "text", None) or "I couldn't produce a reply from the available data."
    return {"reply": text}
