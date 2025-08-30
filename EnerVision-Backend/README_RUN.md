\# EnerVision Backend — quick run



\## 0) Clean up

Delete: app.py, cache\_utils.py, db.py, enervision (sqlite), infra/, node\_modules/, \_\_pycache\_\_.



\## 1) Install deps (Windows)

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt



\## 2) Set environment

\# copy .env.example to .env and fill in:

\# - GEMINI\_API\_KEY

\# - INFLUX\_TOKEN

\# (optional) tweak CORS\_ORIGINS



\## 3) Start services

\# Terminal A: Influx-backed API (forecast/anomalies/advisor)

uvicorn app\_influx:app --reload --host 127.0.0.1 --port 8001



\# Terminal B: Chat/orchestrator

uvicorn main:app --reload --host 127.0.0.1 --port 8000



\## 4) Test

\# Health:

http://127.0.0.1:8001/api/health

http://127.0.0.1:8000/api/health



\# Forecast (needs real UUID \& Influx data):

http://127.0.0.1:8001/api/forecast?building\_id=<UUID>



\# Chat:

POST http://127.0.0.1:8000/api/chat

{

&nbsp; "prompt": "what's the peak and how to reduce my bill?",

&nbsp; "role": "admin"

}



\## 5) Common fixes

\- 401/403 from Influx? => token or org/bucket wrong.

\- Empty forecast? => measurement/field/tag names wrong. Check:

&nbsp; INFLUX\_MEASUREMENT, INFLUX\_FIELD, INFLUX\_TAG\_BUILDING.

\- Redis not running? => in-memory cache will be used automatically.

\- Want demo data? => set ENABLE\_DEMO\_ROUTES=1 in .env and hit /forecast (non-API) or /api/forecast if enabled in main.py.



