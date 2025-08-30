import os
import datetime as dt
from db import PG, write_meter, shutdown_clients

# --- 1) Run schema once ---
sql_path = os.path.join(os.path.dirname(__file__), "sql", "001_init.sql")
with open(sql_path, "r", encoding="utf-8") as f:
    sql = f.read()
with PG.cursor() as cur:
    cur.execute(sql)

BUILDING_NAME = "Alpha Tower"
BUILDING_CITY = "Hyderabad"
METER_KEY = "MTR-001"

# --- 2) Get-or-create building (no unique constraint on name, so SELECT first) ---
with PG.cursor() as cur:
    cur.execute(
        "SELECT id FROM buildings WHERE name = %s AND city = %s LIMIT 1;",
        (BUILDING_NAME, BUILDING_CITY),
    )
    row = cur.fetchone()
    if row:
        bldg = row[0]
    else:
        cur.execute(
            """
            INSERT INTO buildings(name, city, lat, lon)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (BUILDING_NAME, BUILDING_CITY, 17.385, 78.486),
        )
        bldg = cur.fetchone()[0]

# --- 3) Upsert meter (meter_key is UNIQUE in schema) ---
with PG.cursor() as cur:
    cur.execute(
        """
        INSERT INTO meters(building_id, meter_key, phase)
        VALUES (%s, %s, %s)
        ON CONFLICT (meter_key) DO NOTHING
        RETURNING id;
        """,
        (bldg, METER_KEY, "1P"),
    )
    row = cur.fetchone()
    if row:
        mtr = row[0]
    else:
        cur.execute("SELECT id FROM meters WHERE meter_key = %s;", (METER_KEY,))
        mtr = cur.fetchone()[0]

# --- 4) Write 60s of synthetic points to Influx (synchronous writes) ---
now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
for i in range(60):
    ts = now - dt.timedelta(seconds=60 - i)
    write_meter(bldg, mtr, ts, power_w=1500 + (i % 10) * 20)

print("BUILDING_ID:", bldg)
print("METER_ID:", mtr)

# --- 5) Close clients cleanly ---
shutdown_clients()
