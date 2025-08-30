# seed_probe.py — write 600 points (10 hours) safely and verify
import os
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

URL    = os.getenv("INFLUX_URL", "http://127.0.0.1:8087")
ORG    = os.getenv("INFLUX_ORG", "enervision")
BUCKET = os.getenv("INFLUX_BUCKET", "enervision")
TOKEN  = os.getenv("INFLUX_TOKEN")

MEAS   = "building_load"
BID    = "probe-enervision-1"  # simple building_id for demo

print(f"Using URL={URL}, ORG={ORG}, BUCKET={BUCKET}, TOKEN startswith={TOKEN[:6]}...")

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG, timeout=30000)
health = client.health()
assert health.status == "pass", f"Influx health failed: {health.message}"

write_api = client.write_api(write_options=SYNCHRONOUS)

# Write 600 points: one every minute over the past 10 hours
now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
start = now - timedelta(minutes=599)

count = 0
for i in range(600):
    ts = start + timedelta(minutes=i)
    # simple load pattern: base + sinusoidal wave
    kw = 2.0 + (i % 60) * 0.05
    p = (
        Point(MEAS)
        .tag("building_id", BID)
        .field("kW", float(kw))
        .time(ts, WritePrecision.S)
    )
    write_api.write(bucket=BUCKET, org=ORG, record=p)
    count += 1
print(f"✅ wrote {count} points for building_id={BID}")

# Verify via Flux
flux = f'''
from(bucket: "{BUCKET}")
  |> range(start: -12h)
  |> filter(fn: (r) => r._measurement == "{MEAS}")
  |> filter(fn: (r) => r.building_id == "{BID}")
  |> filter(fn: (r) => r._field == "kW")
  |> count()
'''
tables = client.query_api().query(query=flux, org=ORG)

total = 0
for t in tables:
    for r in t.records:
        total += int(r.get_value())

print(f"🔎 verified {total} points for building_id={BID}")
client.close()
