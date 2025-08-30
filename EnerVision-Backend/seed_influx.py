# seed_influx.py — with guaranteed anomalies
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid5, NAMESPACE_URL

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

INFLUX_URL    = "http://127.0.0.1:8087"
INFLUX_TOKEN  = "dev-token"
INFLUX_ORG    = "enervision"
INFLUX_BUCKET = "enervision"
MEASUREMENT   = "building_load"

N_POINTS      = 2880   # ~1 day at 30 sec
STEP_SECONDS  = 30
BUILDING_ID   = str(uuid5(NAMESPACE_URL, "enervision/example/building/1"))

def main():
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    now = datetime.now(timezone.utc)
    start = now - timedelta(seconds=N_POINTS * STEP_SECONDS)

    points = []
    ts = start
    for i in range(N_POINTS):
        hour = ts.hour + ts.minute / 60
        base = 1.5 + 0.8 * (1 if 9 <= hour <= 22 else 0)   # normal load
        ac = 0.4 * max(0, (1 - abs((hour - 18) / 3)))      # evening bump
        kw = round(base + ac, 3)

        # 🔥 Inject anomalies
        if i % 500 == 0:    # every ~4h
            kw += 3.5       # big spike
        if i % 777 == 0:    # random dip
            kw -= 1.5

        p = (
            Point(MEASUREMENT)
            .tag("building_id", BUILDING_ID)
            .field("kW", kw)
            .time(ts, WritePrecision.S)
        )
        points.append(p)
        ts += timedelta(seconds=STEP_SECONDS)

    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
    print(f"✅ Wrote {len(points)} points with anomalies injected")
    client.close()

if __name__ == "__main__":
    main()
