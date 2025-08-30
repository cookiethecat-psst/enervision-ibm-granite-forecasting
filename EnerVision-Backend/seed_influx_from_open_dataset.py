# seed_influx.py — reliable seeder (synchronous write)
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid5, NAMESPACE_URL

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ---------------- Config ----------------
INFLUX_URL    = os.getenv("INFLUX_URL", "http://127.0.0.1:8087")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN", "")
INFLUX_ORG    = os.getenv("INFLUX_ORG", "enervision")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "enervision")
MEASUREMENT   = "building_load"

N_POINTS      = 2880   # ~1 day at 30 sec
STEP_SECONDS  = 30

# deterministic building_id
BUILDING_ID   = str(uuid5(NAMESPACE_URL, "enervision/example/building/1"))


def main():
    print(f"Writing seed data to {INFLUX_URL}, bucket={INFLUX_BUCKET}, org={INFLUX_ORG}")
    print(f"Example building_id: {BUILDING_ID}")

    # create client
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    # verify connection
    health = client.health()
    if health.status != "pass":
        raise RuntimeError(f"InfluxDB not healthy: {health.message}")

    # synchronous write API
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # generate points (synthetic pattern)
    now = datetime.now(timezone.utc)
    start = now - timedelta(seconds=N_POINTS * STEP_SECONDS)

    points = []
    ts = start
    for i in range(N_POINTS):
        hour = ts.hour + ts.minute / 60
        base = 1.5 + 0.8 * (1 if 9 <= hour <= 22 else 0)   # day vs night load
        ac = 0.4 * max(0, (1 - abs((hour - 18) / 3)))       # evening bump
        kw = round(base + ac, 3)

        p = (
            Point(MEASUREMENT)
            .tag("building_id", BUILDING_ID)
            .field("kW", kw)
            .time(ts, WritePrecision.S)
        )
        points.append(p)
        ts += timedelta(seconds=STEP_SECONDS)

    # write in chunks
    BATCH = 1000
    total = 0
    for i in range(0, len(points), BATCH):
        chunk = points[i:i+BATCH]
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=chunk)
        total += len(chunk)
        print(f"wrote {total}/{len(points)} points…")

    print("✅ Done writing")

    # verify count
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -2d)
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> filter(fn: (r) => r.building_id == "{BUILDING_ID}")
      |> filter(fn: (r) => r._field == "kW")
      |> count()
    '''
    tables = client.query_api().query(query=query, org=INFLUX_ORG)
    count = 0
    for table in tables:
        for rec in table.records:
            count += int(rec.get_value())

    print(f"🔎 Verified {count} points in bucket='{INFLUX_BUCKET}' for building_id={BUILDING_ID}")

    client.close()


if __name__ == "__main__":
    main()
