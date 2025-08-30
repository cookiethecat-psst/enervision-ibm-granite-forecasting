# seed_influx.py — write 2 days of minute-level demo data to Influx v2 (synchronous write)
import os, math, random
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

UTC = timezone.utc

INFLUX_URL = os.getenv("INFLUX_URL", "http://127.0.0.1:8087")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "dev-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "enervision")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "enervision")

MEAS = os.getenv("INFLUX_MEASUREMENT", "power")
FIELD = os.getenv("INFLUX_FIELD", "kW")
TAG_BUILDING = os.getenv("INFLUX_TAG_BUILDING", "building_id")

# use your UUID or keep this one
BUILDING_ID = os.getenv("SEED_BUILDING_ID", "016b34c1-0c63-537f-b0ab-11653df967b8")

def main():
    print("Writing seed data to:", INFLUX_URL, "bucket:", INFLUX_BUCKET)
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    start = now - timedelta(days=2)

    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as c:
        write_api = c.write_api(write_options=SYNCHRONOUS)

        t = start
        batch = []
        count = 0
        while t <= now:
            # simple diurnal curve: base 6–18 kW + small noise
            hod = t.hour + t.minute/60.0
            diurnal = 8 + 6 * (0.5 * (1 + math.sin((hod/24.0)*2*math.pi - math.pi/2)))
            val = diurnal + random.uniform(-0.6, 0.6)
            p = Point(MEAS).tag(TAG_BUILDING, BUILDING_ID).field(FIELD, float(val)).time(t)
            batch.append(p)

            if len(batch) >= 5000:
                write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=batch)
                count += len(batch)
                print(f"wrote {count} points…")
                batch = []

            t += timedelta(minutes=1)

        if batch:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=batch)
            count += len(batch)
            print(f"wrote {count} points…")

    print("✅ done. Example building_id:", BUILDING_ID)

if __name__ == "__main__":
    main()
