import os
from influxdb_client import InfluxDBClient

INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_ORG = os.getenv("INFLUX_ORG", "enervision")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "building_power")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "dev-admin-token-please-change")

def main():
    q = f'''
from(bucket:"{INFLUX_BUCKET}")
  |> range(start: 2006-12-16T00:00:00Z)
  |> filter(fn:(r) => r._measurement == "meter")
  |> limit(n:10)
'''
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as c:
        for t in c.query_api().query_stream(q):
            print(t)

if __name__ == "__main__":
    main()
