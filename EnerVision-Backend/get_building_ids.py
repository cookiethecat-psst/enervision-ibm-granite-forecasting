import os
from influxdb_client import InfluxDBClient

URL  = os.getenv("INFLUX_URL", "http://localhost:8086")
ORG  = os.getenv("INFLUX_ORG", "enervision")
BUCK = os.getenv("INFLUX_BUCKET", "building_power")
TOK  = os.getenv("INFLUX_TOKEN", "dev-admin-token-please-change")

q = f'''
import "influxdata/influxdb/schema"
schema.tagValues(
  bucket: "{BUCK}",
  tag: "building_id",
  predicate: (r) => r._measurement == "meter",
  start: 2006-12-16T00:00:00Z,
)
'''
with InfluxDBClient(url=URL, token=TOK, org=ORG) as c:
    res = c.query_api().query(q)
    values = [rec.get_value() for table in res for rec in table.records]
    print("building_ids:")
    for v in values:
        print("  ", v)
