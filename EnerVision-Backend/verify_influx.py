# verify_influx.py — find your actual Influx measurement/field/tag + sample building_id
import os
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient

UTC = timezone.utc
INFLUX_URL = os.getenv("INFLUX_URL", "http://127.0.0.1:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "")
INFLUX_ORG = os.getenv("INFLUX_ORG", "enervision")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "building_power")

def q(query: str):
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN or " ", org=INFLUX_ORG) as c:
        return c.query_api().query(org=INFLUX_ORG, query=query)

def show(title, tables, max_rows=10):
    print(f"\n== {title} ==")
    rows = []
    for t in tables:
        for r in t.records:
            rows.append(r.values)
            if len(rows) >= max_rows: break
        if len(rows) >= max_rows: break
    if rows:
        for i, r in enumerate(rows, 1):
            print(f"{i:02d}. {r}")
    else:
        print("(no rows)")

def main():
    print("Reading Influx bucket schema…")
    show("measurements", q(f'import "influxdata/influxdb/schema"\nschema.measurements(bucket: "{INFLUX_BUCKET}")'))

    # pick likely measurement
    candidates = ["power","energy","load","meter_readings","consumption"]
    chosen = None
    for m in candidates:
        t = q(f'import "influxdata/influxdb/schema"\nschema.fieldKeys(bucket: "{INFLUX_BUCKET}", predicate: (r) => r._measurement == "{m}")')
        if any(True for _ in t):
            chosen = m; break
    if not chosen:
        print("\nCould not auto-pick a measurement; choose from the list above and update .env.")
        return
    print(f"\nAuto-picked measurement: {chosen}")

    show(f"field keys for {chosen}", q(f'import "influxdata/influxdb/schema"\nschema.fieldKeys(bucket: "{INFLUX_BUCKET}", predicate: (r) => r._measurement == "{chosen}")'))
    show(f"tag keys for {chosen}", q(f'import "influxdata/influxdb/schema"\nschema.tagKeys(bucket: "{INFLUX_BUCKET}", predicate: (r) => r._measurement == "{chosen}")'))

    # guess building-like tag
    tag_guess = ["building_id","building","site","meter_id","bldg","home_id"]
    tag_key, tag_val = None, None
    for tg in tag_guess:
        t = q(f'import "influxdata/influxdb/schema"\nschema.tagValues(bucket: "{INFLUX_BUCKET}", predicate: (r) => r._measurement == "{chosen}", tag: "{tg}")')
        vals = []
        for tbl in t:
            for r in tbl.records:
                vals.append(str(r.get_value())); break
            if vals: break
        if vals:
            tag_key, tag_val = tg, vals[0]; break
    if not tag_key:
        print("\nCouldn’t guess a building tag key. Pick one from the tag-keys list above and update .env.")
        return
    print(f"\nAuto-picked tag: {tag_key} = {tag_val}")

    # guess numeric field
    field_candidates = ["kW","kw","value","power","active_power","consumption"]
    chosen_field = None
    for fk in field_candidates:
        stop = datetime.now(UTC)
        start = stop - timedelta(days=7)
        flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
  |> filter(fn: (r) => r._measurement == "{chosen}")
  |> filter(fn: (r) => r._field == "{fk}")
  |> filter(fn: (r) => r.{tag_key} == "{tag_val}")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> limit(n: 1)
'''
        t = q(flux)
        if any(True for _ in t):
            chosen_field = fk; break
    if not chosen_field:
        print("\nCouldn’t auto-pick a field. Choose one from the field-keys list and update .env.")
        return

    print("\n✅ Put these in your .env:")
    print(f"INFLUX_MEASUREMENT={chosen}")
    print(f"INFLUX_FIELD={chosen_field}")
    print(f"INFLUX_TAG_BUILDING={tag_key}")
    print(f'Example building_id to test: {tag_val}')

if __name__ == "__main__":
    main()
