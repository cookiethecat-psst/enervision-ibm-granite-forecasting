# seed_db.py
import sqlite3, uuid, random, datetime, os
DB = "enervision.db"
if os.path.exists(DB):
    print("DB already exists:", DB); raise SystemExit
con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("""
CREATE TABLE meter_readings(
  building_id TEXT NOT NULL,
  ts TEXT NOT NULL,              -- ISO Z
  value REAL NOT NULL
)""")
bid = str(uuid.uuid4())
now = datetime.datetime.utcnow().replace(second=0, microsecond=0)
# 7 days of 1-minute data with a mild evening peak
for d in range(7*24*60):
    ts = now - datetime.timedelta(minutes=(7*24*60 - d))
    # base + evening bump
    val = 1.5 + 0.8*random.random()
    if 13 <= ts.hour <= 18: val += 0.6
    if 18 <= ts.hour <= 22: val += 1.2
    cur.execute("INSERT INTO meter_readings(building_id, ts, value) VALUES (?,?,?)",
                (bid, ts.isoformat()+"Z", val))
con.commit(); con.close()
print("Seeded DB. Use building_id:", bid)
