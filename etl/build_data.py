# etl/build_data.py
import os, io, json, hashlib, re
from datetime import datetime, timezone
from dateutil import parser as dateparser
import pandas as pd
import requests
import yaml
from jsonschema import validate
import sqlite3

ROOT = os.path.dirname(os.path.dirname(__file__))
PUBLIC = os.path.join(ROOT, "public", "data")

def load_config():
    with open(os.path.join(ROOT, "etl", "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_local(path, fmt):
    abspath = os.path.abspath(os.path.join(ROOT, path))
    if fmt == "csv":
        return pd.read_csv(abspath)
    elif fmt == "geojson":
        with open(abspath, "r", encoding="utf-8") as f:
            data = json.load(f)
        feats = data.get("features", [])
        rows = []
        for ft in feats:
            props = ft.get("properties") or {}
            geom = ft.get("geometry") or {}
            if geom and geom.get("type") == "Point":
                coords = geom.get("coordinates") or [None, None]
                props["_lon"] = coords[0]
                props["_lat"] = coords[1]
            rows.append(props)
        return pd.DataFrame(rows)
    else:
        raise SystemExit(f"Unsupported format: {fmt}")

def download_to_df(cfg):
    url = cfg["source_url"]
    fmt = cfg.get("format","csv").lower()
    if url.startswith("http://") or url.startswith("https://"):
        print(f"[etl] downloading {fmt} from {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        if fmt == "csv":
            df = pd.read_csv(io.BytesIO(r.content))
        elif fmt == "geojson":
            data = r.json()
            feats = data.get("features", [])
            rows = []
            for ft in feats:
                props = ft.get("properties") or {}
                geom = ft.get("geometry") or {}
                if geom and geom.get("type") == "Point":
                    coords = geom.get("coordinates") or [None, None]
                    props["_lon"] = coords[0]
                    props["_lat"] = coords[1]
                rows.append(props)
            df = pd.DataFrame(rows)
        else:
            raise SystemExit(f"Unsupported format: {fmt}")
    else:
        print(f"[etl] reading {fmt} from {url}")
        df = read_local(url, fmt)
    print(f"[etl] raw rows: {len(df)}")
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(name: str) -> str:
        return re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower())
    return df.rename(columns={c: norm(c) for c in df.columns})

def get_col(df, candidates):
    for c in candidates:
        if c and c in df.columns:
            return c
    return None

def compute_score(last12):
    score = 100
    closed = int(last12.get("closed", 0) or 0)
    conditional = int(last12.get("conditional", 0) or 0)
    crit = int(last12.get("critical", 0) or 0)
    any_recent = int(last12.get("any_recent", 0) or 0)
    if closed > 0: score -= 40
    score -= 25 * min(conditional, 2)
    score -= 5 * crit
    if not any_recent: score -= 10
    score = max(0, min(score, 100))
    if closed > 0 or score < 40:
        bucket = "avoid for now"
    elif (conditional > 0) or (40 <= score < 70):
        bucket = "caution"
    else:
        bucket = "safe"
    return score, bucket

def write_sqlite(df: pd.DataFrame, db_path: str):
    cols = ["premise_id","name","address","inspection_date","result",
            "violation","action","severity","code","critical_cnt","noncritical_cnt"]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0 if c in ("critical_cnt","noncritical_cnt") else ""
    out["inspection_date"] = pd.to_datetime(out["inspection_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["critical_cnt"] = pd.to_numeric(out["critical_cnt"], errors="coerce").fillna(0).astype(int)
    out["noncritical_cnt"] = pd.to_numeric(out["noncritical_cnt"], errors="coerce").fillna(0).astype(int)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            premise_id      TEXT NOT NULL,
            name            TEXT NOT NULL,
            address         TEXT NOT NULL,
            inspection_date TEXT NOT NULL,
            result          TEXT,
            violation       TEXT,
            action          TEXT,
            severity        TEXT,
            code            TEXT,
            critical_cnt    INTEGER DEFAULT 0,
            noncritical_cnt INTEGER DEFAULT 0,
            PRIMARY KEY (premise_id, inspection_date)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inspections_name    ON inspections(name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inspections_address ON inspections(address);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inspections_date    ON inspections(inspection_date);")

    rows = list(out[cols].itertuples(index=False, name=None))
    cur.executemany("""
        INSERT OR REPLACE INTO inspections
        (premise_id,name,address,inspection_date,result,violation,action,severity,code,critical_cnt,noncritical_cnt)
        VALUES (?,?,?,?,?,?,?,?,?,?,?);
    """, rows)

    cur.execute("DROP VIEW IF EXISTS latest_by_premise;")
    cur.execute("""
        CREATE VIEW latest_by_premise AS
        WITH ranked AS (
          SELECT *,
                 ROW_NUMBER() OVER (
                   PARTITION BY premise_id
                   ORDER BY inspection_date DESC
                 ) AS rn
          FROM inspections
        )
        SELECT * FROM ranked WHERE rn = 1;
    """)
    conn.commit()
    conn.close()

def main():
    cfg = load_config()
    years = int(cfg.get("years", 0))
    df = download_to_df(cfg)
    df = normalize_columns(df)

    fm = cfg.get("field_map", {}) or {}
    # Core columns (try config mapping first, then common fallbacks)
    col_id   = get_col(df, [fm.get("id")])
    col_name = get_col(df, [fm.get("name"), "premise_name","establishmentname","businessname","name","establishment_name"])
    col_addr = get_col(df, [fm.get("address"), "establishment_address","address","address1","location_address"])
    col_lat  = get_col(df, [fm.get("lat"), "latitude","lat","_lat"])
    col_lon  = get_col(df, [fm.get("lon"), "longitude","lon","long","_lon"])
    col_date = get_col(df, [fm.get("inspection_date"), "inspection_date","inspectiondate","date","inspection_date_time"])
    col_res  = get_col(df, [fm.get("result"), "result","outcome","status"])

    # Details columns (new: code)
    col_violation = get_col(df, [fm.get("violation"), "infraction_details","details","violation_description","violation"])
    col_action    = get_col(df, [fm.get("action"), "action","notice"])
    col_severity  = get_col(df, [fm.get("severity"), "severity","severity_category"])
    col_code      = get_col(df, [
        fm.get("infraction_code") if fm else None,
        "infraction_code","infraction code","code","infractioncode","infraction_code_id","infractionid","code_id"
    ])

    col_crit = get_col(df, [fm.get("critical_cnt"), "critical_infractions","critical","criticalcount","infractions_critical"])
    col_ncr  = get_col(df, [fm.get("noncritical_cnt"), "noncritical_infractions","non_critical","noncriticalcount","infractions_noncritical"])

    need = [col_name, col_addr, col_date, col_res]
    if any(x is None for x in need):
        raise SystemExit(f"Missing required columns; got name={col_name}, address={col_addr}, inspection_date={col_date}, result={col_res}. Edit etl/config.yaml field_map.")

    # Parse and time-window
    df["_date"] = pd.to_datetime(df[col_date], errors="coerce", utc=True)
    dropped = df["_date"].isna().sum()
    if dropped:
        print(f"[etl] dropped {dropped} rows with invalid/missing dates")
    df = df.dropna(subset=["_date"])
    if years > 0:
        cutoff = pd.Timestamp.utcnow() - pd.DateOffset(years=years)
        dfw = df[df["_date"] >= cutoff].copy()
    else:
        print(f"[etl] no time filtering (years=0): {len(df)} rows")
        dfw = df.copy()

    # Normalize result labels
    def norm_result(s):
        if not isinstance(s, str): return ""
        t = s.strip().lower()
        if "conditional" in t: return "CONDITIONAL PASS"
        if "closed" in t: return "CLOSED"
        if "pass" in t: return "PASS"
        if "fail" in t: return "CLOSED"
        return s.upper()
    dfw["_result"] = dfw[col_res].apply(norm_result)

    # Premise ID
    if col_id and col_id in dfw.columns:
        dfw["_premise_id"] = dfw[col_id].astype(str)
    else:
        def mkid(row):
            key = (str(row.get(col_name,"")).strip() + "|" + str(row.get(col_addr,"")).strip()).lower()
            return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        dfw["_premise_id"] = dfw.apply(mkid, axis=1)

    # Numeric counts
    for c,name in [(col_crit,"_crit"), (col_ncr,"_ncr")]:
        if c and c in dfw.columns:
            dfw[name] = pd.to_numeric(dfw[c], errors="coerce").fillna(0).astype(int)
        else:
            dfw[name] = 0

    # Prepare detail columns with safe defaults
    dfw["_violation"] = dfw[col_violation] if col_violation else ""
    dfw["_action"]    = dfw[col_action]    if col_action    else ""
    dfw["_severity"]  = dfw[col_severity]  if col_severity  else ""
    dfw["_code"]      = dfw[col_code]      if col_code      else ""

    # Latest per premise (for geojson; UI reads CSV directly)
    latest_idx = dfw.sort_values("_date").groupby("_premise_id")["_date"].idxmax()
    latest = dfw.loc[latest_idx, ["_premise_id", col_name, col_addr, col_lat, col_lon, "_date","_result","_violation","_action","_severity","_code"]].copy()
    latest = latest.rename(columns={col_name:"name", col_addr:"address"})
    latest["latest_date"] = latest["_date"].dt.strftime("%Y-%m-%d")
    latest["latest_result"] = latest["_result"]

    # 12mo aggregates for scoring
    since = pd.Timestamp.utcnow() - pd.DateOffset(years=1)
    d12 = dfw[dfw["_date"] >= since].copy()
    agg = d12.groupby("_premise_id").apply(lambda g: pd.Series({
        "closed": int((g["_result"]=="CLOSED").sum()),
        "conditional": int((g["_result"]=="CONDITIONAL PASS").sum()),
        "critical": int(g["_crit"].sum()),
        "noncritical": int(g["_ncr"].sum()),
        "any_recent": int(len(g) > 0),
    })).reset_index()

    merged = latest.merge(agg, on="_premise_id", how="left").fillna(0)
    scores, buckets = [], []
    for _,row in merged.iterrows():
        sc, bk = compute_score({
            "closed": row["closed"],
            "conditional": row["conditional"],
            "critical": row["critical"],
            "any_recent": row["any_recent"]
        })
        scores.append(sc); buckets.append(bk)
    merged["score"] = scores
    merged["bucket"] = buckets

    # GeoJSON (optional; map not used on UI, but keep)
    feats = []
    for _,row in merged.iterrows():
        lat = row.get(col_lat) if col_lat else None
        lon = row.get(col_lon) if col_lon else None
        geom = None
        if pd.notna(lat) and pd.notna(lon):
            try:
                geom = {"type":"Point","coordinates":[float(lon), float(lat)]}
            except Exception:
                geom = None
        props = {
            "premise_id": row["_premise_id"],
            "name": row["name"],
            "address": row["address"],
            "latest_result": row["latest_result"],
            "latest_date": row["latest_date"],
            "score": float(row["score"]),
            "bucket": row["bucket"],
            "counts_12mo": {
                "closed": int(row["closed"]),
                "conditional": int(row["conditional"]),
                "critical": int(row["critical"]),
                "noncritical": int(row["noncritical"]),
            },
            "violation": str(row.get("_violation") or ""),
            "action": str(row.get("_action") or ""),
            "severity": str(row.get("_severity") or ""),
            "code": str(row.get("_code") or ""),
        }
        feats.append({"type":"Feature","geometry": geom, "properties": props})
    fc = {"type":"FeatureCollection","features": feats}

    with open(os.path.join(ROOT,"etl","schemas","venues.schema.json"),"r",encoding="utf-8") as f:
        schema = json.load(f)
    validate(instance=fc, schema=schema)

    os.makedirs(PUBLIC, exist_ok=True)
    venues_path = os.path.join(PUBLIC, "venues.geojson")
    with open(venues_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    print(f"[etl] wrote {venues_path} (features={len(fc['features'])})")

    # CSV export (now includes code + details)
    export_cols = ["_premise_id", col_name, col_addr, "_date", "_result", "_violation", "_action", "_severity", "_code", "_crit", "_ncr"]
    export = dfw[export_cols].rename(columns={
        "_premise_id":"premise_id",
        col_name:"name",
        col_addr:"address",
        "_date":"inspection_date",
        "_result":"result",
        "_violation":"violation",
        "_action":"action",
        "_severity":"severity",
        "_code":"code",
        "_crit":"critical_cnt",
        "_ncr":"noncritical_cnt"
    }).sort_values(["premise_id","inspection_date"], ascending=[True, False])

    csv_path = os.path.join(PUBLIC, "inspections.csv")
    export.to_csv(csv_path, index=False)
    print(f"[etl] wrote {csv_path} (rows={len(export)})")

    # SQLite (optional, but great signal)
    db_path = os.path.join(PUBLIC, "inspections.db")
    write_sqlite(export, db_path)
    print(f"[etl] wrote {db_path} (rows={len(export)})")

    # metadata.json
    meta = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": cfg["source_url"],
        "total_venues": len(fc["features"]),
        "window": {
            "start": (pd.Timestamp.utcnow() - pd.DateOffset(years=years)).date().isoformat() if years>0 else None,
            "end": datetime.now(timezone.utc).date().isoformat()
        }
    }
    with open(os.path.join(PUBLIC, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[etl] wrote metadata.json")

if __name__ == "__main__":
    main()
