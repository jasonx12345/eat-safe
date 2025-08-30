# Eat Safe – Toronto Food Safety Lookup

Search Toronto restaurant inspections by **name** or **address** and see the **latest inspection result** with a one-click history view.  
Built as a compact, end-to-end data project: **ingest → normalize → validate → publish → UI**.

[🌐 Live site](https://jasonx12345.github.io/eat-safe/) •
[CSV](https://jasonx12345.github.io/eat-safe/data/inspections.csv) •
SQLite (download): `public/data/inspections.db`

![Build & Deploy Pages](https://github.com/jasonx12345/eat-safe/actions/workflows/pages.yml/badge.svg)

---

## Features

-  **Type-ahead search** on restaurant name or address
-  **Outcome badge**: PASS / CONDITIONAL PASS / CLOSED
-  **Inspection history** (expand per venue)
-  **Fresh data** rebuilt by GitHub Actions (daily or on push)
-  **Static hosting** (GitHub Pages) — no server required
-  **SQLite artifact** with indexes + `latest_by_premise` view for analysis

Tech: **HTML, CSS, JavaScript (PapaParse), Python (pandas), SQLite**.

---

## Data pipeline at a glance

public/data/dinesafe.csv ──► etl/build_data.py
├─ writes: public/data/inspections.csv
├─ writes: public/data/inspections.db
└─ writes: public/data/metadata.json
frontend (index.html) ◄──── reads inspections.csv (PapaParse)


### Output schema (`public/data/inspections.csv`)
| column | description |
|---|---|
| `premise_id` | stable id (hash of name+address if none provided) |
| `name` | establishment name |
| `address` | street address |
| `inspection_date` | ISO date/time (UTC) |
| `result` | PASS / CONDITIONAL PASS / CLOSED (normalized) |
| `violation` | details text (when present) |
| `action` | notice / action text (when present) |
| `severity` | severity label (when present) |
| `code` | infraction/violation code (when present) |
| `critical_cnt` | critical infractions count (if provided) |
| `noncritical_cnt` | non-critical infractions count (if provided) |

---

## Repo layout



etl/
build_data.py # ETL: normalize, validate, export CSV + SQLite + metadata
config.yaml # source + optional field mappings
requirements.txt # pandas, requests, pyyaml, jsonschema
public/
index.html # static UI (search + history)
icons/ # logo assets
data/
dinesafe.csv # source CSV (committed)
inspections.csv # generated
inspections.db # generated
metadata.json # generated
.github/workflows/
pages.yml # builds ETL and deploys public/ to GitHub Pages


---

## Run locally

```powershell
# 1) create & activate venv (optional)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) install ETL deps
pip install -r etl/requirements.txt

# 3) ensure etl/config.yaml points to your local CSV:
#    source_url: "public/data/dinesafe.csv"

# 4) build data
python etl/build_data.py

# 5) serve the site
python -m http.server -d public 8080
# open http://127.0.0.1:8080
