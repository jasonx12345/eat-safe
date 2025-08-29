# tools/make_inspections.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src  = ROOT / "datasets" / "toronto_dinesafe.csv"   # adjust if your file lives elsewhere
dst  = ROOT / "public" / "data" / "inspections.csv"

# Load the raw Toronto CSV
df = pd.read_csv(src)

# Map the columns the UI expects
out = pd.DataFrame({
    "name": df.get("Establishment Name"),
    "address": df.get("Establishment Address"),
    "inspection_date": df.get("Inspection Date"),
    "result": df.get("Outcome"),
})

# Clean + drop empties
out = out.dropna(subset=["name","address"]).copy()

# Normalize date strings (optional)
out["inspection_date"] = pd.to_datetime(out["inspection_date"], errors="coerce").dt.date.astype(str)

# Save where the site reads from
dst.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dst, index=False)
print(f"Wrote {len(out):,} rows -> {dst}")
