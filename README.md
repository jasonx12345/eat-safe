# Food Safety Near Campus (Static)

Student-friendly map of restaurant inspection outcomes using official city open data (CSV/GeoJSON). No scraping, no backend. Data is built nightly via GitHub Actions and served on GitHub Pages.

## Configure
Edit `etl/config.yaml`:
- `source_url`: CSV/GeoJSON export or FeatureServer `f=geojson` query
- `format`: `csv` or `geojson`
- `field_map`: match your dataset's column headers

## Build locally
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r etl/requirements.txt
python etl/build_data.py
python -m http.server -d public 8000
