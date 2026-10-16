# 🛰️ SAT-FIRE TRIAGE: Satellite-Derived Active Fire Decision Support System

> **⚠️ OPERATIONAL SAFETY DISCLAIMER**  
> **ANALYST DECISION SUPPORT ONLY — NOT AN EMERGENCY RESPONSE SYSTEM.**  
> Satellite thermal anomaly detections (such as NASA FIRMS MODIS and VIIRS observations) are remote-sensing indicators, **not ground-truth confirmation** of active wildfires. Detections may be caused by controlled agricultural burns, industrial flares, or solar glint, and fires may go undetected due to cloud cover or smoke attenuation. **Never use this system for emergency dispatch, evacuation, or life-safety decisions.**

---

## 📋 System Overview & Architecture

SAT-FIRE TRIAGE is a geospatial AI/ML decision-support dashboard built for wildfire analysts. It imports satellite thermal observations, validates fields, visualizes records on high-resolution satellite maps and filterable tables, groups hotspots into fire complexes using spatial clustering, and prioritizes anomalies using an **explainable multi-factor scoring model**.

Furthermore, it features a **Satellite Fire Observer** that enables analysts to target any location on Earth or pick random global fire regions to simulate polar satellite passes (NOAA-20/21 VIIRS, Aqua/Terra MODIS), swath geometry, and atmospheric attenuation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SAT-FIRE TRIAGE PIPELINE                          │
├─────────────────┬───────────────────┬───────────────────┬───────────────────┤
│ 1. Data Ingestion│ 2. Field Validation│ 3. DBSCAN Cluster │ 4. Explainable ML │
│  - NASA FIRMS CSV│  - Boundary check │  - 4 km Haversine │  - FRP Energy (30%)│
│  - Custom Upload │  - Kelvin checks  │  - Fire perimeters│  - Confidence(20%)│
│  - Satellite Pass│  - Missing values │  - Total FRP / MW │  - Delta-T (15%)  │
│  - Benchmark set │  - Glint warning  │  - Centroid calc  │  - Cluster (15%)  │
└─────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

---

## ✨ Key Features

### 1. Ingestion & Field Validation Engine
- Supports standard NASA FIRMS CSV formats for both **VIIRS (375m)** (`bright_ti4`, `bright_ti5`, `frp`, confidence `l`/`n`/`h`) and **MODIS (1km)** (`brightness`, `bright_t31`, `frp`, confidence `0-100%`).
- Rigorous boundary checks: coordinates in $[-90, 90]$ and $[-180, 180]$, positive Fire Radiative Power (MW), valid Kelvin brightness temperatures ($250\text{ K} - 550\text{ K}$), ISO acquisition dates, UTC timestamps.
- Automated data sanitization report displaying valid rows, sanitized warnings, and dropped invalid rows.

### 2. Explainable Multi-Factor Triage Scoring (0 - 100)
Rather than an uninterpretable black box, the triage priority score is computed from transparent, physically grounded factors:
$$S_{\text{triage}} = \sum_{i} w_i \cdot S_i$$
1. **Fire Radiative Power (FRP, 30% default)**: Logarithmically scaled active combustion power in Megawatts ($S_{\text{frp}} = \min(100, \frac{\ln(1 + \text{FRP})}{\ln(1 + 350)} \times 100)$).
2. **Satellite Detection Confidence (20% default)**: Normalized sensor algorithm confidence (e.g., VIIRS High = 95%, Nominal = 70%, Low = 30%).
3. **Thermal Contrast Delta ($\Delta T$, 15% default)**: Temperature difference between mid-IR fire channel ($4\,\mu\text{m}$) and thermal background ($11\,\mu\text{m}$). Large positive delta indicates high-temperature sub-pixel emitter.
4. **Spatial Multiplicity / Cluster Density (15% default)**: Haversine DBSCAN cluster membership. Distinguishes isolated noise pixels from large advancing fire fronts.
5. **Nighttime Reliability Bonus (10% default)**: Night passes eliminate solar glint off water, metal roofs, or greenhouses, providing higher thermal contrast.
6. **Sensor Nadir Quality (5% default)**: Centers of the swath have crisp spatial resolution ($0.375\,\text{km}$); pixels near scan edge suffer from bow-tie distortion ($>2.0\,\text{km}$).
7. **Temporal Persistence (5% default)**: Multi-pass recurrence across consecutive satellite overpasses.

Priority Tiers:
- 🔴 **CRITICAL** ($\ge 80$): High energy, large multi-pixel front, high confidence.
- 🟠 **HIGH** ($65 - 79$): Significant thermal anomaly.
- 🟡 **ELEVATED** ($45 - 64$): Moderate thermal anomaly or controlled burn.
- 🔵 **MONITOR / LOW** ($< 45$): Marginal, isolated, or low confidence detection.

### 3. Interactive Satellite Map & Swath Viewer
- Powered by Leaflet.js with multi-layer basemaps:
  - **🛰️ High-Resolution Satellite**: Esri World Imagery (Maxar/Airbus Earthstar)
  - **🌑 Mission Control Dark**: CartoDB Dark Matter
  - **🗺️ OpenStreetMap**
- **Fire Complexes**: Visualizes DBSCAN clusters with dynamic radius circles, total cluster FRP, and cluster tier badges.
- **Satellite Footprint**: Visualizes polar sun-synchronous swath polygons ($3,040\text{ km}$ wide) and sub-orbital ground track vectors.
- **Deep Audit Popups**: Click any hotspot to view telemetry and open the explainability breakdown.

### 4. 🛰️ Satellite Fire Observer & Global Location Search Bar
- **Global Search Bar**:
  - Live geocoding autocomplete in the header navigation and observer tab.
  - Type any city, country, region, national park, or direct coordinate pair (e.g. `California`, `Sydney`, `Athens`, `Jasper`, `Pantanal`, `Maui`, `37.77, -122.41`).
  - Powered by local offline fast-lookup and live OpenStreetMap Nominatim geocoding.
  - Automatically points the satellite to the searched area, displays the polar swath corridor, generates active thermal detections, and smoothly flies the Leaflet map.
- **🎲 Random Fire Region**: One-click acquisition of active fire regions across Earth (Canadian Boreal, Brazilian Pantanal, Siberian Taiga, Australian Outback, Greek Archipelago, Indonesian Peatlands, Chilean Valparaíso).
- **Radar Sweep & HUD Telemetry**: Animated radar sweep displaying satellite name (NOAA-20/21 VIIRS, Aqua/Terra MODIS), orbit number, swath corridor, local solar time, cloud cover fraction (%), and atmospheric attenuation.

### 5. Benchmark Evaluation & ML Metrics
- Evaluates triage model against **320 ground-truth annotated samples**:
  - `true_wildfire`: Active uncontrolled vegetation fires
  - `agricultural_burn`: Controlled seasonal crop stubble burns
  - `industrial_flare`: Refinery and gas flare points
  - `false_alarm_glint`: Solar glint off water/greenhouses and hot desert soil
- Computes **2x2 Interactive Confusion Matrix** (TP, FP, FN, TN).
- Computes **Precision, Recall, F1-Score, Overall Accuracy, and Specificity**.
- **Interactive Threshold Slider** ($20 - 90$) with real-time recalculation of metrics and Precision-Recall tradeoff curves.

### 6. Exportable Reports
- **CSV Export**: Downloads processed observations with computed triage scores, ranks, clusters, and validation status.
- **Analyst JSON Briefing**: Structured executive summary report with top-ranked incidents, key factor drivers, and mandatory limitations notice.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- `flask`, `requests`

### Quick Launch
1. Ensure dependencies are installed:
   ```bash
   pip install flask requests
   ```
2. Start the application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 🧪 Running Automated Tests
Run the built-in test suite to verify all 9 API endpoints:
```bash
python test_app.py
```
Output:
```text
Testing Satellite Fire Triage API Endpoints...
✅ GET /: OK (HTML UI Rendered)
✅ GET /api/datasets: OK (5 datasets available)
✅ POST /api/dataset/load: OK (190 records, 15 clusters)
✅ POST /api/evaluation/run: OK (Precision: 79.9%, Recall: 99.4%, F1: 88.6%)
✅ GET /api/observer/presets: OK (12 presets loaded)
✅ POST /api/observer/random: OK (Target: Valparaíso, Chile, Sat: NOAA-20)
✅ POST /api/observer/observe (Custom Coords): OK (Jasper Boreal Alberta)
✅ GET /api/export/csv: OK (CSV successfully generated)
✅ GET /api/export/report: OK (Analyst JSON Briefing generated)

🎉 ALL 9 SYSTEM TESTS PASSED SUCCESSFULLY!
```

---

## 📁 Repository Structure
```
vyom/
├── app.py                     # Flask application & REST API server
├── triage_engine.py           # Field validation, DBSCAN clustering, explainable ranking
├── satellite_observer.py      # Satellite orbital pass simulation & random region engine
├── evaluation_engine.py       # Benchmark evaluation, confusion matrix, PR curves
├── generate_datasets.py       # NASA FIRMS dataset generator
├── test_app.py                # Automated integration test suite
├── data/                      # Datasets directory
│   ├── sample_firms_viirs.csv # Western USA / California Wildfires (VIIRS 375m)
│   ├── sample_amazon_fires.csv# Amazon Basin & Pantanal Wetlands (VIIRS 375m)
│   ├── sample_australia_fires.csv # Australia Bushfires (MODIS 1km)
│   ├── sample_mediterranean_fires.csv # Greece / Aegean Wildfires (VIIRS 375m)
│   └── labelled_benchmark.csv # 320 ground-truth annotated validation samples
├── static/
│   ├── css/
│   │   └── style.css          # NASA mission-control theme styling
│   └── js/
│       ├── app.js             # Main dashboard UI coordinator & table filters
│       ├── map.js             # Leaflet map, satellite layers, cluster rendering
│       ├── charts.js          # Chart.js scatter, donut, and evaluation charts
│       └── observer.js        # Satellite Fire Observer interactive module
└── templates/
    └── index.html             # Responsive dashboard template
```

---

## 🔬 Scientific Limitations & Caveats
1. **Cloud & Smoke Blindness**: Thermal infrared ($3.7 - 4.0\,\mu\text{m}$) cannot penetrate liquid clouds or thick pyrocumulonimbus plumes. Fires under clouds are omitted entirely.
2. **Orbital Latency**: Polar sun-synchronous satellites revisit a given area 2 to 4 times per 24 hours. Fires spreading between overpasses are not detected in real time.
3. **Sub-Pixel Resolution**: A single $375\,\text{m}$ VIIRS pixel reports coordinates at the pixel center; it may contain a small $50\,\text{m}^2$ intense flaming front or a large smoldering patch.
4. **False Alarms**: Industrial flare stacks, solar panels, and permitted agricultural burns generate valid thermal anomalies but are not uncontrolled wildfires.
