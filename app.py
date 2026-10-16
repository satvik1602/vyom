"""
Satellite-Derived Active Fire Triage System
Flask Web Application & API Server

Architecture Overview:
    This Flask application serves as the web API and front-end controller for the
    Satellite-Derived Active Fire Triage System. It orchestrates a three-stage
    processing pipeline:

        1. **Field Validation** (FieldValidator) — Ingests raw CSV records from
           NASA FIRMS (VIIRS 375m / MODIS 1km) format, validates coordinate ranges,
           normalizes brightness temperatures, FRP, confidence levels, and metadata.

        2. **Geospatial Clustering** (GeospatialClusterer) — Groups individual
           hotspot pixels into coherent fire complexes using a Haversine-based DBSCAN
           algorithm with configurable neighborhood radius (eps_km).

        3. **Explainable Triage Ranking** (ExplainableTriageRanker) — Assigns each
           detection a composite priority score (0–100) derived from 7 physically
           meaningful factors, then classifies into priority tiers (CRITICAL / HIGH /
           ELEVATED / MONITOR-LOW).

    The app also provides:
        - A satellite observer module for simulating orbital passes over arbitrary
          coordinates with biome-aware hotspot generation.
        - A benchmark evaluation engine for computing confusion matrices, precision,
          recall, F1, and threshold curves against ground-truth labelled datasets.
        - CSV and JSON export endpoints for analyst reporting.
        - A geocode endpoint with a 3-stage lookup strategy (coordinate parsing →
          curated fast-lookup table → live OSM Nominatim).

    All state is held in-memory in the global STATE dictionary (single-session,
    no persistent database). The front-end is served from templates/index.html
    with static assets in static/.

Dependencies:
    - Flask (web framework)
    - requests (for Nominatim geocoding)
    - triage_engine (FieldValidator, GeospatialClusterer, ExplainableTriageRanker)
    - satellite_observer (SatelliteObserver)
    - evaluation_engine (EvaluationEngine)
"""

# ===== STANDARD LIBRARY IMPORTS =====
import os
import csv
import io
import json
import re
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, make_response

# ===== PROJECT MODULE IMPORTS =====
# triage_engine: the three-stage validation → clustering → ranking pipeline
from triage_engine import FieldValidator, GeospatialClusterer, ExplainableTriageRanker
# satellite_observer: simulated satellite orbital pass and hotspot generation
from satellite_observer import SatelliteObserver
# evaluation_engine: confusion matrix and performance metrics against ground truth
from evaluation_engine import EvaluationEngine

# ===== FLASK APP INITIALIZATION =====
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload

# ===== IN-MEMORY APPLICATION STATE =====
# Global session state dictionary. Holds the currently loaded dataset and all
# intermediate/final pipeline outputs. This is a single-session, non-persistent
# store — refreshing or restarting the server resets to defaults.
STATE = {
    # Name or label of the currently active dataset (displayed in the UI header)
    'current_dataset_name': 'sample_firms_viirs.csv',

    # Raw records as-read from CSV before any validation or transformation
    'raw_records': [],

    # Records that passed field validation and normalization (coordinates, FRP,
    # brightness, confidence mapped to standard schema)
    'validated_records': [],

    # Records after DBSCAN geospatial clustering (each record annotated with
    # cluster_id, cluster_size, and in_cluster flag)
    'clustered_records': [],

    # Final output: records sorted by composite priority_score, with triage_rank,
    # priority_tier, factor decomposition, and analyst caveats attached
    'ranked_records': [],

    # Aggregated per-cluster statistics: centroid, bbox, total/mean/max FRP,
    # radius, count, and cluster tier classification
    'cluster_summaries': [],

    # Summary of the validation pass: total_raw, valid_count, warnings, errors
    'validation_summary': {},

    # Current user-configured scoring weights (sum to 100) for the 7 triage
    # factors: frp, confidence, temp_delta, cluster_size, nighttime,
    # nadir_quality, persistence
    'weights': dict(ExplainableTriageRanker.DEFAULT_WEIGHTS)
}

# ===== PRESET DATASET CATALOG =====
# Pre-packaged CSV datasets stored in the data/ directory. Each entry describes
# the geographic region, sensor platform, biome context, and approximate record
# count to help the user choose a scenario.
PRESET_DATASETS = {
    # Western USA wildfire complex: Sierra Nevada, Dixie fire, Trinity Alps,
    # and Central Valley agricultural anomalies
    'sample_firms_viirs.csv': {
        'name': 'Western USA / California Wildfires (VIIRS 375m)',
        'description': 'Sierra Nevada & Dixie fire complexes, Trinity Alps, Central Valley agricultural burns and scattered anomalies.',
        'satellite': 'NOAA-20 / Suomi-NPP VIIRS',
        'records_count': 190
    },
    # Amazon & Pantanal: deforestation arc fires, agricultural burning,
    # and intense wetland fire fronts in Rondônia and Mato Grosso
    'sample_amazon_fires.csv': {
        'name': 'Amazon Basin & Pantanal Wetlands (VIIRS 375m)',
        'description': 'Rondônia deforestation arc, Mato Grosso agricultural fires, and intense Pantanal wetland fire fronts.',
        'satellite': 'NOAA-20 / NOAA-21 VIIRS',
        'records_count': 220
    },
    # Australian Black Summer-style bushfires: Blue Mountains NSW and
    # East Gippsland Victoria mega-fire complexes with pyrocumulus activity
    'sample_australia_fires.csv': {
        'name': 'Southeastern Australia Bushfire Front (MODIS 1km)',
        'description': 'Mega-fire complexes across Blue Mountains NSW and East Gippsland Victoria with massive pyrocumulus.',
        'satellite': 'Terra & Aqua MODIS',
        'records_count': 180
    },
    # Mediterranean summer fires: Attica, Evia, and Rhodes wind-driven
    # wildland-urban interface events with sea-breeze interaction
    'sample_mediterranean_fires.csv': {
        'name': 'Mediterranean Summer Fires (Greece & Aegean)',
        'description': 'Attica, Evia, and Rhodes high-wind interface wildfires with sea-breeze interaction.',
        'satellite': 'NOAA-20 / NOAA-21 VIIRS',
        'records_count': 150
    },
    # Ground-truth annotated benchmark for evaluation: contains verified
    # true_wildfire, agricultural_burn, industrial_flare, and false_alarm_glint
    # labels for computing precision/recall/F1 and threshold curves
    'labelled_benchmark.csv': {
        'name': 'Labelled Evaluation Benchmark (Ground-Truth Annotated)',
        'description': '320 ground-truth verified observations: True Wildfires, Controlled Agricultural Burns, Industrial Gas Flares, and Solar Glint.',
        'satellite': 'Multi-sensor (VIIRS & MODIS)',
        'records_count': 320
    }
}

# ===== CSV FILE LOADER =====

def load_csv_file(filepath):
    """Parses a CSV file into list of dicts.

    Reads a CSV file from the local filesystem using DictReader, which maps
    each row to a dictionary keyed by the header column names. This preserves
    all original column names and string values; numeric conversion happens
    downstream in FieldValidator.validate_and_normalize().

    Args:
        filepath: Relative or absolute path to the CSV file (e.g., 'data/sample_firms_viirs.csv').

    Returns:
        A list of dictionaries, one per CSV row. Returns an empty list if the
        file does not exist or is empty.
    """
    records = []
    # Guard against missing files (e.g., user-specified path doesn't exist)
    if not os.path.exists(filepath):
        return records
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert each OrderedDict row to a plain dict for downstream compatibility
            records.append(dict(row))
    return records

# ===== TRIAGE PROCESSING PIPELINE =====

def process_pipeline(raw_records, weights=None, eps_km=4.0):
    """Executes the validation -> clustering -> triage ranking pipeline.

    This is the central orchestration function that chains the three core
    processing stages in sequence. Each stage transforms and enriches the
    records:

        Stage 1 — Validation: Rejects records with invalid/missing coordinates,
            normalizes brightness temperatures (K), FRP (MW), confidence levels,
            and extracts metadata (satellite, instrument, day/night flag).

        Stage 2 — Clustering: Groups validated hotspot pixels into fire complexes
            using Haversine-distance DBSCAN. Points within eps_km of each other
            (and meeting min_samples threshold) form coherent clusters. Isolated
            points are labelled "Isolated".

        Stage 3 — Ranking: Computes a 0–100 composite priority_score for each
            record using 7 weighted factors (FRP, confidence, thermal delta,
            cluster size, nighttime bonus, nadir quality, persistence). Assigns
            a priority tier (CRITICAL ≥80, HIGH ≥65, ELEVATED ≥45, MONITOR/LOW <45)
            and generates human-readable factor explanations and analyst caveats.

    Args:
        raw_records: List of dicts as returned by load_csv_file() or CSV upload.
        weights: Optional dict of scoring weights (keys: frp, confidence,
            temp_delta, cluster_size, nighttime, nadir_quality, persistence).
            If None, uses ExplainableTriageRanker.DEFAULT_WEIGHTS.
        eps_km: DBSCAN neighborhood radius in kilometers. Default 4.0 km groups
            hotspots within ~4 km into the same fire complex.

    Returns:
        Tuple of (ranked_records, cluster_summaries, val_summary):
            - ranked_records: List of dicts sorted by priority_score descending,
              each enriched with triage_rank, priority_tier, factors, and caveats.
            - cluster_summaries: List of per-cluster aggregate statistics (centroid,
              bbox, total FRP, cluster tier).
            - val_summary: Dict summarizing validation pass (counts, warnings, errors).
    """
    # 1. Field validation & normalization — converts raw CSV strings to typed,
    #    range-checked, standardized records; invalid rows are rejected with errors
    valid_records, val_summary = FieldValidator.validate_and_normalize(raw_records)
    
    # 2. Geospatial DBSCAN clustering — groups nearby hotspots into fire complexes
    #    using great-circle (Haversine) distance with eps_km radius
    clusterer = GeospatialClusterer(eps_km=eps_km, min_samples=2)
    clustered_records, cluster_summaries = clusterer.cluster(valid_records)
    
    # 3. Explainable triage priority ranking — scores each record on 7 physical
    #    factors, assigns tier, generates analyst-facing explanations and caveats
    ranker = ExplainableTriageRanker(weights=weights)
    ranked_records = ranker.rank(clustered_records)
    
    return ranked_records, cluster_summaries, val_summary

# ===== ROUTE: INDEX PAGE =====

@app.route('/')
def index():
    """Serves the main single-page application HTML.

    HTTP Method: GET
    Response: Rendered index.html template (the full analyst dashboard UI).
    """
    return render_template('index.html')

# ===== ROUTE: LIST AVAILABLE DATASETS =====

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Returns the catalog of available preset datasets and the currently active one.

    HTTP Method: GET
    Response JSON:
        {
            "current": "<active dataset filename>",
            "datasets": { <filename>: { name, description, satellite, records_count }, ... }
        }
    """
    return jsonify({
        "current": STATE['current_dataset_name'],
        "datasets": PRESET_DATASETS
    })

# ===== ROUTE: LOAD A PRESET DATASET =====

@app.route('/api/dataset/load', methods=['POST'])
def load_dataset():
    """Loads a preset CSV dataset from the data/ directory and runs the full pipeline.

    HTTP Method: POST
    Request JSON Body:
        { "dataset_name": "<filename>" }   (defaults to 'sample_firms_viirs.csv')

    Processing Steps:
        1. Resolve the file path under data/ and verify it exists.
        2. Parse the CSV into raw records via load_csv_file().
        3. Run the full triage pipeline (validate → cluster → rank).
        4. Update the global STATE with all results.
        5. Compute summary statistics (counts per tier, total FRP, etc.).

    Response JSON:
        {
            "success": true,
            "dataset_name": "...",
            "info": { <dataset metadata> },
            "stats": { total_records, total_clusters, total_frp_mw, max_frp_mw,
                       critical_count, high_count, elevated_count, low_count,
                       has_ground_truth },
            "validation_summary": { ... },
            "cluster_summaries": [ ... ],
            "records": [ ... ]
        }
    Error: 404 if dataset file is not found.
    """
    data = request.get_json() or {}
    dataset_name = data.get('dataset_name', 'sample_firms_viirs.csv')
    
    # Resolve path relative to project root's data/ directory
    filepath = os.path.join('data', dataset_name)
    if not os.path.exists(filepath):
        return jsonify({"error": f"Dataset file {dataset_name} not found"}), 404
        
    # Parse CSV and run the full 3-stage pipeline
    raw_records = load_csv_file(filepath)
    ranked_records, cluster_summaries, val_summary = process_pipeline(raw_records, STATE['weights'])
    
    # Persist results in the global session state
    STATE['current_dataset_name'] = dataset_name
    STATE['raw_records'] = raw_records
    STATE['ranked_records'] = ranked_records
    STATE['cluster_summaries'] = cluster_summaries
    STATE['validation_summary'] = val_summary
    
    # Compute aggregate statistics for the front-end dashboard cards
    stats = {
        "total_records": len(ranked_records),
        "total_clusters": len(cluster_summaries),
        "total_frp_mw": round(sum(r['frp'] for r in ranked_records), 1),
        "max_frp_mw": max([r['frp'] for r in ranked_records]) if ranked_records else 0,
        # Count records in each priority tier for the tier breakdown bar chart
        "critical_count": sum(1 for r in ranked_records if r['priority_tier'] == 'CRITICAL'),
        "high_count": sum(1 for r in ranked_records if r['priority_tier'] == 'HIGH'),
        "elevated_count": sum(1 for r in ranked_records if r['priority_tier'] == 'ELEVATED'),
        "low_count": sum(1 for r in ranked_records if r['priority_tier'] == 'MONITOR / LOW'),
        # Flag whether this dataset contains ground_truth_label for enabling evaluation
        "has_ground_truth": any(r.get('ground_truth_label') for r in ranked_records)
    }
    
    return jsonify({
        "success": True,
        "dataset_name": dataset_name,
        "info": PRESET_DATASETS.get(dataset_name, {}),
        "stats": stats,
        "validation_summary": val_summary,
        "cluster_summaries": cluster_summaries,
        "records": ranked_records
    })

# ===== ROUTE: UPLOAD CUSTOM CSV DATASET =====

@app.route('/api/dataset/upload', methods=['POST'])
def upload_dataset():
    """Accepts a user-uploaded CSV file, parses it, and runs the full pipeline.

    HTTP Method: POST (multipart/form-data)
    Request: File field named 'file' containing a CSV with FIRMS-compatible columns.

    Processing Steps:
        1. Validate that a file was uploaded and the filename is non-empty.
        2. Read the file stream, decode as UTF-8, parse with csv.DictReader.
        3. Run the full triage pipeline on the parsed records.
        4. Update global STATE and return results identical to load_dataset().

    Response JSON: Same structure as /api/dataset/load.
    Errors:
        - 400 if no file uploaded, empty filename, or empty/invalid CSV.
        - 500 if CSV parsing fails with an exception.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    try:
        # Read the uploaded file stream into memory and decode as UTF-8 text
        content = file.stream.read().decode('utf-8')
        # Wrap in StringIO so csv.DictReader can iterate line by line
        stream = io.StringIO(content)
        reader = csv.DictReader(stream)
        raw_records = [dict(row) for row in reader]
        
        if not raw_records:
            return jsonify({"error": "CSV is empty or has invalid headers"}), 400
            
        # Run the validation → clustering → ranking pipeline
        ranked_records, cluster_summaries, val_summary = process_pipeline(raw_records, STATE['weights'])
        
        # Store with a descriptive name indicating this is a user upload
        STATE['current_dataset_name'] = f"Uploaded: {file.filename}"
        STATE['raw_records'] = raw_records
        STATE['ranked_records'] = ranked_records
        STATE['cluster_summaries'] = cluster_summaries
        STATE['validation_summary'] = val_summary
        
        # Compute dashboard summary statistics (same as load_dataset)
        stats = {
            "total_records": len(ranked_records),
            "total_clusters": len(cluster_summaries),
            "total_frp_mw": round(sum(r['frp'] for r in ranked_records), 1),
            "max_frp_mw": max([r['frp'] for r in ranked_records]) if ranked_records else 0,
            "critical_count": sum(1 for r in ranked_records if r['priority_tier'] == 'CRITICAL'),
            "high_count": sum(1 for r in ranked_records if r['priority_tier'] == 'HIGH'),
            "elevated_count": sum(1 for r in ranked_records if r['priority_tier'] == 'ELEVATED'),
            "low_count": sum(1 for r in ranked_records if r['priority_tier'] == 'MONITOR / LOW'),
            "has_ground_truth": any(r.get('ground_truth_label') for r in ranked_records)
        }
        
        return jsonify({
            "success": True,
            "dataset_name": STATE['current_dataset_name'],
            "stats": stats,
            "validation_summary": val_summary,
            "cluster_summaries": cluster_summaries,
            "records": ranked_records
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse uploaded CSV: {str(e)}"}), 500

# ===== ROUTE: RECALCULATE TRIAGE WITH UPDATED WEIGHTS =====

@app.route('/api/triage/recalculate', methods=['POST'])
def recalculate_triage():
    """Re-runs the triage pipeline on the current dataset with updated weights or eps_km.

    This endpoint allows the analyst to adjust the relative importance of each
    scoring factor (FRP, confidence, thermal delta, cluster size, etc.) and/or
    change the DBSCAN clustering radius, then immediately see updated triage
    rankings without re-uploading the data.

    HTTP Method: POST
    Request JSON Body:
        {
            "weights": { "frp": 30, "confidence": 20, ... },  // optional, defaults to current
            "eps_km": 4.0  // optional, DBSCAN neighborhood radius in km
        }

    Response JSON:
        {
            "success": true,
            "cluster_summaries": [ ... ],
            "records": [ ... ]   // re-ranked records with updated scores and tiers
        }
    Error: 400 if no dataset is currently loaded.
    """
    data = request.get_json() or {}
    weights = data.get('weights', STATE['weights'])
    eps_km = float(data.get('eps_km', 4.0))
    
    if not STATE['raw_records']:
        return jsonify({"error": "No dataset currently loaded"}), 400
        
    # Persist the new weights so subsequent operations use them
    STATE['weights'] = weights
    # Re-run the full pipeline from raw records with updated parameters
    ranked_records, cluster_summaries, val_summary = process_pipeline(STATE['raw_records'], weights, eps_km=eps_km)
    STATE['ranked_records'] = ranked_records
    STATE['cluster_summaries'] = cluster_summaries
    
    return jsonify({
        "success": True,
        "cluster_summaries": cluster_summaries,
        "records": ranked_records
    })

# ===== ROUTE: RUN BENCHMARK EVALUATION =====

@app.route('/api/evaluation/run', methods=['POST'])
def run_evaluation():
    """Runs the evaluation engine to compute performance metrics against ground truth.

    If the currently loaded dataset contains ground_truth_label annotations,
    those records are evaluated directly. Otherwise, the system automatically
    falls back to loading the labelled_benchmark.csv dataset which contains
    320 verified records across 4 categories (true_wildfire, agricultural_burn,
    industrial_flare, false_alarm_glint).

    HTTP Method: POST
    Request JSON Body:
        { "threshold": 60.0 }   // priority_score threshold for positive classification

    Response JSON:
        {
            "has_data": true,
            "sample_size": N,
            "threshold": 60.0,
            "confusion_matrix": { tp, fp, fn, tn },
            "metrics": { accuracy, precision, recall, specificity, f1_score },
            "subgroups": { <label>: { total, flagged_high, flagged_low }, ... },
            "threshold_curve": [ { threshold, precision, recall, f1, fpr }, ... ],
            "scientific_limitations": [ ... ]
        }
    """
    data = request.get_json() or {}
    threshold = float(data.get('threshold', 60.0))
    
    # Check if current dataset has ground truth; if not, load labelled_benchmark.csv
    # as the fallback evaluation dataset
    records_to_eval = STATE['ranked_records']
    if not any(r.get('ground_truth_label') for r in records_to_eval):
        # Load the dedicated benchmark dataset and run the pipeline on it
        benchmark_raw = load_csv_file('data/labelled_benchmark.csv')
        records_to_eval, _, _ = process_pipeline(benchmark_raw, STATE['weights'])
        
    # Compute confusion matrix, metrics, subgroup breakdown, and threshold curves
    eval_result = EvaluationEngine.evaluate(records_to_eval, threshold=threshold)
    return jsonify(eval_result)

# ===== CURATED GEOCODING FAST-LOOKUP TABLE =====
# A hand-curated list of globally significant fire-prone locations. This table
# serves as a fast, offline first-check for the geocode endpoint, enabling
# instant results for well-known wildfire regions without hitting the external
# Nominatim API. Each entry includes:
#   - name: Short canonical name for substring matching
#   - display_name: Full descriptive label shown to the user
#   - lat/lon: Representative center coordinates
#   - biome: Fire ecology / vegetation classification for context
POPULAR_GEO_PLACES = [
    # ---- North America ----
    {"name": "California, USA", "display_name": "California, United States (Wildfire Corridor)", "lat": 38.58, "lon": -121.49, "biome": "Chaparral & Coniferous Forest"},
    {"name": "Los Angeles, CA", "display_name": "Los Angeles, California, USA", "lat": 34.05, "lon": -118.24, "biome": "Coastal Chaparral / WUI"},
    {"name": "San Francisco, CA", "display_name": "San Francisco, California, USA", "lat": 37.77, "lon": -122.42, "biome": "Coastal Mediterranean"},
    {"name": "Jasper National Park, Canada", "display_name": "Jasper National Park, Alberta, Canada", "lat": 52.87, "lon": -118.08, "biome": "Boreal Spruce & Pine Forest"},
    {"name": "Fort McMurray, Canada", "display_name": "Fort McMurray, Alberta, Canada", "lat": 56.73, "lon": -111.38, "biome": "Boreal Peat & Black Spruce"},
    # ---- South America ----
    {"name": "Amazon Rainforest, Brazil", "display_name": "Amazon Rainforest Basin, Brazil", "lat": -3.46, "lon": -62.21, "biome": "Dense Tropical Humid Rainforest"},
    {"name": "Pantanal Basin, Brazil", "display_name": "Pantanal Wetland Basin, Brazil", "lat": -16.85, "lon": -56.80, "biome": "Tropical Wetland Grassland"},
    # ---- Oceania ----
    {"name": "Sydney, Australia", "display_name": "Sydney & Blue Mountains, NSW, Australia", "lat": -33.86, "lon": 151.20, "biome": "Dry Sclerophyll Eucalyptus"},
    {"name": "Melbourne, Australia", "display_name": "Melbourne & Victoria High Country, Australia", "lat": -37.81, "lon": 144.96, "biome": "Eucalyptus Temperate Forest"},
    # ---- Europe / Mediterranean ----
    {"name": "Athens, Greece", "display_name": "Athens & Attica Region, Greece", "lat": 37.98, "lon": 23.72, "biome": "Mediterranean Pine & Scrub"},
    {"name": "Rhodes, Greece", "display_name": "Rhodes, South Aegean, Greece", "lat": 36.43, "lon": 28.22, "biome": "Aegean Mediterranean Scrub"},
    # ---- Pacific Islands ----
    {"name": "Maui, Hawaii", "display_name": "Lahaina / Maui, Hawaii, USA", "lat": 20.88, "lon": -156.68, "biome": "Tropical Dry Grassland / WUI"},
    # ---- Asia / Siberia ----
    {"name": "Yakutia, Siberia", "display_name": "Sakha / Yakutia Larch Taiga, Russia", "lat": 62.03, "lon": 129.73, "biome": "Subarctic Permafrost Taiga"},
    {"name": "Kalimantan, Borneo", "display_name": "Central Kalimantan Peatlands, Indonesia", "lat": -2.21, "lon": 113.92, "biome": "Tropical Peat Swamp Forest"},
    # ---- Africa ----
    {"name": "Okavango Delta, Botswana", "display_name": "Okavango Delta Basin, Botswana", "lat": -19.28, "lon": 22.90, "biome": "Wooded Savanna & Floodplain"},
    # ---- South America (cont.) ----
    {"name": "Valparaíso, Chile", "display_name": "Valparaíso Coastal Hills, Chile", "lat": -33.05, "lon": -71.61, "biome": "Matorral & Pine Plantations"},
    # ---- Europe (cont.) ----
    {"name": "Madrid, Spain", "display_name": "Madrid & Central Plateau, Spain", "lat": 40.41, "lon": -3.70, "biome": "Iberian Dehesa & Scrub"},
    {"name": "Provence, France", "display_name": "Provence-Alpes-Côte d'Azur, France", "lat": 43.52, "lon": 5.44, "biome": "Garrigue & Maritime Pine"},
    # ---- South Asia ----
    {"name": "New Delhi, India", "display_name": "New Delhi & Northern Plains, India", "lat": 28.61, "lon": 77.20, "biome": "Agricultural Stubble Plains"},
    # ---- East Asia ----
    {"name": "Tokyo, Japan", "display_name": "Tokyo & Kanto Plain, Japan", "lat": 35.68, "lon": 139.69, "biome": "Temperate Broadleaf Forest"},
    # ---- Western Europe ----
    {"name": "London, United Kingdom", "display_name": "London, United Kingdom", "lat": 51.50, "lon": -0.12, "biome": "Temperate Oceanic"},
    # ---- Eastern North America ----
    {"name": "New York, USA", "display_name": "New York, NY, USA", "lat": 40.71, "lon": -74.00, "biome": "Atlantic Coastal Plain"},
    # ---- US National Parks ----
    {"name": "Yellowstone, Wyoming", "display_name": "Yellowstone National Park, WY, USA", "lat": 44.42, "lon": -110.58, "biome": "High Plateau Lodgepole Pine"},
    {"name": "Yosemite, California", "display_name": "Yosemite National Park, CA, USA", "lat": 37.86, "lon": -119.53, "biome": "Sierra Mixed Conifer"},
    # ---- Southern Africa ----
    {"name": "Cape Town, South Africa", "display_name": "Cape Town & Table Mountain, South Africa", "lat": -33.92, "lon": 18.42, "biome": "Fynbos Fire-Prone Shrubland"},
    # ---- Iberian Peninsula ----
    {"name": "Lisbon, Portugal", "display_name": "Lisbon & Central Portugal", "lat": 38.72, "lon": -9.13, "biome": "Maritime Pine & Eucalyptus"},
    {"name": "Rome, Italy", "display_name": "Rome & Lazio Region, Italy", "lat": 41.90, "lon": 12.49, "biome": "Mediterranean Coastal Scrub"}
]

# ===== ROUTE: GEOCODE LOCATION (3-STAGE LOOKUP) =====

@app.route('/api/geocode', methods=['GET'])
def geocode_location():
    """Resolves a user's location query to geographic coordinates using a 3-stage strategy.

    The geocode endpoint implements a tiered lookup approach to balance speed,
    offline capability, and global coverage:

        Stage 1 — Coordinate Parsing: Detects if the query is a raw coordinate
            pair (e.g., "37.77, -122.41" or "37.77 -122.41"). If so, returns
            immediately with a synthetic "Coordinates" result. This supports
            analysts who already know their target lat/lon.

        Stage 2 — Curated Fast-Lookup Table: Searches the POPULAR_GEO_PLACES
            list for case-insensitive substring matches against name and
            display_name. This provides instant offline results for well-known
            fire-prone regions without any network call.

        Stage 3 — Live OSM Nominatim Geocoding: Queries the OpenStreetMap
            Nominatim API for global geocoding coverage. Deduplicates results
            that are very close (within 0.15° lat/lon) to entries already found
            in Stage 2. Has a 3.5-second timeout to avoid blocking the UI.
            If Nominatim fails or times out, the local results from Stage 2
            are still returned seamlessly.

    HTTP Method: GET
    Query Parameters:
        q: Search query string (minimum 2 characters)

    Response JSON:
        {
            "query": "...",
            "results": [
                { "name": "...", "display_name": "...", "lat": ..., "lon": ...,
                  "type": "...", "biome": "..." },
                ...
            ]
        }
    Results are capped at 8 entries maximum.
    """
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({"query": query, "results": []})
        
    results = []
    
    # ---- Stage 1: Direct Coordinate Parsing ----
    # Regex pattern matches valid lat/lon pairs separated by comma or whitespace.
    # Latitude range: -90 to 90; Longitude range: -180 to 180.
    coord_pattern = r'^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?)[,\s]+[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$'
    if re.match(coord_pattern, query):
        try:
            # Split on comma or whitespace to extract the two numeric components
            parts = re.split(r'[,\s]+', query)
            lat = float(parts[0])
            lon = float(parts[1])
            # Return a synthetic coordinate result immediately — no need for geocoding
            results.append({
                "name": f"Coordinates ({lat:.4f}°, {lon:.4f}°)",
                "display_name": f"Target Coordinates: {lat:.4f}°, {lon:.4f}°",
                "lat": lat,
                "lon": lon,
                "type": "coordinate_target",
                "biome": "Custom Coordinates Sector"
            })
            return jsonify({"query": query, "results": results})
        except Exception:
            pass

    # ---- Stage 2: Curated Local Fast-Lookup Table ----
    # Case-insensitive substring match against the hand-curated fire-prone places list.
    # This provides instant offline results for well-known regions.
    q_lower = query.lower()
    for p in POPULAR_GEO_PLACES:
        if q_lower in p['name'].lower() or q_lower in p['display_name'].lower():
            results.append(dict(p))
            
    # ---- Stage 3: Live OpenStreetMap Nominatim Geocoding ----
    # Query the external Nominatim API for global coverage. This handles locations
    # not in the curated table (e.g., small towns, specific addresses).
    try:
        url = "https://nominatim.openstreetmap.org/search"
        # User-Agent is required by Nominatim usage policy
        headers = {'User-Agent': 'SatFireTriageApp/1.0 (NASA-FIRMS-Active-Fire-Decision-Support)'}
        params = {'q': query, 'format': 'json', 'limit': 6, 'addressdetails': 1}
        resp = requests.get(url, params=params, headers=headers, timeout=3.5)
        if resp.status_code == 200:
            for item in resp.json():
                try:
                    lat = float(item['lat'])
                    lon = float(item['lon'])
                    # Deduplicate: skip results that are very close (<0.15°) to
                    # entries already found in the curated table (Stage 2)
                    is_dup = any(abs(lat - r['lat']) < 0.15 and abs(lon - r['lon']) < 0.15 for r in results)
                    if not is_dup:
                        disp_name = item.get('display_name', '')
                        name = item.get('name') or disp_name.split(',')[0]
                        results.append({
                            "name": name,
                            "display_name": disp_name,
                            "lat": round(lat, 4),
                            "lon": round(lon, 4),
                            "type": item.get('type', 'place'),
                            "biome": "Regional Fire Sector"
                        })
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        # Nominatim network error or timeout: local results will still be returned seamlessly!
        pass
        
    # Cap results at 8 to keep the dropdown manageable
    return jsonify({"query": query, "results": results[:8]})

# ===== ROUTE: SATELLITE OBSERVER PRESETS =====

@app.route('/api/observer/presets', methods=['GET'])
def get_observer_presets():
    """Returns the list of preset fire-prone regions and satellite platform specs.

    Used by the front-end to populate the "Observe Region" dropdown and display
    satellite capability cards.

    HTTP Method: GET
    Response JSON:
        {
            "presets": [ { id, name, country, lat, lon, biome, fire_regime, ... }, ... ],
            "satellites": [ { name, resolution, revisit }, ... ]
        }
    """
    return jsonify({
        "presets": SatelliteObserver.get_preset_regions(),
        "satellites": [
            {"name": "NOAA-20 / VIIRS", "resolution": "375 m", "revisit": "12 h"},
            {"name": "NOAA-21 / VIIRS", "resolution": "375 m", "revisit": "12 h"},
            {"name": "Suomi-NPP / VIIRS", "resolution": "375 m", "revisit": "12 h"},
            {"name": "Terra / Aqua MODIS", "resolution": "1 km", "revisit": "1-2 days"}
        ]
    })

# ===== ROUTE: OBSERVE CUSTOM LOCATION =====

@app.route('/api/observer/observe', methods=['POST'])
def observe_custom_location():
    """Simulates a satellite observation pass over a user-specified location.

    Generates realistic thermal anomaly detections (hotspots) in the vicinity
    of the target coordinates, runs them through the full triage pipeline, and
    returns the results along with simulated satellite pass metadata (swath
    polygon, ground track, orbit info, atmospheric conditions).

    HTTP Method: POST
    Request JSON Body:
        {
            "lat": <float>,    // target latitude (-85 to 85)
            "lon": <float>,    // target longitude (-180 to 180)
            "name": "...",     // optional place name
            "biome": "..."     // optional biome hint
        }

    Response JSON: Full observation result including target info, satellite_pass
        metadata, validation_summary, cluster_summaries, and ranked observations.
    Error: 400 if lat or lon are missing.
    """
    data = request.get_json() or {}
    lat = data.get('lat')
    lon = data.get('lon')
    name = data.get('name')
    biome = data.get('biome')
    
    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required"}), 400
        
    result = SatelliteObserver.observe_location(
        target_lat=float(lat),
        target_lon=float(lon),
        place_name=name,
        biome_hint=biome
    )
    
    # Also update state with the observed records so user can explore them in table & map
    STATE['current_dataset_name'] = f"Satellite Pass: {result['target']['name']}"
    STATE['ranked_records'] = result['observations']
    STATE['cluster_summaries'] = result['cluster_summaries']
    STATE['validation_summary'] = result['validation_summary']
    
    return jsonify(result)

# ===== ROUTE: OBSERVE RANDOM REGION =====

@app.route('/api/observer/random', methods=['POST'])
def observe_random():
    """Picks a random curated fire-prone region and simulates a satellite pass over it.

    Selects from the GLOBAL_REGIONS list with slight coordinate jitter, then
    delegates to observe_location(). Useful for demonstrating the system's
    capabilities on diverse global biomes.

    HTTP Method: POST
    Request: No body required.
    Response JSON: Same structure as /api/observer/observe.
    """
    result = SatelliteObserver.observe_random_region()
    STATE['current_dataset_name'] = f"Satellite Pass: {result['target']['name']}"
    STATE['ranked_records'] = result['observations']
    STATE['cluster_summaries'] = result['cluster_summaries']
    STATE['validation_summary'] = result['validation_summary']
    return jsonify(result)

# ===== ROUTE: EXPORT RANKED DATA AS CSV =====

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    """Exports the current ranked records as a downloadable CSV file.

    Generates a CSV with a fixed set of triage-relevant columns, suitable for
    import into GIS tools (QGIS, ArcGIS), spreadsheets, or downstream analysis
    pipelines. Records are in priority_score descending order (triage_rank 1 first).

    HTTP Method: GET
    Response: CSV file download (Content-Disposition: attachment) with MIME type text/csv.
    Error: 400 if no records are currently loaded.
    """
    records = STATE.get('ranked_records', [])
    if not records:
        return "No active records to export", 400
        
    output = io.StringIO()
    # Define the column order for the exported CSV — includes triage-specific
    # fields (rank, score, tier) alongside standard FIRMS columns
    fieldnames = [
        'triage_rank', 'priority_score', 'priority_tier', 'obs_id',
        'latitude', 'longitude', 'frp', 'brightness', 'bright_bg', 'temp_delta',
        'confidence_pct', 'confidence_level', 'cluster_id', 'cluster_size',
        'acq_date', 'acq_time', 'satellite', 'instrument', 'daynight', 'scan'
    ]
    # extrasaction='ignore' silently drops record keys not in fieldnames
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(records)
    
    # Build the HTTP response with CSV content and download headers
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=firms_triage_ranked_export.csv'
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    return response

# ===== ROUTE: EXPORT ANALYST BRIEFING REPORT (JSON) =====

@app.route('/api/export/report', methods=['GET'])
def export_report():
    """Exports a structured JSON analyst briefing report for the current dataset.

    Generates a decision-support report containing:
        - Executive summary: total detections, critical/high counts, cluster count,
          aggregate FRP.
        - Top 10 priority incidents with key scoring factors and analyst caveats.
        - Mandatory scientific limitations notice (satellite cannot confirm ground
          truth, cloud obscuration, sub-pixel blending, etc.).

    The report is explicitly marked as "DECISION SUPPORT / NOT GROUND TRUTH /
    NON-EMERGENCY" to prevent misuse as an emergency dispatch trigger.

    HTTP Method: GET
    Response: JSON file download (Content-Disposition: attachment).
    """
    records = STATE.get('ranked_records', [])
    clusters = STATE.get('cluster_summaries', [])
    ds_name = STATE.get('current_dataset_name', 'Unknown')
    
    # Filter for the two highest tiers to highlight in the executive summary
    critical_records = [r for r in records if r['priority_tier'] == 'CRITICAL']
    high_records = [r for r in records if r['priority_tier'] == 'HIGH']
    
    report = {
        "title": "Satellite-Derived Active Fire Triage Assessment Briefing",
        "system": "Antigravity Active Fire Triage Analyst Dashboard",
        # Explicit disclaimer classification to prevent misuse
        "classification": "DECISION SUPPORT / NOT GROUND TRUTH / NON-EMERGENCY",
        "generated_at": datetime.now().isoformat() + "Z",
        "dataset_source": ds_name,
        "executive_summary": {
            "total_detections": len(records),
            "critical_priority_count": len(critical_records),
            "high_priority_count": len(high_records),
            "fire_clusters_identified": len(clusters),
            "total_frp_mw": round(sum(r['frp'] for r in records), 1)
        },
        # Top 10 incidents by triage_rank, with factor decomposition for analyst review
        "top_priority_incidents": [
            {
                "triage_rank": r['triage_rank'],
                "obs_id": r['obs_id'],
                "coordinates": f"{r['latitude']}, {r['longitude']}",
                "priority_score": r['priority_score'],
                "tier": r['priority_tier'],
                "frp_mw": r['frp'],
                "brightness_k": r['brightness'],
                "confidence": r['confidence_level'],
                "cluster": r['cluster_id'],
                # Show the top 3 most impactful scoring factors
                "key_factors": [f['explanation'] for f in r['factors'][:3]],
                "analyst_caveats": r['caveats']
            }
            for r in records[:10]
        ],
        # Mandatory disclaimers that must accompany any operational use
        "mandatory_limitations_notice": [
            "Satellite thermal anomalies do not confirm active uncontrolled wildfires.",
            "Sub-pixel emitters, agricultural burns, and industrial flares cannot be definitively differentiated without optical/ground audit.",
            "Under-canopy and cloud-covered fires cannot be detected by thermal sensors.",
            "This briefing must not be used as an emergency response or evacuation dispatch trigger."
        ]
    }
    
    # Serialize with indentation for human readability
    response = make_response(json.dumps(report, indent=2))
    response.headers['Content-Disposition'] = 'attachment; filename=fire_triage_analyst_briefing.json'
    response.headers['Content-Type'] = 'application/json'
    return response

# ===== APPLICATION STARTUP =====

if __name__ == '__main__':
    # Initialize with default dataset on server startup.
    # Pre-loads sample_firms_viirs.csv so the dashboard has data immediately
    # available when the user first opens the page, without requiring a manual
    # "Load Dataset" action.
    default_records = load_csv_file('data/sample_firms_viirs.csv')
    if default_records:
        # Run the full pipeline to populate STATE with ranked records, clusters,
        # and validation summary before the server starts accepting requests
        r, c, v = process_pipeline(default_records, STATE['weights'])
        STATE['raw_records'] = default_records
        STATE['ranked_records'] = r
        STATE['cluster_summaries'] = c
        STATE['validation_summary'] = v
    print("Satellite Fire Triage Flask Server initialized on port 5000.")
    # Bind to 0.0.0.0 to allow access from other machines on the network;
    # debug=False for production-like stability (no auto-reload or debugger)
    app.run(host='0.0.0.0', port=5000, debug=False)
