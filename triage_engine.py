"""
Satellite-Derived Active Fire Triage Engine
Core modules:
- Field validation and ingestion for NASA FIRMS (VIIRS & MODIS)
- Geospatial Haversine & DBSCAN clustering
- Multi-factor explainable priority ranking algorithm
- Limitations and analyst caveats generator

Architecture Overview (3-Stage Pipeline):
    This module implements the core processing pipeline for satellite-derived
    active fire observations. The pipeline has three sequential stages:

    Stage 1 — Field Validation & Normalization (FieldValidator):
        Ingests raw CSV records (string-valued dictionaries from NASA FIRMS or
        compatible formats), validates coordinate ranges, normalizes brightness
        temperatures (Kelvin), Fire Radiative Power (MW), confidence levels
        (categorical → numeric), and extracts metadata. Invalid records are
        rejected with structured error messages; marginal values generate warnings.

    Stage 2 — Geospatial DBSCAN Clustering (GeospatialClusterer):
        Groups individual hotspot pixel detections into coherent fire complexes
        using a density-based spatial clustering algorithm (DBSCAN) with
        Haversine great-circle distance as the metric. Nearby pixels (within
        eps_km) that meet a minimum neighbor count (min_samples) are assigned
        to the same cluster. Isolated pixels remain unclustered. Each cluster
        is characterized by its centroid, bounding box, aggregate FRP, and a
        tier classification (CRITICAL COMPLEX / MAJOR FIRE FRONT / LOCALIZED).

    Stage 3 — Explainable Triage Ranking (ExplainableTriageRanker):
        Assigns each observation a composite priority score (0–100) based on
        7 physically meaningful, independently normalized factors:
            1. Fire Radiative Power (log-scaled, 30% weight)
            2. Sensor Algorithm Confidence (linear, 20%)
            3. Thermal Delta T4-T11 (linear, 15%)
            4. Spatial Cluster Multiplicity (sqrt-scaled, 15%)
            5. Nighttime Detection Bonus (binary, 10%)
            6. Nadir Scan Geometry Quality (inverse-linear, 5%)
            7. Temporal Multi-Pass Persistence (linear, 5%)
        Each factor produces a human-readable explanation. The final score
        determines the Priority Tier (CRITICAL ≥80, HIGH ≥65, ELEVATED ≥45,
        MONITOR/LOW <45) and analyst caveats are generated for low-confidence
        or ambiguous detections.
"""

import math
import re
from datetime import datetime

# ===== HAVERSINE DISTANCE =====
# Mean radius of Earth in kilometers (WGS-84 mean radius).
# Used for all great-circle distance calculations in clustering.
R_EARTH_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two points in km.

    Implements the Haversine formula, which computes the shortest distance
    between two points on a sphere given their latitudes and longitudes.
    This is the standard distance metric for geospatial fire clustering.

    The Haversine formula:
        a = sin²(Δφ/2) + cos(φ₁) · cos(φ₂) · sin²(Δλ/2)
        c = 2 · atan2(√a, √(1−a))
        d = R · c

    Where φ = latitude in radians, λ = longitude in radians, R = Earth radius.

    Note: This assumes a perfect sphere, which introduces ~0.3% error vs.
    the WGS-84 ellipsoid. Acceptable for fire clustering at km scales.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Great-circle distance in kilometers.
    """
    # Convert decimal degrees to radians for trigonometric functions
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    # Compute the differences in latitude and longitude in radians
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    # Haversine formula: 'a' is the square of half the chord length between the points
    # sin²(Δφ/2) handles the latitude separation; the second term handles longitude
    # separation weighted by the cosines of the latitudes (converging meridians)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    # 'c' is the angular distance in radians, computed via atan2 for numerical stability
    # (avoids issues with antipodal points that arcsin would have)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    # Multiply angular distance by Earth's radius to get distance in kilometers
    return R_EARTH_KM * c


# ===== FIELD VALIDATION =====

class FieldValidator:
    """Validates raw satellite active fire observation records.

    Ingests raw CSV dictionaries (as produced by csv.DictReader from NASA FIRMS
    downloads or compatible formats) and produces validated, type-converted,
    schema-normalized records. Handles both VIIRS and MODIS column naming
    conventions (e.g., bright_ti4 vs brightness, bright_ti5 vs bright_t31).

    Validation steps per record:
        1. Coordinate validation: lat ∈ [-90, 90], lon ∈ [-180, 180]
        2. Brightness temperature normalization (mid-IR channel, Kelvin)
        3. Background brightness temperature (long-wave IR, Kelvin)
        4. Fire Radiative Power (FRP) validation (megawatts, non-negative)
        5. Confidence level normalization (categorical H/N/L → numeric 0–100)
        6. Acquisition date/time extraction
        7. Instrument and satellite metadata
        8. Scan/track pixel geometry (km)
        9. Day/Night flag normalization
        10. Optional ground truth label and land cover type passthrough

    Records that fail coordinate validation are rejected (added to errors).
    Records with unusual but parseable values generate warnings but are kept.
    """
    
    @staticmethod
    def validate_and_normalize(raw_records):
        """Validates and normalizes a list of raw fire observation records.

        Args:
            raw_records: List of dictionaries with string values, as read from
                a NASA FIRMS CSV file. Expected columns include latitude,
                longitude, bright_ti4/brightness, frp, confidence, etc.

        Returns:
            Tuple of (valid_records, summary):
                - valid_records: List of validated dicts with typed numeric values,
                  normalized confidence, computed temp_delta, and standardized schema.
                - summary: Dict with total_raw, valid_count, warnings_count,
                  errors_count, and capped lists of warning/error details.
        """
        valid_records = []
        warnings = []
        errors = []
        
        for idx, row in enumerate(raw_records):
            # Use the existing obs_id if present; otherwise generate a sequential ID
            record_id = row.get('obs_id') or f"REC-{idx+1:04d}"
            try:
                # ---- Step 1: Coordinates Validation ----
                # Support both 'latitude'/'longitude' and short 'lat'/'lon' column names
                lat_str = row.get('latitude') or row.get('lat')
                lon_str = row.get('longitude') or row.get('lon')
                if lat_str is None or lon_str is None:
                    errors.append({"row": idx, "field": "coordinates", "error": "Missing latitude or longitude"})
                    continue
                lat = float(lat_str)
                lon = float(lon_str)
                # Reject coordinates outside valid geographic range
                if not (-90.0 <= lat <= 90.0):
                    errors.append({"row": idx, "field": "latitude", "error": f"Latitude {lat} out of range [-90, 90]"})
                    continue
                if not (-180.0 <= lon <= 180.0):
                    errors.append({"row": idx, "field": "longitude", "error": f"Longitude {lon} out of range [-180, 180]"})
                    continue
                
                # ---- Step 2: Brightness / Thermal Channel Validation ----
                # VIIRS uses 'bright_ti4' (mid-IR I4 band at ~3.75 µm)
                # MODIS uses 'brightness' (channel 21/22 at ~3.96 µm)
                # Some datasets may use 'bright_t4' as an alias
                b_raw = row.get('bright_ti4') or row.get('brightness') or row.get('bright_t4')
                # Default to 320 K if not provided (reasonable ambient + slight anomaly)
                bright = float(b_raw) if b_raw not in (None, '') else 320.0
                # Typical active fire brightness: 300–500 K; values outside 250–550 K
                # are flagged as unusual (but still kept)
                if bright < 250.0 or bright > 550.0:
                    warnings.append({"row": idx, "field": "brightness", "warning": f"Unusual brightness {bright} K (expected 280-510K)"})
                
                # ---- Step 3: Background Thermal Channel ----
                # VIIRS uses 'bright_ti5' (I5 band at ~11.45 µm, atmospheric window)
                # MODIS uses 'bright_t31' (channel 31 at ~11.03 µm)
                # This measures the ambient background temperature for computing
                # the thermal contrast (temp_delta = brightness - bright_bg)
                b_bg_raw = row.get('bright_ti5') or row.get('bright_t31')
                # Default 292 K is a reasonable terrestrial background temperature
                bright_bg = float(b_bg_raw) if b_bg_raw not in (None, '') else 292.0
                
                # ---- Step 4: Fire Radiative Power (FRP) Validation ----
                # FRP is measured in megawatts (MW) and represents the rate of
                # radiant heat energy released by the fire. It is the single most
                # important physical indicator of fire intensity.
                # Typical ranges: smoldering ~1-5 MW, surface fire ~10-50 MW,
                # crown fire ~50-500 MW, extreme events >500 MW
                frp_raw = row.get('frp')
                frp = float(frp_raw) if frp_raw not in (None, '') else 5.0
                # Negative FRP is physically impossible; clamp to zero
                if frp < 0.0:
                    frp = 0.0
                    warnings.append({"row": idx, "field": "frp", "warning": "Negative FRP clamped to 0.0"})
                
                # ---- Step 5: Confidence Level Normalization ----
                # VIIRS reports categorical confidence: 'h' (high), 'n' (nominal), 'l' (low)
                # MODIS reports numeric confidence: 0–100 integer
                # This step normalizes both formats to a consistent 0–100 numeric scale
                # with a corresponding categorical label
                conf_raw = str(row.get('confidence', 'nominal')).strip().lower()
                if conf_raw in ('h', 'high'):
                    # High confidence: strict spectral tests passed, very likely real fire
                    confidence_pct = 95.0
                    conf_level = 'High'
                elif conf_raw in ('n', 'nominal', 'med', 'medium'):
                    # Nominal confidence: standard detection threshold met
                    confidence_pct = 70.0
                    conf_level = 'Nominal'
                elif conf_raw in ('l', 'low'):
                    # Low confidence: marginal detection, higher false-alarm risk
                    confidence_pct = 30.0
                    conf_level = 'Low'
                else:
                    # Could be numeric MODIS confidence (0 - 100)
                    try:
                        c_val = float(conf_raw)
                        # Clamp to valid percentage range [0, 100]
                        confidence_pct = max(0.0, min(100.0, c_val))
                        # Map numeric value to categorical label for display
                        if confidence_pct >= 80:
                            conf_level = 'High'
                        elif confidence_pct >= 50:
                            conf_level = 'Nominal'
                        else:
                            conf_level = 'Low'
                    except ValueError:
                        # Unrecognized string: default to Nominal with a warning
                        confidence_pct = 65.0
                        conf_level = 'Nominal'
                        warnings.append({"row": idx, "field": "confidence", "warning": f"Unknown confidence '{conf_raw}', defaulted to Nominal"})
                
                # ---- Step 6: Acquisition Date & Time ----
                # Date in ISO format (YYYY-MM-DD), time as HHMM (zero-padded 4 digits)
                acq_date = str(row.get('acq_date', '2026-08-14')).strip()
                acq_time = str(row.get('acq_time', '1200')).strip().zfill(4)
                
                # ---- Step 7: Instrument & Satellite Metadata ----
                # Instrument: VIIRS or MODIS (uppercase normalized)
                # Satellite: NOAA-20, NOAA-21, Suomi-NPP, Aqua, Terra, etc.
                instrument = str(row.get('instrument', 'VIIRS')).strip().upper()
                satellite = str(row.get('satellite', 'NOAA-20')).strip()
                
                # ---- Step 8: Scan / Track Pixel Geometry ----
                # Scan and track represent the pixel footprint dimensions in km.
                # At nadir: VIIRS ≈ 0.375 km, MODIS ≈ 1.0 km.
                # At swath edge: pixels stretch up to ~2.0+ km due to viewing angle.
                # Smaller values = sharper spatial resolution = more precise geolocation.
                scan = float(row.get('scan', 0.5)) if row.get('scan') not in (None, '') else 0.5
                track = float(row.get('track', 0.5)) if row.get('track') not in (None, '') else 0.5
                
                # ---- Step 9: Day / Night Flag ----
                # 'D' = daytime pass (solar heating present, potential for solar glint)
                # 'N' = nighttime pass (higher thermal contrast reliability)
                daynight = str(row.get('daynight', 'D')).strip().upper()
                if daynight not in ('D', 'N'):
                    daynight = 'D'
                
                # ---- Step 10: Optional Evaluation & Land Cover Fields ----
                # Ground truth label (if available): used by EvaluationEngine
                # for computing confusion matrix / precision / recall
                ground_truth = row.get('ground_truth_label')
                land_cover = row.get('land_cover_type')
                # Persistence count: how many consecutive overpasses detected this hotspot
                persistence_count = int(row.get('persistence_count', 1)) if row.get('persistence_count') not in (None, '') else 1
                
                # Build the normalized record with standardized schema
                norm_record = {
                    'obs_id': record_id,
                    'latitude': round(lat, 5),
                    'longitude': round(lon, 5),
                    'brightness': round(bright, 1),
                    'bright_bg': round(bright_bg, 1),
                    # Thermal contrast: difference between fire channel and background
                    # channel. Higher delta = more distinct heat source above ambient.
                    'temp_delta': round(bright - bright_bg, 1),
                    'frp': round(frp, 1),
                    'confidence_pct': round(confidence_pct, 1),
                    'confidence_level': conf_level,
                    'scan': round(scan, 2),
                    'track': round(track, 2),
                    'acq_date': acq_date,
                    'acq_time': acq_time,
                    'satellite': satellite,
                    'instrument': instrument,
                    'daynight': daynight,
                    'persistence_count': persistence_count,
                    'ground_truth_label': ground_truth,
                    'land_cover_type': land_cover
                }
                valid_records.append(norm_record)
                
            except Exception as e:
                # Catch-all for unexpected parsing errors (malformed data, type conversion failures)
                errors.append({"row": idx, "field": "parsing", "error": str(e)})
        
        # Build validation summary; cap warning/error lists to avoid bloating API responses
        summary = {
            "total_raw": len(raw_records),
            "valid_count": len(valid_records),
            "warnings_count": len(warnings),
            "errors_count": len(errors),
            "warnings": warnings[:15], # cap for response size
            "errors": errors[:15]
        }
        return valid_records, summary


# ===== GEOSPATIAL CLUSTERING =====

class GeospatialClusterer:
    """Groups individual hotspot pixels into fire complexes using Haversine-based DBSCAN.

    Clusters represent coherent fire fronts / burn complexes. The algorithm uses
    the DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    approach where:
        - Distance metric: Haversine great-circle distance (km)
        - eps_km: Maximum neighborhood radius — two points within this distance
          are considered neighbors
        - min_samples: Minimum number of neighbors required to form a dense
          region (core point threshold)

    Points that don't belong to any cluster are labeled "Isolated" (noise in
    DBSCAN terminology). Isolated single-pixel detections have higher false-alarm
    rates than clustered fire fronts.

    After clustering, each cluster is characterized by:
        - Centroid (mean lat/lon of member points)
        - Bounding box [min_lat, min_lon, max_lat, max_lon]
        - Aggregate FRP statistics (total, mean, max)
        - Estimated perimeter radius (max distance from centroid to any member)
        - Cluster tier: CRITICAL COMPLEX (≥500 MW or ≥20 hotspots),
          MAJOR FIRE FRONT (≥150 MW or ≥8 hotspots), or LOCALIZED CLUSTER

    Args:
        eps_km: DBSCAN neighborhood radius in kilometers. Default 4.0 km.
        min_samples: Minimum number of neighbors to form a core point. Default 2.
    """
    
    def __init__(self, eps_km=4.0, min_samples=2):
        self.eps_km = eps_km
        self.min_samples = min_samples

    def cluster(self, records):
        """Runs DBSCAN clustering on the validated records.

        Algorithm Steps:
            1. Initialize all points as unvisited with label -1 (noise).
            2. For each unvisited point, find all neighbors within eps_km.
            3. If the point has ≥ min_samples neighbors, start a new cluster:
               expand the cluster by recursively adding density-reachable points.
            4. If the point has fewer neighbors, mark it as noise (Isolated).
            5. After clustering, compute per-cluster aggregate statistics.
            6. Annotate each record with cluster_id and cluster_size.

        Args:
            records: List of validated record dicts (must have 'latitude', 'longitude').

        Returns:
            Tuple of (records, cluster_summaries):
                - records: Input records mutated in-place with added cluster_id,
                  in_cluster, and cluster_size fields.
                - cluster_summaries: List of per-cluster statistics dicts.
        """
        n = len(records)
        if n == 0:
            return records, []
            
        # DBSCAN state: track which points have been visited and their cluster labels
        visited = [False] * n
        cluster_labels = [-1] * n # -1 = noise/isolated (no cluster assigned)
        current_cluster = 0
        
        # Precompute coordinates array to avoid repeated dict lookups in _region_query
        coords = [(r['latitude'], r['longitude']) for r in records]
        
        # ---- DBSCAN Core Loop ----
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            
            # Find all points within eps_km of point i (including i itself)
            neighbors = self._region_query(i, coords)
            
            if len(neighbors) < self.min_samples:
                # Not enough neighbors: this point is noise (isolated detection)
                cluster_labels[i] = -1 # Isolated
            else:
                # This is a core point: start a new cluster and expand it
                cluster_labels[i] = current_cluster
                # BFS queue for expanding the cluster to density-reachable points
                queue = list(neighbors)
                idx = 0
                while idx < len(queue):
                    p = queue[idx]
                    idx += 1
                    if not visited[p]:
                        visited[p] = True
                        # Check if this newly visited point is also a core point
                        p_neighbors = self._region_query(p, coords)
                        if len(p_neighbors) >= self.min_samples:
                            # Core point: add its unqueued neighbors to expand the cluster
                            for q in p_neighbors:
                                if q not in queue:
                                    queue.append(q)
                    # Assign this point to the current cluster (even if it was
                    # previously marked as noise — DBSCAN allows border points)
                    if cluster_labels[p] == -1:
                        cluster_labels[p] = current_cluster
                current_cluster += 1
        
        # ---- Aggregate Cluster Statistics ----
        # Group records by their cluster assignment
        cluster_groups = {}
        for idx, cid in enumerate(cluster_labels):
            # Assign human-readable cluster IDs: "CL-01", "CL-02", ... or "Isolated"
            records[idx]['cluster_id'] = f"CL-{cid+1:02d}" if cid >= 0 else "Isolated"
            records[idx]['in_cluster'] = (cid >= 0)
            
            if cid >= 0:
                if cid not in cluster_groups:
                    cluster_groups[cid] = []
                cluster_groups[cid].append(records[idx])
        
        # Compute summary statistics for each cluster
        cluster_summaries = []
        for cid, items in sorted(cluster_groups.items()):
            lats = [it['latitude'] for it in items]
            lons = [it['longitude'] for it in items]
            frps = [it['frp'] for it in items]
            
            # Aggregate FRP (Fire Radiative Power) statistics
            total_frp = round(sum(frps), 1)
            mean_frp = round(total_frp / len(items), 1)
            max_frp = max(frps)
            
            # Cluster centroid: arithmetic mean of member coordinates
            centroid_lat = round(sum(lats) / len(lats), 5)
            centroid_lon = round(sum(lons) / len(lons), 5)
            # Bounding box: [south, west, north, east]
            bbox = [min(lats), min(lons), max(lats), max(lons)]
            
            # Estimate cluster perimeter radius: maximum Haversine distance
            # from the centroid to any member point (minimum 0.5 km floor)
            max_dist = max([haversine_km(centroid_lat, centroid_lon, la, lo) for la, lo in zip(lats, lons)])
            radius_km = round(max(0.5, max_dist), 2)
            
            # ---- Cluster Tier Classification ----
            # Classify the cluster's severity based on aggregate energy and spatial extent
            if total_frp >= 500 or len(items) >= 20:
                # CRITICAL COMPLEX: Very high aggregate energy (≥500 MW) or very
                # large spatial extent (≥20 hotspots) — mega-fire complex
                tier = "CRITICAL COMPLEX"
            elif total_frp >= 150 or len(items) >= 8:
                # MAJOR FIRE FRONT: Significant energy (≥150 MW) or substantial
                # cluster (≥8 hotspots) — active multi-pixel fire front
                tier = "MAJOR FIRE FRONT"
            else:
                # LOCALIZED CLUSTER: Smaller fire group, possibly a single burn
                # unit or emerging ignition
                tier = "LOCALIZED CLUSTER"
                
            cluster_summaries.append({
                "cluster_code": f"CL-{cid+1:02d}",
                "count": len(items),
                "total_frp": total_frp,
                "mean_frp": mean_frp,
                "max_frp": max_frp,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "radius_km": radius_km,
                "bbox": bbox,
                "tier": tier
            })
            
        # Annotate each record with its cluster's total member count
        # (used by the ranker to score the "cluster_size" factor)
        cluster_size_map = {f"CL-{cid+1:02d}": len(items) for cid, items in cluster_groups.items()}
        for r in records:
            # Isolated points get cluster_size=1
            r['cluster_size'] = cluster_size_map.get(r['cluster_id'], 1)
            
        return records, cluster_summaries

    def _region_query(self, p_idx, coords):
        """Finds all points within eps_km of the point at index p_idx.

        Uses a two-stage approach for performance:
            1. Fast rectangular pre-filter: Rejects points whose latitude or
               longitude difference exceeds what could possibly be within eps_km.
               Uses approximate conversion factors:
                 - 1° latitude ≈ 111 km (constant)
                 - 1° longitude ≈ 85 km (conservative estimate at ~40° latitude;
                   actual value varies with cos(lat), but 85 km is a safe lower
                   bound for most fire-prone latitudes)
            2. Exact Haversine check: For points that pass the pre-filter,
               compute the precise great-circle distance.

        This optimization avoids expensive trigonometric Haversine calculations
        for obviously distant points, typically reducing computation by 90%+.

        Args:
            p_idx: Index of the query point in the coords array.
            coords: List of (latitude, longitude) tuples for all points.

        Returns:
            List of indices of all points within eps_km of point p_idx
            (including p_idx itself).
        """
        lat1, lon1 = coords[p_idx]
        neighbors = []
        for j, (lat2, lon2) in enumerate(coords):
            # Fast box filter check first (~1 deg latitude ≈ 111 km)
            # If the latitude difference alone exceeds eps_km, skip immediately
            if abs(lat1 - lat2) * 111.0 > self.eps_km:
                continue
            # Fast longitude filter (~1 deg longitude ≈ 85 km at mid-latitudes)
            # Uses a conservative 85 km factor that works for latitudes up to ~40°
            if abs(lon1 - lon2) * 85.0 > self.eps_km:
                continue
            # Exact Haversine distance check for candidates that pass the pre-filter
            if haversine_km(lat1, lon1, lat2, lon2) <= self.eps_km:
                neighbors.append(j)
        return neighbors


# ===== EXPLAINABLE RANKING =====

class ExplainableTriageRanker:
    """Ranks satellite thermal observations using transparent, explainable physical factors.

    Produces score (0-100), Priority Tier, and Factor Impact Decomposition.

    This ranker is designed for transparency and auditability: every component
    of the final score is individually normalized, weighted, and accompanied
    by a human-readable explanation of what the value means physically and
    why it matters for fire triage.

    Scoring Architecture:
        Each of the 7 factors is independently normalized to [0, 1], then
        multiplied by its weight. The weights sum to 100, so the raw total
        naturally produces a 0–100 score. The normalization functions are
        chosen to match the physical distribution of each variable:

        Factor                  | Normalization          | Weight | Physical Meaning
        ========================|========================|========|==================
        FRP (MW)                | log1p(frp)/log1p(350)  | 30     | Rate of heat release; best single indicator of fire intensity
        Confidence (%)          | linear / 100           | 20     | Sensor algorithm certainty; filters false alarms
        Thermal Delta (K)       | linear / 50            | 15     | Brightness excess over background; heat source distinctness
        Cluster Size (#)        | sqrt(n)/sqrt(16)       | 15     | Spatial coherence; clustered fires = higher credibility
        Nighttime Bonus         | binary (1.0/0.35)      | 10     | No solar glint at night; cleaner thermal signal
        Nadir Quality (km)      | inverse-linear         |  5     | Sharper pixel = more precise geolocation
        Persistence (passes)    | linear / 3             |  5     | Recurrence across overpasses confirms real fire

    Priority Tiers:
        - CRITICAL (≥80): High-energy, clustered fire event with robust confidence.
          Urgent analyst inspection recommended.
        - HIGH (≥65): Significant thermal anomaly with elevated FRP or cluster presence.
        - ELEVATED (≥45): Moderate thermal signature. May represent controlled burn.
        - MONITOR / LOW (<45): Low energy or isolated detection. Routine monitoring.
    """
    
    # Configurable weights (sum to 100)
    # Each weight determines the maximum contribution of that factor to the
    # final priority score
    DEFAULT_WEIGHTS = {
        'frp': 30.0,          # Fire Radiative Power (MW) — dominant intensity signal
        'confidence': 20.0,   # Sensor algorithm confidence — false-alarm filter
        'temp_delta': 15.0,   # Thermal contrast vs background (Kelvin) — heat source distinctness
        'cluster_size': 15.0, # Spatial multiplicity / spreading front — coherence indicator
        'nighttime': 10.0,    # Solar glint absence & contrast bonus — signal cleanliness
        'nadir_quality': 5.0, # Optical footprint quality (minimal distortion) — geolocation precision
        'persistence': 5.0    # Recurrence across multiple overpasses — temporal confirmation
    }
    
    def __init__(self, weights=None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def rank(self, records):
        """Scores all records and returns them sorted by priority (descending).

        Args:
            records: List of validated, clustered record dicts.

        Returns:
            List of record dicts enriched with priority_score, priority_tier,
            triage_rank (1-indexed), factors list, and caveats list. Sorted
            by priority_score descending.
        """
        ranked = []
        for r in records:
            scored_record = self._score_observation(r)
            ranked.append(scored_record)
            
        # Sort descending by priority score (highest priority first)
        ranked.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Assign triage rank position (1 = highest priority, N = lowest)
        for i, r in enumerate(ranked):
            r['triage_rank'] = i + 1
            
        return ranked

    def _score_observation(self, r):
        """Computes the composite priority score for a single observation.

        Evaluates 7 independent factors, each normalized to [0, 1] then
        multiplied by its weight. Generates human-readable explanations
        and analyst caveats for each factor.

        Args:
            r: A single validated, clustered record dict.

        Returns:
            A copy of the record dict enriched with priority_score, priority_tier,
            priority_badge, tier_description, factors (list of factor dicts),
            and caveats (list of warning strings).
        """
        w = self.weights
        factors = []
        caveats = []
        
        # ---- Factor 1: Fire Radiative Power (FRP) ----
        # Normalization: Logarithmic scaling via log1p to compress the wide
        # dynamic range of FRP values (0.1 MW smoldering → 500+ MW crown fire).
        # log1p(x) = ln(1+x), which avoids log(0) and gives:
        #   5 MW → ~0.30,  50 MW → ~0.65,  250 MW → ~0.95,  500+ MW → 1.0
        # Divided by log1p(350) as the reference maximum (~5.86)
        frp = r.get('frp', 0.0)
        norm_frp = min(1.0, math.log1p(frp) / math.log1p(350.0))
        pts_frp = round(norm_frp * w['frp'], 1)
        
        # Generate human-readable description based on FRP magnitude
        if frp >= 200.0:
            frp_desc = f"Extreme radiative energy release ({frp:.1f} MW) indicates intense crown fire or severe biomass burning."
        elif frp >= 50.0:
            frp_desc = f"Significant fire radiative power ({frp:.1f} MW) consistent with active flaming front."
        elif frp >= 15.0:
            frp_desc = f"Moderate thermal intensity ({frp:.1f} MW); surface fire or localized combustion."
        else:
            frp_desc = f"Low thermal energy ({frp:.1f} MW); smoldering, small burn, or sub-pixel anomaly."
            caveats.append("Low FRP: Anomaly may be a small agricultural burn, hot ground, or early nascent ignition.")
            
        factors.append({
            "factor": "Fire Radiative Power (FRP)",
            "score": pts_frp,
            "max_score": w['frp'],
            "value": f"{frp:.1f} MW",
            # Impact classification for visual indicators in the UI
            "impact": "HIGH" if pts_frp > 18 else "MED" if pts_frp > 8 else "LOW",
            "explanation": frp_desc
        })
        
        # ---- Factor 2: Satellite Algorithm Confidence ----
        # Normalization: Simple linear scaling from percentage to [0, 1].
        # confidence_pct is already on a 0–100 scale from FieldValidator.
        conf_pct = r.get('confidence_pct', 70.0)
        norm_conf = conf_pct / 100.0
        pts_conf = round(norm_conf * w['confidence'], 1)
        
        if conf_pct >= 85:
            conf_desc = f"High satellite detection algorithm confidence ({conf_pct:.0f}%); strict spectral tests satisfied."
        elif conf_pct >= 60:
            conf_desc = f"Nominal sensor confidence ({conf_pct:.0f}%); standard threshold met with low false-alarm likelihood."
        else:
            conf_desc = f"Low sensor confidence ({conf_pct:.0f}%); marginal thermal detection threshold."
            caveats.append("Low sensor confidence: Higher likelihood of false trigger (e.g. cloud edge, warm rooftop).")
            
        factors.append({
            "factor": "Satellite Algorithm Confidence",
            "score": pts_conf,
            "max_score": w['confidence'],
            "value": f"{conf_pct:.0f}% ({r.get('confidence_level', 'Nominal')})",
            "impact": "HIGH" if pts_conf > 14 else "MED" if pts_conf > 8 else "LOW",
            "explanation": conf_desc
        })
        
        # ---- Factor 3: Thermal Delta (Brightness vs Background) ----
        # The temperature difference between the fire-sensitive mid-IR channel
        # (T4 at ~3.9 µm) and the background longwave IR channel (T11 at ~11 µm).
        # A high delta indicates a distinct, localized heat source above ambient.
        # Normalization: Linear scaling, capped at 50 K.
        #   0 K → 0.0,  15 K → 0.30,  40 K → 0.80,  50+ K → 1.0
        temp_delta = r.get('temp_delta', 25.0)
        norm_delta = max(0.0, min(1.0, temp_delta / 50.0))
        pts_delta = round(norm_delta * w['temp_delta'], 1)
        
        if temp_delta >= 40.0:
            delta_desc = f"Sharp thermal spike (+{temp_delta:.1f} K above background); strong localized heat source."
        elif temp_delta >= 15.0:
            delta_desc = f"Moderate thermal contrast (+{temp_delta:.1f} K above ambient background)."
        else:
            delta_desc = f"Low thermal contrast (+{temp_delta:.1f} K above background); diffuse or ambient heating."
            caveats.append("Low thermal contrast: Heat difference from surrounding terrain is slight.")
            
        factors.append({
            "factor": "Thermal Delta (T4 - T11)",
            "score": pts_delta,
            "max_score": w['temp_delta'],
            "value": f"+{temp_delta:.1f} K",
            "impact": "HIGH" if pts_delta > 10 else "MED" if pts_delta > 5 else "LOW",
            "explanation": delta_desc
        })
        
        # ---- Factor 4: Spatial Cluster Multiplicity ----
        # How many hotspot pixels are in this record's cluster. More pixels in
        # a coherent spatial group = higher confidence this is a real fire front
        # rather than a single-pixel false alarm.
        # Normalization: Square-root scaling (diminishing returns for very large clusters).
        #   1 pixel → 0.25,  4 pixels → 0.50,  9 pixels → 0.75,  16+ pixels → 1.0
        cluster_size = r.get('cluster_size', 1)
        norm_clust = min(1.0, math.sqrt(cluster_size) / math.sqrt(16.0))
        pts_clust = round(norm_clust * w['cluster_size'], 1)
        
        if cluster_size >= 12:
            clust_desc = f"Extensive fire front: record is part of a dense cluster of {cluster_size} contiguous hotspots within 4 km."
        elif cluster_size >= 4:
            clust_desc = f"Coherent cluster: grouped with {cluster_size} adjacent hotspot detections."
        elif cluster_size >= 2:
            clust_desc = f"Minor pairing: grouped with {cluster_size} neighboring detections."
        else:
            clust_desc = "Isolated single-pixel detection without neighboring thermal anomalies."
            caveats.append("Isolated pixel: Single-pixel anomalies have significantly higher false-alarm rates than clustered fronts.")
            
        factors.append({
            "factor": "Spatial Cluster Multiplicity",
            "score": pts_clust,
            "max_score": w['cluster_size'],
            "value": f"{cluster_size} detections ({r.get('cluster_id', 'Isolated')})",
            "impact": "HIGH" if pts_clust > 10 else "MED" if pts_clust > 4 else "LOW",
            "explanation": clust_desc
        })
        
        # ---- Factor 5: Nighttime Detection Quality Bonus ----
        # Nighttime observations have fundamentally higher thermal contrast
        # reliability because:
        #   - No solar heating of the ground surface (lower background T)
        #   - No solar glint reflections off metal roofs, water, or solar panels
        #   - Lower atmospheric scintillation noise
        # Scoring: Binary — full weight (1.0) for nighttime, reduced (0.35) for daytime
        daynight = r.get('daynight', 'D')
        is_night = (daynight == 'N')
        norm_night = 1.0 if is_night else 0.35
        pts_night = round(norm_night * w['nighttime'], 1)
        
        if is_night:
            night_desc = "Nighttime observation: Zero solar glint risk, lower background noise, high thermal contrast reliability."
        else:
            night_desc = "Daytime observation: Ambient solar heating present; potential for solar glint on reflective surfaces."
            caveats.append("Daytime pass: Solar reflections off metal roofs, solar farms, or water can occasionally cause false alarms.")
            
        factors.append({
            "factor": "Day / Night Thermal Reliability",
            "score": pts_night,
            "max_score": w['nighttime'],
            "value": "Nighttime (N)" if is_night else "Daytime (D)",
            "impact": "HIGH" if is_night else "MED",
            "explanation": night_desc
        })
        
        # ---- Factor 6: Nadir Quality (Scan Angle / Pixel Size) ----
        # The scan value represents the along-scan pixel footprint size in km.
        # Near-nadir (directly beneath the satellite): VIIRS ≈ 0.375 km, MODIS ≈ 1.0 km
        # At swath edge: pixels can stretch to >2.0 km (the "bow-tie" effect)
        # Smaller pixels = sharper spatial resolution = more precise fire geolocation
        # Normalization: Inverse-linear — smaller scan values score higher.
        #   0.35 km → 1.0,  1.35 km → 0.5,  2.35+ km → 0.2 (floor)
        scan = r.get('scan', 0.5)
        norm_nadir = max(0.2, min(1.0, 1.0 - (scan - 0.35) / 2.0))
        pts_nadir = round(norm_nadir * w['nadir_quality'], 1)
        
        if scan <= 0.55:
            nadir_desc = f"Near-nadir observation (pixel size {scan:.2f} km): Sharp spatial resolution with minimal geometric distortion."
        else:
            nadir_desc = f"Off-nadir / edge scan (pixel size {scan:.2f} km): Pixel elongation may blend multiple thermal sources."
            caveats.append("Off-nadir distortion: Pixel footprint is enlarged, reducing spatial precision.")
            
        factors.append({
            "factor": "Sensor Scan Geometry (Nadir Quality)",
            "score": pts_nadir,
            "max_score": w['nadir_quality'],
            "value": f"{scan:.2f} km",
            "impact": "HIGH" if pts_nadir > 3.5 else "LOW",
            "explanation": nadir_desc
        })
        
        # ---- Factor 7: Temporal Persistence / Repeat Detections ----
        # How many consecutive satellite overpasses have detected a hotspot at
        # or very near this location. Multiple detections across different passes
        # strongly confirms a real, sustained fire (transient false alarms don't persist).
        # Normalization: Linear, capped at 3 passes (1 pass → 0.33, 2 → 0.67, 3+ → 1.0)
        persist = r.get('persistence_count', 1)
        norm_persist = min(1.0, persist / 3.0)
        pts_persist = round(norm_persist * w['persistence'], 1)
        
        if persist >= 3:
            persist_desc = f"High temporal persistence: Hotspot detected across {persist} consecutive satellite overpasses."
        elif persist == 2:
            persist_desc = "Repeat observation: Detected in 2 consecutive overpasses."
        else:
            persist_desc = "First-pass observation; temporal persistence not yet established."
            
        factors.append({
            "factor": "Temporal Multi-Pass Persistence",
            "score": pts_persist,
            "max_score": w['persistence'],
            "value": f"{persist} passes",
            "impact": "HIGH" if pts_persist > 3 else "LOW",
            "explanation": persist_desc
        })
        
        # ---- Compute Total Priority Score ----
        # Sum all weighted factor scores. The theoretical maximum is 100 (all
        # factors at maximum), the theoretical minimum is ~0.
        raw_total = pts_frp + pts_conf + pts_delta + pts_clust + pts_night + pts_nadir + pts_persist
        final_score = round(min(100.0, max(0.0, raw_total)), 1)
        
        # ---- Assign Priority Tier ----
        # Tier thresholds are calibrated so that:
        #   CRITICAL (≥80): Requires high scores across most factors — genuinely
        #       alarming fire events that warrant urgent analyst inspection.
        #   HIGH (≥65): Strong signal in several factors but may lack one
        #       confirmatory dimension (e.g., isolated but very high FRP).
        #   ELEVATED (≥45): Moderate signal; could be a controlled burn, growing
        #       front, or industrial source — analyst should review.
        #   MONITOR / LOW (<45): Weak or ambiguous signal; routine monitoring
        #       queue, likely false alarm or minor thermal anomaly.
        if final_score >= 80.0:
            tier = "CRITICAL"
            tier_badge = "badge-critical"
            tier_desc = "High-energy, clustered fire event with robust sensor confidence. Urgent analyst inspection recommended."
        elif final_score >= 65.0:
            tier = "HIGH"
            tier_badge = "badge-high"
            tier_desc = "Significant thermal anomaly with elevated FRP or cluster presence."
        elif final_score >= 45.0:
            tier = "ELEVATED"
            tier_badge = "badge-elevated"
            tier_desc = "Moderate thermal signature. May represent controlled burn or growing front."
        else:
            tier = "MONITOR / LOW"
            tier_badge = "badge-low"
            tier_desc = "Low energy or isolated detection. Low triage priority; routine monitoring."
            
        # Add system disclaimer caveat — always present on every record
        caveats.append("Operational Limitation: Satellite detections indicate thermal anomalies, not ground-verified fires. Not for emergency response.")

        # Build the enriched record (copy to avoid mutating the input)
        r_copy = dict(r)
        r_copy.update({
            'priority_score': final_score,
            'priority_tier': tier,
            'priority_badge': tier_badge,
            'tier_description': tier_desc,
            'factors': factors,
            'caveats': caveats
        })
        return r_copy
