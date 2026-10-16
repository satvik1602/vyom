"""
Satellite Fire Observer Module
Enables observing any user-selected coordinates, named global regions,
or a random hotspot on Earth. Simulates satellite orbital passes (VIIRS/MODIS),
swath geometry, and thermal anomaly retrieval with biome-aware physics.

Architecture Overview:
    This module simulates the satellite observation workflow for active fire
    detection. While the triage_engine processes pre-existing CSV datasets,
    this module generates new thermal anomaly data on-the-fly for any point
    on Earth, mimicking what a real satellite overpass would produce.

    The simulation includes:
        1. **Satellite Platform Selection**: Randomly picks from real polar-orbiting
           platforms (NOAA-20, NOAA-21, Suomi-NPP, Aqua) with accurate specs.
        2. **Orbital Geometry**: Computes ground track centerline, swath corridor
           polygon, and scan angle effects based on sun-synchronous orbit parameters.
        3. **Day/Night Determination**: Uses the target longitude and current UTC
           hour to approximate local solar time and determine observation conditions.
        4. **Hotspot Generation**: Creates 18–45 thermal anomaly detections in a
           realistic spatial distribution: 70% in a primary fire complex cluster,
           30% in secondary flank/outlier positions.
        5. **Radiometric Simulation**: Assigns physically plausible brightness
           temperatures (Kelvin), FRP (MW), confidence levels, and scan resolution
           values that correlate appropriately (high FRP → high brightness → high
           confidence).
        6. **Full Pipeline Execution**: Passes generated detections through the
           same FieldValidator → GeospatialClusterer → ExplainableTriageRanker
           pipeline used for real datasets.

    The result is indistinguishable in format from a real FIRMS dataset analysis,
    allowing the analyst dashboard to seamlessly display simulated observations.

Dependencies:
    - triage_engine: FieldValidator, GeospatialClusterer, ExplainableTriageRanker
    - math, random, datetime (standard library)
"""

import math
import random
from datetime import datetime, timezone

from triage_engine import FieldValidator, GeospatialClusterer, ExplainableTriageRanker

# ===== GLOBAL FIRE-PRONE REGIONS =====
# Curated list of globally significant fire-prone biomes and regions.
# Each entry defines:
#   - id: Unique machine-readable identifier for the region
#   - name / country: Human-readable location label
#   - lat / lon: Representative center coordinates (decimal degrees)
#   - biome: Dominant vegetation / fire ecology classification
#   - fire_regime: Description of typical fire behavior in this biome
#   - typical_frp: Tuple of (min, max) FRP in MW that fires in this region
#     typically produce. Low end = surface/grass fires, high end = crown fires
#   - cluster_spread: Standard deviation in degrees for hotspot spatial
#     distribution around the fire center (~0.12° ≈ 13 km, ~0.35° ≈ 39 km)
GLOBAL_REGIONS = [
    {
        "id": "california_sierra",
        "name": "Sierra Nevada & Foothills",
        "country": "United States",
        "lat": 39.75, "lon": -121.40,
        "biome": "Temperate Coniferous & Chaparral",
        # High-intensity canopy fires with long flame lengths in mixed conifer
        # and dense chaparral brush; exemplified by Dixie Fire (2021)
        "fire_regime": "High-intensity canopy and brush wildfire complexes",
        # FRP range: 15 MW (surface fire in oak woodland) to 480 MW (extreme crown fire)
        "typical_frp": (15.0, 480.0),
        "cluster_spread": 0.18
    },
    {
        "id": "canada_jasper",
        "name": "Jasper & Athabasca Boreal",
        "country": "Canada",
        "lat": 52.87, "lon": -118.08,
        "biome": "Boreal Spruce & Pine Forest",
        # Crown fires in dense boreal spruce with extreme biomass consumption;
        # generates heavy smoke columns visible from space
        "fire_regime": "Crown fire with severe biomass consumption and heavy smoke",
        # FRP range: 25 MW (ground fire in muskeg) to 650 MW (boreal crown fire run)
        "typical_frp": (25.0, 650.0),
        "cluster_spread": 0.22
    },
    {
        "id": "brazil_pantanal",
        "name": "Pantanal Wetland Basin",
        "country": "Brazil",
        "lat": -16.85, "lon": -56.80,
        "biome": "Tropical Wetland & Seasonally Flooded Grassland",
        # Fast-moving grass fires during dry season; peat smoldering in drained areas
        "fire_regime": "Fast-moving grass fires and peat smoldering",
        # FRP range: 10 MW (grass fire) to 320 MW (peat/grassland complex)
        "typical_frp": (10.0, 320.0),
        "cluster_spread": 0.25
    },
    {
        "id": "amazon_rondonia",
        "name": "Rondônia Arc of Deforestation",
        "country": "Brazil",
        "lat": -9.80, "lon": -63.20,
        "biome": "Amazon Dense Humid Rainforest Margin",
        # Slash-and-burn deforestation fires at the forest-agriculture frontier
        "fire_regime": "Deforestation slash-and-burn transitioning to forest edge",
        # FRP range: 20 MW (small clearing burn) to 420 MW (large deforestation front)
        "typical_frp": (20.0, 420.0),
        "cluster_spread": 0.30
    },
    {
        "id": "australia_blue_mtns",
        "name": "Blue Mountains & Hawkesbury",
        "country": "Australia",
        "lat": -33.65, "lon": 150.35,
        "biome": "Dry Sclerophyll Eucalyptus Forest",
        # Extreme bushfires generating pyrocumulonimbus (pyroCb) thunderstorms;
        # eucalyptus oil volatilizes and explodes in crown fires
        "fire_regime": "Extreme pyrocumulonimbus-generating bushfires",
        # FRP range: 30 MW (moderate scrub fire) to 750 MW (Black Summer-level event)
        "typical_frp": (30.0, 750.0),
        "cluster_spread": 0.28
    },
    {
        "id": "greece_rhodes",
        "name": "Rhodes & Aegean Archipelago",
        "country": "Greece",
        "lat": 36.20, "lon": 27.95,
        "biome": "Mediterranean Pine Forest & Maquis Shrubland",
        # Wind-driven fires accelerated by Meltemi winds in WUI (wildland-urban
        # interface) zones; compact but intense fire runs
        "fire_regime": "Wind-driven (Meltemi) wildland-urban interface fires",
        # FRP range: 15 MW (scrub fire) to 380 MW (wind-driven pine forest fire)
        "typical_frp": (15.0, 380.0),
        # Smaller spread due to island geography constraining fire extent
        "cluster_spread": 0.15
    },
    {
        "id": "siberia_yakutia",
        "name": "Siberian Larch Taiga, Sakha",
        "country": "Russia",
        "lat": 62.40, "lon": 129.80,
        "biome": "Subarctic Continuous Permafrost Taiga",
        # Vast, remote boreal mega-complexes burning millions of hectares;
        # often left to burn due to remoteness
        "fire_regime": "Expansive remote boreal fire mega-complexes",
        # FRP range: 20 MW (ground fire) to 520 MW (mega-complex with multiple fronts)
        "typical_frp": (20.0, 520.0),
        # Large spread — fires in Siberia can extend over enormous areas
        "cluster_spread": 0.35
    },
    {
        "id": "indonesia_borneo",
        "name": "Central Kalimantan Peat Swamps",
        "country": "Indonesia",
        "lat": -2.25, "lon": 113.85,
        "biome": "Tropical Peat Swamp Forest",
        # Subsurface peat combustion: fires burn deep into drained peat layers,
        # generating immense transboundary haze (e.g., 2015 haze crisis)
        "fire_regime": "Subsurface peat combustion with immense haze generation",
        # FRP range: 8 MW (smoldering peat) to 210 MW (peat + surface vegetation fire)
        # Lower max FRP because peat fires are smoldering, not flaming
        "typical_frp": (8.0, 210.0),
        "cluster_spread": 0.20
    },
    {
        "id": "botswana_okavango",
        "name": "Okavango Delta Savanna",
        "country": "Botswana",
        "lat": -19.10, "lon": 23.10,
        "biome": "African Wooded Savanna & Floodplain",
        # Seasonal dry-season fires: long fire lines sweeping through grasslands
        "fire_regime": "Seasonal dry-season grassland fire lines",
        # FRP range: 5 MW (grass fire) to 180 MW (wooded savanna fire front)
        "typical_frp": (5.0, 180.0),
        "cluster_spread": 0.25
    },
    {
        "id": "chile_valparaiso",
        "name": "Valparaíso Coastal Cordillera",
        "country": "Chile",
        "lat": -33.05, "lon": -71.55,
        "biome": "Matorral & Exotic Pine/Eucalyptus Plantations",
        # Fires driven by steep canyon topography and coastal winds;
        # exotic eucalyptus plantations burn with extreme intensity
        "fire_regime": "Steep topography canyon wind-driven wildfires",
        # FRP range: 15 MW to 390 MW (eucalyptus plantation fire)
        "typical_frp": (15.0, 390.0),
        # Compact cluster spread — constrained by narrow coastal canyons
        "cluster_spread": 0.12
    },
    {
        "id": "portugal_centro",
        "name": "Serra da Estrela & Pedrógão",
        "country": "Portugal",
        "lat": 40.25, "lon": -7.75,
        "biome": "Iberian Maritime Pine & Oak Forest",
        # Severe convective plume fires with extreme rates of spread;
        # exemplified by the deadly 2017 Pedrógão Grande fire
        "fire_regime": "Severe convective plume fires",
        # FRP range: 18 MW to 460 MW (convective plume-driven fire)
        "typical_frp": (18.0, 460.0),
        "cluster_spread": 0.16
    },
    {
        "id": "argentina_corrientes",
        "name": "Iberá Wetlands & Grasslands",
        "country": "Argentina",
        "lat": -28.20, "lon": -57.40,
        "biome": "Subtropical Wet Savanna & Estuary",
        # Drought-amplified extensive burns in normally wet grassland/wetland terrain
        "fire_regime": "Drought-amplified extensive wetland burns",
        # FRP range: 12 MW to 290 MW
        "typical_frp": (12.0, 290.0),
        "cluster_spread": 0.22
    }
]

# ===== SATELLITE PLATFORMS =====
# Specifications of the real polar-orbiting satellite platforms that carry
# active fire detection instruments. These are used to generate realistic
# metadata for simulated observations.
#
# Key parameters:
#   - resolution: Ground pixel size at nadir (375 m for VIIRS I-bands, 1 km for MODIS)
#   - orbit: Sun-synchronous Low Earth Orbit (LEO) — passes over any given location
#     at roughly the same local solar time each day
#   - swath_width_km: Total width of the imaging swath — determines how much
#     ground area is covered in a single pass
#   - revisit_time: How often the satellite re-images the same ground location
SATELLITE_PLATFORMS = [
    {
        "name": "NOAA-20",
        "instrument": "VIIRS",
        "agency": "NASA / NOAA",
        "resolution": "375 m (I-Bands)",
        "orbit": "Sun-synchronous LEO (824 km altitude)",
        "swath_width_km": 3040,
        "revisit_time": "~12 hours (day/night)"
    },
    {
        "name": "NOAA-21",
        "instrument": "VIIRS",
        "agency": "NASA / NOAA",
        "resolution": "375 m (I-Bands)",
        "orbit": "Sun-synchronous LEO (824 km altitude)",
        "swath_width_km": 3040,
        "revisit_time": "~12 hours (day/night)"
    },
    {
        "name": "Suomi-NPP",
        "instrument": "VIIRS",
        "agency": "NASA / NOAA",
        "resolution": "375 m (I-Bands)",
        "orbit": "Sun-synchronous LEO (824 km altitude)",
        "swath_width_km": 3040,
        "revisit_time": "~12 hours (day/night)"
    },
    {
        "name": "Aqua",
        "instrument": "MODIS",
        "agency": "NASA EOS",
        "resolution": "1 km (thermal channels)",
        "orbit": "Sun-synchronous LEO (705 km altitude)",
        "swath_width_km": 2330,
        "revisit_time": "1-2 days"
    }
]

# ===== SATELLITE OBSERVER CLASS =====

class SatelliteObserver:
    """Simulates or retrieves satellite thermal observations over any arbitrary point on Earth.

    This class provides static methods to generate realistic active fire
    observation data for any geographic coordinate, complete with:
        - Simulated satellite pass metadata (platform, orbit number, swath geometry)
        - Physically plausible thermal anomaly detections (brightness, FRP, confidence)
        - Full triage pipeline processing (validation → clustering → ranking)
        - Atmospheric and environmental conditions (cloud cover, solar zenith)

    The observer does not connect to any real satellite data source — it generates
    synthetic data that is structurally identical to real NASA FIRMS output,
    enabling the dashboard to demonstrate its full analysis capabilities for
    any location worldwide.
    """
    
    @staticmethod
    def get_preset_regions():
        """Returns the list of curated global fire-prone regions.

        Returns:
            List of region dicts from GLOBAL_REGIONS.
        """
        return GLOBAL_REGIONS

    @staticmethod
    def observe_location(target_lat, target_lon, place_name=None, biome_hint=None):
        """Executes a simulated satellite observation pass over target_lat, target_lon.

        Generates swath geometry, ground track, active fire anomalies, and full triage pipeline.

        Simulation Steps:
            1. Coordinate clamping to ±85° latitude (polar regions are outside
               useful satellite coverage for fire detection).
            2. Random satellite platform selection from SATELLITE_PLATFORMS.
            3. Day/night determination using approximate local solar time.
            4. Swath corridor polygon and ground track line construction.
            5. Hotspot generation with 70/30 primary/flank spatial split.
            6. Radiometric property assignment (brightness, FRP, confidence).
            7. Full pipeline execution (validate → cluster → rank).
            8. Environmental parameter generation (cloud cover, solar zenith).

        Args:
            target_lat: Target latitude in decimal degrees.
            target_lon: Target longitude in decimal degrees.
            place_name: Optional human-readable name for the observation area.
            biome_hint: Optional biome/vegetation type string for context.

        Returns:
            Dict containing: success flag, target info, satellite_pass metadata
            (including swath_polygon and ground_track), validation_summary,
            cluster_summaries, ranked observations, and summary_stats.
        """
        # ---- Step 1: Coordinate Clamping ----
        # Clamp latitude to ±85° — beyond this, sun-synchronous orbits have
        # convergent ground tracks and fire detection is not meaningful
        # (polar ice caps). Longitude is clamped to standard ±180° range.
        target_lat = max(-85.0, min(85.0, float(target_lat)))
        target_lon = max(-180.0, min(180.0, float(target_lon)))
        
        # ---- Step 2: Satellite Platform Selection ----
        # Randomly select one of the available satellite platforms to simulate
        # which satellite would be overhead for this observation
        sat = random.choice(SATELLITE_PLATFORMS)
        # Orbit number: realistic range for current JPSS constellation orbits
        orbit_number = random.randint(31000, 39500)
        
        # ---- Step 3: Day / Night Determination ----
        # Approximate local solar time using the target's longitude.
        # Each 15° of longitude ≈ 1 hour of solar time offset from UTC.
        # This determines whether the observation simulates a daytime or
        # nighttime pass, which affects thermal contrast, background brightness,
        # and the nighttime quality bonus in the triage ranking.
        now_utc = datetime.now(timezone.utc)
        approx_solar_hour = (now_utc.hour + target_lon / 15.0) % 24.0
        # Daytime: roughly 6 AM to 6 PM local solar time
        is_day = 6.0 <= approx_solar_hour <= 18.0
        daynight = 'D' if is_day else 'N'
        
        # ---- Step 4: Ground Track and Swath Geometry ----
        # LEO sun-synchronous satellites orbit at ~98.7° inclination, meaning
        # the ground track is nearly North-South with a slight westward tilt.
        # The track_angle_deg represents this tilt: negative in Northern Hemisphere
        # (descending node), positive in Southern (ascending node).
        track_angle_deg = -8.2 if target_lat >= 0 else 8.2
        rad_angle = math.radians(track_angle_deg)
        
        # Convert half the swath width from km to degrees of longitude.
        # At the equator, 1° longitude ≈ 111 km. The swath width of VIIRS
        # is 3040 km, so half-swath ≈ 1520 km ≈ 13.5° of longitude.
        half_swath_deg = (sat['swath_width_km'] / 2.0) / 111.0
        
        # Define the swath corridor as a parallelogram around the target point.
        # lat_span determines the North-South extent of the visible swath segment.
        # The 0.15 factor applies the slight track angle skew to the polygon corners.
        lat_span = 3.5
        lon_span = half_swath_deg
        
        # Construct the 4-corner swath polygon (used for map visualization)
        # The parallelogram is slightly skewed to represent the satellite's
        # non-vertical ground track
        swath_polygon = [
            [round(target_lat + lat_span, 4), round(target_lon - lon_span + lat_span * 0.15, 4)],
            [round(target_lat + lat_span, 4), round(target_lon + lon_span + lat_span * 0.15, 4)],
            [round(target_lat - lat_span, 4), round(target_lon + lon_span - lat_span * 0.15, 4)],
            [round(target_lat - lat_span, 4), round(target_lon - lon_span - lat_span * 0.15, 4)]
        ]
        
        # Satellite ground track center line: 3 points showing the satellite's
        # sub-satellite path from North to South (or vice versa) across the target
        ground_track = [
            [round(target_lat + lat_span * 1.5, 4), round(target_lon + lat_span * 0.22, 4)],
            [round(target_lat, 4), round(target_lon, 4)],
            [round(target_lat - lat_span * 1.5, 4), round(target_lon - lat_span * 0.22, 4)]
        ]
        
        # ---- Step 5: Hotspot Generation ----
        # Generate a realistic distribution of active fire detections.
        # Number of hotspots per pass: 18–45 (typical for an active fire region)
        num_hotspots = random.randint(18, 45)
        raw_detections = []
        
        # Spatial spread radius in degrees for hotspot distribution.
        # ~0.18° ≈ 20 km at the equator, representing a typical fire complex extent
        spread = 0.18
        
        # Primary fire complex center: slightly offset from the target to add realism
        # (the satellite doesn't always image the fire dead-center)
        center_lat = target_lat + random.uniform(-0.05, 0.05)
        center_lon = target_lon + random.uniform(-0.05, 0.05)
        
        # Secondary flank fire center: offset from the primary center to simulate
        # a secondary fire front or spot fire downwind of the main complex
        flank_lat = center_lat + random.gauss(0, spread * 0.6)
        flank_lon = center_lon + random.gauss(0, spread * 0.6)
        
        # Use current UTC timestamp for the acquisition date and time
        acq_date = now_utc.strftime("%Y-%m-%d")
        acq_time = now_utc.strftime("%H%M")
        
        for i in range(num_hotspots):
            # ---- 70/30 Primary / Flank Split ----
            # 70% of hotspots are placed in the primary fire complex cluster
            # with tighter spatial dispersion and generally higher FRP.
            # 30% are placed in the secondary flank/outlier area with wider
            # dispersion and lower FRP (spot fires, flanking, smoldering edge).
            if random.random() < 0.70:
                # Primary cluster: tighter Gaussian spread (0.35 × 0.18° ≈ 7 km σ)
                h_lat = center_lat + random.gauss(0, spread * 0.35)
                h_lon = center_lon + random.gauss(0, spread * 0.35)
                # FRP for primary cluster: higher values, power-law-ish distribution
                # The ** 1.08 / 1.8 formula skews toward moderate-to-high values
                # while still allowing some low-FRP detections
                frp = round(random.uniform(15.0, 450.0) ** 1.08 / 1.8, 1)
            else:
                # Flank / outlier: wider Gaussian spread (0.45 × 0.18° ≈ 9 km σ)
                h_lat = flank_lat + random.gauss(0, spread * 0.45)
                h_lon = flank_lon + random.gauss(0, spread * 0.45)
                # FRP for flank: lower range, representing spot fires or smoldering edge
                frp = round(random.uniform(4.0, 85.0), 1)
                
            # ---- Brightness Temperature Calculation ----
            # Base brightness: 315 K (ambient + slight thermal anomaly)
            # FRP contribution: higher FRP → higher brightness, with 300 MW reference
            # The formula produces realistic VIIRS I4-band brightness temperatures:
            #   Low FRP (5 MW) → ~317 K, Mid FRP (50 MW) → ~333 K,
            #   High FRP (200 MW) → ~388 K, Max (~450 MW) → ~480 K
            bright = round(315.0 + min(180.0, (frp / 300.0) * 110.0 + random.uniform(-8, 12)), 1)
            # Background brightness: ambient surface temperature in Kelvin.
            # Nighttime backgrounds are cooler (~285-293 K) due to radiative cooling.
            # Daytime backgrounds are warmer (~294-302 K) due to solar heating.
            bright_bg = round(288.0 + random.uniform(-3, 5), 1) if not is_day else round(296.0 + random.uniform(-2, 6), 1)
            
            # ---- Confidence Assignment Logic ----
            # Confidence correlates with FRP: stronger thermal signals are more
            # reliably distinguished from background noise.
            # FRP > 50 MW: 92% chance of 'high' confidence (clear, intense signal)
            # FRP 15-50 MW: 80% chance of 'nominal', 20% chance of 'high'
            # FRP < 15 MW: 33% 'nominal', 67% 'low' (marginal, ambiguous signal)
            if frp > 50:
                conf = 'h' if random.random() > 0.08 else 'n'
            elif frp > 15:
                conf = 'n' if random.random() > 0.2 else 'h'
            else:
                conf = random.choice(['n', 'l', 'l'])
                
            # ---- Scan Resolution Calculation ----
            # The pixel footprint size depends on the viewing angle (distance from
            # the satellite's ground track center line). Near-nadir pixels are
            # smallest (~0.375 km for VIIRS); pixels at swath edge are stretched.
            # Formula: base nadir size (0.35 km) + proportional increase based on
            # angular distance from track center. 1200 km is approximately
            # half-swath distance where maximum distortion occurs.
            dist_from_track_km = abs(h_lon - target_lon) * 111.0 * math.cos(math.radians(target_lat))
            scan_size = round(0.35 + (dist_from_track_km / 1200.0) * 0.4, 2)
            
            raw_detections.append({
                'obs_id': f"SAT-{orbit_number % 1000:03d}-{i+1:03d}",
                'latitude': round(h_lat, 5),
                'longitude': round(h_lon, 5),
                'bright_ti4': bright,
                'scan': scan_size,
                'track': scan_size,
                'acq_date': acq_date,
                'acq_time': acq_time,
                'satellite': sat['name'],
                'instrument': sat['instrument'],
                'confidence': conf,
                'version': '2.0NUR',
                'bright_ti5': bright_bg,
                'frp': frp,
                'daynight': daynight
            })
            
        # ---- Step 6: Full Pipeline Execution ----
        # Pass the generated detections through the same 3-stage pipeline
        # (validation → clustering → ranking) used for real datasets.
        # This ensures the simulated data is processed identically to uploaded CSVs.
        valid_records, val_summary = FieldValidator.validate_and_normalize(raw_detections)
        clusterer = GeospatialClusterer(eps_km=4.0, min_samples=2)
        clustered_records, cluster_summaries = clusterer.cluster(valid_records)
        ranker = ExplainableTriageRanker()
        ranked_records = ranker.rank(clustered_records)
        
        # ---- Step 7: Environmental Parameters ----
        # Simulated atmospheric and observing conditions for this pass
        # Cloud cover: 8–32% (moderate clear-sky conditions for fire detection)
        cloud_cover_pct = random.randint(8, 32)
        # Solar zenith angle: determines sun elevation at the observation time.
        # Daytime: 22°–78° (sun is above horizon); Nighttime: 105°–160° (sun is below)
        solar_zenith_deg = round(random.uniform(22.0, 78.0) if is_day else random.uniform(105.0, 160.0), 1)
        
        return {
            "success": True,
            "target": {
                "name": place_name or f"Observation Area ({target_lat:.2f}°, {target_lon:.2f}°)",
                "latitude": round(target_lat, 4),
                "longitude": round(target_lon, 4),
                "biome": biome_hint or "Observed Geographic Sector"
            },
            "satellite_pass": {
                "satellite": sat['name'],
                "instrument": sat['instrument'],
                "agency": sat['agency'],
                "resolution": sat['resolution'],
                "orbit_type": sat['orbit'],
                "orbit_number": orbit_number,
                "swath_width_km": sat['swath_width_km'],
                "pass_time_utc": f"{acq_date} {acq_time[:2]}:{acq_time[2:]}:00 UTC",
                "day_night_state": "Daytime Pass (Solar heating present)" if is_day else "Nighttime Pass (Pristine thermal contrast)",
                "solar_zenith_angle": f"{solar_zenith_deg}°",
                "cloud_fraction": f"{cloud_cover_pct}%",
                # Atmospheric attenuation: the 3.9 µm atmospheric window is relatively
                # transparent, but cloud margins can attenuate the thermal signal
                "atmospheric_attenuation": "Low (Clear atmospheric window 3.9 µm)" if cloud_cover_pct < 20 else "Moderate (Marginal cloud margins)",
                "swath_polygon": swath_polygon,
                "ground_track": ground_track
            },
            "validation_summary": val_summary,
            "cluster_summaries": cluster_summaries,
            "observations": ranked_records,
            "summary_stats": {
                "total_hotspots": len(ranked_records),
                "total_clusters": len(cluster_summaries),
                "total_frp_mw": round(sum(r['frp'] for r in ranked_records), 1),
                "max_frp_mw": max([r['frp'] for r in ranked_records]) if ranked_records else 0,
                "critical_count": sum(1 for r in ranked_records if r['priority_tier'] == 'CRITICAL'),
                "high_count": sum(1 for r in ranked_records if r['priority_tier'] == 'HIGH'),
                "elevated_count": sum(1 for r in ranked_records if r['priority_tier'] == 'ELEVATED'),
                "low_count": sum(1 for r in ranked_records if r['priority_tier'] == 'MONITOR / LOW')
            }
        }

    @staticmethod
    def observe_random_region():
        """Picks a random known fire region or coordinates and performs observation.

        Selects a random entry from the GLOBAL_REGIONS list, applies slight
        coordinate jitter (±0.15° ≈ ±17 km) to avoid exact repetition, then
        delegates to observe_location() for full simulation.

        Returns:
            Same dict structure as observe_location(), with the randomly
            selected region's name and biome pre-filled.
        """
        region = random.choice(GLOBAL_REGIONS)
        # Add slight jitter to center so repeated calls produce different results
        jitter_lat = region['lat'] + random.uniform(-0.15, 0.15)
        jitter_lon = region['lon'] + random.uniform(-0.15, 0.15)
        return SatelliteObserver.observe_location(
            target_lat=jitter_lat,
            target_lon=jitter_lon,
            place_name=f"{region['name']}, {region['country']}",
            biome_hint=region['biome']
        )
