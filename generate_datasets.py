"""
Dataset generator for NASA FIRMS Active Fire observations.
Creates:
1. sample_firms_viirs.csv (Western USA / California fire complex)
2. sample_amazon_fires.csv (Amazon & Pantanal fire cluster)
3. sample_australia_fires.csv (Southeastern Australia bushfire front)
4. sample_mediterranean_fires.csv (Southern Europe / Greece summer fires)
5. labelled_benchmark.csv (Annotated evaluation dataset with ground-truth verification)
"""

import os
import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

def generate_cluster(center_lat, center_lon, num_points, spread_deg=0.15, 
                     frp_min=5.0, frp_max=350.0, base_date="2026-08-14",
                     satellite="NOAA-20", instrument="VIIRS"):
    records = []
    for i in range(num_points):
        # Gaussian distribution around center for realistic fire cluster shape
        lat = center_lat + random.gauss(0, spread_deg * 0.4)
        lon = center_lon + random.gauss(0, spread_deg * 0.4)
        
        # Power-law distribution for FRP (most fires are small, few are massive)
        u = random.random()
        frp = round(frp_min * ((frp_max / frp_min) ** u), 1)
        
        # Brightness temperature (Kelvin): typically 310K - 490K
        # Higher FRP correlates with higher brightness
        bright_ti4 = round(315.0 + (frp / frp_max) * 120.0 + random.uniform(-10, 15), 1)
        bright_ti4 = min(505.0, max(300.0, bright_ti4))
        
        # Background thermal channel bright_ti5 (Kelvin): typically 285K - 310K
        bright_ti5 = round(290.0 + random.uniform(-4, 6), 1)
        
        # Scan and track geometry (nadir is 0.32km, edge up to 0.78km for VIIRS 375m)
        scan = round(random.uniform(0.32, 0.75), 2)
        track = round(random.uniform(0.32, 0.75), 2)
        
        # Acquisition time (UTC)
        hour = random.choice([3, 4, 11, 19, 21])
        minute = random.randint(0, 59)
        acq_time = f"{hour:02d}{minute:02d}"
        
        daynight = 'D' if 6 <= hour <= 18 else 'N'
        
        # Confidence: 'h' for high, 'n' for nominal, 'l' for low
        if frp > 60 or (bright_ti4 - bright_ti5 > 50):
            conf = 'h' if random.random() > 0.1 else 'n'
        elif frp > 15:
            conf = 'n' if random.random() > 0.15 else 'h'
        else:
            conf = random.choice(['l', 'n', 'n'])
            
        records.append({
            'latitude': round(lat, 5),
            'longitude': round(lon, 5),
            'bright_ti4': bright_ti4,
            'scan': scan,
            'track': track,
            'acq_date': base_date,
            'acq_time': acq_time,
            'satellite': satellite,
            'instrument': instrument,
            'confidence': conf,
            'version': '2.0NUR',
            'bright_ti5': bright_ti5,
            'frp': frp,
            'daynight': daynight
        })
    return records

def generate_modis_cluster(center_lat, center_lon, num_points, spread_deg=0.25,
                           base_date="2026-08-14", satellite="Aqua"):
    records = []
    for i in range(num_points):
        lat = center_lat + random.gauss(0, spread_deg * 0.5)
        lon = center_lon + random.gauss(0, spread_deg * 0.5)
        
        frp = round(random.uniform(10.0, 500.0) ** 1.05 / 2.0, 1)
        brightness = round(310.0 + random.uniform(10, 120), 1)
        bright_t31 = round(288.0 + random.uniform(-3, 8), 1)
        
        scan = round(random.uniform(1.0, 2.8), 1)
        track = round(random.uniform(1.0, 1.8), 1)
        
        acq_time = f"{random.choice([13, 14, 21, 22]):02d}{random.randint(0, 59):02d}"
        hour = int(acq_time[:2])
        daynight = 'D' if 6 <= hour <= 18 else 'N'
        
        # MODIS confidence is numeric 0-100%
        if frp > 80:
            confidence = random.randint(85, 100)
        elif frp > 25:
            confidence = random.randint(60, 85)
        else:
            confidence = random.randint(30, 65)
            
        records.append({
            'latitude': round(lat, 5),
            'longitude': round(lon, 5),
            'brightness': brightness,
            'scan': scan,
            'track': track,
            'acq_date': base_date,
            'acq_time': acq_time,
            'satellite': satellite,
            'instrument': 'MODIS',
            'confidence': confidence,
            'version': '6.1N',
            'bright_t31': bright_t31,
            'frp': frp,
            'daynight': daynight
        })
    return records

def save_csv(filepath, fieldnames, rows):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} records to {filepath}")

def main():
    # 1. Western USA / California Complex (Sierra Nevada, Shasta-Trinity, Mendocino)
    viirs_fields = ['latitude', 'longitude', 'bright_ti4', 'scan', 'track', 'acq_date', 'acq_time',
                    'satellite', 'instrument', 'confidence', 'version', 'bright_ti5', 'frp', 'daynight']
    
    cal_records = []
    # Dixie / Sierra Fire Complex (high intensity multi-cluster)
    cal_records.extend(generate_cluster(39.98, -121.25, 75, spread_deg=0.22, frp_min=12.0, frp_max=650.0, satellite="NOAA-20"))
    # Trinity Alps fire flank
    cal_records.extend(generate_cluster(40.85, -122.95, 45, spread_deg=0.18, frp_min=8.0, frp_max=280.0, satellite="Suomi-NPP"))
    # Central Valley agricultural residue burns (lower FRP, distinct pattern)
    cal_records.extend(generate_cluster(36.75, -119.80, 25, spread_deg=0.35, frp_min=3.0, frp_max=22.0, satellite="NOAA-21"))
    # Southern CA brush anomaly
    cal_records.extend(generate_cluster(34.25, -117.85, 30, spread_deg=0.12, frp_min=15.0, frp_max=420.0, satellite="NOAA-20"))
    # Add a few isolated outlier/noise detections
    for _ in range(15):
        cal_records.append({
            'latitude': round(37.5 + random.uniform(-4, 4), 5),
            'longitude': round(-120.0 + random.uniform(-3, 3), 5),
            'bright_ti4': round(random.uniform(305.0, 325.0), 1),
            'scan': 0.55, 'track': 0.48, 'acq_date': "2026-08-14",
            'acq_time': "1945", 'satellite': "NOAA-20", 'instrument': "VIIRS",
            'confidence': 'l', 'version': '2.0NUR',
            'bright_ti5': round(random.uniform(293.0, 302.0), 1),
            'frp': round(random.uniform(1.2, 6.5), 1), 'daynight': 'D'
        })
    save_csv('data/sample_firms_viirs.csv', viirs_fields, cal_records)

    # 2. Amazon & Pantanal Fire Dataset (Brazil / Bolivia)
    amazon_records = []
    # Rondonia deforestation arc
    amazon_records.extend(generate_cluster(-9.50, -63.80, 90, spread_deg=0.45, frp_min=10.0, frp_max=450.0, satellite="NOAA-20"))
    # Mato Grosso agricultural & pasture fires
    amazon_records.extend(generate_cluster(-12.80, -55.60, 70, spread_deg=0.35, frp_min=5.0, frp_max=180.0, satellite="Suomi-NPP"))
    # Pantanal wetland grass fire
    amazon_records.extend(generate_cluster(-17.20, -56.90, 60, spread_deg=0.25, frp_min=20.0, frp_max=750.0, satellite="NOAA-21"))
    save_csv('data/sample_amazon_fires.csv', viirs_fields, amazon_records)

    # 3. Southeastern Australia Bushfires (New South Wales & Victoria)
    modis_fields = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'acq_date', 'acq_time',
                    'satellite', 'instrument', 'confidence', 'version', 'bright_t31', 'frp', 'daynight']
    aus_records = []
    # Blue Mountains mega-fire front
    aus_records.extend(generate_modis_cluster(-33.70, 150.30, 80, spread_deg=0.30, base_date="2026-01-12", satellite="Aqua"))
    # East Gippsland bushfire perimeter
    aus_records.extend(generate_modis_cluster(-37.45, 148.50, 65, spread_deg=0.25, base_date="2026-01-12", satellite="Terra"))
    # Southern Tablelands grass fire
    aus_records.extend(generate_modis_cluster(-35.20, 149.10, 35, spread_deg=0.20, base_date="2026-01-12", satellite="Aqua"))
    save_csv('data/sample_australia_fires.csv', modis_fields, aus_records)

    # 4. Mediterranean / Greece Fire Dataset
    med_records = []
    # Attica / Evia mega fire
    med_records.extend(generate_cluster(38.65, 23.60, 65, spread_deg=0.18, frp_min=15.0, frp_max=580.0, base_date="2026-07-28", satellite="NOAA-20"))
    # Rhodes wildfire front
    med_records.extend(generate_cluster(36.20, 27.95, 45, spread_deg=0.14, frp_min=18.0, frp_max=490.0, base_date="2026-07-28", satellite="NOAA-21"))
    # Peloponnese scrub fire
    med_records.extend(generate_cluster(37.60, 21.80, 40, spread_deg=0.20, frp_min=8.0, frp_max=220.0, base_date="2026-07-28", satellite="Suomi-NPP"))
    save_csv('data/sample_mediterranean_fires.csv', viirs_fields, med_records)

    # 5. Labelled Validation Benchmark (Ground-Truth Annotated)
    # Includes classes:
    # - true_wildfire (Active uncontrolled vegetation/forest fire)
    # - agricultural_burn (Controlled seasonal crop stubble/pasture burn)
    # - industrial_flare (Refinery/gas flare/metallurgical plant)
    # - false_alarm_glint (Solar glint on water/greenhouse or hot bare desert soil)
    benchmark_fields = [
        'obs_id', 'latitude', 'longitude', 'brightness', 'scan', 'track', 
        'acq_date', 'acq_time', 'satellite', 'instrument', 'confidence', 
        'bright_t31', 'frp', 'daynight', 'ground_truth_label', 'land_cover_type',
        'persistence_count', 'distance_to_refinery_km', 'verification_method'
    ]
    
    benchmark_records = []
    obs_id = 1
    
    # Class 1: True Wildfire (160 samples)
    for _ in range(160):
        frp = round(random.uniform(25.0, 750.0), 1)
        bright = round(330.0 + (frp / 750.0) * 120.0 + random.uniform(-10, 10), 1)
        bright_bg = round(290.0 + random.uniform(-3, 5), 1)
        conf = random.choice(['h', 'h', 'h', 'n'])
        daynight = random.choice(['D', 'N', 'D'])
        persistence = random.randint(2, 8) # Observed across multiple satellite passes
        
        benchmark_records.append({
            'obs_id': f"OBS-{obs_id:04d}",
            'latitude': round(39.0 + random.uniform(-4, 4), 4),
            'longitude': round(-121.0 + random.uniform(-4, 4), 4),
            'brightness': bright,
            'scan': round(random.uniform(0.35, 0.75), 2),
            'track': round(random.uniform(0.35, 0.75), 2),
            'acq_date': "2026-08-14",
            'acq_time': f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}",
            'satellite': random.choice(['NOAA-20', 'NOAA-21', 'Aqua']),
            'instrument': 'VIIRS',
            'confidence': conf,
            'bright_t31': bright_bg,
            'frp': frp,
            'daynight': daynight,
            'ground_truth_label': 'true_wildfire',
            'land_cover_type': random.choice(['Coniferous Forest', 'Mixed Woodland', 'Chaparral Shrubland']),
            'persistence_count': persistence,
            'distance_to_refinery_km': round(random.uniform(35.0, 180.0), 1),
            'verification_method': 'Incident Ground Report & Aerial Recon'
        })
        obs_id += 1

    # Class 2: Agricultural Burn (70 samples)
    for _ in range(70):
        frp = round(random.uniform(3.5, 28.0), 1)
        bright = round(312.0 + random.uniform(2, 20), 1)
        bright_bg = round(295.0 + random.uniform(-2, 4), 1)
        conf = random.choice(['n', 'l', 'n'])
        persistence = 1 # Usually single pass
        
        benchmark_records.append({
            'obs_id': f"OBS-{obs_id:04d}",
            'latitude': round(36.5 + random.uniform(-2, 2), 4),
            'longitude': round(-119.5 + random.uniform(-2, 2), 4),
            'brightness': bright,
            'scan': round(random.uniform(0.4, 0.8), 2),
            'track': round(random.uniform(0.4, 0.8), 2),
            'acq_date': "2026-08-14",
            'acq_time': f"{random.choice([11, 13, 14]):02d}{random.randint(0, 59):02d}",
            'satellite': 'NOAA-20',
            'instrument': 'VIIRS',
            'confidence': conf,
            'bright_t31': bright_bg,
            'frp': frp,
            'daynight': 'D',
            'ground_truth_label': 'agricultural_burn',
            'land_cover_type': 'Cropland / Stubble',
            'persistence_count': persistence,
            'distance_to_refinery_km': round(random.uniform(15.0, 90.0), 1),
            'verification_method': 'Agricultural Permit Registry'
        })
        obs_id += 1

    # Class 3: Industrial Flare (40 samples)
    # Very high brightness, moderate FRP, exact same location repeatedly, near known refinery/gas facility
    for _ in range(40):
        frp = round(random.uniform(12.0, 65.0), 1)
        bright = round(345.0 + random.uniform(10, 45), 1)
        bright_bg = round(292.0 + random.uniform(-2, 4), 1)
        conf = 'h'
        persistence = random.randint(15, 60) # Recurrent point source
        
        benchmark_records.append({
            'obs_id': f"OBS-{obs_id:04d}",
            'latitude': round(37.95 + random.uniform(-0.02, 0.02), 4),
            'longitude': round(-122.35 + random.uniform(-0.02, 0.02), 4),
            'brightness': bright,
            'scan': round(random.uniform(0.35, 0.6), 2),
            'track': round(random.uniform(0.35, 0.6), 2),
            'acq_date': "2026-08-14",
            'acq_time': f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}",
            'satellite': 'NOAA-21',
            'instrument': 'VIIRS',
            'confidence': conf,
            'bright_t31': bright_bg,
            'frp': frp,
            'daynight': random.choice(['N', 'D']),
            'ground_truth_label': 'industrial_flare',
            'land_cover_type': 'Industrial / Petroleum Complex',
            'persistence_count': persistence,
            'distance_to_refinery_km': round(random.uniform(0.05, 0.8), 2),
            'verification_method': 'EPA Industrial Point Source Database'
        })
        obs_id += 1

    # Class 4: False Positive / Solar Glint / Hot Surface (50 samples)
    for _ in range(50):
        frp = round(random.uniform(0.5, 7.0), 1)
        bright = round(308.0 + random.uniform(2, 12), 1)
        bright_bg = round(304.0 + random.uniform(0, 6), 1) # Small or negligible delta T
        conf = random.choice(['l', 'l', 'n'])
        
        benchmark_records.append({
            'obs_id': f"OBS-{obs_id:04d}",
            'latitude': round(35.5 + random.uniform(-3, 3), 4),
            'longitude': round(-116.0 + random.uniform(-3, 3), 4),
            'brightness': bright,
            'scan': round(random.uniform(0.6, 1.2), 2),
            'track': round(random.uniform(0.6, 1.2), 2),
            'acq_date': "2026-08-14",
            'acq_time': f"{random.choice([12, 13, 14, 15]):02d}{random.randint(0, 59):02d}",
            'satellite': 'Terra',
            'instrument': 'MODIS',
            'confidence': conf,
            'bright_t31': bright_bg,
            'frp': frp,
            'daynight': 'D', # Always daytime solar glint
            'ground_truth_label': 'false_alarm_glint',
            'land_cover_type': random.choice(['Solar Farm Panel', 'Bare Playa Salt Flat', 'Water Body Edge']),
            'persistence_count': 1,
            'distance_to_refinery_km': round(random.uniform(20.0, 150.0), 1),
            'verification_method': 'Sentinel-2 High-Res Optical Cloud/Glint Audit'
        })
        obs_id += 1

    save_csv('data/labelled_benchmark.csv', benchmark_fields, benchmark_records)
    print("All datasets successfully generated.")

if __name__ == '__main__':
    main()
