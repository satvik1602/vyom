"""
Integration tests for Satellite Fire Triage System API
"""

import urllib.request
import json
import sys
import time

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:5000"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read()

def post(path, data):
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def run_tests():
    print("Testing Satellite Fire Triage API Endpoints...")
    time.sleep(1)

    # 1. Test GET /
    status, body = get('/')
    assert status == 200, f"Expected 200, got {status}"
    assert b"SAT-FIRE TRIAGE" in body
    print("✅ GET /: OK (HTML UI Rendered)")

    # 2. Test GET /api/datasets
    status, body = get('/api/datasets')
    data = json.loads(body.decode('utf-8'))
    assert 'datasets' in data
    print(f"✅ GET /api/datasets: OK ({len(data['datasets'])} datasets available)")

    # 3. Test POST /api/dataset/load
    status, data = post('/api/dataset/load', {'dataset_name': 'sample_firms_viirs.csv'})
    assert data['success'] is True
    assert len(data['records']) > 0
    print(f"✅ POST /api/dataset/load: OK ({len(data['records'])} records, {len(data['cluster_summaries'])} clusters)")

    # 4. Test POST /api/evaluation/run
    status, data = post('/api/evaluation/run', {'threshold': 60.0})
    assert data['has_data'] is True
    assert 'confusion_matrix' in data
    m = data['metrics']
    print(f"✅ POST /api/evaluation/run: OK (Precision: {m['precision']*100:.1f}%, Recall: {m['recall']*100:.1f}%, F1: {m['f1_score']*100:.1f}%)")

    # 5. Test GET /api/observer/presets
    status, body = get('/api/observer/presets')
    data = json.loads(body.decode('utf-8'))
    assert len(data['presets']) > 0
    print(f"✅ GET /api/observer/presets: OK ({len(data['presets'])} presets loaded)")

    # 6. Test POST /api/observer/random
    status, data = post('/api/observer/random', {})
    assert data['success'] is True
    assert 'satellite_pass' in data
    assert len(data['observations']) > 0
    print(f"✅ POST /api/observer/random: OK (Target: {data['target']['name']}, Sat: {data['satellite_pass']['satellite']}, Hotspots: {len(data['observations'])})")

    # 7. Test POST /api/observer/observe with custom coords
    status, data = post('/api/observer/observe', {
        'lat': 52.87, 'lon': -118.08, 'name': 'Jasper Boreal Alberta', 'biome': 'Boreal Spruce'
    })
    assert data['success'] is True
    print(f"✅ POST /api/observer/observe (Custom Coords): OK ({data['target']['name']})")

    # 8. Test GET /api/export/csv
    status, body = get('/api/export/csv')
    assert status == 200
    assert b"triage_rank" in body
    print("✅ GET /api/export/csv: OK (CSV successfully generated)")

    # 9. Test GET /api/export/report
    status, body = get('/api/export/report')
    data = json.loads(body.decode('utf-8'))
    assert 'executive_summary' in data
    print("✅ GET /api/export/report: OK (Analyst JSON Briefing generated)")

    # 10. Test GET /api/geocode
    status, body = get('/api/geocode?q=California')
    data = json.loads(body.decode('utf-8'))
    assert len(data['results']) > 0
    print(f"✅ GET /api/geocode: OK (Found {len(data['results'])} matching places for 'California')")

    print("\n🎉 ALL 10 SYSTEM TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    run_tests()
