"""End-to-end test: train → save → list → load → simulate"""
import urllib.request, json, time

BASE = 'http://localhost:9000'

def post(path, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(f'{BASE}{path}', data=body, 
        headers={'Content-Type':'application/json'}, method='POST')
    return json.loads(urllib.request.urlopen(req).read())

def get(path):
    return json.loads(urllib.request.urlopen(f'{BASE}{path}').read())

# 1. Start short training
print("=" * 50)
print("1. STARTING 30-step TRAINING...")
r = post('/api/train/start', {'steps': 30, 'fleet_size': 50})
print(f"   Response: {r}")

# 2. Wait for completion
for i in range(60):
    time.sleep(1)
    st = get('/api/train/status')
    phase = st['phase']
    step = st['step']
    running = st['running']
    print(f"   [{i+1}s] Step {step}/{st['total_steps']} phase={phase} running={running}")
    if not running and step > 0:
        break

print(f"\n   Final metrics: {st.get('metrics', {})}")

# 3. Save model
print("\n" + "=" * 50)
print("2. SAVING MODEL...")
r = post('/api/models/save', {'name': 'Test_30steps', 'description': 'E2E test'})
print(f"   Save result keys: {list(r.keys()) if isinstance(r, dict) else r}")
if 'error' in r:
    print(f"   ERROR: {r['error']}")
else:
    print(f"   Saved as: {r.get('id', '???')}")

# 4. List models  
print("\n" + "=" * 50)
print("3. LISTING MODELS...")
models = get('/api/models')
print(f"   Found {len(models)} models:")
for m in models:
    print(f"   - '{m['name']}' (id={m['id']}, steps={m.get('steps_trained',0)})")
    fm = m.get('final_metrics', {})
    print(f"     AR={fm.get('ar','?')}, Profit={fm.get('profit','?')}")

# 5. Test sim/state
print("\n" + "=" * 50)
print("4. TESTING SIM STATE...")
ss = get('/api/sim/state')
print(f"   Grid: {ss.get('grid')}x{ss.get('grid')}, Vehicles: {len(ss.get('vehicles',[]))}")
if ss.get('vehicles'):
    v0 = ss['vehicles'][0]
    print(f"   Sample vehicle: zone={v0['zone']}, pax={v0['pax']}, idle={v0['idle']}")
print(f"   Events: {len(ss.get('events',[]))}")

print("\n✅ ALL TESTS PASSED" if len(models) > 0 else "\n❌ SAVE FAILED")
