"""
=============================================================================
Distributed Model-Free Ride-Sharing: DARM + DPRS + DQN
=============================================================================
Paper-faithful implementation of Haliem et al., IEEE TITS 2021:
    • DARM greedy assignment + insertion-based route planning
    • DPRS distributed pricing with customer accept/reject
    • DQN dispatching with multi-agent execution (shared policy)
    • State uses supply/demand forecasts and vehicle features

Run:
    python ridesharing_darm_dprs_dqn.py

Requirements: numpy, matplotlib  (torch optional; kagglehub optional for dataset)
=============================================================================
"""

import math, random, time, collections, os, json, argparse, threading, queue, sys
import csv
import glob
import shutil
import pathlib
import datetime as dt
# Fix Windows console encoding for Unicode output
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PyTorch (optional) ────────────────────────────────────────────────────
try:
    import torch, torch.nn as nn, torch.optim as optim
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False
    print("[WARN] PyTorch not found – using numpy Q-table fallback.")

# ── Simple progress bar (no tqdm required) ───────────────────────────────
class tqdm:
    def __init__(self, iterable=None, total=None, unit="it", desc=""):
        self.iterable = list(iterable) if iterable is not None else list(range(total))
        self.total = len(self.iterable); self.n = 0
        self.desc = desc; self._pf = {}
    def _render(self):
        if self.total <= 0:
            return
        if self.n % max(1, self.total // 20) == 0 or self.n == self.total:
            pct = 100 * self.n / max(1, self.total)
            info = "  ".join(f"{k}={v}" for k,v in self._pf.items())
            print(f"\r  {self.desc}[{pct:5.1f}%] {self.n}/{self.total}  {info}",
                  end="", flush=True)
    def __iter__(self):
        for item in self.iterable:
            yield item; self.n += 1
            self._render()
        print()
    def set_postfix(self, d): self._pf = d
    def update(self, n=1):
        self.n += n
        self._render()

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GRID_W = GRID_H      = 15          # 15×15 = 225 zones  (≈1 mile/cell)
ZONE_KM              = 0.8         # km per cell (calibrated for pricing equilibrium)
MAX_DISPATCH_CELLS   = 7           # ±7 cells → 15×15 action space
ACTION_DIM           = (2*MAX_DISPATCH_CELLS+1)**2  # 225

N_VEHICLES           = 50
N_TYPES              = 3
MAX_CAP              = [2, 4, 6]
MILEAGE_L            = [35, 28, 22]   # km/litre
BASE_FARE            = [5.0, 8.0, 12.0]
RATE_KM              = [1.5, 2.0, 2.5]   # $/km  (ω1)
RATE_WAIT            = [0.1, 0.15, 0.2]  # $/min  (ω3)
PGAS                 = 1.5   # $/litre

REJECT_RADIUS_KM     = 3.0         # max pickup distance (paper: 5 km)
MAX_IDLE_MIN         = 10          # dispatch idle vehicles after this

# Forecast horizon for supply/demand in DQN state (t:t+T)
FORECAST_H           = 1

# Route feasibility constraints (paper: delay windows + detour constraints)
MAX_DETOUR_FACTOR    = 1.5         # max in-vehicle detour vs. direct travel

# DQN
GAMMA=0.95; LR=5e-4; BATCH=64; BUF=5000; MIN_BUF=200
EPS0=1.0; EPS_MIN=0.05; EPS_DEC=0.997; TGT_UPD=150

TRAIN_STEPS = 3000
DEMO_STEPS  = 500
WARMUP_STEPS = 20          # Paper: 20 min without dispatching

# Reward weights (Eq. 6-style).
# Components: served_pax (+), dispatch_time (-), detour_time (-),
# profit_step (+), idle_flag (-).
B1,B2,B3,B4,B5 = 10,-1,-5,12,-8
# Customer weights (Eq. 4)
W4,W5,W6 = 15,1,4
HOTSPOT_FRAC = 0.10

SEED = 42
random.seed(SEED); np.random.seed(SEED)
if TORCH:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# Algorithm presets for UI/CLI selection
DEFAULT_ALGO_CONFIG = dict(dispatch=True, rideshare=True, pricing=True, darm=True)
ALGO_PRESETS = {
    "full":        dict(dispatch=True,  rideshare=True,  pricing=True,  darm=True),
    "dqn_only":    dict(dispatch=True,  rideshare=False, pricing=False, darm=False),
    "no_pricing":  dict(dispatch=True,  rideshare=True,  pricing=False, darm=True),
    "no_rideshare":dict(dispatch=True,  rideshare=False, pricing=True,  darm=True),
    "greedy":      dict(dispatch=True,  rideshare=False, pricing=True,  darm=False),
}

EVAL_INTERVAL = 200
EVAL_STEPS = 100
EVAL_SEED = 123

def resolve_algo(cfg):
    """Resolve algorithm preset or config dict to full config."""
    if cfg is None:
        cfg = "full"
    if isinstance(cfg, str):
        base = DEFAULT_ALGO_CONFIG.copy()
        base.update(ALGO_PRESETS.get(cfg, {}))
        base["name"] = cfg
        return base
    if isinstance(cfg, dict):
        base = DEFAULT_ALGO_CONFIG.copy()
        base.update(cfg)
        base.setdefault("name", "custom")
        return base
    return DEFAULT_ALGO_CONFIG.copy()

def calc_eps_decay_rate(eps_start, eps_end, decay_steps):
    """Convert decay steps to multiplicative epsilon decay rate."""
    if decay_steps is None or decay_steps <= 0:
        return 1.0
    if eps_end >= eps_start:
        return 1.0
    return float(math.exp(math.log(eps_end / eps_start) / max(1.0, decay_steps)))

# ═══════════════════════════════════════════════════════════════════════════
# 1. City Grid
# ═══════════════════════════════════════════════════════════════════════════
class CityGrid:
    def __init__(self):
        self.W, self.H = GRID_W, GRID_H
        self.n = self.W * self.H

    def rc(self, z):  return divmod(int(z), self.W)
    def zid(self, r, c): return max(0,min(r,self.H-1))*self.W + max(0,min(c,self.W-1))
    def rand(self):   return random.randint(0, self.n-1)

    def dist_km(self, a, b):
        r1,c1 = self.rc(a); r2,c2 = self.rc(b)
        return (abs(r1-r2)+abs(c1-c2))*ZONE_KM

    def action_to_zone(self, z, act):
        side = 2*MAX_DISPATCH_CELLS+1
        dr, dc = act//side - MAX_DISPATCH_CELLS, act%side - MAX_DISPATCH_CELLS
        r, c = self.rc(z)
        return self.zid(r+dr, c+dc)

G = CityGrid()

# ═══════════════════════════════════════════════════════════════════════════
# 2. ETA, travel-time uncertainty, and route cost
#    Extension: stochastic congestion model per zone + time-of-day
# ═══════════════════════════════════════════════════════════════════════════
AVG_SPEED = 0.5   # km/min = 30 km/h (free-flow)

# Zone congestion: each zone has a base congestion level that varies by time
_zone_congestion = np.random.uniform(0.0, 0.3, G.n).astype(np.float32)

_osrm_cache = collections.OrderedDict()

def _zone_center_latlon(zone):
    r, c = G.rc(zone)
    lat = NYC_LAT_MAX - (r + 0.5) / GRID_H * (NYC_LAT_MAX - NYC_LAT_MIN)
    lon = NYC_LON_MIN + (c + 0.5) / GRID_W * (NYC_LON_MAX - NYC_LON_MIN)
    return lat, lon

def _osrm_route(a, b):
    if a == b:
        return 0.0, 0.0
    key = (a, b)
    if key in _osrm_cache:
        dist_km, dur_min = _osrm_cache[key]
        _osrm_cache.move_to_end(key)
        return dist_km, dur_min
    lat1, lon1 = _zone_center_latlon(a)
    lat2, lon2 = _zone_center_latlon(b)
    url = (f"{OSRM_BASE_URL}/route/v1/{OSRM_PROFILE}/"
           f"{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}?overview=false")
    try:
        with urllib.request.urlopen(url, timeout=OSRM_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        routes = payload.get("routes") or []
        if not routes:
            raise RuntimeError("OSRM route not found")
        dist_km = float(routes[0]["distance"]) / 1000.0
        dur_min = float(routes[0]["duration"]) / 60.0
    except Exception as exc:
        # Fallback to Manhattan distance on error
        dist_km = G.dist_km(a, b)
        dur_min = dist_km / AVG_SPEED
        if OSRM_ENABLED:
            print(f"[WARN] OSRM fallback for {a}->{b}: {exc}")
    _osrm_cache[key] = (dist_km, dur_min)
    if len(_osrm_cache) > OSRM_CACHE_SIZE:
        _osrm_cache.popitem(last=False)
    return dist_km, dur_min

def dist_km(a, b):
    if OSRM_ENABLED:
        return _osrm_route(a, b)[0]
    return G.dist_km(a, b)

def travel_time_min(a, b, t=0):
    if OSRM_ENABLED:
        dist_km_val, dur_min = _osrm_route(a, b)
        if CONGESTION_ENABLED:
            return max(0.3, dur_min * congestion_factor(a, t))
        return max(0.3, dur_min)
    dist = dist_km(a, b)
    cf = congestion_factor(a, t)
    return max(0.3, dist / AVG_SPEED * cf)

def congestion_factor(zone, t):
    """Time-varying congestion multiplier for a zone.
    Returns >1.0 (slower travel), peaks during rush hours (t=480,1020)."""
    if not CONGESTION_ENABLED:
        return 1.0
    base = 1.0 + _zone_congestion[zone]
    # Rush-hour peaks at t=480 (8am) and t=1020 (5pm) in minutes
    hour_min = t % 1440
    rush = 0.3 * (math.exp(-((hour_min-480)**2)/5000) +
                  math.exp(-((hour_min-1020)**2)/5000))
    noise = random.gauss(0, 0.08)  # stochastic component
    return max(0.6, base + rush + noise)

def eta_min(a, b, t=0):
    """Travel time with optional OSRM and congestion."""
    return travel_time_min(a, b, t)

def route_cost(zone_list):
    """Total km for a list of zone stops."""
    return sum(dist_km(zone_list[i], zone_list[i+1])
               for i in range(len(zone_list)-1)) if len(zone_list) > 1 else 0.

# Extensions (off by default to match the paper)
CONGESTION_ENABLED = False
USE_TIME_FEATURES = False
RIDER_CHANGE_PROB = 0.0
RIDER_CANCEL_PROB = 0.0

# OSRM routing (off by default)
OSRM_ENABLED = False
OSRM_BASE_URL = "http://router.project-osrm.org"
OSRM_PROFILE = "driving"
OSRM_TIMEOUT_SEC = 2.0
OSRM_CACHE_SIZE = 50000

# Dataset-driven demand (KaggleHub). Enabled by default; falls back to synthetic.
USE_DATASET = True
DATASET_PATH = ""
DATASET_MAX_FILES = 1
DATASET_MAX_ROWS = 100000
DATASET_TIME_BIN = 1
DATASET_DEMAND_SCALE = 0.1
DATASET_FALLBACK_BASE = 10.0

# NYC bounding box (approx) for lat/lon mapping to grid
NYC_LAT_MIN = 40.4774
NYC_LAT_MAX = 40.9176
NYC_LON_MIN = -74.2591
NYC_LON_MAX = -73.7004
TLC_MAX_ZONE_ID = 263

# ═══════════════════════════════════════════════════════════════════════════
# 3. Demand model – spatially-biased Poisson demand with rush-hour peaks
# ═══════════════════════════════════════════════════════════════════════════
class DemandModel:
    def __init__(self):
        w = np.ones(G.n)
        mr,mc = G.H//2, G.W//2   # midtown
        dr,dc = G.H//4, G.W//4   # downtown
        for z in range(G.n):
            r,c = G.rc(z)
            w[z] = (5*math.exp(-(abs(r-mr)+abs(c-mc))/3)
                  + 3*math.exp(-(abs(r-dr)+abs(c-dc))/3) + 0.5)
        self.weights = w / w.sum()

    def time_mult(self, t_min):
        h = (t_min // 60) % 24
        if 7<=h<9 or 17<=h<19: return 2.5
        if 22<=h or h<5:        return 0.4
        return 1.2

    def generate(self, t, base=10.0):
        n = np.random.poisson(base * self.time_mult(t))
        out = []
        for _ in range(n):
            o = np.random.choice(G.n, p=self.weights)
            d = np.random.choice(G.n, p=self.weights)
            while d == o: d = np.random.choice(G.n, p=self.weights)
            n_pax = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            out.append((int(o), int(d), int(n_pax)))
        return out

    def future(self, t, horizon=30):
        return self.weights * 10.0 * self.time_mult(t) * horizon

def _safe_int(val, default=None):
    try:
        if val is None:
            return default
        return int(float(val))
    except (TypeError, ValueError):
        return default

def _safe_float(val, default=None):
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

def _parse_pickup_time(value):
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None

def _latlon_to_zone(lat, lon):
    if lat is None or lon is None:
        return None
    if not (NYC_LAT_MIN <= lat <= NYC_LAT_MAX and NYC_LON_MIN <= lon <= NYC_LON_MAX):
        return None
    r = int((NYC_LAT_MAX - lat) / (NYC_LAT_MAX - NYC_LAT_MIN) * GRID_H)
    c = int((lon - NYC_LON_MIN) / (NYC_LON_MAX - NYC_LON_MIN) * GRID_W)
    return G.zid(r, c)

def _zone_id_to_zone(zone_id, max_zone_id=TLC_MAX_ZONE_ID):
    zid = _safe_int(zone_id, None)
    if zid is None or zid <= 0:
        return None
    return int((zid - 1) / max(1, max_zone_id) * (G.n - 1))

def _detect_columns(fieldnames):
    if not fieldnames:
        return {}
    lower_map = {f.lower().strip(): f for f in fieldnames}
    def pick(*cands):
        for c in cands:
            key = c.lower().strip()
            if key in lower_map:
                return lower_map[key]
        return None
    return {
        "pickup_time": pick("tpep_pickup_datetime", "pickup_datetime", "lpep_pickup_datetime"),
        "passenger_count": pick("passenger_count"),
        "pu_loc": pick("pulocationid", "pu_location_id"),
        "do_loc": pick("dolocationid", "do_location_id"),
        "pu_lat": pick("pickup_latitude"),
        "pu_lon": pick("pickup_longitude"),
        "do_lat": pick("dropoff_latitude"),
        "do_lon": pick("dropoff_longitude"),
    }

def _find_dataset_files(root_dir, max_files=1):
    if not root_dir or not os.path.isdir(root_dir):
        return []
    csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    csv_files.sort()
    if max_files:
        return csv_files[:max_files]
    return csv_files

def _download_dataset_kagglehub():
    """Download NYC taxi dataset from KaggleHub with local caching."""
    
    # Use ~/.cache/darm_dprs_dqn/ as the cache directory
    cache_dir = pathlib.Path.home() / '.cache' / 'darm_dprs_dqn'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_marker = cache_dir / 'dataset_cached.txt'
    
    # Check if we already have the dataset cached
    if cache_marker.exists():
        cached_path = str(cache_dir)
        # Verify at least one CSV file exists
        csv_files = glob.glob(os.path.join(cached_path, '**', '*.csv'), recursive=True)
        if csv_files:
            print(f"[INFO] Using cached dataset from {cached_path}")
            return cached_path
    
    try:
        import kagglehub
    except Exception as exc:
        print(f"[WARN] kagglehub not available: {exc}")
        return ""
    try:
        # Download to cache directory
        print("[INFO] Downloading NYC taxi dataset (this may take a moment)...")
        path = kagglehub.dataset_download("elemento/nyc-yellow-taxi-trip-data")
        print(f"[INFO] Downloaded dataset to: {path}")
        
        # Move/copy dataset files to cache if not already there
        csv_files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
        if csv_files:
            print(f"[INFO] Caching dataset ({len(csv_files)} CSV files)...")
            for csv_file in csv_files[:1]:  # Cache first file
                dst = cache_dir / os.path.basename(csv_file)
                shutil.copy2(csv_file, dst)
            cache_marker.write_text("Dataset cached from KaggleHub\n")
            print(f"[INFO] Dataset cached to {cache_dir}")
            return str(cache_dir)
        
        return path
    except Exception as exc:
        print(f"[WARN] KaggleHub download failed: {exc}")
        return ""

def _scale_trip_list(trips, scale):
    if scale == 1.0:
        return list(trips)
    if scale <= 0.0:
        return []
    if not trips:
        return []
    if scale < 1.0:
        k = int(len(trips) * scale)
        return random.sample(trips, k) if k > 0 else []
    k = int(len(trips) * scale)
    reps, rem = divmod(k, len(trips))
    out = []
    for _ in range(reps):
        out.extend(trips)
    if rem:
        out.extend(random.sample(trips, rem))
    return out

class DatasetDemandModel:
    def __init__(self, dataset_dir, max_files=1, max_rows=200000, time_bin=1, demand_scale=1.0):
        self.dataset_dir = dataset_dir
        self.max_files = max_files
        self.max_rows = max_rows
        self.time_bin = max(1, int(time_bin))
        self.demand_scale = float(demand_scale)
        self.minutes_per_day = 1440 // self.time_bin
        self.trips_by_minute = [list() for _ in range(self.minutes_per_day)]
        self.counts_by_minute = [np.zeros(G.n, np.float32) for _ in range(self.minutes_per_day)]
        self.weights = np.ones(G.n, np.float32) / max(1, G.n)
        self.total_rows = 0
        self.mean_per_minute = 0.0
        self._load()

    def _load(self):
        files = _find_dataset_files(self.dataset_dir, self.max_files)
        if not files:
            raise RuntimeError("No CSV files found in dataset path")
        total_rows = 0
        for path in files:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = _detect_columns(reader.fieldnames)
                if not cols.get("pickup_time"):
                    continue
                for row in reader:
                    if self.max_rows and total_rows >= self.max_rows:
                        break
                    pu_time = _parse_pickup_time(row.get(cols["pickup_time"]))
                    if pu_time is None:
                        continue
                    pax = _safe_int(row.get(cols.get("passenger_count")), 1) or 1
                    if pax <= 0:
                        pax = 1
                    if cols.get("pu_loc") and cols.get("do_loc"):
                        o = _zone_id_to_zone(row.get(cols["pu_loc"]))
                        d = _zone_id_to_zone(row.get(cols["do_loc"]))
                    else:
                        pu_lat = _safe_float(row.get(cols.get("pu_lat")))
                        pu_lon = _safe_float(row.get(cols.get("pu_lon")))
                        do_lat = _safe_float(row.get(cols.get("do_lat")))
                        do_lon = _safe_float(row.get(cols.get("do_lon")))
                        o = _latlon_to_zone(pu_lat, pu_lon)
                        d = _latlon_to_zone(do_lat, do_lon)
                    if o is None or d is None or o == d:
                        continue
                    minute = (pu_time.hour * 60 + pu_time.minute) // self.time_bin
                    if minute < 0 or minute >= self.minutes_per_day:
                        continue
                    self.trips_by_minute[minute].append((int(o), int(d), int(pax)))
                    self.counts_by_minute[minute][int(o)] += 1.0
                    total_rows += 1
                if self.max_rows and total_rows >= self.max_rows:
                    break
        self.total_rows = total_rows
        totals = np.sum(np.stack(self.counts_by_minute), axis=0)
        if totals.sum() > 0:
            self.weights = totals / totals.sum()
        self.mean_per_minute = float(total_rows / max(1, self.minutes_per_day))
        print(f"[INFO] Dataset loaded: {total_rows} rows from {len(files)} file(s)")

    def _fallback_generate(self, t):
        base = self.mean_per_minute if self.mean_per_minute > 0 else DATASET_FALLBACK_BASE
        n = np.random.poisson(base * self.demand_scale)
        out = []
        for _ in range(n):
            o = np.random.choice(G.n, p=self.weights)
            d = np.random.choice(G.n, p=self.weights)
            while d == o:
                d = np.random.choice(G.n, p=self.weights)
            n_pax = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            out.append((int(o), int(d), int(n_pax)))
        return out

    def generate(self, t, base=10.0):
        minute = (t % (self.minutes_per_day * self.time_bin)) // self.time_bin
        trips = self.trips_by_minute[minute]
        if trips:
            return _scale_trip_list(trips, self.demand_scale)
        return self._fallback_generate(t)

    def future(self, t, horizon=30):
        minute = (t % (self.minutes_per_day * self.time_bin)) // self.time_bin
        counts = self.counts_by_minute[minute]
        if counts.sum() <= 0:
            base = self.mean_per_minute if self.mean_per_minute > 0 else DATASET_FALLBACK_BASE
            return self.weights * base * self.demand_scale * horizon
        return counts * self.demand_scale * horizon

_DEMAND_MODEL_CACHE = {"key": None, "model": None}

def build_demand_model(use_dataset=True, dataset_path="", max_files=1, max_rows=200000,
                       time_bin=1, demand_scale=1.0):
    if not use_dataset:
        return DemandModel()
    if not dataset_path:
        dataset_path = _download_dataset_kagglehub()
    if not dataset_path:
        print("[WARN] Dataset unavailable, using synthetic demand.")
        return DemandModel()
    key = (dataset_path, max_files, max_rows, time_bin, demand_scale)
    if _DEMAND_MODEL_CACHE["key"] == key:
        return _DEMAND_MODEL_CACHE["model"]
    try:
        model = DatasetDemandModel(dataset_path, max_files=max_files, max_rows=max_rows,
                                   time_bin=time_bin, demand_scale=demand_scale)
    except Exception as exc:
        print(f"[WARN] Dataset load failed: {exc}")
        return DemandModel()
    _DEMAND_MODEL_CACHE["key"] = key
    _DEMAND_MODEL_CACHE["model"] = model
    return model

def apply_runtime_config(cfg=None):
    global CONGESTION_ENABLED, USE_TIME_FEATURES
    global RIDER_CHANGE_PROB, RIDER_CANCEL_PROB
    global OSRM_ENABLED, OSRM_BASE_URL, OSRM_TIMEOUT_SEC, OSRM_CACHE_SIZE
    global OSRM_PROFILE
    global USE_DATASET, DATASET_PATH, DATASET_MAX_FILES, DATASET_MAX_ROWS
    global DATASET_TIME_BIN, DATASET_DEMAND_SCALE, DEMAND_MODEL
    cfg = cfg or {}
    if "enable_congestion" in cfg:
        CONGESTION_ENABLED = bool(cfg["enable_congestion"])
    if "use_time_features" in cfg:
        USE_TIME_FEATURES = bool(cfg["use_time_features"])
    if "enable_osrm" in cfg:
        OSRM_ENABLED = bool(cfg["enable_osrm"])
    if "osrm_base_url" in cfg and cfg["osrm_base_url"]:
        OSRM_BASE_URL = str(cfg["osrm_base_url"]).rstrip("/")
    if "osrm_profile" in cfg and cfg["osrm_profile"]:
        OSRM_PROFILE = str(cfg["osrm_profile"])
    if "osrm_timeout_sec" in cfg:
        OSRM_TIMEOUT_SEC = float(cfg["osrm_timeout_sec"])
    if "osrm_cache_size" in cfg:
        OSRM_CACHE_SIZE = int(cfg["osrm_cache_size"])
        while len(_osrm_cache) > OSRM_CACHE_SIZE:
            _osrm_cache.popitem(last=False)
    if "rider_change_prob" in cfg:
        RIDER_CHANGE_PROB = float(cfg["rider_change_prob"])
    if "rider_cancel_prob" in cfg:
        RIDER_CANCEL_PROB = float(cfg["rider_cancel_prob"])
    if "use_dataset" in cfg:
        USE_DATASET = bool(cfg["use_dataset"])
    if "dataset_path" in cfg:
        DATASET_PATH = cfg["dataset_path"] or ""
    if "dataset_max_files" in cfg:
        DATASET_MAX_FILES = int(cfg["dataset_max_files"])
    if "dataset_max_rows" in cfg:
        DATASET_MAX_ROWS = int(cfg["dataset_max_rows"])
    if "dataset_time_bin" in cfg:
        DATASET_TIME_BIN = int(cfg["dataset_time_bin"])
    if "dataset_demand_scale" in cfg:
        DATASET_DEMAND_SCALE = float(cfg["dataset_demand_scale"])
    DEMAND_MODEL = build_demand_model(
        USE_DATASET, DATASET_PATH, DATASET_MAX_FILES, DATASET_MAX_ROWS,
        DATASET_TIME_BIN, DATASET_DEMAND_SCALE
    )

DEMAND_MODEL = DemandModel()

def _metrics_to_json(metrics):
    out = {}
    for k, v in metrics.items():
        out[k] = [float(x) for x in v]
    return out

def export_metrics(out_dir, train_metrics, demo_metrics, config):
    bundle = {
        "train": _metrics_to_json(train_metrics),
        "demo": _metrics_to_json(demo_metrics),
        "config": config,
    }
    json_path = os.path.join(out_dir, "metrics_bundle.json")
    js_path = os.path.join(out_dir, "metrics_bundle.js")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("window.__METRICS_BUNDLE__ = ")
        json.dump(bundle, f)
        f.write(";")
    return json_path, js_path

# ═══════════════════════════════════════════════════════════════════════════
# 4. Data classes
# ═══════════════════════════════════════════════════════════════════════════
_rid = 0
@dataclass
class Req:
    o: int; d: int; np: int; t: int; rid: int = 0
    max_wait: float = field(default_factory=lambda: random.uniform(5,20))
    share: bool     = field(default_factory=lambda: random.random()>0.25)
    pref_type: int  = field(default_factory=lambda: random.randint(0,N_TYPES-1))
    delta: float    = field(default_factory=lambda: random.uniform(3.0,15.0))
    def __post_init__(self):
        global _rid; _rid+=1; self.rid = _rid

@dataclass
class Veh:
    vid: int; vt: int; zone: int; t_enter: int = 0
    cap: int = 0
    route: List[Tuple] = field(default_factory=list)  # [('pu'|'do', Req)]
    pax: List = field(default_factory=list)
    profit: float = 0.; km: float = 0.
    trips: int = 0; occ_steps: int = 0; total_steps: int = 0
    idle_since: int = 0
    dispatching: bool = False

    @property
    def maxcap(self): return MAX_CAP[self.vt]
    @property
    def free(self): return self.maxcap - self.cap
    @property
    def avail(self): return self.free > 0 and not self.dispatching

# ═══════════════════════════════════════════════════════════════════════════
# 5. Greedy Matcher (Algorithm 2)
# ═══════════════════════════════════════════════════════════════════════════
def greedy_match(vehicles, reqs):
    """Assign each request to nearest available vehicle within REJECT_RADIUS_KM.
    Returns dict {vid: [Req, ...]}.  Does NOT modify vehicle state."""
    asgn = {v.vid: [] for v in vehicles}
    temp_cap = {v.vid: v.cap for v in vehicles}
    for req in reqs:
        best_v, best_d = None, float('inf')
        for v in vehicles:
            free = v.maxcap - temp_cap[v.vid]
            if not v.avail or free < req.np:
                continue
            d = dist_km(v.zone, req.o)
            if d <= REJECT_RADIUS_KM and d < best_d:
                best_d, best_v = d, v
        if best_v is not None:
            asgn[best_v.vid].append(req)
            temp_cap[best_v.vid] += req.np
    return asgn

# ═══════════════════════════════════════════════════════════════════════════
# 6. Insertion-Based Route Planner (Algorithm 3)
# ═══════════════════════════════════════════════════════════════════════════
def route_waypoints(v_zone, route):
    wps = [v_zone]
    for tag, r in route:
        if tag == 'mv':
            wps.append(int(r))
        else:
            wps.append(r.o if tag=='pu' else r.d)
    return wps

def route_times(v_zone, route, t=0):
    """Arrival times at each pickup/dropoff in minutes."""
    wps = route_waypoints(v_zone, route)
    arr = [0.0]
    for i in range(len(wps)-1):
        arr.append(arr[-1] + travel_time_min(wps[i], wps[i+1], t))
    pickup_t, drop_t = {}, {}
    wp_idx = 1  # wps[0] is v_zone (current position)
    for tag, req in route:
        if tag == 'mv':
            wp_idx += 1
            continue
        t = arr[wp_idx]
        if tag == 'pu':
            pickup_t[req.rid] = t
        else:
            drop_t[req.rid] = t
        wp_idx += 1
    return pickup_t, drop_t

def route_feasible(v, route, new_req, t=0):
    """Check delay and detour constraints for a candidate route."""
    pickup_t, drop_t = route_times(v.zone, route, t)
    req_map = {}
    for tag, r in route:
        if tag != 'mv':
            req_map[r.rid] = r
    for r in v.pax:
        req_map[r.rid] = r

    for rid, r in req_map.items():
        if rid not in drop_t:
            return False, None
        if rid in pickup_t:
            trip_t = drop_t[rid] - pickup_t[rid]
            direct_t = travel_time_min(r.o, r.d, t)
        else:
            trip_t = drop_t[rid]
            direct_t = travel_time_min(v.zone, r.d, t)
        if trip_t > direct_t * MAX_DETOUR_FACTOR:
            return False, None

    wait_t = pickup_t.get(new_req.rid, 0.0)
    if wait_t > new_req.max_wait:
        return False, None
    return True, wait_t

def insert_req(v, req, t=0):
    """Try inserting req into v.route.
    Returns (route_km, extra_km, new_route, wait_min) or (inf, inf, None, None)."""
    n = len(v.route)
    base_wps  = route_waypoints(v.zone, v.route)
    base_cost = route_cost(base_wps)
    best_extra, best_route = float('inf'), None
    best_cost, best_wait = float('inf'), None

    for i_o in range(n+1):
        for i_d in range(i_o+1, n+2):
            nr = list(v.route)
            nr.insert(i_o, ('pu', req))
            nr.insert(i_d, ('do', req))

            # Capacity check: simulate capacity evolution
            cap = v.cap   # current (undo already done before this call)
            ok = True
            for tag, r in nr:
                if tag == 'mv':
                    continue
                cap += (r.np if tag=='pu' else -r.np)
                if cap > v.maxcap or cap < 0:
                    ok = False; break
            if not ok: continue

            wps  = route_waypoints(v.zone, nr)
            route_km = route_cost(wps)
            extra_km = route_km - base_cost

            feasible, wait_t = route_feasible(v, nr, req, t)
            if not feasible:
                continue

            if extra_km < best_extra:
                best_extra, best_route = extra_km, nr
                best_cost, best_wait = route_km, wait_t

    if best_route is None:
        return float('inf'), float('inf'), None, None
    return best_cost, best_extra, best_route, best_wait

# ═══════════════════════════════════════════════════════════════════════════
# 7. Pricing (Equations 2 & 3)
# ═══════════════════════════════════════════════════════════════════════════
def price_initial(v, req, cost_km, wait_min):
    """Equation (2)."""
    shared = max(1, v.cap + req.np)
    fuel   = (cost_km/shared) * (PGAS/MILEAGE_L[v.vt])
    p      = BASE_FARE[v.vt] + RATE_KM[v.vt]*(cost_km/shared) + fuel \
             - RATE_WAIT[v.vt]*wait_min
    return max(BASE_FARE[v.vt], p)

def price_driver(v, req, p_init, hotspot_zones, zone_rank):
    """Equation (3) -- driver adjusts price based on destination zone Q-value.
    If destination is a hotspot, driver keeps base price (willing to go there).
    Otherwise, driver adds a small markup proportional to how undesirable the zone is.
    Paper: markup should be small enough that customer rejection stays ~5%.
    """
    if req.d in hotspot_zones:
        return p_init
    alpha_rank = zone_rank.get(req.d, G.n-1)
    alpha = alpha_rank / max(1, (G.n - 1))  # 0=best zone, 1=worst zone
    # Gentle markup: at most ~30% of initial price for the worst zones
    markup = p_init * alpha * 0.30
    return p_init + markup

# ═══════════════════════════════════════════════════════════════════════════
# 8. Customer Decision (Equations 4 & 5)
# ═══════════════════════════════════════════════════════════════════════════
def customer_decide(req, v, price, wait_min):
    """Return True if customer accepts (Eq. 4 & 5)."""
    # Sharing preference: customer who doesn't want pooling rejects if vehicle has pax
    if not req.share and len(v.pax) > 0: return False
    # Vehicle type is a soft preference (adds to utility), NOT a hard filter
    # Paper: type score contributes to utility, not a rejection gate
    type_bonus = 1.0 if v.vt >= req.pref_type else 0.5
    # Utility (Eq. 4)
    u = (W4 / max(1, v.cap+1) + W5 / max(1., wait_min) +
        W6 * type_bonus * (v.vt+1))
    # Accept (Eq. 5)
    return u >= (price - req.delta)

# ═══════════════════════════════════════════════════════════════════════════
# 9. DQN (Double DQN with numpy fallback)
# ═══════════════════════════════════════════════════════════════════════════
class ReplayBuf:
    def __init__(self, cap): self.buf = collections.deque(maxlen=cap)
    def push(self, *t):      self.buf.append(t)
    def sample(self, n):
        b = random.sample(self.buf, n)
        s,a,r,ns,d = zip(*b)
        return (np.array(s,np.float32), np.array(a,np.int64),
                np.array(r,np.float32), np.array(ns,np.float32),
                np.array(d,np.float32))
    def __len__(self): return len(self.buf)

if TORCH:
    class QNet(nn.Module):
        def __init__(self, sd, ad):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(sd,256), nn.ReLU(),
                nn.Linear(256,256), nn.ReLU(),
                nn.Linear(256,128), nn.ReLU(),
                nn.Linear(128, ad))
        def forward(self, x): return self.net(x)

@dataclass
class AgentCtx:
    eps: float = EPS0
    last_state: Optional[np.ndarray] = None
    last_action: Optional[int] = None

def _state_dim():
    dim = G.n + (FORECAST_H+1) * 2 * G.n + 1 + N_TYPES
    if USE_TIME_FEATURES:
        dim += 2
    return dim

class DQNAgent:
    """Multi-agent DQN dispatcher with shared policy and per-agent epsilon."""

    def __init__(
        self,
        n_agents,
        lr=LR,
        gamma=GAMMA,
        batch_size=BATCH,
        replay_size=BUF,
        min_buf=MIN_BUF,
        eps_start=EPS0,
        eps_end=EPS_MIN,
        eps_decay=EPS_DEC,
        target_update=TGT_UPD,
    ):
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.min_buf = int(min_buf)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)
        self.target_update = int(target_update)

        self.state_dim = _state_dim()
        self.ctx = {i: AgentCtx(eps=self.eps_start) for i in range(n_agents)}
        self.buf = ReplayBuf(replay_size)
        self.steps = 0
        self.zone_q = np.zeros(G.n)  # Q-value per destination zone (for pricing)

        if TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.online = QNet(self.state_dim, ACTION_DIM)
            self.target = QNet(self.state_dim, ACTION_DIM)
            self.online.to(self.device)
            self.target.to(self.device)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
            self.opt = optim.Adam(self.online.parameters(), lr=self.lr)
        else:
            self.qtab = np.zeros((G.n, ACTION_DIM))

    def avg_eps(self):
        return float(np.mean([c.eps for c in self.ctx.values()]))

    def state(self, v, supply_f, demand_f, t_min):
        s = np.zeros(self.state_dim, np.float32)
        idx = 0
        s[idx + v.zone] = 1.0
        idx += G.n
        for h in range(FORECAST_H+1):
            sf = supply_f[h]
            df = demand_f[h]
            s[idx:idx+G.n] = sf / max(1., sf.max())
            idx += G.n
            s[idx:idx+G.n] = df / max(1., df.max())
            idx += G.n
        s[idx] = v.cap / max(1., v.maxcap); idx += 1
        s[idx + v.vt] = 1.0; idx += N_TYPES
        if USE_TIME_FEATURES:
            hr = (t_min // 60) % 24
            s[idx] = math.sin(2*math.pi*hr/24)
            s[idx+1] = math.cos(2*math.pi*hr/24)
        return s

    def act(self, v, state, explore=True, track=True):
        ctx = self.ctx[v.vid]
        if explore and random.random() < ctx.eps:
            a = random.randrange(ACTION_DIM)
        else:
            if TORCH:
                with torch.no_grad():
                    s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    a = int(self.online(s).argmax(dim=1).item())
            else:
                a = int(self.qtab[v.zone].argmax())
        if track:
            ctx.last_state = state
            ctx.last_action = a
        return a

    def observe(self, vid, reward, next_state, done=False):
        ctx = self.ctx[vid]
        if ctx.last_state is None or ctx.last_action is None:
            return
        self.buf.push(ctx.last_state, ctx.last_action, reward, next_state, done)
        ctx.last_state = None
        ctx.last_action = None
        ctx.eps = max(self.eps_end, ctx.eps * self.eps_decay)

    def update_zone_q(self, supply_f, demand_f, t_min):
        """Compute per-zone max-Q (used by pricing)."""
        if TORCH:
            ss = np.zeros((G.n, self.state_dim), np.float32)
            ss[:, :G.n] = np.eye(G.n, dtype=np.float32)
            idx = G.n
            for h in range(FORECAST_H+1):
                sf = supply_f[h]; df = demand_f[h]
                ss[:, idx:idx+G.n] = sf / max(1., sf.max())
                idx += G.n
                ss[:, idx:idx+G.n] = df / max(1., df.max())
                idx += G.n
            ss[:, idx] = 0.0; idx += 1
            ss[:, idx] = 1.0; idx += N_TYPES  # type-0 baseline
            if USE_TIME_FEATURES:
                hr = (t_min // 60) % 24
                ss[:, idx] = math.sin(2*math.pi*hr/24)
                ss[:, idx+1] = math.cos(2*math.pi*hr/24)
            with torch.no_grad():
                q = self.online(torch.as_tensor(ss, dtype=torch.float32, device=self.device))
                self.zone_q = q.max(1).values.detach().cpu().numpy()
        else:
            self.zone_q = self.qtab.max(axis=1)

    def learn(self):
        if len(self.buf) < self.min_buf:
            return 0.0
        s,a,r,ns,d = self.buf.sample(self.batch_size)
        if TORCH:
            s  = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            a  = torch.as_tensor(a, dtype=torch.int64, device=self.device)
            r  = torch.as_tensor(r, dtype=torch.float32, device=self.device)
            ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
            d  = torch.as_tensor(d, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                na  = self.online(ns).argmax(1)
                q_t = self.target(ns).gather(1,na.unsqueeze(1)).squeeze()
                y   = r + self.gamma*q_t*(1-d)
            qp = self.online(s).gather(1,a.unsqueeze(1)).squeeze()
            loss = F.mse_loss(qp, y)
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), 1.)
            self.opt.step()
            self.steps += 1
            if self.steps % self.target_update == 0:
                self.target.load_state_dict(self.online.state_dict())
            return float(loss.item())
        else:
            for i in range(len(s)):
                zi = int(s[i,:G.n].argmax())
                nzi= int(ns[i,:G.n].argmax())
                td = r[i] + self.gamma*self.qtab[nzi].max()*(1-d[i])
                self.qtab[zi,a[i]] += self.lr*(td - self.qtab[zi,a[i]])
            return 0.0

    def configure(
        self,
        lr=None,
        gamma=None,
        batch_size=None,
        replay_size=None,
        min_buf=None,
        eps_start=None,
        eps_end=None,
        eps_decay=None,
        target_update=None,
        reset_eps=False,
        reset_buf=False,
    ):
        if lr is not None:
            self.lr = float(lr)
            if TORCH:
                for pg in self.opt.param_groups:
                    pg["lr"] = self.lr
        if gamma is not None:
            self.gamma = float(gamma)
        if batch_size is not None:
            self.batch_size = int(batch_size)
        if min_buf is not None:
            self.min_buf = int(min_buf)
        if eps_start is not None:
            self.eps_start = float(eps_start)
        if eps_end is not None:
            self.eps_end = float(eps_end)
        if eps_decay is not None:
            self.eps_decay = float(eps_decay)
        if target_update is not None:
            self.target_update = int(target_update)
        if replay_size is not None and reset_buf:
            self.buf = ReplayBuf(int(replay_size))
        if reset_eps:
            for ctx in self.ctx.values():
                ctx.eps = self.eps_start

    def save(self, path):
        if TORCH:
            payload = {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "steps": self.steps,
            }
            torch.save(payload, path)
        else:
            np.save(path, self.qtab)

    def load(self, path):
        if TORCH:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                # weights_only param not supported in older PyTorch
                payload = torch.load(path, map_location="cpu")
            self.online.load_state_dict(payload["online"])
            self.target.load_state_dict(payload["target"])
            self.steps = int(payload.get("steps", 0))
        else:
            self.qtab = np.load(path)

    def clone_for_eval(self):
        maxlen = getattr(self.buf.buf, "maxlen", BUF)
        clone = DQNAgent(
            len(self.ctx),
            lr=self.lr,
            gamma=self.gamma,
            batch_size=self.batch_size,
            replay_size=maxlen,
            min_buf=self.min_buf,
            eps_start=0.0,
            eps_end=0.0,
            eps_decay=1.0,
            target_update=self.target_update,
        )
        if TORCH:
            clone.online.load_state_dict(self.online.state_dict())
            clone.target.load_state_dict(self.target.state_dict())
            clone.online.eval()
            clone.target.eval()
        else:
            clone.qtab = np.array(self.qtab, copy=True)
        clone.steps = self.steps
        for ctx in clone.ctx.values():
            ctx.eps = 0.0
        return clone

def run_eval(agent, algo_cfg=None, steps=EVAL_STEPS, seed=EVAL_SEED):
    """Run evaluation with fixed seed, no exploration, and no learning."""
    global _rid
    py_state = random.getstate()
    np_state = np.random.get_state()
    rid_state = _rid
    try:
        random.seed(seed)
        np.random.seed(seed)
        eval_agent = agent.clone_for_eval() if hasattr(agent, "clone_for_eval") else agent
        env = RSEnv(cfg=algo_cfg, agent=eval_agent, train_mode=False)
        metrics = collections.defaultdict(list)
        for _ in range(int(steps)):
            m = env.step()
            for k, v in m.items():
                metrics[k].append(v)
        return env, metrics
    finally:
        _rid = rid_state
        random.setstate(py_state)
        np.random.set_state(np_state)

# ═══════════════════════════════════════════════════════════════════════════
# 10. Master Environment (Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════════
class RSEnv:
    def __init__(self, cfg=None, agent=None, agent_cfg=None, train_mode=True):
        self.cfg = resolve_algo(cfg)
        self.train_mode = bool(train_mode)
        if agent is None:
            agent_cfg = agent_cfg or {}
            self.agent = DQNAgent(N_VEHICLES, **agent_cfg)
        else:
            self.agent = agent
        self.t      = 0
        self.supply = np.zeros(G.n, np.float32)
        self.demand = np.zeros(G.n, np.float32)
        self.demand_model = DEMAND_MODEL
        self.pending: List[Req] = []
        self.vehicles: List[Veh] = []
        self.metrics  = collections.defaultdict(list)
        self.total_req = 0
        self.served_req = 0
        self.rejected_req = 0
        self.events = collections.deque(maxlen=200)
        self.event_id = 0
        self.model_path = ""
        self.warmup_done = False
        self.qmax_history = []   # Track Q-max convergence (paper Fig.6)
        self._spawn_fleet()

    def _spawn_fleet(self):
        for i in range(N_VEHICLES):
            self.vehicles.append(Veh(vid=i, vt=random.randint(0,N_TYPES-1),
                                     zone=G.rand()))

    def _forecast_supply(self, horizon):
        out = np.zeros((horizon+1, G.n), np.float32)
        for v in self.vehicles:
            zones = [v.zone]
            caps = [v.cap]
            for tag, req in v.route:
                if tag == 'mv':
                    zones.append(int(req))
                    caps.append(caps[-1])
                else:
                    zones.append(req.o if tag=='pu' else req.d)
                    caps.append(caps[-1] + (req.np if tag=='pu' else -req.np))
            for h in range(horizon+1):
                idx = min(h, len(zones)-1)
                z = zones[idx]
                cap = caps[idx]
                if cap < v.maxcap and not (v.dispatching and h == 0):
                    out[h, z] += 1
        return out

    def _forecast_demand(self, t_min, horizon):
        out = np.zeros((horizon+1, G.n), np.float32)
        for h in range(horizon+1):
            out[h] = self.demand_model.future(t_min + h, horizon=1).astype(np.float32)
        return out

    def _upd_supply_demand(self):
        self.supply = self._forecast_supply(0)[0]
        self.demand = self._forecast_demand(self.t, 0)[0]

    def _hotspots(self):
        top  = max(1, int(HOTSPOT_FRAC*G.n))
        rank = np.argsort(-self.agent.zone_q)
        return list(rank[:top]), {int(z): int(r) for r,z in enumerate(rank)}

    def _dispatch_idle(self, supply_f, demand_f):
        dispatch_time = collections.defaultdict(float)
        for v in self.vehicles:
            if v.route:
                continue
            idle = self.t - v.idle_since
            if idle < MAX_IDLE_MIN and v.trips > 0:
                continue
            state = self.agent.state(v, supply_f, demand_f, self.t)
            act  = self.agent.act(v, state, explore=self.train_mode, track=self.train_mode)
            nz   = G.action_to_zone(v.zone, act)
            dispatch_time[v.vid] += travel_time_min(v.zone, nz, self.t)
            if nz != v.zone:
                v.route = [('mv', nz)]
                v.dispatching = True
        return dispatch_time

    def _advance(self):
        """Advance each vehicle by one stop and collect per-vehicle stats.
        Extension: riders may change destination or cancel mid-trip."""
        completed = 0
        step = {v.vid: dict(served=0, km=0.0, fuel=0.0) for v in self.vehicles}

        # --- On-the-fly rider changes (optional extension) ---
        if RIDER_CHANGE_PROB > 0.0 or RIDER_CANCEL_PROB > 0.0:
            for v in self.vehicles:
                if not v.pax:
                    continue
                for req in list(v.pax):
                    # Rider cancellation mid-trip
                    if RIDER_CANCEL_PROB > 0.0 and random.random() < RIDER_CANCEL_PROB:
                        v.pax.remove(req)
                        v.cap = max(0, v.cap - req.np)
                        # Remove dropoff from route
                        v.route = [(t,r) for t,r in v.route if not (t=='do' and r is req)]
                        # Partial refund
                        v.profit -= 0.5 * (BASE_FARE[v.vt] + RATE_KM[v.vt] * dist_km(req.o, req.d))
                        self.event_id += 1
                        self.events.append({"id": self.event_id, "t": self.t,
                            "type": "cancel", "msg": f"R{req.rid} cancelled mid-trip on V{v.vid}"})
                        continue
                    # Rider destination change
                    if RIDER_CHANGE_PROB > 0.0 and random.random() < RIDER_CHANGE_PROB:
                        old_d = req.d
                        new_d = G.rand()
                        while new_d == old_d or new_d == v.zone:
                            new_d = G.rand()
                        req.d = new_d
                        # Update route: find dropoff for this req and change target
                        # (keeps route order, just changes destination zone)
                        self.event_id += 1
                        self.events.append({"id": self.event_id, "t": self.t,
                            "type": "change", "msg": f"R{req.rid} changed dest on V{v.vid}"})

        for v in self.vehicles:
            v.total_steps += 1
            if v.pax:
                v.occ_steps += 1
            if not v.route:
                if not v.pax:
                    v.idle_since = self.t
                continue
            # Pop ONE waypoint (one stop per timestep)
            tag, req = v.route.pop(0)
            target = int(req) if tag == 'mv' else (req.o if tag=='pu' else req.d)
            # Apply congestion to travel time (affects km tracking)
            km = dist_km(v.zone, target)
            cf = congestion_factor(v.zone, self.t)
            effective_km = km * cf  # congestion increases effective travel cost
            v.km += km
            v.zone = target
            step[v.vid]['km'] += km
            fuel = effective_km * (PGAS / MILEAGE_L[v.vt])  # fuel based on congestion
            step[v.vid]['fuel'] += fuel
            v.profit -= fuel

            if tag == 'mv':
                v.dispatching = False
                continue

            if tag == 'pu':
                v.pax.append(req)
                v.cap += req.np
            else:
                if req in v.pax:
                    v.pax.remove(req)
                v.cap -= req.np
                v.cap = max(0, v.cap)
                v.trips += 1
                completed += 1
                step[v.vid]['served'] += req.np
            if not v.route and not v.pax:
                v.idle_since = self.t
        return completed, step

    def step(self):
        self.t += 1
        use_dispatch = bool(self.cfg.get("dispatch", True))
        use_pricing = bool(self.cfg.get("pricing", True))
        use_darm = bool(self.cfg.get("darm", True))
        allow_rideshare = bool(self.cfg.get("rideshare", True))

        self.log_event(f"Step {self.t}: Starting simulation cycle", "sys")

        supply_f = self._forecast_supply(FORECAST_H)
        demand_f = self._forecast_demand(self.t, FORECAST_H)
        self.supply = supply_f[0]
        self.demand = demand_f[0]
        if use_dispatch or use_pricing:
            self.agent.update_zone_q(supply_f, demand_f, self.t)
        hot, rank = self._hotspots() if use_pricing else ([], {})

        # 1. New requests + expire stale pending (timeout after 10 steps)
        raw   = self.demand_model.generate(self.t)
        reqs  = [Req(o,d,n,self.t) for o,d,n in raw]
        fresh_pending = [r for r in self.pending if (self.t - r.t) <= 10]
        expired = len(self.pending) - len(fresh_pending)
        self.rejected_req += expired
        all_r = fresh_pending + reqs
        self.pending = []
        self.log_event(f"Generated {len(reqs)} requests, {expired} expired. Total pool: {len(all_r)}", "sys")

        # 2. Greedy match
        avail = [v for v in self.vehicles if v.avail and (allow_rideshare or not v.pax)]
        asgn  = greedy_match(avail, all_r)
        self.log_event(f"Greedy match: assigned {sum(len(r) for r in asgn.values())} requests to {len(avail)} available vehicles", "sys")

        acc=0; rej=0; wait_sum=0.0
        step_profit = {v.vid: 0.0 for v in self.vehicles}
        extra_km = {v.vid: 0.0 for v in self.vehicles}
        self.total_req += len(all_r)

        # 4. Per-vehicle: insertion + pricing + customer decision
        for v in self.vehicles:
            assigned = asgn.get(v.vid, [])
            if not assigned:
                continue
            if not allow_rideshare or not use_darm:
                assigned = assigned[:1]
            assigned.sort(key=lambda r: dist_km(v.zone, r.o))

            for req in assigned:
                if use_darm:
                    route_km, extra, new_route, wait = insert_req(v, req, self.t)
                    if new_route is None:
                        self.pending.append(req); rej+=1; continue
                else:
                    new_route = [('pu', req), ('do', req)]
                    route_km = dist_km(v.zone, req.o) + dist_km(req.o, req.d)
                    extra = route_km
                    wait = travel_time_min(v.zone, req.o, self.t)

                if use_pricing:
                    trip_km = dist_km(req.o, req.d)
                    p0   = price_initial(v, req, trip_km, wait)
                    p    = price_driver(v, req, p0, hot, rank)
                    ok   = customer_decide(req, v, p, wait)
                else:
                    p = BASE_FARE[v.vt] + RATE_KM[v.vt] * dist_km(req.o, req.d)
                    ok = True

                if ok:
                    v.route  = new_route
                    v.profit += p
                    step_profit[v.vid] += p
                    extra_km[v.vid] += extra
                    acc += 1; wait_sum += wait
                    self.log_event(f"V{v.vid} accepted R{req.rid} @ ${p:.2f} (Wait: {wait:.1f}m)", "accept")
                else:
                    self.pending.append(req); rej += 1
                    self.log_event(f"R{req.rid} rejected by V{v.vid} (Price: ${p:.2f})", "reject")

        # 3. Dispatch idle vehicles (after matching, per paper)
        supply_f = self._forecast_supply(FORECAST_H)
        demand_f = self._forecast_demand(self.t, FORECAST_H)
        if self.t <= WARMUP_STEPS or not use_dispatch:
            dispatch_km = collections.defaultdict(float)
        else:
            self.warmup_done = True
            dispatch_km = self._dispatch_idle(supply_f, demand_f)
            num_disp = sum(1 for v in self.vehicles if v.dispatching)
            self.log_event(f"DQN Dispatch: {num_disp} vehicles re-positioned", "dqn")

        # 5. Advance vehicles
        _, step_move = self._advance()

        # 6. DQN observe + learn
        next_supply_f = self._forecast_supply(FORECAST_H)
        next_demand_f = self._forecast_demand(self.t + 1, FORECAST_H)
        profit_step_vals = []
        for v in self.vehicles:
            served = step_move[v.vid]['served']
            dispatch_time = dispatch_km.get(v.vid, 0.0)
            detour_time = extra_km[v.vid] / AVG_SPEED
            profit_step = step_profit[v.vid] - step_move[v.vid]['fuel']
            profit_step_vals.append(profit_step)
            idle_flag = 1.0 if (not v.route and not v.pax) else 0.0
            if self.train_mode and use_dispatch:
                reward = (B1 * served + B2 * dispatch_time + B3 * detour_time +
                          B4 * profit_step + B5 * idle_flag)
                ns = self.agent.state(v, next_supply_f, next_demand_f, self.t + 1)
                self.agent.observe(v.vid, float(reward), ns, False)
        
        self.supply = next_supply_f[0]
        self.demand = next_demand_f[0]
        loss = self.agent.learn() if (self.train_mode and use_dispatch) else 0.0
        if self.train_mode and use_dispatch:
            self.log_event(f"Learning: Loss={loss:.4f}, ε={self.agent.avg_eps():.3f}", "dqn")

        # Track Q-max convergence (paper Fig. 6)
        qmax_avg = float(self.agent.zone_q.max()) if self.agent.zone_q.max() > 0 else 0.0
        self.qmax_history.append(qmax_avg)

        n_req = len(all_r)
        ar    = acc / max(1, n_req)
        idle  = sum(1 for v in self.vehicles if not v.route)
        idle_frac = sum(1 for v in self.vehicles if not v.pax) / max(1, len(self.vehicles))
        profit_step_avg = float(np.mean(profit_step_vals)) if profit_step_vals else 0.0
        profit_total_avg = float(np.mean([v.profit for v in self.vehicles]))
        awt   = wait_sum / max(1, acc)
        occ   = float(np.mean([v.occ_steps/max(1,v.total_steps)
                                for v in self.vehicles]))
        akm   = float(np.mean([v.km for v in self.vehicles]))
        # Travel distance per hour (paper metric)
        hrs = max(1, self.t) / 60.0
        km_hr = float(np.mean([v.km / max(0.01, hrs) for v in self.vehicles]))

        m = dict(t=self.t, n_req=n_req, acc=acc, rej=rej,
             ar=ar, idle=idle, idle_frac=idle_frac, profit=profit_step_avg, profit_total=profit_total_avg,
                 wait=awt, occ=occ, km=akm, km_hr=km_hr,
                 loss=float(loss), eps=self.agent.avg_eps(),
             qmax=qmax_avg, buf=len(self.agent.buf))
        for k,val in m.items(): self.metrics[k].append(val)
        self.served_req += acc
        self.rejected_req += rej
        return m

    def log_event(self, msg, etype="info"):
        self.event_id += 1
        self.events.append({
            "id": self.event_id,
            "t": self.t,
            "type": etype,
            "msg": msg,
        })
        # Keep only last 1000 events to avoid memory blowup
        if len(self.events) > 1000:
            self.events = self.events[-1000:]

        demand = [float(x) for x in self.demand]
        zone_q = [float(x) for x in self.agent.zone_q]
        pending = []
        for r in self.pending:
            ro, co = G.rc(r.o)
            rd, cd = G.rc(r.d)
            pending.append({
                "id": r.rid,
                "o": r.o,
                "d": r.d,
                "r": ro,
                "c": co,
                "dr": rd,
                "dc": cd,
                "np": r.np,
            })
        vehicles = []
        for v in self.vehicles:
            r, c = G.rc(v.zone)
            if v.dispatching:
                status = "dispatching"
            elif v.pax:
                status = "occupied"
            elif v.route:
                status = "enroute"
            else:
                status = "idle"
            route = []
            for tag, req in v.route:
                if tag == "mv":
                    rz, cz = G.rc(int(req))
                    route.append({"r": rz, "c": cz, "type": "move"})
                elif tag == "pu":
                    rz, cz = G.rc(req.o)
                    route.append({"r": rz, "c": cz, "type": "pickup"})
                else:
                    rz, cz = G.rc(req.d)
                    route.append({"r": rz, "c": cz, "type": "dropoff"})
            vehicles.append({
                "id": v.vid,
                "vt": v.vt,
                "r": r,
                "c": c,
                "zone": v.zone,
                "status": status,
                "passengers": len(v.pax),
                "maxcap": v.maxcap,
                "profit": float(v.profit),
                "route": route,
            })
        return {
            "t": self.t,
            "grid": {"w": GRID_W, "h": GRID_H},
            "demand": demand,
            "zone_q": zone_q,
            "pending": pending,
            "vehicles": vehicles,
            "events": list(self.events),
            "model": {"path": self.model_path},
            "metrics": metrics,
            "totals": {
                "requests": int(self.total_req),
                "served": int(self.served_req),
                "rejected": int(self.rejected_req),
            },
        }

# ═══════════════════════════════════════════════════════════════════════════
# 11. Configurable Baseline (Paper Section VI-D)
#     Supports all 6 baselines by toggling flags:
#       1. (!D, !RS, !PS, GM)  2. (!D, RS, !PS, GM)
#       3. (D, !RS, !PS, GM)   4. (D, RS, !PS, GM)
#       5. (D, RS, PS, GM)     6. (D, RS, PS, DARM) ← full framework
# ═══════════════════════════════════════════════════════════════════════════
BASELINE_CONFIGS = {
    "!D,!RS,!PS,GM": dict(dispatch=False, rideshare=False, pricing=False, darm=False),
    "!D,RS,!PS,GM":  dict(dispatch=False, rideshare=True,  pricing=False, darm=False),
    "D,!RS,!PS,GM":  dict(dispatch=True,  rideshare=False, pricing=False, darm=False),
    "D,RS,!PS,GM":   dict(dispatch=True,  rideshare=True,  pricing=False, darm=False),
    "D,RS,PS,GM":    dict(dispatch=True,  rideshare=True,  pricing=True,  darm=False),
    "D,RS,PS,DARM":  dict(dispatch=True,  rideshare=True,  pricing=True,  darm=True),
}

class BaselineEnv:
    """Simplified env for running baselines without full DQN training."""
    def __init__(self, cfg_name, agent=None, demand_model=None):
        self.cfg = BASELINE_CONFIGS[cfg_name]
        self.name = cfg_name
        self.t = 0
        self.vehicles = [Veh(vid=i, vt=random.randint(0,N_TYPES-1),
                             zone=G.rand()) for i in range(N_VEHICLES)]
        self.pending = []
        self.metrics = collections.defaultdict(list)
        self.agent = agent  # shared trained DQN agent (for dispatch baselines)
        self.demand_model = demand_model or DEMAND_MODEL

    def step(self):
        self.t += 1
        raw  = self.demand_model.generate(self.t)
        reqs = [Req(o,d,n,self.t) for o,d,n in raw] + self.pending
        self.pending = []; acc = 0; rej = 0; wait_sum = 0.0

        # Greedy match
        avail = [v for v in self.vehicles if v.free > 0]
        asgn = greedy_match(avail, reqs)

        hot_zones, zone_rank = [], {}
        if self.cfg['pricing'] and self.agent:
            self.agent.update_zone_q(
                np.zeros((FORECAST_H+1, G.n), np.float32),
                np.zeros((FORECAST_H+1, G.n), np.float32), self.t)
            top = max(1, int(HOTSPOT_FRAC * G.n))
            rank = np.argsort(-self.agent.zone_q)
            hot_zones = list(rank[:top])
            zone_rank = {int(z): int(r) for r, z in enumerate(rank)}

        for v in self.vehicles:
            assigned = asgn.get(v.vid, [])
            if not assigned: continue

            if not self.cfg['rideshare']:
                assigned = assigned[:1]  # only one request

            for req in assigned:
                if self.cfg['darm']:
                    route_km, extra, new_route, wait = insert_req(v, req, self.t)
                    if new_route is None:
                        self.pending.append(req); rej += 1; continue
                else:
                    # Simple direct route
                    new_route = [('pu', req), ('do', req)]
                    route_km = dist_km(v.zone, req.o) + dist_km(req.o, req.d)
                    wait = travel_time_min(v.zone, req.o, self.t)

                if self.cfg['pricing']:
                    p0 = price_initial(v, req, route_km, wait)
                    p  = price_driver(v, req, p0, hot_zones, zone_rank)
                    ok = customer_decide(req, v, p, wait)
                else:
                    p  = BASE_FARE[v.vt] + RATE_KM[v.vt] * dist_km(req.o, req.d)
                    ok = True

                if ok:
                    v.route = new_route
                    v.profit += p
                    fuel = route_km * (PGAS / MILEAGE_L[v.vt])
                    v.profit -= fuel
                    v.km += route_km
                    acc += 1; wait_sum += wait
                else:
                    self.pending.append(req); rej += 1

        # Dispatch idle vehicles using DQN
        if self.cfg['dispatch'] and self.agent:
            for v in self.vehicles:
                if not v.route and not v.pax:
                    sf = np.zeros((FORECAST_H+1, G.n), np.float32)
                    df = np.zeros((FORECAST_H+1, G.n), np.float32)
                    state = self.agent.state(v, sf, df, self.t)
                    act = self.agent.act(v, state, explore=False, track=False)
                    nz = G.action_to_zone(v.zone, act)
                    if nz != v.zone:
                        v.zone = nz  # instant move for baseline simplicity

        # Advance: pop one route stop
        for v in self.vehicles:
            v.total_steps += 1
            if v.pax:
                v.occ_steps += 1
            if v.route:
                tag, req = v.route.pop(0)
                target = int(req) if tag == 'mv' else (req.o if tag == 'pu' else req.d)
                v.zone = target
                if tag == 'pu':
                    v.pax.append(req); v.cap += req.np
                elif tag == 'do':
                    if req in v.pax: v.pax.remove(req)
                    v.cap = max(0, v.cap - req.np)
                    v.trips += 1

        n = len(reqs)
        ar = acc / max(1, n)
        self.metrics['ar'].append(ar)
        self.metrics['profit'].append(float(np.mean([v.profit for v in self.vehicles])))
        self.metrics['km'].append(float(np.mean([v.km for v in self.vehicles])))
        hrs = max(1, self.t) / 60.0
        self.metrics['km_hr'].append(float(np.mean([v.km/max(0.01,hrs) for v in self.vehicles])))
        self.metrics['wait'].append(wait_sum / max(1, acc))
        self.metrics['occ'].append(float(np.mean([v.occ_steps/max(1,v.total_steps) for v in self.vehicles])))
        idle_frac = sum(1 for v in self.vehicles if not v.pax) / max(1, len(self.vehicles))
        self.metrics['idle_frac'].append(idle_frac)

# ═══════════════════════════════════════════════════════════════════════════
# 12. Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════
DARK='#0d1b2a'; PANEL='#16213e'
COLS=['#00e5ff','#69ff47','#ff6e40','#ffd740','#e040fb','#ff4081']

def smooth(x, w=40):
    if len(x)<w: return np.array(x,float)
    return np.convolve(x, np.ones(w)/w, 'valid')

def styled_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors='white', labelsize=8)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#444')
    ax.set_xlabel('Timestep', color='#999', fontsize=8)

def plot_training(metrics, path):
    keys   = ['ar','profit','wait','occ','km','loss']
    titles = ['Accept Rate','Avg Profit ($)','Avg Wait (min)',
              'Occupancy Rate','Avg km/Vehicle','DQN Loss']
    fig, axes = plt.subplots(2,3, figsize=(15,9))
    fig.suptitle('DARM + DPRS + DQN — Training Curves',
                 color='white', fontsize=14, fontweight='bold')
    fig.patch.set_facecolor(DARK)
    for ax,(k,ttl,col) in zip(axes.flat, zip(keys,titles,COLS)):
        styled_ax(ax,ttl)
        vals = metrics.get(k,[])
        if not vals: continue
        ax.plot(vals, alpha=0.2, color=col, lw=0.8)
        sv = smooth(vals)
        ax.plot(range(len(sv)), sv, color=col, lw=2)
        if k=='ar' or k=='occ': ax.set_ylim(0,1)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close(); print(f"[Plot] Training  → {path}")

def plot_demo_snapshot(env, metrics, path):
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    fig.suptitle(f'Demo Snapshot (t={env.t})',
                 color='white', fontsize=13, fontweight='bold')
    fig.patch.set_facecolor(DARK)

    # Panel 1: demand heat-map
    ax = axes[0]; ax.set_facecolor(DARK)
    dm = env.demand_model.future(env.t).reshape(GRID_H, GRID_W)
    im = ax.imshow(dm, cmap='inferno', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Demand Heat-Map', color='white', fontweight='bold')
    ax.tick_params(colors='white')

    # Panel 2: vehicle scatter
    ax = axes[1]; ax.set_facecolor(DARK)
    bg = env.supply.reshape(GRID_H,GRID_W)
    ax.imshow(bg, cmap='Blues', interpolation='nearest', alpha=0.3)
    tc = ['#00e5ff','#69ff47','#ff6e40']
    for v in env.vehicles:
        r,c = G.rc(v.zone)
        ax.scatter(c, r, color=tc[v.vt], s=15+8*v.cap,
                   alpha=0.6, linewidths=0)
    patches = [mpatches.Patch(color=tc[t],label=f'Type-{t}')
               for t in range(N_TYPES)]
    ax.legend(handles=patches, loc='upper right', fontsize=7,
              facecolor='#1a1a2e', labelcolor='white')
    ax.set_title('Fleet Positions', color='white', fontweight='bold')
    ax.tick_params(colors='white')

    # Panel 3: accept rate over demo
    ax = axes[2]; ax.set_facecolor(PANEL)
    ar = metrics.get('ar',[])
    ax.plot(ar, color='#00e5ff', alpha=0.3, lw=0.8)
    sv = smooth(ar,20)
    ax.plot(range(len(sv)), sv, color='#00e5ff', lw=2)
    mu = float(np.mean(ar)) if ar else 0.
    ax.axhline(mu, color='#ffd740', ls='--', lw=1.5,
               label=f'Mean={mu:.3f}')
    ax.set_ylim(0,1); ax.set_title('Accept Rate (Demo)', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#444')

    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close(); print(f"[Plot] Demo      → {path}")

def plot_comparison(baseline_results, n, path):
    """Plot all baselines side-by-side (paper Fig. 3 style)."""
    keys   = ['ar','profit','wait','occ','idle_frac','km_hr']
    titles = ['Accept Rate','Avg Profit ($)','Avg Wait (min)',
              'Occupancy Rate','Idle Time Fraction','km/hr per Vehicle']
    fig, axes = plt.subplots(2,3, figsize=(18,10))
    fig.suptitle('Baseline Comparison (Paper Fig. 3 Style)',
                 color='white', fontsize=14, fontweight='bold')
    fig.patch.set_facecolor(DARK)
    for ax, k, ttl in zip(axes.flat, keys, titles):
        styled_ax(ax, ttl)
        for i, (name, metrics) in enumerate(baseline_results.items()):
            vals = metrics.get(k, [])[-n:]
            if not vals: continue
            col = COLS[i % len(COLS)]
            sv = smooth(vals, min(20, len(vals)))
            ax.plot(range(len(sv)), sv, color=col, lw=2, label=name)
        if k == 'ar' or k == 'occ' or k == 'idle_frac':
            ax.set_ylim(0, 1)
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=7, loc='best')
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close(); print(f"[Plot] Comparison→ {path}")

# ═══════════════════════════════════════════════════════════════════════════
# 12b. Ablation Study
# ═══════════════════════════════════════════════════════════════════════════
def run_ablation(train_steps=300, eval_steps=150):
    """Run ablation study testing individual component contributions."""
    print("\n" + "="*80)
    print("  ABLATION STUDY")
    print("="*80)

    # 1. Component ablation: remove one component at a time from full framework
    ablation_configs = collections.OrderedDict()
    ablation_configs["Full (D+RS+PS+DARM)"] = dict(dispatch=True, rideshare=True, pricing=True, darm=True)
    ablation_configs["-Dispatch"]            = dict(dispatch=False, rideshare=True, pricing=True, darm=True)
    ablation_configs["-Ridesharing"]         = dict(dispatch=True, rideshare=False, pricing=True, darm=True)
    ablation_configs["-Pricing (DPRS)"]      = dict(dispatch=True, rideshare=True, pricing=False, darm=True)
    ablation_configs["-DARM (greedy only)"]  = dict(dispatch=True, rideshare=True, pricing=True, darm=False)

    # Train a shared DQN agent first
    print("\n  Training shared DQN agent...")
    env = RSEnv()
    for _ in tqdm(range(train_steps), desc="  Train"):
        env.step()
    agent = env.agent

    # 2. Evaluate each ablation config
    results = collections.OrderedDict()
    for name, cfg in ablation_configs.items():
        print(f"\n  Ablation: {name}")
        # Create a temporary baseline config
        BASELINE_CONFIGS[f"_ablation_{name}"] = cfg
        bl = BaselineEnv(f"_ablation_{name}", agent=agent, demand_model=env.demand_model)
        for _ in tqdm(range(eval_steps), desc=f"    {name[:20]:20s}"):
            bl.step()
        results[name] = bl.metrics
        del BASELINE_CONFIGS[f"_ablation_{name}"]

    # 3. Test rider change sensitivity
    print("\n  Testing rider-change sensitivity...")
    global RIDER_CHANGE_PROB, RIDER_CANCEL_PROB
    orig_change, orig_cancel = RIDER_CHANGE_PROB, RIDER_CANCEL_PROB
    change_results = collections.OrderedDict()
    for change_rate in [0.0, 0.03, 0.06, 0.10]:
        RIDER_CHANGE_PROB = change_rate
        RIDER_CANCEL_PROB = change_rate / 3.0
        label = f"Change={change_rate*100:.0f}%"
        env2 = RSEnv()
        env2.agent = agent
        for _ in tqdm(range(eval_steps), desc=f"    {label:20s}"):
            env2.step()
        change_results[label] = {
            'ar': float(np.mean(env2.metrics.get('ar', [0]))),
            'profit': float(np.mean(env2.metrics.get('profit', [0]))),
            'wait': float(np.mean(env2.metrics.get('wait', [0]))),
        }
    RIDER_CHANGE_PROB, RIDER_CANCEL_PROB = orig_change, orig_cancel

    # 4. Print results
    avg = lambda d, k: float(np.mean(d.get(k, [0.])))

    print("\n" + "-"*80)
    print("  Component Ablation Results")
    print("-"*80)
    fmt = "  {:<24} {:>8} {:>10} {:>10} {:>10}"
    print(fmt.format("Config", "AR", "Profit($)", "Wait(min)", "Occ.Rate"))
    print("  " + "-"*66)
    for name, m in results.items():
        print(fmt.format(name[:24],
            f"{avg(m,'ar'):.4f}", f"{avg(m,'profit'):.2f}",
            f"{avg(m,'wait'):.2f}", f"{avg(m,'occ'):.4f}"))

    print("\n" + "-"*80)
    print("  Rider Change Sensitivity")
    print("-"*80)
    for label, r in change_results.items():
        print(f"  {label:20s}  AR={r['ar']:.4f}  Profit=${r['profit']:.2f}  Wait={r['wait']:.2f}min")

    # 5. Plot ablation
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Ablation Study: Component Impact", color='white', fontsize=14, fontweight='bold')
    fig.patch.set_facecolor(DARK)
    abl_keys = ['ar', 'profit', 'wait', 'occ']
    abl_titles = ['Accept Rate', 'Avg Profit ($)', 'Avg Wait (min)', 'Occupancy']
    names = list(results.keys())
    vals_per_key = {k: [avg(results[n], k) for n in names] for k in abl_keys}
    colors_bar = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
    for ax, k, ttl in zip(axes, abl_keys, abl_titles):
        styled_ax(ax, ttl)
        bars = vals_per_key[k]
        x = range(len(names))
        ax.bar(x, bars, color=colors_bar[:len(names)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([n[:12] for n in names], rotation=30, ha='right', fontsize=7, color='#94a3b8')
        if k == 'ar' or k == 'occ': ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_study.png")
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"\n[Plot] Ablation  -> {path}")
    return results, change_results

# ═══════════════════════════════════════════════════════════════════════════
# 13. Main
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="DARM + DPRS + DQN ridesharing sim")
    p.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--demo-steps", type=int, default=DEMO_STEPS)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-demo", action="store_true")
    p.add_argument("--save-model", type=str, default="")
    p.add_argument("--load-model", type=str, default="")
    p.add_argument("--save-best", type=str, default="")
    p.add_argument("--best-metric", choices=["ar", "profit", "loss"], default="ar")
    p.add_argument("--best-window", type=int, default=100)
    p.add_argument("--no-export-metrics", action="store_true")
    p.add_argument("--state-interval", type=int, default=1)
    p.add_argument("--ablation", action="store_true", help="Run ablation study")
    p.add_argument("--ablation-train", type=int, default=300, help="Training steps for ablation")
    p.add_argument("--ablation-eval", type=int, default=150, help="Eval steps per ablation config")
    p.add_argument("--use-dataset", action="store_true", default=None)
    p.add_argument("--no-dataset", action="store_true")
    p.add_argument("--dataset-path", type=str, default="")
    p.add_argument("--dataset-max-files", type=int, default=DATASET_MAX_FILES)
    p.add_argument("--dataset-max-rows", type=int, default=DATASET_MAX_ROWS)
    p.add_argument("--dataset-time-bin", type=int, default=DATASET_TIME_BIN)
    p.add_argument("--dataset-demand-scale", type=float, default=DATASET_DEMAND_SCALE)
    p.add_argument("--enable-congestion", action="store_true")
    p.add_argument("--enable-rider-changes", action="store_true")
    p.add_argument("--rider-change-prob", type=float, default=0.03)
    p.add_argument("--rider-cancel-prob", type=float, default=0.01)
    p.add_argument("--use-time-features", action="store_true")
    p.add_argument("--enable-osrm", action="store_true")
    p.add_argument("--osrm-base-url", type=str, default="")
    p.add_argument("--osrm-profile", type=str, default="")
    p.add_argument("--osrm-timeout-sec", type=float, default=2.0)
    p.add_argument("--osrm-cache-size", type=int, default=OSRM_CACHE_SIZE)
    return p.parse_args()

def main():
    args = parse_args()
    t0 = time.time()
    if args.no_dataset:
        use_dataset = False
    elif args.use_dataset is not None:
        use_dataset = bool(args.use_dataset)
    else:
        use_dataset = USE_DATASET

    apply_runtime_config({
        "use_dataset": use_dataset,
        "dataset_path": args.dataset_path,
        "dataset_max_files": args.dataset_max_files,
        "dataset_max_rows": args.dataset_max_rows,
        "dataset_time_bin": args.dataset_time_bin,
        "dataset_demand_scale": args.dataset_demand_scale,
        "enable_congestion": bool(args.enable_congestion),
        "use_time_features": bool(args.use_time_features),
        "enable_osrm": bool(args.enable_osrm),
        "osrm_base_url": args.osrm_base_url,
        "osrm_profile": args.osrm_profile,
        "osrm_timeout_sec": args.osrm_timeout_sec,
        "osrm_cache_size": args.osrm_cache_size,
        "rider_change_prob": args.rider_change_prob if args.enable_rider_changes else 0.0,
        "rider_cancel_prob": args.rider_cancel_prob if args.enable_rider_changes else 0.0,
    })
    print("\n" + "═"*60)
    print("  DARM + DPRS + Double-DQN Ride-Sharing Framework")
    print("  Haliem et al. IEEE TITS 2021 + improvements")
    print("═"*60)
    print(f"  PyTorch: {TORCH}  |  Grid: {GRID_W}×{GRID_H}  |"
          f"  Fleet: {N_VEHICLES}  |  Steps: {TRAIN_STEPS}")
    print("═"*60+"\n")

    # ── Training ──────────────────────────────────────────────────────────
    agent_cfg = {
        "lr": LR,
        "gamma": GAMMA,
        "batch_size": BATCH,
        "replay_size": BUF,
        "min_buf": MIN_BUF,
        "eps_start": EPS0,
        "eps_end": EPS_MIN,
        "eps_decay": calc_eps_decay_rate(EPS0, EPS_MIN, args.train_steps),
        "target_update": TGT_UPD,
    }
    train_env = RSEnv(agent_cfg=agent_cfg, train_mode=True)
    live_server = None
    control = None
    if args.load_model:
        train_env.agent.load(args.load_model)
        print(f"Loaded model: {args.load_model}")

    if not args.skip_train:
        roll = collections.deque(maxlen=args.best_window)
        best_score = -float("inf")
        bar = tqdm(total=args.train_steps, desc="Train ")
        step_idx = 0
        training_started = False
        if control and not control.start_event.is_set():
            print("▶  Training pending — click Start in UI")
        elif not control:
            print("▶  Training DARM+DPRS+DQN …")
            training_started = True

        while step_idx < args.train_steps:
            if control:
                if control.save_event.is_set():
                    path = control.save_path or os.path.join(os.path.dirname(__file__), "outputs", "model_latest.pt")
                    if not os.path.isabs(path):
                        path = os.path.join(os.path.dirname(__file__), path)
                    train_env.agent.save(path)
                    train_env.model_path = path
                    control.save_event.clear()
                    print(f"\n[Save] Model saved: {path}")
                if control.load_event.is_set():
                    path = control.load_path
                    if path:
                        if not os.path.isabs(path):
                            path = os.path.join(os.path.dirname(__file__), path)
                        train_env.agent.load(path)
                        train_env.model_path = path
                        print(f"\n[Load] Model loaded: {path}")
                    control.load_event.clear()
                if control.reset_event.is_set():
                    train_env = RSEnv(agent_cfg=agent_cfg, train_mode=True)
                    control.reset_event.clear()
                    roll.clear()
                    best_score = -float("inf")
                    step_idx = 0
                    bar.n = 0
                    training_started = False
                    continue
                if not control.start_event.is_set():
                    time.sleep(0.1)
                    continue
                if control.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                if not training_started:
                    print("▶  Training DARM+DPRS+DQN …")
                    training_started = True

            m = train_env.step()
            step_idx += 1
            bar.update(1)
            if live_server and step_idx % args.state_interval == 0:
                state = train_env.export_state(m)
                live_server.push(json.dumps(state))
            roll.append(m['ar'])
            if args.save_best:
                if args.best_metric == "ar":
                    score = float(np.mean(roll))
                elif args.best_metric == "profit":
                    score = float(m['profit'])
                else:
                    score = -float(m['loss'])
                if score > best_score:
                    train_env.agent.save(args.save_best)
                    best_score = score
                    print(f"\n[Best] Saved model: {args.save_best} ({args.best_metric}={score:.4f})")
            bar.set_postfix({'AR': f"{np.mean(roll):.3f}",
                             '$':  f"{m['profit']:.0f}",
                             'W':  f"{m['wait']:.1f}m",
                             'ε':  f"{m['eps']:.3f}"})
        print(f"\n  Final mean accept-rate: {np.mean(roll):.3f}")

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    if not args.skip_train:
        plot_training(train_env.metrics, os.path.join(out_dir, "training_curves.png"))

    # ── Demo ──────────────────────────────────────────────────────────────
    eval_agent = train_env.agent.clone_for_eval() if hasattr(train_env.agent, "clone_for_eval") else train_env.agent
    demo_env = None
    demo_m = collections.defaultdict(list)
    if not args.skip_demo:
        print("\n▶  Demo run …")
        demo_env = RSEnv(cfg=train_env.cfg, agent=eval_agent, train_mode=False)
        for i in tqdm(range(args.demo_steps), desc="Demo  "):
            m = demo_env.step()
            if live_server and i % args.state_interval == 0:
                state = demo_env.export_state(m)
                live_server.push(json.dumps(state))
            for k,v in m.items():
                demo_m[k].append(v)

        plot_demo_snapshot(demo_env, demo_m, os.path.join(out_dir, "demo_snapshot.png"))

        print(f"\n  Demo accept-rate : {np.mean(demo_m['ar']):.3f}")
        print(f"  Demo avg profit  : ${np.mean(demo_m['profit']):.2f}")
        print(f"  Demo avg wait    : {np.mean(demo_m['wait']):.2f} min")
        print(f"  Demo occupancy   : {np.mean(demo_m['occ']):.3f}")

    # ── Run all paper baselines ────────────────────────────────────────────
    baseline_results = collections.OrderedDict()
    baseline_results["D,RS,PS,DARM (Ours)"] = demo_m

    if not args.skip_demo:
        baseline_names_to_run = [
            "!D,!RS,!PS,GM",
            "!D,RS,!PS,GM",
            "D,!RS,!PS,GM",
            "D,RS,!PS,GM",
            "D,RS,PS,GM",
        ]
        for bname in baseline_names_to_run:
            print(f"\n▶  Baseline: {bname} …")
            bl = BaselineEnv(bname, agent=eval_agent, demand_model=demo_env.demand_model)
            for _ in tqdm(range(args.demo_steps), desc=f"  {bname[:16]:16s}"):
                bl.step()
            baseline_results[bname] = bl.metrics

        plot_comparison(baseline_results, args.demo_steps,
                        os.path.join(out_dir, "comparison.png"))

    # ── Final table (paper style) ─────────────────────────────────────────
    def mm(d, k): return f"{np.mean(d.get(k,[0.])):.4f}"
    print("\n" + "═"*90)
    print("  FINAL COMPARISON (Paper Table Style)")
    print("═"*90)
    header_keys = ["ar", "profit", "wait", "occ", "idle_frac", "km", "km_hr"]
    header_names = ["AR", "Profit($)", "Wait(min)", "Occ.Rate", "Idle%", "km", "km/hr"]
    fmt = "  {:<24}" + " {:>10}" * len(header_keys)
    print(fmt.format("Baseline", *header_names))
    print("  " + "-"*84)
    for bname, bmetrics in baseline_results.items():
        vals = [mm(bmetrics, k) for k in header_keys]
        print(fmt.format(bname[:24], *vals))
    print("═"*90)
    print(f"\n  Wall-clock time: {time.time()-t0:.1f}s")

    if not args.no_export_metrics:
        config = {
            "grid": GRID_W,
            "vehicles": N_VEHICLES,
            "types": N_TYPES,
            "reject_radius_km": REJECT_RADIUS_KM,
            "max_idle_min": MAX_IDLE_MIN,
            "forecast_h": FORECAST_H,
            "max_detour_factor": MAX_DETOUR_FACTOR,
            "train_steps": args.train_steps,
            "demo_steps": args.demo_steps,
            "warmup_steps": WARMUP_STEPS,
            "reward_weights": {"B1":B1,"B2":B2,"B3":B3,"B4":B4,"B5":B5},
            "use_dataset": USE_DATASET,
            "dataset_path": DATASET_PATH,
            "dataset_max_files": DATASET_MAX_FILES,
            "dataset_max_rows": DATASET_MAX_ROWS,
            "dataset_time_bin": DATASET_TIME_BIN,
            "dataset_demand_scale": DATASET_DEMAND_SCALE,
            "enable_congestion": CONGESTION_ENABLED,
            "use_time_features": USE_TIME_FEATURES,
            "rider_change_prob": RIDER_CHANGE_PROB,
            "rider_cancel_prob": RIDER_CANCEL_PROB,
            "baselines": list(baseline_results.keys()),
        }
        # Add baseline results to the export bundle
        baseline_export = {}
        for bname, bmetrics in baseline_results.items():
            baseline_export[bname] = _metrics_to_json(bmetrics)
        bundle = {
            "train": _metrics_to_json(train_env.metrics),
            "demo": _metrics_to_json(demo_m),
            "baselines": baseline_export,
            "qmax_history": [float(x) for x in train_env.qmax_history],
            "config": config,
        }
        json_path = os.path.join(out_dir, "metrics_bundle.json")
        js_path = os.path.join(out_dir, "metrics_bundle.js")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        with open(js_path, "w", encoding="utf-8") as f:
            f.write("window.__METRICS_BUNDLE__ = ")
            json.dump(bundle, f)
            f.write(";")
        print(f"\nMetrics bundle saved:\n  {json_path}\n  {js_path}")

    if args.save_model:
        train_env.agent.save(args.save_model)
        print(f"\nModel saved: {args.save_model}")
    print("\nOutputs saved:")
    for f in ["training_curves.png","demo_snapshot.png","comparison.png"]:
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            print(f"  {p}")

    # --- Ablation study ---
    if args.ablation:
        abl_res, change_res = run_ablation(
            train_steps=args.ablation_train,
            eval_steps=args.ablation_eval)
        p = os.path.join(out_dir, "ablation_study.png")
        if os.path.exists(p):
            print(f"  {p}")

    if live_server:
        live_server.stop()

if __name__=="__main__":
    main()
