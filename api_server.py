"""
API Server for DARM+DPRS+DQN Interactive Training Dashboard.
Provides REST endpoints for training control, model management, and comparison.
"""
import sys, os, json, time, threading, queue, uuid, glob, shutil, copy
sys.stdout.reconfigure(encoding='utf-8')

from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import importlib

# Import the simulation module
import ridesharing_darm_dprs_dqn as sim

# ═══════════════════════════════════════════════════════════════════════════
# Globals
# ═══════════════════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
BASE_DIR = os.path.dirname(__file__)
os.makedirs(MODELS_DIR, exist_ok=True)

# Training state
train_state = {
    "running": False,
    "paused": False,
    "step": 0,
    "total_steps": 0,
    "phase": "idle",  # idle, training, demo, baselines, ablation
    "metrics": {},
    "history": {
        "ar": [],
        "profit": [],
        "profit_total": [],
        "wait": [],
        "occ": [],
        "idle_frac": [],
        "km": [],
        "loss": [],
        "eps": [],
        "qmax": [],
        "buf": [],
        "val_ar": [],
        "val_profit": [],
        "val_profit_total": [],
        "val_wait": [],
        "val_occ": [],
        "val_idle": [],
        "val_km": [],
    },
    "config": {},
    "model_id": None,
}
train_lock = threading.Lock()
train_thread = None
sse_clients = []
current_env = None

# ═══════════════════════════════════════════════════════════════════════════
# SSE Broadcasting
# ═══════════════════════════════════════════════════════════════════════════
def broadcast_sse(data):
    """Send data to all connected SSE clients."""
    msg = f"data: {json.dumps(data)}\n\n"
    dead = []
    for q in sse_clients:
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(q)
    for q in dead:
        sse_clients.remove(q)

def emit_state():
    """Broadcast current training state to all SSE clients."""
    with train_lock:
        snapshot = {
            "type": "train_update",
            "step": train_state["step"],
            "total_steps": train_state["total_steps"],
            "phase": train_state["phase"],
            "running": train_state["running"],
            "paused": train_state["paused"],
            "metrics": train_state["metrics"],
            "history": {k: v[-100:] for k, v in train_state["history"].items()},
        }
    broadcast_sse(snapshot)

# ═══════════════════════════════════════════════════════════════════════════
# Training Worker
# ═══════════════════════════════════════════════════════════════════════════
def training_worker(config):
    """Run training in background thread with live metric emission."""
    global current_env
    try:
        steps = int(config.get("steps", 500))
        algo = config.get("algorithm", "full")
        algo_cfg = sim.resolve_algo(algo)
        eval_interval = int(config.get("eval_interval", sim.EVAL_INTERVAL))
        eval_steps = int(config.get("eval_steps", sim.EVAL_STEPS))
        eval_seed = int(config.get("eval_seed", sim.EVAL_SEED))
        # Apply config overrides
        sim.N_VEHICLES = config.get("fleet_size", sim.N_VEHICLES)
        sim.B1 = config.get("beta1", sim.B1)
        sim.B2 = config.get("beta2", sim.B2)
        sim.B3 = config.get("beta3", sim.B3)
        sim.B4 = config.get("beta4", sim.B4)
        sim.B5 = config.get("beta5", sim.B5)
        sim.WARMUP_STEPS = config.get("warmup_steps", sim.WARMUP_STEPS)
        sim.apply_runtime_config({
            "use_dataset": config.get("use_dataset", sim.USE_DATASET),
            "dataset_path": config.get("dataset_path", ""),
            "dataset_max_files": config.get("dataset_max_files", sim.DATASET_MAX_FILES),
            "dataset_max_rows": config.get("dataset_max_rows", sim.DATASET_MAX_ROWS),
            "dataset_time_bin": config.get("dataset_time_bin", sim.DATASET_TIME_BIN),
            "dataset_demand_scale": config.get("dataset_demand_scale", sim.DATASET_DEMAND_SCALE),
            "enable_osrm": config.get("enable_osrm", False),
            "osrm_base_url": config.get("osrm_base_url", ""),
            "osrm_profile": config.get("osrm_profile", ""),
            "osrm_timeout_sec": config.get("osrm_timeout_sec", sim.OSRM_TIMEOUT_SEC),
            "osrm_cache_size": config.get("osrm_cache_size", sim.OSRM_CACHE_SIZE),
            "enable_congestion": config.get("enable_congestion", False),
            "use_time_features": config.get("use_time_features", False),
            "rider_change_prob": config.get("rider_change_prob", 0.0),
            "rider_cancel_prob": config.get("rider_cancel_prob", 0.0),
        })

        # DQN hyperparams
        lr = config.get("learning_rate", 1e-3)
        gamma = config.get("gamma", 0.97)
        eps_start = config.get("eps_start", 1.0)
        eps_end = config.get("eps_end", 0.05)
        eps_decay_steps = config.get("eps_decay", 5000)
        eps_decay = sim.calc_eps_decay_rate(eps_start, eps_end, eps_decay_steps)
        batch_size = config.get("batch_size", 64)
        replay_size = config.get("replay_size", 5000)

        with train_lock:
            train_state["running"] = True
            train_state["phase"] = "training"
            train_state["total_steps"] = steps
            train_state["step"] = 0
            train_state["config"] = config
            train_state["history"] = {
                "ar": [],
                "profit": [],
                "profit_total": [],
                "wait": [],
                "occ": [],
                "idle_frac": [],
                "km": [],
                "loss": [],
                "eps": [],
                "qmax": [],
                "buf": [],
                "val_ar": [],
                "val_profit": [],
                "val_profit_total": [],
                "val_wait": [],
                "val_occ": [],
                "val_idle": [],
                "val_km": [],
            }

        agent_cfg = {
            "lr": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "replay_size": replay_size,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
        }
        env = sim.RSEnv(cfg=algo_cfg, agent_cfg=agent_cfg, train_mode=True)
        current_env = env

        last_val = {
            "val_ar": None,
            "val_profit": None,
            "val_profit_total": None,
            "val_wait": None,
            "val_occ": None,
            "val_idle": None,
            "val_km": None,
        }

        for step in range(1, steps + 1):
            # Check pause
            while train_state["paused"] and train_state["running"]:
                time.sleep(0.1)
            if not train_state["running"]:
                break

            env.step()

            did_eval = False
            if eval_interval > 0 and (step % max(1, eval_interval) == 0):
                _, eval_metrics = sim.run_eval(
                    env.agent,
                    algo_cfg=algo_cfg,
                    steps=eval_steps,
                    seed=eval_seed,
                )
                avg = lambda k: float(sum(eval_metrics.get(k, [])) / max(1, len(eval_metrics.get(k, []))))
                last_val = {
                    "val_ar": avg("ar"),
                    "val_profit": avg("profit"),
                    "val_profit_total": avg("profit_total"),
                    "val_wait": avg("wait"),
                    "val_occ": avg("occ"),
                    "val_idle": avg("idle_frac"),
                    "val_km": avg("km"),
                }
                did_eval = True

            with train_lock:
                train_state["step"] = step
                m = env.metrics
                ar = m.get("ar", [0])[-1] if m.get("ar") else 0
                profit = m.get("profit", [0])[-1] if m.get("profit") else 0
                profit_total = m.get("profit_total", [0])[-1] if m.get("profit_total") else 0
                wait = m.get("wait", [0])[-1] if m.get("wait") else 0
                occ = m.get("occ", [0])[-1] if m.get("occ") else 0
                idle_frac = m.get("idle_frac", [0])[-1] if m.get("idle_frac") else 0
                km = m.get("km", [0])[-1] if m.get("km") else 0
                loss = m.get("loss", [0])[-1] if m.get("loss") else 0
                eps = m.get("eps", [0])[-1] if m.get("eps") else 0
                buf = m.get("buf", [0])[-1] if m.get("buf") else 0
                qmax = float(env.agent.zone_q.max()) if hasattr(env.agent, 'zone_q') else 0

                train_state["metrics"] = {
                    "ar": ar,
                    "profit": profit,
                    "profit_total": profit_total,
                    "wait": wait,
                    "occ": occ,
                    "idle_frac": idle_frac,
                    "km": km,
                    "loss": loss,
                    "eps": eps,
                    "qmax": qmax,
                    "buf": buf,
                    **last_val,
                }
                train_state["history"]["ar"].append(ar)
                train_state["history"]["profit"].append(profit)
                train_state["history"]["profit_total"].append(profit_total)
                train_state["history"]["wait"].append(wait)
                train_state["history"]["occ"].append(occ)
                train_state["history"]["idle_frac"].append(idle_frac)
                train_state["history"]["km"].append(km)
                train_state["history"]["loss"].append(loss)
                train_state["history"]["eps"].append(eps)
                train_state["history"]["qmax"].append(qmax)
                train_state["history"]["buf"].append(buf)
                if did_eval:
                    train_state["history"]["val_ar"].append(last_val["val_ar"])
                    train_state["history"]["val_profit"].append(last_val["val_profit"])
                    train_state["history"]["val_profit_total"].append(last_val["val_profit_total"])
                    train_state["history"]["val_wait"].append(last_val["val_wait"])
                    train_state["history"]["val_occ"].append(last_val["val_occ"])
                    train_state["history"]["val_idle"].append(last_val["val_idle"])
                    train_state["history"]["val_km"].append(last_val["val_km"])

            if step % max(1, steps // 100) == 0:
                emit_state()

        with train_lock:
            train_state["phase"] = "done"
            train_state["running"] = False
        emit_state()

    except Exception as e:
        with train_lock:
            train_state["phase"] = "error"
            train_state["running"] = False
            train_state["metrics"]["error"] = str(e)
        emit_state()
        import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════
# Model Management
# ═══════════════════════════════════════════════════════════════════════════
def list_models():
    """List all saved models with metadata."""
    models = []
    for meta_file in glob.glob(os.path.join(MODELS_DIR, "*/meta.json")):
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            models.append(meta)
        except Exception:
            pass
    models.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
    return models

def save_model(name, description=""):
    """Save current model to disk with metadata."""
    global current_env
    if current_env is None:
        return {"error": "No trained model to save"}

    model_id = f"{name.replace(' ','_')}_{int(time.time())}"
    model_dir = os.path.join(MODELS_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(model_dir, "model.pt")
    current_env.agent.save(model_path)

    # Save metadata
    meta = {
        "id": model_id,
        "name": name,
        "description": description,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": train_state.get("config", {}),
        "final_metrics": train_state.get("metrics", {}),
        "steps_trained": train_state.get("step", 0),
        "history_summary": {
            k: {"min": min(v) if v else 0, "max": max(v) if v else 0,
                "last": v[-1] if v else 0, "mean": sum(v)/len(v) if v else 0}
            for k, v in train_state.get("history", {}).items()
        },
        "model_path": model_path,
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save full history
    with open(os.path.join(model_dir, "history.json"), "w") as f:
        json.dump(train_state.get("history", {}), f)

    return meta

def load_model(model_id):
    """Load a saved model."""
    global current_env
    model_dir = os.path.join(MODELS_DIR, model_id)
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {"error": f"Model {model_id} not found"}

    with open(meta_path, "r") as f:
        meta = json.load(f)

    sim.apply_runtime_config(meta.get("config", {}))

    if current_env is None:
        current_env = sim.RSEnv()
    current_env.agent.load(meta["model_path"])

    # Load history if available
    hist_path = os.path.join(model_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            hist = json.load(f)
        with train_lock:
            train_state["history"] = hist
            train_state["step"] = meta.get("steps_trained", 0)
            train_state["config"] = meta.get("config", {})
            train_state["metrics"] = meta.get("final_metrics", {})
            train_state["model_id"] = model_id
            train_state["phase"] = "loaded"

    return meta

def delete_model(model_id):
    """Delete a saved model."""
    model_dir = os.path.join(MODELS_DIR, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        return {"status": "deleted", "id": model_id}
    return {"error": "Model not found"}

def compare_models(model_ids, eval_steps=100):
    """Evaluate multiple models and return comparison data."""
    results = {}
    orig_fleet = sim.N_VEHICLES
    try:
        for mid in model_ids:
            model_dir = os.path.join(MODELS_DIR, mid)
            meta_path = os.path.join(model_dir, "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r") as f:
                meta = json.load(f)

            cfg = meta.get("config", {})
            sim.N_VEHICLES = cfg.get("fleet_size", sim.N_VEHICLES)
            algo_cfg = sim.resolve_algo(cfg.get("algorithm", "full"))
            sim.apply_runtime_config({
                "use_dataset": cfg.get("use_dataset", sim.USE_DATASET),
                "dataset_path": cfg.get("dataset_path", ""),
                "dataset_max_files": cfg.get("dataset_max_files", sim.DATASET_MAX_FILES),
                "dataset_max_rows": cfg.get("dataset_max_rows", sim.DATASET_MAX_ROWS),
                "dataset_time_bin": cfg.get("dataset_time_bin", sim.DATASET_TIME_BIN),
                "dataset_demand_scale": cfg.get("dataset_demand_scale", sim.DATASET_DEMAND_SCALE),
                "enable_congestion": cfg.get("enable_congestion", False),
                "use_time_features": cfg.get("use_time_features", False),
                "rider_change_prob": cfg.get("rider_change_prob", 0.0),
                "rider_cancel_prob": cfg.get("rider_cancel_prob", 0.0),
            })
            agent = sim.DQNAgent(sim.N_VEHICLES)
            agent.load(meta["model_path"])
            _, eval_metrics = sim.run_eval(
                agent,
                algo_cfg=algo_cfg,
                steps=eval_steps,
                seed=sim.EVAL_SEED,
            )
            avg = lambda k: float(sum(eval_metrics.get(k, [])) / max(1, len(eval_metrics.get(k, []))))
            results[meta["name"]] = {
                "id": mid,
                "ar": avg("ar"), "profit": avg("profit"),
                "wait": avg("wait"), "occ": avg("occ"),
                "idle_frac": avg("idle_frac"),
                "km": avg("km"),
                "config": cfg,
                "steps_trained": meta.get("steps_trained", 0),
            }
    finally:
        sim.N_VEHICLES = orig_fleet
    return results

def summarize_metrics(metrics):
    """Convert metric histories into a compact JSON-friendly summary."""
    avg = lambda k: float(sum(metrics.get(k, [])) / max(1, len(metrics.get(k, []))))
    return {
        "ar": avg("ar"),
        "profit": avg("profit"),
        "wait": avg("wait"),
        "occ": avg("occ"),
        "idle_frac": avg("idle_frac"),
        "km": avg("km"),
    }

# ═══════════════════════════════════════════════════════════════════════════
# HTTP API Handler
# ═══════════════════════════════════════════════════════════════════════════
class APIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress request logging

    def handle(self):
        """Override to catch Windows ConnectionAbortedError gracefully."""
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            pass  # Normal on Windows when browser closes connection

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.path = "/dashboard.html"
            return super().do_GET()

        # Handle favicon.ico to prevent 404 noise
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if path == "/api/stream":
            self._handle_sse()
        elif path == "/api/train/status":
            with train_lock:
                self._json({
                    "step": train_state["step"],
                    "total_steps": train_state["total_steps"],
                    "phase": train_state["phase"],
                    "running": train_state["running"],
                    "paused": train_state["paused"],
                    "metrics": train_state["metrics"],
                    "history": {k: v[-200:] for k, v in train_state["history"].items()},
                    "config": train_state["config"],
                    "model_id": train_state.get("model_id"),
                })
        elif path == "/api/models":
            self._json(list_models())
        elif path == "/api/config":
            self._json({
                "fleet_size": sim.N_VEHICLES,
                "beta1": sim.B1, "beta2": sim.B2, "beta3": sim.B3,
                "beta4": sim.B4, "beta5": sim.B5,
                "warmup_steps": sim.WARMUP_STEPS,
                "rider_change_prob": sim.RIDER_CHANGE_PROB,
                "rider_cancel_prob": sim.RIDER_CANCEL_PROB,
                "use_dataset": sim.USE_DATASET,
                "dataset_path": sim.DATASET_PATH,
                "dataset_max_files": sim.DATASET_MAX_FILES,
                "dataset_max_rows": sim.DATASET_MAX_ROWS,
                "dataset_time_bin": sim.DATASET_TIME_BIN,
                "dataset_demand_scale": sim.DATASET_DEMAND_SCALE,
                "enable_congestion": sim.CONGESTION_ENABLED,
                "use_time_features": sim.USE_TIME_FEATURES,
                "enable_osrm": sim.OSRM_ENABLED,
                "osrm_base_url": sim.OSRM_BASE_URL,
                "osrm_profile": sim.OSRM_PROFILE,
                "osrm_timeout_sec": sim.OSRM_TIMEOUT_SEC,
                "osrm_cache_size": sim.OSRM_CACHE_SIZE,
                "learning_rate": sim.LR, "gamma": sim.GAMMA,
                "eps_start": sim.EPS0, "eps_end": sim.EPS_MIN, "eps_decay": 5000,
                "batch_size": sim.BATCH, "replay_size": sim.BUF,
                "eval_interval": sim.EVAL_INTERVAL,
                "eval_steps": sim.EVAL_STEPS,
                "eval_seed": sim.EVAL_SEED,
                "algorithms": list(sim.ALGO_PRESETS.keys()),
            })
        elif path == "/api/sim/state":
            env = current_env
            if env is None:
                self._json({
                    "t": 0,
                    "grid": {"w": sim.GRID_W, "h": sim.GRID_H},
                    "demand": [],
                    "zone_q": [],
                    "vehicles": [],
                    "events": [],
                    "pending": [],
                    "totals": {"requests": 0, "served": 0, "rejected": 0},
                    "metrics": {},
                })
            else:
                m = env.metrics
                metrics = {
                    "ar": m.get("ar", [0])[-1] if m.get("ar") else 0,
                    "idle_frac": m.get("idle_frac", [0])[-1] if m.get("idle_frac") else 0,
                    "occ": m.get("occ", [0])[-1] if m.get("occ") else 0,
                    "wait": m.get("wait", [0])[-1] if m.get("wait") else 0,
                    "km": m.get("km", [0])[-1] if m.get("km") else 0,
                }
                self._json(env.export_state(metrics))
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = {}
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                body = json.loads(self.rfile.read(length))
        except Exception:
            pass

        if path == "/api/train/start":
            global train_thread
            if train_state["running"]:
                self._json({"error": "Training already running"}, 409)
                return
            # Reset state BEFORE starting thread so polls see "starting" not "done"
            config = body
            with train_lock:
                train_state["running"] = True
                train_state["paused"] = False
                train_state["phase"] = "starting"
                train_state["step"] = 0
                train_state["total_steps"] = config.get("steps", 500)
            train_thread = threading.Thread(target=training_worker, args=(config,), daemon=True)
            train_thread.start()
            self._json({"status": "started", "steps": config.get("steps", 500)})

        elif path == "/api/train/pause":
            with train_lock:
                train_state["paused"] = not train_state["paused"]
            self._json({"paused": train_state["paused"]})

        elif path == "/api/train/stop":
            with train_lock:
                train_state["running"] = False
                train_state["paused"] = False
            self._json({"status": "stopped"})

        elif path == "/api/models/save":
            name = body.get("name", f"model_{int(time.time())}")
            desc = body.get("description", "")
            result = save_model(name, desc)
            self._json(result)

        elif path == "/api/models/load":
            model_id = body.get("id", "")
            result = load_model(model_id)
            self._json(result)

        elif path == "/api/compare":
            model_ids = body.get("model_ids", [])
            eval_steps = body.get("eval_steps", 100)
            results = compare_models(model_ids, eval_steps)
            self._json(results)

        elif path == "/api/ablation":
            if train_state["running"]:
                self._json({"error": "Training is running"}, 409)
                return
            train_steps = int(body.get("train_steps", 300))
            eval_steps = int(body.get("eval_steps", 150))
            results, sensitivity = sim.run_ablation(train_steps=train_steps, eval_steps=eval_steps)
            self._json({
                "components": {name: summarize_metrics(metrics) for name, metrics in results.items()},
                "sensitivity": sensitivity,
                "plot": "/outputs/ablation_study.png",
            })

        else:
            self._json({"error": "Unknown endpoint"}, 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/models/"):
            model_id = path.split("/api/models/")[1]
            result = delete_model(model_id)
            self._json(result)
        else:
            self._json({"error": "Unknown endpoint"}, 404)

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = queue.Queue(maxsize=50)
        sse_clients.append(q)
        try:
            # Send initial state
            emit_state()
            while True:
                try:
                    msg = q.get(timeout=2)
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionError):
            pass
        finally:
            if q in sse_clients:
                sse_clients.remove(q)

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    import socket
    from http.server import ThreadingHTTPServer
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    # Allow port reuse to prevent "Address already in use" errors
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.address_family = socket.AF_INET
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), APIHandler)
    except OSError:
        port = 9000
        server = ThreadingHTTPServer(("0.0.0.0", port), APIHandler)
    server.daemon_threads = True
    print(f"\n  DARM+DPRS+DQN Training Dashboard")
    print(f"  Open: http://localhost:{port}/\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
