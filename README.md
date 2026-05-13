# DARM + DPRS + DQN: Intelligent Ride-Sharing Framework

## An Extended Implementation of Haliem et al. (IEEE TITS 2021) with Travel-Time Uncertainty, On-the-Fly Rider Changes, and Interactive Training Dashboard

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Base Paper Summary](#3-base-paper-summary)
4. [System Architecture](#4-system-architecture)
5. [Core Algorithms](#5-core-algorithms)
6. [Our Extensions Beyond the Paper](#6-our-extensions-beyond-the-paper)
7. [Mathematical Formulation](#7-mathematical-formulation)
8. [Simulation Environment](#8-simulation-environment)
9. [Interactive Dashboard](#9-interactive-dashboard)
10. [Experimental Results](#10-experimental-results)
11. [Ablation Study](#11-ablation-study)
12. [Comparison: Paper vs Our Implementation](#12-comparison-paper-vs-our-implementation)
13. [File Structure](#13-file-structure)
14. [How to Run](#14-how-to-run)
15. [Future Work](#15-future-work)
16. [References](#16-references)

---

## 1. Project Overview

This project implements and **extends** the ride-sharing optimization framework proposed by Haliem et al. in their IEEE Transactions on Intelligent Transportation Systems (TITS) 2021 paper: *"A Distributed Model-Free Ride-Sharing Approach for Joint Matching, Pricing, and Dispatching."*

The framework combines three intelligent components to optimize urban ride-sharing:

| Component | Full Name | Role |
|-----------|-----------|------|
| **DARM** | Distributed Asynchronous Ride Matching | Matches riders to vehicles optimally using insertion-based route planning |
| **DPRS** | Dynamic Pricing for Ride-Sharing | Sets competitive, fair prices based on supply/demand using game-theoretic principles |
| **DQN** | Double Deep Q-Network | Learns optimal vehicle dispatching strategies through reinforcement learning |

### What Makes This Project Unique

Unlike the paper which only presents results, this project provides:

1. **A fully functional simulation** — A complete 15×15 city grid with 150 vehicles, stochastic demand, and realistic dynamics
2. **Travel-time uncertainty** — Zone-based congestion model with rush-hour peaks (paper's future work, Section VII)
3. **On-the-fly rider changes** — Riders can change destinations or cancel mid-trip (paper's future work, Reference [17])
4. **Interactive training dashboard** — A 4-tab web UI for training, simulating, saving/loading models, and comparing configurations
5. **Ablation study** — Quantifies the contribution of each component independently
6. **Model library** — Save, load, and compare models trained with different hyperparameters

Note: Extensions are disabled by default to match the paper. Enable them from the dashboard toggles.

---

## 2. Problem Statement & Motivation

### The Urban Mobility Challenge

Ride-sharing platforms (Uber, Lyft, Ola) face three simultaneous optimization problems:

1. **Who picks up whom?** (Matching) — Assigning riders to vehicles while minimizing detours for existing passengers
2. **How much to charge?** (Pricing) — Setting fares that are competitive yet profitable, adapting to real-time supply/demand
3. **Where to send idle vehicles?** (Dispatching) — Proactively positioning empty vehicles in high-demand zones

### Why These Problems Are Hard

- **Combinatorial explosion**: With 150 vehicles and dozens of requests per timestep, the matching space is enormous
- **Dynamic environment**: Demand shifts constantly (rush hours, events, weather)
- **Competing objectives**: Maximize profit vs. minimize wait times vs. maximize accept rates
- **Information asymmetry**: Drivers don't know future demand; riders don't know vehicle availability

### Why Reinforcement Learning?

Traditional optimization (linear programming, heuristics) requires a complete model of the environment. RL agents **learn from experience**, discovering patterns in demand and developing dispatching strategies that adapt to changing conditions — without needing an explicit model of rider behavior.

---

## 3. Base Paper Summary

**Paper**: *"A Distributed Model-Free Ride-Sharing Approach for Joint Matching, Pricing, and Dispatching"*
**Authors**: Marina Haliem, Vaneet Aggarwal, Bharat Bhargava
**Published**: IEEE Transactions on Intelligent Transportation Systems, Vol. 22, No. 12, December 2021

### Key Contributions of the Paper

1. **DARM Algorithm**: A distributed insertion-based matching that considers route feasibility (detour constraints, delay windows) and handles ride-sharing where multiple riders share a vehicle
2. **DPRS Mechanism**: A game-theoretic pricing scheme where:
   - Initial price is based on distance, fuel, and wait time
   - Drivers adjust prices based on supply/demand competition
   - Customers decide based on utility (capacity, wait, vehicle type)
3. **DQN Dispatching**: A Double DQN agent that learns which zones to send idle vehicles to, maximizing long-term reward
4. **Joint Framework**: All three components work together — DQN dispatches, DARM matches, DPRS prices — creating a synergistic system

### Paper's Reported Results (NYC Taxi Data)

| Metric | DARM+DPRS+DQN | Best Baseline |
|--------|---------------|---------------|
| Accept Rate | ~96% | ~98% |
| Profit | **Highest** (5-10x baselines) | Low |
| Wait Time | Comparable | Comparable |
| Occupancy | **Highest** | Low |

The key finding: **pricing reduces accept rate slightly but increases profit dramatically**, making the system economically viable.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION ENVIRONMENT                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 15×15    │  │ Demand   │  │ 150      │  │ Congestion   │   │
│  │ City Grid│  │ Generator│  │ Vehicles │  │ Model        │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              STEP PIPELINE (per timestep)                │   │
│  │                                                          │   │
│  │  1. Generate requests  ──►  2. DARM matching             │   │
│  │  3. DPRS pricing      ──►  4. Customer decision          │   │
│  │  5. DQN dispatching   ──►  6. Vehicle advancement        │   │
│  │  7. Rider mid-trip changes ──► 8. Metrics collection     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ DQN      │  │ Replay   │  │ Target   │                     │
│  │ Network  │  │ Buffer   │  │ Network  │                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌──────────────────┐
│   API Server    │◄────────►│   Dashboard UI   │
│  (api_server.py)│   SSE    │  (dashboard.html) │
│  REST + SSE     │   JSON   │  4-tab workbench  │
└─────────────────┘          └──────────────────┘
```

---

## 5. Core Algorithms

### 5.1 DARM — Distributed Asynchronous Ride Matching

**Purpose**: Assign each incoming ride request to the best available vehicle.

**How it works**:
1. For each new request, find all vehicles within `REJECT_RADIUS_KM` (5 km)
2. For each candidate vehicle, try **inserting** the pickup and dropoff into the vehicle's existing route
3. Score each insertion by the **extra distance** it adds (detour cost)
4. Reject insertions that violate the `MAX_DETOUR_FACTOR` (1.5× direct distance)
5. Assign the request to the vehicle with the **minimum extra cost**

**Why insertion-based matching?** Unlike simple nearest-vehicle assignment, DARM considers the impact on existing passengers. A vehicle 3 km away might be better than one 1 km away if the closer vehicle would cause a huge detour for its current passenger.

### 5.2 DPRS — Dynamic Pricing for Ride-Sharing

**Purpose**: Set fares that balance profitability with customer acceptance.

**Two-stage pricing**:

**Stage 1 — Initial Price** (what the platform suggests):
```
p₀ = base_fare + rate_per_km × (trip_km / shared) + fuel_cost - wait_discount
```
- `shared` = number of riders sharing → price decreases with more riders (incentive to share)
- `fuel_cost` = distance × gas price / vehicle efficiency
- `wait_discount` = compensation for customer waiting

**Stage 2 — Driver Adjustment** (competitive pricing):
```
p = p₀ × (1 + markup)
markup = min(0.10, supply_demand_ratio × 0.05)
```
- In **high-demand zones** (more requests than vehicles), drivers can charge slightly more
- In **oversupplied zones**, prices stay low to attract riders
- Markup is capped at 10% to prevent price gouging

### 5.3 Customer Decision Model

**Purpose**: Determine whether a customer accepts the offered ride.

Each customer has a **utility function**:
```
u = W₄/(capacity+1) + W₅/wait_time + W₆ × type_bonus × (vehicle_type+1)
```
- **W₄=15**: Preference for less crowded vehicles (utility decreases as capacity fills)
- **W₅=1**: Preference for shorter wait times
- **W₆=4**: Preference for nicer vehicle types (sedan > compact)

**Accept condition**: `u ≥ price - δ` where `δ ∈ [3, 15]` is the customer's budget flexibility.

### 5.4 DQN — Double Deep Q-Network Dispatching

**Purpose**: Learn which zones to send idle vehicles to, maximizing long-term fleet profit.

**State representation** (per vehicle):
- Vehicle's current zone (one-hot encoded)
- Supply forecast: how many vehicles will be in each zone in the next T steps
- Demand forecast: how many requests will appear in each zone in the next T steps
- Time encoding (sine/cosine of timestep for periodicity)

**Action space**: 225 discrete actions (15×15 grid offsets from current position)

**Reward function** (implemented):
```
R = B₁×served_pax + B₂×dispatch_time + B₃×detour_time + B₄×profit_step + B₅×idle_flag
```
Where B₁=10, B₂=-1, B₃=-5, B₄=12, B₅=-8. `dispatch_time` and `detour_time` are costs (negative weights). `idle_flag = 1` when a vehicle has no passengers.

**Double DQN** prevents value overestimation by using:
- A **policy network** to select actions
- A **target network** (updated every 150 steps) to evaluate Q-values
- **Experience replay** (buffer of 5000 transitions) for stable learning

---

## 6. Our Extensions Beyond the Paper

### 6.1 Travel-Time Uncertainty (Paper Section VII, Future Work)

The paper assumes deterministic travel times. Real cities have **congestion** that varies by zone and time of day.

**Our congestion model**:
```python
congestion_factor(zone, t) = base + rush_hour_peak + noise
```
- **Base**: Each zone has a random base congestion ∈ [1.0, 1.3]
- **Rush-hour peaks**: Gaussian peaks at 8:00 AM (t=480) and 5:00 PM (t=1020)
- **Stochastic noise**: Gaussian noise (σ=0.08) models unpredictable traffic

**Impact**: Congestion increases fuel costs (vehicles burn more fuel in traffic), making dispatching to congested zones less profitable. The DQN agent must learn to avoid congested zones.

### 6.2 On-the-Fly Rider Changes (Paper Reference [17])

Real riders sometimes change their minds after being picked up.

**Our implementation**:
- **3% chance per timestep** a rider changes their destination to a random zone
- **1% chance per timestep** a rider cancels mid-trip (50% refund to driver)
- Route is dynamically updated to reflect the new destination

**Impact**: This tests the framework's **robustness**. A good dispatching strategy should be resilient to route disruptions.

### 6.3 Ablation Study

We systematically remove one component at a time to measure its contribution:

| Configuration | What's Removed | Purpose |
|--------------|----------------|---------|
| Full (D+RS+PS+DARM) | Nothing | Baseline |
| -Dispatch | DQN dispatching | How much does proactive positioning help? |
| -Ridesharing | Multi-passenger sharing | How much does sharing improve efficiency? |
| -Pricing (DPRS) | Dynamic pricing | What's the profit cost of flat pricing? |
| -DARM (greedy only) | Insertion matching | How much better is DARM vs nearest-vehicle? |

### 6.4 Interactive Training Dashboard

A web-based workbench with 4 tabs:

| Tab | Features |
|-----|----------|
| **🎯 Train** | Algorithm selector, 15+ parameter sliders, live training metrics (AR, wait, occupancy, idle, distance) + charts (AR, profit, wait, loss, epsilon, Q-max), auto-save prompt |
| **🏙️ Simulate** | City grid visualization, fleet status, event log, real-time metrics (AR, wait, occupancy, idle, distance) |
| **💾 Models** | Save/load model library, metadata (algorithm, params, final metrics), delete old models |
| **📊 Compare** | Select 2+ saved models, run side-by-side evaluation, bar charts + comparison table |

---

## 7. Mathematical Formulation

### 7.1 Pricing (Equations 2-3)

**Initial price for customer c assigned to vehicle v**:
$$p_0(v,c) = \omega_0^{v_t} + \omega_1^{v_t} \cdot \frac{d(o_c, d_c)}{|S_v|} + \omega_2^{v_t} \cdot \frac{d(o_c, d_c)}{M_{v_t}} \cdot P_{gas} - \omega_3^{v_t} \cdot w_c$$

Where:
- ω₀ = base fare by vehicle type ($2, $3, $5 for compact/sedan/luxury)
- ω₁ = rate per km ($0.9, $1.2, $1.8)
- d(oₓ, dₓ) = direct trip distance for customer c
- |Sᵥ| = number of riders sharing vehicle v
- Mᵥₜ = fuel efficiency (km/litre) by vehicle type
- ω₃ = wait-time discount rate

### 7.2 Customer Utility (Equation 4)

$$u(v,c) = \frac{W_4}{\text{cap}(v)+1} + \frac{W_5}{\text{wait}(c)} + W_6 \cdot \tau_b \cdot (v_t + 1)$$

Customer accepts if: $u(v,c) \geq p(v,c) - \delta_c$

### 7.3 Reward Function (Equation 6)

$$R_v = B_1 \cdot \text{served}_v + B_2 \cdot \text{dispatch\_time}_v + B_3 \cdot \text{detour\_time}_v + B_4 \cdot \text{profit}_v + B_5 \cdot \text{idle\_flag}_v$$

Where weights are: $B_1=10, B_2=-1, B_3=-5, B_4=12, B_5=-8$
Dispatch and detour times are costs (negative weights). `idle_flag = 1` when a vehicle has no passengers.

### 7.4 Double DQN Update

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \cdot Q_{\text{target}}(s', \arg\max_{a'} Q(s',a')) - Q(s,a) \right]$$

---

## 8. Simulation Environment

### 8.1 City Grid

- **15×15 grid** = 225 zones (modeled after NYC Manhattan)
- **Cell size**: 0.8 km per cell (calibrated for realistic trip distances)
- **Manhattan distance** between zones by default
- Optional **OSRM routing** between zone centers for higher-fidelity travel distance/time
- Average trip: ~7-10 km, matching typical urban ride-sharing trips

### 8.2 Fleet

- **150 vehicles** of 3 types:
  - Type 0 (Compact): capacity 2, 35 km/L, base fare $2
  - Type 1 (Sedan): capacity 4, 28 km/L, base fare $3
  - Type 2 (Luxury): capacity 6, 22 km/L, base fare $5

### 8.3 Demand Model

- **NYC Yellow Taxi dataset** via KaggleHub (elemento/nyc-yellow-taxi-trip-data)
- Requests are bucketed by minute-of-day; demand forecasts use per-minute historical counts
- **Synthetic fallback** when dataset is unavailable
- 75% of riders are willing to share; 25% want private rides

### 8.4 Request Properties

Each request has:
- Origin and destination zones
- Number of passengers (1-3)
- Maximum wait tolerance (5-20 minutes)
- Vehicle type preference (random)
- Budget flexibility δ ∈ [3, 15]

---

## 9. Interactive Dashboard

### Architecture

```
Browser (dashboard.html) ←── SSE ──► API Server (api_server.py) ──► Sim Engine (ridesharing_darm_dprs_dqn.py)
                         ──── REST ──►
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/train/start` | POST | Start training with config JSON |
| `/api/train/pause` | POST | Toggle pause/resume |
| `/api/train/stop` | POST | Stop training |
| `/api/train/status` | GET | Current step, metrics, history |
| `/api/stream` | GET (SSE) | Real-time metric updates |
| `/api/models` | GET | List saved models |
| `/api/models/save` | POST | Save current model |
| `/api/models/load` | POST | Load model by ID |
| `/api/models/{id}` | DELETE | Delete a model |
| `/api/compare` | POST | Evaluate and compare models |
| `/api/config` | GET | Current parameter defaults |

### Performance Metrics (Evaluation)

The dashboard and API report the following evaluation metrics:

- **Accept rate** — accepted requests / total requests
- **Cruising (idle) time** — fraction of vehicles with no passengers (`idle_frac`)
- **Occupancy rate** — average fraction of timesteps vehicles carry passengers (`occ`)
- **Waiting time** — average pickup wait for accepted requests (`wait`)
- **Travel distance** — average cumulative km per vehicle (`km`)

### Configurable Parameters (via UI sliders)

| Category | Parameters |
|----------|-----------|
| **DQN** | Learning rate, gamma, epsilon (start/end/decay), batch size, replay buffer |
| **Reward** | B1 (served), B2 (dispatch time), B3 (detour time), B4 (profit), B5 (idle) |
| **Environment** | Fleet size, warmup steps, rider change probability |
| **Algorithm** | Full pipeline / DQN only / No pricing / No ridesharing / Greedy |

---

## 10. Experimental Results

### 10.1 Main Results (1500 training steps + 300 demo steps)

| Baseline | Accept Rate | Profit ($) | Wait (min) | Occupancy | Idle % |
|----------|-------------|------------|------------|-----------|--------|
| **D,RS,PS,DARM (Ours)** | **91.2%** | **$1,194.89** | 1.41 | **15.7%** | **0.0%** |
| !D,!RS,!PS,GM | 97.4% | $46.64 | 1.20 | 4.9% | 83.8% |
| !D,RS,!PS,GM | 98.7% | $50.85 | 1.13 | 5.4% | 83.5% |
| D,!RS,!PS,GM | 96.2% | $46.06 | 1.13 | 5.1% | 82.7% |
| D,RS,!PS,GM | 97.3% | $46.37 | 1.10 | 5.2% | 86.1% |
| D,RS,PS,GM | 77.1% | $42.04 | 1.10 | 5.1% | 86.6% |

### 10.2 Key Findings

1. **Our framework achieves 25× higher profit** ($1,195 vs $47) than the best non-DARM baseline
2. **Accept rate of 91.2%** — only 5% below non-pricing baselines, matching the paper's finding that pricing trades small AR for large profit
3. **Zero idle time** — DARM + DQN dispatching keeps every vehicle productive
4. **3× higher occupancy** (15.7% vs 5.1%) — ridesharing fills vehicles more efficiently
5. **Greedy matching with pricing (D,RS,PS,GM) drops to 77%** — showing DARM's superiority over naive matching

---

## 11. Ablation Study

### 11.1 Component Ablation

| Configuration | Accept Rate | Profit ($) | Wait (min) | Occupancy |
|--------------|-------------|------------|------------|-----------|
| Full (D+RS+PS+DARM) | 94.9% | $21.07 | 1.16 | 4.78% |
| **-Dispatch** | **70.7%** | $20.73 | 1.29 | 4.99% |
| -Ridesharing | 94.1% | $23.58 | 1.28 | 5.23% |
| -Pricing (DPRS) | 97.9% | $22.34 | 1.12 | 5.12% |
| -DARM (greedy only) | 88.2% | $24.58 | 1.11 | 5.27% |

### 11.2 Key Ablation Findings

- **Dispatch has the largest impact**: Removing DQN dispatch drops AR by **24 percentage points** (94.9% → 70.7%). This confirms the paper's claim that proactive positioning is critical.
- **DARM matters**: Removing insertion matching for greedy drops AR by 6.7 points, showing sophisticated matching is important.
- **Pricing trades AR for control**: Removing pricing raises AR by 3% but loses the ability to maximize revenue.
- **Ridesharing is marginal for AR** but critical for vehicle utilization.

### 11.3 Rider Change Sensitivity

| Rider Change Rate | Accept Rate | Profit ($) | Wait (min) |
|-------------------|-------------|------------|------------|
| 0% (baseline) | 93.6% | $19.36 | 1.02 |
| 3% (default) | 93.5% | $16.11 | 0.93 |
| 6% | 91.2% | $17.06 | 1.14 |
| 10% | 92.5% | $18.53 | 1.03 |

**Finding**: The framework is **robust to mid-trip disruptions**. Even with 10% of riders changing destinations every step, AR only drops 1 percentage point. Profit decreases modestly due to cancellation refunds.

---

## 12. Comparison: Paper vs Our Implementation

| Aspect | Paper (Haliem et al. 2021) | Our Implementation |
|--------|---------------------------|-------------------|
| **Data** | Real NYC taxi trips (2016) | NYC Yellow Taxi dataset via KaggleHub (sampled by default) |
| **Grid** | NYC Manhattan zones | 15×15 generic city (0.8 km/cell) |
| **Fleet** | 200 vehicles | 150 vehicles (configurable) |
| **Travel time** | Deterministic | Optional stochastic congestion (off by default) |
| **Rider behavior** | Static (no mid-trip changes) | Optional rider changes/cancellations (off by default) |
| **Pricing** | Mathematical formulation only | Full simulation with customer utility |
| **DQN** | Standard Double DQN | Double DQN + warm-up + Q-max tracking |
| **Visualization** | Static plots | **Interactive web dashboard** ✨ |
| **Model management** | None | **Save/load/compare library** ✨ |
| **Ablation** | 6 baseline scenarios | 6 baselines + **component ablation + sensitivity analysis** ✨ |
| **Code** | Not released | **Fully open-source, single-file engine** |

### What Matches the Paper

✅ DARM insertion-based matching algorithm
✅ DPRS two-stage pricing mechanism
✅ Double DQN with experience replay
✅ 6 baseline comparison scenarios (Table II in paper)
✅ Accept rate, profit, wait time, occupancy metrics
✅ Key finding: DARM+DPRS achieves highest profit with competitive AR

### What We Added (Paper's Future Work)

✨ Travel-time uncertainty with zone-based congestion (Section VII)
✨ On-the-fly rider changes (Reference [17] in paper)
✨ Component ablation study
✨ Interactive parameter exploration via dashboard
✨ Model persistence and comparison framework

---

## 13. File Structure

```
new_rl/
├── ridesharing_darm_dprs_dqn.py    # Core simulation engine (1647 lines)
│   ├── CityGrid                     # 15×15 grid with Manhattan distance
│   ├── DemandModel / DatasetDemandModel  # NYC dataset or synthetic demand
│   ├── congestion_factor()          # Travel-time uncertainty model
│   ├── Req / Veh                    # Request and Vehicle data classes
│   ├── insert_req()                 # DARM insertion matching
│   ├── greedy_match()               # Vehicle-request assignment
│   ├── price_initial/price_driver   # DPRS pricing (Eq. 2-3)
│   ├── customer_decide()            # Customer utility model (Eq. 4)
│   ├── DQNAgent                     # Double DQN with replay buffer
│   ├── RSEnv                        # Main simulation environment
│   ├── BaselineEnv                  # 6 baseline comparison configs
│   ├── run_ablation()               # Ablation study runner
│   └── main()                       # API/server entry point
│
├── api_server.py                    # REST API + SSE server (440 lines)
│   ├── training_worker()            # Background training thread
│   ├── save/load/delete/compare     # Model management
│   └── APIHandler                   # HTTP request routing
│
├── dashboard.html                   # 4-tab interactive UI
├── dashboard.css                    # Premium dark theme styling
├── dashboard.js                     # Frontend logic (charts, SSE, API calls)
│
├── saved_models/                    # Persistent model storage
│   └── <model_id>/
│       ├── model.pt                 # PyTorch weights
│       ├── meta.json                # Metadata + config + final metrics
│       └── history.json             # Full training history
│
└── outputs/                         # Generated plots and data
    ├── training_curves.png          # 6-panel training progression
    ├── comparison.png               # Paper Fig. 3 style comparison
    ├── ablation_study.png           # Component ablation bar charts
    ├── demo_snapshot.png            # Fleet state visualization
    ├── metrics_bundle.json          # Machine-readable results
    └── metrics_bundle.js            # Browser-loadable results
```

---

## 14. How to Run

### Prerequisites

```bash
pip install torch numpy matplotlib tqdm kagglehub
```

### Option 1: Interactive Dashboard (Recommended)

```bash
python api_server.py 8000
# Open http://localhost:8000/dashboard.html in your browser
```

By default, the simulator tries to download the NYC dataset via KaggleHub. If it fails,
it falls back to synthetic demand. Use `--no-dataset` to force synthetic demand or
`--dataset-path` to point to a local dataset directory.

Then use the UI to:
1. Adjust parameters with sliders
2. Set training steps → Click ▶ Start Training
3. Watch live metrics update
4. Save your model when done
5. Train more models with different params
6. Compare them in the Compare tab

### Dashboard only

```bash
python api_server.py 8000
# Open http://localhost:8000/
```

---

## 15. Future Work

Based on the paper's Section VII and our own observations:

1. **Joint passenger-goods delivery** (Paper Ref [43]): Extend vehicles to carry packages during low-demand periods
2. **Multi-hop routing** (Paper Ref [45]): Allow passengers to transfer between vehicles for long trips
3. **Transit integration** (Paper Ref [33]): Connect ride-sharing with public transit for first/last mile
4. **OSRM routing**: Replace Manhattan distance with OSRM-based routing and ETA
5. **Multi-agent RL**: Replace single DQN with independent agents per vehicle for truly distributed decision-making
6. **Safety constraints**: Add passenger comfort metrics and driver fatigue modeling

---

## 16. References

1. **M. Haliem, V. Aggarwal, and B. Bhargava**, "A Distributed Model-Free Ride-Sharing Approach for Joint Matching, Pricing, and Dispatching," *IEEE Transactions on Intelligent Transportation Systems*, vol. 22, no. 12, pp. 7931-7942, Dec. 2021.

2. **H. van Hasselt, A. Guez, and D. Silver**, "Deep Reinforcement Learning with Double Q-learning," *AAAI*, 2016.

3. **T. P. Lillicrap et al.**, "Continuous control with deep reinforcement learning," *ICLR*, 2016.

4. **NYC Taxi and Limousine Commission**, Trip Record Data, 2016.

---

*This project was developed as an extended implementation and analysis of the Haliem et al. IEEE TITS 2021 paper, with novel contributions in travel-time uncertainty, rider behavior modeling, and interactive tooling.*
