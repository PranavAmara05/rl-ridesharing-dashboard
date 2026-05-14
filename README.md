# DARM + DPRS + DQN: Intelligent Ride-Sharing Analytical Workbench

## An Advanced Implementation of Haliem et al. (IEEE TITS 2021) with High-Fidelity Simulation and RL Diagnostics

This project transforms the research findings of Haliem et al. into a professional-grade **Analytical Workbench**. It doesn't just implement the DARM+DPRS+DQN framework; it provides a suite of diagnostic tools to visualize, measure, and optimize reinforcement learning (RL) behavior in urban mobility.

---

## 🚀 The Analytical Workbench Paradigm

While the original paper provides static results, this implementation treats the ride-sharing problem as an interactive experiment. The system is designed to expose the "black box" of RL dispatching through three pillars of analysis:

### 1. Behavioral Diagnostics (The "Why")
- **Rejection Analytics**: Instead of just tracking "Accept Rate," the system decomposes rejections into **Price-driven** vs. **Wait-driven**. This reveals whether the pricing model is too aggressive or the fleet is too sparse.
- **Profit Fairness (Gini Coefficient)**: Implements the Gini Coefficient to measure how evenly profit is distributed across the fleet. This prevents "super-driver" scenarios and ensures system-wide stability.
- **Value Mapping (Q-Map)**: A toggleable heatmap overlay that visualizes the DQN agent's internal Q-values for every city zone, showing where the agent *perceives* the highest long-term value.

### 2. Engineering Excellence (The "How")
- **Delta-Compressed SSE**: To handle real-time fleet updates without lagging the browser, the API implements delta-compression, streaming only the modified state components.
- **Adaptive $\epsilon$-Decay**: To prevent convergence to local optima, the agent monitors the Accept Rate. If it stagnates for 50 steps, the exploration rate ($\epsilon$) is automatically increased, forcing new strategy discovery.
- **Auto-Checkpointing**: The system automatically tracks and saves the "Best-Model" based on a balanced Profit $\times$ Accept Rate peak, ensuring the most robust policy is persisted.
- **Verified Logic**: A comprehensive unit test suite validates the core DARM insertion logic, DPRS pricing equilibrium, and customer utility models.

### 3. Visual Fidelity (The "What")
- **Position Interpolation**: Vehicles no longer "jump" between zones. Using linear interpolation (LERP), vehicles glide smoothly across the grid, making transit visible and intuitive.
- **High-Transparency UI**: A detailed fleet legend, Model ID pills, and real-time passenger counts provide a clear view of the simulation state.
- **Semantic Event Log**: A color-coded live feed exposes the agent's internal reasoning (e.g., "DQN: Dispatching to Zone 42", "Customer: Rejected due to Price").

---

## 🛠️ System Architecture

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
│  REST + SSE     │   JSON   │  Analysis Suite   │
└─────────────────┘          └──────────────────┘
```

---

## 🧠 Core Algorithms

### DARM — Distributed Asynchronous Ride Matching
Matches riders to vehicles using an **insertion-based** approach. Instead of nearest-neighbor, DARM checks if a new request can be inserted into a vehicle's route without violating:
- **Detour Constraint**: Total route length $\le 1.5\times$ direct distance.
- **Delay Window**: Pickup occurs within the rider's max wait tolerance.
- **Capacity**: Vehicle does not exceed its max passenger limit.

### DPRS — Dynamic Pricing for Ride-Sharing
A game-theoretic pricing model consisting of two stages:
1. **Initial Price**: Based on distance, fuel efficiency, vehicle type, and wait time.
2. **Driver Markup**: Drivers adjust the price based on the destination zone's Q-value (perceived future reward).

### DQN — Double Deep Q-Network Dispatching
Learns optimal "empty-vehicle" repositioning.
- **State**: Current zone + Supply/Demand forecasts + Time features.
- **Action**: Dispatch to one of 225 city zones.
- **Reward**: $\text{Served Pax} + \text{Profit} - \text{Dispatch Cost} - \text{Detour Cost}$.

---

## 🏙️ Simulation Grounding: The NYC Dataset

This framework is grounded in real-world urban mobility using the **NYC Yellow Taxi dataset** (via KaggleHub).

**Why this matters:**
- **Authentic Flow**: Captures real origin-destination patterns, avoiding the pitfalls of synthetic "random" demand.
- **Temporal Periodicity**: Implements realistic rush-hour peaks (8 AM / 5 PM), forcing the DQN agent to learn time-dependent strategies.
- **Calibration**: The $15\times 15$ grid and $0.8\text{km}$ cell size are calibrated to match typical Manhattan trip distributions.

---

## 📊 Interactive Dashboard Features

### 🚂 Training & Optimization
- **Hyperparameter Tuning**: Real-time sliders for LR, $\gamma$, $\epsilon$, and Reward Weights ($B_1$-$B_5$).
- **Algorithm Ablation**: Toggle DARM, DPRS, or DQN on/off to measure individual component contribution.
- **Live Curves**: 6-panel chart tracking AR, Profit, Wait, Loss, $\epsilon$, and Q-max.

### 🏙️ Live Simulation
- **High-Fidelity Grid**: Visualizes vehicle status (Idle $\rightarrow$ Dispatching $\rightarrow$ Occupied $\rightarrow$ Carrying).
- **Smooth Transit**: Position interpolation prevents "teleporting" and shows actual vehicle movement.
- **Demand Heatmap**: Visualizes high-demand hotspots in real-time.

### 🔬 Analysis Suite
- **Q-Map Overlay**: See what the RL agent is "thinking" by visualizing zone values.
- **Gini Gauge**: Track profit fairness across the fleet.
- **Rejection Breakdown**: Analyze why customers are rejecting rides (Price vs. Wait).

---

## 🛠️ How to Run

### 1. Prerequisites
```bash
pip install torch numpy matplotlib tqdm kagglehub
```

### 2. Launch the Workbench
```bash
python api_server.py 8000
```
Open **`http://localhost:8000/`** in your browser.

### 3. Example Workflow
1. **Train**: Go to **Train Tab** $\rightarrow$ Adjust parameters $\rightarrow$ Start Training $\rightarrow$ Save Best Model.
2. **Analyze**: Go to **Simulate Tab** $\rightarrow$ Load Model $\rightarrow$ Run Demo $\rightarrow$ Toggle **Q-Map Overlay**.
3. **Verify**: Check the **Semantic Event Log** to trace individual matching and pricing decisions.

---

## 📂 File Structure
- `api_server.py`: Unified REST/SSE server.
- `ridesharing_darm_dprs_dqn.py`: Core RL engine and simulation logic.
- `dashboard.html/js/css`: The interactive analytical frontend.
- `tests/test_rl_engine.py`: Unit tests for core algorithms.

## 📝 References
1. **M. Haliem et al.**, "A Distributed Model-Free Ride-Sharing Approach for Joint Matching, Pricing, and Dispatching," *IEEE TITS*, 2021.
2. **Van Hasselt et al.**, "Deep Reinforcement Learning with Double Q-learning," *AAAI*, 2016.
