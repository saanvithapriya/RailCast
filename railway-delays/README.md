# RailCast — Railway Network Intelligence

RailCast is a railway network analytics and delay intelligence system for Indian Railways. It models high-density trunk routes as directed graphs (stations and junctions as nodes, block sections as edges), identifies how delays propagate through junctions, and predicts downstream arrival delays 1 to 6 hours ahead using strictly observable pre-cutoff states.

> **Illustrative Data Disclosure:** RailCast currently uses simulated railway data based on the Corridor B (New Delhi to Pt. Deen Dayal Upadhyaya Jn) timetable. Results are illustrative and are not live Indian Railways operational data.

---

## Key Features

- **🌐 Network Overview:** High-level summary of Corridor B physical topology (108 stations, 214 directed sections, 794.2 km), physical trunk segments, active passenger services, and baseline delay distributions.
- **🗺️ Railway Network Explorer:** Searchable station directory and inspector detailing chain distances from New Delhi, junction classifications, and corridor segment groupings.
- **⚡ Delay Insights & Downstream Forecasting:**
  - **Critical Chokepoint Identification:** Ranked bottleneck sections (such as `ETW-EKL` and `CNB-CNBI`) responsible for the highest primary delay propagation.
  - **Multi-Horizon Forecasting:** Predict downstream arrival delays across 1h, 2h, 3h, 4h, 5h, and 6h horizons using machine learning models trained on pre-cutoff observables.
  - **Operational Interpretations:** Contextual insights explaining whether delays compound or stabilize based on headway coupling and weather conditions.

---

## Project Structure

```
railway-delays/
├── data/
│   ├── raw/            # Timetable and station data
│   └── processed/      # Cleaned graph data, simulations, and trained model bundles
├── src/
│   ├── app/            # Web application server, static UI, and Streamlit app
│   │   ├── static/     # Clean single-page application (HTML/CSS/JS)
│   │   ├── server.py   # Production HTTP server and REST API
│   │   ├── streamlit_app.py # Streamlit multi-page dashboard
│   │   └── data_loader.py   # Cached data access layer
│   ├── data/           # Timetable cleaning and validation
│   ├── graph/          # Station and timetable network construction
│   ├── simulation/     # Train movement and delay propagation engine
│   ├── analysis/       # Cascade mining and propagation algorithms
│   └── models/         # Multi-horizon feature pipelines and predictors
├── reports/            # Analysis summaries and figures
├── tests/              # Comprehensive automated regression suite
├── requirements.txt    # Pinned Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows 11 / Linux / macOS

### Installation (Windows PowerShell)

```powershell
cd railway-delays
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If PowerShell script execution is restricted:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install --upgrade pip
pip install -r requirements.txt
```

*Note on PyTorch:* If installing on CPU:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

---

## Running the Application

RailCast offers two user interfaces:

### Option 1: Web Application & REST API (Recommended)
Launches the built-in HTTP service with the streamlined single-page interface:
```powershell
python -m src.app.server --port 8501
```
- Open browser at: **`http://localhost:8501`**
- Includes REST API endpoints (`/api/status`, `/api/stations`, `/api/trains`, `/api/critical_sections`, `/api/predict`).

### Option 2: Interactive Streamlit Dashboard
Launches the Streamlit application:
```powershell
python -m streamlit run src/app/streamlit_app.py
```
- Open browser at: **`http://localhost:8501`**

---

## Preparing Data & Models (Reproducibility)

Precomputed artifacts and model bundles are included in `data/processed/`. If you wish to re-generate the pipeline from scratch:

1. **Clean and Validate Timetable:**
   ```powershell
   python -m src.data.clean
   python -m src.data.validate
   ```

2. **Run Train Simulation & Disruptions:**
   ```powershell
   python -m src.simulation.scenarios
   ```

3. **Construct Network Graphs:**
   ```powershell
   python -m src.graph.build
   ```

4. **Train Multi-Horizon Delay Forecasting Pipeline:**
   ```powershell
   python -m src.models.run_pipeline --stride 4
   ```

5. **Verify Inference CLI:**
   ```powershell
   python -m src.models.predict --sample
   ```

---

## Testing & Quality Assurance

The project includes an extensive test suite verifying train movement mechanics, graph topology, cascade mining invariance, anti-leakage compliance, model inference, server endpoints, and app interfaces:

```powershell
pytest
```

**Test Status:** 121 passed, 6 skipped (cleanly skipped when multi-gigabyte raw simulation tables are omitted).

---

## Known Assumptions & Limitations

- **Simulated Delays:** No open dataset of minute-level operational Indian Railways delays is publicly available. Delay cascades are simulated using a train-movement engine running on the pre-2018 Corridor B timetable.
- **Daily Running Assumption:** Because the historical timetable lacks day-of-week metadata, all 225 passenger trains run daily in the simulation, creating denser traffic than real-world conditions.
- **Strict Single-Lane Movement:** The simulation models one lane per direction between consecutive core stations with a strict 4-minute minimum headway. In reality, Indian Railways utilizes dynamic loop lines, multi-aspect signaling blocks, and dispatch overrides.
- **Illustrative Results:** Predictions demonstrate machine learning and graph mining capabilities on networked infrastructure; they should not be used for actual railway operations.
