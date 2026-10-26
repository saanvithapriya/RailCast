# Mining Cascading Delays in the Indian Railways Network

This project models the Indian Railways network as a graph (stations and junctions as nodes, train routes as edges), mines how delays propagate and cascade through junctions, and builds models that predict downstream delays 1-6 hours ahead. It combines graph analysis (NetworkX), classical machine learning (scikit-learn, XGBoost, LightGBM), graph neural networks (PyTorch Geometric), and an interactive Streamlit app for exploring results.

> Status: Phase 0 (project setup) only. No analysis or modelling code yet.

## Folder layout

```
railway-delays/
├── data/
│   ├── raw/            # original downloaded data (git-ignored)
│   └── processed/      # cleaned / derived data (git-ignored)
├── notebooks/          # exploration and environment checks
├── src/
│   ├── data/           # loading and cleaning
│   ├── graph/          # network construction
│   ├── analysis/       # delay-propagation mining
│   ├── models/         # prediction models
│   └── app/            # Streamlit app
├── reports/            # figures and write-ups
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup (Windows 11, PowerShell)

```powershell
cd railway-delays
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used. PyTorch is the CPU build; if `pip install -r requirements.txt` cannot find it, install it explicitly first:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

Verify the environment by running `notebooks/00_setup_check.ipynb`:

```powershell
jupyter notebook notebooks/00_setup_check.ipynb
```

## Phase 2: cleaned Corridor B timetable

```powershell
python -m src.data.clean      # writes data/processed/*.parquet + csv and reports/phase2_cleaning_log.md
python -m src.data.validate   # exits non-zero if any check fails
```

Rules, counts and known data problems are in `reports/phase2_cleaning_log.md`.

## Synthetic data

> **All delay data produced by `src/simulation` is SIMULATED.** No open dataset of real Indian Railways delays exists for this project, so the simulator runs the real (old, pre-2018) Corridor B timetable through a train-movement model and injects random disruptions. Findings from it say whether mining and prediction methods work in a world we designed, not how Indian Railways behaves.

```powershell
python -m pytest tests -q                      # unit tests for the movement rules and delay sources
python -m src.simulation.pilot                 # January + June 2025 pilot -> data/processed/sim_pilot/
python -m src.simulation.scenarios             # low / medium / high congestion, full 2025 -> data/processed/sim/
```

Parameters live in `src/simulation/config.yaml` (seed 42); the design is in `reports/phase2b_simulator_design.md`.

**What is modelled:** one lane per direction between consecutive core stations, a minimum headway, no passing inside a section, priority yielding and overtaking only at junctions or real halts, slack recovery and shortened halts, rake reuse between a train and its return service, random freight blockers, section incidents (heavy tailed), halt extensions, block/signal disruptions, synthetic fog/rain (swappable weather source) and an unknown incoming delay for trains that enter mid-route. Every delay minute is labelled with its cause, causing train and root event (`delay_attribution`).

**This world is cascade-heavy compared with the real corridor.** In the simulated data more than half of all delay minutes (about 59-75% depending on the scenario) are caused by trains waiting for, or yielding to, other trains, and queues of up to 17-20 trains form over a year and every section sees queues of 3 or more. The real corridor is very likely less coupled than this. Reasons: (1) the timetable has no day-of-week information, so every train runs every day, which makes traffic much denser than it really is; (2) the model has one lane per direction between stations, no passing inside a section and no signalling blocks, loops or parallel lines beyond the loop-capacity rule, so a delayed train directly holds up those behind it; (3) there is no real slack or recovery data, so recovery is a flat 5-8% of section time; (4) freight is random noise. Treat cascade counts, queue lengths and "share of delay caused by other trains" from this data as properties of the simulator, not estimates for Indian Railways.

**Key assumptions and limitations**
- Every train runs daily (the timetable has no day-of-week information). Pairs of trains the timetable itself schedules closer than the headway keep their scheduled spacing, otherwise the daily assumption alone creates large artificial delays.
- Scheduled times at pass-through stations are interpolated between real halts; raw times are kept in `raw_scheduled_*`.
- Prayagraj (ALD, NYN, COI) is one node; the 39 trains that leave the main line use a fixed-time off-corridor leg.
- Priority trains can overtake at any station with limited loop capacity (2 trains held aside, 2-minute penalty); the earlier junction-or-halt rule is available as `movement.overtake_mode`.
- The yield look-ahead is one section (`movement.yield_lookback_sections: 1`); two sections was tried once and made results worse. Four Shatabdi trains (12033, 12003 mainly) are delayed more than their priority suggests; this is a known model artifact.
- No platform capacity, crews or locomotives.
- The Kanchausi to Para Jani Halt segment (0.12 km apart, 16-22 minutes in the timetable) is a data quirk, not a bottleneck.
- Parameters were set by hand and not fitted to real data; the check targets (70-85% within 10 minutes, 15-35% of delay caused by other trains) are assumptions and are not met.

**Scenarios (calendar 2025, written to `data/processed/sim/{low,medium,high}/`):** headway 3 / 4 / 6 minutes with freight 0.02 / 0.04 / 0.06 per section-hour, everything else equal, same seed and weather. Run with `python -m src.simulation.scenarios`. Summary table in `reports/phase2b_simulator_design.md` section 13.

## Graphs (Phase 3)

`python -m src.graph.build` builds the Corridor B graphs into `data/processed/graph/`: a station graph (108 nodes, 214 directed edges, Prayagraj merged into one node), a train-dependency graph (timetable only), delay attributes for the training window (Jan-Aug 2025) and for all of 2025 (analysis only), and a snapshot lookup. Delay-based parts use the **SIMULATED** medium scenario. The ground-truth cascade tables (`data/processed/sim/medium/groundtruth_cascades*.parquet`) are **for Phase 5 validation only**; no mining or prediction method may read them. Details: `reports/phase3_graph_notes.md`; visuals: `notebooks/04_graphs.ipynb`.

## Time split and simulated data sources

Defined once in `configs/splits.yaml` (read through `src/splits.py`): **train = all of 2024 + Jan-Aug 2025, validation = Sep-Oct 2025, test = Nov-Dec 2025**, strictly by time. The medium scenario is two simulation runs of the same timetable: 2025 (seed 42, `data/processed/sim/medium/`) and 2024 (seed 2024, `data/processed/sim/medium_2024/`, made with `python -m src.simulation.scenarios --medium-2024`). Weather and incidents differ between the runs; the timetable, headway and every other parameter are the same, so it is the same simulated world twice, not two real years. Fog is synthetic and falls in Nov-Feb: the training window contains full fog seasons (48 fog days), the validation window has none, the test window has 10 (all in December). Details: `reports/phase3_graph_notes.md`, section 7.

**Rolling-origin folds** (also in `configs/splits.yaml`): fold 1 trains on 2024-01-01..2024-10-31 and validates on Nov-Dec 2024 (12 fog days); fold 2 trains on 2024-01-01..2025-08-31 and validates on Sep-Oct 2025 (no fog). Select models on the mean over both folds and report fog performance separately; the test window (Nov-Dec 2025) is untouched. See `reports/phase3_graph_notes.md`, section 8.

## Exploratory analysis (Phase 4)

`notebooks/05_eda.ipynb` and `reports/phase4_findings.md` explore the SIMULATED medium-scenario delays. Decisions about modelling use only 2024-01-01..2024-10-31 (fold 1's training range); later months of the training window appear only in descriptive plots; validation fold 2 and the test window are never loaded. Helpers: `src/analysis/eda.py` (tests in `tests/test_eda_guard.py` check the window and that no label tables or columns are read).
