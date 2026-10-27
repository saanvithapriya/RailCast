# RailCast: Graph-Based Mining and Prediction of Cascading Delays in the Indian Railways Network
## Final Project Research Report (Phases 0–8)

> **MANDATORY SCIENTIFIC CAVEAT:** All delay figures, timetable simulations, cascade trees, and prediction metrics in this research report originate strictly from the **SIMULATED** Corridor B environment. They do not describe, evaluate, or represent real-world Indian Railways operations.

---

## 1. Executive Summary & Problem Formulation
The Indian Railways network represents one of the world's most heavily loaded mixed passenger-and-freight rail systems. On high-density trunk routes such as **Corridor B (New Delhi to Pt. Deen Dayal Upadhyaya Jn, 794.2 km)**, train movements are tightly coupled by headway spacing, shared bidirectional junctions, and limited loop overtaking capacities. A primary disruption to a single train—such as an engineering failure, signal fault, or weather restriction—inevitably forces following and conflicting services to yield, triggering widespread **cascading delay propagation**.

**RailCast** provides a unified computational framework to:
1. Model the railway corridor as a spatio-temporal network graph.
2. Simulate realistic microscopic train movements and synthetic micro-disruptions across multi-year scenarios.
3. Unsupervisedly mine delay cascades, contagion rates, and topological bottlenecks without prior knowledge of root causes.
4. Predict downstream train arrival delays across multiple forecast horizons (1 to 6 hours ahead) using graph-enriched machine learning under strict temporal isolation.
5. Provide an interactive web dashboard for dispatchers and transport researchers to explore network vulnerability and test predictive interventions.

---

## 2. Corridor B Network & Timetable Pipeline (Phases 1–3)
- **Corridor Topology:** 108 stations, 214 directed section edges spanning 794.2 km.
- **Junction Merger:** ALD, NYN, and COI are merged into a unified node (`PRYG`) reflecting dispatching realities.
- **Timetable:** 225 active trains per day across Rajdhani, Shatabdi, Duronto, Superfast, Mail/Express, and Passenger classes. Because raw timetables lack day-of-week metadata, all services run daily, generating a dense, stress-tested synthetic operating regime.

---

## 3. Microscopic Delay Simulation Engine (Phase 2B)
To evaluate mining and predictive algorithms where real open delay logs are unavailable:
- **Coupling Mechanics:** 4-minute minimum headway coupling, single-lane unidirectional tracks without passing inside sections, and priority yielding at stations with limited loop capacity.
- **Disruption Sources:** Random freight blockers (rate 0.04/hr), heavy-tailed section incidents, signal failures, dwell extensions, and synthetic seasonal weather regimes (winter fog and monsoon rain).
- **Slack Recovery:** Section run time slack (5–8%) and dwell slack (50%).
- **Attribution Guard:** Every simulated delay minute is labelled with its internal root cause and causing train for validation purposes only. **These ground-truth tables are strictly quarantined from all mining algorithms and prediction models.**

---

## 4. Delay Cascade Mining & Network Criticality (Phases 4–5)
Cascade mining identifies how localized delays propagate across trains and sections:
- **Delay-Threshold Mining:** Tracks consecutive train delay increments above temporal propagation thresholds.
- **Mining Robustness (v1 vs v2):**
  - *Method v1:* Fixed 5-minute thresholding identifies 1,642 cascade trees.
  - *Method v2:* Dynamic headway-relative thresholding reduces false merges, yielding 1,418 cleaner propagation trees.
  - *Robustness:* Spearman rank correlation between section bottleneck rankings across v1 and v2 is $\rho = 0.89$, confirming that bottleneck identification is invariant to the specific threshold definition.
- **Top Bottleneck Sections:**
  1. `ETW-EKL` (Up): 454 origin cascades, 20,135 primary delay minutes.
  2. `CNB-CNBI` (Down): 274 origin cascades, 11,840 primary delay minutes.
  3. `ARW-JEP` (Down): 237 origin cascades, 10,220 primary delay minutes.

---

## 5. Downstream Delay Prediction Pipeline (Phase 6)

### Temporal Split Isolation
To guarantee leakage-free operational forecasting:
- **Training Set:** Calendar Year 2024 and Jan–Aug 2025.
- **Validation Set:** Sep–Oct 2025 (2 months, non-fog regime).
- **Test Set:** Nov–Dec 2025 (2 months, untouched until final evaluation; contains 10 synthetic winter fog days).

### Feature Engineering & Anti-Leakage
- **Observable Cutoff Rule:** Features are derived exclusively from observations timestamped at or before the forecast cutoff time.
- **Feature Set:** Current train delay, rolling section statistics (30m, 60m active trains, mean delay, max delay), graph centrality (degree, betweenness, section criticality), timetable context (scheduled departure hour, day of week), and ambient weather regime.
- **Regression Verification:** Automated tests (`test_prediction_leakage.py`) mathematically confirm zero leakage from future timestamps, unobserved incidents, or ground-truth cascade tables.

### Benchmark Evaluation on Untouched Test Set (SIMULATED)

| Horizon | Samples | Persistence MAE | Historical MAE | LightGBM/XGB MAE | RMSE | Gain vs Baseline |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1h** | 219,383 | 3.71 min | 15.75 min | **3.57 min** | 7.30 min | **+3.78%** |
| **2h** | 186,410 | 6.97 min | 16.40 min | **6.46 min** | 11.16 min | **+7.34%** |
| **3h** | 158,124 | 10.10 min | 17.29 min | **8.97 min** | 14.80 min | **+11.16%** |
| **4h** | 134,921 | 13.13 min | 18.25 min | **11.22 min** | 18.12 min | **+14.56%** |
| **5h** | 115,310 | 16.16 min | 19.28 min | **13.01 min** | 20.80 min | **+19.45%** |
| **6h** | 98,740 | 19.43 min | 20.48 min | **14.86 min** | 23.53 min | **+23.55%** |
| **Macro Avg** | — | 11.58 min | 17.91 min | **9.68 min** | 15.95 min | **+16.42%** |

*Winter Fog Breakdown:* During synthetic fog episodes in December 2025, MAE rises to 5.12m at 1h and 22.84m at 6h, highlighting the compounding impact of visibility reductions on headway spacing.

---

## 6. Interactive Streamlit Dashboard (Phase 7)
The application (`src/app/streamlit_app.py`) provides an interactive interface across six pages:
1. **Network Overview:** Network metrics, track segments, and simulation notices.
2. **Railway Network Explorer:** Stations, chain distances, junction indicators, and topological features.
3. **Cascade Mining Results:** Critical section rankings, propagation dynamics, and v1 vs v2 robustness.
4. **Downstream Delay Prediction:** Real-time multi-horizon inference (1h to 6h) loaded from persisted model bundles.
5. **Model Evaluation:** Quantified evaluation tables, residual error distributions, and winter fog comparisons.
6. **Data & Methodology:** Comprehensive explanation of simulator mechanics, timetable assumptions, and scientific limits.

---

## 7. Verification & Quality Assurance (Phase 8)
- **Regression Suite:** 114 passing tests, 0 failures, 6 cleanly skipped (due to absent multi-gigabyte raw parquets).
- **Environment Compatibility:** Fully verified on Windows PowerShell with offline Python 3.13 virtual environment.
- **Reproducibility:** All seeds (42 and 2024), feature transformation pipelines, and model artifacts are deterministically saved and executable via documented CLI commands.

---

## 8. Limitations & Future Work
1. **Synthetic Nature:** Results evaluate algorithmic efficacy in a controlled synthetic environment; validation against empirical Indian Railways NTES delay telemetry remains necessary.
2. **Simplified Infrastructure:** Real corridors feature triple/quadruple tracks, automatic block signalling divisions, dynamic dispatching loops, and crew/power rostering constraints not captured in the 1-lane model.
3. **Future Extension:** Integration of real-time GPS telemetry feeds, dynamic junction dispatching optimization, and bidirectional multi-corridor network expansion.
