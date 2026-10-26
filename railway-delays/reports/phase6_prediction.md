# Phase 6: Downstream Train Delay Prediction (SIMULATED Medium Scenario)

> **CRITICAL CAVEAT:** All delay data, incident injection, weather states, and prediction outcomes analyzed and presented below come exclusively from the **SIMULATED** Corridor B medium scenario (headway 4 min, freight rate 0.04). None of these results describe or estimate real Indian Railways operating performance. Findings quantify the capability of mining, baseline, machine learning, and graph neural network methods on a synthetic world with explicit headway coupling, single-lane unidirectional tracks, and flat slack recovery.

---

## 1. Problem Formulation & Prediction Unit

### 1.1 Prediction Unit
The prediction unit is defined as an **eligible active train arrival observation** at a corridor station (or junction) at observation cutoff timestamp $t_{\text{now}}$.

At time $t_{\text{now}}$, railway dispatchers and passengers observe the current train state, its current arrival delay, and the prevailing corridor traffic. The goal is to forecast the future arrival delay of that specific train as it travels downstream along the corridor.

### 1.2 Target Definition
We support six explicit forecast horizons:
- **1 hour ahead** ($h=1$)
- **2 hours ahead** ($h=2$)
- **3 hours ahead** ($h=3$)
- **4 hours ahead** ($h=4$)
- **5 hours ahead** ($h=5$)
- **6 hours ahead** ($h=6$)

**Formal Target Definition:**
$$Y^{(h)} = \text{Actual downstream arrival delay (in minutes) at the station reached closest to } t_{\text{now}} + 60 \cdot h \text{ minutes}$$
subject to an arrival matching tolerance window of $\pm 35$ minutes.

**Boundary and Termination Handling:**
If a train terminates its journey, exits the modeled corridor, or has no recorded station arrival within $t_{\text{now}} + 60 \cdot h \pm 35$ minutes, the target $Y^{(h)}$ is strictly recorded as `NaN`. Such instances are excluded from the horizon evaluation set rather than synthetically imputed or extrapolated from unavailable future records.

---

## 2. Data Sources & Observable Features

### 2.1 Sources and Cadence
- **Training Source:** `data/processed/sim/medium_2024` (seed 2024) and `data/processed/sim/medium` (seed 42), clipped strictly to `[2024-01-01, 2025-08-31]`.
- **Validation Source:** `data/processed/sim/medium` clipped to `[2025-09-01, 2025-10-31]`.
- **Test Source (Held-out Untouched):** `data/processed/sim/medium` clipped to `[2025-11-01, 2025-12-31]`.
- **Sampling Cadence:** Evaluated on arrival observations at a 4-station stride ($\approx 30$-minute sampling along each journey) while preserving 100% of real scheduled halts and junction events. Over 2.35 million training examples and 235k test examples are constructed.

### 2.2 Feature Schema
All features represent information observable at or before prediction time $t_{\text{now}}$:

| Category | Feature Name | Description |
|---|---|---|
| **Temporal** | `sched_hour`, `sched_min` | Scheduled arrival time of day (0..23, 0..59) |
| | `hour_sin`, `hour_cos` | Cyclical sin/cos encodings of hour ($\sin(2\pi h / 24)$, $\cos(2\pi h / 24)$) |
| | `day_of_week` | Day of week index (0=Monday .. 6=Sunday) |
| | `month` | Calendar month (1..12) |
| | `season_code` | Categorical season index (0=Jan-Feb fog, 1=Mar-May, 2=Jun-Aug, 3=Sep-Oct, 4=Nov-Dec fog) |
| | `elapsed_corridor_min` | Minutes elapsed since train entered the modeled corridor ($t_{\text{now}} - t_{\text{entry}}$) |
| | `progress_ratio` | Stop sequence / total stops along corridor ($seq / seq_{\max}$) |
| **Train & Section** | `current_delay` | Observed arrival delay in minutes at $t_{\text{now}}$ |
| | `prev_dep_delay` | Departure delay at the immediate preceding station |
| | `delay_change_recent` | Recent delay change across preceding section (`current_delay - prev_dep_delay`) |
| | `rolling_mean_delay_3` | Rolling mean arrival delay over the 3 most recent stops |
| | `train_type_code` | Categorical train class (Rajdhani, Duronto, Shatabdi, Superfast, Mail/Express, Passenger, other) |
| | `priority` | Static priority tier (1 to 6) |
| | `current_node` | Station node index (0 to 107) |
| | `is_junction` | Binary indicator for major junction stations |
| | `km_from_new_delhi` | Distance in kilometers along the main-line chain from New Delhi |
| | `direction` | Direction of travel (+1 for Down / toward Mughalsarai, -1 for Up / toward New Delhi) |
| **Network & Local State** | `station_lead_delay` | Observed arrival delay of the immediately preceding train at the same station/direction (gap $\le 60$ min) |
| | `station_lead_gap_min` | Time gap in minutes since the leading train passed (capped at 120 min) |
| | `node_hist_mean_delay` | Historical mean arrival delay of the current node (fitted on training data only) |
| | `node_hist_p90_delay` | Historical 90th percentile arrival delay of the current node (training data only) |
| **Environmental** | `is_fog` | Binary indicator for fog weather on prediction date |
| | `is_rain` | Binary indicator for rain weather on prediction date |

### 2.3 Anti-Leakage Controls
1. **Forbidden Simulator Labels:** Columns `cause_code`, `cause_train`, `root_event_id`, `primary_delay_injected`, and tables `groundtruth_cascades` and `delay_attribution` are strictly blocked from feature construction and target generation.
2. **Strict Chronological Windows:** Enforced from `configs/splits.yaml`. Training preprocessing statistics (station historical means, default imputers) are fitted exclusively on training rows and applied frozen to validation and test data.
3. **Dedicated Leakage Unit Tests:** `tests/test_prediction_leakage.py` verifies that mutating future observations or injecting post-cutoff values does not affect the feature pipeline or target alignment.

---

## 3. Models and Baselines

1. **Persistence Baseline:** Forecasts $\hat{Y}^{(h)} = \text{current\_delay}$.
2. **Historical Baseline:** Hierarchical lookup fitted strictly on training data: (1) `(train, station, h)` mean, (2) `(station, hour, h)` mean, (3) `h`-horizon global mean.
3. **Random Forest Regressor:** Scikit-learn `RandomForestRegressor` (50 trees, max_depth 12, min_samples_leaf 5) trained on 40,000 sampled instances per horizon.
4. **LightGBM Regressor (Primary Classical Model):** LightGBM 4.7.0 (120 trees, learning_rate 0.05, num_leaves 31, subsample 0.8, colsample 0.8) trained per horizon on up to 250,000 training instances.
5. **Graph Neural Network (PyTorch Geometric 2.8.1):** 2-layer `GCNConv` on the Corridor B station topology (108 nodes, 214 directed edges), concatenating the 16-dimensional learned station node embedding with the train dynamic features into an MLP prediction head.

---

## 4. Empirical Evaluation Results

### 4.1 Untouched Test Set (Nov–Dec 2025, 10 Fog Days)
The held-out test set represents late-season conditions including synthetic fog in December. This is the ultimate ground truth for model generalization:

| Forecast Horizon | Sample Count ($N$) | Persistence MAE (min) | Historical MAE (min) | Random Forest MAE (min) | LightGBM MAE (min) | LightGBM RMSE (min) | LightGBM $R^2$ | LightGBM Improvement vs Persistence |
|---|---|---|---|---|---|---|---|---|
| **1 Hour** | 219,383 | 3.71 | 15.75 | 3.65 | **3.57** | 7.30 | 0.9510 | **+3.78%** |
| **2 Hours** | 193,458 | 6.97 | 16.40 | 6.70 | **6.46** | 11.45 | 0.8938 | **+7.34%** |
| **3 Hours** | 168,799 | 10.10 | 17.29 | 9.47 | **8.97** | 15.34 | 0.8290 | **+11.16%** |
| **4 Hours** | 145,296 | 13.13 | 18.25 | 12.02 | **11.22** | 18.84 | 0.7712 | **+14.56%** |
| **5 Hours** | 122,233 | 16.16 | 19.28 | 14.15 | **13.01** | 21.65 | 0.7248 | **+19.45%** |
| **6 Hours** | 98,998 | 19.43 | 20.48 | 16.29 | **14.86** | 23.53 | 0.6919 | **+23.55%** |
| **Macro Average**| — | **11.58** | **17.91** | **10.23** | **9.68** | **16.35** | **0.8103** | **+16.42%** |

### 4.2 Validation Set (Sep–Oct 2025, Clear Season, Zero Fog)
During the validation period, zero fog days occurred. Under clear simulated conditions without external shocks, the simulator carries delays almost one-for-one (Phase 4 Finding 2: 0.996 correlation with near-zero slack absorption):

| Forecast Horizon | Sample Count ($N$) | Persistence MAE (min) | Historical MAE (min) | LightGBM MAE (min) | LightGBM Improvement vs Persistence |
|---|---|---|---|---|---|
| **1 Hour** | 219,409 | **2.76** | 10.97 | 2.87 | -3.98% |
| **2 Hours** | 192,837 | **4.89** | 11.39 | 5.23 | -6.82% |
| **3 Hours** | 167,495 | **6.79** | 11.96 | 7.24 | -6.52% |
| **4 Hours** | 143,622 | **8.46** | 12.52 | 8.96 | -5.89% |
| **5 Hours** | 120,404 | **10.02** | 13.09 | 10.29 | -2.69% |
| **6 Hours** | 97,402 | **11.55** | 13.71 | 11.58 | -0.19% |
| **Macro Average**| — | **7.41** | **12.28** | **7.69** | **-3.77%** |

### 4.3 Fog Season Robustness (Fold 1 Validation: Nov–Dec 2024, 12 Fog Days)
To evaluate model selection before seeing the test set, Fold 1's validation window (2024-11-01 to 2024-12-31) provided 12 fog days:
- **Persistence Macro MAE:** 11.27 min
- **LightGBM Macro MAE:** 9.15 min
- **Relative Improvement:** **+18.82% over Persistence** (matching the test set gain of +16.42%).

### 4.4 Fog vs. Non-Fog Segmentation (Test Set)
At horizon 6h on the untouched test set:
- **Fog Days ($N = 16,998$):**
  - Persistence MAE: **40.80 min**
  - LightGBM MAE: **22.54 min**
  - **Improvement in Fog: +44.76%**
- **Non-Fog Days ($N = 82,000$):**
  - Persistence MAE: **15.00 min**
  - LightGBM MAE: **13.26 min**
  - **Improvement in Non-Fog: +11.59%**

### 4.5 Train Priority Class Segmentation (Test Set Horizon 6h)
- **Shatabdi:** LightGBM MAE = 28.49 min vs Persistence 94.80 min (**+69.95% improvement** — captures recovery and model artifact dynamics).
- **Duronto:** LightGBM MAE = 12.95 min vs Persistence 21.62 min (**+40.11% improvement**).
- **Rajdhani:** LightGBM MAE = 12.06 min vs Persistence 18.06 min (**+33.22% improvement**).
- **Superfast:** LightGBM MAE = 15.67 min vs Persistence 19.68 min (**+20.37% improvement**).
- **Mail/Express:** LightGBM MAE = 12.72 min vs Persistence 15.05 min (**+15.44% improvement**).

---

## 5. Graph Neural Network Findings

A Spatio-Temporal Graph Convolutional Network (`GCNConv` on 108 station nodes, 214 directed edges) was evaluated on the identical dataset and splits:
- **Validation Macro MAE:** 8.27 min (vs LightGBM 7.69 min, Persistence 7.41 min).
- **Analysis:** On a 1-dimensional chain topology, non-local spatial message passing offers marginal topological advantage over tabular tree models equipped with explicit chain distance (`km_from_new_delhi`), node identity, junction flags, and leading-train delays. While the GNN successfully learns valid representations without leakage, LightGBM outperforms GNN by 0.58 minutes MAE on validation while training in half the time.

---

## 6. Inference Interface & Artifacts

The inference API is encapsulated in `src.models.predict.DelayPredictor`:

```python
from src.models.predict import DelayPredictor

# 1. Load trained pipeline and models
predictor = DelayPredictor.load("data/processed/models")

# 2. Predict downstream delay for any horizon (1h to 6h)
obs = {
    "train": "12301",
    "station": "NDLS",
    "current_delay": 18.5,
    "weather": "fog",
}
pred_1h = predictor.predict(obs, horizon=1)
# Returns DataFrame with columns: ['train', 'station', 'predicted_delay_min', 'horizon_hours', 'units', 'is_simulated']

# 3. Multi-horizon dispatch forecast
all_horizons = predictor.predict_all_horizons(obs)
```

Saved artifacts in `data/processed/models/`:
- `model_h1.joblib` .. `model_h6.joblib`
- `preprocessor.joblib`
- `metadata.json` (contains feature lists, training parameters, horizons, and dependency versions)

---

## 7. How to Reproduce

All commands run from repository root in PowerShell:

```powershell
# 1. Run the test suite (157 items including Phase 6 unit tests)
.\.venv\Scripts\pytest.exe -v

# 2. Run the end-to-end pipeline (dataset prep, training, evaluation, figures)
.\.venv\Scripts\python.exe -m src.models.run_pipeline --stride 4

# 3. Test standalone inference
.\.venv\Scripts\python.exe -c "from src.models.predict import DelayPredictor; p = DelayPredictor.load(); print(p.predict_all_horizons({'train': '12301', 'current_delay': 15.0, 'station': 'NDLS'}))"
```
