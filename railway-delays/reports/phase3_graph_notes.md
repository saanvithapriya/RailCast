# Phase 3: graph notes (Corridor B)

Delay-based numbers come from the **SIMULATED** medium scenario (`data/processed/sim/medium/`, headway 4, freight 0.04). Nothing here describes real Indian Railways delays. Low and high scenarios were not touched.

Code: `src/graph/` (`timetable.py`, `station_graph.py`, `train_graph.py`, `delay_attrs.py`, `snapshot.py`, `groundtruth.py`, `build.py`). Rebuild everything with `python -m src.graph.build`. Outputs go to `data/processed/graph/` (untracked); the ground-truth tables go to the scenario folder.

## 1. Time split and leakage rules

**Confirmed split (see section 7 for the change from the first proposal): train = all of 2024 + Jan-Aug 2025, validation = Sep-Oct 2025, test = Nov-Dec 2025, strictly by time.** The windows and data sources are defined only in `configs/splits.yaml` (read through `src/splits.py`); no date is hard-coded in the graph code.

- Every delay-based attribute is computed by a function that takes a date range (`node_delay_attributes`, `edge_delay_attributes` in `delay_attrs.py`; `snapshot` for a single instant). Two versions are saved:
  - `*_SIMULATED_train_2024-01-01_2025-08-31`: the only version that may be used for development, fitting or choosing anything (it replaces the first Jan-Aug-2025-only version, which was deleted).
  - `*_SIMULATED_ANALYSIS_ONLY_all_2025`: for analysis only, never for training.
- **Filtering is by actual event time, not by run date.** A row counts only if its actual arrival lies in `[start 00:00, end+1 day 00:00)`. A run that enters on 31 August and arrives on 1-3 September contributes nothing to the training window. Queue waits are filtered by the time the wait ended. The one row used from before the window is the upstream departure of a passage that ends inside it (past information).
- Tests build small hand-made data and show that a run spilling over midnight belongs to the window of its arrival time, that adding data after the window leaves the window's numbers unchanged, and that queue waits follow the window.
- **Timetable-only graphs** (`station_graph_timetable.pkl`, `train_graph_timetable.pkl`, the timetable columns of the parquet tables) are built from the Phase 2 tables and the `return_train` field of `trains.json` only. A test builds them while any access to a simulation output path raises, and another checks that the timetable modules do not import the delay modules. They do reuse the network and schedule logic of `src/simulation/network.py` (chain order, PRYG merge, distance-split run times), so they depend on simulator *configuration* (`config.yaml`) but not on any simulated delay.
- **Ground-truth cascades** are kept strictly separate: `groundtruth_cascades.parquet` and `groundtruth_cascade_runs.parquet` in `data/processed/sim/medium/`, marked `validation_only` in every row, in the parquet metadata and in a README next to them. `load_groundtruth()` raises unless called with `validation_only=True`. No notebook or graph in this phase reads them.
- Seasonal coverage of the synthetic fog under the final split: see section 7 (the first proposal lacked the December peak in training; adding 2024 fixed that).

## 2. Station graph

- **Nodes: 108** (110 original stations). **Prayagraj is one node `PRYG`**: ALD (Prayagraj Jn), NYN (Naini Jn) and COI (Prayagraj Chheoki) are merged because 17 trains reverse or loop between them and a straight chain cannot hold them. The node sits at ALD's coordinates, is a junction, and `station_node_map.parquet` maps all 110 original codes to their node (`original_codes = "ALD|NYN|COI"`). Its display name is "Prayagraj area (Prayagraj Jn + Naini Jn + Prayagraj Chheoki)". Per-day counts for PRYG count a train once per node visit.
- **Edges: 214 directed** = 107 sections x 2 directions between chain-consecutive nodes. A graph test shows every edge joins nodes one step apart on the chain and that the graph is strongly connected.
- Node attributes (timetable only): `display_name`, `lat`, `lon`, `is_junction` (guess from Phase 2: name has JN, an anchor, or Ghaziabad), `km_from_new_delhi` (sum of straight-line distances), `trains_per_day` (distinct trains with a visit; every train is assumed to run daily), `halts_per_day` (real halts plus true origin/terminus), `trains_down/up_per_day` (by direction of the next hop; a train's last visit has no direction, so arrivals at a terminus are not counted in up/down), `share_<type>` (share of trains by type).
- Edge attributes (timetable only): `trains_per_day` (per direction), `mean/median_sched_run_min` (the simulator's scheduled section time: the timetable time between halts split over sections by distance, **not** the raw per-stop times), `trains_hour_00..23` (scheduled section entries per hour of day), `n_fast/n_slow` and `share_fast/share_slow` (fast = priority 1-3: Rajdhani, Duronto, Shatabdi, Superfast), `km`.
- Delay attributes (**SIMULATED**, separate tables and graphs): per node `mean/median/p90_arrival_delay`, `share_late_over_10`, `n_arrivals`; per edge `mean/median/p90_arrival_delay` at the downstream node, `mean_delay_added` (downstream arrival delay minus upstream departure delay), `mean_queue_wait_per_passage` and `share_passages_waited` (waits above 0.5 minute are logged), `n_passages`. Edges exclude the 39 off-corridor legs and rare multi-section hops.
- Values (medium, Jan-Aug): 81-199 trains/day per node (New Delhi 81, Kanpur Central 199), 40-93 trains/day per edge direction; edge mean arrival delay 10.7 minutes on average (0.3 to 23.1), mean delay added -0.02 minutes (-0.19 to +0.82), mean queue wait 0.25 minutes (0.02 to 0.82), 9% of passages waited (0-18%). Passages per edge in the window match the timetable (0.85-1.00 of trains per day x days).

## 3. Train-dependency graph (timetable only)

225 trains, 1,964 edges:

| type | edges | meaning | attributes |
|---|---|---|---|
| rake | 218 (214 `return_train`, 4 `odd_even`; 7 trains have no partner) | directed feeder to next service of the rake | `rule`, `mean_gap_min` = scheduled turnaround |
| section_sharing | 734 | scheduled to enter the same section, same direction, within 10 minutes (circular over the day) | `count` = number of sections (11,721 in total), `mean_gap_min` 4.7 |
| overtake_station | 425 | X passes a node while Y has a scheduled real halt there | `count` = nodes (732 total), `mean_gap_min` = minutes Y still waits (8.8) |
| overtake_section | 587 | X enters a section after Y but is scheduled to arrive first | `count` = sections (598 total), `mean_gap_min` 2.9 |

Degrees (distinct neighbours, all types): mean 8.1, median 7, maximum 40; every train has at least one neighbour. Most connected: 13040 Janata Express (40 neighbours, 34 of them by in-section crossings), 22409 Garib Rath (23), 19570 Varanasi-Okha (22), 12329 West Bengal Sampark Kranti (22), 12379 Sealdah-Amritsar (21). Rake edges: 218 trains have a feeder; counting both directions, 212 trains have 2 rake edges (feeder and follower, the usual return pair), 3 have 3, 3 have 1, and 7 have none.

## 4. Snapshot lookup (time-expanded view)

`snapshot(scenario_dir, date, hour, tt, minute=0)` returns one row per (train, node): `at_station` (arrived and not yet left) or `in_section` (left the previous node, not yet arrived, attached to the downstream node), with the **current delay as of that instant** (arrival delay if at a station, departure delay at the previous node if in a section), the node, the previous node, and the timetable times at that node. Only events up to the instant are used. Trains in an off-corridor leg are not attached to any node. Surprise: a train passing a station without halting is only "at the station" for the instant it passes (zero dwell), so almost every train in a snapshot is `in_section`; only halting trains show as `at_station`. Verified on known hand-made cases in `tests/test_graph.py`.

## 5. Ground-truth cascades (SIMULATED, validation only)

`groundtruth_cascades.parquet`: 62,976 cascades, one per injected event with at least one delayed run (entry delay 23,296; section incident 18,562; freight 11,683; halt extension 9,380; disruption 55). Columns: `cascade_id`, `origin_event_id/type/place/train/date/start_min`, `origin_injected_min`, `size_runs_delayed`, `n_trains_delayed`, `max_depth_hops`, `duration_min` (from event start to the last delayed arrival), `total_delay_minutes_added`, first/last delay time. `groundtruth_cascade_runs.parquet` holds the runs and hop of each cascade.

Distribution: median size 1 run; 21,235 cascades delay 2 or more runs, 5,082 delay 5 or more (median 7), 1,429 delay 10 or more, 1,886 reach 3 or more hops; the largest delays 52 runs, the deepest has 7 hops. Hop 0: 69,815 runs, hop 1: 35,643, hop 2: 13,975, hop 3: 4,112. 3,065 runs (2.4%) have a root event but no path through "caused by" links and have no hop value.

Definitions: a run belongs to a cascade if some component of its delay traces back to the event (`root_event_id`); a root is the root of the *largest* component of the causing train's delay when the wait was added, so a run waiting for a train delayed by two events is attributed to the larger one only. Hops follow "waited for / yielded to / inherited rake delay from train X" links.

## 6. Surprises and suspicious items

1. **Delay added across a section is almost zero everywhere** (mean -0.02 minutes, extremes -0.19 and +0.82). Delay is added at entry and in a few places and carried, while slack recovery nearly cancels additions. The edge attribute `mean_delay_added` carries little signal; `mean_arrival_delay` and queue wait do.
2. **Highest edge delays are on the up direction into New Delhi** (CNJ to ANVT 23.1, CSB to NDLS 22.8, SBB to CNJ 22.8 minutes mean, p90 about 71) and New Delhi has the highest node delay (22.4). Up trains arrive at the terminus having accumulated their entry and en-route delay. Not a data artifact; a directional accumulation effect.
3. **Congested junction sections**: HNG to TDL (up) has the highest mean queue wait (0.82 min per passage, 18% of passages wait) and the largest delay added after EKL to ETW (+0.82). Both trace to the model's congestion near Tundla and Etawah, not to a data quirk.
4. **KNS-PJY quirk: no extreme edge.** The 0.12 km pair shows 16-22 minutes between stops in the raw timetable, but section run times are the whole halt-to-halt time split by distance, so KNS to PJY has a scheduled run of 0.33 minutes and ordinary delay statistics (mean arrival delay 13.4 down and 6.8 up; delay added +0.09 and +0.04). The quirk is hidden, not reported as a bottleneck. It would reappear if anyone used raw per-stop times.
5. **Shatabdi artifact: not visible in the graphs.** The four Shatabdi trains are 1.8% of trains and appear in cascades in proportion (1.5% of runs in cascades of 5 or more); none ranks among the most connected trains. Anything computed per train type or per Shatabdi train would still be affected (12003 and 12033 are the most delayed Shatabdi trains, about 28 minutes in clear weather in the pilot).
6. **`overtake_section` edges (587) are likely partly an artifact of the interpolation**: scheduled times between halts are split by distance, so two trains with different halt patterns can appear to cross inside a section. Treat them as "scheduled crossings", not confirmed overtakes. Station overtakes (425) rest on real halt windows and are more reliable.
7. **Train 13040 (Janata Express)** has 40 neighbours, more than any other train, because it crosses nearly the whole corridor (101 sections, 24 halts, about 20 hours), so it shares slots with most other trains.
8. Dependency edges are built on the daily assumption: trains that really run on different days are linked here.
9. Share of trains by type per node is almost constant along the chain, because most trains traverse most of it; type shares carry little spatial signal.

## 7. Update: final split, the 2024 run, and two cautions

**Split (confirmed):** train = all of 2024 + Jan-Aug 2025, validation = Sep-Oct 2025, test = Nov-Dec 2025, strictly by time. It is defined once, in `configs/splits.yaml`; `src/splits.py` reads it and every date-range function, `build.py` and the file-name tags use it. Test `test_split_config_is_the_single_source_of_the_windows` checks the windows are contiguous and do not overlap, and that `build.py` contains no hard-coded dates.

**Data behind the split (all SIMULATED, medium scenario, headway 4, freight 0.04):**
- 2025: the existing run (seed 42), `data/processed/sim/medium/`.
- 2024: a new run for calendar 2024 with a **different seed (2024)** and the same configuration otherwise, `data/processed/sim/medium_2024/`: 82,350 train runs, 5.67 M observation rows, 7.4 minutes to simulate, 223 MB on disk. Ledger check (same as the scenario runs): worst engine mismatch 1.7e-9 minutes; 99.998% of 2.5 M late observations have labelled components that match the observed delay within 0.05 minutes (largest difference 0.109 minutes, because components under 0.01 minute are not stored). Within 10 minutes: 71.8% (2025 medium: 73.5%); mean delay 12.2 minutes; share of added delay caused by other trains 62%.
- **2024 and 2025 share the same timetable** (same 225 trains, same scheduled times, every train daily). Only the random seed differs, so weather and incidents differ but the schedule, headway, freight rate and every model assumption are identical. This is not two real years; it is the same simulated world twice, so it adds weather and incident variety, not timetable change. 2024 is a leap year (366 days).
- Each source may supply only rows whose actual time lies in its own calendar range, so the two runs never overlap: rows of the 2024 run that spill past 2024-12-31 are dropped. The first days of each source (January 1 of 2024 and 2025) lack the runs that were already in progress at midnight, because the warm-up days are not recorded; this edge effect is small and exists in every run.

**Fog (and rain) coverage under the final split** (synthetic weather: fog only in Nov-Feb):

| window | fog days | by month | rain days |
|---|---|---|---|
| train (609 days) | 48 | 2024: Jan 9, Feb 13, Nov 2, Dec 10; 2025: Jan 10, Feb 4 | 76 (mostly Jun-Sep) |
| validation, Sep-Oct 2025 (61 days) | **0** | none | 9 (Sep 7, Oct 2) |
| test, Nov-Dec 2025 (61 days) | 10 | Dec 10 (Nov 0) | 3 |

So training now contains a full fog season including a December (10 fog days), the validation window has **no fog at all**, and the test window has fog in December only. Validation scores will look better than test scores for reasons that have nothing to do with the model; do not use validation delay levels to predict test delay levels.

**Cautions that carry forward**
- **"Mean delay added across a section" is near zero and must not be used as a feature.** Training window: mean -0.009 minutes, range -0.18 to +0.75 across the 214 edges. Delay is added in a few places and carried, while slack recovery cancels most additions, so the attribute carries almost no signal. Use mean arrival delay, queue wait and the carried delay itself instead.
- **In-section crossing edges (`overtake_section`, 587 edges) are partly an artifact.** The scheduled time between halts is split over sections by distance, so two trains with different halt patterns can appear to cross inside a section even if the real timetable has them follow each other. Treat them as scheduled crossings, not as confirmed overtakes; station overtakes (`overtake_station`) rest on real halt windows and are more reliable.

**Rebuild and tests:** `python -m src.graph.build` rebuilt the training-window delay tables and graphs on the new window (9.1 M passages over 214 edges; mean edge arrival delay 11.7 minutes against 11.1 for 2025 alone) and the all-2025 analysis-only version; the first Jan-Aug-only files were removed. The leakage tests, including the spill-over-midnight case, pass; new tests cover two sources, the year boundary (a 2024 run arriving in 2025 is not double counted) and the validation boundary (a run entering on 31 August and arriving on 1 September stays out of training). The ground-truth cascade tables were not rebuilt; they still describe the 2025 medium run only.

## 8. Update: rolling-origin validation (two folds)

Defined in `configs/splits.yaml` (`folds:`), read through `src/splits.py` (`folds()`, `fold(name)`, `fold_names()`, `fold_tag()`, `eda_windows()`):

| fold | train | validate | validation weather |
|---|---|---|---|
| `fold1_fog` | 2024-01-01 .. 2024-10-31 | 2024-11-01 .. 2024-12-31 | **12 fog days** (Nov 2, Dec 10) |
| `fold2_clear` | 2024-01-01 .. 2025-08-31 | 2025-09-01 .. 2025-10-31 | **no fog** (9 rain days) |

The test window (2025-11-01 .. 2025-12-31, 10 fog days in December) and its rules are unchanged; no fold touches it.

**Rules for model selection (later phases):**
- Any model or setting is chosen on the **mean of the two folds' validation scores**, never on one fold.
- **Fog performance is reported separately** (fold 1 is the only validation data with fog; fold 2 has none), so a model that is good on average but poor in fog is visible. Because fold 1 contains only 12 fog days, treat fog scores as indicative, with wide uncertainty.
- Fold 1 validation (Nov-Dec 2024) is not available to anything that makes modelling decisions when the training side is fold 1's training range; Phase 4 follows the same rule (decisions from 2024-01-01 .. 2024-10-31 only, the rest of the training window only for descriptive plots).

**No leakage between a fold's validation data and its training data:** `fold_train_delay_tables(fold_name, node_of, part="train")` in `src/graph/delay_attrs.py` builds node and edge delay tables only from that fold's training range, by actual event time. Tests (`tests/test_folds.py`) show that (a) a run entering on 2024-10-31 at 23:50 and arriving on 2024-11-01 belongs to fold 1's validation range, not its training range, (b) rewriting every observation in November 2024 leaves fold 1's training tables unchanged, (c) with real data the fold 1 training tables are identical when the source is clipped at 2024-10-31, so nothing after that date contributes, and (d) the fold ranges are contiguous, in time order, and stay inside the training window, with fold 2's training range equal to the full training window.

The fold 1 training tables are saved by `python -m src.graph.build` (it took about 50 seconds): `station_nodes_delay_SIMULATED_fold1_fog_train_2024-01-01_2024-10-31.parquet` and `station_edges_delay_...`. Fold 2's training tables are the main training-window files already saved (`..._SIMULATED_train_2024-01-01_2025-08-31`).
