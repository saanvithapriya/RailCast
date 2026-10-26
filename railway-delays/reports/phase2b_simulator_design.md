# Phase 2b: delay simulator design (for approval, no code written yet)

**All output of this simulator is synthetic.** Every table carries `is_simulated = True`, and every notebook, figure and report that uses it must say "simulated". Nothing here has been calibrated against real Indian Railways delays; the calibration targets in section 8 are plausible assumptions, not verified facts.

## 0. Items I need you to approve or change

1. **Rake pairing.** You asked for consecutive odd/even numbers. In the timetable only 109 odd/even pairs exist among the corridor trains, but `trains.json` has a `return_train` field and 214 of the 225 trains have their return partner in the corridor set. **Proposal: use `return_train`, fall back to odd/even.** Both are assumptions about rake use (the real rake diagrams are not public in this data).
2. **Two extra columns in the Phase 2 `trains` table.** Rake reuse needs each train's clock time at its true origin departure and true terminus arrival, which Phase 2 did not keep. I would add `origin_departure_min` and `terminus_arrival_min` (minutes from the start of the journey) by re-running `src.data.clean`. Existing columns and counts do not change.
3. **Scheduled times at pass-through rows.** The raw pass times are whole minutes and unreliable. Per your rule, the simulator uses only the scheduled time between real halts and splits it over the sections in proportion to distance. So for pass-through rows, `scheduled_*` in the output is the interpolated time, not the raw timetable value. An on-time train then shows exactly zero delay. Real halts keep their raw scheduled times.
4. **A fuller ground-truth table.** Your observation schema has one `cause_code` and one `cause_train` per row, which cannot hold a row that is late for two reasons. I would add a second table, `delay_attribution`, with one row per (observation, cause) holding the minutes and the causing train/event. The `cause_code` in `observations` is the dominant new cause at that row.
5. **Observation rows for every core station** (pass-through included), about 15,550 per simulated day, so roughly 5.7 million rows for 12 months. Parquet should be a few hundred MB. Say so if you would rather log halts and junctions only.
6. **Dependencies.** I would install `pytest` into the venv and add it to `requirements.txt` for the unit tests. PyYAML is already installed.
7. **Simulated year.** Calendar 2025 (365 days), purely to give months and weekdays meaning.

## 1. Inputs and facts the design relies on

- Timetable: `data/processed/stops.parquet`, `trains.parquet`, `stations.parquet` from Phase 2.
- After dropping the 4 slip entries: **225 trains**, 15,550 core stop rows, 1,325 real halts (8.5%).
- 160 trains enter the corridor mid-route (not 163; 3 slips were among them); 65 start inside it. 159 leave before their terminus.
- 110 core stations on one chain (109 sections). 11 junction guesses: ALD, ALJN, CNB, ETW, GZB, HRS, KRJ, MGS, NDLS, SKB, TDL.
- Busiest sections carry about 183 train passages per day (both directions), so about 90 per direction. At a 5-minute headway that is about 32% of capacity on average with higher peaks, which is enough to produce queues.
- Journeys span up to 4 calendar days (3,315 minutes), so trains from one service day are still running on the next.

## 2. Network model

- **Sections:** consecutive core stations on the chain, one lane per direction (up and down are independent resources). 109 sections, each with a length in km (haversine on station coordinates, sum along the chain).
- **Run times:** for each train, take the scheduled time between consecutive real halts (departure at one to arrival at the next) and split it over the sections in between in proportion to section length. The timetable's own pass times and per-segment speeds are never used.
- **Prayagraj special case.** ALD, NYN and COI form a triangle that 17 trains traverse out of chain order (reversing or looping). I treat them as **one node, `PRYG`**, placed where ALD is on the chain. A train's visit to the node starts at its first arrival among those three stations and ends at its last departure. The scheduled time inside the node is its dwell time (it contains any reversal). Its three original rows in the output get the node's arrival delay on arrival rows and the node's departure delay on the final departure row, and the intermediate rows take the arrival delay. Consequence: no conflicts are modelled inside the triangle. This is a deliberate simplification.
- **Off-corridor legs (39 `leaves_main_line` trains).** The hop between the two core rows more than 60 km apart is a fixed-time leg: it takes its scheduled time, carries the delay unchanged (no recovery, no weather effect), uses no section, and causes or suffers no conflicts. The sections it jumps over are not occupied by that train.
- **Entering mid-route:** an `entered_mid_route` train appears at its first core station with an incoming delay (section 5.3).

## 3. Movement rules

The simulator is event-driven. An event is "train *k* is ready to leave station *i*", processed in time order from a heap. One service day is processed at a time, but the heap and section state carry over, so trains still running at midnight continue into the next day. Random numbers for each day come from `SeedSequence([seed, day_index])`, so any day is reproducible independently.

### 3.1 Departure time

`departure = max(scheduled departure, arrival + min_halt, next section free, rake ready)`

- **Scheduled departure:** a train never leaves early at a real halt. Pass-through nodes have no hold, so a train passes as soon as it arrives.
- **min_halt:** `max(1, round(min_halt_factor x scheduled halt))`, never more than the scheduled halt. Default factor 0.5. A late train can shorten a halt down to this value (recovery).
- **Next section free:** see 3.2 and 3.3.
- **Rake ready:** only at a train's true origin; see 3.6.

### 3.2 Section occupancy and headway

- A train may enter a section no earlier than `last entry in the same direction + headway`. Default `headway_min = 5`; per-section overrides in the config.
- Trains cannot pass inside a section (one lane). A follower may not arrive at the end of a section earlier than `leader arrival + exit_gap_min` (default 2). If it would, it is slowed and the extra minutes are logged as a section wait caused by the leader.

### 3.3 Priority and overtaking

Priority (lower number wins): Rajdhani 1, Duronto 2, Shatabdi 2, Superfast 3, Mail/Express 4, Passenger 5, other 4.

- **Overtaking is allowed only at junction stations, or at any station where the slower train has a real halt.** Everywhere else trains keep their order and the follower waits behind the leader (section 3.2).
- **Yielding:** when a train L is ready to leave a station where it can be overtaken, it checks trains in the previous section heading the same way (one section look-back, `yield_lookback_sections = 1`). If a higher-priority train H will be ready to enter the next section within `yield_window_min` (default 10) of L's ready time, L waits until `H's entry + headway`, but at most `max_yield_hold_min` (default 30). The held minutes are logged as yielding, caused by H. At a junction L is held even if it was not scheduled to stop there.
- If L cannot be overtaken (no halt, not a junction) it does not yield, and H later waits behind it.
- Because the simulator only knows arrival times for trains one section behind, a higher-priority train further back cannot trigger a yield. This is a limitation.

### 3.4 Recovery and slack

- Each train type has a slack fraction `s` (default Rajdhani 0.05, Duronto 0.05, Shatabdi 0.06, Superfast 0.06, Mail/Express 0.07, Passenger 0.08, other 0.06).
- A section's scheduled time `T` includes slack. A late train runs it in `T - min(current delay, s x T)`, so it recovers up to `s x T` per section and never runs faster than `T(1 - s)`.
- A train also recovers by shortening halts to `min_halt`.
- Off-corridor legs never recover.

### 3.5 Freight blockers (no freight data exists)

- Random freight occupations per section: Poisson rate `freight_rate_per_section_hour` (default 0.25), multiplied by `freight_junction_multiplier` (default 2.5) for sections touching NDLS, GZB, TDL, CNB, PRYG and MGS. Each blocks entry to the section (both directions) for a log-normal duration (median 6 min, sigma 0.6).
- A passenger train that wants to enter during a block waits until it ends, except Rajdhani, Duronto and Shatabdi, for which freight yields (no delay). Wait minutes are logged as freight, with the freight event id.

### 3.6 Rake reuse

- Linked pair A (arriving) and B (departing): the partner of each train comes from `return_train`, with odd/even as fallback (item 0.1).
- B's rake becomes ready at `A's actual arrival at its true terminus + min_turnaround_min` (default 45). If A's terminus is off-corridor, A's arrival delay is estimated as `A's delay when leaving the corridor x (1 - offcorridor_recovery)` plus noise (default recovery 0.3, noise sd 3 min).
- The link only applies if B's scheduled origin departure is 45 min to 36 h after A's scheduled terminus arrival (`max_turnaround_hours = 36`). Otherwise B has no rake delay.
- B's resulting delay at the origin is then carried to its first core station (reduced by the same off-corridor recovery). The delay is logged as rake reuse, caused by A.

## 4. Delay sources (primary delays)

### 4.1 Section incidents

Each train traversal of a section can trigger an incident with probability
`p = base_rate x station_factor x hour_factor x season_factor x weather_factor`, default `base_rate = 0.003`.

- Station factor: junction sections 2.0, sections touching NDLS 2.5, CNB 2.0, TDL 1.8, PRYG 2.0, otherwise 1.0.
- Hour factor: 1.3 for 06-09 and 17-21, 1.0 otherwise.
- Season factor: Dec-Jan 1.5, Feb and Nov 1.2, Jun-Sep (monsoon) 1.2, otherwise 1.0.
- Weather factor: fog 2.0, rain 1.3, clear 1.0.
- Delay minutes: 95% log-normal (median 5, sigma 0.9), 5% Pareto tail (minimum 15, alpha 1.7), capped at 240. Added as extra running time in that section.

### 4.2 Halt extensions (station incidents)

At each real halt, probability 0.02 (same hour and season factors) of an extension of 1-10 minutes (log-normal, median 3).

### 4.3 Weather

Swappable: `src/simulation/weather.py` defines an interface `daily_state(date) -> {"clear","fog","rain"}`; the default is the synthetic generator, and a real Open-Meteo reader can replace it without touching the simulator. The output `weather` table stores `date, state, source`.

- **Synthetic daily state:** a Markov chain with monthly base probabilities. Fog: Dec 0.35, Jan 0.40, Feb 0.12, Nov 0.08, otherwise 0. Rain: Jul-Sep 0.25, Jun and Oct 0.10, otherwise 0.02. Persistence 0.6 (a fog or rain day makes the next one more likely).
- **Fog is regional:** intensity 1.0 from NDLS to CNB, 0.5 from CNB to PRYG, 0.2 beyond.
- **Effect:** section run time x `(1 + fog_slowdown x intensity)`, default `fog_slowdown = 0.12`; rain x 1.03. This is logged as weather delay. Slack recovery still applies on top.

### 4.4 Entry delay (160 mid-route trains)

Unknown incoming delay is drawn at the first core station: with probability 0.45 it is zero, otherwise log-normal (median 12 min, sigma 0.9), multiplied by
`(1 + 0.5 x journey fraction already run)`, by a type factor (Rajdhani 0.6, Duronto 0.7, Shatabdi 0.7, Superfast 1.0, Mail/Express 1.1, Passenger 1.2), by a season factor (same as 4.1) and by 2.5 on fog days. If the train has a rake link the rake delay (3.6) is added. Logged as entry delay.

### 4.5 Disruptions (block or signal failure)

Per section, `disruption_rate_per_day = 0.0015` (about 60 a year on the corridor), duration log-normal clipped to 30-180 min (median 70). 60% signal failure (one direction closed), 40% block (both). No train may enter the section during the interval; trains that arrive wait and are released one headway apart, which creates the queue. Wait minutes are logged as disruption with the event id. Trains already inside the section are not stopped.

## 5. Ground-truth labels

Every minute of delay is tracked in a per-train ledger of `(cause, causing train or event, root event, minutes)`. Cause codes:

| code | cause | causing id |
|---|---|---|
| 0 | none | |
| 1 | injected section incident | event id |
| 2 | disruption (block or signal failure) | event id |
| 3 | weather slowdown | |
| 4 | section wait: headway or following another train | the leading train |
| 5 | yielding to a higher-priority train | the priority train |
| 6 | freight blocker | freight event id |
| 7 | rake reuse | the arriving train A |
| 8 | halt extension | event id |
| 9 | entry delay (unknown incoming) | |
| 10 | inherited: delay carried unchanged from upstream | |
| -1 | recovery (negative minutes: slack used, halt shortened) | |

- Delay added in a section or at a station gets a local cause (1-9). Delay already carried is code 10 and keeps the `root_event_id` it came from, so Phase 5 can follow a cascade back to its first cause even through other trains (for example "waited behind train X, which was late because of event E").
- Recovery subtracts proportionally from the carried components, so the ledger always sums to the observed delay.
- The `cause_code` in `observations` is the code of the largest delay added on the section ending at that row, or at that station (0 if none). The full split is in `delay_attribution`.

## 6. Output (`data/processed/sim/`, parquet, partitioned by month)

- **observations:** `date` (service date of the train's first core stop), `train`, `station`, `seq`, `scheduled_arrival`, `scheduled_departure`, `actual_arrival`, `actual_departure` (all as minutes from midnight of `date`, plus a timestamp), `arrival_delay`, `departure_delay`, `primary_delay_injected` (minutes injected at that section or station), `cause_code`, `cause_train`, `root_event_id`, `is_simulated`.
- **delay_attribution:** `obs_id`, `cause_code`, `minutes`, `cause_train`, `event_id`, `root_event_id`.
- **events:** `event_id`, `date`, `type` (section_incident, halt_extension, disruption_block, disruption_signal, freight, entry_delay), `section` or `station`, `start`, `duration`, `train` (where it applies).
- **weather:** `date`, `state`, `source`.
- **config_used.yaml:** every parameter and the seed (default `seed = 42`), copied next to the output. The same config and seed reproduce the output exactly.

## 7. Code layout and tests

```
src/simulation/
  config.yaml        all parameters above
  config.py          loading and validation
  network.py         sections, node merging, run-time allocation
  weather.py         swappable weather source
  delays.py          incident, entry and disruption generators
  engine.py          event loop, headway, priority, overtaking, rake, slack
  ledger.py          cause accounting
  run.py             12-month run and output writer
tests/test_simulation.py
```

Unit tests (small hand-built networks, fixed seed): headway enforced; follower waits behind a leader and cannot pass inside a section; priority yield at a junction and at a real halt, but not at a non-halt station; slack recovery limited to `s x T`; halts shortening to `min_halt`; rake reuse passes delay on only when the turnaround buffer is exceeded; ledger sums to the observed delay; same seed gives identical output.

## 8. Calibration targets and checks (assumptions, not facts)

Roughly aiming for these, to be tuned and then reported honestly in `notebooks/03_simulator_check.ipynb`:

- Most arrivals within 10 minutes of schedule (aim 70-85% overall).
- A heavy tail (p99 of at least 90 minutes).
- Fog months (Dec-Jan) clearly worse than the monsoon and summer; long-distance, low-priority trains worse than Rajdhani/Shatabdi.
- Delay growing along a route unless slack absorbs it.
- 15-35% of delay minutes caused by other trains (codes 4, 5, 7), so cascades are clearly visible.

The notebook shows: delay by station, month, hour and train type; delay along three trains' routes; one traced cascade; the share of delay by cause; and a sensitivity run varying headway (3, 5, 8 min) and slack (3%, 6%, 10%) on a shortened period (about 2 months each) to keep it fast.

## 9. Runtime and size (estimate, to be measured)

Pure Python event loop with about 5.7 million train-station steps for 12 months. I expect 15-40 minutes; if it is slower I will run months in parallel with a 4-day warm-up overlap. I will report the real runtime and file sizes.

## 10. Limitations (also going into the README "Synthetic data" section)

- All numbers are invented; nothing is calibrated on real delays. Findings from this data say whether methods work on a world we designed, not about Indian Railways.
- Every train runs daily (inherited from Phase 2), which overstates traffic.
- Pass-through scheduled times are interpolated.
- Prayagraj is one node, so no conflicts inside the triangle.
- No platform or loop capacity, no signalling blocks inside sections, no crew or loco constraints.
- Freight is random noise, not real paths.
- Yield look-ahead is one section.
- Off-corridor behaviour (entry delay, off-corridor leg of rake partners) is a simple random model.
- The timetable itself is old (pre-2018) and has no freight.

## 11. Decisions after review (approved) and additions

- **Rake pairing:** `return_train` first, odd/even fallback. The simulator logs how many trains each rule covers (measured on the 225 trains: 214 by `return_train`, 4 by odd/even, 7 with no partner, 0 outside the turnaround window).
- **Phase 2 re-run:** `trains` gained `origin_departure_min` and `terminus_arrival_min`. The validator (`python -m src.data.validate --baseline <dir>`) confirmed that existing columns, values and counts are unchanged.
- **Interpolated pass-through times:** the raw timetable time stays in separate columns (`raw_scheduled_arrival`, `raw_scheduled_departure`).
- **`delay_attribution`** is written next to `observations`. `cause_code` in `observations` stays the dominant cause of the delay added since the previous row; if nothing was added but the train is still late it is 10 (inherited), and 0 if on time.
- **Every core station is logged**, with float32 and categorical dtypes, partitioned by month. All output tables carry `is_simulated = True` and the parquet metadata says `SIMULATED`.
- **Simulated year 2025**; `pytest` added to `requirements.txt`.
- **Lookback depth** is the parameter `movement.yield_lookback_sections` (default 1).
- **Per-run random streams:** each train run draws from its own stream seeded by (seed, entry date, train). Results therefore do not depend on the interleaving of other trains, but they are only reproducible for the same config, seed and date range (rake links and queues carry state between runs).

### Known limitations added at review

- **Yield look-ahead defaults to one section.** A higher-priority train two or more sections behind is not seen, so a lower-priority train can leave just before it and later hold it up. Raising `yield_lookback_sections` reduces this (it uses an estimate, not an exact arrival time).
- **KNS-PJY is a data quirk, not a bottleneck.** In the Phase 2 data the pair Kanchausi-Para Jani Halt (KNS-PJY) is 0.12 km apart but shows 16-22 minutes between stops. The simulator allocates time by section length between real halts, and any analysis of simulated or real-looking delays must not report this segment as a bottleneck.

## 12. Changes made while building (all SIMULATED-data assumptions, none calibrated on real delays)

1. **Scheduled pairs are assumed feasible.** With 225 trains all running daily, 37% of consecutive entries into the same section are scheduled closer than the 5-minute headway (many at the same minute), so the model produced structural delay even with every random source off (41% of arrivals within 10 minutes). Pairs that the timetable itself schedules closer than the headway now keep their scheduled spacing (`movement.assume_scheduled_pairs_feasible`); all other pairs use the headway. With this, the no-randomness baseline has 98% of arrivals within 10 minutes and a mean delay of 0.65 minutes.
2. **Yield only when it helps.** A slower train yields to a priority train only if entering first would delay the priority train by more than `movement.yield_min_conflict_min` (default 1 minute), including the case where it would catch up inside the section. The first version held every slower train near a priority train even when no conflict existed.
3. **Lowered defaults after a first check** (the first values gave about 47% of arrivals within 10 minutes): freight rate 0.25 to 0.04 per section-hour (median 6 to 5 minutes), incident base rate 0.003 to 0.002, entry delay zero probability 0.45 to 0.6 (median 12 to 8 minutes, sigma 0.9 to 0.8), exit gap 2 to 1 minute. These were chosen by hand to reach the plausible targets in section 8; they are not estimates of real behaviour.
4. **Zero-length sections.** The scheduled time between halts can be shorter than the sum of section minutes would suggest for very short sections; a section can have scheduled time 0 (the earlier 0.2-minute floor made the ledger disagree with the observed delay by 0.2 minutes).
5. **Queue log.** The engine can log every wait to enter a section (`queue_log`), used by the pilot notebook to show queues.

## 13. Overtaking at any station, headway 4, and the second pilot (parameters NOT tuned to the targets)

**Changes (as requested):**
- `movement.overtake_mode: any_station` (default). A lower-priority train can be held aside at any station so a priority train passes, if fewer than `loop_capacity` (default 2) trains are already held there; it pays `overtake_penalty_min` (default 2 minutes) on top of the wait. The earlier rule (junction or real halt only, no penalty, no capacity limit) is kept as `overtake_mode: junction_or_halt`.
- Default headway is 4 minutes; 3 and 6 are the sensitivity values.
- No other parameter was changed to hit the 70-85% and 15-35% targets.

**Where the second pilot lands (January and June 2025, seed 42, SIMULATED):**

| | January | June |
|---|---|---|
| Arrivals within 10 minutes | 58% | 78% |
| Mean / p90 / p99 delay (min) | 21 / 63 / 169 | 8 / 25 / 103 |
| Clear-weather share within 10 min | 71% | 78% |
| Fog days: share within 10 min / mean delay | 30% / 41 min (10 fog days) | no fog |
| Delay minutes added by other trains (section wait + yield + rake) | 53% | 69% |
| Weather share of added minutes | 23% | 2% |
| Largest simultaneous queue; section-directions with 3+ waiting of 214 | 12; 202 | 15; 193 |

Both months together: 68% within 10 minutes (target 70-85%, an assumption). Winter is worse than summer, fog is clearly worse than clear weather, and the tail is heavy (p99 above 100 minutes in both months).

**Priority sanity check (clear weather, 14 days of June and the clear days of the pilot): it does not pass as specified.**
- Raw mean delay (pilot, clear): Rajdhani 6.5, Duronto 7.1, Passenger 6.4, Mail/Express 9.3, Superfast 10.3, **Shatabdi 15.3** minutes. Rajdhani and Duronto are now below Superfast (before the change, with the old rule, Rajdhani was the worst class at 15.7). But Shatabdi is above Superfast, Mail/Express is below Superfast, and Passenger is below Mail/Express.
- Why: (1) **Exposure.** Passenger trains run about 20 rows in the corridor and 88% start inside it, so they have little time to accumulate delay and no mid-route entry delay; 93% of Mail/Express trains enter mid-route with an unknown incoming delay. Raw mean delay mixes priority with route length and entry. (2) **Shatabdi is four trains.** Two of them, 12033 (Kanpur to New Delhi) and 12003 (Lucknow to New Delhi), average 28 minutes late in clear weather; they spend their delay at ETW, SKB, TDL, FZD and HNG waiting behind slower trains (18631, 51809, 12379, 12453, 12439). That is the behaviour the one-section look-ahead and the loop capacity allow: a priority train that catches a slower train between stations waits until the next station, where the slower train is held aside only if a loop is free and the priority train is already within the window.
- On a measure that removes exposure, **delay minutes added by other trains per 100 logged rows**, the ordering holds except for Shatabdi: Rajdhani 18.8, Duronto 19.2, Superfast 28.8, Mail/Express 37.4, Passenger 50.5, Shatabdi 49.9. So priority does reduce how much other trains delay a class, but not enough to make raw delay monotone, and Shatabdi remains an unexplained outlier at this sample size.
- Tests: the raw ordering is recorded as a strict expected failure (`tests/test_priority.py`); the exposure-adjusted ordering (excluding Shatabdi) is a passing test.

**Cascade-heaviness, honestly:** this is a cascade-heavy world. More than half of all delay minutes are added by waiting for or yielding to other trains (53% in January, 69% in June), queues of 8-15 trains form on most busy sections, and one 34-minute incident on GZB-MIU led to 45 train runs being delayed up to 8 hops away. The causes are the one-lane-per-direction, headway-spaced model with a timetable that all runs daily, so the corridor is denser than the real one. The 15-35% target is not met.

**Sensitivity (same seed; January / June, share of arrivals within 10 min and share of added minutes caused by other trains):**

| variant | within 10 min | caused by other trains |
|---|---|---|
| headway 3 | 61% / 81% | 48% / 64% |
| headway 4 (default) | 58% / 78% | 53% / 69% |
| headway 6 | 50% / 68% | 65% / 80% |
| old overtaking rule | 56% / 76% | 56% / 73% |
| slack ~10% | 67% / 85% | 49% / 64% |
| slack ~3% | 49% / 68% | 57% / 74% |

The overtaking change on its own improves on-time share only by about two points; its main effect is to stop the priority classes being the worst-hit. Runtime is now about 36-38 seconds per month (about 8 minutes for a year, because yield checks run at every station) and about 16-24 MB per month, 240 MB per year.


## 14. Lookback depth 2 (tried once) and the low / medium / high scenarios

### 14.1 `yield_lookback_sections = 2`: tried once, not adopted

The decision rule was fixed before the run: Shatabdi counts as fixed if, in clear weather, both its mean delay and its delay added by other trains per 100 rows are at or below Superfast's; another type counts as worse if its clear-weather mean delay rises by more than 1 minute or the overall share within 10 minutes falls by more than 1 percentage point. January and June 2025, seed 42, SIMULATED.

| | lookback 1 | lookback 2 |
|---|---|---|
| Share within 10 min (both months) | 67.7% | 66.4% |
| Mean delay, all weather (min) | 14.8 | 15.5 |
| p99 delay (min) | 149 | 153 |
| Shatabdi clear-weather mean (min) | 15.3 | 17.6 |
| Shatabdi added by others per 100 rows | 49.9 | 55.4 |
| Superfast clear-weather mean (min) | 10.3 | 10.8 |
| Passenger clear-weather mean (min) | 6.4 | 9.4 |
| Rajdhani / Duronto clear-weather mean (min) | 6.5 / 7.1 | 7.3 / 7.7 |

Result: **Shatabdi was not fixed** (12033 stayed at about 29 minutes late and 12003 rose from 28 to 36), and every other type was slightly worse, Passenger by 3.1 minutes and the overall on-time share by 1.3 points. Lookback 2 uses an estimate of a train's arrival rather than an exact time, so it holds slower trains for priority trains that then do not arrive when expected. **The default stays at 1 and no other value was tried.** The four Shatabdi trains (mainly 12033 Kanpur to New Delhi and 12003 Lucknow to New Delhi, delayed at ETW, SKB and TDL behind slower trains) are a **known model artifact**: the model cannot let a priority train get past a slower train between stations, and the loop capacity and one-section look-ahead do not always release it at the next station. Findings that rely on Shatabdi delays should not be trusted.

### 14.2 Scenarios (calendar 2025, lookback 1, seed 42, all SIMULATED)

Scenarios differ only in headway and freight rate, both set in advance and not tuned. Same seed and weather in all three. Written to `data/processed/sim/{low,medium,high}/`; each folder holds `observations`, `delay_attribution` (partitioned by month), `events`, `weather`, `queues`, `config_used.yaml` and `scenario_summary.json`.

| | low | medium (default) | high |
|---|---|---|---|
| Headway (min) / freight (per section-hour) | 3 / 0.02 | 4 / 0.04 | 6 / 0.06 |
| Train runs / observation rows | 82,125 / 5.65 M | 82,125 / 5.65 M | 82,125 / 5.65 M |
| Share of arrivals within 10 min | 78.1% | 73.5% | 63.4% |
| Mean delay (min) | 8.8 | 11.1 | 18.2 |
| p99 delay (min) | 111 | 126 | 171 |
| Share of delay minutes added by other trains | 59% | 65% | 75% |
| Share added by weather | 11% | 9% | 7% |
| Largest queue (trains waiting at once, one station and direction) | 17 | 20 | 19 |
| Section-directions with a queue of 3+ at some time (of 214) | 214 | 214 | 214 |
| Runtime (simulate + write) | 9.5 min | 8.5 min | 10.8 min |
| Size on disk | 191 MB | 212 MB | 257 MB |

Versus the plausible targets (assumptions, not met by design; no parameter was tuned): the low scenario is inside the 70-85% on-time range, the medium scenario is just inside (73.5%) and the high scenario is below it (63.4%). Delay caused by other trains is 59-75%, far above the 15-35% target in every scenario, so all three worlds are cascade-heavy (see the README for why).

### 14.3 Ledger check on the full runs

For every observation in each scenario that is at least 0.5 minutes late and has labelled components, the component minutes were compared with the observed arrival delay (2.07 M, 2.36 M and 2.91 M rows in low, medium, high). 99.998-99.9996% of them agree within 0.05 minutes (a few dozen rows per scenario differ by more), and the largest difference is 0.077, 0.094 and 0.073 minutes respectively (components below 0.01 minute are not stored). The engine-side check (sum of components vs delay at every arrival and departure) shows a worst mismatch of about 1e-9 minutes in each run.

## 15. Calendar 2024 for the medium scenario (added for the time split)

To give the training window a full year before 2025, the medium scenario was also simulated for calendar 2024 into `data/processed/sim/medium_2024/` with a different seed (2024) so weather and incidents differ; headway 4, freight 0.04, lookback 1 and everything else are unchanged. The weather generator's start point now moves back when a run starts before 2025 (existing 2025 runs are unchanged). Results (SIMULATED): 82,350 train runs, 5.67 M observation rows, 7.4 minutes to simulate, 223 MB; 71.8% of arrivals within 10 minutes, mean delay 12.2 minutes, p99 134 minutes, 62% of added delay caused by other trains, largest queue 24 trains. Ledger check as in section 14.3: worst engine mismatch 1.7e-9 minutes, 99.998% of 2.5 M late observations match their labelled components within 0.05 minutes (largest difference 0.109 minutes). The same timetable is used for 2024 and 2025.
