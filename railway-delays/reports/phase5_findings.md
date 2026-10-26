# Phase 5 findings: mining delay propagation and cascades (SIMULATED medium scenario)

**Everything below comes from the SIMULATED medium scenario. None of it describes real Indian Railways delays.** The mining uses observable data only (scheduled and actual times, delays, train types, the timetable graphs, the weather state); it never read `delay_attribution` or the ground-truth cascades. Figures are in `reports/figures/` (`mining_NN_*.png`, each titled "[SIMULATED]"); the analysis is `notebooks/06_mining.ipynb`; the code is in `src/analysis/` (`passages`, `decomposition`, `contagion`, `cascades`, `criticality`, `patterns`, `communities`, `mining_pipeline`).

## How the method was frozen and applied

- **Decision window 2024-01-01 .. 2024-10-31:** every threshold and design choice was set here and written to `configs/mining_frozen.yaml`: the slack fractions per train type and for halts (estimated from late, uncontested trains in clear weather: 0.05 Rajdhani and Duronto, 0.06 Superfast and Shatabdi, 0.07 Mail/Express, 0.08 Passenger, 0.5 of the scheduled dwell at halts), the gap window G = 6 minutes (where a late leader stops raising the chance that a follower adds 2 or more minutes), the rake turnaround limit of 180 minutes (beyond it the entry delay no longer follows the feeder's delay), and the cluster count k = 10 for communities. Thresholds fixed in advance: a leader is "late" at 5 minutes, a follower is linked at 2 minutes added, a primary event is 15 minutes, an external entry trigger is 15 minutes, a rake link needs 5 minutes.
- **Application window 2024-11-01 .. 2025-08-31:** the frozen method was run unchanged. All results below are for this window unless stated; the decision-window result is given where it shows stability.
- Validation fold 2 (Sep-Oct 2025) and the test window were not loaded.
- **Design changes made inside the decision window** (documented because they changed the method): the first section increment (arrival at the exit station minus departure from the entry station) missed the wait to enter a section, which happens at the station; the increment now runs from arrival at the entry station to arrival at the exit station, which roughly quintupled the visible effect of a late leader. Rake links first chained unrelated late runs into week-long cascades; the 180-minute turnaround limit fixed that.
- Caveat for everything: 2024 and 2025 are the same timetable simulated twice with different seeds; every train runs daily; one lane per direction; slack is a flat 5-8%; freight is random; headway is 4 minutes.

## Findings

### 1. Delay is almost all inherited: sections generate little, and slack absorbs very little
Per section passage the mean delay is **11.1 minutes inherited from upstream, 0.46 minutes added in the section, 0.25 minutes absorbed by slack**; slack absorbs only **2.2%** of the delay present (1.0-3.0% for most train types, 8% for Passenger trains, which have the most slack). The slack fractions that the method estimated from observable data agree with the values the simulator was configured with (5-8% running slack, half of the scheduled dwell at halts), which the method did not see. Added delay per passage is highest for Passenger (0.92) and Shatabdi (0.65) and lowest for Rajdhani (0.30) and Duronto (0.31).
*Figures:* `mining_01_decomposition`, `mining_03_slack_absorbed`.
*SIMULATED caveat:* numbers describe this simulated corridor.
*Likely artifact?* **Yes, largely.** The tiny absorption is a direct consequence of the flat 5-8% slack and of entry delays being injected once and then carried; real railways have more varied recovery.

### 2. About half of what sections add is explained by a late train ahead; added delay is concentrated in a few sections
**40% of the added delay minutes** are linked to a late leading train, and per section the explained share has a median of **0.49 (range 0.12-0.73)**. The sections adding the most delay are TDL-HNG (up, 1.5 minutes per passage, 17% of passages add 2 or more minutes), ETW-EKL (up), RURA-RMW (down) and HNG-FZD (up). There are **28.6 primary events per day** (a section adding at least 15 minutes with no late leader to explain it): 3,886 of 8,697 are Superfast trains and 3,354 Mail/Express (in line with their share of trains), and fog doubles the rate (**50 per day on fog days against 26 on clear days, 35 on rain days**; fog days are 8.6% of the window and carry 15% of the primary events).
*Figures:* `mining_01_decomposition`, `mining_02_contagion` (right panel).
*SIMULATED caveat:* the explanation rule is observable but approximate: a late leader that entered within 6 minutes and left within 6 minutes before the follower.
*Likely artifact?* **Largely yes.** One lane per direction, no passing between stations, a 4-minute headway and daily running (a corridor about twice as dense as a real one) make trains wait for each other; the hot sections sit next to junctions where the simulator doubles the incident rate.

### 3. Between trains the average transfer of delay is close to zero; it happens in rare, local events, and no section amplifies at one hop
The linear coefficient (follower added delay per minute of leader delay at entry, on timetable-adjacent pairs, with controls) has a median of **0.0035** (95th percentile 0.03, maximum 0.11); the interval excludes zero on the positive side in 119 of 214 sections and every section is labelled "absorb". The pass-on ratio (follower delay minutes caused per minute of leader delay, one hop at the same section) is **0.012 over all late leaders but 0.34 after primary events** (median over 103 sections with at least 30 events; range 0.05-0.58): a primary event passes on about a third of its minutes to the trains right behind it, 103 sections are labelled "absorb" and the other 111 are "uncertain" for lack of events; none is "amplify".
*Figures:* `mining_02_contagion`.
*SIMULATED caveat:* the linear coefficient is near zero partly because late leaders are usually followed by trains already late in the same platoon; the pass-on ratio is a one-hop, same-section measure and understates chains (finding 4).
*Likely artifact?* **Very likely.** In the simulator, trains the timetable puts closer than the headway are assumed not to block each other, sections are first-in-first-out, and recovery is flat, so one-hop amplification is structurally limited. A real network may show stronger or weaker transfer.

### 4. Cascades are mostly small and shallow, with a long tail; their number and size depend strongly on the thresholds
In the application window the method finds **12,518 cascades**: 57% involve one run, 43% two or more, the 90th percentile is 5 runs, the largest has 43 runs and 18 hops; 23% reach two or more hops. Cascades started by a primary event are larger (mean 2.83 runs, 4,861 cascades) than those started by a late entry into the corridor (mean 2.07, 7,657 cascades). Durations are heavy-tailed because rake chains (a late train delaying the next service of its rake) carry delay across days. On **fog days 71 cascades start per day against 38 on clear days (42 on rain days), but they are not larger** (mean size 2.21 against 2.37; most of the extra fog cascades come from late entries into the corridor, since primary-rooted cascades rise only from 15.6 to 18.7 per day). The thresholds matter: raising the primary threshold from 10 to 30 minutes cuts the number of cascades from 18,942 to 4,086 and raises the mean size from 1.98 to 2.79; widening the gap window from 4 to 15 minutes raises the mean size from 2.06 to 2.87 and the maximum from 31 to 55. The decision-window results (12,221 cascades, mean size 2.39) are very close to the application window (2.37).
*Figures:* `mining_04_cascade_distributions`, `mining_05_sensitivity`, `mining_09_cascade_time_space` (one cascade; map in `mining_09_cascade_map_SIMULATED.html`).
*SIMULATED caveat:* a cascade here is a tree built by an extraction rule, not an observed fact; sizes depend on the rule.
*Likely artifact?* **Partly.** Daily running makes rake chains endless, and one lane and headway queueing create most of the chains; but the strong dependence on P and G is a property of any threshold-based extraction.

### 5. A few sections start most cascades, and they are the ones the simulator makes risky; graph centrality does not find them
By how often a primary event starts a cascade there, the most critical sections are **ETW-EKL (454 cascades, 20,135 primary-event minutes, 904 runs triggered), CNB-CNBI (274), ARW-JEP (237), TDL-HNG (136), PATA-PHD (123)**. The three cascade-based rankings (origin frequency, primary delay minutes, cascade size) agree strongly (Spearman 0.82-0.93) and are stable between the decision and the application window (0.81-0.89). Centrality rankings agree poorly with them: PageRank on the station graph weighted by contagion (0.16-0.30), edge betweenness (0.25-0.30) and the train-dependency graph score (0.21-0.32); betweenness on a chain is topological (unique shortest paths) and mostly ranks the middle of the line. Among trains, Janata Express 13040 leads the most delayed followers (745 links) and Tundla-Kanpur Passenger 51810 starts the most cascades (166).
*Figures:* `mining_06_critical_sections`.
*Artifact flags:* the only top-ranked appearance of KNS-PJY is edge betweenness (the middle of the chain), not a cascade-based rank, so it does not trace to the KNS-PJY data quirk; no section has half or more of its primary-event minutes from the Shatabdi-artifact trains (12003, 12033), and no Shatabdi train is among the top trains.
*SIMULATED caveat:* a ranking of this simulated corridor.
*Likely artifact?* **Yes, very likely.** The top sections touch junctions or congested nodes (ETW, CNB, TDL, MGS), where the simulator doubles the incident probability and raises the freight rate; the method recovers the design's hot spots. Whether it would do so on real data cannot be told from this.

### 6. Propagation mostly stays inside a section or a segment; fog changes how often cascades start, not where they go
The most frequent section sequences repeat the **same section**: RURA-RMW then RURA-RMW then RURA-RMW (134 chains, support 3.6%, confidence 48%) and TDL-HNG (113, confidence 27%) are queues of trains waiting behind a late leader. Sequences that move along the line are rare and weak: the best triple is ARW-JEP, then KCN-BEP, then GHSR-ULD (48 cascades, support 1.6%, confidence 52%), and the most frequent pairs are JEP-MGS then SFG-PRYG (179) and TDL-HNG then HNG-FZD (113). **60% of chain steps stay inside the same physical segment** (NDLS-TDL, TDL-ETW, ETW-CNB, CNB-PRYG, PRYG-MGS). Patterns specific to fog are essentially absent: at most one pair per fog season is at least twice as common as elsewhere, with 15-18 occurrences.
*Figures:* `mining_07_segment_transitions`.
*SIMULATED caveat:* patterns of section ids in a simulated network.
*Likely artifact?* **Likely.** No passing between stations makes queues within a section the dominant pattern, and daily running makes the same sections interact every day; moving patterns are rare because most cascades have one or two hops.

### 7. Stations that move together form contiguous groups that only partly match the physical segments
Clustering stations by the correlation of their hourly delay (daily pattern removed) gives groups that are contiguous along the line (92-96% of neighbouring stations share a cluster) but their boundaries only partly follow the segments: the adjusted Rand index against the five segments is **0.47-0.50 with the frozen k = 10, 0.68 with k = 5, and 0.64-0.69 for a Louvain clustering** (5-6 groups). With k = 5 the boundaries fall at SBB, SKB, RURA and KCN against the physical TDL, ETW, CNB and PRYG: near TDL and PRYG but shifted, and not at ETW or CNB. The silhouette score on the decision window kept rising to the end of the search range (0.19 at k = 2, 0.41 at k = 10), so k = 10 is the cap of the range, not a natural cluster number.
*Figures:* `mining_08_communities`.
*SIMULATED caveat:* correlation of hourly delay along a single line is dominated by trains carrying their delay from one station to the next.
*Likely artifact?* **Partly.** Because delay travels with trains (finding 1), correlation mostly reflects distance along the line and shared trains, which makes groups contiguous by construction; the segment boundaries have no special meaning in the simulator.

## Items to keep in mind
- Findings 1-3 depend on the flat-slack and headway design; the robustness run on the low and high scenarios (item 9) shows which numbers move with congestion.
- Validation against the ground truth (item 8, `07_mining_validation.ipynb`) is the only place the true labels are read; its results decide how much of findings 2-5 can be trusted.
- Per-train results for 12003 and 12033 were not interpreted (known Shatabdi artifact).

## Validation against the simulator's ground truth (item 8; SIMULATED)

Notebook `notebooks/07_mining_validation.ipynb`, the only place the true labels are read (`validation_only=True`). Window 2025-01-01..2025-08-31, medium scenario. The v1 method is frozen and was not changed after these results. Figures: `validation_01..05`.

| Question | Result |
|---|---|
| Link precision (inferred link = follower, station, leader train; true = section wait or yield) | **0.81** (0.95 for actual gaps of 4-6 min, 0.89 for 2-4 min, 0.35 for gaps under 2 min) |
| Link recall (true section waits and yields recovered) | **0.27** for section waits, 0.05 for yields, 0 for everything not caused by a train (by design) |
| Rake links | precision 0.52, recall 0.42 (116 of 279 true rake delays) |
| Entry-delay cascades | 5,674 of 5,752 external-entry cascades match a true entry delay of the same train; entry delays are found well |
| True cause of the inferred "primary events" | **38% yields**, 28% incidents, 28% section waits, 4% freight, 1% disruption, 0.5% halt extension |
| Large incidents (15 min or more) found as an inferred primary origin | 61%; only 30% of inferred primary-origin cascades match a true incident |
| Inferred vs true cascade size | inferred mean 2.8 runs vs true 6.1 (runs with 2+ min); Spearman 0.38 (size is underestimated and only weakly ordered; larger incidents do give larger inferred cascades: 2.5, 3.1, 4.8 runs for incidents of 15-30, 30-60, 60+ min) |
| Origin-section ranking vs truth (Spearman) | 0.69 (frequency vs true large incidents), 0.75 (size triggered), 0.69 (primary minutes vs true generated minutes); top-8 lists overlap on 4 sections |
| Contagion vs true share of delay caused by other trains (214 section-directions) | explained share rho: Spearman **0.63**, but its level is too low (median 0.49 vs true 0.78); pass-on ratio 0.29 / 0.14; the linear coefficients beta and c: about 0 |

**What this means for findings 1-7.**
- Reliable: where the mined links exist they are mostly right (precision 0.8, higher in the 2-6 min range), entry delays, the ordering of sections by where delay starts (Spearman about 0.7), the ordering of sections by how much of their delay is inherited (rho, 0.63).
- Not reliable: absolute cascade sizes (about half the true size), the proportion of delay that is primary (v1 counts yields as primary delay), the linear contagion coefficients beta and c as measures of contagion (no relation to the true share), and rake links as a class (half of them wrong).
- Failure modes: (1) **yields**: a train held for a faster train enters after it, so the cause is not its leader and v1 calls the delay primary; 38% of v1 primary events. (2) **Waits behind a leader with a gap under 2 min** (precision 0.35) and gaps over 6 min (0.13): the ordering of entry is unclear or the leader was not the cause. (3) **Rake reuse versus section waits**: a late rake delaying the next service is hard to tell from the line being busy, so half of the rake links are false and 58% of true rake delays are missed. (4) Cascades are cut where the delay of a follower falls below the 2-minute link threshold, which truncates trees and shrinks sizes; the simulator also records delay spreading to followers of followers in ways the link rule does not see.

**Proposed v2 of the primary-event rule (not applied to v1 results).** A large delay is not primary if its leader is of a higher class, entered at most `G_lead` min earlier and was on time (`src/analysis/mining_v2.py`, tests in `tests/test_mining_v2.py`). `G_lead = 8` min is calibrated from observable data on the decision window only (`configs/mining_v2.yaml`): a faster on-time leader raises the chance of a 5-minute delay only for gaps up to 8 min and the effect vanishes after that. An earlier attempt using the faster train *behind* (`G = 2`, then 6) had no signal and was dropped; it is not part of the proposal.

| Primary events | v1 (frozen) | v2 (proposed) |
|---|---|---|
| 2025-01..08 (design-informed, optimistic): events | 6,728 | 4,087 |
| share really generated (incident, disruption, freight, halt ext.) | 0.34 | 0.48 |
| share that were yields | 0.38 | 0.08 |
| recall of true generated delays of 15 min or more | 0.88 | 0.78 |
| 2024-11..12 (independent check): share really generated | 0.37 | 0.53 |
| share yields | 0.38 | 0.07 |
| recall | 0.92 | 0.86 |

v2 removes most of the yields, but it also discards some real delays (recall drops 6-10 points) and the share really generated is still about half, because section waits mislabelled as primary remain (45% of v2 primary events). It was designed after seeing the 2025 labels, so it is a proposal for a later phase, not validated as a replacement. The independent 2024-11..12 check gives the same picture.

## Robustness across scenarios (item 9; SIMULATED)

The frozen pipeline was run unchanged on the low, medium and high scenarios for 2025-01-01..2025-08-31 (notebook `08_mining_robustness.ipynb`, code `src/analysis/robustness.py`, figures `robustness_01..03`; no ground truth read, no re-tuning). The scenarios differ in headway (3 / 4 / 6 min) and freight (0.02 / 0.04 / 0.06) only; same timetable and trains.

- **Critical-junction ranking: fairly stable.** Spearman of the section rankings between scenarios is 0.91 / 0.78 / 0.85 (low-medium / low-high / medium-high) for origin frequency, 0.89 / 0.81 / 0.84 for cascade size triggered, 0.94 / 0.82 / 0.86 for primary minutes, 0.90 / 0.81 / 0.78 for station PageRank. The same sections lead everywhere (ETW-EKL, CNB-CNBI, ARW-JEP, PATA-PHD, PHD-KNS in the top 6 of every scenario; TDL-HNG and SKB-KAA join with congestion), but the top-10 lists share only 5-9 sections. Edge betweenness is identical (Spearman 1.0) because it depends on the graph, which is the same in all scenarios: it says nothing about delay.
- **Cascade size distribution: nearly the same shape.** Median 1 run, 90th percentile 5 in all three; mean 2.2 / 2.4 / 2.4, 99th percentile 12 / 14 / 14; share of cascades with 10+ runs 2.0% / 2.9% / 3.2%. Congestion changes the **number** of cascades far more than their size: 8,946 / 9,640 / 12,033 cascades, and primary events 4,926 / 6,728 / 14,696 (the high scenario triples them, mostly longer waits that pass the 15-minute threshold). The upper tail (maximum 35 / 43 / 31) is noisy.
- **Contagion coefficients: levels stable, per-section ordering stable for the aggregate measures only.** Median explained share rho 0.46 / 0.49 / 0.48, rank correlation between scenarios 0.88-0.94; the pass-on ratio of all added delay A_all 0.012 in all three (rank correlation 0.88-0.91). The regression coefficients beta and c are small (median 0.003 and 0.001-0.003) and much less stable (0.51-0.69), and the pass-on ratio after primary events A_primary rests on few sections (34-77) and has rank correlation 0.51-0.59. All sections are labelled "absorb" in every scenario.

**Caveats.** Rankings are from one simulated year-part per scenario, and the same timetable and event schedule seeds are used for the three runs, so part of the stability is shared structure (stations, one lane, priorities), not evidence the method would be stable on real data. Together with the validation, this supports using the origin-section ranking and rho; it does not support using beta or c.
