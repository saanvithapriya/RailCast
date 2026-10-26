# Phase 1: Data source and scope decision

Date: 2026-10-17. Status: draft for review. Nothing has been downloaded or scraped.

**Evidence quality.** Facts below are tagged: **[verified]** = read on the source page itself; **[snippet]** = from a web-search summary, not confirmed on the source; **[unverified]** = page would not load or showed no details (Kaggle pages render with JavaScript and returned only titles); **[estimate]** = my own rough knowledge, to be checked against the timetable once we have it. Treat anything not marked [verified] as a lead, not a fact.

## 1. Candidate corridors

Train counts are **[estimate]** (approximate daily-ish passenger services including weekly ones; the real figures should be computed from the timetable in Phase 2). Station counts are for all stops on the main line plus its immediate branches, not only major halts.

| # | Corridor | Approx. stations | Major junctions | Approx. trains (both directions) |
|---|----------|------------------|-----------------|----------------------------------|
| A | Delhi - Mathura - Agra - Gwalior - Jhansi - Bhopal (North Central / West Central) | 60-90 | New Delhi, Mathura Jn, Agra Cantt, Gwalior, Jhansi, Bina, Bhopal | 150-250 |
| B | Delhi - Kanpur - Prayagraj - Mughalsarai/DDU (Howrah main line, west half) | **[verified, datameet] 435 stations with any corridor stop; 125 "core" stations served by 20+ corridor trains; 235 served by 5+** (estimate was 70-110) | New Delhi, Tundla, Etawah, Kanpur Central, Prayagraj, Mughalsarai (DDU) | **[verified, datameet] 237 trains stop at 2+ anchor junctions; 165 at 4+** (estimate was 200-300). Counts are by train number, not by operating day, and the data is old (see section 2a) |
| C | Mughalsarai - Patna - Asansol - Howrah (Howrah main line, east half) | 80-130 | DDU, Buxar, Patna/Danapur, Jasidih, Dhanbad, Asansol, Burdwan, Howrah | 200-300 |

Notes: B and C are among the densest passenger corridors in the country, so they have the most interacting trains (good for cascade mining). A is somewhat simpler and shorter, with a dense Delhi-Agra section and a thinner Jhansi-Bhopal section. Freight is heavy on B and C and is not in any passenger timetable, which matters for cascades (see risks).

## 2. Timetable sources

| Source | Link | License | Size | Contents | Trains / stations | Recency | Arr/dep per stop | Distance per stop |
|---|---|---|---|---|---|---|---|---|
| **datameet/railways** | https://github.com/datameet/railways | **CC0** [verified] | Not shown on page [unverified] | `schedules.json` (one row per train stop: train_number, station_code, day, arrival, departure), `trains.json` (GeoJSON; train-level distance, duration, from/to), `stations.json` (GeoJSON; code, name, zone, state, coordinates) [verified] | Counts not stated [unverified] | No date stated; 3 commits; collected over "a few months" by two people [verified]. Likely years old [estimate] | **Yes**, plus a `day` field [verified]. Arrival is the string `"None"` for origin stops [verified] | **No per-stop distance** in schedules (train-total only) [verified] |
| **Neo2308/indianrailways-gtfs** (GTFS feed) | https://github.com/Neo2308/indianrailways-gtfs ; mirror https://mobilitydatabase.org/feeds/gtfs/mdb-2867 | **Not stated** [verified: absent] | 3.98 MB zip / 18 MB unzipped [verified] | GTFS: 10,594 routes [verified]; stops/trips counts not shown | Not shown | Generated about Nov 2025; mirror fetched 2026-08-31 [verified]. Labeled "unofficial"; 2 validation errors [verified] | Likely (GTFS stop_times) but **not confirmed** | `shape_dist_traveled` **not confirmed** |
| **data.gov.in "Indian Railways Train Time Table"** | https://www.data.gov.in/catalog/indian-railways-train-time-table | Government open data licence typical, **not confirmed** | Resource list showed "No Result Found" [snippet] | Described as train-wise arrival/departure, route, distance | - | - | Described yes [snippet] | Described yes [snippet] | 
| **Kaggle: bhavyarajdev "Indian Railways schedule, prices, availability"** | listed via baselight mirror (baselight has shut down); Kaggle page not read | [unverified] | [unverified] | IRCTC scrape, Oct 2023 [snippet] | [unverified] | Oct 2023 [snippet] | [unverified] | [unverified] |
| **Kaggle: sripaadsrinivasan "Indian Railways Dataset"**, **dnyaneshyeole "Indian Trains"** | https://www.kaggle.com/datasets/sripaadsrinivasan/indian-railways-dataset , https://www.kaggle.com/datasets/dnyaneshyeole/indian-trains | [unverified] | [unverified] | Page content not retrievable | [unverified] | [unverified] | [unverified] | [unverified] |

Takeaway: the only timetable source I could fully inspect is **datameet** (CC0, per-stop arrival/departure/day, no per-stop distance). The **GTFS feed** is newer and probably better, but its license is unknown and its internals are unverified. Per-stop distance can be recovered from `stations.json` coordinates or from the GTFS shapes if needed, but that is Phase 2 work.

### 2a. Update after downloading datameet and inspecting it (2026-10-17)

Downloaded untouched into `data/raw/`: `schedules.json` 82.2 MB, `trains.json` 14.8 MB, `stations.json` 1.9 MB. Explored in `notebooks/01_explore_timetable.ipynb`.

- **[verified]** 5,208 trains, 417,080 train-stop rows, 8,990 stations in `stations.json` (8,539 appear in schedules).
- **[verified]** `schedules.json` columns: arrival, day, train_name, station_name, station_code, id, train_number, departure. `id` is consecutive within a train and gives route order.
- **[verified] Missing values:** arrival is the string "None" in 27,835 rows and departure in 27,827 (origin/terminal stops); `day` is null in 22,561 rows (stops with no time at all). In `trains.json`, 15 trains have "None" arrival/departure and null distance/duration; 599 have no return train. 4,532 stations have no state/zone/address and 293 have no geometry type (8,990 have no top-level `geometry` field, coordinates are under `geometry.coordinates`).
- **[verified] The data is old.** It uses pre-rename names (ALLAHABAD JN, MUGHAL SARAI JN) and codes (ALD, MGS rather than PRYJ, DDU). Allahabad was renamed Prayagraj in 2018 and Mughal Sarai to Deen Dayal Upadhyaya in 2018, so it predates that. With 5,208 trains it is also well below the current network size. Treat it as a structural timetable (useful for topology and realistic scheduling), not as the current schedule.
- **[verified]** Some trains have more than 600 rows (e.g. 15905/15906 Dibrugarh-Kanyakumari, 698), including 443 exact duplicate (train, station, times, day) rows across the file. Not fixed; flagged for Phase 2.
- **[verified]** Per-stop distance is absent from `schedules.json`, as expected.

### 2b. Neo2308 GTFS feed: license and contents (not downloaded)

- **[verified] No license.** The GitHub API returns `license: null` and the repository root has no LICENSE file. With no license, default copyright applies and reuse is not permitted. The feed is therefore **not permissively licensed**.
- **[verified]** Newer: repo last pushed 2026-08-30; Mobility Database fetched the feed on 2026-08-31; README mentions data generated around Nov 2025. 3.98 MB zip, 10,594 routes.
- **[verified]** It is generated by a Go program that queries an external API with an API key (the README does not say which service). So its upstream source and terms are also unknown.
- **Not confirmed:** stop counts, whether stop_times and `shape_dist_traveled` are populated.
- **Conclusion:** newer but unlicensed. I would only use it with the author's written permission. I have not downloaded it. Asking the author to add a license is a cheap next step.

## 3. Delay data sources

| Source | What it contains | Span / coverage | License / terms | Scraping allowed? |
|---|---|---|---|---|
| **RSTGCN dataset** (arXiv 2510.01262), scraped from runningstatus.in | Per train, per station, per day: scheduled/actual arrival and departure, arrival/departure delay, distance from source [verified] | 1-30 Sep 2024; 3,892 long-distance passenger trains, 4,735 stations, 17 zones; local and freight excluded [verified] | Not stated. Authors say it will be released "upon acceptance"; **no download link** [verified] | N/A (not obtainable yet). Could email the authors |
| **runningstatus.in** (their upstream) | Live/historical running status | - | Terms page and robots.txt returned **HTTP 403** to my fetch; unknown [verified: blocked] | **Unknown; assume not permitted** until read in a browser |
| **NTES** (enquiry.indianrail.gov.in, CRIS) | Official live running status | Live only; no bulk history known | Terms not found; captcha-protected [snippet, 2013 article] | **Unknown; likely restricted.** Needs the footer terms read in a browser |
| **RailYatri / "Where is my train"** | Live status; RailYatri reportedly supplies the data behind RailRadar with a 5-minute delay [snippet, anonymous comment] | Live | Terms not found | Unknown; assume restricted |
| **Kaggle: naijilaji "Indian Railways Train Delays Dataset 2025"** | Per train-station **average** delay, on-time %, severity (~1,900 train-station rows), reportedly scraped from ETrain.info [snippet] | Aggregated, not per-run [snippet] | [unverified] | Source-site terms unknown |
| **Kaggle: vishwassrivastava1 "Indian Railway Delay Dataset"** | Reportedly **simulated** for learning [snippet] | One sampled date/year [snippet] | [unverified] | N/A; not real |
| **Gaurav & Srivastava 2018 (IEEE)** | 135 trains' running status over two years [snippet] | Small, old | Data availability unknown | - |
| **Ministry of Railways punctuality index** (via dataful.in) | Annual aggregate on-time % (e.g. 77.1% in 2024-25) [snippet] | Yearly, national | Government data | Aggregate only; not usable for cascades |

Takeaway: **I found no openly downloadable, per-run, per-station historical delay dataset with a clear license.** The best match (RSTGCN) is not yet released. Everything else is aggregated, simulated, live-only, or has unclear/blocked terms.

## 4. Real delay data per corridor

| Corridor | Does real delay data exist? | How much we can get today |
|---|---|---|
| A (Delhi-Agra-Jhansi-Bhopal) | Only inside the RSTGCN data (Sep 2024), which would cover its trains | **None accessible now.** If RSTGCN is released: roughly one month of per-run data |
| B (Delhi-Kanpur-Prayagraj-DDU) | Same | Same |
| C (DDU-Patna-Asansol-Howrah) | Same | Same |

I cannot say whether any of these is better covered, because I have no access to the data. The aggregated ETrain-style Kaggle set could give average delay per train-station for any corridor, but not time-ordered runs, so it cannot support cascade mining or 1-6 hour prediction.

## 5. Recommendation

**Hybrid, leading with a simulator on the real timetable, with real delays as a validation/calibration layer if we can get them legitimately.**

Reasons:
1. The project needs time-ordered, per-run, per-station delays to mine cascades and train 1-6 h predictors. No open source with a clear license provides that today.
2. The one dataset that does (RSTGCN, one month) is unreleased. A month is also thin for seasonality (fog season, monsoon), though adequate for method development.
3. Scraping runningstatus.in, NTES or RailYatri is legally and ethically unclear, and two of those returned 403 or captcha signals. We should not build the project on data we may not be allowed to collect.
4. A simulator on the real timetable (propagating delays through shared tracks, junction conflicts and rake/turnaround links) lets us **test whether the mining and prediction methods recover known cascades**, which real data cannot do. Its weakness: findings are only as realistic as the simulator's assumptions, so results must be labeled as simulated.
5. Hybrid keeps the door open: if the RSTGCN data arrives (or NTES permission is granted), we calibrate simulator parameters on it and evaluate on it.

**Recommended corridor: B (Delhi - Kanpur - Prayagraj - Mughalsarai).** It has dense mixed traffic, well-known bottleneck junctions (Tundla, Kanpur, Prayagraj), and the size (70-110 stations) fits the 50-150 target. Choose A instead if you want a smaller first pass. This is a judgment call on estimates; Phase 2 should confirm train counts from the timetable.

## 6. Risks and what to verify next

- **Timetable currency.** datameet's age is unknown; train numbers, timings and even stations may have changed (renumbering, new Vande Bharat services, electrification). Check commit dates, and compare a handful of trains against the current official timetable.
- **GTFS license and quality.** Unlicensed and unofficial; contents unverified. Ask the author about the license before relying on it.
- **Missing per-stop distance** in datameet; need coordinates or GTFS shapes.
- **Freight and non-passenger traffic** is absent from every source, yet it drives much real congestion. A cascade model on passenger trains alone will under-explain delays.
- **Kaggle claims** (naijilaji source, "simulated" label, licenses) come from search snippets and must be read in a browser.
- **Terms of use.** Read runningstatus.in, NTES and RailYatri terms in a browser; consider contacting RSTGCN authors for data access and their license.
- **Simulator realism.** Without real delays to calibrate against, simulated cascades could look plausible but be wrong. Keep simulated and real results clearly separated.

## 7. Decisions made

1. Corridor B approved.
2. Hybrid approved: simulator on the real timetable, with the design kept open so real delay data can replace simulated delays later.
3. datameet downloaded (section 2a). The GTFS feed is not downloaded (section 2b).
4. Email to the RSTGCN authors drafted in `reports/email_rstgcn_authors.md`; not sent.

## 8. Next to verify

- Decide whether the datameet timetable's age is acceptable, or whether to ask the GTFS author for a license (section 2b).
- Corridor B station list: the 435-station "any stop" set includes branch and loop stops; the 125-station core set is the closer match to the 50-150 target. Pick a definition in Phase 2.
- Whether the RSTGCN authors reply and with what license.

## 9. Final decisions (addendum, after Phases 2 and 2b)

These record what was actually decided and built; sections 1-8 above are the original Phase 1 analysis and are left as written.

- **Corridor:** B, Delhi - Kanpur - Prayagraj - Mughalsarai. Anchors for the corridor definition are NDLS, TDL, ETW, CNB, ALD, MGS (Old Delhi dropped).
- **Stations:** **110** core stations (served by 20+ corridor trains, off-line stations removed); 430 in the wider set. Prayagraj (ALD, NYN, COI) is merged into one node in the simulator, giving 108 nodes and 107 sections.
- **Trains:** 229 corridor trains in the cleaned timetable, **225 after removing the 4 coach-slip entries**. 160 of the 225 enter the corridor mid-route, 39 leave the main line for part of their journey (treated as fixed-time off-corridor legs).
- **Approach: hybrid.** A simulator on the real timetable produces the delay data; the design keeps the weather source and delay layer replaceable so real delays can be swapped in if they are ever obtained. No real delay data is in the project.
- **Timetable: datameet/railways accepted** (CC0), despite its age (pre-2018 names and codes, no recent trains, no freight, no day-of-week information, so every train is treated as running daily).
- **GTFS feed (Neo2308): skipped.** It has no license, so it was not downloaded.
- **RSTGCN dataset:** a request email was drafted (`reports/email_rstgcn_authors.md`) and left for the project owner to send; nothing in the project depends on a reply.
- **Consequence:** every delay result in the project is **simulated**. Findings say whether methods work in a world we designed, not how Indian Railways behaves. See `reports/phase2b_simulator_design.md` for the simulator and its limits.
