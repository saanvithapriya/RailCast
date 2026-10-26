"""Item 2: contagion between trains, per section and direction (SIMULATED data, observable inputs only).

Pairs are chosen from the TIMETABLE, not from what happened: the leader of a passage is its scheduled predecessor
(the previous train scheduled to enter the same section in the same direction), and a pair is used only if the
scheduled gap is at most G minutes. Choosing pairs by actual gaps would select on the outcome (a follower that
waited has a larger actual gap).

Per section and direction the follower added delay (item 1, net of slack) is regressed on
    leader delay at the moment it enters the section (departure delay at the entry node), scheduled gap, the follower own entry delay, fog, rain, peak hour,
    leader fast, follower fast
(OLS, HC1 robust standard errors). beta = minutes of follower added delay per minute of leader delay.
m = mean number of followers scheduled to enter within G minutes after a leader; contagion strength c = beta * m is the
expected downstream delay per minute of leader delay in one hop. Label: "amplify" if the 95% interval of c is above 1,
"absorb" if below 1, otherwise "uncertain".
"""
import numpy as np
import pandas as pd

PEAK = (6, 7, 8, 17, 18, 19, 20)
FEATURES = ["const", "leader_delay", "sched_gap", "own_entry_delay", "fog", "rain", "peak", "leader_fast", "follower_fast"]


def scheduled_neighbours(p, G):
    """Leader (scheduled predecessor), scheduled gap, and the number of followers within G minutes of each passage."""
    s = p.sort_values(["key", "sched_in"], kind="stable").reset_index(drop=True)
    g = s.groupby("key", sort=False)
    for src, dst in (("sched_in", "l_sched_in"), ("d_ent", "l_d_ent"), ("d_out", "l_d_out"), ("fast", "l_fast"), ("run_id", "l_run")):
        s[dst] = g[src].shift()
    s["gap_sched"] = s.sched_in - s.l_sched_in
    comp = s.key.to_numpy().astype("float64") * 1e8 + (s.sched_in.to_numpy() + 1e6)
    idx = np.arange(len(s))
    s["n_followers"] = np.searchsorted(comp, comp + G, side="right") - idx - 1
    return s


def ols_hc1(X, y):
    """Coefficients and HC1 robust standard errors."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    n, k = X.shape
    xe = X * e[:, None]
    cov = xtx_inv @ (xe.T @ xe) @ xtx_inv * n / max(n - k, 1)
    return beta, np.sqrt(np.maximum(np.diag(cov), 0))


def design(pairs):
    return np.column_stack([
        np.ones(len(pairs)), pairs.l_d_ent.to_numpy("float64"), pairs.gap_sched.to_numpy("float64"), pairs.d_in.to_numpy("float64"),
        (pairs.weather.astype(str) == "fog").to_numpy("float64"), (pairs.weather.astype(str) == "rain").to_numpy("float64"),
        pairs.sched_hour.isin(PEAK).to_numpy("float64"), pairs.l_fast.astype("float64").to_numpy(), pairs.fast.astype("float64").to_numpy()])


def contagion_coefficients(p, G, n_min=300):
    """One row per (section, direction). `p` must carry the item 1 columns (added) and come from decompose()."""
    s = scheduled_neighbours(p, G)
    pair = s[s.l_run.notna() & (s.gap_sched >= 0) & (s.gap_sched <= G) & (s.l_run != s.run_id)]
    m = s.groupby("key").n_followers.mean()
    rows = []
    for key, g in pair.groupby("key"):
        sec, di = key // 2, (1 if key % 2 else -1)
        base = {"key": key, "sec": sec, "dir": di, "n_pairs": len(g), "m_followers": float(m.get(key, np.nan))}
        if len(g) < n_min:
            rows.append(base)
            continue
        X, y = design(g), g.added.to_numpy("float64")
        beta, se = ols_hc1(X, y)
        i = FEATURES.index("leader_delay")
        c, lo, hi = beta[i] * base["m_followers"], (beta[i] - 1.96 * se[i]) * base["m_followers"], (beta[i] + 1.96 * se[i]) * base["m_followers"]
        rows.append({**base, "beta": beta[i], "se": se[i], "beta_lo": beta[i] - 1.96 * se[i], "beta_hi": beta[i] + 1.96 * se[i],
                     "beta_sched_gap": beta[FEATURES.index("sched_gap")], "beta_own_entry": beta[FEATURES.index("own_entry_delay")],
                     "c": c, "c_lo": lo, "c_hi": hi, "mean_added_followers": float(y.mean()),
                     "label": "amplify" if lo > 1 else ("absorb" if hi < 1 else "uncertain")})
    out = pd.DataFrame(rows)
    sec_rows = s.groupby("key").agg(from_node=("from_node", "first"), to_node=("to_node", "first"))
    return out.merge(sec_rows, left_on="key", right_index=True, how="left")


def slack_absorbed(p, keys=("sec", "dir")):
    """Share of the delay present on a section that slack absorbs: sum(absorbed) / sum(inherited + absorbed + added>0)."""
    d = p.assign(present=p.inherited + p.absorbed + p.added.clip(lower=0))
    g = d.groupby(list(keys), observed=True)
    out = pd.DataFrame({"absorbed_min": g.absorbed.sum(), "present_min": g.present.sum(), "passages": g.size()})
    out["share_absorbed"] = out.absorbed_min / out.present_min
    return out.reset_index()


# ----------------------------------------------------------------------------- pass-on ratio (one hop, same section)
def pass_on_ratio(p, explained, primary, params, n_boot=200, seed=0, min_events=30):
    """Follower delay caused per minute of leader delay, per (section, direction). Needs attach_leaders (lead_pid),
    decompose (added) and the explanation / primary flags of src.analysis.decomposition.

    A follower counts if it is explained by a late leader (same section, actual gap <= G) and adds at least A_link
    minutes. For each leader, followers' added minutes are summed (one hop only) and divided by the leader delay on
    entering the section.
      A_all     = sum of followers' added minutes / sum of delay at entry of late leaders that have a follower slot
      A_primary = the same, over leaders that are primary events (large added delay no leader explains), where the
                  leader delay is its added delay in this section
      rho       = share of the delay added on the section that a leader explains
    A_primary is also given with a 95% bootstrap interval over leaders. Label: "amplify" if the interval is above 1,
    "absorb" if below 1, otherwise "uncertain" (too few primary events gives "uncertain").
    """
    fol = p[explained & (p.added >= params.A_link) & p.lead_pid.notna()]
    child = fol.groupby("lead_pid").added.sum()
    lead_ids = child.index.to_numpy().astype("int64")
    n = len(p)
    child_all = np.zeros(n)
    child_all[lead_ids] = child.to_numpy()
    p = p.assign(child=child_all, expl=explained.to_numpy(), prim=primary.to_numpy())
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in p.groupby("key"):
        late = g[g.d_ent >= params.L_late]
        pr = g[g.prim]
        row = {"key": key, "sec": key // 2, "dir": 1 if key % 2 else -1,
               "A_all": float(late.child.sum() / max(late.d_ent.sum(), 1e-9)), "n_late_leaders": len(late),
               "rho": float(g.added.clip(lower=0)[g.expl].sum() / max(g.added.clip(lower=0).sum(), 1e-9)), "n_primary": len(pr)}
        if len(pr) >= min_events:
            num, den = pr.child.to_numpy(), pr.added.to_numpy()
            row["A_primary"] = float(num.sum() / den.sum())
            idx = rng.integers(0, len(pr), (n_boot, len(pr)))
            bs = num[idx].sum(axis=1) / den[idx].sum(axis=1)
            row["A_primary_lo"], row["A_primary_hi"] = np.percentile(bs, [2.5, 97.5])
            row["label_pass_on"] = "amplify" if row["A_primary_lo"] > 1 else ("absorb" if row["A_primary_hi"] < 1 else "uncertain")
        else:
            row["label_pass_on"] = "uncertain"
        rows.append(row)
    return pd.DataFrame(rows)
