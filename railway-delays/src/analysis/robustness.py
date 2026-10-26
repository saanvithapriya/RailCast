"""Stability of the frozen mining results across the low, medium and high SIMULATED scenarios (no re-tuning)."""
import numpy as np
import pandas as pd


def _sp(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def ranking_stability(crit, cols=("origin_freq", "size_triggered", "primary_minutes", "pagerank_station", "edge_betweenness"), top=10):
    """Spearman between scenarios of each criticality measure per section, and overlap of the top-`top` sections.

    crit: dict scenario -> critical_sections table (index by sec)."""
    names, rows = list(crit), []
    t = {n: crit[n].set_index("sec") for n in names}
    for col in cols:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x, y = t[a][col].align(t[b][col], join="inner")
                rows.append({"measure": col, "pair": f"{a} vs {b}", "spearman": _sp(x, y),
                             "top_overlap": len(set(x.nlargest(top).index) & set(y.nlargest(top).index))})
    return pd.DataFrame(rows)


def size_distribution(casc, cuts=(1, 2, 3, 5, 10, 20)):
    """Summary of cascade size (runs) and the share of cascades of at least each size."""
    s = casc.size_runs
    out = {"n_cascades": len(s), "mean": s.mean(), "median": s.median(), "p90": s.quantile(0.9), "p99": s.quantile(0.99), "max": s.max(),
           "mean_depth": casc.depth.mean(), "mean_extent": casc.extent_sections.mean()}
    out.update({f"share>={c}": float((s >= c).mean()) for c in cuts})
    return out


def contagion_stability(cont, cols=("rho", "A_all", "A_primary", "beta", "c")):
    """Spearman of each contagion measure per section-direction between scenarios, and the median level in each."""
    names, rows = list(cont), []
    t = {n: cont[n].set_index(["sec", "dir"]) for n in names}
    for col in cols:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x, y = t[a][col].align(t[b][col], join="inner")
                ok = x.notna() & y.notna()
                rows.append({"measure": col, "pair": f"{a} vs {b}", "spearman": _sp(x[ok], y[ok]) if ok.sum() > 3 else np.nan, "n": int(ok.sum()),
                             **{f"median_{n}": float(t[n][col].median()) for n in names}})
    return pd.DataFrame(rows)
