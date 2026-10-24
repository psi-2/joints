"""Which forecast of a cluster's surge ratio transfers across events: its past surge or its occupancy?

The Lebaran analog forecasts each cluster's 2026 Lebaran ratio from the same cluster's 2025 Lebaran total
(year-on-year persistence). eda/121-122 suggest an alternative: clusters with low baseline occupancy surge
more (capacity saturation). Train has 7 labelled weekday-holiday surges (eda/122 E2). For every event:
  cluster uplift u_c = log(sum holiday tickets / sum baseline tickets) over that cluster's pairs,
then for each event e, forecast u_c(e) from
  P  persistence: mean u_c over the OTHER events (cluster effect),
  O  occupancy:   slope x log cluster baseline occupancy, slope fitted on the other events,
  PO average of P and O,
and score by the weighted correlation / MSE against the realised u_c(e) (weights = baseline tickets).
Also reports the split-half reliability of cluster surge ratios (how stable they are at all).
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eda"))
from common import base_title, fig_dir, load, style
from features import calendar

bt = importlib.import_module("122_surge_allocation_backtest")


def main():
    out = fig_dir("124_cluster_surge_stability")
    D = load()
    tr = D["train"].assign(base=lambda x: base_title(x.movie_title))
    E = bt.event_holidays(tr, calendar(D["hol"]))
    E["w_occ"] = E.occ * E.base
    C = E.groupby(["unit", "cinema_ids"]).agg(y=("y", "sum"), b=("base", "sum"), wo=("w_occ", "sum"), n=("y", "size")).reset_index()
    C = C[C.n >= 3]
    C["u"] = np.log((C.y + 1) / (C.b + 1))
    C["u"] -= C.groupby("unit").u.transform(lambda v: np.average(v, weights=C.loc[v.index, "b"]))
    C["lo"] = np.log(C.wo / C.b)
    C["lo"] -= C.groupby("unit").lo.transform("mean")
    U = C.pivot_table(index="cinema_ids", columns="unit", values="u")
    events = list(U.columns)
    rows = []
    for e in events:
        tgt = C[C.unit == e].set_index("cinema_ids")
        oth = C[C.unit != e]
        P = oth.groupby("cinema_ids").apply(lambda g: np.average(g.u, weights=g.b), include_groups=False)
        k = np.polyfit(oth.lo, oth.u, 1, w=np.sqrt(oth.b))[0]
        O = k * tgt.lo
        d = tgt.assign(P=P.reindex(tgt.index), O=O).dropna()
        d["PO"] = (d.P + d.O) / 2
        w = d.b
        for name in ["P", "O", "PO"]:
            rows.append(dict(event=e, forecast=name, clusters=len(d),
                             corr=np.corrcoef(d[name], d.u)[0, 1],
                             mse=np.average((d.u - d[name]) ** 2, weights=w),
                             mse_zero=np.average(d.u ** 2, weights=w), slope_occ=k))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{len(events)} holiday events, {U.shape[0]} clusters")
    print(R.round(3).to_string(index=False))
    S = R.groupby("forecast")[["corr", "mse", "mse_zero"]].mean()
    S["skill_vs_zero"] = 1 - S.mse / S.mse_zero
    print("\nmean over events:"); print(S.round(3).to_string())
    # split-half reliability: average u over two random halves of the events
    rng = np.random.default_rng(2026)
    rel = []
    for _ in range(200):
        p = rng.permutation(events)
        a, b = U[p[: len(p) // 2]].mean(axis=1), U[p[len(p) // 2:]].mean(axis=1)
        m = a.notna() & b.notna()
        rel.append(np.corrcoef(a[m], b[m])[0, 1])
    print(f"\nsplit-half correlation of cluster surge effects: median {np.median(rel):.3f} [{np.quantile(rel, .05):.3f}, {np.quantile(rel, .95):.3f}]")
    R.to_csv(out / "forecast_scores.csv", index=False)
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, f in enumerate(["P", "O", "PO"]):
        q = R[R.forecast == f]
        ax[0].scatter(np.arange(len(q)) + i * .2, q["corr"], label=f, s=20)
    ax[0].set_xticks(range(len(events)), events, rotation=30); ax[0].axhline(0, color="k", lw=.7)
    ax[0].set(title="Correlation of forecast vs realised cluster surge (leave-one-event-out)"); ax[0].legend()
    ax[1].scatter(C.lo, C.u, s=6, alpha=.4); ax[1].set(title="Cluster surge vs cluster baseline occupancy (event-demeaned)",
                                                        xlabel="log occupancy (demeaned)", ylabel="log uplift (demeaned)")
    fig.tight_layout(); fig.savefig(out / "cluster_surge_stability.png")


if __name__ == "__main__":
    main()
