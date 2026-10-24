"""Preprocessing check for Lebaran v2: mix the year-on-year cluster ratio with a capacity cluster ratio.

Current analog (v10a / v18 with rho 0.92), pair p in cluster c on Lebaran day k:
    r_an = rho * sqrt(r_cl(c,k) * r_nat(k)),  r_cl = cluster total 2025 day k / cluster slate D1-D3 mean 2026
Capacity ratio (eda/121): seats of the 2026 slate in c on D3 x 2025 slate occupancy of c on day k / c's D1-D3 mean.
eda/124 on 7 labelled train surges: forecasting a cluster's surge from its past surge (P) has skill 0.20,
from occupancy (O) 0.39, from their average 0.44. v2 therefore uses the geometric mean of r_an and r_cap per
cluster-day, then rescales every Lebaran day so the national slate total equals the analog's (the national
level is confirmed by two independent routes in eda/121: 543k vs 570k tickets/day). Only the split across
clusters changes. Clusters without 2025 data (new venues) get capacity information instead of the national
fallback. Checks: day totals identical, factor distribution, largest movers, new clusters, plot.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, style

RHO, DAY1 = 0.92, 0.75
SLATE25 = ["PABRIK GULA", "PABRIK GULA UNCUT", "QODRAT 2", "JUMBO", "KOMANG", "NORMA: ANTARA MERTUA DAN MENANTU"]
LEB25, LEB26 = pd.Timestamp("2025-03-31"), pd.Timestamp("2026-03-21")


def seats(df):
    ok = (df.occupation_rate >= 1) & (df.occupation_rate <= 100) & (df.total_show > 0)
    return pd.Series(np.where(ok, df.total_ticket / (df.occupation_rate / 100), np.nan), index=df.index)


def lebaran_v2(train, hist, rows):
    """rows: DataFrame with movie_title, cinema_ids, date_show, scale (Lebaran-week test rows).
    Returns (r_analog, r_v2) arrays aligned with rows."""
    base = base_title(train.movie_title)
    day25 = train.groupby("date_show").total_ticket.sum()
    m25 = pd.Series({k: day25[LEB25 + pd.Timedelta(days=k - 1)] for k in range(2, 8)})
    m25.loc[1] = DAY1 * m25.loc[2]
    slate = sorted(rows.movie_title.unique())
    hs = hist[hist.movie_title.isin(slate)].copy()
    S26 = hs.groupby("date_show").total_ticket.sum().mean()
    c26 = hs.groupby("cinema_ids").total_ticket.sum() / 3
    hs["seats"] = seats(hs)
    seat26 = hs[hs.date_show == hs.date_show.max()].groupby("cinema_ids").seats.sum(min_count=1)
    s25 = train[base.isin(SLATE25)].copy()
    s25["seats"] = seats(s25)
    kday = (rows.date_show - LEB26).dt.days.values + 1
    r_an, r_v2 = np.zeros(len(rows)), np.zeros(len(rows))
    for k in range(1, 8):
        m = kday == k
        if not m.any():
            continue
        dk = LEB25 + pd.Timedelta(days=max(k, 2) - 1)
        f = DAY1 if k == 1 else 1.0
        c25 = train[train.date_show == dk].groupby("cinema_ids").total_ticket.sum() * f
        g = s25[s25.date_show == dk].groupby("cinema_ids")
        occ = (g.total_ticket.sum() / g.seats.sum(min_count=1)) * f
        occ_nat = s25[s25.date_show == dk].total_ticket.sum() / s25[s25.date_show == dk].seats.sum() * f
        r_nat = m25.loc[k] / S26
        cl = rows.cinema_ids.values[m]
        rc = pd.Series(cl).map(c25).values / pd.Series(cl).map(c26).values
        rc = np.where(np.isfinite(rc) & (rc > 0), rc, r_nat)
        an = RHO * np.sqrt(rc * r_nat)
        cap = pd.Series(cl).map(seat26).values * pd.Series(cl).map(occ).fillna(occ_nat).values / pd.Series(cl).map(c26).values
        cap = np.where(np.isfinite(cap) & (cap > 0), cap, an)
        v2 = np.sqrt(an * cap)
        sc = rows.scale.values[m]
        v2 *= (an * sc).sum() / (v2 * sc).sum()  # national slate total of the day = analog's
        r_an[m], r_v2[m] = an, v2
    return r_an, r_v2


def main():
    out = fig_dir("127_lebaran_v2")
    D = load()
    tr, h, te = D["train"], D["hist"], D["test"]
    rows = te[te.date_show.between(LEB26, LEB26 + pd.Timedelta(days=6))].copy()
    sc = (h.groupby(KEY).total_ticket.sum() / 3).clip(lower=1).rename("scale")
    rows = rows.join(sc, on=KEY)
    r_an, r_v2 = lebaran_v2(tr, h, rows)
    rows["r_an"], rows["r_v2"] = r_an, r_v2
    rows["k"] = (rows.date_show - LEB26).dt.days + 1
    tot = rows.groupby("k").apply(lambda g: pd.Series(dict(analog=(g.r_an * g.scale).sum(), v2=(g.r_v2 * g.scale).sum(),
                                                           mean_r_an=g.r_an.mean(), mean_r_v2=g.r_v2.mean())), include_groups=False)
    print(f"Lebaran rows {len(rows)} / pairs {rows.groupby(KEY).ngroups}")
    print("A. national predicted tickets per Lebaran day (must match):"); print(tot.round(2).to_string())
    assert np.allclose(tot.analog, tot.v2, rtol=1e-9)
    fac = rows.r_v2 / rows.r_an
    print(f"\nB. v2/analog factor per row: quantiles {fac.quantile([.05, .25, .5, .75, .95]).round(3).tolist()}; "
          f"mean |log| {np.abs(np.log(fac)).mean():.3f}")
    cf = rows.groupby(["cinema_ids", "city_name"]).apply(lambda g: pd.Series(dict(rows=len(g), factor=np.exp(np.log(g.r_v2 / g.r_an).mean()))),
                                                        include_groups=False).reset_index().sort_values("factor")
    print("   clusters with the largest moves:"); print(pd.concat([cf.head(6), cf.tail(6)]).round(3).to_string(index=False))
    jab = rows.city_name.isin(["JAKARTA", "BOGOR", "DEPOK", "TANGERANG", "BEKASI"])
    print(f"   Jabodetabek factor {np.exp(np.log(fac[jab]).mean()):.3f} vs others {np.exp(np.log(fac[~jab]).mean()):.3f}")
    new = ~rows.cinema_ids.isin(tr.cinema_ids.unique())
    print(f"C. rows in clusters absent from train (new venues): {int(new.sum())}; mean r analog {rows.r_an[new].mean():.2f} -> v2 {rows.r_v2[new].mean():.2f}")
    rows[KEY + ["date_show", "city_name", "scale", "r_an", "r_v2"]].to_csv(out / "lebaran_v2_rows.csv", index=False)
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist(np.log(fac), bins=50); ax[0].set(title="log(v2 / analog) per Lebaran row")
    ax[1].scatter(np.log(rows.r_an), np.log(rows.r_v2), s=4, alpha=.4); lim = [np.log(rows.r_an).min(), np.log(rows.r_an).max()]
    ax[1].plot(lim, lim, color="k", lw=.7); ax[1].set(title="log r: analog vs v2", xlabel="analog", ylabel="v2")
    fig.tight_layout(); fig.savefig(out / "lebaran_v2.png")


if __name__ == "__main__":
    main()
