"""Backtest: during a demand surge, should a pair's uplift be flat or scale with occupancy^beta?

eda/121: within film, the 2025 Lebaran/normal ticket ratio falls with the pair's occupancy (slope -0.65);
train weekday-holiday surges show the same (slope -0.51). The Lebaran analog (v8-v18, v10a) gives every pair
in a cluster the same r. Here the allocation rules are scored on surges whose labels ARE in train, with
the occupancy exponent beta always estimated out-of-sample (leave one film / one date out):

  E1. 2025 Lebaran: baseline week 21-27 Apr (normal) -> target 1-7 Apr per day, slate-2025 pairs.
  E2. train weekday holidays (Apr-Sep, outage excluded): baseline = geometric mean of the same pair
      +-7 days -> target = holiday.
Rules (all keep the true total uplift of the event, so only the ALLOCATION is tested):
  flat_nat   r = national ratio
  flat_clu   r = sqrt(cluster ratio x national ratio)            (the current analog's form)
  occ_nat    r = national ratio x (occ / mean occ)^beta, renormalised to the national total
  occ_clu    flat_clu x (occ / cluster occ)^beta-ish, renormalised per cluster
Score = mean |y - y_hat| / baseline scale (MASE-like), plus film-level bootstrap of the difference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, style
from features import calendar

SLATE25 = ["PABRIK GULA", "PABRIK GULA UNCUT", "QODRAT 2", "JUMBO", "KOMANG", "NORMA: ANTARA MERTUA DAN MENANTU"]


def slope(d, grp):
    dm = d[["locc", "lr"]] - d.groupby(grp)[["locc", "lr"]].transform("mean")
    return float(np.polyfit(dm.locc, dm.lr, 1)[0])


def allocate(d, beta, unit):
    """y_hat for every row of d (one event unit = film-day or holiday-date) keeping each unit's true total."""
    tot = d.groupby(unit).y.transform("sum")
    btot = d.groupby(unit).base.transform("sum")
    nat = tot / btot
    ctot = d.groupby([unit, "cinema_ids"]).y.transform("sum")
    cb = d.groupby([unit, "cinema_ids"]).base.transform("sum")
    clu = np.sqrt((ctot / cb) * nat)
    out = {"flat_nat": d.base * nat}
    f = d.base * clu
    out["flat_clu"] = f * tot / f.groupby(d[unit]).transform("sum")
    w = np.exp(beta * (d.locc - d.groupby(unit).locc.transform("mean")))
    g = d.base * w
    out["occ_nat"] = g * tot / g.groupby(d[unit]).transform("sum")
    g2 = d.base * clu * w
    out["occ_clu"] = g2 * tot / g2.groupby(d[unit]).transform("sum")
    return out


def score(d, preds, film_col):
    rows, diffs = {}, {}
    for k, p in preds.items():
        e = np.abs(d.y - p) / d.scale
        rows[k] = e.mean()
        diffs[k] = e
    rng = np.random.default_rng(2026)
    films = d[film_col].values
    uf = np.unique(films)
    res = []
    for k in preds:
        dd = (diffs[k] - diffs["flat_clu"]).groupby(films).agg(["sum", "size"])
        ix = rng.integers(0, len(uf), (2000, len(uf)))
        b = dd["sum"].values[ix].sum(1) / dd["size"].values[ix].sum(1)
        res.append(dict(rule=k, score=rows[k], delta_vs_flat_clu=rows[k] - rows["flat_clu"],
                        ci_low=np.quantile(b, .025), ci_high=np.quantile(b, .975)))
    return pd.DataFrame(res)


def event_lebaran(tr):
    s = tr[tr.base.isin(SLATE25)]
    base = s[s.date_show.between("2025-04-21", "2025-04-27")].groupby(KEY).agg(
        bsum=("total_ticket", "sum"), occ=("occupation_rate", "mean"), n=("date_show", "size"))
    base = base[(base.n == 7) & (base.occ > 0)]
    tgt = s[s.date_show.between("2025-04-01", "2025-04-07")][KEY + ["date_show", "total_ticket"]]
    d = base.reset_index()[KEY].merge(pd.DataFrame({"date_show": sorted(tgt.date_show.unique())}), how="cross")
    d = d.merge(tgt.groupby(KEY + ["date_show"], as_index=False).total_ticket.sum(), on=KEY + ["date_show"],
                how="left").fillna({"total_ticket": 0})
    d = d.join(base, on=KEY)
    d["base"] = d.bsum / 7
    d["scale"] = d.base.clip(lower=1)
    d["y"] = d.total_ticket
    d["locc"] = np.log(d.occ)
    d["film"] = base_title(d.movie_title)
    d["unit"] = d.film + "|" + d.date_show.astype(str)
    d["lr"] = np.log((d.y + 1) / (d.base + 1))
    return d


def event_holidays(tr, cal):
    day = tr.groupby(KEY + ["date_show"]).agg(y=("total_ticket", "sum"), occ=("occupation_rate", "mean"))
    bad = set(pd.date_range("2025-06-01", "2025-06-16"))
    rows = []
    for dte in cal.index[(cal.is_hol == 1) & (cal.dow < 5)]:
        dte = pd.Timestamp(dte)
        if not (pd.Timestamp("2025-04-14") <= dte <= pd.Timestamp("2025-09-23")) or dte in bad:
            continue
        nb = [dte - pd.Timedelta(days=7), dte + pd.Timedelta(days=7)]
        if any(n in bad or cal.is_hol.get(n, 0) == 1 for n in nb):
            continue
        a, b = day.xs(nb[0], level="date_show"), day.xs(nb[1], level="date_show")
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j.y_a >= 3) & (j.y_b >= 3) & (j.occ_a > 0) & (j.occ_b > 0)]
        j = j.join(day.xs(dte, level="date_show").y.rename("y"), how="left").fillna({"y": 0}).reset_index()
        j["base"] = np.sqrt(j.y_a * j.y_b)
        j["occ"] = np.sqrt(j.occ_a * j.occ_b)
        j["date_show"] = dte
        rows.append(j)
    d = pd.concat(rows, ignore_index=True)
    d["scale"] = d.base.clip(lower=1)
    d["locc"] = np.log(d.occ)
    d["film"] = base_title(d.movie_title)
    d["unit"] = d.date_show.astype(str)
    d["lr"] = np.log((d.y + 1) / (d.base + 1))
    return d


def run(d, unit, loo_col, name):
    preds = {k: np.zeros(len(d)) for k in ["flat_nat", "flat_clu", "occ_nat", "occ_clu"]}
    betas = {}
    for g in d[loo_col].unique():
        m = (d[loo_col] == g).values
        b = slope(d[~m], unit)  # within-unit slope, other films/dates only
        betas[g] = b
        p = allocate(d, b, unit)
        for k in preds:
            preds[k][m] = p[k].values[m]
    S = score(d, preds, "film")
    print(f"\n{name}: {len(d)} rows, {d.film.nunique()} films, {d[unit].nunique()} units; LOO beta "
          f"{np.mean(list(betas.values())):.3f} (range {min(betas.values()):.3f}..{max(betas.values()):.3f})")
    print(S.round(4).to_string(index=False))
    return S, preds


def main():
    out = fig_dir("122_surge_allocation")
    D = load()
    tr = D["train"].assign(base=lambda x: base_title(x.movie_title))
    cal = calendar(D["hol"])
    pd.set_option("display.width", 200)
    E1 = event_lebaran(tr)
    S1, P1 = run(E1, "unit", "film", "E1 2025 Lebaran (baseline 21-27 Apr -> 1-7 Apr)")
    E2 = event_holidays(tr, cal)
    S2, P2 = run(E2, "unit", "unit", "E2 train weekday holidays (baseline +-7 days -> holiday)")
    # across-film part: does a film's mean occupancy also predict its uplift within a date?
    F = E2.groupby(["unit", "film"]).agg(lr=("lr", "mean"), locc=("locc", "mean"), n=("lr", "size"))
    F = F[F.n >= 10].reset_index()
    print(f"\nE2 across films within date: slope of film mean log uplift on film mean log occ {slope(F, 'unit'):.3f} ({len(F)} film-dates)")
    pd.concat([S1.assign(event="lebaran2025"), S2.assign(event="holidays")]).to_csv(out / "scores.csv", index=False)
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, d, P, t in [(ax[0], E1, P1, "E1 2025 Lebaran"), (ax[1], E2, P2, "E2 train holidays")]:
        q = pd.qcut(d.occ, 5)
        for k, c in [("flat_clu", "#e34948"), ("occ_clu", "#2a78d6")]:
            r = ((d.y - P[k]) / d.scale).groupby(q, observed=True).mean()
            a.plot(range(5), r.values, marker="o", color=c, label=k)
        a.axhline(0, color="k", lw=.7); a.set_xticks(range(5), [f"Q{i + 1}" for i in range(5)])
        a.set(title=f"{t}: mean (y - y_hat)/scale by occupancy quintile", xlabel="baseline occupancy quintile"); a.legend()
    fig.tight_layout(); fig.savefig(out / "allocation_bias.png")


if __name__ == "__main__":
    main()
