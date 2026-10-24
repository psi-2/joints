"""Lebaran week: do tickets saturate toward seat capacity, so weak (low-occupancy) pairs gain far more?

The v8-v18 analog gives every pair in a cluster the same ratio r = y / scale on Lebaran day k
(rho x geometric mean of cluster and national 2025/2026 ratios). The 2026 slate D1-D3 (18-20 Mar, last
Ramadan days, mudik) sold at ~17% occupancy, while the 2025 slate ran at ~73% during Lebaran week. If
Lebaran demand fills seats, tickets follow capacity (shows x seats per show) rather than D1-D3 tickets, and
r should be much larger for pairs whose D1-D3 occupancy was low.

A. 2025 slate pairs, Lebaran week (1-7 Apr) vs the first normal week after (21-27 Apr): occupancy spread
   (is it compressed near capacity?) and, within film, how the Lebaran/normal ticket ratio depends on the
   pair's normal-week occupancy (slope -1 = pure saturation, 0 = proportional).
B. Same question on train holiday surges (weekday holidays, pair vs +-7 days) - more events, smaller surge.
C. 2026 slate pairs: implied capacity, the current analog's r, and a capacity-based alternative
   r_cap = seats x occupancy_2025(cluster, day) / scale; national totals of both vs 2025 Lebaran.
No test labels are used; 2025 facts come from train.csv, 2026 from test_history.csv (official inputs).
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
LEB25, LEB26 = pd.Timestamp("2025-03-31"), pd.Timestamp("2026-03-21")


def seats(df):
    ok = (df.occupation_rate >= 1) & (df.occupation_rate <= 100) & (df.total_show > 0)
    return np.where(ok, df.total_ticket / (df.occupation_rate / 100), np.nan)


def within_film_slope(d, x, y, film="movie_title"):
    """OLS slope of y on x after demeaning both within film."""
    dm = d[[x, y]] - d.groupby(film)[[x, y]].transform("mean")
    return float(np.polyfit(dm[x], dm[y], 1)[0]), len(d)


def part_a(tr):
    s = tr[tr.base.isin(SLATE25)]
    p = lambda a, b: s[s.date_show.between(a, b)].groupby(KEY).agg(  # noqa: E731
        y=("total_ticket", "sum"), sh=("total_show", "sum"), occ=("occupation_rate", "mean"), n=("date_show", "size"))
    L, N = p("2025-04-01", "2025-04-07"), p("2025-04-21", "2025-04-27")
    J = L.join(N, lsuffix="_leb", rsuffix="_norm", how="inner")
    J = J[(J.n_leb == 7) & (J.n_norm == 7) & (J.occ_norm > 0)].reset_index()
    J["lr"] = np.log(J.y_leb / J.y_norm)
    J["locc"] = np.log(J.occ_norm)
    J["lr_show"] = np.log(J.sh_leb / J.sh_norm)
    J["lr_tps"] = J.lr - J.lr_show
    spread = pd.DataFrame({"Lebaran week": J.occ_leb.describe(percentiles=[.1, .25, .5, .75, .9]),
                           "normal week (21-27 Apr)": J.occ_norm.describe(percentiles=[.1, .25, .5, .75, .9])}).T
    sl = {k: within_film_slope(J, "locc", k) for k in ["lr", "lr_show", "lr_tps"]}
    return J, spread, sl


def part_b(tr, cal):
    day = tr.groupby(KEY + ["date_show"]).agg(y=("total_ticket", "sum"), occ=("occupation_rate", "mean"))
    bad = set(pd.date_range("2025-06-01", "2025-06-16"))
    rows = []
    for d in cal.index[(cal.is_hol == 1) & (cal.dow < 5)]:
        d = pd.Timestamp(d)
        if not (pd.Timestamp("2025-04-14") <= d <= pd.Timestamp("2025-09-23")) or d in bad:
            continue
        nb = [d - pd.Timedelta(days=7), d + pd.Timedelta(days=7)]
        if any(n in bad or cal.is_hol.get(n, 0) == 1 for n in nb):
            continue
        g = [day.xs(x, level="date_show") for x in [d] + nb]
        j = g[0].join(g[1], rsuffix="_a", how="inner").join(g[2], rsuffix="_b", how="inner")
        j = j[(j.y_a >= 3) & (j.y_b >= 3) & (j.occ_a > 0) & (j.occ_b > 0)].reset_index()
        j["lr"] = np.log(j.y / np.sqrt(j.y_a * j.y_b))
        j["locc"] = np.log(np.sqrt(j.occ_a * j.occ_b))
        j["date"] = d.date()
        rows.append(j)
    B = pd.concat(rows, ignore_index=True)
    B["film_date"] = B.movie_title + "|" + B.date.astype(str)
    return B, within_film_slope(B, "locc", "lr", film="film_date")


def part_c(D, tr):
    h = D["hist"].assign(base=lambda d: base_title(d.movie_title))
    slate = h[h.date_show.between("2026-03-18", "2026-03-20")]
    films = slate.groupby("movie_title").date_show.min()
    films = films[films == pd.Timestamp("2026-03-18")].index
    hs = h[h.movie_title.isin(films)].copy()
    hs["seats"] = seats(hs)
    P = hs.groupby(KEY).agg(scale=("total_ticket", "sum"), sh=("total_show", "sum"), seats=("seats", "sum"),
                            occ=("occupation_rate", "mean"), y3=("total_ticket", lambda v: 0)).reset_index()
    P["scale"] = (P.scale / 3).clip(lower=1)
    last = hs[hs.date_show == "2026-03-20"].groupby(KEY).agg(sh3=("total_show", "sum"), seats3=("seats", "sum"))
    P = P.join(last, on=KEY)
    test = D["test"]
    P = P[P.set_index(KEY).index.isin(test.set_index(KEY).index)]
    # 2025 Lebaran: per cluster slate occupancy and total tickets by Lebaran day k (k=2..7 observed)
    s25 = tr[tr.base.isin(SLATE25) & tr.date_show.between("2025-04-01", "2025-04-06")].copy()
    s25["seats"] = seats(s25)
    occ25 = s25.groupby("cinema_ids").apply(lambda g: g.total_ticket.sum() / np.nansum(g.seats), include_groups=False)
    occ25_nat = s25.total_ticket.sum() / np.nansum(s25.seats)
    all25 = tr[tr.date_show.between("2025-04-01", "2025-04-06")].groupby("cinema_ids").total_ticket.sum() / 6
    c26 = hs.groupby("cinema_ids").total_ticket.sum() / 3
    r_cl = (P.cinema_ids.map(all25) / P.cinema_ids.map(c26)).values
    r_nat = all25.sum() / (hs.total_ticket.sum() / 3)
    r_cl = np.where(np.isfinite(r_cl) & (r_cl > 0), r_cl, r_nat)
    P["r_analog"] = 0.92 * np.sqrt(r_cl * r_nat)              # v10a (rho 0.92), days 2-7 average
    seats_day = P.seats3.fillna(P.seats / 3)
    o = P.cinema_ids.map(occ25).fillna(occ25_nat).clip(upper=0.95)
    P["r_cap"] = seats_day * o / P.scale                       # capacity at 2025 Lebaran occupancy, shows as on D3
    tot = pd.Series({"2025 slate, Lebaran days 2-7 (per day)": s25.total_ticket.sum() / 6,
                     "2025 all films, Lebaran days 2-7 (per day)": tr[tr.date_show.between("2025-04-01", "2025-04-06")].total_ticket.sum() / 6,
                     "2026 slate D1-D3 (per day)": hs.total_ticket.sum() / 3,
                     "2026 slate seats on D3": np.nansum(hs[hs.date_show == "2026-03-20"].seats),
                     "2025 slate seats per Lebaran day": np.nansum(s25.seats) / 6,
                     "implied 2026 per day, analog rho 0.92": float((P.r_analog * P.scale).sum()),
                     "implied 2026 per day, capacity model": float((P.r_cap * P.scale).sum())})
    return P, tot, occ25_nat


def main():
    out = fig_dir("121_lebaran_saturation")
    D = load()
    tr = D["train"].assign(base=lambda d: base_title(d.movie_title))
    cal = calendar(D["hol"])
    pd.set_option("display.width", 220)

    J, spread, sl = part_a(tr)
    print(f"A. 2025 slate pairs with all 7 days in both weeks: {len(J)} pairs, {J.movie_title.nunique()} titles")
    print("occupancy (%) distribution:"); print(spread.round(1).to_string())
    print("within-film slope on log(normal-week occupancy):",
          {k: (round(v[0], 3), v[1]) for k, v in sl.items()}, "(tickets = shows + tickets/show)")
    print(f"median Lebaran/normal ticket ratio {np.exp(J.lr.median()):.2f}, show ratio {np.exp(J.lr_show.median()):.2f}")
    J.to_csv(out / "a_2025_slate_pairs.csv", index=False)

    B, slb = part_b(tr, cal)
    print(f"\nB. train weekday-holiday surges: {len(B)} pair-days, within film-date slope of log uplift on log occupancy {slb[0]:.3f}")
    B["occ_bin"] = pd.qcut(np.exp(B.locc), 5)
    print(B.groupby("occ_bin", observed=True).lr.agg(["size", lambda v: np.exp(v.median())]).rename(
        columns={"<lambda_0>": "median_uplift"}).round(3).to_string())

    P, tot, occ25 = part_c(D, tr)
    print(f"\nC. 2026 slate pairs in test: {len(P)}; 2025 slate national occupancy during Lebaran days 2-7: {100 * occ25:.1f}%")
    print(tot.round(0).to_string())
    P["occ_bin"] = pd.qcut(P.occ, 5)
    print(P.groupby("occ_bin", observed=True)[["scale", "r_analog", "r_cap"]].median().round(2).to_string())
    print("film-level weighted mean r:")
    print(P.groupby("movie_title")[["r_analog", "r_cap"]].mean().round(2).to_string())
    P.to_csv(out / "c_2026_slate_pairs.csv", index=False)

    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(J.occ_norm, bins=40, alpha=.6, label="normal week 21-27 Apr")
    ax[0].hist(J.occ_leb, bins=40, alpha=.6, label="Lebaran week 1-7 Apr")
    ax[0].set(title="A. 2025 slate pair occupancy (%)"); ax[0].legend()
    ax[1].scatter(J.locc, J.lr, s=5, alpha=.4)
    ax[1].set(title=f"A. log(Lebaran/normal tickets) vs log normal occ (within-film slope {sl['lr'][0]:.2f})",
              xlabel="log occupancy, normal week", ylabel="log ticket ratio")
    ax[2].scatter(np.log(P.occ.clip(lower=.5)), np.log(P.r_cap), s=5, alpha=.4, label="capacity model")
    ax[2].scatter(np.log(P.occ.clip(lower=.5)), np.log(P.r_analog), s=5, alpha=.4, label="analog rho 0.92")
    ax[2].set(title="C. 2026 slate: implied log r vs D1-D3 occupancy", xlabel="log D1-D3 occupancy (%)"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out / "lebaran_saturation.png")


if __name__ == "__main__":
    main()
