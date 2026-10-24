"""Is the weekend / weekday contrast stronger in school-term weeks (the test period) than DOW_PROF assumes?

DOW_PROF was estimated on all of train (Apr-Sep 2025), which contains the post-Lebaran school break
(Apr) and the June-July school break, when weekdays are busier. The test period (Oct-Mar) is school term
except late December. Test films starting on Thursday rise from D2 (Fri) to D3 (Sat) ~11% more than train
films (eda notes), while Wednesday films (Thu->Fri) match.

A. national daily profile (train, holidays/outage removed): log tickets minus the week mean, per weekday,
   term weeks vs school-break weeks.
B. pair-level day-to-day transitions in D1-D3 (calendar-free): log(y_{d+1} / y_d) by weekday transition,
   train term / train break / test (test_history), pairs with >= 5 tickets on both days.
C. v18 OOF median calibration: share of positive-or-zero targets above the prediction (0.5 = calibrated)
   by target weekday, term vs break, and the TW-MASE change from re-scaling weekend predictions (5-fold by
   film for the scale factor, so the gain is out-of-sample).
D. Test exposure: share of test target rows on Sat/Sun and on Mon-Thu.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, release_dates, style
from evaluate import test_weights_fd
from features import DOW_PROF, calendar

DOWN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BREAKS = [("2025-04-01", "2025-04-13"), ("2025-06-28", "2025-07-12")]  # post-Lebaran + mid-year school break
BAD = pd.date_range("2025-06-01", "2025-06-16")


def is_break(d):
    m = np.zeros(len(d), bool)
    for a, b in BREAKS:
        m |= np.asarray(d.between(a, b))
    return m


def part_a(tr, cal):
    day = tr.groupby("date_show").total_ticket.sum()
    day = day[~day.index.isin(BAD) & (cal.is_hol.reindex(day.index).values == 0)]
    df = pd.DataFrame({"lg": np.log(day.values), "dow": day.index.dayofweek,
                       "week": day.index.to_period("W-SUN"), "brk": is_break(pd.Series(day.index))}, index=day.index)
    full = df.groupby("week").dow.transform("size") >= 6
    df = df[full]
    df["dev"] = df.lg - df.groupby("week").lg.transform("mean")
    t = df.groupby(["brk", "dow"]).dev.mean().unstack(0)
    t.columns = ["term", "break"]
    t = np.exp(t)
    t["DOW_PROF (norm.)"] = DOW_PROF / np.exp(np.log(DOW_PROF).mean())
    t.index = DOWN
    return t


def transitions(tx, d1, label):
    x = tx.assign(base=base_title(tx.movie_title))
    x = x.join(d1.rename("d1"), on="base").dropna(subset=["d1"])
    x["age"] = (x.date_show - x.d1).dt.days + 1
    x = x[x.age.between(1, 3)]
    p = x.groupby(KEY + ["age"]).agg(y=("total_ticket", "sum"), date=("date_show", "first")).reset_index()
    w = p.pivot_table(index=KEY, columns="age", values="y")
    dt = p.pivot_table(index=KEY, columns="age", values="date", aggfunc="first")
    rows = []
    for a in (1, 2):
        ok = (w[a] >= 5) & (w[a + 1] >= 5)
        d0 = pd.to_datetime(dt.loc[ok, a])
        rows.append(pd.DataFrame({"from": d0.dt.dayofweek.values, "lr": np.log(w.loc[ok, a + 1] / w.loc[ok, a]).values,
                                  "date": d0.values, "age_from": a}))
    r = pd.concat(rows)
    r["set"] = label
    return r


def part_b(D, cal):
    tr = D["train"]
    d1 = release_dates(tr)
    d1b = pd.Series(d1.values, index=base_title(pd.Series(d1.index)).values).groupby(level=0).min()
    A = transitions(tr, d1b, "train")
    A = A[~pd.Series(A.date).isin(BAD).values]
    A.loc[is_break(pd.Series(A.date)), "set"] = "train break"
    A.loc[A.set == "train", "set"] = "train term"
    h = D["hist"]
    hd = h.assign(base=base_title(h.movie_title)).groupby("base").date_show.min()
    B = transitions(h, hd, "test")
    xm = pd.Series(B.date).between("2025-12-20", "2026-01-04").values | pd.Series(B.date).between("2026-02-19", "2026-03-27").values
    B.loc[xm, "set"] = "test xmas/ramadan"
    R = pd.concat([A, B], ignore_index=True)
    nxt = lambda d: (pd.Series(d) + pd.Timedelta(days=1))  # noqa: E731
    hol = (cal.is_hol.reindex(pd.to_datetime(R.date)).values == 1) | (cal.is_hol.reindex(nxt(R.date)).values == 1)
    R = R[~hol]
    R["trans"] = [f"{DOWN[f]}->{DOWN[(f + 1) % 7]}" for f in R["from"]]
    T = R.groupby(["trans", "set"]).lr.agg(["median", "size"]).unstack("set")
    cal_imp = pd.Series({f"{DOWN[f]}->{DOWN[(f + 1) % 7]}": np.log(DOW_PROF[(f + 1) % 7] / DOW_PROF[f]) for f in range(7)})
    return T, cal_imp, R


def part_c():
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    Te = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    o = X[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v18/oof.csv"), on=KEY + ["h"], how="left")
    p, y, s = o.oof_final.values, X.total_ticket.values, X.scale.values
    w = test_weights_fd(X, Te)
    dow = X.date_show.dt.dayofweek.values
    brk = is_break(X.date_show)
    cal = X.t_hol.values == 1
    df = pd.DataFrame({"dow": dow, "period": np.where(brk, "break", "term"), "above": (y > p).astype(float),
                       "below": (y < p).astype(float), "hol": cal})
    df = df[~df.hol]
    C = df.groupby(["dow", "period"]).apply(lambda g: g.above.mean() / (g.above.mean() + g.below.mean()), include_groups=False).unstack()
    C.index = DOWN
    # out-of-sample weekend scale factor: choose m on 4/5 of films, apply to the held-out fifth
    films = base_title(X.movie_title).values
    uf = np.array(sorted(set(films)))
    fo = dict(zip(uf, np.random.RandomState(2026).permutation(len(uf)) % 5))
    fold = np.array([fo[f] for f in films])
    wk = dow >= 5
    grid = np.round(np.arange(0.85, 1.31, 0.025), 3)
    tw = lambda q, m: np.average(np.abs(y[m] - q[m]) / s[m], weights=w[m])  # noqa: E731
    adj = p.copy()
    chosen = []
    for k in range(5):
        trn, val = fold != k, fold == k
        best = min(grid, key=lambda g: tw(np.where(wk, p * g, p), trn & ~brk))
        chosen.append(best)
        adj[val] = np.where(wk[val], p[val] * best, p[val])
    term = ~brk
    res = dict(chosen=chosen, TW_term_before=tw(p, term), TW_term_after=tw(adj, term),
               TW_all_before=tw(p, np.ones(len(p), bool)), TW_all_after=tw(adj, np.ones(len(p), bool)))
    curve = pd.Series({g: tw(np.where(wk, p * g, p), term) for g in grid})
    return C, res, curve


def main():
    out = fig_dir("123_weekday_profile")
    D = load()
    cal = calendar(D["hol"])
    pd.set_option("display.width", 220)
    A = part_a(D["train"], cal)
    print("A. national weekday level relative to the week's geometric mean (train, holidays removed):")
    print(A.round(3).to_string())
    print(f"   Sat/Mon-Thu ratio: term {A.loc['Sat', 'term'] / A.loc[['Mon', 'Tue', 'Wed', 'Thu'], 'term'].mean():.3f} | "
          f"break {A.loc['Sat', 'break'] / A.loc[['Mon', 'Tue', 'Wed', 'Thu'], 'break'].mean():.3f} | "
          f"DOW_PROF {DOW_PROF[5] / DOW_PROF[:4].mean():.3f}")
    A.to_csv(out / "a_national_profile.csv")

    T, cal_imp, R = part_b(D, cal)
    print("\nB. pair-level log(y_{d+1}/y_d) inside D1-D3 (holiday days removed), median and n:")
    T[("median", "calendar")] = cal_imp.reindex(T.index)
    print(T.round(3).to_string())
    T.to_csv(out / "b_transitions.csv")

    C, res, curve = part_c()
    print("\nC. v18 OOF: P(y > pred | y != pred) by target weekday (0.5 = median-calibrated):")
    print(C.round(3).to_string())
    print("   weekend scale factor chosen per fold (term rows):", res["chosen"])
    print(f"   TW term {res['TW_term_before']:.4f} -> {res['TW_term_after']:.4f} | TW all {res['TW_all_before']:.4f} -> {res['TW_all_after']:.4f}")
    curve.to_csv(out / "c_weekend_scale_curve.csv")

    Te = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    dw = Te.date_show.dt.dayofweek
    print(f"\nD. test target rows on Sat/Sun {np.mean(dw >= 5):.3f}; Mon-Thu {np.mean(dw <= 3):.3f}; Fri {np.mean(dw == 4):.3f}")

    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    A[["term", "break", "DOW_PROF (norm.)"]].plot(ax=ax[0], marker="o"); ax[0].set(title="A. Weekday level / week mean")
    tt = T["median"][["train term", "train break", "test", "calendar"]].dropna(how="all")
    tt.plot(ax=ax[1], marker="o"); ax[1].set_xticks(range(len(tt)), tt.index, rotation=30); ax[1].axhline(0, color="k", lw=.7)
    ax[1].set(title="B. Pair log(y_{d+1}/y_d) in D1-D3")
    C.plot(ax=ax[2], marker="o"); ax[2].axhline(.5, color="k", ls="--", lw=.8); ax[2].set(title="C. v18 OOF P(y > pred)")
    fig.tight_layout(); fig.savefig(out / "weekday_profile.png")


if __name__ == "__main__":
    main()
