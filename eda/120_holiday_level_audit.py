"""Is the calendar holiday level (weekday holiday = Saturday level, HOL_LEVEL 1.29) too low?

Friend's EDA: cuti 18 Aug 2025 tickets 2.56x neighbouring Mondays; test films crossing Christmas have
D2/D1 1.45x what the v8 calendar expects. Our c(t) caps a weekday holiday at the Saturday level, so a
too-low holiday level makes c_mult wrong in two places: target dates on holidays (under-prediction) and
D1-D3 windows containing holidays (scale inflated -> over-prediction afterwards).

A. train, pair level: tickets on a weekday holiday / geometric mean of the same pair 7 days before and after
   (same weekday, non-holiday, not outage), vs the ratio our calendar implies.
B. film level, train + test_history: log T(film, day) = film FE + age + weekday + holiday terms (OLS),
   holiday split into train-period / test non-Ramadan / test Ramadan, Christmas-New Year separately.
   Expressed as "Saturday equivalents" = level of a weekday holiday relative to a Saturday.
C. v15 / v18 OOF residuals on train target rows by target-date type and by holiday-in-D1-D3.
D. exposure: test target rows on holidays and test pairs whose D1-D3 contain a holiday, by segment.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, OUTAGE, base_title, fig_dir, load, release_dates, style
from features import CUTI_BERSAMA, DOW_PROF, HOL_LEVEL, calendar

DOWN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
XMAS = ("2025-12-20", "2026-01-04")
BAD = set(pd.date_range("2025-06-01", "2025-06-16"))  # outage + 1-5 Jun anomaly (friend EDA 106)


def part_a(tr, cal):
    day = tr.groupby(KEY + ["date_show"]).agg(y=("total_ticket", "sum"), sh=("total_show", "sum"))
    hol_days = [pd.Timestamp(d) for d in cal.index[(cal.is_hol == 1) & (cal.dow < 5)]
                if pd.Timestamp("2025-04-08") <= d <= pd.Timestamp("2025-09-23") and d not in BAD]
    rows = []
    for d in hol_days:
        nb = [d - pd.Timedelta(days=7), d + pd.Timedelta(days=7)]
        if any(n in BAD or cal.is_hol.get(n, 0) == 1 for n in nb):
            continue
        g = [day.xs(x, level="date_show") for x in [d] + nb]
        j = g[0].join(g[1], rsuffix="_a", how="inner").join(g[2], rsuffix="_b", how="inner")
        j = j[(j.y_a >= 5) & (j.y_b >= 5)]
        if len(j) < 30:
            continue
        r = j.y / np.sqrt(j.y_a * j.y_b)
        rs = j.sh / np.sqrt(j.sh_a * j.sh_b)
        imp = cal.cal[d] / np.sqrt(cal.cal[nb[0]] * cal.cal[nb[1]])
        rows.append(dict(date=d.date(), dow=DOWN[d.dayofweek], pairs=len(j), ticket_ratio=np.exp(np.log(r).median()),
                         show_ratio=np.exp(np.log(rs).median()), calendar_implied=imp,
                         sat_equiv=np.exp(np.log(r).median()) * DOW_PROF[d.dayofweek] / DOW_PROF[5]))
    return pd.DataFrame(rows)


def film_days(tx, d1):
    x = tx.assign(base=base_title(tx.movie_title))
    b1 = pd.Series(d1.values, index=base_title(pd.Series(d1.index)).values).groupby(level=0).min()
    x = x.join(b1.rename("d1"), on="base").dropna(subset=["d1"])
    x["age"] = (x.date_show - x.d1).dt.days + 1
    return x[x.age.between(1, 3)].groupby(["base", "age", "date_show"]).total_ticket.sum().reset_index()


def part_b(D, cal):
    """Film-level excess: log(T_d / T_1) minus the median of non-holiday films with the same D1 weekday.
    For a film whose day d is a weekday holiday (and D1 is not), the excess estimates
    log(holiday level / same weekday); the calendar implies log(max(DOW, HOL) / DOW)."""
    tr = D["train"]
    A = film_days(tr[tr.date_show >= "2025-04-08"], release_dates(tr)).assign(period="train")
    h = D["hist"]
    hd1 = h.assign(base=base_title(h.movie_title)).groupby("base").date_show.min()
    B = film_days(h, pd.Series(hd1.values, index=hd1.index)).assign(period="test")
    F = pd.concat([A, B], ignore_index=True)
    W = F.pivot_table(index=["base", "period"], columns="age", values="total_ticket")
    dts = F.pivot_table(index=["base", "period"], columns="age", values="date_show", aggfunc="first")
    W = W[(W > 0).all(axis=1)]
    dts = dts.loc[W.index]
    if len(W) == 0:
        return pd.DataFrame(), W
    R = pd.DataFrame(index=W.index)
    R["d1"] = pd.to_datetime(dts[1])
    R["dow1"] = R.d1.dt.dayofweek
    for d in (1, 2, 3):
        dd = pd.to_datetime(dts[d])
        c = cal.reindex(dd)
        R[f"hol{d}"] = ((c.is_hol.values == 1) & (c.dow.values < 5)).astype(int)
        R[f"spec{d}"] = ((c.is_hol.values == 1) | (c.school.values == 1) | dd.isin(BAD)).astype(int)
        R[f"ram{d}"] = c.ramadan.values
        R[f"cal{d}"] = c.cal.values
        R[f"date{d}"] = dd.values
    for d in (2, 3):
        R[f"g{d}"] = np.log(W[d] / W[1]).values
    clean = (R[["spec1", "spec2", "spec3"]].sum(axis=1) == 0) & (R[["ram1", "ram2", "ram3"]].sum(axis=1) == 0)
    ref = R[clean].groupby("dow1")[["g2", "g3"]].median()
    nref = R[clean].groupby("dow1").size()
    rows = []
    for d in (2, 3):
        m = (R[f"hol{d}"] == 1) & (R.hol1 == 0) & (R.spec1 == 0) & R.dow1.isin(nref[nref >= 8].index)
        for idx, r in R[m].iterrows():
            dow_d = pd.Timestamp(r[f"date{d}"]).dayofweek
            ex = r[f"g{d}"] - ref.loc[r.dow1, f"g{d}"]
            rows.append(dict(film=idx[0][:32], period=idx[1], day=d, date=pd.Timestamp(r[f"date{d}"]).date(),
                             dow=DOWN[dow_d], ramadan=int(r[f"ram{d}"]), actual_mult=np.exp(ex),
                             calendar_mult=max(DOW_PROF[dow_d], HOL_LEVEL) / DOW_PROF[dow_d] if r[f"ram{d}"] == 0 else 1.0))
    return pd.DataFrame(rows), R


def part_c(cal):
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    res = {}
    for v in ["v15", "v18"]:
        o = X[KEY + ["h"]].merge(pd.read_csv(ROOT / f"results/{v}/oof.csv"), on=KEY + ["h"], how="left")
        res[v] = o.oof_final.values
    c = cal.reindex(X.date_show)
    typ = np.select([(c.is_hol.values == 1) & (c.dow.values < 5), c.dow.values >= 5], ["weekday holiday", "weekend"], "normal weekday")
    df = pd.DataFrame({"type": typ, "hol_in_hist": X.hol_in_hist.values > 0, "y": X.total_ticket.values, "s": X.scale.values})
    rows = []
    for v, p in res.items():
        df["se"] = (df.y - p) / df.s
        df["lr"] = np.log((df.y + 1) / (p + 1))
        pos = df.y > 0
        for (k, g) in df.groupby(["type", "hol_in_hist"]):
            gp = g[pos.loc[g.index]]
            rows.append(dict(model=v, target_day=k[0], hol_in_D1_D3=k[1], rows=len(g), mean_signed_scaled_err=g.se.mean(),
                             median_log_ratio_positive=gp.lr.median(), mase=g.se.abs().mean()))
    return pd.DataFrame(rows)


def part_d(cal):
    T = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    c = cal.reindex(T.date_show)
    seg = np.select([T.date_show.between("2026-03-21", "2026-03-27"), T.date_show.between(*XMAS),
                     T.date_show.between("2026-02-19", "2026-03-20")], ["lebaran", "xmas", "ramadan"], "normal")
    df = pd.DataFrame({"seg": seg, "target_weekday_hol": (c.is_hol.values == 1) & (c.dow.values < 5),
                       "hol_in_D1_D3": T.hol_in_hist.values > 0})
    out = df.groupby("seg").agg(rows=("seg", "size"), target_weekday_hol=("target_weekday_hol", "mean"),
                                hol_in_D1_D3=("hol_in_D1_D3", "mean"))
    out.loc["all"] = [len(df), df.target_weekday_hol.mean(), df.hol_in_D1_D3.mean()]
    hd = sorted(set(T.date_show[(c.is_hol.values == 1) & (c.dow.values < 5)].dt.date))
    return out, hd


def main():
    out = fig_dir("120_holiday_level")
    D = load()
    cal = calendar(D["hol"])
    pd.set_option("display.width", 220)
    print(f"calendar: weekday holiday -> max(DOW, {HOL_LEVEL}); DOW_PROF {DOW_PROF.tolist()}; cuti dates {len(CUTI_BERSAMA)}")

    A = part_a(D["train"], cal)
    print("\nA. train pair-level: ticket ratio on weekday holiday vs same pair +-7 days (geometric mean)")
    print(A.round(3).to_string(index=False))
    print(f"   median ticket ratio {A.ticket_ratio.median():.2f} vs calendar-implied {A.calendar_implied.median():.2f}; "
          f"median Saturday-equivalent {A.sat_equiv.median():.2f} (calendar assumes 1.00)")
    A.to_csv(out / "a_train_pair_holidays.csv", index=False)

    B, R = part_b(D, cal)
    print("\nB. film-level excess of log(T_d/T_1) over non-holiday films with the same D1 weekday, day d = weekday holiday:")
    print(B.round(3).to_string(index=False))
    B["grp"] = np.select([B.ramadan.eq(1), B.period.eq("train"), B.date.astype(str).between("2025-12-20", "2026-01-04")],
                         ["test Ramadan", "train", "test Christmas-New Year"], "test other")
    G = B.groupby("grp").apply(lambda g: pd.Series(dict(films=len(g), actual_mult=np.exp(np.log(g.actual_mult).median()),
                                                       calendar_mult=g.calendar_mult.median())), include_groups=False)
    G["actual_over_calendar"] = G.actual_mult / G.calendar_mult
    print(G.round(3).to_string())
    B.to_csv(out / "b_film_holiday_excess.csv", index=False); G.to_csv(out / "b_summary.csv")

    C = part_c(cal)
    print("\nC. OOF residual on train target rows (positive = model under-predicts):")
    print(C.round(4).to_string(index=False))
    C.to_csv(out / "c_oof_residual_by_daytype.csv", index=False)

    Dx, hd = part_d(cal)
    print("\nD. test exposure:"); print(Dx.round(4).to_string())
    print("   test target weekday holidays:", hd)
    Dx.to_csv(out / "d_test_exposure.csv")

    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    lab = A.date.astype(str) + " " + A.dow
    xi = np.arange(len(A))
    ax[0].bar(xi - .2, A.ticket_ratio, .4, label="actual (pairs)")
    ax[0].bar(xi + .2, A.calendar_implied, .4, label="calendar c(t)")
    ax[0].set_xticks(xi, lab, rotation=45, ha="right"); ax[0].axhline(1, color="k", lw=.7)
    ax[0].set(title="A. Train: weekday holiday vs same pair +-7 days"); ax[0].legend()
    for i, (k, g) in enumerate(B.groupby("grp")):
        ax[1].scatter(np.full(len(g), i) + np.random.default_rng(0).uniform(-.15, .15, len(g)), g.actual_mult, s=14, alpha=.7)
        ax[1].scatter([i], [g.calendar_mult.median()], marker="_", s=600, color="k")
    ax[1].set_xticks(range(B.grp.nunique()), sorted(B.grp.unique())); ax[1].set_yscale("log")
    ax[1].set(title="B. Film-level holiday multiplier (dots) vs calendar (bar)")
    cc = C[C.model == "v18"].assign(k=lambda d: d.target_day + "\nD1-3 hol=" + d.hol_in_D1_D3.astype(str))
    ax[2].barh(cc.k, cc.mean_signed_scaled_err, color=["#e34948" if v > 0 else "#2a78d6" for v in cc.mean_signed_scaled_err])
    ax[2].axvline(0, color="k", lw=.7); ax[2].set(title="C. v18 OOF mean (y - pred)/scale by day type")
    fig.tight_layout(); fig.savefig(out / "holiday_level.png")


if __name__ == "__main__":
    main()
