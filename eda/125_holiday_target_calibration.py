"""Post-hoc correction for target rows on weekday holidays (outside Lebaran), validated leave-one-holiday-out.

eda/120: on train target rows that fall on a weekday holiday, v18 OOF MASE is 0.52 vs 0.29 on normal
weekdays and the mean signed error is larger (+0.22 vs +0.13): the calendar's "holiday = Saturday" level is
too low for running films. eda/121-122: the surge is larger for low-occupancy pairs (slope ~ -0.5).
Correction on the final prediction of a holiday target row:  y_hat * exp(a + b * (log occ_D1-D3 - c)),
c = median log occupancy. (a, b) are chosen on the OTHER holiday dates (leave one date out) by minimising
TW-weighted MAE / scale, from a grid, so the score below is out-of-sample.
Test exposure: weekday-holiday target rows outside Lebaran (Christmas 25-26 Dec, 1 Jan, 16 Jan, 16-17 Feb,
18-20 Mar inside Ramadan - the last group is reported separately since train has no Ramadan holiday).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, fig_dir, load, style
from evaluate import test_weights_fd
from features import calendar

A_GRID = np.round(np.arange(-0.1, 0.51, 0.05), 2)
B_GRID = np.round(np.arange(-0.6, 0.01, 0.1), 2)


def main():
    out = fig_dir("125_holiday_target_calibration")
    D = load()
    cal = calendar(D["hol"])
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    Te = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(X, Te)
    o = X[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v18/oof.csv"), on=KEY + ["h"], how="left")
    p, y, s = o.oof_final.values, X.total_ticket.values, X.scale.values
    c = cal.reindex(X.date_show)
    hol = (c.is_hol.values == 1) & (c.dow.values < 5) & (c.ramadan.values == 0)
    lo = np.log(X.occ_mean.clip(lower=0.5).values)
    cmed = np.median(lo[hol])
    dates = X.date_show[hol].dt.date.values
    idx = np.flatnonzero(hol)
    ud = sorted(set(dates))
    err = lambda q, m: np.average(np.abs(y[m] - q[m]) / s[m], weights=w[m])  # noqa: E731
    adj = p.copy()
    picks = []
    for dte in ud:
        val = idx[dates == dte]
        trn = idx[dates != dte]
        mt = np.zeros(len(p), bool); mt[trn] = True
        best = min(((a, b) for a in A_GRID for b in B_GRID),
                   key=lambda ab: err(p * np.exp(ab[0] + ab[1] * (lo - cmed)), mt))
        picks.append(dict(date=dte, rows=len(val), a=best[0], b=best[1]))
        adj[val] = p[val] * np.exp(best[0] + best[1] * (lo[val] - cmed))
    Pk = pd.DataFrame(picks)
    mh = np.zeros(len(p), bool); mh[idx] = True
    allm = np.ones(len(p), bool)
    print(f"train weekday-holiday target rows (non-Ramadan): {hol.sum()} on {len(ud)} dates; median occupancy {np.exp(cmed):.1f}%")
    print(Pk.to_string(index=False))
    rows = []
    for dte in ud:
        m = np.zeros(len(p), bool); m[idx[dates == dte]] = True
        rows.append(dict(date=dte, before=err(p, m), after=err(adj, m)))
    R = pd.DataFrame(rows)
    print(R.round(4).to_string(index=False))
    print(f"\nholiday rows TW: {err(p, mh):.4f} -> {err(adj, mh):.4f} | all rows TW: {err(p, allm):.4f} -> {err(adj, allm):.4f}"
          f" | dates improved {int((R.after < R.before).sum())}/{len(R)}")
    full = min(((a, b) for a in A_GRID for b in B_GRID), key=lambda ab: err(p * np.exp(ab[0] + ab[1] * (lo - cmed)), mh))
    print(f"full-sample choice a={full[0]}, b={full[1]} (multiplier at median occupancy {np.exp(full[0]):.2f})")
    ct = cal.reindex(Te.date_show)
    th = (ct.is_hol.values == 1) & (ct.dow.values < 5)
    leb = Te.date_show.between("2026-03-21", "2026-03-27").values
    ram = ct.ramadan.values == 1
    print(f"test exposure: weekday-holiday target rows outside Lebaran and Ramadan {int((th & ~leb & ~ram).sum())} "
          f"({np.mean(th & ~leb & ~ram):.2%}); inside Ramadan {int((th & ram).sum())}")
    pd.DataFrame({"a": [full[0]], "b": [full[1]], "c_log_occ": [cmed]}).to_csv(out / "chosen.csv", index=False)
    R.to_csv(out / "per_date.csv", index=False)
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xi = np.arange(len(R))
    ax[0].bar(xi - .2, R.before, .4, label="v18 OOF"); ax[0].bar(xi + .2, R.after, .4, label="calibrated (LODO)")
    ax[0].set_xticks(xi, [str(d) for d in R.date], rotation=30); ax[0].set(title="TW-MASE on weekday-holiday target rows"); ax[0].legend()
    q = pd.qcut(lo[hol], 5, labels=False)
    for k, (arr, lab) in enumerate([(p, "v18"), (adj, "calibrated")]):
        se = pd.Series((y[hol] - arr[hol]) / s[hol]).groupby(q).mean()
        ax[1].plot(range(5), se.values, marker="o", label=lab)
    ax[1].axhline(0, color="k", lw=.7); ax[1].set(title="Mean (y - pred)/scale by occupancy quintile", xlabel="quintile"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "holiday_calibration.png")


if __name__ == "__main__":
    main()
