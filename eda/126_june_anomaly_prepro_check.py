"""Preprocessing check: extend the outage exclusion from 5 June dates to the whole 1-16 June 2025 block.

Friend EDA 106: total shows fell 6,272 -> 2,044 from 31 May to 1 Jun 2025 with 112/114 clusters at <= half
their shows, while weekend controls were flat; 1-5 June are not in OUTAGE, so windows of films released
28 May keep target rows on depressed dates. Here:
 A. daily network shows/tickets 20 May - 25 Jun with the current OUTAGE marked (does the anomaly exist?).
 B. simulate() with the current OUTAGE vs with all of 1-16 June: which windows / target rows are removed,
    and the per-horizon zero rate + mean y/scale of the removed rows vs the same horizons elsewhere
    (expected: removed rows are abnormally low = false decay labels).
 C. after cleaning: no remaining target row inside 1-16 June, row counts by month.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, OUTAGE, fig_dir, load, release_dates, scale, simulate, style

BLOCK = pd.date_range("2025-06-01", "2025-06-16")


def main():
    out = fig_dir("126_june_anomaly")
    D = load()
    tr = D["train"]
    day = tr.groupby("date_show").agg(shows=("total_show", "sum"), tickets=("total_ticket", "sum"), clusters=("cinema_ids", "nunique"))
    win = day.loc["2025-05-20":"2025-06-25"]
    print("A. network per day (current OUTAGE marked *):")
    print(win.assign(outage=win.index.isin(OUTAGE).astype(int)).to_string())

    d1 = release_dates(tr)
    h0, t0 = simulate(tr, d1)
    h1, t1 = simulate(tr, d1, bad=BLOCK)
    k0 = t0.assign(d1=t0.movie_title.map(d1))
    k0["h"] = (k0.date_show - k0.d1).dt.days + 1
    k0 = k0.join(scale(h0), on=KEY)
    gone = k0.merge(t1[KEY + ["date_show"]], on=KEY + ["date_show"], how="left", indicator=True)
    gone = gone[gone._merge == "left_only"]
    print(f"\nB. current: {len(t0)} target rows / {t0.movie_title.nunique()} titles; block 1-16 Jun: {len(t1)} / {t1.movie_title.nunique()}")
    print("   removed rows by title:", gone.groupby("movie_title").size().to_dict())
    print("   removed rows by date :", gone.groupby(gone.date_show.dt.date).size().to_dict())
    keep = k0[~k0.set_index(KEY + ["date_show"]).index.isin(gone.set_index(KEY + ["date_show"]).index)]
    cmp = pd.DataFrame({
        "removed_zero": gone.groupby("h").total_ticket.apply(lambda v: (v == 0).mean()),
        "kept_zero": keep.groupby("h").total_ticket.apply(lambda v: (v == 0).mean()),
        "removed_r": gone.assign(r=gone.total_ticket / gone.scale).groupby("h").r.median(),
        "kept_r": keep.assign(r=keep.total_ticket / keep.scale).groupby("h").r.median(),
        "removed_n": gone.groupby("h").size()})
    print("   zero rate and median y/scale by horizon, removed vs kept:"); print(cmp.round(3).to_string())
    assert not t1.date_show.isin(BLOCK).any(), "no target row may remain inside 1-16 June"
    print(f"\nC. target rows inside 1-16 Jun after cleaning: {int(t1.date_show.isin(BLOCK).sum())}; rows by month:",
          t1.groupby(t1.date_show.dt.to_period('M')).size().to_dict())
    cmp.to_csv(out / "removed_vs_kept.csv")
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    ax[0].plot(win.index, win.shows, marker="o", label="shows")
    ax[0].scatter(BLOCK.intersection(win.index), win.shows.reindex(BLOCK).dropna(), color="#e34948", zorder=3, label="1-16 Jun block")
    ax[0].set(title="Network shows per day"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=30)
    ax[1].plot(cmp.index, cmp.removed_r, marker="o", label="removed rows"); ax[1].plot(cmp.index, cmp.kept_r, marker="o", label="kept rows")
    ax[1].set(title="Median y/scale by horizon", xlabel="h"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "june_anomaly.png")


if __name__ == "__main__":
    main()
