"""
Report figures, generated from reports/ so they regenerate rather than being
hand-made.

Every figure reads its numbers from a file in reports/ and writes both the image
(300dpi PNG plus a vector PDF) and the data behind it to figures/data/, so any
figure in the report can be traced to a CSV. One figure (07) is the exception
and says so on its face: it compares runs from early exploration that predate
the current scripts, so its values are constants below with their source noted.

Style notes, for anyone changing this: light mode only, because these go into a
printed document. Every mark carries a direct value label — partly because a
print figure has no tooltip, and partly because one of the three series colours
sits below 3:1 contrast on the page, which obliges visible labels. No chart here
has two y-axes; where two measures of different scale needed showing, the figure
is faceted instead.

Run:
    python make_figures.py                 # all figures
    python make_figures.py --only 1 5 8    # just those
    python make_figures.py --outdir figs   # somewhere else
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config import END_DATE_STR, REPORT_DIR, START_DATE_STR, STOCKS

warnings.filterwarnings("ignore", category=FutureWarning)

# --- palette -----------------------------------------------------------------
# Three categorical slots, validated all-pairs for colour-vision deficiency.
# Do not add a fourth: the next slot puts yellow beside orange, which fails the
# separation floor. If a figure needs more than three series, facet it instead.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"               # diverging partner for BLUE, negative pole
BLUE_LIGHT = "#86b6ef"        # sequential step, for an ordinal pair with BLUE
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#c3c2b7"
SURFACE = "#fcfcfb"

PLATFORM = {"news": BLUE, "reddit": ORANGE}
ORDER = ["AAPL", "GOOG", "NVDA", "TTWO", "BLK", "JPM"]
CATEGORY = {s.symbol: s.category for s in STOCKS}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK,
    "legend.frameon": False,
    "axes.grid": False,
})

FIG_DIR = "figures"


# --- plumbing ----------------------------------------------------------------

def rp(name):
    return os.path.join(REPORT_DIR, name)


def read(name, **kw):
    """Read a report file, or return None so one missing file skips one figure."""
    path = rp(name)
    if not os.path.exists(path):
        print(f"  ! {path} not found")
        return None
    return pd.read_csv(path, **kw)


def finish(fig, num, slug, outdir, data=None):
    """Save PNG + PDF, and the figure's own numbers beside them."""
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"fig{num:02d}_{slug}")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    if data is not None:
        ddir = os.path.join(outdir, "data")
        os.makedirs(ddir, exist_ok=True)
        data.to_csv(os.path.join(ddir, f"fig{num:02d}_{slug}.csv"), index=False)
    print(f"  -> {stem}.png / .pdf")


def hgrid(ax, axis="x"):
    """Recessive gridlines, behind the marks, on one axis only."""
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def arm_accuracies(fam, target="open_to_close"):
    """
    Per-arm accuracy, recovered from the pairwise comparison rows.

    compare() writes acc_a/acc_b alongside model_a/model_b, so every arm appears
    in at least one row and the mapping can be rebuilt without a separate file.
    """
    df = read(f"tuned_results_{fam}_{target}.csv")
    if df is None or "model_a" not in df.columns:
        return None, None
    acc = {}
    for _, r in df.iterrows():
        acc[r["model_a"]] = float(r["acc_a"])
        acc[r["model_b"]] = float(r["acc_b"])
    return acc, df


def majority_baseline(fam, target="open_to_close"):
    """
    Baseline from the arm's own predictions, not reused across families.

    The two families pool a different number of sessions (3,336 against 3,294),
    so one baseline figure would be wrong for one of them.
    """
    df = read(f"tuned_preds_{fam}_{target}_price_only.csv")
    if df is None or "y_true" not in df.columns:
        return None
    m = float(df["y_true"].mean())
    return max(m, 1 - m) * 100


def corr_table(source):
    df = read(f"correlation_analysis_{source}.csv")
    if df is None:
        return None
    df["_o"] = df["ticker"].map({t: i for i, t in enumerate(ORDER)})
    return df.sort_values("_o").drop(columns="_o")


def panel(source):
    name = ("daily_panel_{start}_{end}.csv" if source == "news"
            else "daily_panel_reddit_{start}_{end}.csv")
    return read(name.format(start=START_DATE_STR, end=END_DATE_STR),
                parse_dates=["date"])


# --- 01 ----------------------------------------------------------------------

def fig01(outdir):
    """Forest plot: every arm against its dimensionality-matched control."""
    want = {
        "price_plus_news vs price_plus_noise": "News vs matched noise",
        "price_plus_reddit vs price_plus_noise": "Reddit vs matched noise",
        "price_plus_both vs price_plus_noise10": "Combined vs matched noise",
    }
    rows = []
    for fam, label in (("gbm", "Gradient boosting"), ("lstm", "LSTM")):
        df = read(f"tuned_results_{fam}_open_to_close.csv")
        if df is None:
            continue
        for _, r in df.iterrows():
            if r["comparison"] in want:
                rows.append({"family": label, "label": want[r["comparison"]],
                             "diff": r["diff_pp"], "lo": r["ci_low_pp"],
                             "hi": r["ci_high_pp"], "p": r["p"]})
    if not rows:
        return
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    hgrid(ax)
    fams = ["Gradient boosting", "LSTM"]
    labels = list(want.values())
    # Value labels sit in a fixed column to the right of the data, not at the
    # end of each interval: anchoring them to the interval end puts a short
    # interval's label on top of a neighbouring one.
    LABEL_X = 3.3
    ticks, ticklabels = [], []
    for i, lab in enumerate(labels):
        for j, fam in enumerate(fams):
            sub = d[(d.label == lab) & (d.family == fam)]
            if sub.empty:
                continue
            y = -(i * 2.0 + j * 0.75)
            r = sub.iloc[0]
            c = BLUE if fam == fams[0] else ORANGE
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=2, solid_capstyle="round",
                    zorder=3)
            ax.plot([r["diff"]], [y], "o", color=c, ms=8, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            txt = f"{r['diff']:+.2f}pp".replace("-", "−")
            ax.annotate(f"{txt}   p = {r['p']:.3f}", (LABEL_X, y),
                        va="center", ha="left", fontsize=7.5, color=INK2)
            ticks.append(y)
            ticklabels.append(fam)
        ax.annotate(lab, (-6.4, -(i * 2.0 - 0.55)), ha="left", va="center",
                    fontsize=8.5, color=INK, fontweight="bold")

    ax.axvline(0, color=INK, lw=1, zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=7.5, color=INK2)
    ax.set_xlabel("Difference in accuracy vs the matched noise control "
                  "(pp) — negative favours the control")
    ax.set_xlim(-6.5, 8.6)
    ax.set_xticks([-6, -4, -2, 0, 2])
    ax.set_ylim(-5.0, 1.1)
    ax.set_title("No arm beats its dimensionality-matched control",
                 loc="left", color=INK, pad=22)
    finish(fig, 1, "pre_registered_forest", outdir, d)


# --- 02 ----------------------------------------------------------------------

def fig02(outdir):
    """Arm accuracies against each family's own majority-class baseline."""
    arms = ["price_only", "price_plus_news", "price_plus_reddit",
            "price_plus_noise", "price_plus_both", "price_plus_noise10"]
    pretty = {"price_only": "Price only", "price_plus_news": "+ News",
              "price_plus_reddit": "+ Reddit", "price_plus_noise": "+ Noise (control)",
              "price_plus_both": "+ Both", "price_plus_noise10": "+ Noise ×10 (control)"}
    fams = [("gbm", "Gradient boosting", BLUE), ("lstm", "LSTM", ORANGE)]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.3), sharey=True)
    out = []
    for ax, (fam, label, colour) in zip(axes, fams):
        acc, _ = arm_accuracies(fam)
        base = majority_baseline(fam)
        if acc is None:
            ax.set_visible(False)
            continue
        # An arm missing from the results file would otherwise be drawn as a
        # silent gap, which reads as "this arm scored zero" rather than "this
        # arm was not found". Say so instead.
        missing = [a for a in arms if a not in acc]
        if missing:
            print(f"  ! {fam}: no accuracy recovered for {', '.join(missing)}")
        hgrid(ax)
        y = np.arange(len(arms))[::-1]
        for yi, a in zip(y, arms):
            if a not in acc:
                ax.annotate("not in results file", (46.2, yi), va="center",
                            fontsize=7, color=MUTED, style="italic")
                continue
            v = acc[a] * 100
            control = "noise" in a
            ax.barh(yi, v, height=0.62,
                    color=NEUTRAL if control else colour,
                    hatch="///" if control else None,
                    edgecolor=SURFACE, linewidth=1.2, zorder=3)
            # Label inside the bar: outside, it collides with the baseline rule.
            ax.annotate(f"{v:.2f}%", (v, yi), xytext=(-5, 0),
                        textcoords="offset points", va="center", ha="right",
                        fontsize=7.5, color=INK if control else "#ffffff",
                        zorder=5)
            out.append({"family": label, "arm": a, "accuracy_pct": v,
                        "baseline_pct": base})
        if base:
            ax.axvline(base, color=INK, lw=1.1, ls=(0, (4, 2)), zorder=4)
            ax.annotate(f"majority baseline  {base:.2f}%", (base, len(arms) - 0.3),
                        xytext=(-4, 0), textcoords="offset points",
                        fontsize=7, color=INK, ha="right", va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([pretty[a] for a in arms], fontsize=8, color=INK)
        ax.set_xlim(46, 56)
        ax.set_ylim(-0.7, len(arms) - 0.1)
        ax.set_title(label, loc="left", color=INK, fontsize=9.5, pad=14)
        ax.set_xlabel("Out-of-sample accuracy (%)")

    fig.legend(handles=[Patch(facecolor=NEUTRAL, hatch="///", edgecolor=SURFACE,
                              label="Random-feature control")],
               loc="lower center", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Neither family clears its own baseline", x=0.02, ha="left",
                 color=INK, fontsize=10.5, y=1.04)
    finish(fig, 2, "arm_accuracies", outdir, pd.DataFrame(out))


# --- 03 ----------------------------------------------------------------------

def fig03(outdir):
    """Same-day against next-day correlation, both platforms, shared scale."""
    tabs = {s: corr_table(s) for s in ("news", "reddit")}
    if all(v is None for v in tabs.values()):
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), sharey=True)
    out = []
    # Significance markers only, not a number on every bar: twenty-four rotated
    # labels is clutter, and the exact values are in the figure's data file and
    # in the report's correlation table.
    for ax, (col, title) in zip(axes, [("same_day", "Same session"),
                                       ("next_day", "Next session")]):
        hgrid(ax, axis="y")
        x = np.arange(len(ORDER))
        for k, (src, c) in enumerate(PLATFORM.items()):
            t = tabs[src]
            if t is None:
                continue
            t = t.set_index("ticker")
            for tk, xi in zip(ORDER, x + (k - 0.5) * 0.38):
                v, p = t[col].get(tk, np.nan), t[f"{col}_p"].get(tk, np.nan)
                if not np.isfinite(v):
                    continue
                ax.bar(xi, v, width=0.36, color=c, edgecolor=SURFACE,
                       linewidth=1.2, zorder=3,
                       label=src.capitalize()
                       if (ax is axes[0] and tk == ORDER[0]) else None)
                s = stars(p)
                if s:
                    ax.annotate(s, (xi, v), xytext=(0, 2 if v >= 0 else -9),
                                textcoords="offset points", ha="center",
                                fontsize=8, color=INK2)
                out.append({"window": title, "platform": src, "ticker": tk,
                            "r": v, "p": p})
        ax.axhline(0, color=BASELINE, lw=0.9, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, fontsize=8, color=INK)
        ax.set_title(title, loc="left", color=INK, fontsize=9.5, pad=8)
        ax.set_ylim(-0.14, 0.31)
    axes[0].set_ylabel("Correlation with return")
    axes[0].legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
    fig.suptitle("Sentiment tracks the current session, not the next one",
                 x=0.02, ha="left", color=INK, fontsize=10.5, y=1.04)
    fig.text(0.02, -0.07, "*** p<0.001   ** p<0.01   * p<0.05", fontsize=7,
             color=MUTED, ha="left")
    finish(fig, 3, "sameday_vs_nextday", outdir, pd.DataFrame(out))


# --- 04 ----------------------------------------------------------------------

def fig04(outdir):
    """Coverage dumbbell: news against Reddit, per ticker."""
    n = read("daily_coverage.csv")
    r = read("daily_coverage_reddit.csv")
    if n is None or r is None:
        return
    n, r = n.set_index("ticker"), r.set_index("ticker")

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    hgrid(ax)
    out = []
    for i, tk in enumerate(ORDER):
        y = len(ORDER) - i - 1
        a, b = n["pct_covered"].get(tk), r["pct_covered"].get(tk)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        ax.plot([a, b], [y, y], color=BASELINE, lw=2, zorder=2,
                solid_capstyle="round")
        ax.plot([a], [y], "o", color=BLUE, ms=9, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.plot([b], [y], "o", color=ORANGE, ms=9, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        lo, hi = (a, b) if a < b else (b, a)
        ax.annotate(f"{lo:.1f}%", (lo, y), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.5, color=INK2)
        ax.annotate(f"{hi:.1f}%", (hi, y), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.5, color=INK2)
        out.append({"ticker": tk, "news_pct": a, "reddit_pct": b})
    ax.set_yticks(range(len(ORDER))[::-1])
    ax.set_yticklabels(ORDER, fontsize=8.5, color=INK)
    ax.set_xlabel("Share of the 1,258 trading sessions with coverage (%)")
    ax.set_xlim(10, 108)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=8, lw=0,
                               label="Financial news"),
                        Line2D([], [], color=ORANGE, marker="o", ms=8, lw=0,
                               label="Reddit")],
               loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    # Deliberately not "Reddit covers more": it does so for only two of the six
    # tickers. The finding is the spread, and AAPL in particular.
    ax.set_title("Coverage differs sharply by platform and by stock",
                 loc="left", color=INK, pad=8)
    finish(fig, 4, "coverage_dumbbell", outdir, pd.DataFrame(out))


# --- 05 ----------------------------------------------------------------------

def fig05(outdir):
    """Volume against next-session volatility, before and after the control."""
    tabs = {s: corr_table(s) for s in ("news", "reddit")}
    if all(v is None for v in tabs.values()):
        return

    # A before/after pair reads better as movement than as two bars: the
    # question is how far each correlation falls once clustering is controlled,
    # and a dumbbell shows that as a distance. Raw is an open marker and
    # controlled a filled one, so the two are distinguishable by shape as well
    # as position.
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), sharey=True)
    out = []
    for ax, src in zip(axes, ("news", "reddit")):
        t = tabs[src]
        if t is None:
            ax.set_visible(False)
            continue
        hgrid(ax)
        t = t.set_index("ticker")
        colour = PLATFORM[src]
        for i, tk in enumerate(ORDER):
            y = len(ORDER) - i - 1
            raw, ctl = t["volume_vol"].get(tk), t["volume_vol_pc"].get(tk)
            p_ctl = t["volume_vol_pc_p"].get(tk, np.nan)
            if not (np.isfinite(raw) and np.isfinite(ctl)):
                continue
            ax.plot([ctl, raw], [y, y], color=BASELINE, lw=1.8, zorder=2,
                    solid_capstyle="round")
            ax.plot([raw], [y], "o", ms=8, color=SURFACE, zorder=3,
                    markeredgecolor=colour, markeredgewidth=1.8)
            ax.plot([ctl], [y], "o", ms=8, color=colour, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.annotate(f"{raw:+.3f}", (raw, y), xytext=(7, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=7, color=MUTED)
            ax.annotate(f"{ctl:+.3f}{stars(p_ctl)}", (ctl, y), xytext=(-7, 0),
                        textcoords="offset points", va="center", ha="right",
                        fontsize=7, color=INK2)
            out.append({"platform": src, "ticker": tk, "raw": raw,
                        "controlled": ctl, "controlled_p": p_ctl})
        ax.axvline(0, color=BASELINE, lw=0.9, zorder=1)
        ax.set_yticks(range(len(ORDER))[::-1])
        ax.set_yticklabels(ORDER, fontsize=8, color=INK)
        ax.set_xlim(-0.09, 0.235)
        ax.set_xticks([0.0, 0.1, 0.2])
        ax.set_title("Financial news" if src == "news" else "Reddit",
                     loc="left", color=INK, fontsize=9.5, pad=8)
        ax.set_xlabel("Correlation with next-session |return|")
    fig.legend(handles=[
        Line2D([], [], marker="o", lw=0, ms=7, color=SURFACE,
               markeredgecolor=INK2, markeredgewidth=1.6, label="Raw"),
        Line2D([], [], marker="o", lw=0, ms=7, color=INK2,
               label="After volatility control")],
        loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    fig.suptitle("Most of the volume–volatility link is volatility clustering",
                 x=0.02, ha="left", color=INK, fontsize=10.5, y=1.04)
    fig.text(0.02, -0.21,
             "Controls are today's absolute return and trailing 5-day volatility.   "
             "*** p<0.001   ** p<0.01   * p<0.05",
             fontsize=7, color=MUTED, ha="left")
    finish(fig, 5, "volume_volatility", outdir, pd.DataFrame(out))


# --- 06 ----------------------------------------------------------------------

def fig06(outdir):
    """Sentiment distribution as 100% stacked bars, one panel per platform."""
    tabs = {}
    for s in ("news", "reddit"):
        d = read(f"sentiment_summary_{s}.csv")
        if d is not None:
            tabs[s] = d.set_index("ticker")
    if not tabs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), sharey=True)
    out = []
    for ax, src in zip(axes, ("news", "reddit")):
        if src not in tabs:
            ax.set_visible(False)
            continue
        hgrid(ax)
        t = tabs[src]
        y = np.arange(len(ORDER))[::-1]
        for yi, tk in zip(y, ORDER):
            neg = float(t["pct_negative"].get(tk, np.nan))
            pos = float(t["pct_positive"].get(tk, np.nan))
            if not (np.isfinite(neg) and np.isfinite(pos)):
                continue
            neu = 100 - neg - pos
            left = 0
            for v, c, name in [(neg, RED, "negative"), (neu, NEUTRAL, "neutral"),
                               (pos, BLUE, "positive")]:
                ax.barh(yi, v, left=left, height=0.62, color=c,
                        edgecolor=SURFACE, linewidth=1.2, zorder=3)
                if v > 9:
                    ax.annotate(f"{v:.0f}", (left + v / 2, yi), ha="center",
                                va="center", fontsize=7,
                                color="#ffffff" if c != NEUTRAL else INK)
                left += v
                out.append({"platform": src, "ticker": tk, "class": name,
                            "pct": v})
        ax.set_yticks(y)
        ax.set_yticklabels(ORDER, fontsize=8.5, color=INK)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of documents (%)")
        ax.set_title("Financial news" if src == "news" else "Reddit",
                     loc="left", color=INK, fontsize=9.5)
    fig.legend(handles=[Patch(facecolor=RED, label="Negative"),
                        Patch(facecolor=NEUTRAL, label="Neutral (|score| ≤ 0.1)"),
                        Patch(facecolor=BLUE, label="Positive")],
               loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.suptitle("FinBERT is far less decisive on Reddit than on news",
                 x=0.02, ha="left", color=INK, fontsize=10.5, y=1.02)
    finish(fig, 6, "sentiment_distribution", outdir, pd.DataFrame(out))


# --- 07 ----------------------------------------------------------------------
# The only figure not read from reports/. These four numbers come from early
# exploration runs that predate model_tuned.py, and are recorded in
# docs/METHODOLOGY.md §6 ("The dimensionality control"). They are constants here
# rather than silently recomputed, because the scripts that produced them no
# longer exist in this form — and the figure's whole point is historical.
JUDGED = [
    # (dataset, comparator, difference pp, p, verdict)
    ("Real data", "vs price-only", -2.99, 0.003, "“sentiment hurts”"),
    ("Real data", "vs matched noise", +0.93, 0.300, "no effect"),
    ("Synthetic random data", "vs price-only", +3.42, 0.034, "“sentiment helps”"),
    ("Synthetic random data", "vs matched noise", +2.88, 0.130, "no effect"),
]


def fig07(outdir):
    """The comparator decides the conclusion — in both directions."""
    d = pd.DataFrame(JUDGED, columns=["dataset", "comparator", "diff_pp", "p",
                                      "verdict"])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    # As in figure 1, value labels sit in a fixed column rather than at the bar
    # end — a negative bar's label would otherwise run back into the tick label.
    LABEL_X = 4.4
    for ax, ds in zip(axes, ["Real data", "Synthetic random data"]):
        hgrid(ax)
        sub = d[d.dataset == ds].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        for yi, (_, r) in zip(y, sub.iterrows()):
            sig = r["p"] < 0.05
            ax.barh(yi, r["diff_pp"], height=0.44,
                    color=ORANGE if sig else NEUTRAL,
                    hatch=None if sig else "///",
                    edgecolor=SURFACE, linewidth=1.2, zorder=3)
            txt = f"{r['diff_pp']:+.2f}pp".replace("-", "−")
            ax.annotate(f"{txt}  p = {r['p']:.3f}", (LABEL_X, yi + 0.12),
                        ha="left", va="center", fontsize=7.5, color=INK2)
            ax.annotate(r["verdict"], (LABEL_X, yi - 0.17), ha="left",
                        va="center", fontsize=7, color=MUTED, style="italic")
        ax.axvline(0, color=INK, lw=1, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["comparator"], fontsize=8.5, color=INK)
        ax.set_xlim(-5.0, 10.4)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(ds, loc="left", color=INK, fontsize=9.5, pad=8)
        ax.set_xlabel("Difference in accuracy (pp)")
    fig.legend(handles=[Patch(facecolor=ORANGE, label="Significant at p < 0.05"),
                        Patch(facecolor=NEUTRAL, hatch="///", edgecolor=SURFACE,
                              label="Not significant")],
               loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.suptitle("The same arm, judged against two comparators",
                 x=0.02, ha="left", color=INK, fontsize=10.5, y=1.03)
    fig.text(0.02, -0.19,
             "Early-exploration runs recorded in docs/METHODOLOGY.md §6. On real data a price-only comparison "
             "calls sentiment harmful;\non synthetic noise it calls the same arm helpful. Only the matched-noise "
             "comparison is right both times.",
             fontsize=7, color=MUTED, ha="left")
    finish(fig, 7, "judged_two_ways", outdir, d)


# --- 08 ----------------------------------------------------------------------

def fig08(outdir):
    """
    Lag profile: where in time the sentiment-return association actually sits.

    Lag k correlates sentiment on session d with the return of session d+k, so
    negative lags ask whether sentiment follows a move it could not have caused.
    Computed per ticker and averaged, with the individual tickers shown, because
    a single pooled line would hide how much of the lag-0 spike is NVDA.
    """
    lags = [-2, -1, 0, 1, 2]
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    hgrid(ax, axis="y")
    out = []
    for src, colour in PLATFORM.items():
        p = panel(src)
        if p is None:
            continue
        per = {}
        for tk, g in p.groupby("ticker"):
            g = g.sort_values("date")
            price = "adj_close" if "adj_close" in g.columns else "close"
            ret = g[price].pct_change()
            s = g["sentiment_mean"]
            per[tk] = [s.corr(ret.shift(-k)) for k in lags]
        if not per:
            continue
        arr = np.array([per[t] for t in per])
        for row in arr:
            ax.plot(lags, row, color=colour, lw=0.8, alpha=0.3, zorder=2)
        mean = np.nanmean(arr, axis=0)
        ax.plot(lags, mean, color=colour, lw=2.2, marker="o", ms=8, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=src.capitalize())
        # Label each series' own maximum rather than assuming it falls at lag
        # zero: news peaks at lag 0, but Reddit peaks at lag -1, which is the
        # more interesting half of this figure.
        k = int(np.nanargmax(mean))
        ax.annotate(f"{mean[k]:+.3f}", (lags[k], mean[k]),
                    xytext=(10, 6 if src == "news" else -15),
                    textcoords="offset points", ha="left", fontsize=8,
                    color=colour, fontweight="bold")
        for tk, vals in per.items():
            for L, v in zip(lags, vals):
                out.append({"platform": src, "ticker": tk, "lag": L, "r": v})
    ax.axhline(0, color=BASELINE, lw=0.9, zorder=1)
    ax.axvline(0, color=GRID, lw=6, zorder=0)
    ax.set_xticks(lags)
    ax.set_xticklabels(["\u22122", "\u22121", "0", "+1", "+2"], fontsize=9,
                       color=INK)
    ax.annotate("same session", (0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -12), textcoords="offset points", ha="center",
                fontsize=7, color=MUTED)
    ax.set_xlabel("Lag between sentiment and return (sessions)")
    ax.set_ylabel("Correlation (mean across tickers)")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Neither platform's sentiment leads the market",
                 loc="left", color=INK, pad=8)
    fig.text(0.02, -0.06,
             "Faint lines are individual tickers. Negative lags correlate sentiment with earlier returns; "
             "positive lags with later ones.",
             fontsize=7, color=MUTED, ha="left")
    finish(fig, 8, "lag_profile", outdir, pd.DataFrame(out))


# --- 09 ----------------------------------------------------------------------

def fig09(outdir):
    """
    Material against immaterial stocks on each metric.

    The sub-question asks whether stock type moderates which platform works.
    This is the figure that shows it cannot be answered: the two categories'
    ranges overlap on every metric, so there is no separation for a moderator to
    act on. A null is still worth drawing.
    """
    metrics = [("same_day", "Same-session\nsentiment"),
               ("next_day", "Next-session\nsentiment"),
               ("volume_vol", "Volume vs volatility\n(raw)"),
               ("volume_vol_pc", "Volume vs volatility\n(controlled)")]
    tabs = {s: corr_table(s) for s in ("news", "reddit")}
    if all(v is None for v in tabs.values()):
        return

    fig, axes = plt.subplots(1, 4, figsize=(6.9, 3.0), sharey=True)
    out = []
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, metrics):
        hgrid(ax, axis="y")
        for xi, cat in enumerate(["material", "immaterial"]):
            vals = []
            for src, colour in PLATFORM.items():
                t = tabs[src]
                if t is None:
                    continue
                sub = t[t["ticker"].map(CATEGORY) == cat]
                v = sub[col].to_numpy(dtype=float)
                jitter = rng.uniform(-0.09, 0.09, len(v))
                ax.plot(np.full(len(v), xi) + jitter, v, "o", ms=6,
                        color=colour, alpha=0.85, zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.0,
                        label=src.capitalize() if (ax is axes[0] and xi == 0) else None)
                vals.extend(v[np.isfinite(v)])
                for tk, x in zip(sub["ticker"], v):
                    out.append({"metric": col, "category": cat, "platform": src,
                                "ticker": tk, "r": x})
            if vals:
                m = float(np.mean(vals))
                ax.plot([xi - 0.22, xi + 0.22], [m, m], color=INK, lw=1.6,
                        zorder=4)
        ax.axhline(0, color=BASELINE, lw=0.9, zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Material", "Immaterial"], fontsize=7.5, color=INK,
                           rotation=20, ha="right")
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title, loc="left", color=INK, fontsize=8)
    axes[0].set_ylabel("Correlation")
    fig.legend(handles=[Line2D([], [], marker="o", lw=0, ms=7, color=BLUE,
                               label="News"),
                        Line2D([], [], marker="o", lw=0, ms=7, color=ORANGE,
                               label="Reddit")],
               loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.suptitle("Stock type does not separate on any metric",
                 x=0.02, ha="left", color=INK, fontsize=10.5, y=1.04)
    fig.text(0.02, -0.26, "Black bars are category means across both platforms.",
             fontsize=7, color=MUTED, ha="left")
    finish(fig, 9, "material_vs_immaterial", outdir, pd.DataFrame(out))


# --- 10 ---------------------------------------------------------------------

def fig10(outdir):
    """
    Monthly coverage per ticker — the temporal unevenness the proposal denied.

    Small multiples rather than one crowded axis, and each panel keeps its own
    y-scale because the question is the shape over time, not whether NVDA
    out-publishes TTWO. The y-axis is therefore unlabelled by value; the
    accompanying data file carries the counts.
    """
    pans = {s: panel(s) for s in ("news", "reddit")}
    if all(v is None for v in pans.values()):
        return

    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.6), sharex=True)
    out = []
    for ax, tk in zip(axes.ravel(), ORDER):
        hgrid(ax, axis="y")
        for src, colour in PLATFORM.items():
            p = pans[src]
            if p is None:
                continue
            g = p[p["ticker"] == tk]
            if g.empty:
                continue
            m = (g.set_index("date")["article_count"]
                   .resample("ME").sum())
            # Each series is indexed to its own peak month. Plotted as raw
            # counts the two share a y-axis, and wherever one platform dwarfs
            # the other — AAPL's 2020-21 Reddit surge, for instance — the
            # smaller series is flattened into the baseline and unreadable.
            peak = m.max()
            idx = m / peak if peak else m
            ax.plot(idx.index, idx.values, color=colour, lw=1.4, zorder=3,
                    label=src.capitalize())
            ax.fill_between(idx.index, idx.values, color=colour, alpha=0.12,
                            zorder=2)
            for d, v in m.items():
                out.append({"platform": src, "ticker": tk,
                            "month": d.strftime("%Y-%m"), "count": int(v),
                            "share_of_peak": round(v / peak, 4) if peak else 0})
        ax.set_title(f"{tk}  ({CATEGORY.get(tk, '')})", loc="left", color=INK,
                     fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="News"),
                        Line2D([], [], color=ORANGE, lw=2, label="Reddit")],
               loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2)
    fig.suptitle("Coverage is not evenly distributed across the window",
                 x=0.02, ha="left", color=INK, fontsize=10.5, y=1.02)
    fig.text(0.02, -0.11,
             "Each series indexed to its own peak month, so both platforms stay legible where one dwarfs the other. "
             "Counts are in the data file.",
             fontsize=7, color=MUTED, ha="left")
    fig.tight_layout()
    finish(fig, 10, "monthly_coverage", outdir, pd.DataFrame(out))


FIGURES = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05,
           6: fig06, 7: fig07, 8: fig08, 9: fig09, 10: fig10}


def main():
    ap = argparse.ArgumentParser(description="Generate the report figures.")
    ap.add_argument("--only", nargs="*", type=int, help="Figure numbers to build")
    ap.add_argument("--outdir", default=FIG_DIR)
    args = ap.parse_args()

    want = args.only or sorted(FIGURES)
    for n in want:
        fn = FIGURES.get(n)
        if fn is None:
            print(f"no figure {n}")
            continue
        print(f"Figure {n}: {fn.__doc__.strip().splitlines()[0]}")
        fn(args.outdir)

    print(f"\nFigures in {args.outdir}/ (PNG at 300dpi + PDF); "
          f"the numbers behind each are in {args.outdir}/data/.")


if __name__ == "__main__":
    main()
