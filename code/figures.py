"""The three figures in the paper. Fonts: Nimbus Sans (a Helvetica clone) if its .otf files are
in a fonts/ folder next to this file, otherwise matplotlib's default sans-serif."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

from confirmatory import M3, ols

BLUE, ORANGE, INK, INK2, GREY, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8883", "#e6e5e1"
fonts = list((Path(__file__).parent / "fonts").glob("NimbusSans-*.otf"))
for f in fonts:
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.unicode_minus": True, **({"font.family": "Nimbus Sans"} if fonts else {})})
MINUS = lambda s: s.replace("-", "−")


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def simple_slopes(df, out):
    """Simple-slopes figure: predicted intrusion at -1 and +1 SD of privacy concern (model M3)."""
    m = ols(M3.format(y="intrusion"), df.dropna(subset=["privacy_z"]))

    def pred(c, r, z):
        p = m.get_prediction(pd.DataFrame({"c_ctrl": [c], "relevance": [r], "privacy_z": [z]}))
        f = p.summary_frame(alpha=0.05).iloc[0]
        return f["mean"], f["mean_ci_lower"], f["mean_ci_upper"]
    groups = [("Low privacy concern (−1 SD)", BLUE, -1, "o", "-", -0.04),
              ("High privacy concern (+1 SD)", ORANGE, +1, "s", "--", 0.04)]
    panels = [("A. Viewer control (H2a)", ["Forced\n(average of three placements)", "Viewer-chosen"],
               lambda z: (pred(-0.25, 0, z), pred(0.75, 0, z)), "c_ctrl"),
              ("B. Contextual relevance (H2b)", ["Irrelevant ad", "Relevant ad"],
               lambda z: (pred(0, -0.5, z), pred(0, 0.5, z)), "relevance")]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.1), sharey=True)
    for ax, (title, ticks, ends, term) in zip(axs, panels):
        labels = []
        for lab, col, z, mk, ls, dx in groups:
            (y0, l0, h0), (y1, l1, h1) = ends(z)
            xs = np.array([0, 1]) + dx
            ax.errorbar(xs, [y0, y1], yerr=[[y0 - l0, y1 - l1], [h0 - y0, h1 - y1]], fmt="none",
                        ecolor=col, elinewidth=1.2, capsize=3, alpha=0.8, zorder=2)
            ax.plot(xs, [y0, y1], color=col, lw=2, ls=ls, marker=mk, ms=8, mec="white", mew=1.5, zorder=3)
            b = m.params[term] + m.params[f"{term}:privacy_z"] * z
            labels.append([y1, MINUS(f"{b:+.2f}"), col])
        if abs(labels[0][0] - labels[1][0]) < 0.14:      # keep the two end labels apart
            mid, o = (labels[0][0] + labels[1][0]) / 2, np.argsort([labels[0][0], labels[1][0]])
            labels[o[0]][0], labels[o[1]][0] = mid - 0.07, mid + 0.07
        for y, t, col in labels:
            ax.text(1.17, y, t, va="center", fontsize=10, color=col, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.35, 1.45)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.tick_params(axis="y", labelleft=True)
        style(ax)
    axs[0].set_ylabel("Predicted perceived intrusion\n(standard units; lower = less intrusive)")
    axs[0].set_ylim(-1.0, 1.0)
    axs[0].set_yticks([-1, -0.5, 0, 0.5, 1])
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2, ls=ls, marker=mk, ms=7, mec="white", label=l)
                        for l, c, z, mk, ls, dx in groups],
               loc="upper center", ncol=2, frameon=False, fontsize=9.5, handlelength=3.2,
               bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def placement(df, out):
    """Placement figure: intrusion at each placement, assigned vs chosen, with 95% CIs."""
    slots = [("pre", "A1", "Pre-roll"), ("mid", "A2", "Mid-roll"), ("post", "A0", "Post-roll")]
    ch = df[df["condition"].eq("A3")]
    n_chose = int(ch["chosen_ad_timing"].isin(["pre", "mid", "post"]).sum())

    def mean_ci(s):
        h = stats.t.ppf(0.975, len(s) - 1) * s.std() / np.sqrt(len(s))
        return s.mean(), h, len(s)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ticks = []
    for i, (k, cond, lab) in enumerate(slots):
        n = {}
        for s, col, mk, dx, key in [(df.loc[df["condition"].eq(cond), "intrusion"], ORANGE, "s", -0.14, "a"),
                                    (ch.loc[ch["chosen_ad_timing"].eq(k), "intrusion"], BLUE, "o", 0.14, "c")]:
            mu, h, n[key] = mean_ci(s.dropna())
            ax.errorbar(i + dx, mu, yerr=h, fmt=mk, color=col, ms=8, mec="white", mew=1.5, capsize=3,
                        elinewidth=1.2)
        ticks.append(f"{lab}\nassigned n = {n['a']}\nchose n = {n['c']} ({round(100 * n['c'] / n_chose)}%)")
    ax.set_xticks(range(3))
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-1.35, 1.0)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Perceived intrusion\n(standard units; lower = less intrusive)")
    style(ax)
    ax.legend(handles=[Line2D([0], [0], color=ORANGE, marker="s", ls="", ms=7, label="Assigned this placement"),
                       Line2D([0], [0], color=BLUE, marker="o", ls="", ms=7, label="Chose this placement")],
              frameon=False, loc="upper left", fontsize=9, ncol=2, bbox_to_anchor=(0, 1.12))
    plt.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def sem_paths(eff, out):
    """SEM figure: the structural model, drawn from the estimates returned by sem.run."""
    c, r, sp = eff["c_ctrl"], eff["relevance"], eff["spillover"]
    f = lambda v: MINUS(f"{v:+.2f}")
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_xlim(0, 148)
    ax.set_ylim(-4, 54)
    ax.axis("off")

    def box(x, y, t, bold=False, w=15, h=6.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="white", ec=INK2, lw=1.4))
        ax.text(x, y, t, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold" if bold else "normal")

    def arrow(p, q, col, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.6, color=col, ls=ls))

    def num(x, y, s, col, ha="center", va="bottom"):
        ax.text(x, y, s, ha=ha, va=va, fontsize=10, color=col, fontweight="bold")

    def note(x, y, s, ha="center", va="top", col=INK2):
        ax.text(x, y, s, ha=ha, va=va, fontsize=7.8, color=col, linespacing=1.15)
    box(9, 42, "Viewer control\n(chose timing)")
    box(44, 42, "Perceived\ncontrol")
    box(9, 10, "Contextual\nrelevance\n(matched ad)")
    box(44, 10, "Perceived\nfit")
    box(76, 26, "Perceived\nintrusion", bold=True, h=7)
    box(107, 26, "Attitude to\nthe ad")
    box(139, 26, "Attitude to\nthe platform")
    ax.text(107, 33.5, "Spillover to attitudes", ha="center", fontsize=8.5, color=INK2, fontstyle="italic")
    arrow((16.8, 42), (36.2, 42), BLUE)
    num(26.5, 43, f(c["a"]), BLUE)
    note(26.5, 41, f"perceived control\n{c['a']:.2f} SD higher")
    arrow((51.8, 40.2), (68.4, 29.8), BLUE)
    num(61.5, 39.5, f(c["b"]), BLUE, ha="left")
    note(61.5, 39.2, "more control,\nless intrusion", ha="left")
    arrow((16.8, 10), (36.2, 10), ORANGE)
    num(26.5, 11, f(r["a"]), ORANGE)
    note(26.5, 9, f"perceived fit\n{r['a']:.2f} SD higher")
    arrow((51.8, 11.8), (68.4, 22.2), ORANGE)
    num(61.5, 13, f(r["b"]), ORANGE, ha="left", va="top")
    note(61.5, 10.2, "better fit,\nless intrusion", ha="left")
    arrow((9, 38.4), (68.2, 27.6), GREY, "--")
    note(24, 32.4, f"direct: {f(c['direct'])} (n.s.)", ha="left", col=GREY)
    arrow((9, 13.8), (68.2, 24.4), GREY, "--")
    note(24, 21.4, f"direct: {f(r['direct'])} (n.s.)", ha="left", col=GREY)
    arrow((84.2, 26), (98.8, 26), INK2)
    num(91.5, 27, f(sp["ad"]), INK2)
    note(91.5, 24.8, "more intrusion,\nworse attitude\nto the ad")
    arrow((115.2, 26), (130.8, 26), INK2)
    num(123, 27, f(sp["platform"]), INK2)
    note(123, 24.8, "better attitude\nto the ad, better\nto the platform")
    ci = lambda e: f"{f(e['indirect'])} SD of intrusion [{f(e['ci'][0])}, {f(e['ci'][1])}]"
    ax.text(44, 50.5, f"Indirect effect of control via perceived control: {ci(c)}",
            ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(44, 1.6, f"Indirect effect of relevance via perceived fit: {ci(r)}",
            ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    ax.text(74, -3.4, "How to read the numbers: + means more of what the arrow points to, − means less. "
            "Numbers are in standard deviations (SD); n.s. = not significant.", ha="center", fontsize=8, color=INK2)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
