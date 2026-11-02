"""Participants, reliability and manipulation checks (Section 4.1, the cell-means table and Appendices C and D)."""
import numpy as np
import pandas as pd
from scipy import stats

from prepare import (ADATT, FIT, INT, IRR, IUIPC, IUIPC_SUB, PCTRL, PK, PLATT, TIMING, TRACK,
                     events, truthy)


def dropout(data):
    """Where the participants who did not complete left (from their event logs)."""
    nc = data[~data["post_survey_complete"].map(truthy)]

    def stage(js):
        kinds = {e["kind"] for e in events(js)}
        if kinds & {"content_ended", "watch_end"}:
            return "after the clip"
        if kinds & {"watch_start", "content_play", "ad_start"}:
            return "during the clip"
        return "before consent"
    print(f"started {len(data)}, completed {len(data) - len(nc)}, did not complete {len(nc)}:",
          nc["events_json"].map(stage).value_counts().to_dict())


def cronbach_alpha(df, cols):
    X = df[cols].dropna()
    return len(cols) / (len(cols) - 1) * (1 - X.var().sum() / X.sum(axis=1).var())


def mcdonald_omega(df, cols):
    from statsmodels.multivariate.factor import Factor
    lam = np.abs(Factor(df[cols].dropna().values, n_factor=1, method="ml").fit().loadings[:, 0])
    return lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())


def reliability(df):
    scales = {"intrusiveness": INT, "irritation": IRR, "perceived fit": FIT,
              "perceived control": PCTRL, "ad attitude": ADATT, "platform attitude": PLATT,
              "persuasion knowledge": PK, **{f"IUIPC {k}": v for k, v in IUIPC_SUB.items()}}
    for name, cols in scales.items():
        print(f"  {name:22s} alpha = {cronbach_alpha(df, cols):.2f}")
    print(f"  {'IUIPC (9 items)':22s} omega = {mcdonald_omega(df, IUIPC):.2f}")
    print(f"  {'perceived tracking':22s} r = {df[TRACK].corr().iloc[0, 1]:.2f}")
    print(f"  intrusiveness-irritation r = {df['intrusiveness_m'].corr(df['irritation_m']):.2f}")


def describe(df):
    t = df["total_time_s"] / 60
    print(f"session time: median {t.median():.1f} min (IQR {t.quantile(.25):.1f}-{t.quantile(.75):.1f})")
    print(f"under 35: {1 - df['older'].mean():.0%}; daily VOD use: "
          f"{df['vod_freq'].eq('Daily').sum() / df['vod_freq'].notna().sum():.0%}")
    t = pd.crosstab(df["timing"], df["older"])
    chi2, p = stats.chi2_contingency(t)[:2]
    print(f"aged 35+ by timing: {(t[1.0] / t.sum(axis=1)).round(2).to_dict()}; "
          f"chi2({(t.shape[0] - 1) * (t.shape[1] - 1)}) = {chi2:.2f}, p = {p:.3f}")
    print(f"with privacy concern: {df['privacy_z'].notna().sum()}; "
          f"with persuasion knowledge: {df['pk_z'].notna().sum()}")


def participants_table(df):
    """Appendix D: participants by ad-timing condition."""
    ages = ["18–24", "25–34", "35–44", "45–54", "55 or older"]

    def cnt(mask, base):
        return f"{int(mask.sum())} ({round(100 * mask.sum() / base.sum())})"
    cols = {}
    for name, g in [("All", df)] + [(t, df[df["timing"].eq(t)]) for t in TIMING.values()]:
        rec = g["gender"].notna()
        r = {"Participants": len(g), "Women": cnt(g["gender"].eq("Female"), rec),
             "Men": cnt(g["gender"].eq("Male"), rec),
             "Prefer not to say": int(g["gender"].eq("Prefer not to say").sum())}
        r.update({a: cnt(g["age"].eq(a), g["age"].notna()) for a in ages})
        r["Uses VOD daily"] = cnt(g["vod_freq"].eq("Daily"), g["vod_freq"].notna())
        r["Demographics not recorded"] = int((~rec).sum())
        for lab, items in [("Privacy concern", IUIPC), ("Persuasion knowledge", PK)]:
            s = g[items].mean(axis=1)
            r[f"{lab} M (SD)"] = f"{s.mean():.2f} ({s.std():.2f})"
        cols[name] = r
    print(pd.DataFrame(cols).to_string())


def welch_d(a, b):
    a, b = a.dropna(), b.dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return f"M = {a.mean():.2f} vs {b.mean():.2f}; t = {t:.2f}, p = {p / 2:.2g} (one-sided), d = {(a.mean() - b.mean()) / sp:.2f}"


def manipulation_checks(df):
    rel, chose = df["relevant"], df["condition"].eq("A3")
    print("control check (perceived control):", welch_d(df.loc[chose, "pctrl_m"], df.loc[~chose, "pctrl_m"]))
    print("relevance check (ad chosen for the video):",
          welch_d(df.loc[rel, "manip_relevance"], df.loc[~rel, "manip_relevance"]))
    print("perceived fit:", welch_d(df.loc[rel, "fit_m"], df.loc[~rel, "fit_m"]))
    print(f"relevant ad rated below the fit midpoint: {(df.loc[rel, 'fit_m'] < 4).mean():.0%}")


def cell_means(df):
    """Cell-means table: mean (SD), n of perceived intrusion by condition."""
    g = df.groupby(["timing", "ad_relevance_arm"])["intrusion"].agg(["mean", "std", "count"])
    print(g.round(2).unstack().to_string())
    print("timing means:", df.groupby("timing")["intrusion"].mean().round(2).to_dict())
