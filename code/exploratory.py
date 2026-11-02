"""Analyses registered as exploratory (E1-E7), reported without correction (Sections 4.2-4.6).

The mediation and spillover models (E1, E2, E6) are in sem.py; the regression of intrusion on
perceived fit (E1) is here.
"""
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from confirmatory import M1, M3, ols, one_sided, show
from prepare import zscore

VOD_CODES = {"never or almost never": 0, "less than weekly": 1, "weekly": 2,
             "several times a week": 3, "daily": 4}


def lincomb(m, expr, label, w=40):
    t = m.t_test(f"{expr} = 0")
    b, (lo, hi), p = float(np.squeeze(t.effect)), np.squeeze(t.conf_int()), float(np.squeeze(t.pvalue))
    print(f"  {label:{w}s} {b:+.2f} [{lo:+.2f}, {hi:+.2f}], p = {p:.3f}")


def omnibus(df):
    """4 x 2 ANOVA (type II): does the relevance effect differ across placements?"""
    tab = anova_lm(smf.ols("intrusion ~ C(condition)*C(relevant)", data=df).fit(), typ=2)
    r, res = tab.loc["C(condition):C(relevant)"], tab.loc["Residual"]
    print(f"  timing x relevance: F({int(r['df'])}, {int(res['df'])}) = {r['F']:.2f}, p = {r['PR(>F)']:.3f}")


def e3_moderators(df):
    """Persuasion knowledge and VOD-use frequency, one at a time, in the shape of M3."""
    d = df.assign(vod_z=zscore(df["vod_freq"].str.strip().str.lower().map(VOD_CODES)))
    for z, name, signs in [("pk_z", "persuasion knowledge", (+1, 0, +1)),
                           ("vod_z", "VOD-use frequency", (0, 0, 0))]:
        m = ols(f"intrusion ~ c_ctrl*{z} + relevance*{z}", d)
        print(f"  {name} (per SD, N = {int(m.nobs)})")
        for term, lab, s in zip([z, f"c_ctrl:{z}", f"relevance:{z}"],
                                ["main effect", "x control", "x relevance"], signs):
            show("  " + lab, m, term, s)


def e4_control_vs_relevance(df):
    """Viewer-chosen irrelevant ad vs forced relevant ad (forced placements weighted equally)."""
    d = df.assign(cell=df["condition"] + np.where(df["relevant"], "R", "I"))
    m = ols("intrusion ~ 0 + C(cell)", d)
    forced_rel = "(C(cell)[A0R] + C(cell)[A1R] + C(cell)[A2R]) / 3"
    for expr, lab in [("C(cell)[A3I]", "chosen, irrelevant ad (mean)"),
                      (forced_rel, "forced, relevant ad (mean)"),
                      (f"C(cell)[A3I] - {forced_rel}", "difference")]:
        lincomb(m, expr, lab)


def e5_placements(df):
    """Timing means (averaged over the relevance arms) and their differences."""
    m = ols("intrusion ~ C(condition, Treatment('A0'))*relevance", df)
    T = {"A0": None, **{k: f"C(condition, Treatment('A0'))[T.{k}]" for k in ("A1", "A2", "A3")}}
    for a, b, lab in [("A0", "A2", "post-roll - mid-roll"), ("A1", "A0", "pre-roll - post-roll"),
                      ("A3", "A2", "chosen - mid-roll"), ("A3", "A0", "chosen - post-roll"),
                      ("A3", "A1", "chosen - pre-roll")]:
        expr = f"-{T[b]}" if a == "A0" else T[a] if b == "A0" else f"{T[a]} - {T[b]}"
        lincomb(m, expr, lab)


def e7_chosen_timing(df):
    """Which timing viewers chose, and deferring the ad to the end by privacy concern."""
    a = df[df["condition"].eq("A3") & df["chosen_ad_timing"].isin(["pre", "mid", "post"])].copy()
    counts = a["chosen_ad_timing"].value_counts().reindex(["pre", "mid", "post"])
    chi = stats.chisquare(counts.values)
    print(f"  chosen (n = {len(a)}): " + ", ".join(f"{k} {v / len(a):.0%}" for k, v in counts.items())
          + f"; chi2(2) = {chi.statistic:.1f}, p = {chi.pvalue:.2g}")
    a["defer"] = a["chosen_ad_timing"].eq("post").astype(int)
    lg = smf.logit("defer ~ privacy_z", data=a).fit(disp=0, cov_type="HC0")
    b, (lo, hi), p = lg.params["privacy_z"], lg.conf_int().loc["privacy_z"], lg.pvalues["privacy_z"]
    print(f"  post-roll per SD privacy concern: OR = {np.exp(b):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}], "
          f"p = {one_sided(b, p, +1):.3f} (one-sided)")


def e1_perceived_fit(df):
    """Intrusion on perceived fit and the assigned relevance condition."""
    m = ols("intrusion ~ fit_z + " + M1.format(y="").split("~")[1],
            df.assign(fit_z=zscore(df["fit_m"])))
    for t in ("fit_z", "relevance"):
        print(f"  {t:10s} b = {m.params[t]:+.2f} (SE {m.bse[t]:.2f})")


def tracking(df):
    """Perceived tracking: level, what predicts it, and what it predicts.
    The last two lines (intrusion on tracking, H2b adjusted for tracking) were not registered."""
    t = df["tracking_m"]
    print(f"  M = {t.mean():.2f} (SD {t.std():.2f}); above the midpoint: {(t > 4).mean():.1%}")
    show("relevance -> tracking", ols(M1.format(y="tracking_m"), df), "relevance")
    dp = df.dropna(subset=["privacy_z"]).assign(tracking_z=lambda d: zscore(d["tracking_m"]))
    show("privacy concern -> tracking", ols(M3.format(y="tracking_m"), dp), "privacy_z")
    show("tracking (per SD) -> intrusion", ols("intrusion ~ tracking_z + c_ctrl + relevance", dp), "tracking_z")
    show("H2b, adjusted for tracking", ols(M3.format(y="intrusion") + " + tracking_z", dp), "relevance:privacy_z")
