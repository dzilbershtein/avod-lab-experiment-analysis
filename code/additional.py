"""Analyses added after the registered ones (labelled "not registered" in the paper), and the
robustness checks in Appendix E. None is corrected for multiple tests unless stated.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from confirmatory import M1, M3, five_tests, ols, show
from prepare import ADATT, PLATT, analysed, zscore

SESOI_D = 0.5        # the medium effect the study was powered for
N_BOOT = 5000
SEED = 20261023
BASE = M1.format(y="intrusion").split("~")[1]


def equivalence(df):
    """TOST of the null results against d = 0.5 (in composite units), with 90% CIs."""
    bound = SESOI_D * df["intrusion"].std()
    m1 = ols(M1.format(y="intrusion"), df)
    m3 = ols(M3.format(y="intrusion"), df.dropna(subset=["privacy_z"]))
    print(f"  bound = {bound:.2f} composite units")
    for lab, m, t in [("H1b", m1, "relevance"), ("H1c", m1, "c_ctrl:relevance"), ("H2a", m3, "c_ctrl:privacy_z")]:
        b, se, (lo, hi) = m.params[t], m.bse[t], m.conf_int(alpha=0.10).loc[t]
        p = max(1 - stats.norm.cdf((b + bound) / se), stats.norm.cdf((b - bound) / se))
        print(f"  {lab}: 90% CI [{lo:+.2f}, {hi:+.2f}], TOST p = {p:.3f}")


def johnson_neyman(df):
    """Where along privacy concern the relevance effect is reliably non-zero."""
    dp = df.dropna(subset=["privacy_z"])
    m = ols(M3.format(y="intrusion"), dp)
    V, b1, b3 = m.cov_params(), m.params["relevance"], m.params["relevance:privacy_z"]
    v11, v13 = V.loc["relevance", "relevance"], V.loc["relevance", "relevance:privacy_z"]
    v33 = V.loc["relevance:privacy_z", "relevance:privacy_z"]
    zc = stats.norm.ppf(0.975)
    effect = lambda z: (b1 + b3 * z, np.sqrt(v11 + z ** 2 * v33 + 2 * z * v13))
    grid = np.linspace(-3, 3, 6001)
    e, se = effect(grid)
    lower = grid[e + zc * se < 0].min()
    print(f"  relevance lowers intrusion above {lower:+.2f} SD ({(dp['privacy_z'] >= lower).mean():.0%} "
          f"of the sample); raises it nowhere: {not (e - zc * se > 0).any()}")
    z = dp["privacy_z"].quantile(0.90)
    e, se = effect(z)
    ctrl = m.params["c_ctrl"] + m.params["c_ctrl:privacy_z"] * z
    print(f"  at the 90th percentile: relevance {e:+.2f} [{e - zc * se:+.2f}, {e + zc * se:+.2f}], "
          f"control {ctrl:+.2f}")


def choosing_vs_assigned(df):
    """Chosen vs assigned placement, and a bootstrap split of the control effect into the
    placements viewers picked and choosing beyond them."""
    d = df.assign(slot=np.where(df["condition"].eq("A3"), df["chosen_ad_timing"],
                                df["condition"].map({"A0": "post", "A1": "pre", "A2": "mid"})),
                  chosen=df["condition"].eq("A3").astype(int))
    d = d[d["slot"].isin(["pre", "mid", "post"])]
    print(f"  N = {len(d)}")
    show("chosen vs assigned, same placement", ols("intrusion ~ chosen + C(slot) + relevance", d), "chosen")
    forced, ch = d[d["chosen"].eq(0)], d[d["chosen"].eq(1)]
    for s in ("pre", "mid", "post"):
        a, f = ch.loc[ch["slot"].eq(s), "intrusion"], forced.loc[forced["slot"].eq(s), "intrusion"]
        print(f"    {s:4s} chosen {a.mean():+.2f} (n = {len(a)}) vs assigned {f.mean():+.2f} (n = {len(f)})")

    def split(f, c):
        share = c["slot"].value_counts(normalize=True)
        means = f.groupby("slot")["intrusion"].mean()
        bench = sum(share.get(s, 0) * means[s] for s in ("pre", "mid", "post"))
        return c["intrusion"].mean() - bench, bench - means.mean(), c["intrusion"].mean() - means.mean()
    rng = np.random.default_rng(SEED)
    groups = [g for _, g in forced.groupby(["slot", "relevance"])]
    boot = np.array([split(pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1e9))
                                      for g in groups]),
                           ch.sample(len(ch), replace=True, random_state=rng.integers(1e9)))
                     for _ in range(N_BOOT)])
    for k, lab in enumerate(["choosing, beyond the placements picked", "placements picked",
                             "total (vs the average forced placement)"]):
        lo, hi = np.percentile(boot[:, k], [2.5, 97.5])
        print(f"  {lab:40s} {split(forced, ch)[k]:+.2f} [{lo:+.2f}, {hi:+.2f}]")


def relevance_by_placement(df):
    rows = [(t, ols("intrusion ~ relevance", df[df["timing"].eq(t)])) for t in df["timing"].unique()]
    adj = multipletests([m.pvalues["relevance"] for _, m in rows], method="holm")[1]
    for (t, m), p in zip(rows, adj):
        lo, hi = m.conf_int().loc["relevance"]
        print(f"  {t:14s} b = {m.params['relevance']:+.2f} [{lo:+.2f}, {hi:+.2f}], Holm p = {p:.3f}")


def age_and_gender(df):
    """Does the control effect differ by age group or gender? (Holm across the two)"""
    d = df.assign(woman=df["gender"].map({"Female": 1.0, "Male": 0.0}))
    ps = []
    for v, lab in [("older", "aged 35+"), ("woman", "women")]:
        dd = d.dropna(subset=[v])
        m = ols(f"intrusion ~ c_ctrl*{v} + relevance + c_n1 + c_n2", dd)
        b, (lo, hi) = m.params[f"c_ctrl:{v}"], m.conf_int().loc[f"c_ctrl:{v}"]
        ps.append(m.pvalues[f"c_ctrl:{v}"])
        print(f"  control x {lab}: b = {b:+.2f} [{lo:+.2f}, {hi:+.2f}]; control effect "
              f"{m.params['c_ctrl']:+.2f} (others) vs {m.params['c_ctrl'] + b:+.2f} ({lab}), N = {len(dd)}")
    print(f"  Holm p: {np.round(multipletests(ps, method='holm')[1], 3).tolist()}")
    print("  aged 35+ per condition:", df[df["older"].eq(1)]["timing"].value_counts().to_dict())
    show("control effect adjusted for age group", ols(M1.format(y="intrusion") + " + older", df), "c_ctrl")
    forced = df[df["condition"].ne("A3")]
    for f in ("intrusion ~ C(timing)", "intrusion ~ C(timing) + older"):
        print(f"  forced timing coefficients ({f}):", ols(f, forced).params.round(2).to_dict())


def downstream(df):
    d = df.assign(ad_att=zscore(df[ADATT].mean(axis=1)), plat_att=zscore(df[PLATT].mean(axis=1)),
                  satisfaction=zscore(df["sat_1"]))
    for y in ("ad_att", "plat_att", "satisfaction"):
        m = ols(M1.format(y=y), d)
        for t in ("c_ctrl", "relevance"):
            show(f"{y}: {t}", m, t)


def registered_rules(data):
    """The five tests under the registered versions of the two exclusion rules (Appendix E)."""
    variants = {"without the attention-based exclusions": dict(attention=False),
                "registered technical-malfunction rule": dict(registered_tech_fail=True),
                "registered ad-seen exclusion": dict(ad_seen=True),
                "both registered rules": dict(ad_seen=True, registered_tech_fail=True)}
    for name, kw in variants.items():
        d = analysed(data, **kw)
        _, rows, decisions = five_tests(d, verbose=False)
        print(f"  {name} (N = {len(d)}): " + "; ".join(
            f"{lab[:3]} {b:+.2f}" + (f" [{lo:+.2f}, {hi:+.2f}]" if lab.startswith("H1a") else "")
            + ("*" if decisions[lab] else "") for lab, b, lo, hi, _ in rows))
    print("  (* = supported after the Holm correction)")


def control_effect_checks(df):
    """The robustness table in Appendix E, below the line, plus the rank-based test."""
    dd = df.dropna(subset=["privacy_z", "pk_z", "age", "gender"]).assign(
        woman=lambda d: d["gender"].eq("Female").astype(float))
    show(f"covariates (N = {len(dd)})", ols(M1.format(y="intrusion") + " + privacy_z + pk_z + older + woman", dd), "c_ctrl")
    s = df.dropna(subset=["ad_id_shown"])
    show(f"one intercept per creative (N = {len(s)})", ols(M1.format(y="intrusion") + " + C(ad_id_shown)", s), "c_ctrl")
    s = df.dropna(subset=["title_watched"])
    show(f"one intercept per programme (N = {len(s)})", ols(M1.format(y="intrusion") + " + C(title_watched)", s), "c_ctrl")
    for y in ("intrusiveness_z", "irritation_z"):
        show(y, ols(M1.format(y=y), df), "c_ctrl")
    for arm in ("irrelevant", "relevant"):
        s = df[df["ad_relevance_arm"].eq(arm)]
        show(f"{arm} ads only (N = {len(s)})", ols("intrusion ~ c_ctrl + c_n1 + c_n2", s), "c_ctrl")
    u = stats.mannwhitneyu(df.loc[df["condition"].eq("A3"), "intrusion"],
                           df.loc[df["condition"].ne("A3"), "intrusion"], alternative="less")
    print(f"  Mann-Whitney, chosen < forced: p = {u.pvalue:.1g}")


def programme_checks(df, catalogue_json):
    """Did the programme watched, or the title featured at the top of the catalogue, matter?"""
    cat = json.load(open(catalogue_json))
    s = df.dropna(subset=["title_watched"]).copy()
    s["category"] = s["title_watched"].map(lambda t: cat[t]["relevant_ad_category"])
    s["classic"] = s["title_watched"].map(lambda t: float(cat[t]["year"] < 2000))
    print(f"  programme missing for {df['title_watched'].isna().sum()} participants")

    def omnibus(m, prefix):
        names = [n for n in m.params.index if n.startswith(prefix) and ":" not in n]
        w = m.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
        return f"chi2({len(names)}) = {w.statistic:.2f}, p = {w.pvalue:.2f}"
    print("  programme watched:", omnibus(ols(f"intrusion ~ {BASE} + C(title_watched)", s), "C(title_watched)"))
    h = s.dropna(subset=["hero_title"])
    print("  featured title:", omnibus(ols(f"intrusion ~ {BASE} + C(hero_title)", h), "C(hero_title)"))
    for x in ("relevance", "c_ctrl"):
        m = ols(f"intrusion ~ {BASE} + C(category)*{x}", s)
        names = [n for n in m.params.index if n.startswith("C(category)") and n.endswith(f":{x}")]
        w = m.wald_test(", ".join(f"{n} = 0" for n in names), scalar=True)
        print(f"  category x {x}: chi2({len(names)}) = {w.statistic:.2f}, p = {w.pvalue:.2f}")
        m = ols(f"intrusion ~ {BASE} + classic*{x}", s)
        print(f"  classic film x {x}: p = {m.pvalues[f'classic:{x}']:.2f}")


def creatives_and_completion(df):
    s = df.dropna(subset=["ad_id_shown"])
    t = pd.crosstab(s["ad_id_shown"], s["ad_relevance_arm"])
    print(f"  creatives shown: {len(t)}; in both relevance arms: {int((t > 0).all(axis=1).sum())}")
    e = df.dropna(subset=["ad_view_duration_sec"])
    print(f"  sessions with a logged ad: {len(e)}; played the full 15 s: "
          f"{int(e['ad_view_duration_sec'].eq(15).sum())}")
