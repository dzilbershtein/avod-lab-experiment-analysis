"""The five pre-registered confirmatory tests (Sections 4.2-4.4 and the confirmatory-tests table).

H1a, H1b and H1c come from one 4 x 2 factorial OLS model (M1), H2a and H2b from a moderated
regression (M3), all with HC3 standard errors. H1c is two-sided, the other four one-sided in the
predicted direction, and the five form one Holm-Bonferroni family (alpha = 0.05).
"""
import numpy as np
import statsmodels.formula.api as smf

M1 = "{y} ~ c_ctrl*relevance + c_n1*relevance + c_n2*relevance"
M3 = "{y} ~ c_ctrl*privacy_z + relevance*privacy_z"
HYPOTHESES = [  # label, model, term, predicted sign (0 = two-sided)
    ("H1a control lowers intrusion", "M1", "c_ctrl", -1),
    ("H1b relevance lowers intrusion", "M1", "relevance", -1),
    ("H1c control x relevance != 0", "M1", "c_ctrl:relevance", 0),
    ("H2a privacy x control", "M3", "c_ctrl:privacy_z", -1),
    ("H2b privacy x relevance", "M3", "relevance:privacy_z", +1),
]


def ols(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="HC3")


def one_sided(b, p2, sign):
    return p2 / 2 if np.sign(b) == sign else 1 - p2 / 2


def show(label, m, term, sign=0, w=40):
    b, (lo, hi), p2 = m.params[term], m.conf_int().loc[term], m.pvalues[term]
    p = p2 if sign == 0 else one_sided(b, p2, sign)
    print(f"  {label:{w}s} b = {b:+.2f} [{lo:+.2f}, {hi:+.2f}], p = {p:.3f}"
          + ("" if sign == 0 else " (one-sided)"))


def holm(pvals, alpha=0.05):
    reject, out = True, {}
    for k, (name, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        reject = reject and p <= alpha / (len(pvals) - k)
        out[name] = reject
    return out


def five_tests(df, verbose=True):
    models = {"M1": ols(M1.format(y="intrusion"), df),
              "M3": ols(M3.format(y="intrusion"), df.dropna(subset=["privacy_z"]))}
    rows, pvals = [], {}
    for label, model, term, sign in HYPOTHESES:
        m = models[model]
        b, (lo, hi), p2 = m.params[term], m.conf_int().loc[term], m.pvalues[term]
        p = p2 if sign == 0 else one_sided(b, p2, sign)
        rows.append((label, b, lo, hi, p))
        pvals[label] = p
    decisions = holm(pvals)
    if verbose:
        print(f"N = {int(models['M1'].nobs)} (H2: {int(models['M3'].nobs)})")
        for label, b, lo, hi, p in rows:
            print(f"  {label:32s} b = {b:+.2f} [{lo:+.2f}, {hi:+.2f}]  p = {p:.3f}  "
                  f"{'supported' if decisions[label] else 'not supported'}")
    return models, rows, decisions


def simple_slopes(m3):
    """Effect of each intervention at -1 and +1 SD of privacy concern."""
    for x in ("c_ctrl", "relevance"):
        effects = [m3.params[x] + m3.params[f"{x}:privacy_z"] * z for z in (-1, 1)]
        print(f"  {x:10s} effect at -1 SD {effects[0]:+.2f}, at +1 SD {effects[1]:+.2f}")
