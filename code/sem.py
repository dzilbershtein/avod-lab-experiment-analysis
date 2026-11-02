"""Structural equation model (E1, E2, E6; Section 4.5 and its figure). Requires semopy.

Perceived intrusion is a second-order factor over the intrusiveness and irritation factors
(they correlated above the registered r = 0.50). Control acts through perceived control and
relevance through perceived fit, with direct effects and the control x relevance term alongside.
The extended model adds the chain intrusion -> ad attitude -> platform attitude. ML estimation.
Indirect effects are standardised (SD of intrusion) with bias-corrected bootstrap 95% CIs.
"""
import numpy as np
from scipy.stats import norm
from semopy import Model, calc_stats

from prepare import ADATT, FIT, INT, IRR, PCTRL, PLATT

N_BOOT = 5000
SEED = 20260929
ITEMS = INT + IRR + FIT + PCTRL + ADATT + PLATT
FACTORS = f"""
intrusiveness =~ {' + '.join(INT)}
irritation =~ {' + '.join(IRR)}
fit =~ {' + '.join(FIT)}
pctrl =~ {' + '.join(PCTRL)}
"""
ATTITUDES = f"""
adatt =~ {' + '.join(ADATT)}
platatt =~ {' + '.join(PLATT)}
"""
SECOND_ORDER = "annoyance =~ 1*intrusiveness + 1*irritation\n"
STRUCTURE = """
fit ~ relevance
pctrl ~ c_ctrl
fit ~~ pctrl
annoyance ~ fit + pctrl + relevance + c_ctrl + ctrlXrel
"""
MEDIATION = FACTORS + SECOND_ORDER + STRUCTURE
EXTENDED = FACTORS + ATTITUDES + SECOND_ORDER + STRUCTURE + """
adatt ~ annoyance
platatt ~ adatt
"""


def sem_data(df):
    return df.dropna(subset=ITEMS).assign(ctrlXrel=df["c_ctrl"] * df["relevance"])


def srmr(m, d):
    obs = m.vars["observed"]
    S = d[obs].cov().values
    sd = np.sqrt(np.diag(S))
    R = (S - m.calc_sigma()[0]) / np.outer(sd, sd)
    return float(np.sqrt(np.mean(R[np.tril_indices_from(R)] ** 2)))


def fit(desc, d, label):
    m = Model(desc)
    m.fit(d)
    s = calc_stats(m).T
    print(f"  {label}: N = {len(d)}, CFI = {s.loc['CFI'].iloc[0]:.2f}, TLI = {s.loc['TLI'].iloc[0]:.2f}, "
          f"RMSEA = {s.loc['RMSEA'].iloc[0]:.2f}, SRMR = {srmr(m, d):.2f}")
    return m


def std_effects(m, d):
    """a (SD of mediator per unit of the contrast), b (standardised), direct and indirect
    (SD of intrusion per unit of the contrast) for each intervention."""
    s = m.inspect(std_est=True)
    g = lambda l, r: float(s[(s.lval == l) & (s.op == "~") & (s.rval == r)]["Est. Std"].iloc[0])
    out = {}
    for x, med in [("c_ctrl", "pctrl"), ("relevance", "fit")]:
        sd = d[x].std(ddof=0)
        a, b, c = g(med, x), g("annoyance", med), g("annoyance", x)
        out[x] = dict(a=a / sd, b=b, direct=c / sd, indirect=a * b / sd)
    return out


def bc_interval(boot, theta, level=0.95):
    """Bias-corrected percentile interval."""
    z0 = norm.ppf(np.clip(np.mean(boot < theta), 1e-6, 1 - 1e-6))
    zc = norm.ppf(0.5 + level / 2)
    return np.quantile(boot, norm.cdf(2 * z0 - zc)), np.quantile(boot, norm.cdf(2 * z0 + zc))


def run(df, n_boot=N_BOOT):
    d = sem_data(df)
    m = fit(MEDIATION, d, "mediation model")
    est = m.inspect(se_robust=True)                    # robust (sandwich) standard errors
    for x in ("c_ctrl", "relevance"):
        p = est[(est.lval == "annoyance") & (est.op == "~") & (est.rval == x)]["p-value"].iloc[0]
        print(f"  direct effect of {x}: p = {float(p):.2f}")
    eff = std_effects(m, d)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(n_boot):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        try:
            mb = Model(MEDIATION)
            mb.fit(b)
            e = std_effects(mb, b)
            boot.append([e["c_ctrl"]["indirect"], e["relevance"]["indirect"]])
        except Exception:
            continue
    boot = np.array(boot)
    print(f"  bootstrap: {len(boot)}/{n_boot} resamples converged")
    for j, (x, med) in enumerate([("c_ctrl", "perceived control"), ("relevance", "perceived fit")]):
        e = eff[x]
        e["ci"] = bc_interval(boot[:, j], e["indirect"])
        print(f"  {x} -> {med}: a = {e['a']:+.2f} SD, b = {e['b']:+.2f}, direct = {e['direct']:+.2f}, "
              f"indirect = {e['indirect']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]")

    m = fit(EXTENDED, d, "extended model")
    s = m.inspect(std_est=True)
    g = lambda l, r: float(s[(s.lval == l) & (s.op == "~") & (s.rval == r)]["Est. Std"].iloc[0])
    eff["spillover"] = dict(ad=g("adatt", "annoyance"), platform=g("platatt", "adatt"))
    print(f"  intrusion -> ad attitude {eff['spillover']['ad']:+.2f}; "
          f"ad -> platform attitude {eff['spillover']['platform']:+.2f} (standardised)")
    return eff
