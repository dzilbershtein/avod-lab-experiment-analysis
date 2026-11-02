"""Build the analysed sample from the raw AVOD-Lab export.

The export has one row per session: the survey answers, a few playback columns, and the
session's event log as JSON (`events_json`). `derive` adds the measures that exist only
in the event log; `analysed` applies the pre-registered exclusions and computes the
scores and contrast codes used by every analysis.
"""
import json

import numpy as np
import pandas as pd

# survey items
INT = [f"int_{i}" for i in range(1, 8)]            # perceived intrusiveness
IRR = [f"irr_{i}" for i in range(1, 6)]            # irritation
FIT = [f"rel_{i}" for i in range(1, 4)]            # perceived ad-content fit
PCTRL = [f"ctrl_{i}" for i in range(1, 4)]         # perceived control
ADATT = [f"ad_att_{i}" for i in range(1, 4)]
PLATT = [f"plat_att_{i}" for i in range(1, 4)]
IUIPC_SUB = {"collection": [f"iuipc_col_{i}" for i in range(1, 4)],
             "control": [f"iuipc_ctl_{i}" for i in range(1, 4)],
             "awareness": [f"iuipc_awa_{i}" for i in range(1, 4)]}
IUIPC = sum(IUIPC_SUB.values(), [])
PK = [f"pk_{i}" for i in range(1, 4)]              # persuasion knowledge
TRACK = ["track_1", "track_2"]                     # perceived tracking
STRAIGHTLINE_ITEMS = INT + IRR + FIT + PCTRL       # the 18 post-exposure items

TIMING = {"A0": "Post-roll", "A1": "Pre-roll", "A2": "Mid-roll", "A3": "Viewer-chosen"}
OLDER = ["35–44", "45–54", "55 or older"]

ATTN_CORRECT = {"attn_pre": 6, "attn_post": 5}
TECH_FAIL_KINDS = {"content_play_blocked", "ad_play_blocked", "ad_skipped_due_to_error"}
CHOICE = {"preroll": "pre", "midroll": "mid", "postroll": "post"}
FULL_CLIP_S = 479                                  # clips are 480 s
SUBSEC_MS = 1000


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std(ddof=1)


def truthy(v):
    return v in (True, "True", "true", 1, "1")


def falsy(v):
    return v in (False, "False", "false", 0, "0")


def events(js):
    return json.loads(js) if isinstance(js, str) and js else []


# --------------------------------------------------------------------------- #
#  measures from the event log
# --------------------------------------------------------------------------- #
def per_session(ev):
    kinds = [e["kind"] for e in ev]
    ts = [pd.to_numeric(e.get("ts_client"), errors="coerce") for e in ev]
    pl = [e.get("payload") or {} for e in ev]
    first = lambda k: next((i for i, x in enumerate(kinds) if x == k), None)
    out = {}

    # technical malfunction. Registered rule: any blocked/error event, an ad that ended on a
    # source error, or no ad_end logged. Rule used (Section 3.7): events were lost under server
    # load, so a missing ad_end counts only if the clip was not watched to the end, and a
    # blocked play that still ended "complete" did play.
    ad_end = [i for i, k in enumerate(kinds) if k == "ad_end"]
    src_err = any(pl[i].get("reason") == "source_error" for i in ad_end)
    blocked = bool(TECH_FAIL_KINDS & set(kinds))
    out["tech_fail_registered"] = blocked or src_err or not ad_end
    watched = next((pl[i].get("t_content") for i, k in enumerate(kinds) if k == "watch_end"), None)
    full = watched is not None and float(watched) >= FULL_CLIP_S
    done = any(pl[i].get("reason") == "complete" for i in ad_end)
    out["tech_fail"] = src_err or (blocked and not done) or (not ad_end and not full)

    # session time: session start to the end of the post-exposure survey
    s0 = first("session_start")
    post = [i for i, k in enumerate(kinds) if k == "survey_complete" and pl[i].get("survey") == "post"]
    out["total_time_s"] = (ts[post[0]] - ts[s0]) / 1000 if s0 is not None and post else None

    # careless responding: share of survey pages submitted < 1 s after the previous one
    gaps = []
    for survey in ("pre", "post"):
        idx = [i for i, k in enumerate(kinds) if k == "survey_page" and pl[i].get("survey") == survey]
        gaps += [ts[b] - ts[a] for a, b in zip(idx, idx[1:])]
    out["subsec_page_share"] = sum(g < SUBSEC_MS for g in gaps) / len(gaps) if gaps else None

    # viewer-chosen timing; if the choice event was lost, the placement the ad played at
    c = first("ad_timing_chosen")
    if c is not None:
        out["chosen_ad_timing"] = CHOICE.get(pl[c].get("choice"), "")
    else:
        s = first("ad_start")
        out["chosen_ad_timing"] = CHOICE.get(pl[s].get("placement"), "") if s is not None else ""

    # pre-registered: only a session's first post-exposure survey is analysed
    out["n_post_surveys"] = len(post)
    out["_first_post"] = (pl[post[0]].get("all_responses") or {}) if post else {}
    # answers logged page by page, to fill items whose survey_complete event was lost
    cut = post[0] if post else len(kinds)
    out["_pages"] = {}
    for i, k in enumerate(kinds):
        if k == "survey_page" and (pl[i].get("survey") == "pre" or i < cut):
            out["_pages"].update(pl[i].get("responses") or {})
    return out


def derive(raw):
    """One row per session: the export plus the event-log measures."""
    p = raw.set_index("session_id")
    logs = {sid: events(js) for sid, js in p.pop("events_json").items()}
    d = pd.DataFrame.from_dict({sid: per_session(ev) for sid, ev in logs.items() if ev},
                               orient="index")
    first_post, pages = d.pop("_first_post"), d.pop("_pages")
    for sid in d.index[d["n_post_surveys"] > 1]:
        for k, v in first_post[sid].items():
            if k in p.columns:
                p.at[sid, k] = v
    n_filled = 0
    for sid in d.index:
        missing = [k for k, v in pages[sid].items() if k in p.columns and pd.isna(p.at[sid, k])]
        for k in missing:
            p.at[sid, k] = pages[sid][k]
        n_filled += bool(missing)
    print(f"items filled from page events for {n_filled} sessions; "
          f"{len(p) - len(d)} sessions have no event log")
    for item, correct in ATTN_CORRECT.items():
        v = pd.to_numeric(p[item], errors="coerce")
        p[f"{item}_pass"] = v.eq(correct).where(v.notna())
    out = p.join(d.drop(columns="n_post_surveys")).reset_index()
    out.loc[out["condition"].ne("A3"), "chosen_ad_timing"] = ""
    return out


# --------------------------------------------------------------------------- #
#  exclusions, scores, contrasts
# --------------------------------------------------------------------------- #
ATTENTION_RULES = ["attention", "subsec_dwell", "straightline"]


def exclusion_flags(df, registered_tech_fail=False):
    tf = "tech_fail_registered" if registered_tech_fail else "tech_fail"
    fl = pd.DataFrame({
        "attention": df["attn_pre_pass"].map(falsy) & df["attn_post_pass"].map(falsy),
        "ad_not_seen": df["manip_ad_seen"].eq("No"),
        "tech_fail": df[tf].map(truthy),
        "subsec_dwell": pd.to_numeric(df["subsec_page_share"], errors="coerce") > 0.5,
        "straightline": df[STRAIGHTLINE_ITEMS].nunique(axis=1).eq(1),
    }, index=df.index)
    return fl.fillna(False).astype(bool)


def analysed(data, attention=True, ad_seen=False, registered_tech_fail=False, verbose=False):
    """The analysed sample. Defaults are the rules as applied in the paper (Section 3.7):
    the ad-seen rule is not applied and the revised technical-malfunction rule is used."""
    df = data.copy()
    for c in INT + IRR + FIT + PCTRL + ADATT + PLATT + IUIPC + PK + TRACK + ["manip_relevance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[~df["is_test"].map(truthy) & df["post_survey_complete"].map(truthy)].copy()
    fl = exclusion_flags(df, registered_tech_fail)
    rules = [r for r in fl.columns if (attention or r not in ATTENTION_RULES)
             and (ad_seen or r != "ad_not_seen")]
    if verbose:
        print(f"completed the post-exposure survey: {len(df)}")
        for r in fl.columns:
            print(f"  {r:13s} {int(fl[r].sum()):3d}" + ("" if r in rules else "  (not applied)"))
    df = df[~fl[rules].any(axis=1)].copy()
    if verbose:
        print(f"analysed: {len(df)}")

    df["intrusiveness_m"] = df[INT].mean(axis=1)
    df["irritation_m"] = df[IRR].mean(axis=1)
    df["intrusiveness_z"] = zscore(df["intrusiveness_m"])
    df["irritation_z"] = zscore(df["irritation_m"])
    df["intrusion"] = (df["intrusiveness_z"] + df["irritation_z"]) / 2   # primary outcome
    df["fit_m"] = df[FIT].mean(axis=1)
    df["pctrl_m"] = df[PCTRL].mean(axis=1)
    df["privacy_z"] = zscore(df[IUIPC].mean(axis=1))
    df["pk_z"] = zscore(df[PK].mean(axis=1))
    df["tracking_m"] = df[TRACK].mean(axis=1)
    df["older"] = df["age"].isin(OLDER).astype(float).where(df["age"].notna())

    # contrasts (A0 post, A1 pre, A2 mid, A3 viewer-chosen)
    c = df["condition"]
    df["relevant"] = df["ad_relevance_arm"].eq("relevant")
    df["relevance"] = np.where(df["relevant"], 0.5, -0.5)
    df["c_ctrl"] = np.where(c.eq("A3"), 0.75, -0.25)                   # chosen vs mean of forced
    df["c_n1"] = np.select([c.eq("A0"), c.eq("A1")], [0.5, -0.5], 0.0)  # post vs pre
    df["c_n2"] = np.select([c.eq("A0"), c.eq("A1"), c.eq("A2")], [0.5, 0.5, -1.0], 0.0)
    df["timing"] = c.map(TIMING)
    return df
