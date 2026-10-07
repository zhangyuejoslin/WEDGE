"""Steps 6-9: the four validation gates, run in order; all must pass.

Requires numpy, scipy, scikit-learn.
Every gate returns (passed: bool, info: dict); info goes into the ledger verbatim.
"""
import numpy as np
from scipy.stats import spearmanr, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sandbox import call_with_timeout

SEED = 0
MIN_COVERAGE = 0.95
RHO_MAX = 0.85
TRAIN_GAP_PP = 10.0
HOLDOUT_GAP_PP = 5.0
P_MAX = 0.05
MIN_GROUP = 5
MIN_DELTA_AUC = 0.01


def gate1_coverage(fn, records):
    """Compute the feature on every episode; >=95% valid, non-constant; median-fill.
    Fast-fail: 3 timeouts, or first 10 episodes all invalid, aborts immediately —
    a pathological candidate must not cost 1s x 655 episodes."""
    vals, errs, timeouts = [], [], 0
    for i, ep in enumerate(records):
        ok, v = call_with_timeout(fn, ep, seconds=1)
        if not ok and v == "timeout":
            timeouts += 1
            if timeouts >= 3:
                return False, {"coverage": 0.0, "reason": "code failed to execute (timeout)"}, None
        if ok:
            try:
                v = float(v)
            except (TypeError, ValueError):
                ok, v = False, "non-numeric return"
        if ok and np.isfinite(v):
            vals.append(v)
        else:
            vals.append(None)
            errs.append(v if isinstance(v, str) else "nan/inf")
        if i == 9 and len(errs) == 10:
            return False, {"coverage": 0.0,
                           "reason": f"unknown field / code crashed ({errs[0]})"}, None
    n = len(records)
    valid = [v for v in vals if v is not None]
    cov = len(valid) / n
    info = {"coverage": round(cov, 3),
            "errors": dict(zip(*np.unique(errs, return_counts=True))) if errs else {}}
    info["errors"] = {k: int(v) for k, v in info["errors"].items()}
    if cov < MIN_COVERAGE:
        return False, {**info, "reason": "unknown field / code crashed (coverage<95%)"}, None
    if len(set(valid)) == 1:
        return False, {**info, "reason": "constant feature"}, None
    med = float(np.median(valid))
    filled = np.array([med if v is None else v for v in vals], dtype=float)
    return True, info, filled


def gate2_redundancy(vals, library):
    """Spearman vs every library feature (incl. the 6 baselines); any |rho|>=0.85 rejects."""
    worst_name, worst_rho = None, 0.0
    for name, lv in library.items():
        rho = spearmanr(vals, lv).statistic
        if np.isnan(rho):
            rho = 0.0
        if abs(rho) > abs(worst_rho):
            worst_name, worst_rho = name, rho
    info = {"max_abs_spearman": round(float(worst_rho), 3), "vs": worst_name}
    if abs(worst_rho) >= RHO_MAX:
        return False, {**info, "reason": f"redundant with {worst_name} (|rho|={abs(worst_rho):.2f})"}
    return True, info


def _two_prop_z(s1, n1, s2, n2):
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def gate3_predictive(vals, succ):
    """70/30 split (fixed seed). Train: bottom vs top tertile SR gap >=10pp, p<0.05.
    Holdout (same thresholds): same direction, gap >=5pp. Any group <5 rejects."""
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(vals))
    cut = int(0.7 * len(vals))
    tr, ho = idx[:cut], idx[cut:]

    q33, q66 = np.percentile(vals[tr], [100 / 3, 200 / 3])
    if q33 != q66:
        def groups(ii):
            return ii[vals[ii] <= q33], ii[vals[ii] >= q66]
    else:
        # boolean / heavily-tied feature: binary split at the tied value, picking
        # the side assignment that leaves both groups non-degenerate
        lo_le = (vals[tr] <= q33).sum()
        if 0 < lo_le < len(tr):
            def groups(ii):
                return ii[vals[ii] <= q33], ii[vals[ii] > q33]
        elif 0 < (vals[tr] < q33).sum() < len(tr):
            def groups(ii):
                return ii[vals[ii] < q33], ii[vals[ii] >= q33]
        else:
            return False, {"reason": "values too concentrated to split"}

    lo, hi = groups(tr)
    if len(lo) < MIN_GROUP or len(hi) < MIN_GROUP:
        return False, {"reason": f"train groups too small (lo={len(lo)}, hi={len(hi)})"}
    sr_lo, sr_hi = succ[lo].mean(), succ[hi].mean()
    gap = 100 * (sr_lo - sr_hi)
    z, p = _two_prop_z(succ[lo].sum(), len(lo), succ[hi].sum(), len(hi))
    info = {"train": {"sr_low_tertile": round(100 * sr_lo, 1), "sr_high_tertile": round(100 * sr_hi, 1),
                      "gap_pp": round(gap, 1), "z": round(z, 2), "p": round(p, 4),
                      "n_lo": int(len(lo)), "n_hi": int(len(hi))}}
    if abs(gap) < TRAIN_GAP_PP or p >= P_MAX:
        return False, {**info, "reason": f"train gap/significance insufficient (gap={gap:.1f}pp p={p:.3f})"}

    lo, hi = groups(ho)
    if len(lo) < MIN_GROUP or len(hi) < MIN_GROUP:
        return False, {**info, "reason": f"holdout groups too small (lo={len(lo)}, hi={len(hi)})"}
    hgap = 100 * (succ[lo].mean() - succ[hi].mean())
    info["holdout"] = {"gap_pp": round(hgap, 1), "n_lo": int(len(lo)), "n_hi": int(len(hi))}
    if np.sign(hgap) != np.sign(gap) or abs(hgap) < HOLDOUT_GAP_PP:
        return False, {**info, "reason": f"holdout did not replicate (gap={hgap:.1f}pp)"}
    return True, info


def gate4_incremental(vals, library, succ):
    """5-fold CV logistic AUC: library-only vs library+candidate; mean delta >=0.01.
    Empty library (from-scratch mode): the no-candidate model is the base rate, AUC 0.5."""
    X1 = np.column_stack(([*library.values(), vals]) if library else [vals])
    X0 = X1[:, :-1] if library else None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    d0, d1 = [], []
    for tr, te in skf.split(X1, succ):
        for X, out in ((X0, d0), (X1, d1)):
            if X is None:
                out.append(0.5)
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), succ[tr])
            out.append(roc_auc_score(succ[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    a0, a1 = float(np.mean(d0)), float(np.mean(d1))
    info = {"auc_library": round(a0, 4), "auc_with_candidate": round(a1, 4),
            "delta_auc": round(a1 - a0, 4)}
    if a1 - a0 < MIN_DELTA_AUC:
        return False, {**info, "reason": f"AUC gain insufficient ({a1 - a0:+.4f})"}
    return True, info
