"""Feature selection helpers: how much each column tells us about the target, what is redundant, what to transform."""
from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from pandas.api import types as ptypes
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

from app.config import Settings
from app.services.ingestion import column_kind
from app.services.modeling import identify_exclusions
from app.services.tasks import is_regression

MAX_ROWS = 5000
HIGH_MISSING = 0.4
REDUNDANT_R = 0.9
WEAK_MI_REL = 0.03
SKEW_LIMIT = 1.5
OUTLIER_SHARE = 0.05


def _numeric(s: pd.Series) -> bool:
    return bool(ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s))


def _mutual_info(x: pd.DataFrame, y: pd.Series, regression: bool, seed: int) -> dict[str, float]:
    """Mutual information of each column with the target on a sample (missing values filled only for this estimate)."""
    keep = y.notna()
    x, y = x[keep], y[keep]
    if len(y) < 10 or x.shape[1] == 0:
        return {c: 0.0 for c in x.columns}
    enc = pd.DataFrame(index=x.index)
    discrete = []
    for c in x.columns:
        s = x[c]
        if _numeric(s) or ptypes.is_bool_dtype(s):
            v = s.astype("float64")
            enc[c] = v.fillna(v.median() if v.notna().any() else 0.0)
            discrete.append(False)
        elif ptypes.is_datetime64_any_dtype(s):
            enc[c] = ((s - pd.Timestamp("1970-01-01")).dt.total_seconds()).fillna(0.0)
            discrete.append(False)
        else:
            enc[c] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(s.astype("object").where(s.notna(), "(missing)").astype(str).to_frame()).ravel()
            discrete.append(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if regression:
            mi = mutual_info_regression(enc.to_numpy(), y.astype("float64").to_numpy(), discrete_features=discrete, random_state=seed)
        else:
            codes, _ = pd.factorize(y, sort=True)
            mi = mutual_info_classif(enc.to_numpy(), codes, discrete_features=discrete, random_state=seed)
    return {c: float(max(v, 0.0)) for c, v in zip(x.columns, mi)}


def feature_report(df: pd.DataFrame, target: str, task: str, settings: Settings) -> dict[str, Any]:
    """Per-feature table, redundant pairs and ready-to-apply suggestions (each carries the transform it would run)."""
    seed = settings.random_seed
    regression = is_regression(task)
    sample = df if len(df) <= MAX_ROWS else df.sample(MAX_ROWS, random_state=seed)
    cols = [c for c in df.columns if c != target]
    excl = identify_exclusions(df, target)
    scored = [c for c in cols if c not in excl["constant"] and c not in excl["id_like"]]  # an ID would score as 'perfectly informative'
    mi = _mutual_info(sample[scored], sample[target], regression, seed)
    top = max(mi.values(), default=0.0) or 1.0
    ranked = sorted((c for c in cols if c not in excl["constant"] and c not in excl["id_like"]), key=lambda c: mi.get(c, 0.0), reverse=True)
    rank = {c: i + 1 for i, c in enumerate(ranked)}

    features: list[dict[str, Any]] = []
    for c in cols:
        s = df[c]
        numeric = _numeric(s)
        row: dict[str, Any] = {
            "name": c, "kind": column_kind(s), "missing_pct": round(float(s.isna().mean()) * 100, 2), "unique": int(s.nunique(dropna=True)),
            "mi": round(mi.get(c, 0.0), 5), "mi_rel": round(mi.get(c, 0.0) / top, 4), "rank": rank.get(c),
            "constant": c in excl["constant"], "id_like": c in excl["id_like"], "skew": None, "outlier_pct": None, "min": None,
        }
        if numeric and s.notna().sum() > 3:
            v = s.dropna().astype("float64")
            row["skew"] = round(float(v.skew()), 3) if v.nunique() > 2 else 0.0
            row["min"] = round(float(v.min()), 6)
            q1, q3 = v.quantile([0.25, 0.75])
            iqr = q3 - q1
            row["outlier_pct"] = round(float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()) * 100, 2) if iqr > 0 else 0.0
        features.append(row)
    by_name = {f["name"]: f for f in features}

    pairs: list[dict[str, Any]] = []
    num_cols = [c for c in cols if _numeric(sample[c]) and sample[c].nunique() > 1]
    if len(num_cols) >= 2:
        corr = sample[num_cols].corr()
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) >= REDUNDANT_R:
                    pairs.append({"a": a, "b": b, "r": round(float(r), 3)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)

    suggestions: list[dict[str, Any]] = []
    dropped: set[str] = set()

    def add(action: str, columns: list[str], reason: str, severity: str, op: dict[str, Any]) -> None:
        suggestions.append({"id": f"{action}:{','.join(columns)}", "action": action, "columns": columns, "reason": reason, "severity": severity, "op": op})

    for f in features:
        if f["missing_pct"] > HIGH_MISSING * 100 and not f["constant"]:
            add("drop", [f["name"]], f"{f['missing_pct']:.0f}% of its values are missing, so imputing would mostly invent data.", "high", {"op": "drop_columns", "columns": [f["name"]]})
            dropped.add(f["name"])
    for p in pairs:
        if p["a"] in dropped or p["b"] in dropped:
            continue
        weaker = p["a"] if mi.get(p["a"], 0) < mi.get(p["b"], 0) else p["b"]
        keeper = p["b"] if weaker == p["a"] else p["a"]
        add("drop", [weaker], f"Almost duplicates '{keeper}' (correlation {p['r']:+.2f}) and tells the target less. Redundant columns add noise and make linear models unstable.", "medium", {"op": "drop_columns", "columns": [weaker]})
        dropped.add(weaker)
    n_ranked = len(ranked)
    weak = [c for c in ranked if c not in dropped and by_name[c]["mi_rel"] < WEAK_MI_REL and by_name[c]["rank"] > max(3, n_ranked // 2)]
    if weak:
        add("drop", weak, f"{len(weak)} column(s) share almost no information with the target (relative score under {WEAK_MI_REL:.2f}). Dropping weak columns can reduce overfitting and speed up training; check the list before applying.", "low", {"op": "drop_columns", "columns": weak})
        dropped.update(weak)
    for f in features:
        c = f["name"]
        if c in dropped or f["constant"] or f["id_like"]:
            continue
        if f["skew"] is not None and abs(f["skew"]) > SKEW_LIMIT and f["min"] is not None and f["min"] >= 0 and f["unique"] > 10:
            add("transform", [c], f"Strongly skewed (skew {f['skew']:+.1f}). A log transform makes it closer to symmetric, which helps linear and distance-based models.", "medium", {"op": "log_transform", "columns": [c]})
        elif f["outlier_pct"] is not None and f["outlier_pct"] > OUTLIER_SHARE * 100:
            add("transform", [c], f"{f['outlier_pct']:.0f}% of values are outliers (beyond 1.5 x IQR). Clipping them stops a few extreme rows from dominating.", "low", {"op": "outliers", "columns": [c], "method": "iqr", "factor": 3.0, "action": "clip"})
    order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: order[s["severity"]])
    return {
        "target": target, "task": task, "rows_sampled": int(len(sample)), "features": sorted(features, key=lambda f: -f["mi"]),
        "redundant_pairs": pairs, "suggestions": suggestions,
        "auto_excluded": excl,
        "notes": ["Relevance is mutual information with the target on a sample: it catches non-linear links that correlation misses, but it ranks single columns, so two weak columns can still help together."],
    }
