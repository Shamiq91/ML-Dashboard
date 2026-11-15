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
REDUNDANT_R = 0.9


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
    """Per-feature relevance table (mutual information with the target, skew, outliers) and redundant numeric pairs."""
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
            "constant": c in excl["constant"], "id_like": c in excl["id_like"], "skew": None, "outlier_pct": None,
        }
        if numeric and s.notna().sum() > 3:
            v = s.dropna().astype("float64")
            row["skew"] = round(float(v.skew()), 3) if v.nunique() > 2 else 0.0
            q1, q3 = v.quantile([0.25, 0.75])
            iqr = q3 - q1
            row["outlier_pct"] = round(float(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()) * 100, 2) if iqr > 0 else 0.0
        features.append(row)

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

    return {
        "target": target, "task": task, "rows_sampled": int(len(sample)), "features": sorted(features, key=lambda f: -f["mi"]),
        "redundant_pairs": pairs,
        "auto_excluded": excl,
        "notes": ["Relevance is mutual information with the target on a sample: it catches non-linear links that correlation misses, but it ranks single columns, so two weak columns can still help together."],
    }
