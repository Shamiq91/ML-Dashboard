"""Linear-regression assumption tests (normality, independence, homoscedasticity)."""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api import types as ptypes
from scipy import linalg, stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from app.config import Settings
from app.errors import BadRequestError
from app.services.eda import numeric_feature_columns
from app.utils.safe_math import round_or_none

_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}
_NOTE = " These are linear-regression assumptions; tree-based models do not require them."


# --- pure status mapping -------------------------------------------------
def p_value_status(p_value: float, settings: Settings) -> str:
    """GREEN p > 0.05, YELLOW 0.01 < p <= 0.05, RED p <= 0.01."""
    if p_value > settings.shapiro_alpha:
        return "GREEN"
    if p_value > settings.strict_alpha:
        return "YELLOW"
    return "RED"


def independence_status(dw: float, settings: Settings) -> str:
    """GREEN 1.5-2.5, YELLOW 1.0-1.5 or 2.5-3.0, RED otherwise."""
    if settings.dw_pass_low <= dw <= settings.dw_pass_high:
        return "GREEN"
    if settings.dw_warn_low <= dw <= settings.dw_warn_high:
        return "YELLOW"
    return "RED"


def overall_status(statuses: list[str]) -> str:
    """Worst status of the list (GREEN < YELLOW < RED)."""
    return max(statuses, key=lambda s: _RANK[s]) if statuses else "YELLOW"


def result_for(status: str) -> str:
    """PASS only for GREEN."""
    return "PASS" if status == "GREEN" else "FAIL"


def _skipped(test: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"test": test, "result": "SKIPPED", "status": "YELLOW", "detail": f"SKIPPED: {reason}.", **extra}


def _not_applicable(test: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"test": test, "result": "N/A", "status": "GREEN", "detail": f"N/A: {reason}", **extra}


def is_categorical_target(series: pd.Series, settings: Settings) -> bool:
    """Same rule as training: non-numeric, bool, or few distinct numeric values means classes."""
    if not ptypes.is_numeric_dtype(series) or ptypes.is_bool_dtype(series):
        return True
    return int(series.nunique()) <= settings.continuous_target_unique


_CATEGORICAL_REASON = (
    "the target is categorical, so OLS residuals are discrete by construction and cannot be "
    "normal or constant-variance. This does not affect the classifiers."
)


# --- design matrix -------------------------------------------------------
def _encode_target(series: pd.Series) -> np.ndarray:
    if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
        return series.astype("float64").to_numpy()
    return pd.Categorical(series.astype(str)).codes.astype("float64")


def _drop_collinear(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns that are linearly dependent (pivoted QR on centred data)."""
    if x.shape[1] <= 1:
        return x, []
    centred = x.to_numpy(dtype="float64") - x.to_numpy(dtype="float64").mean(axis=0)
    _, r, pivot = linalg.qr(centred, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    tol = max(centred.shape) * np.finfo(float).eps * (diag[0] if diag.size else 0.0)
    rank = int((diag > tol).sum())
    keep = sorted(int(i) for i in pivot[:rank])
    dropped = [x.columns[i] for i in range(x.shape[1]) if i not in keep]
    return x.iloc[:, keep], dropped


def _prepare(df: pd.DataFrame, target: str, settings: Settings) -> dict[str, Any]:
    features = numeric_feature_columns(df, target)
    info: dict[str, Any] = {"x": None, "y": None, "dropped": {"constant": [], "collinear": []}, "reason": None}
    if not features:
        info["reason"] = "no numeric features available"
        return info
    frame = df[features + [target]].copy()
    mask = frame[features].notna().all(axis=1) & frame[target].notna()
    frame = frame[mask]
    y = _encode_target(frame[target])
    x = frame[features].astype("float64")
    constant = [c for c in x.columns if x[c].nunique() <= 1]
    x = x.drop(columns=constant)
    info["dropped"]["constant"] = constant
    if x.shape[1] == 0:
        info["reason"] = "all numeric features are constant"
        return info
    x, collinear = _drop_collinear(x)
    info["dropped"]["collinear"] = collinear
    if len(x) < max(x.shape[1] + 2, 3):
        info["reason"] = f"too few rows ({len(x)}) for {x.shape[1]} predictor(s)"
        return info
    if np.ptp(y) == 0:
        info["reason"] = "target is constant"
        return info
    info["x"], info["y"] = x.reset_index(drop=True), y
    return info


# --- individual tests ----------------------------------------------------
def _shapiro(values: np.ndarray, basis: str, settings: Settings) -> dict[str, Any]:
    test = "Shapiro-Wilk"
    values = values[np.isfinite(values)]
    if values.size < 3:
        return _skipped(test, "at least 3 observations are required", statistic=None, p_value=None)
    if np.ptp(values) == 0:
        return _skipped(test, "all values are identical", statistic=None, p_value=None)
    note = ""
    if values.size > settings.shapiro_max_samples:
        rng = np.random.default_rng(settings.random_seed)
        values = rng.choice(values, settings.shapiro_max_samples, replace=False)
        note = f" Subsampled to {settings.shapiro_max_samples} points (Shapiro is unreliable on larger samples)."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value = stats.shapiro(values)
    status = p_value_status(float(p_value), settings)
    return {
        "test": test,
        "statistic": round_or_none(statistic),
        "p_value": round_or_none(p_value),
        "result": result_for(status),
        "status": status,
        "detail": f"Shapiro-Wilk on {basis}: W={statistic:.4f}, p={p_value:.4g}.{note}{_NOTE}",
    }


def _durbin_watson(resid: np.ndarray, settings: Settings) -> dict[str, Any]:
    dw = float(durbin_watson(resid))
    if not math.isfinite(dw):
        return _skipped("Durbin-Watson", "statistic is undefined for these residuals", statistic=None)
    status = independence_status(dw, settings)
    return {
        "test": "Durbin-Watson",
        "statistic": round_or_none(dw),
        "result": result_for(status),
        "status": status,
        "detail": (
            f"Durbin-Watson on OLS residuals: DW={dw:.3f} (pass range {settings.dw_pass_low}-{settings.dw_pass_high}). "
            f"Only meaningful if row order is meaningful (time-ordered data).{_NOTE}"
        ),
    }


def _breusch_pagan(resid: np.ndarray, exog: np.ndarray, settings: Settings) -> dict[str, Any]:
    test = "Breusch-Pagan"
    empty = {"lm_statistic": None, "lm_p_value": None, "f_statistic": None, "f_p_value": None}
    try:
        lm, lm_p, f_stat, f_p = het_breuschpagan(resid, exog)
    except Exception as exc:  # noqa: BLE001 - data-driven failure must not 500
        return _skipped(test, f"test could not run ({type(exc).__name__})", **empty)
    if not math.isfinite(float(lm_p)):
        return _skipped(test, "p-value is undefined", **empty)
    status = p_value_status(float(lm_p), settings)
    return {
        "test": test,
        "lm_statistic": round_or_none(lm),
        "lm_p_value": round_or_none(lm_p),
        "f_statistic": round_or_none(f_stat),
        "f_p_value": round_or_none(f_p),
        "result": result_for(status),
        "status": status,
        "detail": f"Breusch-Pagan on OLS residuals: LM={lm:.3f}, p={lm_p:.4g}.{_NOTE}",
    }


def run_assumptions(
    df: pd.DataFrame, target: str, settings: Settings, column: str | None = None, task: str | None = None
) -> dict[str, Any]:
    """Run all three tests on OLS residuals (optionally Shapiro on one raw ``column``)."""
    column_values: np.ndarray | None = None
    if column is not None:
        if column not in df.columns:
            raise BadRequestError(f"Column '{column}' does not exist.", error_code="UNKNOWN_COLUMNS")
        if not (ptypes.is_numeric_dtype(df[column]) and not ptypes.is_bool_dtype(df[column])):
            raise BadRequestError(f"Column '{column}' is not numeric.", error_code="COLUMN_NOT_NUMERIC")
        column_values = df[column].dropna().astype("float64").to_numpy()

    design = _prepare(df, target, settings)
    reason = design["reason"]
    if reason:
        skipped = {
            "normality": _skipped("Shapiro-Wilk", reason, statistic=None, p_value=None),
            "independence": _skipped("Durbin-Watson", reason, statistic=None),
            "homoscedasticity": _skipped(
                "Breusch-Pagan", reason, lm_statistic=None, lm_p_value=None, f_statistic=None, f_p_value=None
            ),
        }
        if column_values is not None:
            skipped["normality"] = _shapiro(column_values, f"column '{column}'", settings)
        payload = skipped
        n_obs = 0
    else:
        x, y = design["x"], design["y"]
        exog = sm.add_constant(x.to_numpy(), has_constant="add")
        fit = sm.OLS(y, exog).fit()
        resid = np.asarray(fit.resid, dtype="float64")
        basis = f"OLS residuals (target ~ {x.shape[1]} numeric feature(s), n={len(x)})"
        categorical = task == "classification" if task else is_categorical_target(df[target], settings)
        if column_values is not None:
            normality = _shapiro(column_values, f"column '{column}'", settings)
        elif categorical:
            normality = _not_applicable("Shapiro-Wilk", _CATEGORICAL_REASON, statistic=None, p_value=None)
        else:
            normality = _shapiro(resid, basis, settings)
        homoscedasticity = (
            _not_applicable(
                "Breusch-Pagan", _CATEGORICAL_REASON,
                lm_statistic=None, lm_p_value=None, f_statistic=None, f_p_value=None,
            )
            if categorical
            else _breusch_pagan(resid, exog, settings)
        )
        payload = {
            "normality": normality,
            "independence": _durbin_watson(resid, settings),
            "homoscedasticity": homoscedasticity,
        }
        n_obs = int(len(x))
    payload["overall_status"] = overall_status(
        [payload[k]["status"] for k in ("normality", "independence", "homoscedasticity")
         if payload[k]["result"] != "N/A"]
    )
    payload["dropped_columns"] = design["dropped"]
    payload["n_observations"] = n_obs
    return payload
