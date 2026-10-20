"""Pure-function tests: downsampling, overfit rule, ranking, status mapping, flags, balance."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.services import assumptions as A
from app.services.eda import correlation, eda_summary, find_multicollinear_pairs
from app.services.modeling import Prepared, cap_rows, compute_learning_curve, rank_results, resolve_average
from app.services.preprocessing import class_balance
from app.services.telemetry import detect_gap_overfit, detect_overfit, detect_underfit
from app.utils.downsample import select_indices
from app.utils.safe_math import safe_div
from app.utils.sanitize import sanitize
from app.ws.radio import format_offset

S = Settings()


# --- downsampling --------------------------------------------------------
@pytest.mark.parametrize("n", [41, 100, 200, 1000, 1234])
def test_downsample_length_and_endpoints(n: int) -> None:
    idx = select_indices(n, 40)
    assert len(idx) <= 40
    assert idx[0] == 0 and idx[-1] == n - 1
    assert idx == sorted(set(idx))


def test_downsample_short_series_unchanged() -> None:
    assert select_indices(3, 40) == [0, 1, 2]
    assert select_indices(0, 40) == []


# --- overfit -------------------------------------------------------------
def test_detect_overfit_positive() -> None:
    train = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    val = [1.0, 0.8, 0.7, 0.75, 0.8, 0.85]
    result = detect_overfit(train, val, 3)
    assert result.detected and result.first_detected_epoch == 5 and result.label == "OVERFIT DETECTED"


def test_detect_overfit_negative_cases() -> None:
    assert not detect_overfit([1, 0.9, 0.8, 0.7], [1, 0.9, 0.8, 0.7]).detected
    assert not detect_overfit([1, 0.9, 0.8, 0.7, 0.6], [1, 1.1, 1.0, 1.1, 1.0]).detected  # not consecutive
    assert not detect_overfit([1.0, 0.9, 0.8], [1.0, 1.1, 1.2]).detected  # series shorter than window+1
    assert not detect_overfit([], []).detected
    assert detect_overfit([1, 0.9, 0.8, 0.7], [1, 1.1, 1.2, 1.3], window=3).first_detected_epoch == 3


# --- ranking / gap -------------------------------------------------------
def test_gap_to_top_and_tie_breaks() -> None:
    rows = {
        "a": {"model": "a", "f1": 0.874, "accuracy": 0.9, "train_time_sec": 2.0},
        "b": {"model": "b", "f1": 0.866, "accuracy": 0.9, "train_time_sec": 1.0},
        "c": {"model": "c", "f1": 0.866, "accuracy": 0.91, "train_time_sec": 5.0},
    }
    board = rank_results(rows)
    assert [r["model"] for r in board] == ["a", "c", "b"]
    assert board[0]["gap_to_top"] == 0.0 and board[0]["rank"] == 1
    assert board[1]["gap_to_top"] == pytest.approx(0.008, abs=1e-9)
    assert board[2]["gap_to_top"] == round(0.874 - 0.866, 6)
    assert rank_results({}) == []


def test_resolve_average() -> None:
    assert resolve_average(2, None) == ("binary", None)
    assert resolve_average(4, None)[0] == "weighted"
    assert resolve_average(4, "binary")[0] == "weighted" and resolve_average(4, "binary")[1]
    assert resolve_average(2, "macro") == ("macro", None)


# --- assumption status mapping ------------------------------------------
@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.5, "GREEN"), (0.0500001, "GREEN"), (0.05, "YELLOW"), (0.02, "YELLOW"), (0.0100001, "YELLOW"),
     (0.01, "RED"), (0.0, "RED")],
)
def test_p_value_status(p: float, expected: str) -> None:
    assert A.p_value_status(p, S) == expected


@pytest.mark.parametrize(
    ("dw", "expected"),
    [(2.0, "GREEN"), (1.5, "GREEN"), (2.5, "GREEN"), (1.2, "YELLOW"), (2.8, "YELLOW"), (1.0, "YELLOW"),
     (3.0, "YELLOW"), (0.5, "RED"), (3.4, "RED")],
)
def test_independence_status(dw: float, expected: str) -> None:
    assert A.independence_status(dw, S) == expected


def test_overall_status_is_worst() -> None:
    assert A.overall_status(["GREEN", "GREEN"]) == "GREEN"
    assert A.overall_status(["GREEN", "YELLOW", "GREEN"]) == "YELLOW"
    assert A.overall_status(["YELLOW", "RED", "GREEN"]) == "RED"
    assert A.result_for("GREEN") == "PASS" and A.result_for("YELLOW") == "FAIL"


def _assump_df(n: int = 300, **extra: object) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": 2 * x1 - x2 + rng.normal(size=n)})
    for key, value in extra.items():
        df[key] = value
    return df


def test_assumptions_happy_path_and_edge_cases() -> None:
    out = A.run_assumptions(_assump_df(), "y", S)
    assert out["normality"]["result"] == "PASS" and "linearity" not in out
    assert out["overall_status"] in {"GREEN", "YELLOW", "RED"}
    # perfectly collinear + constant column are dropped and reported
    df = _assump_df()
    df["dup"] = df["x1"] * 2
    df["const"] = 1.0
    out = A.run_assumptions(df, "y", S)
    assert out["dropped_columns"]["constant"] == ["const"]
    assert out["dropped_columns"]["collinear"]
    assert out["homoscedasticity"]["result"] in {"PASS", "FAIL"}


def test_assumptions_skipped_never_raises() -> None:
    no_numeric = pd.DataFrame({"c": list("abcdef"), "y": list("ababab")})
    out = A.run_assumptions(no_numeric, "y", S)
    assert all(out[k]["result"] == "SKIPPED" and out[k]["status"] == "YELLOW"
               for k in ("normality", "independence", "homoscedasticity"))
    tiny = pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
    assert A.run_assumptions(tiny, "y", S)["normality"]["result"] == "SKIPPED"
    constant = pd.DataFrame({"x": [1.0] * 10, "y": [0, 1] * 5})
    assert A.run_assumptions(constant, "y", S)["overall_status"] == "YELLOW"


def test_categorical_target_marks_residual_tests_not_applicable() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    df = pd.DataFrame({"x": x, "y": (x + rng.normal(size=400) > 0).astype(int)})
    out = A.run_assumptions(df, "y", S)
    assert out["normality"]["result"] == "N/A" and out["homoscedasticity"]["result"] == "N/A"
    assert out["independence"]["result"] in {"PASS", "FAIL"}
    assert out["overall_status"] == out["independence"]["status"]
    # an explicit column still gets a real Shapiro-Wilk
    assert A.run_assumptions(df, "y", S, column="x")["normality"]["result"] in {"PASS", "FAIL"}


def test_shapiro_subsamples_large_samples() -> None:
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(size=n)})
    out = A.run_assumptions(df, "y", S)
    assert "Subsampled to 5000" in out["normality"]["detail"]
    assert "tree-based" in out["normality"]["detail"]
    single = A.run_assumptions(df, "y", S, column="x")
    assert "column 'x'" in single["normality"]["detail"]


# --- correlation flags ---------------------------------------------------
def test_flags_are_upper_triangle_unique_sorted() -> None:
    cols = ["a", "b", "c"]
    m = [[1.0, 0.9, -0.95], [0.9, 1.0, 0.1], [-0.95, 0.1, 1.0]]
    flags = find_multicollinear_pairs(cols, m, 0.85)
    assert [(f["feature_a"], f["feature_b"]) for f in flags] == [("a", "c"), ("a", "b")]
    assert all(f["feature_a"] != f["feature_b"] for f in flags)
    assert find_multicollinear_pairs(cols, [[1, 0.85, 0], [0.85, 1, 0], [0, 0, 1]], 0.85) == []  # strict >


def test_correlation_edge_cases() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4.0], "b": [2, 4, 6, 8.0], "k": [1, 1, 1, 1.0], "y": [0, 1, 0, 1]})
    out = correlation(df, "y", 0.85)
    assert out["matrix"][0][2] is None  # constant column -> null
    assert [(f["feature_a"], f["feature_b"]) for f in out["multicollinearity_flags"]] == [("a", "b")]
    assert correlation(df[["a", "y"]], "y", 0.85)["matrix"] == [[1.0]]
    assert correlation(df[["y"]], "y", 0.85)["matrix"] == []


def test_eda_summary_handles_constant_and_all_nan() -> None:
    df = pd.DataFrame({"c": [5.0] * 4, "n": [np.nan] * 4, "x": [1.0, 2.0, 3.0, np.nan], "y": [0, 1, 0, 1]})
    out = eda_summary(df, "y")["features"]
    assert out["c"]["std"] == 0.0 and out["c"]["constant"]
    assert out["n"]["mean"] is None and out["n"]["count"] == 0
    assert out["x"]["null_count"] == 1 and out["x"]["max"] == 3.0
    assert "y" not in out


# --- class balance -------------------------------------------------------
def test_imbalance_threshold_boundary() -> None:
    at = class_balance(pd.Series(["a"] * 7 + ["b"] * 3), "t", 0.30)
    assert at["minority_ratio"] == 0.3 and at["imbalance_warning"] is False and at["severity"] == "NONE"
    below = class_balance(pd.Series(["a"] * 71 + ["b"] * 29), "t", 0.30)
    assert below["imbalance_warning"] is True and below["severity"] == "STRICT"
    assert "29.00%" in below["message"]


def test_class_balance_single_multi_and_nan() -> None:
    single = class_balance(pd.Series(["a"] * 5), "t", 0.30)
    assert single["single_class"] and not single["imbalance_warning"]
    multi = class_balance(pd.Series(["a"] * 80 + ["b"] * 15 + ["c"] * 5), "t", 0.30)
    assert multi["minority_class"] == "c" and multi["imbalance_warning"]
    assert class_balance(pd.Series(["a", "b", None, None]), "t", 0.3)["total"] == 2


# --- helpers -------------------------------------------------------------
def test_sanitize_and_safe_div_and_offset() -> None:
    out = sanitize({"a": np.float64("nan"), "b": np.inf, "c": np.int64(3), "d": [np.float32(1.5), -np.inf], "e": pd.NaT})
    assert out == {"a": None, "b": None, "c": 3, "d": [1.5, None], "e": None}
    assert safe_div(1, 0) == 0.0 and safe_div(1, 0, 9.0) == 9.0 and safe_div(1, 4) == 0.25
    assert not math.isnan(safe_div(float("nan"), 2))
    assert format_offset(75) == "T+01:15" and format_offset(3725) == "T+01:02:05" and format_offset(-3) == "T+00:00"


def test_cap_rows_keeps_class_proportions() -> None:
    y = np.array([0] * 900 + [1] * 100)
    x = np.arange(1000).reshape(-1, 1).astype(float)
    xs, ys, capped = cap_rows(x, y, 200, 0)
    assert capped and len(ys) == 200 and int(ys.sum()) == 20
    assert np.all(x[xs[:, 0].astype(int), 0] == xs[:, 0])  # rows stay aligned with their labels
    assert np.array_equal(ys, y[xs[:, 0].astype(int)])
    same = cap_rows(x, y, 5000, 0)
    assert same[2] is False and len(same[1]) == 1000


def test_detect_underfit_uses_chance_adjusted_skill() -> None:
    chance = detect_underfit(0.5, 2, 0.2)  # e.g. always predicting the majority class
    assert chance.detected and chance.skill == 0.0 and chance.label == "UNDERFIT DETECTED"
    assert detect_underfit(0.58, 2, 0.2).detected  # skill 0.16
    assert not detect_underfit(0.65, 2, 0.2).detected  # skill 0.30
    assert detect_underfit(0.40, 3, 0.2).detected and not detect_underfit(0.50, 3, 0.2).detected
    assert detect_underfit(0.95, 2, 0.2).label is None


def test_detect_gap_overfit() -> None:
    assert detect_gap_overfit(1.0, 0.80, 0.10)  # memorised the training rows
    assert not detect_gap_overfit(0.88, 0.85, 0.10)
    assert not detect_gap_overfit(0.80, 0.85, 0.10)  # test better than train is not overfit


def test_learning_curve_shapes_and_edge_cases() -> None:
    from app.services.model_registry import resolve_model

    rng = np.random.default_rng(3)
    n = 600
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    prepared = Prepared(X=pd.DataFrame(x, columns=["a", "b", "c"]), y=y, class_labels=["0", "1"], exclusions={})
    fit_idx, val_idx = np.arange(0, 450), np.arange(450, n)
    lc = compute_learning_curve(
        resolve_model("logistic_regression"), prepared, fit_idx, val_idx, scale=True, handle_imbalance="none", settings=S
    )
    assert lc is not None and lc["metric"] == "balanced_accuracy"
    assert lc["train_sizes"] == sorted(lc["train_sizes"]) and lc["train_sizes"][-1] == 450
    assert len(lc["train_sizes"]) == len(lc["train_score"]) == len(lc["val_score"]) >= 2
    assert lc["val_score"][-1] > 0.7  # real signal is learned as data grows
    # too little data for a curve -> None rather than an error
    tiny = compute_learning_curve(
        resolve_model("logistic_regression"), prepared, np.arange(20), val_idx, scale=True,
        handle_imbalance="none", settings=S,
    )
    assert tiny is None
    # a spent time budget still yields at least the two smallest points
    quick = S.model_copy(update={"learning_curve_time_budget_sec": 0.0})
    short = compute_learning_curve(
        resolve_model("decision_tree"), prepared, fit_idx, val_idx, scale=True, handle_imbalance="none", settings=quick
    )
    assert short is not None and len(short["train_sizes"]) == 2
