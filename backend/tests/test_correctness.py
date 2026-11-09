"""Regression tests for calculation bugs: scores vs predictions, class weighting, positive class, target safety."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.config import Settings
from app.errors import BadRequestError
from app.services import analysis, inference_runtime as rt, transforms
from app.services.crossval import cv_pipeline
from app.services.model_registry import ThresholdedLinearRegression
from app.services.modeling import apply_imbalance, identify_exclusions, prepare_dataset
from app.services.preprocessing import clean_dataframe
from tests.conftest import make_df, ready_session
from tests.test_training import train

U = "/api/v1/sessions"


def _binary(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = ((x.a + 0.5 * x.b + rng.normal(scale=0.8, size=n)) > 0.6).astype(int).to_numpy()
    return x, y


@pytest.mark.parametrize("est", [SVC(), ThresholdedLinearRegression()])
def test_scores_agree_with_predictions(est) -> None:
    x, y = _binary()
    pipe = Pipeline([("s", StandardScaler()), ("clf", est)]).fit(x, y)
    pred = pipe.predict(x)
    dash, _ = analysis.score_matrix(pipe, x)
    exported = rt.class_scores(pipe, x)
    assert np.array_equal((dash[:, 1] >= 0.5).astype(int), pred)
    assert np.array_equal((exported[:, 1] >= 0.5).astype(int), pred)


def test_linear_regression_classifier_multiclass_is_one_vs_rest() -> None:
    # class 1 is the "middle" code but sits far from 0 and 2 in feature space: rounding a single regression fails here
    rng = np.random.default_rng(1)
    centres = {0: (-3, 0), 1: (0, 4), 2: (3, 0)}
    y = np.repeat([0, 1, 2], 100)
    x = np.vstack([rng.normal(centres[c], 0.5, size=(100, 2)) for c in (0, 1, 2)])
    model = ThresholdedLinearRegression().fit(x, y)
    assert (model.predict(x) == y).mean() > 0.95
    assert np.array_equal(model.predict(x), model.classes_[model.predict_proba(x).argmax(axis=1)])


def test_cv_and_tuning_pipelines_keep_class_weights_and_smote() -> None:
    from sklearn.compose import ColumnTransformer

    y = np.array([0] * 90 + [1] * 10)
    prep = ColumnTransformer([("num", "passthrough", ["a"])])
    _, params = cv_pipeline(prep, GradientBoostingClassifier(), y, 5, task="classification", n_classes=2,
                            handle_imbalance="class_weight", seed=0, apply_imbalance=apply_imbalance)
    w = params["clf__sample_weight"]
    assert w[y == 1][0] > w[y == 0][0] * 5  # minority rows weigh ~9x more
    pipe, params = cv_pipeline(prep, GradientBoostingClassifier(), y, 5, task="classification", n_classes=2,
                               handle_imbalance="smote", seed=0, apply_imbalance=apply_imbalance)
    assert "smote" in dict(pipe.steps) and params == {}


def test_torch_mlp_uses_class_weights() -> None:
    pytest.importorskip("torch")
    from app.services.torch_models import TorchMLPClassifier

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 3)).astype("float32")
    y = (x[:, 0] + rng.normal(scale=1.5, size=600) > 2.2).astype(int)  # ~10 % positives, heavily overlapping
    plain = TorchMLPClassifier(max_epochs=40, random_state=0).fit(x, y)
    weights = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
    weighted = TorchMLPClassifier(max_epochs=40, random_state=0).fit(x, y, sample_weight=weights)
    assert weighted.predict(x).sum() > plain.predict(x).sum()


@pytest.mark.parametrize("op", [
    {"op": "log_transform", "columns": ["price"]},
    {"op": "rename", "mapping": {"price": "p2"}},
    {"op": "clip", "column": "price", "lower": 1, "upper": 2},
    {"op": "fill_constant", "columns": ["price"], "value": 0},
    {"op": "outliers", "columns": ["price"], "method": "iqr", "factor": 1.5, "action": "clip"},
])
def test_target_cannot_be_transformed(op) -> None:
    df = pd.DataFrame({"x": np.arange(20.0), "price": np.arange(1.0, 21.0)})
    with pytest.raises(BadRequestError):
        transforms.apply_operations(df, [op], {"price"})


def test_positive_class_default_is_minority_and_selectable() -> None:
    s = Settings()
    df = pd.DataFrame({"x": np.arange(100.0), "y": ["stayed"] * 80 + ["churned"] * 20})
    auto = prepare_dataset(df, "y", s)
    assert auto.class_labels == ["stayed", "churned"] and auto.y.sum() == 20
    chosen = prepare_dataset(df, "y", s, positive_class="stayed")
    assert chosen.class_labels == ["churned", "stayed"] and chosen.y.sum() == 80
    with pytest.raises(BadRequestError):
        prepare_dataset(df, "y", s, positive_class="nope")


def test_id_detection_keeps_real_numeric_features() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "income": rng.choice(np.arange(20_000, 200_000), 60, replace=False),  # unique per row but a real feature
        "id": rng.choice(np.arange(1, 99_999), 60, replace=False),  # random ids, named like one
        "row_no": np.arange(100, 160),  # consecutive run
        "CustomerID": [f"C{i}" for i in range(60)],
        "y": rng.integers(0, 2, 60),
    })
    assert sorted(identify_exclusions(df, "y")["id_like"]) == ["CustomerID", "id", "row_no"]


def test_interpolation_uses_time_gaps_and_never_fills_the_target() -> None:
    df = pd.DataFrame({
        "t": ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-03"],
        "v": [0.0, np.nan, 4.0, np.nan],
        "y": [1.0, 2.0, 3.0, np.nan],
    })
    out, report = clean_dataframe(df, "y", "impute", "interpolate", None, sort_by="t")
    assert out["y"].notna().all() and len(out) == 3  # the row with a missing target was dropped, not invented
    assert out.loc[out["t"] == "2024-01-02", "v"].item() == pytest.approx(1.0)  # 1 day of a 4-day gap


def test_knn_imputation_with_categorical_columns() -> None:
    df = make_df(rows=120, missing=8)
    out, _ = clean_dataframe(df, "churn", "impute", "knn", None)
    assert out["total_charges"].notna().all()


def test_api_positive_class_and_full_training_rows(client: TestClient) -> None:
    sid = ready_session(client)
    bal = client.get(f"{U}/{sid}/class-balance").json()
    minority = min(bal["classes"], key=lambda c: c["count"])["label"]
    assert bal["default_positive"] == minority
    job = train(client, sid, ["logistic_regression", "mlp"])
    assert job["result"]["positive_class"] == minority
    other = next(c["label"] for c in bal["classes"] if c["label"] != minority)
    job = train(client, sid, ["logistic_regression"], positive_class=other)
    assert job["result"]["positive_class"] == other
    assert client.post(f"{U}/{sid}/train", json={"models": ["logistic_regression"], "positive_class": "nope"}).status_code == 400


def test_iterative_models_are_refit_on_all_training_rows(client: TestClient) -> None:
    sid = ready_session(client)
    job = train(client, sid, ["mlp"])
    split = job["result"]["split"]
    session = client.app.state.store.get(sid)
    scaler = session.fitted["mlp"].pipeline.named_steps["prep"].named_transformers_["num"]
    assert int(scaler.n_samples_seen_[0] if np.ndim(scaler.n_samples_seen_) else scaler.n_samples_seen_) == split["train_rows"]
    assert session.telemetry["mlp"]["train_loss"]  # the telemetry curve is still recorded
