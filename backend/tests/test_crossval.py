"""Cross-validation during training: reported per model, leak-free, and never able to fail a model."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import make_df, ready_session
from tests.test_training import train

U = "/api/v1/sessions"


def test_cv_is_reported_by_default_with_mean_std_and_folds(client: TestClient) -> None:
    sid = ready_session(client)
    body = train(client, sid, ["logistic_regression", "decision_tree"])
    rows = {r["abbr"]: r for r in body["result"]["leaderboard"]}
    for abbr in ("LR", "DT"):
        r = rows[abbr]
        cv = r["cv"]
        assert cv["folds"] == 5 and cv["metric"] == "f1"
        assert r["cv_score"] == cv["score"] and r["cv_std"] == cv["score_std"]
        assert set(cv["metrics"]) >= {"f1", "accuracy", "balanced_accuracy", "precision", "recall", "roc_auc"}
        folds = cv["metrics"]["f1"]["folds"]
        assert len(folds) == 5 and abs(np.mean(folds) - cv["score"]) < 1e-5
        assert cv["score_sem"] == round(cv["score_std"] / np.sqrt(5), 6)
        assert 0 <= cv["score"] <= 1


def test_cv_can_be_switched_off_or_resized(client: TestClient) -> None:
    sid = ready_session(client)
    off = train(client, sid, ["logistic_regression"], cross_validate=False)
    assert off["result"]["leaderboard"][0]["cv"] is None and off["result"]["leaderboard"][0]["cv_score"] is None
    three = train(client, sid, ["logistic_regression"], cv_folds=3)
    assert three["result"]["leaderboard"][0]["cv"]["folds"] == 3
    assert client.post(f"{U}/{sid}/train", json={"models": ["logistic_regression"], "cv_folds": 2}).status_code == 422


def test_cv_regression_reports_rmse_in_the_targets_units(client: TestClient) -> None:
    df = make_df(300)
    sid = ready_session(client, df, clean=True)
    assert client.post(f"{U}/{sid}/target", json={"target_column": "monthly_charges", "task": "regression"}).status_code == 200
    client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "median"})
    body = train(client, sid, ["linear_regression"])
    row = body["result"]["leaderboard"][0]
    cv = row["cv"]
    assert cv["metric"] == "r2" and set(cv["metrics"]) == {"r2", "rmse", "mae"}
    # RMSE on the original scale is comparable to the target's spread, not ~1 (the standardised scale)
    assert cv["metrics"]["rmse"]["mean"] > 1.0
    assert abs(cv["metrics"]["rmse"]["mean"] - row["rmse"]) < 0.6 * row["rmse"]


def test_cv_never_uses_test_rows(client: TestClient) -> None:
    sid = ready_session(client, make_df(300))
    body = train(client, sid, ["logistic_regression"])
    cv = body["result"]["leaderboard"][0]["cv"]
    assert cv["rows_used"] <= body["result"]["split"]["train_rows"]


def test_cv_with_smote_and_class_weights_runs(client: TestClient) -> None:
    sid = ready_session(client)
    for mode in ("smote", "class_weight"):
        body = train(client, sid, ["logistic_regression"], handle_imbalance=mode)
        assert body["result"]["leaderboard"][0]["cv"] is not None, mode


def test_cv_is_saved_with_the_run(client: TestClient) -> None:
    sid = ready_session(client)
    body = train(client, sid, ["logistic_regression"])
    run = client.get(f"/api/v1/runs/{body['run_id']}").json()
    assert run["leaderboard"][0]["cv_score"] is not None
    assert run["settings"]["cross_validate"] is True and run["settings"]["cv_folds"] == 5


def test_logs_endpoint_accepts_tune_and_cv_stages(client: TestClient) -> None:
    sid = ready_session(client)
    train(client, sid, ["logistic_regression"], tune=True, tune_trials=3)
    logs = client.get(f"{U}/{sid}/logs")
    assert logs.status_code == 200
    stages = {m["stage"] for m in logs.json()["logs"]}
    assert {"CV", "TUNE"} <= stages
