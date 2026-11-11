"""Model lab: catalog, per-model fits and diagnostics, sweeps, learning curves and validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_df, ready_session, upload

U = "/api/v1/sessions"
SUPERVISED = ["logistic_regression", "knn", "xgboost", "decision_tree", "random_forest", "svm"]


def fit(client: TestClient, sid: str, model: str, **body: object) -> dict:
    r = client.post(f"{U}/{sid}/lab/{model}/fit", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def regression_session(client: TestClient) -> str:
    df = make_df(missing=0).drop(columns=["churn"])
    sid = upload(client, df)
    r = client.post(f"{U}/{sid}/target", json={"target_column": "total_charges", "task": "regression"})
    assert r.status_code == 200 and r.json()["task"] == "regression"
    return sid


def test_catalog_lists_eight_models(client: TestClient) -> None:
    body = client.get("/api/v1/lab/models").json()
    keys = [m["key"] for m in body["models"]]
    assert keys == ["linear_regression", "logistic_regression", "knn", "xgboost", "decision_tree", "random_forest", "kmeans", "svm"]
    knn = next(m for m in body["models"] if m["key"] == "knn")
    k = next(p for p in knn["params"] if p["name"] == "n_neighbors")
    assert k["kind"] == "int" and k["default"] == 15 and k["sweep"] and k["help"]


@pytest.mark.parametrize("model", SUPERVISED)
def test_classification_fit_has_metrics_and_charts(client: TestClient, model: str) -> None:
    sid = ready_session(client)
    b = fit(client, sid, model, params={"n_estimators": 40} if model in ("xgboost", "random_forest") else {})
    assert b["task"] == "classification" and b["metric"] == "F1"
    assert 0 <= b["test"]["f1"] <= 1 and 0 <= b["train"]["accuracy"] <= 1
    assert b["cv"]["mean"] is not None and len(b["cv"]["folds"]) == 5
    perf = b["performance"]
    assert perf["binary"] and sum(map(sum, perf["confusion"])) == b["data"]["test_rows"]
    assert b["notes"] and "customer_id" not in b["data"]["features_used"]
    charts = b["charts"]
    expected = {
        "logistic_regression": {"logistic", "boundary"}, "knn": {"knn", "boundary"},
        "xgboost": {"loss", "importance", "tree"}, "decision_tree": {"tree", "rules", "pruning", "importance"},
        "random_forest": {"growth", "importance", "agreement", "tree"}, "svm": {"svm", "boundary"},
    }[model]
    assert expected <= set(charts)


def test_threshold_sweep_is_consistent(client: TestClient) -> None:
    sid = ready_session(client)
    t = fit(client, sid, "logistic_regression")["performance"]["thresholds"]
    assert len(t["grid"]) == 101 and t["recall"][0] == 1.0
    n = t["tp"][50] + t["fp"][50] + t["tn"][50] + t["fn"][50]
    assert all(t["tp"][i] + t["fp"][i] + t["tn"][i] + t["fn"][i] == n for i in range(101))


def test_logistic_penalty_none_gives_inference_and_l1_zeros(client: TestClient) -> None:
    sid = ready_session(client)
    plain = fit(client, sid, "logistic_regression", params={"penalty": "none"})["charts"]["logistic"]
    assert plain["inference"] and plain["coefficients"][0]["p"] is not None
    lasso = fit(client, sid, "logistic_regression", params={"penalty": "l1", "C": 0.01})["charts"]["logistic"]
    assert lasso["n_zero"] > 0 and len(lasso["path"]["points"]) == 13


def test_decision_tree_structure_and_readable_rules(client: TestClient) -> None:
    sid = ready_session(client)
    b = fit(client, sid, "decision_tree", params={"max_depth": 3}, tree_depth=2)
    tree = b["charts"]["tree"]
    assert tree["depth"] <= 3 and tree["root"]["samples"] == b["data"]["train_rows"]
    assert len(tree["root"]["children"]) == 2 and tree["root"]["children"][0]["children"][0].get("hidden") is not None
    rule = b["charts"]["rules"][0]
    assert rule["conditions"] and rule["pred"] in ("Yes", "No")
    assert b["charts"]["pruning"]["points"][0]["leaves"] >= b["charts"]["pruning"]["points"][-1]["leaves"]


def test_tree_thresholds_are_in_original_units(client: TestClient) -> None:
    sid = ready_session(client)
    scaled = fit(client, sid, "decision_tree", params={"max_depth": 1}, scale=True)["charts"]["tree"]["root"]["yes"]
    raw = fit(client, sid, "decision_tree", params={"max_depth": 1}, scale=False)["charts"]["tree"]["root"]["yes"]
    assert scaled.split("<=")[0] == raw.split("<=")[0]
    assert abs(float(scaled.split("<=")[1]) - float(raw.split("<=")[1])) < 1e-2


def test_xgboost_loss_curves_and_tree_index(client: TestClient) -> None:
    sid = ready_session(client)
    b = fit(client, sid, "xgboost", params={"n_estimators": 60, "early_stopping_rounds": 5}, tree_index=3)
    loss = b["charts"]["loss"]
    assert loss["metrics"] == ["error", "logloss"] and len(loss["curves"]["logloss"]["val"]) == len(loss["rounds"])
    assert b["charts"]["tree"]["index"] == 3 and abs(sum(r["value"] for r in b["charts"]["importance"]["gain"]) - 1) < 1e-3


def test_knn_explains_a_row(client: TestClient) -> None:
    sid = ready_session(client)
    b = fit(client, sid, "knn", params={"n_neighbors": 5}, row=3)
    ex = b["charts"]["knn"]["explain"]
    assert ex["row"] == 3 and len(ex["neighbours"]) == 5 and sum(ex["votes"].values()) == 5
    assert b["charts"]["boundary"]["n"] ** 2 == len(b["charts"]["boundary"]["grid"])


def test_feature_subset_and_scaling_note(client: TestClient) -> None:
    sid = ready_session(client)
    b = fit(client, sid, "knn", features=["tenure", "monthly_charges"], scale=False)
    assert b["data"]["features_used"] == ["tenure", "monthly_charges"]
    assert any("Scaling is off" in n for n in b["notes"])
    r = client.post(f"{U}/{sid}/lab/knn/fit", json={"features": ["nope"]})
    assert r.status_code == 400 and r.json()["error_code"] == "NO_FEATURES"


def test_regression_models(client: TestClient) -> None:
    sid = regression_session(client)
    lin = fit(client, sid, "linear_regression")
    assert lin["metric"] == "R2" and lin["test"]["r2"] > 0.5 and lin["charts"]["linear"]["inference"]
    assert lin["performance"]["qq"] and lin["charts"]["linear"]["equation"]
    lasso = fit(client, sid, "linear_regression", params={"regularization": "lasso", "alpha": 0.5})["charts"]["linear"]
    assert not lasso["inference"] and any(c["zero"] for c in lasso["coefficients"])
    for model in ("knn", "decision_tree", "svm"):
        assert fit(client, sid, model)["test"]["rmse"] > 0
    assert client.post(f"{U}/{sid}/lab/logistic_regression/fit", json={}).json()["error_code"] == "MODEL_NOT_FOR_TASK"


def test_kmeans_without_target_and_with_target(client: TestClient) -> None:
    sid = upload(client, make_df(missing=0))
    b = fit(client, sid, "kmeans", params={"n_clusters": 4})
    assert b["task"] == "clustering" and sum(b["sizes"]) == 400 and b["charts"]["target"] is None
    assert len(b["charts"]["profiles"]["centers"]) == 4 and -1 <= b["metrics"]["silhouette"] <= 1
    sid2 = ready_session(client)
    t = fit(client, sid2, "kmeans")["charts"]["target"]
    assert t["kind"] == "classes" and np.array(t["counts"]).sum() == 400


def test_lab_needs_clean_data_and_a_target(client: TestClient) -> None:
    sid = ready_session(client, clean=False)
    r = client.post(f"{U}/{sid}/lab/knn/fit", json={})
    assert r.status_code == 409 and r.json()["error_code"] == "NAN_REMAINING"
    sid2 = upload(client, make_df(missing=0))
    assert client.post(f"{U}/{sid2}/lab/knn/fit", json={}).json()["error_code"] == "TARGET_REQUIRED"
    assert client.post(f"{U}/{sid2}/lab/naive_bayes/fit", json={}).status_code == 400
    assert client.post(f"{U}/{sid2}/lab/kmeans/fit", json={"params": {"n_clusters": "x"}}).json()["error_code"] == "BAD_PARAM"


def test_sweeps(client: TestClient) -> None:
    sid = ready_session(client)
    curve = client.post(f"{U}/{sid}/lab/knn/sweep", json={"param": "n_neighbors", "values": [1, 5, 25]}).json()
    assert curve["kind"] == "curve" and curve["values"] == [1, 5, 25] and curve["train"][0] >= curve["train"][-1]
    grid = client.post(f"{U}/{sid}/lab/svm/sweep",
                       json={"param": "C", "values": [0.1, 1], "param2": "gamma", "values2": [0.01, 0.1]}).json()
    assert grid["kind"] == "grid" and len(grid["val"]) == 2 and len(grid["val"][0]) == 2
    assert grid["changed"] == {"gamma_mode": "custom"} and grid["best"]["score"] is not None
    lc = client.post(f"{U}/{sid}/lab/decision_tree/sweep", json={"param": "train_size"}).json()
    assert lc["kind"] == "learning_curve" and lc["values"] == sorted(lc["values"])
    km = client.post(f"{U}/{sid}/lab/kmeans/sweep", json={"param": "n_clusters", "values": [2, 3, 4]}).json()
    assert km["kind"] == "cluster_sweep" and km["inertia"][0] >= km["inertia"][-1]
    bad = client.post(f"{U}/{sid}/lab/knn/sweep", json={"param": "metric"})
    assert bad.status_code == 400 and bad.json()["error_code"] == "BAD_SWEEP_PARAM"


def test_lab_leaves_dashboard_results_alone(client: TestClient) -> None:
    sid = ready_session(client)
    before = client.get(f"{U}/{sid}").json()
    fit(client, sid, "decision_tree")
    after = client.get(f"{U}/{sid}").json()
    assert before["stage"] == after["stage"] == "CLEANED"


def test_multiclass_target(client: TestClient) -> None:
    df = make_df(missing=0)
    df["churn"] = pd.cut(df["tenure"], [-1, 20, 45, 100], labels=["new", "mid", "loyal"]).astype(str)
    sid = upload(client, df)
    assert client.post(f"{U}/{sid}/target", json={"target_column": "churn"}).status_code == 200
    b = fit(client, sid, "logistic_regression")
    assert b["metric"] == "F1 (weighted)" and len(b["charts"]["logistic"]["classes"]) == 3
    assert len(b["performance"]["roc"]) == 3 and b["performance"]["thresholds"] is None
    s = fit(client, sid, "svm")["charts"]["svm"]
    assert s["margins"]["kind"] == "multiclass" and set(s["per_class"]) == {"new", "mid", "loyal"}


def test_tree_nodes_report_real_class_mix_under_class_weights(client: TestClient) -> None:
    """With class_weight='balanced' the tree's own shares are weighted; nodes must also carry the real row counts."""
    sid = ready_session(client)
    body = fit(client, sid, "decision_tree", params={"max_depth": 3, "class_weight": "balanced"}, tree_depth=3)
    tree, labels = body["charts"]["tree"], body["data"]["class_labels"]
    root = tree["root"]
    assert tree["weighted"] is True and root["samples"] == tree["total"] == body["data"]["train_rows"]
    assert sum(root["counts"]) == root["samples"] and abs(sum(root["mix"]) - 1) < 1e-3
    assert root["dist"][1] == pytest.approx(0.5, abs=0.01)  # balanced weights: the tree sees 50/50 at the root
    assert root["mix"][1] < 0.45  # ...while the real share of the positive class is the base rate
    for child in root["children"]:
        assert sum(child["counts"]) == child["samples"]
    assert sum(c["samples"] for c in root["children"]) == root["samples"]
    for rule in body["charts"]["rules"]:
        j = labels.index(rule["pred"])
        assert rule["confidence"] == pytest.approx(rule["mix"][j])
    plain = fit(client, sid, "decision_tree", params={"max_depth": 3}, tree_depth=2)["charts"]["tree"]
    assert plain["weighted"] is False and plain["root"]["mix"][1] == pytest.approx(plain["root"]["dist"][1], abs=1e-3)


def test_summary_reads_the_fit_in_plain_words(client: TestClient) -> None:
    sid = ready_session(client)
    good = fit(client, sid, "logistic_regression")["summary"]
    labels = [line["label"] for line in good["lines"]]
    assert labels == ["WHAT", "SCORE", "RELIABILITY", "DRIVERS", "SETTINGS", "NEXT"]
    text = {line["label"]: line["text"] for line in good["lines"]}
    assert "`churn`" in text["WHAT"] and "always guessing" in text["SCORE"]
    assert text["SETTINGS"] == "All hyper-parameters are at their defaults."

    deep = fit(client, sid, "decision_tree", params={"max_depth": 0, "min_samples_leaf": 1, "min_samples_split": 2})["summary"]
    assert deep["verdict"] == "overfit" and any("ccp_alpha" in line["text"] for line in deep["lines"])
    assert "max depth" in next(line["text"] for line in deep["lines"] if line["label"] == "SETTINGS")

    km = fit(client, sid, "kmeans", params={"n_clusters": 4})["summary"]
    assert km["verdict"] in ("good", "weak") and "4 clusters" in km["lines"][0]["text"]


def test_regression_summary_compares_error_to_the_average(client: TestClient) -> None:
    sid = regression_session(client)
    s = fit(client, sid, "linear_regression")["summary"]
    score = next(line["text"] for line in s["lines"] if line["label"] == "SCORE")
    assert score.startswith("R2 ") and "MAE" in score and "predicting the average" in score
    assert "(a number)" in s["lines"][0]["text"]


def test_summary_advice_matches_the_model(client: TestClient) -> None:
    sid = ready_session(client)
    # k = 1 neighbour memorises: overfit advice is to raise k
    over = fit(client, sid, "knn", params={"n_neighbors": 1})["summary"]
    assert over["verdict"] == "overfit" and "raise k" in over["lines"][-1]["text"]
    from app.services.lab import _UNDERFIT_HINTS
    assert "lower k" in _UNDERFIT_HINTS["knn"]  # more neighbours would make it smoother, not more flexible
    # coefficient sizes are not comparable without scaling, so no drivers are named
    unscaled = fit(client, sid, "logistic_regression", scale=False)["summary"]
    assert "DRIVERS" not in [line["label"] for line in unscaled["lines"]]
    scaled = fit(client, sid, "logistic_regression")["summary"]
    assert "DRIVERS" in [line["label"] for line in scaled["lines"]]


def test_summary_names_coded_classes_with_their_column(client: TestClient) -> None:
    df = make_df(missing=0)
    df["churn"] = (df["churn"] == "Yes").astype(int)
    sid = upload(client, df)
    assert client.post(f"{U}/{sid}/target", json={"target_column": "churn"}).status_code == 200
    score = next(line["text"] for line in fit(client, sid, "logistic_regression")["summary"]["lines"] if line["label"] == "SCORE")
    assert "on the `churn = 1` class" in score and "always guessing `churn = 0`" in score
