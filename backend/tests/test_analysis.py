"""Tuning, curves/threshold, importance, SHAP, PyTorch/TabNet models, data preview/transform, runs and comparison."""
from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import make_df, ready_session, upload, wait_job

U = "/api/v1/sessions"


def train(client: TestClient, sid: str, models: list[str], **extra: object) -> dict:
    r = client.post(f"{U}/{sid}/train", json={"models": models, **extra})
    assert r.status_code == 202, r.text
    job = wait_job(client, sid, r.json()["job_id"])
    assert job["status"] == "COMPLETE", job
    return job


def test_tuning_reports_and_never_worsens(client: TestClient) -> None:
    sid = ready_session(client)
    job = train(client, sid, ["random_forest", "logistic_regression", "naive_bayes", "voting"], tune=True, tune_trials=4)
    rows = {r["abbr"]: r for r in job["result"]["leaderboard"]}
    assert rows["RFR"]["tuning"]["n_trials"] == 4 and rows["RFR"]["tuning"]["metric"] in {"f1", "f1_weighted", "f1_macro"}
    assert rows["VOT"]["tuned"] is False and rows["VOT"]["tuning"] is None  # no search space
    detail = client.get(f"{U}/{sid}/tuning/random_forest").json()
    assert len(detail["history"]) == 4 and detail["default_score"] is not None
    if not detail["applied"]:  # defaults are kept unless a trial beats them
        assert detail["best_score"] <= detail["default_score"] + 1e-9
    assert client.get(f"{U}/{sid}/tuning/voting").status_code in (200, 404)
    untuned = train(client, sid, ["decision_tree"])
    assert client.get(f"{U}/{sid}/tuning/decision_tree").status_code == 404
    assert untuned["run_id"]


def test_too_many_trials_rejected(client: TestClient) -> None:
    sid = ready_session(client)
    assert client.post(f"{U}/{sid}/train", json={"models": ["knn"], "tune": True, "tune_trials": 61}).status_code == 422


def test_curves_threshold_importance_shap(client: TestClient) -> None:
    sid = ready_session(client)
    train(client, sid, ["xgboost", "svm", "knn"])
    c = client.get(f"{U}/{sid}/curves/xgboost?threshold=0.3").json()
    assert c["binary"] and c["positive_class"] == "Yes" and 0.5 < c["roc"]["macro_auc"] <= 1
    t = c["thresholds"]
    assert len(t["grid"]) == 101 and t["at"]["threshold"] == 0.3
    at = t["at"]
    assert at["tp"] + at["fp"] + at["tn"] + at["fn"] == c["n_test"]
    assert t["recall"][0] == 1.0 and t["recall"][-1] == 0.0  # threshold 0 flags everything, 1 flags nothing
    assert all(0 <= v <= 1 for v in c["roc"]["classes"][0]["fpr"])
    svm = client.get(f"{U}/{sid}/curves/svm").json()
    assert svm["score_kind"] == "decision"  # SVC has no probabilities; its margin is min-max scaled
    imp = client.get(f"{U}/{sid}/importance/xgboost").json()
    names = {f["feature"] for f in imp["features"]}
    assert "tenure" in names and "customer_id" not in names and imp["native"]
    sh = client.get(f"{U}/{sid}/shap/xgboost").json()
    assert sh["explainer"] == "TreeExplainer" and sh["global"][0]["mean_abs"] >= sh["global"][-1]["mean_abs"]
    loc = sh["local"]
    total = loc["base"] + sum(c_["shap"] for c_ in loc["contributions"]) + loc["other"]["shap"]
    assert abs(total - loc["output"]) < 1e-3  # SHAP values are additive
    assert client.get(f"{U}/{sid}/shap/knn?row=3").json()["explainer"] == "KernelExplainer"
    assert client.get(f"{U}/{sid}/residuals/xgboost").status_code == 400
    assert client.get(f"{U}/{sid}/curves/lightgbm").status_code == 404  # not trained


def test_regression_residuals_and_shap(client: TestClient) -> None:
    df = make_df()
    sid = upload(client, df)
    client.post(f"{U}/{sid}/target", json={"target_column": "monthly_charges", "task": "regression"})
    client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "median"})
    train(client, sid, ["ridge", "random_forest"])
    r = client.get(f"{U}/{sid}/residuals/ridge").json()
    assert len(r["scatter"]["actual"]) == len(r["scatter"]["predicted"]) and "bias" in r
    assert client.get(f"{U}/{sid}/curves/ridge").status_code == 400
    sh = client.get(f"{U}/{sid}/shap/ridge").json()
    assert sh["explainer"] == "LinearExplainer" and sh["output_space"] == "target units"
    assert abs(sh["local"]["base"] + sum(c["shap"] for c in sh["local"]["contributions"]) + sh["local"]["other"]["shap"] - sh["local"]["output"]) < 1e-3


def test_pytorch_models(client: TestClient) -> None:
    sid = ready_session(client)
    job = train(client, sid, ["torch_mlp", "tabnet"])
    rows = {r["abbr"]: r for r in job["result"]["leaderboard"]}
    assert set(rows) == {"PTM", "TAB"} and all(0 <= r["f1"] <= 1 and r["roc_auc"] for r in rows.values())
    for key in ("torch_mlp", "tabnet"):
        tele = client.get(f"{U}/{sid}/telemetry/{key}").json()
        assert tele["kind"] == "epochs" and len(tele["train_loss"]) >= 1
    assert client.get(f"{U}/{sid}/importance/tabnet").json()["native"]


def test_data_preview_profile_and_transform(client: TestClient) -> None:
    sid = ready_session(client, clean=False)
    p = client.get(f"{U}/{sid}/preview?limit=5&sort=tenure&descending=true").json()
    assert len(p["rows"]) == 5 and p["total"] == 400 and p["rows"][0]["tenure"] >= p["rows"][1]["tenure"]
    assert next(c for c in p["columns"] if c["name"] == "total_charges")["missing"] == 5
    assert client.get(f"{U}/{sid}/preview?search=zzzzzz").json()["filtered"] == 0
    assert client.get(f"{U}/{sid}/preview?sort=nope").status_code == 400
    prof = client.get(f"{U}/{sid}/profile").json()["columns"]
    assert next(c for c in prof if c["name"] == "tenure")["histogram"]["counts"]
    assert next(c for c in prof if c["name"] == "plan")["top"]
    ops = {"operations": [
        {"op": "outliers", "columns": ["monthly_charges"], "method": "iqr", "factor": 1.0, "action": "clip"},
        {"op": "log_transform", "columns": ["support_calls"]},
        {"op": "drop_columns", "columns": ["plan"]},
    ]}
    r = client.post(f"{U}/{sid}/transform", json=ops)
    assert r.status_code == 200 and r.json()["columns_after"] == r.json()["columns_before"] - 1
    cols = [c["name"] for c in client.get(f"{U}/{sid}/preview").json()["columns"]]
    assert "plan" not in cols
    bad = client.post(f"{U}/{sid}/transform", json={"operations": [{"op": "drop_columns", "columns": ["churn"]}]})
    assert bad.status_code == 400 and bad.json()["error_code"] == "PROTECTED_COLUMN"
    # atomic: a failing second step leaves the first one uncommitted
    n_before = client.get(f"{U}/{sid}/preview").json()["total"]
    client.post(f"{U}/{sid}/transform", json={"operations": [{"op": "drop_duplicates"}, {"op": "drop_columns", "columns": ["nope"]}]})
    assert client.get(f"{U}/{sid}/preview").json()["total"] == n_before
    assert client.post(f"{U}/{sid}/transform", json={"operations": [{"op": "bogus"}]}).status_code == 422
    assert client.get(f"{U}/{sid}/preview?source=raw").json()["columns"][0]["name"] == "customer_id"


def test_runs_datasets_and_comparison(client: TestClient) -> None:
    df = make_df()
    a = ready_session(client, df)
    train(client, a, ["logistic_regression", "decision_tree"], label="baseline")
    train(client, a, ["random_forest"], label="more models")
    b = ready_session(client, make_df(rows=300, seed=9))
    train(client, b, ["logistic_regression", "decision_tree"], tune=True, tune_trials=3, label="other data")
    runs = client.get("/api/v1/runs").json()["runs"]
    assert [r["label"] for r in runs][:3] == ["other data", "more models", "baseline"]
    ds = client.get("/api/v1/datasets").json()
    assert ds["total"] == 2 and sorted(d["n_runs"] for d in ds["datasets"]) == [1, 2]
    one = client.get(f"/api/v1/runs/{runs[2]['id']}").json()
    assert {r["abbr"] for r in one["leaderboard"]} == {"LR", "DT"} and one["curves"] and one["target_profile"]["n_classes"] == 2
    cmp = client.post("/api/v1/compare", json={"run_ids": [runs[2]["id"], runs[0]["id"]]}).json()
    assert cmp["common_models"] == ["LR", "DT"] or set(cmp["common_models"]) == {"LR", "DT"}
    assert any("different datasets" in w for w in cmp["warnings"]) and len(cmp["datasets"]) == 2 and cmp["same_task"]
    same = client.post("/api/v1/compare", json={"run_ids": [runs[2]["id"], runs[1]["id"]]}).json()
    assert len(same["datasets"]) == 1 and not same["warnings"]
    assert client.post("/api/v1/compare", json={"run_ids": [runs[0]["id"]]}).status_code == 422
    patched = client.patch(f"/api/v1/runs/{runs[0]['id']}", json={"label": "renamed", "notes": "n"}).json()
    assert patched["label"] == "renamed"
    assert client.delete(f"/api/v1/runs/{runs[0]['id']}").status_code == 200
    assert client.get(f"/api/v1/runs/{runs[0]['id']}").status_code == 404
    assert client.delete(f"/api/v1/datasets/{ds['datasets'][0]['id']}").status_code == 200


def test_shap_binary_explains_positive_class_and_mirrors(client: TestClient) -> None:
    sid = ready_session(client)
    train(client, sid, ["xgboost"])
    pos = client.get(f"{U}/{sid}/shap/xgboost").json()
    neg = client.get(f"{U}/{sid}/shap/xgboost?class_index=0").json()
    assert pos["class_label"] == "Yes" and neg["class_label"] == "No"
    assert pos["global"][0]["feature"] == neg["global"][0]["feature"]
    assert abs(pos["local"]["output"] + neg["local"]["output"]) < 1e-3  # mirror image in log-odds
