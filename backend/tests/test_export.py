"""Model export: recipe recording, model card, and the zip's predict.py reproducing the dashboard's predictions."""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.services import inference_runtime as rt
from app.services.modeling import prepare_dataset
from tests.conftest import csv_bytes, make_df, ready_session, upload, wait_job

U = "/api/v1/sessions"
OPS = {"operations": [
    {"op": "log_transform", "columns": ["support_calls"]},
    {"op": "rename", "mapping": {"tenure": "months"}},
    {"op": "outliers", "columns": ["monthly_charges"], "method": "iqr", "factor": 1.0, "action": "clip"},
    {"op": "filter_rows", "column": "months", "comparison": "ge", "value": 1},
]}


def train(client: TestClient, sid: str, models: list[str]) -> dict:
    r = client.post(f"{U}/{sid}/train", json={"models": models, "cross_validate": False})
    assert r.status_code == 202, r.text
    job = wait_job(client, sid, r.json()["job_id"])
    assert job["status"] == "COMPLETE", job
    return job


def transformed_session(client: TestClient) -> str:
    sid = ready_session(client)
    r = client.post(f"{U}/{sid}/transform", json=OPS)
    assert r.status_code == 200, r.text
    return sid


def unzip(client: TestClient, sid: str, model: str, dest: Path, lab: dict | None = None):
    """Download a dashboard export (or, with ``lab`` = a lab fit body, the lab's export) and load it."""
    r = client.post(f"{U}/{sid}/lab/{model}/export", json=lab) if lab is not None else client.get(f"{U}/{sid}/export/{model}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip" and r.headers["content-disposition"].endswith('.zip"')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert set(z.namelist()) == {"model.joblib", "predict.py", "MODEL_CARD.md", "model_card.json", "requirements.txt", "example_input.csv"}
    z.extractall(dest)
    spec = importlib.util.spec_from_file_location(f"exported_{model}", dest / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, joblib.load(dest / "model.joblib")


def test_recipe_replays_transforms_and_reset_clears_it(client: TestClient) -> None:
    sid = transformed_session(client)
    session = client.app.state.store.get(sid)
    assert [s["op"] for s in session.recipe] == ["log_transform", "rename", "clip"]  # the row filter is not replayed
    replayed = rt.replay_recipe(session.raw_df, session.recipe).set_index("customer_id").loc[session.working_df["customer_id"]]
    for col in ("monthly_charges", "months", "support_calls"):
        np.testing.assert_allclose(replayed[col].to_numpy(dtype=float), session.working_df[col].to_numpy(dtype=float))
    client.post(f"{U}/{sid}/reset")
    assert session.recipe == [] and session.cleaning == []


def test_card_and_export_reproduce_dashboard_predictions(client: TestClient, tmp_path: Path) -> None:
    sid = transformed_session(client)
    train(client, sid, ["random_forest", "logistic_regression"])
    card = client.get(f"{U}/{sid}/export/random_forest/card").json()
    assert card["task"] == "classification" and card["classes"] == ["No", "Yes"] and card["metrics"]["f1"] > 0
    assert "tenure" in card["input"]["columns"] and "months" not in card["input"]["columns"]  # original names
    assert "customer_id" not in card["input"]["columns"]
    assert [s["op"] for s in card["input"]["recipe"]] == ["log_transform", "rename", "clip"]
    assert card["portable"] and any(r.startswith("scikit-learn==") for r in card["requirements"])
    assert not any("Missing values in the training data" in x for x in card["limitations"])  # median = what predict.py does

    predict, bundle = unzip(client, sid, "random_forest", tmp_path)
    assert "## Input" in (md := (tmp_path / "MODEL_CARD.md").read_text()) and "log1p `support_calls`" in md
    session = client.app.state.store.get(sid)
    # rows that survived the filter, in their original (pre-transform) form
    raw = session.raw_df.set_index("customer_id").loc[session.working_df["customer_id"]].reset_index()
    report: dict = {}
    out = predict.predict(bundle, raw, report)
    prepared = prepare_dataset(session.working_df, "churn", client.app.state.settings)
    expected = np.asarray(["No", "Yes"], dtype=object)[np.asarray(session.fitted["random_forest"].pipeline.predict(prepared.X))]
    complete = raw["total_charges"].notna().to_numpy()  # the dashboard imputed these with the median, as predict.py does
    assert (out["prediction"].to_numpy()[complete] == expected[complete]).all()
    assert np.allclose(out[["prob_No", "prob_Yes"]].sum(axis=1), 1)
    assert report["filled"] == {"total_charges": int((~complete).sum())} or not (~complete).any()

    # the CLI, on new rows with an unseen category, a missing column value and an extra column
    new = make_df(40, seed=11).assign(extra="x")
    new.loc[0, "plan"] = "zzz"
    (tmp_path / "in.csv").write_bytes(csv_bytes(new))
    done = subprocess.run([sys.executable, "predict.py", "model.joblib", "in.csv", "out.csv"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "never seen in training: plan (1)" in done.stdout
    got = pd.read_csv(tmp_path / "out.csv")
    assert len(got) == 40 and {"prediction", "prob_Yes", "extra"} <= set(got.columns)
    done = subprocess.run([sys.executable, "predict.py", "model.joblib", "example_input.csv", "ex.csv"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

    try:
        predict.predict(bundle, new.drop(columns=["tenure"]))
        raise AssertionError("expected a missing-column error")
    except ValueError as exc:
        assert "tenure" in str(exc)  # reported by its original name


def test_regression_export_and_errors(client: TestClient, tmp_path: Path) -> None:
    df = make_df()
    sid = upload(client, df)
    assert client.get(f"{U}/{sid}/export/ridge").status_code == 409  # nothing trained yet
    client.post(f"{U}/{sid}/target", json={"target_column": "monthly_charges", "task": "regression"})
    client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "median"})
    train(client, sid, ["ridge"])
    assert client.get(f"{U}/{sid}/export/xgboost").status_code == 404
    card = client.get(f"{U}/{sid}/export/ridge/card").json()
    assert card["classes"] is None and set(card["metrics"]) == {"r2", "rmse", "mae"}
    predict, bundle = unzip(client, sid, "ridge", tmp_path)
    session = client.app.state.store.get(sid)
    m = session.fitted["ridge"]
    out = predict.predict(bundle, session.raw_df.dropna())
    prepared = prepare_dataset(session.working_df, "monthly_charges", client.app.state.settings, "regression")
    keep = session.raw_df.notna().all(axis=1).to_numpy()
    expected = m.pipeline.predict(prepared.X) * m.y_sd + m.y_mu
    np.testing.assert_allclose(out["prediction"].to_numpy(), expected[keep], rtol=1e-9)


def test_torch_model_export_is_marked_not_portable(client: TestClient) -> None:
    sid = ready_session(client)
    train(client, sid, ["torch_mlp"])
    card = client.get(f"{U}/{sid}/export/torch_mlp/card").json()
    assert card["portable"] is False and any(r.startswith("torch") for r in card["requirements"])
    assert any("PYTHONPATH" in x for x in card["limitations"])
    assert client.get(f"{U}/{sid}/export/torch_mlp").status_code == 200


def test_card_warns_when_imputation_differs_from_predict_fill(client: TestClient) -> None:
    sid = upload(client)
    client.post(f"{U}/{sid}/target", json={"target_column": "churn"})
    assert client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "knn"}).status_code == 200
    train(client, sid, ["logistic_regression"])
    card = client.get(f"{U}/{sid}/export/logistic_regression/card").json()
    assert any("knn" in x for x in card["limitations"])


# --- model lab export --------------------------------------------------------------------------------------------
def test_lab_export_carries_the_lab_settings(client: TestClient, tmp_path: Path) -> None:
    sid = transformed_session(client)
    body = {"params": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.2}, "test_size": 0.3, "scale": False,
            "features": ["months", "monthly_charges", "plan", "support_calls"], "cv": False}
    shown = client.post(f"{U}/{sid}/lab/xgboost/fit", json=body).json()
    predict, bundle = unzip(client, sid, "xgboost", tmp_path, lab=body)
    card = json.loads((tmp_path / "model_card.json").read_text())
    assert card["origin"] == "lab" and card["hyperparams"] == shown["params"]
    assert card["metrics"]["f1"] == shown["test"]["f1"] and card["rows_tested"] == shown["data"]["test_rows"]
    assert card["settings"]["test_size"] == 0.3 and card["settings"]["scale"] is False
    # only the chosen features, by their original names (``months`` was ``tenure`` in the upload)
    assert sorted(card["input"]["columns"]) == ["monthly_charges", "plan", "support_calls", "tenure"]
    md = (tmp_path / "MODEL_CARD.md").read_text()
    assert "Training settings (model lab)" in md and "| max_depth | 2 |" in md
    assert bundle["model"] == "xgboost"

    session = client.app.state.store.get(sid)
    raw = session.raw_df.set_index("customer_id").loc[session.working_df["customer_id"]].reset_index()
    complete = raw["total_charges"].notna().to_numpy()
    out = predict.predict(bundle, raw)
    x = prepare_dataset(session.working_df, "churn", client.app.state.settings).X[bundle["features"]]
    expected = np.asarray(["No", "Yes"], dtype=object)[np.asarray(bundle["pipeline"].predict(x))]
    assert (out["prediction"].to_numpy()[complete] == expected[complete]).all()
    done = subprocess.run([sys.executable, "predict.py", "model.joblib", "example_input.csv", "ex.csv"], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_lab_regression_export_and_clustering_refused(client: TestClient, tmp_path: Path) -> None:
    df = make_df(missing=0).drop(columns=["churn"])
    sid = upload(client, df)
    client.post(f"{U}/{sid}/target", json={"target_column": "total_charges", "task": "regression"})
    body = {"params": {"regularization": "ridge", "alpha": 3.0}, "cv": True}
    shown = client.post(f"{U}/{sid}/lab/linear_regression/fit", json=body).json()
    r = client.post(f"{U}/{sid}/lab/linear_regression/export", json=body)
    assert r.status_code == 200 and r.headers["content-disposition"].endswith('_lab.zip"')
    predict, bundle = unzip(client, sid, "linear_regression", tmp_path, lab=body)
    card = json.loads((tmp_path / "model_card.json").read_text())
    assert set(card["metrics"]) == {"r2", "rmse", "mae"} and card["metrics"]["rmse"] == shown["test"]["rmse"]
    assert card["cv"]["score"] == shown["cv"]["mean"] and card["hyperparams"]["alpha"] == 3.0
    out = predict.predict(bundle, df)
    assert np.corrcoef(out["prediction"], df["total_charges"])[0, 1] > 0.8  # predictions on the original scale

    r = client.post(f"{U}/{sid}/lab/kmeans/export", json={})
    assert r.status_code == 400 and r.json()["error_code"] == "NOT_EXPORTABLE"
