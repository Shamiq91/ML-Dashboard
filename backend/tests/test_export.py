"""Model export: recipe recording, model card, and the zip's predict.py reproducing the dashboard's predictions."""
from __future__ import annotations

import importlib.util
import io
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


def unzip(client: TestClient, sid: str, model: str, dest: Path):
    r = client.get(f"{U}/{sid}/export/{model}")
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
