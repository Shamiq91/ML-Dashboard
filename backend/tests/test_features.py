"""Feature selection suggestions and column drops."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from tests.conftest import make_df, upload
from tests.test_api import assert_error

U = "/api/v1/sessions"


def with_target(client: TestClient, df: pd.DataFrame) -> str:
    sid = upload(client, df)
    assert client.post(f"{U}/{sid}/target", json={"target_column": "churn"}).status_code == 200
    return sid


def test_ranking_puts_informative_columns_first_and_ignores_ids(client: TestClient) -> None:
    df = make_df(500, missing=0)
    df["noise"] = np.random.default_rng(1).normal(size=len(df))
    sid = with_target(client, df)
    d = client.get(f"{U}/{sid}/features/suggest").json()
    names = [f["name"] for f in d["features"]]
    by = {f["name"]: f for f in d["features"]}
    assert by["customer_id"]["id_like"] and by["customer_id"]["rank"] is None and by["customer_id"]["mi"] == 0
    assert by["tenure"]["mi"] > by["noise"]["mi"]
    assert names.index("tenure") < names.index("noise")
    assert max(f["mi_rel"] for f in d["features"]) == 1.0


def test_suggestions_cover_redundant_missing_skewed_and_weak_columns(client: TestClient) -> None:
    rng = np.random.default_rng(2)
    df = make_df(600, missing=0)
    df["tenure_copy"] = df["tenure"] * 1.0 + rng.normal(0, 0.01, len(df))
    df["mostly_empty"] = np.where(rng.random(len(df)) < 0.7, np.nan, 1.0)
    df["skewed"] = np.exp(1.5 * (df["churn"] == "Yes") + rng.normal(0, 1.2, len(df)))  # right-skewed and tied to the target
    for i in range(5):
        df[f"noise{i}"] = rng.normal(size=len(df))
    sid = with_target(client, df)
    d = client.get(f"{U}/{sid}/features/suggest").json()
    ops = {(s["action"], tuple(s["columns"])): s for s in d["suggestions"]}
    assert ("drop", ("mostly_empty",)) in ops and ops[("drop", ("mostly_empty",))]["severity"] == "high"
    assert any(p["a"] in ("tenure", "tenure_copy") and p["b"] in ("tenure", "tenure_copy") for p in d["redundant_pairs"])
    assert any(s["action"] == "drop" and s["columns"][0] in ("tenure", "tenure_copy") and "duplicates" in s["reason"] for s in d["suggestions"])
    assert any(s["op"]["op"] == "log_transform" and s["columns"] == ["skewed"] for s in d["suggestions"])
    weak = [s for s in d["suggestions"] if s["severity"] == "low" and s["op"]["op"] == "drop_columns" and len(s["columns"]) > 1]
    assert len(weak) <= 1  # weak columns come as one grouped suggestion, not a flood
    # every suggestion carries an operation the transform endpoint accepts
    for s in d["suggestions"]:
        r = client.post(f"{U}/{sid}/transform", json={"operations": [s["op"]]})
        assert r.status_code == 200, (s, r.text)
        client.post(f"{U}/{sid}/reset")
        client.post(f"{U}/{sid}/target", json={"target_column": "churn"})


def test_suggest_needs_a_target(client: TestClient) -> None:
    sid = upload(client)
    assert_error(client.get(f"{U}/{sid}/features/suggest"), 409, "STAGE_CONFLICT")


def test_dropping_columns_removes_them_from_training_and_protects_the_target(client: TestClient) -> None:
    sid = with_target(client, make_df(200, missing=0))
    assert_error(client.post(f"{U}/{sid}/transform", json={"operations": [{"op": "drop_columns", "columns": ["churn"]}]}), 400, "PROTECTED_COLUMN")
    r = client.post(f"{U}/{sid}/transform", json={"operations": [{"op": "drop_columns", "columns": ["plan", "support_calls"]}]}).json()
    assert "plan" not in r["column_kinds"] and r["columns_after"] == r["columns_before"] - 2


def test_suggest_scores_categorical_columns_with_missing_values(client: TestClient) -> None:
    df = make_df(200, missing=0)
    df.loc[df.sample(30, random_state=1).index, "plan"] = np.nan
    sid = with_target(client, df)
    r = client.get(f"{U}/{sid}/features/suggest")
    assert r.status_code == 200
    assert {f["name"] for f in r.json()["features"]} >= {"plan"}
