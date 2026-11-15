"""Feature relevance report and column drops."""
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
    d = client.get(f"{U}/{sid}/features").json()
    names = [f["name"] for f in d["features"]]
    by = {f["name"]: f for f in d["features"]}
    assert by["customer_id"]["id_like"] and by["customer_id"]["rank"] is None and by["customer_id"]["mi"] == 0
    assert by["tenure"]["mi"] > by["noise"]["mi"]
    assert names.index("tenure") < names.index("noise")
    assert max(f["mi_rel"] for f in d["features"]) == 1.0


def test_report_flags_redundant_pairs_and_skew(client: TestClient) -> None:
    rng = np.random.default_rng(2)
    df = make_df(600, missing=0)
    df["tenure_copy"] = df["tenure"] * 1.0 + rng.normal(0, 0.01, len(df))
    df["skewed"] = np.exp(1.5 * (df["churn"] == "Yes") + rng.normal(0, 1.2, len(df)))  # right-skewed and tied to the target
    sid = with_target(client, df)
    d = client.get(f"{U}/{sid}/features").json()
    assert any(p["a"] in ("tenure", "tenure_copy") and p["b"] in ("tenure", "tenure_copy") for p in d["redundant_pairs"])
    assert next(f for f in d["features"] if f["name"] == "skewed")["skew"] > 1.5
    assert "suggestions" not in d


def test_features_needs_a_target(client: TestClient) -> None:
    sid = upload(client)
    assert_error(client.get(f"{U}/{sid}/features"), 409, "STAGE_CONFLICT")


def test_dropping_columns_removes_them_from_training_and_protects_the_target(client: TestClient) -> None:
    sid = with_target(client, make_df(200, missing=0))
    assert_error(client.post(f"{U}/{sid}/transform", json={"operations": [{"op": "drop_columns", "columns": ["churn"]}]}), 400, "PROTECTED_COLUMN")
    r = client.post(f"{U}/{sid}/transform", json={"operations": [{"op": "drop_columns", "columns": ["plan", "support_calls"]}]}).json()
    assert "plan" not in r["column_kinds"] and r["columns_after"] == r["columns_before"] - 2


def test_features_scores_categorical_columns_with_missing_values(client: TestClient) -> None:
    df = make_df(200, missing=0)
    df.loc[df.sample(30, random_state=1).index, "plan"] = np.nan
    sid = with_target(client, df)
    r = client.get(f"{U}/{sid}/features")
    assert r.status_code == 200
    assert {f["name"] for f in r.json()["features"]} >= {"plan"}


def test_log_transform_reports_skew_change_and_flags_zero_tail() -> None:
    from app.services import transforms

    rng = np.random.default_rng(0)
    w = pd.Series(np.exp(rng.normal(6, 0.6, 2000)).round())
    w.iloc[:20] = 0  # 'unknown' coded as 0
    out, summary = transforms.apply_operation(pd.DataFrame({"weight": w}), {"op": "log_transform", "columns": ["weight"]}, set())
    assert summary.startswith("applied log1p to 1 column(s): weight skew +")
    assert "20 zero value(s)" in summary and out["weight"].skew() < -1.5


def test_left_skew_is_reported_with_its_sign() -> None:
    from app.config import Settings
    from app.services.features import feature_report

    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 1500)
    left = 100 - np.exp(rng.normal(0, 1, 1500)) + y  # long low tail, all positive
    rep = feature_report(pd.DataFrame({"left": left, "y": y}), "y", "regression", Settings())
    assert next(f for f in rep["features"] if f["name"] == "left")["skew"] < -1.5
