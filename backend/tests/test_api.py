"""HTTP-level tests: ingestion, preprocessing, EDA, assumptions, guards and error schema."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.state.session_store import SessionStore
from tests.conftest import csv_bytes, make_df, ready_session, upload

U = "/api/v1/sessions"


def assert_error(response, status: int, code: str | None = None) -> dict:
    body = response.json()
    assert response.status_code == status, response.text
    assert set(body) == {"error_code", "message", "details"}
    assert "Traceback" not in response.text
    if code:
        assert body["error_code"] == code
    return body


def test_health_and_docs(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "version": "1.0.0"}
    spec = client.get("/openapi.json").json()
    assert "/api/v1/sessions/upload" in spec["paths"] and "/ws/radio" not in spec["paths"]


# --- ingestion -----------------------------------------------------------
def test_upload_csv_response_shape(client: TestClient) -> None:
    df = make_df(120)
    body = client.post(f"{U}/upload", files={"file": ("a.csv", csv_bytes(df), "text/csv")}).json()
    assert body["total_rows"] == 120 and body["total_columns"] == 7
    assert body["column_kinds"]["tenure"] == "numeric" and body["column_kinds"]["plan"] == "categorical"
    assert len(body["preview"]) == 5 and isinstance(body["memory_usage_mb"], float)
    assert body["dtypes"]["tenure"].startswith("int")


def test_upload_parquet_and_latin1(client: TestClient) -> None:
    buf = io.BytesIO()
    make_df(50).to_parquet(buf)
    r = client.post(f"{U}/upload", files={"file": ("a.parquet", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 200 and r.json()["total_rows"] == 50
    latin = "name,v\ncaf\xe9,1\nna\xefve,2\n".encode("latin-1")
    r = client.post(f"{U}/upload", files={"file": ("l.csv", latin, "text/csv")})
    assert r.status_code == 200 and r.json()["preview"][0]["name"] == "caf\xe9"


def test_upload_rejections(client: TestClient) -> None:
    def post(name: str, data: bytes):
        return client.post(f"{U}/upload", files={"file": (name, data, "application/octet-stream")})

    assert_error(post("a.txt", b"x"), 400, "UNSUPPORTED_FILE_TYPE")
    assert_error(post("a.csv", b""), 400, "EMPTY_FILE")
    assert_error(post("a.csv", b"a,b,c\n"), 400, "EMPTY_DATASET")
    assert_error(post("a.parquet", b"not a parquet"), 400, "CORRUPT_FILE")
    assert_error(post("a.csv", b"\x00\x01\x02binary"), 400, "CORRUPT_FILE")
    assert_error(post("big.csv", b"a\n" + b"1\n" * 700_000), 413, "FILE_TOO_LARGE")
    assert_error(client.post(f"{U}/upload"), 422, "VALIDATION_ERROR")


def test_unknown_session_and_routes(client: TestClient) -> None:
    for path in ("", "/missing", "/class-balance", "/eda/summary", "/assumptions", "/logs", "/leaderboard"):
        assert_error(client.get(f"{U}/nope{path}"), 404, "SESSION_NOT_FOUND")
    assert_error(client.delete(f"{U}/nope"), 404, "SESSION_NOT_FOUND")
    assert_error(client.get("/api/v1/unknown"), 404, "NOT_FOUND")


def test_session_expiry_ttl() -> None:
    now = [0.0]
    store = SessionStore(ttl_seconds=10, clock=lambda: now[0])
    session = store.create("a.csv", make_df(10))
    now[0] = 5
    assert store.get(session.session_id) is session
    now[0] = 30
    assert store.purge_expired() == [session.session_id] and len(store) == 0


def test_session_info_and_delete(client: TestClient) -> None:
    sid = upload(client)
    info = client.get(f"{U}/{sid}").json()
    assert info["stage"] == "UPLOADED" and info["target_column"] is None and info["has_missing"] is True
    assert client.delete(f"{U}/{sid}").json() == {"session_id": sid, "deleted": True}
    assert_error(client.get(f"{U}/{sid}"), 404)


# --- target & stage guards ----------------------------------------------
def test_target_validation_and_warnings(client: TestClient) -> None:
    sid = upload(client)
    assert_error(client.post(f"{U}/{sid}/target", json={"target_column": "zzz"}), 400, "TARGET_NOT_FOUND")
    assert_error(client.post(f"{U}/{sid}/target", json={}), 422, "VALIDATION_ERROR")
    ok = client.post(f"{U}/{sid}/target", json={"target_column": "churn"}).json()
    assert ok["kind"] == "binary" and ok["n_classes"] == 2 and ok["stage"] == "TARGET_SET"
    assert ok["task"] == "classification"
    cont = client.post(f"{U}/{sid}/target", json={"target_column": "monthly_charges"}).json()
    assert cont["kind"] == "continuous" and cont["task"] == "regression" and cont["warnings"] == []
    assert client.get(f"{U}/{sid}").json()["task"] == "regression"
    # manual override: forcing classification on a continuous target warns (every value would be a class)
    forced = client.post(f"{U}/{sid}/target", json={"target_column": "monthly_charges", "task": "classification"}).json()
    assert forced["task"] == "classification" and forced["warnings"]
    # regression needs a numeric target
    assert_error(
        client.post(f"{U}/{sid}/target", json={"target_column": "churn", "task": "regression"}), 400, "TARGET_NOT_NUMERIC"
    )


def test_target_all_nan_and_single_class(client: TestClient) -> None:
    df = make_df(30, missing=0)
    df["empty"] = np.nan
    df["one"] = "x"
    sid = upload(client, df)
    assert_error(client.post(f"{U}/{sid}/target", json={"target_column": "empty"}), 400, "TARGET_ALL_NAN")
    single = client.post(f"{U}/{sid}/target", json={"target_column": "one"}).json()
    assert single["kind"] == "single_class" and single["warnings"]
    assert client.get(f"{U}/{sid}/class-balance").json()["single_class"] is True
    assert_error(client.post(f"{U}/{sid}/train", json={"models": ["lr"]}), 400, "SINGLE_CLASS_TARGET")


def test_stage_guard_409(client: TestClient) -> None:
    sid = upload(client)
    for method, path, body in [
        ("post", "clean", {"strategy": "drop"}),
        ("get", "class-balance", None),
        ("get", "eda/summary", None),
        ("get", "eda/correlation", None),
        ("get", "assumptions", None),
        ("post", "train", {"models": ["lr"]}),
        ("get", "leaderboard", None),
        ("get", "telemetry/mlp", None),
    ]:
        r = getattr(client, method)(f"{U}/{sid}/{path}", **({"json": body} if body else {}))
        assert_error(r, 409, "STAGE_CONFLICT")
    assert client.get(f"{U}/{sid}/missing").status_code == 200  # allowed right after upload


# --- preprocessing -------------------------------------------------------
def test_missing_report(client: TestClient) -> None:
    sid = upload(client)
    body = client.get(f"{U}/{sid}/missing").json()
    assert body["has_missing"] and body["total_missing"] == 5
    assert body["columns"]["total_charges"]["count"] == 5 and body["columns"]["tenure"]["percent"] == 0.0


def test_clean_validation(client: TestClient) -> None:
    sid = ready_session(client, clean=False)
    assert_error(client.post(f"{U}/{sid}/clean", json={"strategy": "impute"}), 422, "VALIDATION_ERROR")
    assert_error(client.post(f"{U}/{sid}/clean", json={"strategy": "magic"}), 422)
    assert_error(client.post(f"{U}/{sid}/clean", json={"strategy": "drop", "columns": ["zzz"]}), 400, "UNKNOWN_COLUMNS")


def test_clean_drop_impute_and_reset(client: TestClient) -> None:
    sid = ready_session(client, clean=False)
    dropped = client.post(f"{U}/{sid}/clean", json={"strategy": "drop"}).json()
    assert dropped["rows_before"] == 400 and dropped["rows_after"] == 395 and dropped["missing_after"] == 0
    assert client.get(f"{U}/{sid}").json()["stage"] == "CLEANED"
    assert client.post(f"{U}/{sid}/reset").json() == {"rows": 400, "stage": "TARGET_SET"}
    assert client.get(f"{U}/{sid}/missing").json()["total_missing"] == 5
    for method in ("mean", "median", "mode", "knn"):
        client.post(f"{U}/{sid}/reset")
        r = client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": method, "knn_neighbors": 3}).json()
        assert r["missing_after"] == 0 and r["rows_after"] == 400, method
        assert "total_charges" in r["methods_applied"]


def test_impute_categorical_falls_back_to_mode(client: TestClient) -> None:
    df = make_df(60, missing=0)
    df.loc[[1, 2, 3], "plan"] = np.nan
    sid = ready_session(client, df, clean=False)
    r = client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "mean"}).json()
    assert r["missing_after"] == 0 and r["fallbacks"] and r["methods_applied"]["plan"] == "mode (fallback)"


def test_knn_preserves_scale_and_reduces_k_on_tiny_data(client: TestClient) -> None:
    df = make_df(8, missing=0)
    df.loc[2, "monthly_charges"] = np.nan
    sid = ready_session(client, df, clean=False)
    r = client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "knn", "knn_neighbors": 50}).json()
    assert r["missing_after"] == 0 and any("reduced" in w for w in r["warnings"])
    # restored scale: imputed value stays in the original data range, not z-score range
    sid2 = ready_session(client, make_df(200), clean=False)
    client.post(f"{U}/{sid2}/clean", json={"strategy": "impute", "method": "knn"})
    assert client.get(f"{U}/{sid2}/eda/summary").json()["features"]["total_charges"]["max"] > 100


def test_drop_that_empties_dataset_is_409(client: TestClient) -> None:
    df = make_df(20, missing=0)
    df["all_missing"] = np.nan
    sid = ready_session(client, df, clean=False)
    assert_error(client.post(f"{U}/{sid}/clean", json={"strategy": "drop"}), 409, "EMPTY_RESULT")
    assert client.get(f"{U}/{sid}").json()["rows"] == 20  # untouched
    r = client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "mean"}).json()
    assert r["methods_applied"]["all_missing"].startswith("skipped")


def test_class_balance_endpoint(client: TestClient) -> None:
    sid = ready_session(client)
    body = client.get(f"{U}/{sid}/class-balance").json()
    assert {c["label"] for c in body["classes"]} == {"Yes", "No"}
    assert body["imbalance_warning"] == (body["minority_ratio"] < 0.30)
    assert body["severity"] == ("STRICT" if body["imbalance_warning"] else "NONE")


# --- EDA & assumptions ---------------------------------------------------
def test_eda_endpoints(client: TestClient) -> None:
    sid = ready_session(client)
    feats = client.get(f"{U}/{sid}/eda/summary").json()["features"]
    assert set(feats) == {"tenure", "monthly_charges", "total_charges", "support_calls"}
    assert set(feats["tenure"]) >= {"mean", "std", "min", "max", "count", "median"}
    corr = client.get(f"{U}/{sid}/eda/correlation").json()
    assert corr["columns"] == list(feats) and len(corr["matrix"]) == 4
    assert all(corr["matrix"][i][i] == 1.0 for i in range(4))
    flagged = {frozenset((f["feature_a"], f["feature_b"])) for f in corr["multicollinearity_flags"]}
    cols = corr["columns"]
    for i in range(4):
        for j in range(i + 1, 4):
            r = corr["matrix"][i][j]
            assert corr["matrix"][j][i] == r  # symmetric
            assert (frozenset((cols[i], cols[j])) in flagged) == (abs(r) > 0.85)


def test_assumptions_endpoint(client: TestClient) -> None:
    sid = ready_session(client)
    body = client.get(f"{U}/{sid}/assumptions").json()
    assert body["normality"]["test"] == "Shapiro-Wilk" and body["independence"]["test"] == "Durbin-Watson"
    assert body["homoscedasticity"]["test"] == "Breusch-Pagan"
    # churn is categorical: residual normality / homoscedasticity are not applicable and don't count
    assert body["normality"]["result"] == "N/A" and body["homoscedasticity"]["result"] == "N/A"
    assert body["overall_status"] == body["independence"]["status"]
    one = client.get(f"{U}/{sid}/assumptions", params={"column": "tenure"}).json()
    assert "column 'tenure'" in one["normality"]["detail"]
    assert_error(client.get(f"{U}/{sid}/assumptions", params={"column": "plan"}), 400, "COLUMN_NOT_NUMERIC")
    assert_error(client.get(f"{U}/{sid}/assumptions", params={"column": "zzz"}), 400, "UNKNOWN_COLUMNS")


def test_assumptions_with_no_numeric_features_is_200(client: TestClient) -> None:
    df = pd.DataFrame({"a": list("abcdefgh"), "churn": list("ababbaab")})
    sid = ready_session(client, df, clean=False)
    body = client.get(f"{U}/{sid}/assumptions").json()
    assert body["normality"]["result"] == "SKIPPED" and body["overall_status"] == "YELLOW"
    assert client.get(f"{U}/{sid}/eda/correlation").json()["matrix"] == []


def test_nan_and_inf_never_reach_json(client: TestClient) -> None:
    df = make_df(40, missing=0)
    df["huge"] = [np.inf, -np.inf] + [1.0] * 38
    sid = ready_session(client, df, clean=False)
    r = client.get(f"{U}/{sid}/eda/summary")
    assert r.status_code == 200 and "NaN" not in r.text and "Infinity" not in r.text
    assert client.post(f"{U}/upload", files={"file": ("x.csv", csv_bytes(df), "text/csv")}).status_code == 200


def test_unexpected_errors_return_clean_500(settings: Settings, monkeypatch) -> None:
    from app.routers import eda as eda_router

    def boom(*_a, **_k):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(eda_router.eda, "eda_summary", boom)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        sid = ready_session(c)
        r = c.get(f"{U}/{sid}/eda/summary")
        body = assert_error(r, 500, "INTERNAL_ERROR")
        assert "secret" not in r.text and body["message"] == "An unexpected error occurred."


def test_regression_target_summary_and_class_balance_guard(client: TestClient) -> None:
    sid = upload(client, make_df(300, missing=0))
    client.post(f"{U}/{sid}/target", json={"target_column": "monthly_charges"})
    summary = client.get(f"{U}/{sid}/target-summary").json()
    h = summary["histogram"]
    assert summary["count"] == 300 and summary["min"] <= summary["q1"] <= summary["median"] <= summary["q3"] <= summary["max"]
    assert len(h["edges"]) == len(h["counts"]) + 1 and sum(h["counts"]) == 300
    assert_error(client.get(f"{U}/{sid}/class-balance"), 400, "NOT_CLASSIFICATION")
    # classification target: the reverse guard
    client.post(f"{U}/{sid}/target", json={"target_column": "churn"})
    assert_error(client.get(f"{U}/{sid}/target-summary"), 400, "NOT_REGRESSION")
    assert client.get(f"{U}/{sid}/class-balance").status_code == 200


def test_assumptions_follow_the_chosen_task(client: TestClient) -> None:
    sid = upload(client, make_df(300, missing=0))
    client.post(f"{U}/{sid}/target", json={"target_column": "monthly_charges"})
    reg = client.get(f"{U}/{sid}/assumptions").json()
    assert reg["normality"]["result"] in {"PASS", "FAIL"} and reg["homoscedasticity"]["result"] in {"PASS", "FAIL"}
    # a numeric target with few distinct values is categorical by default, but regression can be forced
    df = make_df(300, missing=0)
    df["rating"] = (df["tenure"] % 7).astype(float)
    sid2 = upload(client, df)
    client.post(f"{U}/{sid2}/target", json={"target_column": "rating"})
    assert client.get(f"{U}/{sid2}/assumptions").json()["normality"]["result"] == "N/A"
    client.post(f"{U}/{sid2}/target", json={"target_column": "rating", "task": "regression"})
    assert client.get(f"{U}/{sid2}/assumptions").json()["normality"]["result"] in {"PASS", "FAIL"}
