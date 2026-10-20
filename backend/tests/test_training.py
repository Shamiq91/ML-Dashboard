"""Training jobs, leaderboard, telemetry, registry and WebSocket radio."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from tests.conftest import make_df, ready_session, upload, wait_job
from tests.test_api import assert_error

U = "/api/v1/sessions"


def train(client: TestClient, sid: str, models: list[str], **extra):
    r = client.post(f"{U}/{sid}/train", json={"models": models, **extra})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "STARTED"
    return wait_job(client, sid, body["job_id"])


def test_registry_catalog(client: TestClient) -> None:
    body = client.get("/api/v1/models").json()
    assert body["total"] == 23
    by = {m["abbr"]: m for m in body["models"]}
    assert by["XGB"]["key"] == "xgboost" and by["XGB"]["family"] == "ENS" and by["XGB"]["iterative"]
    assert by["TAB"]["available"] is True and by["TAB"]["regression"] is True and by["TAB"]["tunable"] is True
    assert by["PTM"]["family"] == "NN" and "hidden" in by["PTM"]["search_space"]
    assert "thresholded at 0.5" in by["LNR"]["note"] and by["LNR"]["hyperparams_text"] == "fit_intercept=True"
    sid = ready_session(client)
    assert all(m["state"] in {"AVAIL", "N/A"} for m in client.get(f"{U}/{sid}/models").json()["models"])


def test_train_refuses_nans_and_bad_models(client: TestClient) -> None:
    sid = ready_session(client, clean=False)
    assert_error(client.post(f"{U}/{sid}/train", json={"models": ["xgboost"]}), 409, "NAN_REMAINING")
    client.post(f"{U}/{sid}/clean", json={"strategy": "drop"})
    assert_error(client.post(f"{U}/{sid}/train", json={"models": ["nope"]}), 400, "UNKNOWN_MODEL")
    assert_error(client.post(f"{U}/{sid}/train", json={"models": ["lr"], "tune": True, "tune_trials": 2}), 422)
    assert_error(client.post(f"{U}/{sid}/train", json={"models": []}), 422, "VALIDATION_ERROR")
    assert_error(client.post(f"{U}/{sid}/train", json={"models": ["lr"], "test_size": 0.9}), 422)
    assert_error(client.get(f"{U}/{sid}/train/zzz"), 404, "JOB_NOT_FOUND")


def test_full_training_flow(client: TestClient) -> None:
    sid = ready_session(client)
    job = train(client, sid, ["xgboost", "random_forest", "svm", "mlp", "logistic_regression"])
    assert job["status"] == "COMPLETE" and job["done"] == job["total"] == 5
    res = job["result"]
    board = res["leaderboard"]
    assert [r["rank"] for r in board] == [1, 2, 3, 4, 5]
    f1s = [r["f1"] for r in board]
    assert f1s == sorted(f1s, reverse=True)
    assert res["top_model"]["f1"] == f1s[0] and board[0]["gap_to_top"] == 0.0
    for row in board:
        assert row["gap_to_top"] == round(f1s[0] - row["f1"], 6)
        assert 0 <= row["accuracy"] <= 1 and row["train_time_sec"] >= 0 and len(row["confusion_matrix"]) == 2
        assert row["cte"] == ["DONE", "DONE", "DONE"] or "WARN" in row["cte"]
        assert row["underfit_warning"] in (None, "UNDERFIT DETECTED")
        assert 0 <= row["train_balanced_accuracy"] <= 1
        assert row["train_test_gap"] == round(row["train_balanced_accuracy"] - row["test_balanced_accuracy"], 6)
        assert isinstance(row["curve_overfit"], bool)
        if row["gap_check"] and row["train_test_gap"] > 0.10:
            assert row["overfit_warning"] == "OVERFIT DETECTED"
    assert res["split"]["stratified"] and res["split"]["train_rows"] + res["split"]["test_rows"] == 400
    assert "customer_id" in res["exclusions"]["id_like"] and res["failed_models"] == []
    assert client.get(f"{U}/{sid}").json()["stage"] == "TRAINED"
    assert client.get(f"{U}/{sid}/leaderboard").json()["leaderboard"] == board
    states = {m["abbr"]: m["state"] for m in client.get(f"{U}/{sid}/models").json()["models"]}
    assert states["XGB"] == "TRAINED" and states["CAT"] == "AVAIL" and states["TAB"] == "AVAIL"


def test_telemetry_for_iterative_models(client: TestClient) -> None:
    sid = ready_session(client, make_df(900))
    train(client, sid, ["xgboost", "mlp", "random_forest"])
    for name in ("xgboost", "MLP"):
        t = client.get(f"{U}/{sid}/telemetry/{name}").json()
        n = len(t["epochs"])
        assert 2 <= n <= 40 and t["epochs"][0] == 0 and t["epochs"][-1] == t["total_epochs"] - 1
        assert len(t["train_loss"]) == len(t["val_loss"]) == len(t["val_accuracy"]) == n
        assert t["overfit_warning"] in (None, "OVERFIT DETECTED")
    xgb = client.get(f"{U}/{sid}/telemetry/xgboost").json()
    assert xgb["total_epochs"] == 200 and xgb["train_loss"][-1] < xgb["train_loss"][0]
    assert xgb["overfit_warning"] == "OVERFIT DETECTED" and xgb["first_detected_epoch"] is not None  # deep trees overfit
    rf = client.get(f"{U}/{sid}/telemetry/random_forest").json()  # no epoch curve -> learning curve instead
    assert rf["kind"] == "learning_curve" and rf["metric"] == "balanced_accuracy"
    n = len(rf["train_sizes"])
    assert n >= 2 and n == len(rf["train_score"]) == len(rf["val_score"])
    assert rf["train_sizes"] == sorted(rf["train_sizes"]) and all(0 <= v <= 1 for v in rf["train_score"] + rf["val_score"])
    assert xgb["kind"] == "epochs"
    assert_error(client.get(f"{U}/{sid}/telemetry/svm"), 404, "MODEL_NOT_TRAINED")
    assert_error(client.get(f"{U}/{sid}/telemetry/bogus"), 400, "UNKNOWN_MODEL")
    flagged = {r["abbr"]: r["overfit_warning"] for r in client.get(f"{U}/{sid}/leaderboard").json()["leaderboard"]}
    board = {r["abbr"]: r for r in client.get(f"{U}/{sid}/leaderboard").json()["leaderboard"]}
    assert flagged["XGB"] == "OVERFIT DETECTED"
    # RFR memorises its training rows by design: the gap check is skipped and it has no curve
    assert board["RFR"]["gap_check"] is False and flagged["RFR"] is None
    assert board["RFR"]["first_detected_epoch"] is None


def test_merge_runs_and_reset_on_settings_change(client: TestClient) -> None:
    sid = ready_session(client)
    train(client, sid, ["logistic_regression"])
    job = train(client, sid, ["decision_tree", "knn", "naive_bayes", "lda", "ridge", "sgd", "linear_regression"])
    assert {r["abbr"] for r in job["result"]["leaderboard"]} == {"LR", "DT", "KNN", "NB", "LDA", "RDG", "SGD", "LNR"}
    job = train(client, sid, ["logistic_regression"], test_size=0.3)  # different settings -> fresh board
    assert [r["abbr"] for r in job["result"]["leaderboard"]] == ["LR"]


def test_extended_models_and_imbalance_modes(client: TestClient) -> None:
    sid = ready_session(client)
    allm = ["lightgbm", "catboost", "extra_trees", "gradient_boosting", "adaboost", "bagging", "stacking",
            "voting", "qda"]
    job = train(client, sid, allm, handle_imbalance="class_weight")
    assert job["status"] == "COMPLETE"
    done = {r["abbr"] for r in job["result"]["leaderboard"]} | {f["model"] for f in job["result"]["failed_models"]}
    assert len(done) == len(allm)
    job = train(client, sid, ["xgboost", "svm", "mlp"], handle_imbalance="smote")
    assert job["status"] == "COMPLETE" and len(job["result"]["leaderboard"]) == 3


def test_one_model_failing_does_not_abort_others(client: TestClient, monkeypatch) -> None:
    from app.services import model_registry as reg

    original = reg._BY_KEY["svm"]

    def boom(*_a, **_k):
        raise ValueError("kaboom")

    monkeypatch.setitem(reg._BY_KEY, "svm", type(original)(**{**original.__dict__, "factory": boom}))
    from app.routers import training as tr
    monkeypatch.setattr(tr, "resolve_model", reg.resolve_model)
    sid = ready_session(client)
    job = train(client, sid, ["logistic_regression", "svm", "decision_tree"])
    assert job["status"] == "COMPLETE" and len(job["result"]["leaderboard"]) == 2
    assert job["result"]["failed_models"][0]["model"] == "svm" and "kaboom" in job["result"]["failed_models"][0]["error"]
    assert job["model_status"]["SVM"] == ["DONE", "FAILED", "PENDING"]


def test_all_models_failing_marks_job_failed(client: TestClient, monkeypatch) -> None:
    from app.services import model_registry as reg
    from app.routers import training as tr

    spec = reg._BY_KEY["decision_tree"]
    monkeypatch.setitem(reg._BY_KEY, "decision_tree",
                        type(spec)(**{**spec.__dict__, "factory": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))}))
    monkeypatch.setattr(tr, "resolve_model", reg.resolve_model)
    sid = ready_session(client)
    job = train(client, sid, ["decision_tree"])
    assert job["status"] == "FAILED" and job["error"] == "ALL MODELS FAILED" and job["result"]["leaderboard"] == []
    assert_error(client.get(f"{U}/{sid}/leaderboard"), 409, "STAGE_CONFLICT")


def test_second_job_while_running_is_409(client: TestClient) -> None:
    sid = ready_session(client, make_df(900))
    first = client.post(f"{U}/{sid}/train", json={"models": ["mlp", "svm"]}).json()
    assert_error(client.post(f"{U}/{sid}/train", json={"models": ["lr"]}), 409, "TRAINING_IN_PROGRESS")
    wait_job(client, sid, first["job_id"])


def test_tiny_dataset_and_multiclass(client: TestClient) -> None:
    df = make_df(14, missing=0)
    sid = ready_session(client, df, clean=False)
    job = train(client, sid, ["logistic_regression", "decision_tree"])
    assert job["status"] == "COMPLETE" and any("small" in w.lower() for w in job["result"]["warnings"])
    rng = np.random.default_rng(1)
    multi = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    multi["churn"] = np.where(multi.a > 0.5, "hi", np.where(multi.b > 0, "mid", "lo"))
    sid = ready_session(client, multi, clean=False)
    job = train(client, sid, ["xgboost", "mlp", "random_forest", "linear_regression"], average="binary")
    assert job["status"] == "COMPLETE" and len(job["result"]["leaderboard"][0]["confusion_matrix"]) == 3
    assert job["result"]["average"] == "weighted" and any("binary" in w for w in job["result"]["warnings"])


def test_rare_class_falls_back_to_unstratified(client: TestClient) -> None:
    df = make_df(60, missing=0)
    df["churn"] = "No"
    df.loc[0, "churn"] = "Yes"
    sid = ready_session(client, df, clean=False)
    job = train(client, sid, ["logistic_regression"])
    assert job["result"]["split"]["stratified"] is False and any("Stratified" in w for w in job["result"]["warnings"])


def _regression_session(client: TestClient, rows: int = 500) -> str:
    """A continuous target (``total_charges``) predicted from the other columns."""
    sid = upload(client, make_df(rows, missing=0))
    r = client.post(f"{U}/{sid}/target", json={"target_column": "total_charges"})
    assert r.json()["task"] == "regression"
    assert client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "median"}).status_code == 200
    return sid


def test_regression_training_flow(client: TestClient) -> None:
    sid = _regression_session(client)
    job = train(client, sid, ["xgboost", "random_forest", "mlp", "svm", "linear_regression", "ridge", "decision_tree"])
    assert job["status"] == "COMPLETE", job
    result = job["result"]
    assert result["task"] == "regression" and result["failed_models"] == []
    board = result["leaderboard"]
    assert {r["abbr"] for r in board} == {"XGB", "RFR", "MLP", "SVM", "LNR", "RDG", "DT"}
    r2s = [r["r2"] for r in board]
    assert r2s == sorted(r2s, reverse=True)  # ranked by R2
    for row in board:
        assert row["task"] == "regression" and row["score_name"] == "R2" and row["score"] == row["r2"]
        assert row["f1"] is None and row["accuracy"] is None and row["confusion_matrix"] == []
        assert row["rmse"] >= 0 and row["mae"] >= 0 and row["rmse"] >= row["mae"]
        assert row["score_metric"] == "r2"
        assert abs(row["train_test_gap"] - (row["train_score"] - row["test_score"])) < 2e-6
        assert row["train_balanced_accuracy"] is None
    assert board[0]["gap_to_top"] == 0.0 and result["top_model"]["score_name"] == "R2"
    assert result["split"]["stratified"] is False
    # total_charges ~ tenure * monthly_charges is learnable: the best model must beat the mean predictor by far
    assert board[0]["r2"] > 0.8
    # RMSE is reported in the target's own units (total_charges is in the thousands), not standardised ones
    assert max(r["rmse"] for r in board) > 10
    xgb = client.get(f"{U}/{sid}/telemetry/xgboost").json()
    assert xgb["kind"] == "epochs" and xgb["val_metric"] == "r2" and xgb["total_epochs"] == 200
    mlp = client.get(f"{U}/{sid}/telemetry/mlp").json()
    assert mlp["val_metric"] == "r2" and len(mlp["epochs"]) >= 2
    rf = client.get(f"{U}/{sid}/telemetry/random_forest").json()
    assert rf["kind"] == "learning_curve" and rf["metric"] == "r2"
    assert all(-1 <= v <= 1 for v in rf["train_score"] + rf["val_score"])
    assert client.get(f"{U}/{sid}").json()["task"] == "regression"
    models = {m["abbr"]: m for m in client.get(f"{U}/{sid}/models").json()["models"]}
    assert models["LR"]["regression"] is False and models["LNR"]["regression"] is True
    assert models["RFR"]["last_score"] == next(r["r2"] for r in board if r["abbr"] == "RFR")


def test_regression_rejects_classifiers_and_ignores_imbalance(client: TestClient) -> None:
    sid = _regression_session(client, 200)
    for abbr in ("lr", "naive_bayes", "lda", "qda"):
        assert_error(client.post(f"{U}/{sid}/train", json={"models": [abbr]}), 400, "MODEL_NOT_FOR_TASK")
    job = train(client, sid, ["ridge"], handle_imbalance="smote")
    assert job["status"] == "COMPLETE"
    assert any("only applies to classification" in w for w in job["result"]["warnings"])


def test_forcing_classification_on_continuous_target_is_rejected(client: TestClient) -> None:
    sid = upload(client, make_df(80, missing=0))
    client.post(f"{U}/{sid}/target", json={"target_column": "monthly_charges", "task": "classification"})
    assert_error(client.post(f"{U}/{sid}/train", json={"models": ["lr"]}), 400, "TARGET_NOT_CLASSIFIABLE")


# --- radio ---------------------------------------------------------------
def collect(ws, stop_on: str, limit: int = 80) -> list[dict]:
    lines = []
    for _ in range(limit):
        msg = ws.receive_json()
        if "message" in msg:
            lines.append(msg)
            if stop_on in msg["message"]:
                return lines
    raise AssertionError(f"never saw {stop_on}: {lines[-3:]}")


def test_ws_receives_clean_started(client: TestClient) -> None:
    sid = ready_session(client, clean=False)
    with client.websocket_connect(f"/ws/radio?session_id={sid}") as ws:
        client.post(f"{U}/{sid}/clean", json={"strategy": "impute", "method": "mean"})
        lines = collect(ws, "CLEAN DATA STARTED")
    last = lines[-1]
    assert set(last) == {"timestamp", "level", "stage", "message", "ts_iso"}
    assert last["stage"] == "CLEAN" and last["timestamp"].startswith("T+") and last["message"].isupper()


def test_ws_replay_training_events_and_logs_rest(client: TestClient) -> None:
    sid = ready_session(client, make_df(600))
    with client.websocket_connect(f"/ws/radio?session_id={sid}") as ws:
        train(client, sid, ["logistic_regression", "xgboost"])
        seen = collect(ws, "TRAINING COMPLETE")
    text = " | ".join(m["message"] for m in seen)
    for needle in ("LR TRAINING STARTED", "LR EVALUATION COMPLETE F1=", "XGB EVALUATION COMPLETE", "OVERFIT DETECTED ON XGB",
                   "TRAINING COMPLETE TOP MODEL"):
        assert needle in text, needle
    rest = client.get(f"{U}/{sid}/logs").json()["logs"]
    assert any("UPLOAD COMPLETE" in m["message"] for m in rest) and rest[-1]["message"].startswith("TRAINING COMPLETE")
    with client.websocket_connect(f"/ws/radio?session_id={sid}") as ws2:  # reconnect replays history
        replay = [ws2.receive_json() for _ in range(min(50, len(rest)))]
    assert replay[-1]["message"] == rest[-1]["message"] if len(rest) <= 50 else True


def test_ws_unknown_session_and_global_channel_and_heartbeat(client: TestClient) -> None:
    with client.websocket_connect("/ws/radio?session_id=nope") as ws:
        assert ws.receive_json()["error_code"] == "SESSION_NOT_FOUND"
    with client.websocket_connect("/ws/radio") as ws:  # server-level channel, heartbeat every 0.3 s in tests
        assert ws.receive_json()["type"] == "heartbeat"
        upload(client)
        got = []
        for _ in range(6):
            m = ws.receive_json()
            got.append(m)
            if "message" in m and "OPENED" in m["message"]:
                break
        assert any("message" in m and "OPENED" in m["message"] for m in got)


def test_ws_multiple_clients_and_dead_client_do_not_block(client: TestClient) -> None:
    sid = ready_session(client, clean=False)
    with client.websocket_connect(f"/ws/radio?session_id={sid}") as a, client.websocket_connect(f"/ws/radio?session_id={sid}") as b:
        with client.websocket_connect(f"/ws/radio?session_id={sid}"):
            pass  # third client disconnects immediately
        client.post(f"{U}/{sid}/clean", json={"strategy": "drop"})
        assert "CLEAN DATA STARTED" in collect(a, "CLEAN DATA STARTED")[-1]["message"]
        assert "CLEAN DATA STARTED" in collect(b, "CLEAN DATA STARTED")[-1]["message"]
        assert client.app.state.radio.client_count == 2
