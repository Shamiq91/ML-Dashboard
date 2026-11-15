"""Shared fixtures: an app per test (with lifespan) and small synthetic datasets."""
from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_df(rows: int = 400, seed: int = 3, missing: int = 5) -> pd.DataFrame:
    """Small churn-like frame with a learnable signal and a few NaNs."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, rows)
    monthly = rng.normal(65, 25, rows).round(2)
    calls = rng.poisson(1.6, rows)
    total = (tenure * monthly * rng.normal(1, 0.05, rows)).round(2)
    logit = -0.5 + 0.03 * (monthly - 65) - 0.05 * tenure + 0.4 * calls
    churn = np.where(rng.random(rows) < 1 / (1 + np.exp(-logit - 0.6)), "Yes", "No")
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(rows)],
            "plan": rng.choice(["a", "b", "c"], rows),
            "tenure": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": calls,
            "churn": churn,
        }
    )
    if missing:
        df.loc[rng.choice(rows, missing, replace=False), "total_charges"] = np.nan
    return df


def csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a frame to CSV bytes."""
    return df.to_csv(index=False).encode()


@pytest.fixture()
def settings(tmp_path: Any) -> Settings:
    """Fast test settings (each test gets its own run-history database)."""
    return Settings(max_upload_mb=1, ws_heartbeat_seconds=0.3, max_epochs=40, n_jobs=1, data_dir=str(tmp_path / "data"),
                    tune_timeout_sec=20, shap_max_samples=40, shap_background=30, perm_max_rows=200)


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    """TestClient with the lifespan running (needed for background jobs and the radio)."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def upload(client: TestClient, df: pd.DataFrame | None = None, name: str = "data.csv") -> str:
    """Upload a frame and return the session id."""
    df = make_df() if df is None else df
    response = client.post("/api/v1/sessions/upload", files={"file": (name, csv_bytes(df), "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def ready_session(client: TestClient, df: pd.DataFrame | None = None, clean: bool = True) -> str:
    """Upload, set target ``churn`` and (optionally) impute so training can start."""
    sid = upload(client, df)
    assert client.post(f"/api/v1/sessions/{sid}/target", json={"target_column": "churn"}).status_code == 200
    if clean:
        r = client.post(f"/api/v1/sessions/{sid}/clean", json={"strategy": "impute", "method": "median"})
        assert r.status_code == 200, r.text
    return sid


def wait_job(client: TestClient, sid: str, job_id: str, timeout: float = 120) -> dict[str, Any]:
    """Poll a job until it leaves RUNNING."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/sessions/{sid}/train/{job_id}").json()
        if body["status"] != "RUNNING":
            return body
        time.sleep(0.1)
    raise AssertionError("training job timed out")
