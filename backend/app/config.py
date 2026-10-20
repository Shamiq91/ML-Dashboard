"""Central configuration: every threshold and limit lives here, never inline."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, overridable with ``MLDASH_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="MLDASH_", env_file=".env", extra="ignore")

    app_name: str = "ML Pipeline Dashboard API"
    app_version: str = "1.0.0"

    # --- HTTP / CORS -------------------------------------------------------
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    frontend_dist: str | None = None  # defaults to ../frontend/dist when it exists

    # --- Upload / sessions -------------------------------------------------
    max_upload_mb: int = 50
    session_ttl_seconds: int = 2 * 60 * 60
    cleanup_interval_seconds: int = 60
    max_log_lines: int = 1000

    # --- Reproducibility / resources --------------------------------------
    random_seed: int = 42
    n_jobs: int = 2
    test_size: float = 0.2
    val_fraction_of_train: float = 0.2  # 0.8 * 0.2 = 16% -> 64/16/20 split
    min_rows_warning: int = 20
    min_rows_hard: int = 6
    # Kernel SVM scales ~O(n^2)-O(n^3); above this many training rows it fits on a stratified sample.
    svm_max_train_rows: int = 10000

    # --- Data-quality thresholds ------------------------------------------
    imbalance_threshold: float = 0.30
    multicollinearity_threshold: float = 0.85
    continuous_target_unique: int = 20
    onehot_max_cardinality: int = 10

    # --- Assumption tests --------------------------------------------------
    dw_pass_low: float = 1.5
    dw_pass_high: float = 2.5
    dw_warn_low: float = 1.0
    dw_warn_high: float = 3.0
    shapiro_alpha: float = 0.05
    strict_alpha: float = 0.01
    shapiro_max_samples: int = 5000

    # --- Telemetry ---------------------------------------------------------
    max_epochs: int = 200
    convergence_tol: float = 1e-4
    convergence_patience: int = 10
    telemetry_max_points: int = 40
    overfit_window: int = 3
    # Overfit (any model) when balanced accuracy on training rows beats the test set by more than this.
    overfit_max_gap: float = 0.10
    # Underfit when the model barely beats chance on its own training rows:
    # skill = (balanced_acc - 1/k) / (1 - 1/k) below this value.
    underfit_min_skill: float = 0.2
    underfit_max_eval_rows: int = 20000

    # Learning curves for models without a per-epoch loss curve (train vs validation score by training size).
    learning_curves: bool = True
    learning_curve_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    learning_curve_min_rows: int = 40
    learning_curve_time_budget_sec: float = 30.0

    # --- Hyper-parameter tuning (Optuna) --------------------------------------------
    tune_max_trials: int = 60
    tune_cv_folds: int = 3
    tune_max_rows: int = 4000  # tuning CV runs on at most this many (stratified / most recent) training rows
    tune_timeout_sec: float = 60.0  # per model

    # --- Cross-validation of every trained model (training rows only) -----------------
    cv_max_rows: int = 5000  # each model is cross-validated on at most this many (stratified / most recent) training rows

    # --- Model analysis (curves, importance, SHAP) ---------------------------
    curve_max_points: int = 200
    threshold_points: int = 101
    perm_repeats: int = 5
    perm_max_rows: int = 1500
    shap_max_samples: int = 150
    shap_background: int = 80
    shap_kernel_rows: int = 60  # rows explained by the (slow) model-agnostic KernelExplainer
    shap_kernel_nsamples: int = 120
    shap_top_features: int = 12

    # --- Data preview / run history ------------------------------------------
    preview_max_limit: int = 500
    profile_histogram_bins: int = 12
    data_dir: str | None = None  # run-history database lives here (default: backend/data)
    run_db_name: str = "mldashboard.db"
    curve_points_stored: int = 60  # ROC/PR points kept per model in a saved run

    # --- WebSocket ---------------------------------------------------------
    log_replay: int = 50
    ws_heartbeat_seconds: float = 25.0
    ws_send_timeout_seconds: float = 2.0

    @property
    def max_upload_bytes(self) -> int:
        """Upload limit in bytes."""
        return self.max_upload_mb * 1024 * 1024

    def resolved_data_dir(self) -> Path:
        """Directory holding the SQLite run history (created on demand)."""
        path = Path(self.data_dir) if self.data_dir else Path(__file__).resolve().parents[1] / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolved_frontend_dist(self) -> Path | None:
        """Return the built frontend directory if one exists."""
        candidate = (
            Path(self.frontend_dist)
            if self.frontend_dist
            else Path(__file__).resolve().parents[2] / "frontend" / "dist"
        )
        return candidate if (candidate / "index.html").exists() else None


@lru_cache
def get_settings() -> Settings:
    """Return the cached process-wide settings object."""
    return Settings()
