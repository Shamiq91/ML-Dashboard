"""Model lab catalog: the eight lab models, their hand-tunable hyper-parameters and how to build each estimator.

Every control the lab page shows comes from here (kind, range, default, help text), so the UI renders it generically
and the backend validates the same definition it advertises.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.errors import BadRequestError

CLASSIFICATION = "classification"
REGRESSION = "regression"
CLUSTERING = "clustering"


@dataclass(frozen=True)
class Param:
    """One hyper-parameter control.

    ``kind``: ``int`` | ``float`` | ``log`` (float on a log slider) | ``cat`` | ``bool``.
    ``when`` shows the control only while another parameter has one of the listed values.
    ``tasks`` limits the control to some tasks (empty = all). ``zero_means`` labels a special 0 (e.g. no depth limit).
    """

    name: str
    label: str
    kind: str
    default: Any
    help: str
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] = ()
    when: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    tasks: tuple[str, ...] = ()
    zero_means: str | None = None
    sweep: tuple[Any, ...] = ()  # default values for a sweep over this parameter (empty = not sweepable)

    def applies(self, task: str) -> bool:
        """True when this control is used for ``task``."""
        return not self.tasks or task in self.tasks

    def coerce(self, raw: Any) -> Any:
        """Validate one incoming value (clamped to the range; bad values raise 400)."""
        if raw is None:
            return self.default
        try:
            if self.kind == "bool":
                if isinstance(raw, str):
                    return raw.strip().lower() in {"1", "true", "yes", "on"}
                return bool(raw)
            if self.kind == "cat":
                for choice in self.choices:
                    if raw == choice or str(raw) == str(choice):
                        return choice
                raise ValueError(raw)
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(raw)
            if self.lo is not None:
                value = max(value, self.lo)
            if self.hi is not None:
                value = min(value, self.hi)
            return int(round(value)) if self.kind == "int" else value
        except (TypeError, ValueError) as exc:
            raise BadRequestError(
                f"Invalid value {raw!r} for '{self.name}'.", error_code="BAD_PARAM",
                details={"param": self.name, "kind": self.kind, "choices": list(self.choices), "lo": self.lo, "hi": self.hi},
            ) from exc

    def public(self) -> dict[str, Any]:
        """JSON description for the UI."""
        return {
            "name": self.name, "label": self.label, "kind": self.kind, "default": self.default, "help": self.help,
            "lo": self.lo, "hi": self.hi, "step": self.step, "choices": list(self.choices),
            "when": {k: list(v) for k, v in self.when.items()}, "tasks": list(self.tasks), "zero_means": self.zero_means,
            "sweep": list(self.sweep),
        }


@dataclass(frozen=True)
class LabModel:
    """One model the lab can open."""

    key: str
    abbr: str
    name: str
    tasks: tuple[str, ...]
    summary: str
    params: tuple[Param, ...]
    default_sweep: tuple[str, ...]  # one parameter (validation curve) or two (score grid)
    scaled: bool = True  # whether scaling changes this model's behaviour (shown as a hint)
    task_note: str | None = None

    def params_for(self, task: str) -> list[Param]:
        """Controls that apply to ``task``."""
        return [p for p in self.params if p.applies(task)]

    def resolve(self, raw: dict[str, Any] | None, task: str) -> dict[str, Any]:
        """Defaults overlaid with validated user values (unknown names are ignored)."""
        raw = raw or {}
        return {p.name: p.coerce(raw.get(p.name)) for p in self.params_for(task)}

    def sweepable(self, task: str) -> dict[str, Param]:
        """Parameters that can be swept for ``task``."""
        return {p.name: p for p in self.params_for(task) if p.sweep}

    def public(self) -> dict[str, Any]:
        """JSON description for the UI."""
        return {
            "key": self.key, "abbr": self.abbr, "name": self.name, "tasks": list(self.tasks), "summary": self.summary,
            "params": [p.public() for p in self.params], "default_sweep": list(self.default_sweep),
            "scaled": self.scaled, "task_note": self.task_note,
        }


_CW = Param("class_weight", "Class weight", "cat", "none",
            "balanced re-weights rows so rare classes count as much as common ones.",
            choices=("none", "balanced"), tasks=(CLASSIFICATION,))

LAB_MODELS: dict[str, LabModel] = {
    m.key: m
    for m in (
        LabModel(
            "linear_regression", "LNR", "Linear regression", (REGRESSION,),
            "Fits the straight-line relationship y = b0 + b1*x1 + ... that minimises squared error. Each coefficient is "
            "the change in the target for a one-unit (one standard deviation, when scaled) change in that feature, "
            "holding the others fixed. Ridge and lasso add a penalty that shrinks coefficients; lasso can set them to 0.",
            (
                Param("regularization", "Regularization", "cat", "none",
                      "none = ordinary least squares (with confidence intervals and p-values). ridge shrinks every "
                      "coefficient, lasso drops weak features to exactly 0, elasticnet mixes both.",
                      choices=("none", "ridge", "lasso", "elasticnet")),
                Param("alpha", "Alpha (penalty strength)", "log", 0.1,
                      "Higher alpha = stronger shrinkage = simpler model. Too high and every coefficient collapses to 0 "
                      "(the target is standardised, so lasso usually needs alpha below 1).",
                      lo=1e-4, hi=100.0, when={"regularization": ("ridge", "lasso", "elasticnet")},
                      sweep=(1e-4, 1e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)),
                Param("l1_ratio", "L1 ratio", "float", 0.5, "0 = pure ridge, 1 = pure lasso.",
                      lo=0.0, hi=1.0, step=0.05, when={"regularization": ("elasticnet",)}),
                Param("fit_intercept", "Fit intercept", "bool", True,
                      "Off forces the line through the origin; almost always leave it on."),
            ),
            ("alpha",), task_note="Needs a numeric target (task: regression).",
        ),
        LabModel(
            "logistic_regression", "LR", "Logistic regression", (CLASSIFICATION,),
            "Models the log-odds of the positive class as a straight-line function of the features, then squashes it "
            "through a sigmoid to get a probability. exp(coefficient) is the odds ratio: how much one unit (one "
            "standard deviation, when scaled) of a feature multiplies the odds.",
            (
                Param("penalty", "Penalty", "cat", "l2",
                      "l2 shrinks all coefficients, l1 can zero some out (built-in feature selection), none = plain "
                      "maximum likelihood (shows confidence intervals and p-values for binary targets).",
                      choices=("l2", "l1", "elasticnet", "none")),
                Param("C", "C (inverse penalty)", "log", 1.0,
                      "Small C = strong regularization (simpler, may underfit). Large C = weak regularization.",
                      lo=1e-3, hi=1000.0, when={"penalty": ("l2", "l1", "elasticnet")},
                      sweep=(1e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0)),
                Param("l1_ratio", "L1 ratio", "float", 0.5, "0 = pure l2, 1 = pure l1.",
                      lo=0.0, hi=1.0, step=0.05, when={"penalty": ("elasticnet",)}),
                _CW,
                Param("max_iter", "Max iterations", "int", 1000, "Solver iterations; raise it if you see a convergence note.",
                      lo=50, hi=5000, step=50),
            ),
            ("C",), task_note="Needs a categorical target (task: classification).",
        ),
        LabModel(
            "knn", "KNN", "K-nearest neighbors", (CLASSIFICATION, REGRESSION),
            "Stores the training rows and predicts a new row from its k closest neighbours: a majority vote for "
            "classes, an average for numbers. Distances depend on feature scale, so scaling matters a lot here.",
            (
                Param("n_neighbors", "k (neighbours)", "int", 15,
                      "Small k follows the data closely (noisy, overfits); large k smooths (may underfit).",
                      lo=1, hi=100, step=1, sweep=(1, 2, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 75)),
                Param("weights", "Weights", "cat", "uniform",
                      "distance gives closer neighbours a bigger say.", choices=("uniform", "distance")),
                Param("metric", "Distance metric", "cat", "euclidean",
                      "euclidean = straight line, manhattan = sum of differences (robust to outliers), "
                      "chebyshev = largest single difference.",
                      choices=("euclidean", "manhattan", "chebyshev")),
            ),
            ("n_neighbors",),
        ),
        LabModel(
            "xgboost", "XGB", "XGBoost", (CLASSIFICATION, REGRESSION),
            "Gradient-boosted trees: adds small trees one at a time, each one fitted to the errors left by the trees "
            "before it, scaled down by the learning rate. Regularization (lambda, alpha, gamma) and row/column "
            "subsampling keep it from memorising the data.",
            (
                Param("n_estimators", "Trees (rounds)", "int", 200, "Maximum boosting rounds.", lo=10, hi=1000, step=10,
                      sweep=(25, 50, 100, 200, 300, 500)),
                Param("learning_rate", "Learning rate (eta)", "log", 0.1,
                      "How much each tree contributes. Lower = needs more trees but generalises better.",
                      lo=0.005, hi=0.5, sweep=(0.01, 0.03, 0.1, 0.3)),
                Param("max_depth", "Max depth", "int", 6, "Depth of each tree; deeper trees capture interactions but overfit.",
                      lo=1, hi=12, step=1, sweep=(2, 3, 4, 6, 8, 10)),
                Param("min_child_weight", "Min child weight", "float", 1.0,
                      "Minimum evidence needed to create a leaf; higher = more conservative.", lo=0.0, hi=20.0, step=0.5,
                      sweep=(0.0, 1.0, 3.0, 5.0, 10.0, 20.0)),
                Param("subsample", "Row subsample", "float", 1.0, "Fraction of rows each tree sees.", lo=0.3, hi=1.0, step=0.05,
                      sweep=(0.4, 0.6, 0.8, 1.0)),
                Param("colsample_bytree", "Column subsample", "float", 1.0, "Fraction of features each tree sees.",
                      lo=0.3, hi=1.0, step=0.05, sweep=(0.4, 0.6, 0.8, 1.0)),
                Param("reg_lambda", "L2 (lambda)", "log", 1.0, "L2 penalty on leaf weights.", lo=1e-3, hi=100.0,
                      sweep=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)),
                Param("reg_alpha", "L1 (alpha)", "float", 0.0, "L1 penalty on leaf weights.", lo=0.0, hi=10.0, step=0.1),
                Param("gamma", "Gamma (min split loss)", "float", 0.0, "A split must reduce loss by at least this much.",
                      lo=0.0, hi=10.0, step=0.1, sweep=(0.0, 0.5, 1.0, 2.0, 5.0)),
                Param("early_stopping_rounds", "Early stopping", "int", 0,
                      "Stop when validation loss has not improved for this many rounds.", lo=0, hi=100, step=5,
                      zero_means="off"),
            ),
            ("max_depth", "learning_rate"), scaled=False,
        ),
        LabModel(
            "decision_tree", "DT", "Decision tree", (CLASSIFICATION, REGRESSION),
            "Splits the data with yes/no questions (feature <= threshold), choosing at each step the question that "
            "makes the groups purest. Each leaf predicts its majority class (or mean value). Fully readable, but a "
            "deep tree memorises the training rows.",
            (
                Param("max_depth", "Max depth", "int", 6, "Longest question chain.", lo=0, hi=30, step=1, zero_means="no limit",
                      sweep=(1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 0)),
                Param("min_samples_leaf", "Min samples per leaf", "int", 1, "Leaves smaller than this are not allowed.",
                      lo=1, hi=200, step=1, sweep=(1, 2, 5, 10, 20, 50, 100, 200)),
                Param("min_samples_split", "Min samples to split", "int", 2, "A node needs this many rows to be split.",
                      lo=2, hi=400, step=1, sweep=(2, 5, 10, 20, 50, 100, 200, 400)),
                Param("criterion", "Criterion", "cat", "gini", "How purity is measured.",
                      choices=("gini", "entropy", "log_loss"), tasks=(CLASSIFICATION,)),
                Param("criterion", "Criterion", "cat", "squared_error", "How split quality is measured.",
                      choices=("squared_error", "friedman_mse", "absolute_error"), tasks=(REGRESSION,)),
                Param("max_features", "Max features", "cat", "all", "Features considered at each split.",
                      choices=("all", "sqrt", "log2")),
                Param("ccp_alpha", "Pruning (ccp_alpha)", "float", 0.0,
                      "Cost-complexity pruning: removes branches whose gain is below alpha.", lo=0.0, hi=0.05, step=0.0005),
                _CW,
            ),
            ("max_depth",), scaled=False,
        ),
        LabModel(
            "random_forest", "RFR", "Random forest", (CLASSIFICATION, REGRESSION),
            "Trains many deep trees, each on a bootstrap sample of rows and a random subset of features at each "
            "split, then averages them. Averaging decorrelated trees cancels much of each tree's overfitting. "
            "Rows a tree never saw (out-of-bag) give a free validation score.",
            (
                Param("n_estimators", "Trees", "int", 200, "More trees = more stable, never worse, but slower.",
                      lo=10, hi=600, step=10, sweep=(10, 25, 50, 100, 200, 400)),
                Param("max_depth", "Max depth", "int", 0, "Depth of each tree.", lo=0, hi=40, step=1, zero_means="no limit",
                      sweep=(2, 4, 6, 8, 10, 15, 20, 0)),
                Param("min_samples_leaf", "Min samples per leaf", "int", 1, "Bigger leaves = smoother forest.",
                      lo=1, hi=100, step=1, sweep=(1, 2, 5, 10, 20, 50)),
                Param("max_features", "Max features", "cat", "sqrt",
                      "Features tried at each split; fewer = more decorrelated trees.",
                      choices=("sqrt", "log2", "0.3", "0.5", "all"), sweep=("sqrt", "log2", "0.3", "0.5", "all")),
                Param("bootstrap", "Bootstrap (+ out-of-bag score)", "bool", True,
                      "Each tree sees a resample of the rows; enables the out-of-bag score."),
                _CW,
            ),
            ("max_depth",), scaled=False,
        ),
        LabModel(
            "kmeans", "KMN", "K-means", (CLUSTERING,),
            "Unsupervised: groups rows into k clusters by repeatedly assigning each row to its nearest centre and "
            "moving each centre to the mean of its rows. It never sees the target; when a target is set the lab "
            "shows how the clusters line up with it.",
            (
                Param("n_clusters", "k (clusters)", "int", 3, "Number of clusters.", lo=2, hi=20, step=1,
                      sweep=(2, 3, 4, 5, 6, 7, 8, 9, 10, 12)),
                Param("init", "Initialisation", "cat", "k-means++", "k-means++ spreads the starting centres out.",
                      choices=("k-means++", "random")),
                Param("n_init", "Restarts", "int", 10, "Runs with different starts; the best is kept.", lo=1, hi=30, step=1),
                Param("max_iter", "Max iterations", "int", 300, "Iterations per run.", lo=10, hi=1000, step=10),
                Param("algorithm", "Algorithm", "cat", "lloyd", "elkan is faster on well-separated clusters.",
                      choices=("lloyd", "elkan")),
            ),
            ("n_clusters",), task_note="Unsupervised: works without a target.",
        ),
        LabModel(
            "svm", "SVM", "Support vector machine", (CLASSIFICATION, REGRESSION),
            "Finds the boundary with the widest margin between classes; only the rows on or inside the margin "
            "(support vectors) define it. Kernels (rbf, poly) let the boundary curve. C trades margin width for "
            "training errors; gamma sets how far one row's influence reaches.",
            (
                Param("kernel", "Kernel", "cat", "rbf", "Shape of the boundary.", choices=("linear", "rbf", "poly", "sigmoid")),
                Param("C", "C", "log", 1.0, "High C = fewer training errors, narrower margin (overfit risk).",
                      lo=1e-3, hi=1000.0, sweep=(0.01, 0.1, 1.0, 10.0, 100.0)),
                Param("gamma_mode", "Gamma", "cat", "scale", "scale / auto pick gamma from the data; custom sets it.",
                      choices=("scale", "auto", "custom"), when={"kernel": ("rbf", "poly", "sigmoid")}),
                Param("gamma", "Gamma value", "log", 0.1, "High gamma = very local influence = wiggly boundary.",
                      lo=1e-4, hi=10.0, when={"gamma_mode": ("custom",)}, sweep=(0.001, 0.01, 0.1, 1.0)),
                Param("degree", "Degree", "int", 3, "Polynomial degree.", lo=2, hi=6, step=1, when={"kernel": ("poly",)}),
                Param("epsilon", "Epsilon (tube)", "float", 0.1,
                      "Errors smaller than epsilon (in standard deviations of the target) are ignored.",
                      lo=0.0, hi=1.0, step=0.01, tasks=(REGRESSION,)),
                _CW,
            ),
            ("C", "gamma"),
        ),
    )
}


def get_lab_model(key: str) -> LabModel:
    """Look up a lab model by key or abbreviation (400 for unknown names)."""
    token = key.strip().lower()
    for model in LAB_MODELS.values():
        if token in (model.key, model.abbr.lower()):
            return model
    raise BadRequestError(f"'{key}' is not a model-lab model.", error_code="UNKNOWN_LAB_MODEL",
                          details={"valid": list(LAB_MODELS)})


def lab_task(model: LabModel, session_task: str, has_target: bool) -> str:
    """The task the lab runs ``model`` as, or 400/409 with a reason it cannot run."""
    if CLUSTERING in model.tasks:
        return CLUSTERING
    if not has_target:
        from app.errors import StageConflictError

        raise StageConflictError(f"{model.name} needs a target column. Set one on the dashboard first.",
                                 error_code="TARGET_REQUIRED")
    if session_task not in model.tasks:
        raise BadRequestError(f"{model.name} cannot run on a {session_task} target. {model.task_note or ''}".strip(),
                              error_code="MODEL_NOT_FOR_TASK")
    return session_task


def _max_features(value: str) -> Any:
    if value == "all":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def build_estimator(model: LabModel, task: str, p: dict[str, Any], seed: int, n_jobs: int, *, fit_time: bool = False) -> Any:
    """A fresh estimator for ``model`` with resolved parameters ``p``.

    ``fit_time`` turns on settings only the main fit uses (XGBoost early stopping needs an eval set).
    """
    clf = task == CLASSIFICATION
    cw = p.get("class_weight")
    cw = "balanced" if cw == "balanced" else None
    key = model.key
    if key == "linear_regression":
        from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

        reg, fi = p["regularization"], p["fit_intercept"]
        if reg == "ridge":
            return Ridge(alpha=p["alpha"], fit_intercept=fi)
        if reg == "lasso":
            return Lasso(alpha=p["alpha"], fit_intercept=fi, max_iter=10000)
        if reg == "elasticnet":
            return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], fit_intercept=fi, max_iter=10000)
        return LinearRegression(fit_intercept=fi)
    if key == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        penalty = p["penalty"]
        if penalty == "none":
            return LogisticRegression(C=np.inf, max_iter=p["max_iter"], class_weight=cw)
        if penalty == "l2":
            return LogisticRegression(C=p["C"], l1_ratio=0.0, max_iter=p["max_iter"], class_weight=cw)
        ratio = 1.0 if penalty == "l1" else p["l1_ratio"]
        return LogisticRegression(C=p["C"], l1_ratio=ratio, solver="saga", max_iter=p["max_iter"], class_weight=cw,
                                  random_state=seed)
    if key == "knn":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        cls = KNeighborsClassifier if clf else KNeighborsRegressor
        return cls(n_neighbors=p["n_neighbors"], weights=p["weights"], metric=p["metric"], n_jobs=n_jobs)
    if key == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        kw: dict[str, Any] = dict(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            min_child_weight=p["min_child_weight"], subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"], reg_alpha=p["reg_alpha"], gamma=p["gamma"], tree_method="hist",
            random_state=seed, n_jobs=n_jobs, verbosity=0,
        )
        if fit_time and p["early_stopping_rounds"] > 0:
            kw["early_stopping_rounds"] = p["early_stopping_rounds"]
        return XGBClassifier(**kw) if clf else XGBRegressor(**kw)
    if key == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        kw = dict(
            max_depth=p["max_depth"] or None, min_samples_leaf=p["min_samples_leaf"],
            min_samples_split=p["min_samples_split"], criterion=p["criterion"],
            max_features=_max_features(p["max_features"]), ccp_alpha=p["ccp_alpha"], random_state=seed,
        )
        return DecisionTreeClassifier(**kw, class_weight=cw) if clf else DecisionTreeRegressor(**kw)
    if key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        kw = dict(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"] or None, min_samples_leaf=p["min_samples_leaf"],
            max_features=_max_features(p["max_features"]), bootstrap=p["bootstrap"], random_state=seed, n_jobs=n_jobs,
        )
        return RandomForestClassifier(**kw, class_weight=cw) if clf else RandomForestRegressor(**kw)
    if key == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=p["n_clusters"], init=p["init"], n_init=p["n_init"], max_iter=p["max_iter"],
                      algorithm=p["algorithm"], random_state=seed)
    if key == "svm":
        from sklearn.svm import SVC, SVR

        gamma: Any = p["gamma"] if p.get("gamma_mode") == "custom" else p.get("gamma_mode", "scale")
        kw = dict(kernel=p["kernel"], C=p["C"], gamma=gamma, degree=p["degree"])
        if clf:
            return SVC(**kw, class_weight=cw, random_state=seed)
        return SVR(**kw, epsilon=p["epsilon"])
    raise BadRequestError(f"No estimator for '{key}'.", error_code="UNKNOWN_LAB_MODEL")  # pragma: no cover


def catalog_payload() -> dict[str, Any]:
    """Everything the lab page needs to draw its controls."""
    return {"models": [m.public() for m in LAB_MODELS.values()]}
