"""Catalog of every model the pipeline can run (23 classifiers; 19 also have a regression counterpart)."""
from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression

from app.errors import BadRequestError
from app.services.tasks import is_regression

Factory = Callable[[int, int, dict[str, Any]], Any]


class ThresholdedLinearRegression(ClassifierMixin, BaseEstimator):
    """Linear regression used as a classifier: binary output thresholded at 0.5.

    For more than two classes the prediction is rounded and clipped to the label range.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "ThresholdedLinearRegression":
        """Fit OLS on the label-encoded target."""
        self.classes_ = np.unique(y)
        self.model_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Raw regression output (binary only), usable as a ranking score for ROC/PR curves."""
        return self.model_.predict(X)

    def predict(self, X: Any) -> np.ndarray:
        """Threshold (binary) or round (multiclass) the regression output."""
        raw = self.model_.predict(X)
        if len(self.classes_) == 2:
            return np.where(raw >= 0.5, self.classes_[1], self.classes_[0])
        return np.clip(np.rint(raw), self.classes_.min(), self.classes_.max()).astype(self.classes_.dtype)


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one model."""

    key: str
    abbr: str
    name: str
    family: str
    hw: str
    hyperparams: dict[str, Any]
    est_time: str
    factory: Factory | None
    scaled: bool = False
    iterative: bool = False
    # Fits its own training rows ~perfectly by design (deep/unpruned trees), so a train-vs-test gap says nothing.
    memorizes: bool = False
    requires: str | None = None
    unavailable_reason: str | None = None
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def supports(self, task: str) -> bool:
        """Whether the model can run for ``task`` (``classification`` or ``regression``)."""
        return not is_regression(task) or self.key in _REGRESSORS

    def availability(self) -> tuple[bool, str | None]:
        """Return ``(available, reason_if_not)``."""
        if self.requires and importlib.util.find_spec(self.requires) is None:
            return False, self.unavailable_reason or f"needs {self.requires}"
        if self.factory is None:
            return False, self.unavailable_reason or "not available"
        return True, None

    def search_space(self) -> dict[str, tuple[Any, ...]]:
        """Tunable hyper-parameters (declarative; see ``SEARCH_SPACES``). Empty when the model has none."""
        return SEARCH_SPACES.get(self.key, {})

    def hyperparams_text(self) -> str:
        """Human string such as ``n_estimators=200, max_depth=6, lr=0.1``."""
        return ", ".join(f"{'lr' if k == 'learning_rate' else k}={v}" for k, v in self.hyperparams.items())

    def build(self, seed: int, n_jobs: int, task: str = "classification") -> Any:
        """Instantiate a fresh estimator (the regressor counterpart when ``task`` is regression)."""
        if is_regression(task):
            factory = _REGRESSORS.get(self.key)
            if factory is None:
                raise BadRequestError(
                    f"Model '{self.abbr}' does not support regression.", error_code="MODEL_NOT_FOR_TASK"
                )
            return factory(seed, n_jobs, dict(self.hyperparams))
        if self.factory is None:
            raise BadRequestError(f"Model '{self.abbr}' is unavailable.", error_code="MODEL_UNAVAILABLE")
        return self.factory(seed, n_jobs, dict(self.hyperparams))


def _xgb(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(**hp, tree_method="hist", n_jobs=n_jobs, random_state=seed, verbosity=0)


def _lgb(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**hp, n_jobs=n_jobs, random_state=seed, verbose=-1)


def _cat(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=hp["iterations"], depth=hp["depth"], learning_rate=hp["learning_rate"],
        random_seed=seed, thread_count=n_jobs, verbose=0, allow_writing_files=False,
    )


def _rf(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**hp, n_jobs=n_jobs, random_state=seed)


def _etc(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(**hp, n_jobs=n_jobs, random_state=seed)


def _mlp(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=(64, 32), learning_rate_init=hp["learning_rate_init"], random_state=seed
    )


def _svm(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.svm import SVC

    return SVC(**hp, cache_size=500, random_state=seed)


def _lr(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**hp, random_state=seed)


def _gbm(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(**hp, random_state=seed)


def _ada(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import AdaBoostClassifier

    return AdaBoostClassifier(**hp, random_state=seed)


def _bag(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import BaggingClassifier

    return BaggingClassifier(**hp, n_jobs=n_jobs, random_state=seed)


def _stack(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression

    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=seed)),
            ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=hp["cv"],
    )


def _vote(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    return VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
            ("rf", RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=seed)),
            ("nb", GaussianNB()),
        ],
        voting=hp["voting"],
    )


def _dt(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(**hp, random_state=seed)


def _knn(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(**hp, n_jobs=n_jobs)


def _nb(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB()


def _lda(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return LinearDiscriminantAnalysis()


def _qda(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    return QuadraticDiscriminantAnalysis(**hp)


def _ridge(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import RidgeClassifier

    return RidgeClassifier(**hp, random_state=seed)


def _sgd(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier(**hp, random_state=seed)


def _lnr(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    return ThresholdedLinearRegression()


def _tab(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from app.services.torch_models import TabNetClassifierW

    return TabNetClassifierW(random_state=seed, **hp)


def _ptm(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from app.services.torch_models import TorchMLPClassifier

    return TorchMLPClassifier(random_state=seed, **hp)


# --- regression counterparts (keyed like the catalog; models absent here are classification-only) ------
def _r_xgb(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**hp, tree_method="hist", n_jobs=n_jobs, random_state=seed, verbosity=0)


def _r_lgb(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**hp, n_jobs=n_jobs, random_state=seed, verbose=-1)


def _r_cat(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=hp["iterations"], depth=hp["depth"], learning_rate=hp["learning_rate"],
        random_seed=seed, thread_count=n_jobs, verbose=0, allow_writing_files=False,
    )


def _r_rf(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**hp, n_jobs=n_jobs, random_state=seed)


def _r_etc(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(**hp, n_jobs=n_jobs, random_state=seed)


def _r_mlp(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate_init=hp["learning_rate_init"], random_state=seed)


def _r_svm(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.svm import SVR

    return SVR(**hp, cache_size=500)


def _r_lnr(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import LinearRegression

    return LinearRegression(**hp)


def _r_ridge(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import Ridge

    return Ridge(**hp, random_state=seed)


def _r_sgd(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.linear_model import SGDRegressor

    return SGDRegressor(max_iter=hp.get("max_iter", 1000), tol=hp.get("tol", 1e-3), random_state=seed)


def _r_gbm(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(**hp, random_state=seed)


def _r_ada(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import AdaBoostRegressor

    return AdaBoostRegressor(**hp, random_state=seed)


def _r_bag(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import BaggingRegressor

    return BaggingRegressor(**hp, n_jobs=n_jobs, random_state=seed)


def _r_stack(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import Ridge

    return StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=seed)),
            ("ridge", Ridge(random_state=seed)),
        ],
        final_estimator=Ridge(random_state=seed),
        cv=hp["cv"],
    )


def _r_vote(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    return VotingRegressor(
        estimators=[
            ("ridge", Ridge(random_state=seed)),
            ("rf", RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=seed)),
            ("knn", KNeighborsRegressor(n_neighbors=15, n_jobs=n_jobs)),
        ],
    )


def _r_dt(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(**hp, random_state=seed)


def _r_knn(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from sklearn.neighbors import KNeighborsRegressor

    return KNeighborsRegressor(**hp, n_jobs=n_jobs)


def _r_tab(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from app.services.torch_models import TabNetRegressorW

    return TabNetRegressorW(random_state=seed, **hp)


def _r_ptm(seed: int, n_jobs: int, hp: dict[str, Any]) -> Any:
    from app.services.torch_models import TorchMLPRegressor

    return TorchMLPRegressor(random_state=seed, **hp)


_REGRESSORS: dict[str, Factory] = {
    "xgboost": _r_xgb, "lightgbm": _r_lgb, "catboost": _r_cat, "random_forest": _r_rf, "extra_trees": _r_etc,
    "mlp": _r_mlp, "svm": _r_svm, "linear_regression": _r_lnr, "ridge": _r_ridge, "sgd": _r_sgd,
    "gradient_boosting": _r_gbm, "adaboost": _r_ada, "bagging": _r_bag, "stacking": _r_stack,
    "voting": _r_vote, "decision_tree": _r_dt, "knn": _r_knn, "tabnet": _r_tab, "torch_mlp": _r_ptm,
}

_LNR_NOTE = (
    "Linear Regression is a true regressor for continuous targets. For a classification target "
    "its output is thresholded at 0.5 to produce class labels."
)

CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("xgboost", "XGB", "XGBoost", "ENS", "GPU",
              {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}, "~3 s", _xgb,
              iterative=True, requires="xgboost"),
    ModelSpec("lightgbm", "LGB", "LightGBM", "ENS", "CPU",
              {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1}, "~2 s", _lgb,
              iterative=True, requires="lightgbm"),
    ModelSpec("random_forest", "RFR", "Random Forest", "ENS", "CPU",
              {"n_estimators": 200, "max_depth": 12}, "~4 s", _rf, memorizes=True),
    ModelSpec("mlp", "MLP", "Multi-Layer Perceptron", "NN", "GPU",
              {"hidden_layers": "64-32", "learning_rate_init": 0.001, "max_epochs": 200}, "~10 s", _mlp,
              scaled=True, iterative=True),
    ModelSpec("svm", "SVM", "Support Vector Machine", "SVM", "CPU",
              {"kernel": "rbf", "C": 1.0}, "~12 s", _svm, scaled=True),
    ModelSpec("logistic_regression", "LR", "Logistic Regression", "LIN", "CPU",
              {"C": 1.0, "max_iter": 1000}, "~1 s", _lr, scaled=True),
    ModelSpec("catboost", "CAT", "CatBoost", "ENS", "GPU",
              {"iterations": 300, "depth": 6, "learning_rate": 0.1}, "~8 s", _cat, requires="catboost"),
    ModelSpec("extra_trees", "ETC", "Extra Trees", "ENS", "CPU",
              {"n_estimators": 200, "max_depth": 12}, "~3 s", _etc, memorizes=True),
    ModelSpec("gradient_boosting", "GBM", "Gradient Boosting", "ENS", "CPU",
              {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3}, "~9 s", _gbm),
    ModelSpec("adaboost", "ADA", "AdaBoost", "ENS", "CPU",
              {"n_estimators": 100, "learning_rate": 1.0}, "~3 s", _ada),
    ModelSpec("bagging", "BAG", "Bagging Classifier", "ENS", "CPU",
              {"n_estimators": 50}, "~4 s", _bag, memorizes=True),
    ModelSpec("stacking", "STK", "Stacking Ensemble", "ENS", "CPU",
              {"cv": 3, "base": "RF+LR", "final": "LR"}, "~15 s", _stack, scaled=True),
    ModelSpec("voting", "VOT", "Voting Ensemble", "ENS", "CPU",
              {"voting": "soft", "base": "LR+RF+NB"}, "~5 s", _vote, scaled=True),
    ModelSpec("decision_tree", "DT", "Decision Tree", "TREE", "CPU",
              {"max_depth": 6}, "<1 s", _dt),
    ModelSpec("knn", "KNN", "K-Nearest Neighbors", "KNN", "CPU",
              {"n_neighbors": 15}, "~1 s", _knn, scaled=True),
    ModelSpec("naive_bayes", "NB", "Gaussian Naive Bayes", "NB", "CPU", {"var_smoothing": 1e-9}, "<1 s", _nb),
    ModelSpec("lda", "LDA", "Linear Discriminant Analysis", "LIN", "CPU", {"solver": "svd"}, "<1 s", _lda),
    ModelSpec("ridge", "RDG", "Ridge Classifier", "LIN", "CPU", {"alpha": 1.0}, "<1 s", _ridge, scaled=True),
    ModelSpec("sgd", "SGD", "SGD Classifier", "LIN", "CPU",
              {"loss": "log_loss", "max_iter": 1000, "tol": 0.001}, "~1 s", _sgd, scaled=True),
    ModelSpec("qda", "QDA", "Quadratic Discriminant Analysis", "LIN", "CPU",
              {"reg_param": 0.1}, "<1 s", _qda),
    ModelSpec("linear_regression", "LNR", "Linear Regression", "LIN", "CPU",
              {"fit_intercept": True}, "<1 s", _lnr, scaled=True, note=_LNR_NOTE),
    ModelSpec("tabnet", "TAB", "TabNet", "NN", "GPU",
              {"n_d": 8, "n_a": 8, "n_steps": 3, "max_epochs": 60}, "~30 s", _tab,
              scaled=True, iterative=True, requires="pytorch_tabnet", unavailable_reason="needs pytorch-tabnet"),
    ModelSpec("torch_mlp", "PTM", "PyTorch MLP", "NN", "GPU",
              {"hidden": "64-32", "lr": 0.001, "dropout": 0.1, "max_epochs": 100}, "~10 s", _ptm,
              scaled=True, iterative=True, requires="torch", unavailable_reason="needs torch"),
)

_BY_KEY = {spec.key: spec for spec in CATALOG}
_BY_ABBR = {spec.abbr.lower(): spec for spec in CATALOG}


def resolve_model(identifier: str) -> ModelSpec:
    """Find a model by key (``random_forest``) or abbreviation (``RFR``)."""
    token = identifier.strip().lower()
    spec = _BY_KEY.get(token) or _BY_ABBR.get(token)
    if spec is None:
        raise BadRequestError(
            f"Unknown model '{identifier}'.",
            error_code="UNKNOWN_MODEL",
            details={"valid": [s.key for s in CATALOG]},
        )
    return spec


# --- hyper-parameter search spaces ---------------------------------------------------------------------------
# ("int", lo, hi[, step]) | ("float", lo, hi) | ("log", lo, hi) | ("cat", [choices]). Names are valid for both the
# classifier and the regressor of a model, so one space serves every task.
SEARCH_SPACES: dict[str, dict[str, tuple[Any, ...]]] = {
    "xgboost": {
        "n_estimators": ("int", 50, 400, 50), "max_depth": ("int", 2, 10), "learning_rate": ("log", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0), "colsample_bytree": ("float", 0.6, 1.0), "min_child_weight": ("int", 1, 10),
    },
    "lightgbm": {
        "n_estimators": ("int", 50, 400, 50), "num_leaves": ("int", 8, 128), "learning_rate": ("log", 0.01, 0.3),
        "min_child_samples": ("int", 5, 60), "colsample_bytree": ("float", 0.6, 1.0),
    },
    "catboost": {
        "iterations": ("int", 100, 400, 100), "depth": ("int", 3, 8), "learning_rate": ("log", 0.02, 0.3),
        "l2_leaf_reg": ("float", 1.0, 10.0),
    },
    "random_forest": {
        "n_estimators": ("int", 100, 400, 50), "max_depth": ("int", 3, 20), "min_samples_leaf": ("int", 1, 10),
        "max_features": ("cat", ["sqrt", "log2", 0.5, 1.0]),
    },
    "extra_trees": {
        "n_estimators": ("int", 100, 400, 50), "max_depth": ("int", 3, 20), "min_samples_leaf": ("int", 1, 10),
        "max_features": ("cat", ["sqrt", "log2", 0.5, 1.0]),
    },
    "gradient_boosting": {
        "n_estimators": ("int", 50, 300, 50), "learning_rate": ("log", 0.01, 0.3), "max_depth": ("int", 2, 6),
        "subsample": ("float", 0.6, 1.0),
    },
    "adaboost": {"n_estimators": ("int", 50, 300, 50), "learning_rate": ("log", 0.01, 2.0)},
    "bagging": {"n_estimators": ("int", 10, 100, 10), "max_samples": ("float", 0.5, 1.0), "max_features": ("float", 0.5, 1.0)},
    "decision_tree": {"max_depth": ("int", 2, 20), "min_samples_leaf": ("int", 1, 30)},
    "knn": {"n_neighbors": ("int", 3, 50), "weights": ("cat", ["uniform", "distance"]), "p": ("cat", [1, 2])},
    "svm": {"C": ("log", 0.1, 100.0), "gamma": ("log", 1e-4, 1.0)},
    "logistic_regression": {"C": ("log", 0.01, 100.0)},
    "ridge": {"alpha": ("log", 0.01, 100.0)},
    "sgd": {"alpha": ("log", 1e-6, 1e-1)},
    "mlp": {
        "hidden_layer_sizes": ("cat", ["32", "64", "64-32", "128-64", "128-64-32"]),
        "alpha": ("log", 1e-6, 1e-1), "learning_rate_init": ("log", 1e-4, 1e-2),
    },
    "qda": {"reg_param": ("float", 0.0, 1.0)},
    "naive_bayes": {"var_smoothing": ("log", 1e-11, 1e-5)},
    "tabnet": {"n_d": ("cat", [8, 16, 32]), "n_steps": ("int", 3, 5), "gamma": ("float", 1.0, 2.0), "lr": ("log", 0.005, 0.05)},
    "torch_mlp": {
        "hidden": ("cat", ["32", "64", "64-32", "128-64"]), "lr": ("log", 1e-4, 1e-2),
        "weight_decay": ("log", 1e-6, 1e-2), "dropout": ("float", 0.0, 0.4),
    },
}
