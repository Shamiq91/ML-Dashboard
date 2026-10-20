"""sklearn-compatible PyTorch estimators: a small MLP and a TabNet wrapper.

Both expose ``fit(X, y, eval_set=...)`` and fill ``history_`` (per-epoch train/val loss and a val score) so the
dashboard can draw the same training telemetry it draws for XGBoost/LightGBM/MLP. Torch is imported lazily so the rest
of the backend starts fast and still works if torch is not installed.
"""
from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split


def pick_device() -> str:
    """``cuda`` when available, otherwise ``cpu`` (small tabular nets are slower on MPS than on the CPU)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _holdout(x: np.ndarray, y: np.ndarray, classification: bool, seed: int) -> tuple[np.ndarray, ...]:
    """10 % internal validation split used for early stopping when the caller supplies no eval set."""
    if len(y) < 20:
        return x, y, x, y
    try:
        xa, xb, ya, yb = train_test_split(x, y, test_size=0.1, random_state=seed, stratify=y if classification else None)
    except ValueError:
        xa, xb, ya, yb = train_test_split(x, y, test_size=0.1, random_state=seed)
    return xa, ya, xb, yb


def _hidden(spec: str | tuple[int, ...] | list[int]) -> list[int]:
    if isinstance(spec, str):
        return [int(p) for p in spec.replace(",", "-").split("-") if p.strip()]
    return [int(p) for p in spec]


# --- PyTorch MLP ------------------------------------------------------------------------------------------
class _TorchMLPBase(BaseEstimator):
    """Shared training loop; subclasses fix the task (classification / regression)."""

    _classification = True

    def __init__(
        self, hidden: str = "64-32", lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.1,
        max_epochs: int = 100, batch_size: int = 256, patience: int = 10, random_state: int = 42,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def _build(self, n_in: int, n_out: int) -> Any:
        import torch.nn as nn

        layers: list[Any] = []
        width = n_in
        for size in _hidden(self.hidden):
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        layers.append(nn.Linear(width, n_out))
        return nn.Sequential(*layers)

    def fit(self, X: Any, y: Any, eval_set: tuple[Any, Any] | None = None, sample_weight: Any = None) -> "_TorchMLPBase":
        """Train with Adam and early stopping on validation loss; keeps the best weights."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        x = np.asarray(X, dtype="float32")
        y = np.asarray(y)
        if eval_set is None:
            x, y, xv, yv = _holdout(x, y, self._classification, self.random_state)
        else:
            xv, yv = np.asarray(eval_set[0], dtype="float32"), np.asarray(eval_set[1])
        device = pick_device()
        self.device_ = device
        if self._classification:
            self.classes_ = np.unique(np.concatenate([y, yv]))
            lookup = {c: i for i, c in enumerate(self.classes_)}
            y_idx = np.array([lookup[c] for c in y]); yv_idx = np.array([lookup[c] for c in yv])
            n_out = len(self.classes_)
            loss_fn: Any = nn.CrossEntropyLoss()
            yt, yvt = torch.as_tensor(y_idx, dtype=torch.long), torch.as_tensor(yv_idx, dtype=torch.long)
        else:
            n_out = 1
            loss_fn = nn.MSELoss()
            yt, yvt = torch.as_tensor(y, dtype=torch.float32).view(-1, 1), torch.as_tensor(yv, dtype=torch.float32).view(-1, 1)
        xt, xvt = torch.as_tensor(x), torch.as_tensor(xv)
        net = self._build(x.shape[1], n_out).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = len(xt)
        bs = max(1, min(self.batch_size, n))
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_metric": []}
        best, best_state, stale = np.inf, None, 0
        gen = torch.Generator().manual_seed(self.random_state)
        for _ in range(self.max_epochs):
            net.train()
            order = torch.randperm(n, generator=gen)
            for start in range(0, n, bs):
                idx = order[start : start + bs]
                if len(idx) < 2 and n > 1:
                    continue
                opt.zero_grad()
                loss = loss_fn(net(xt[idx].to(device)), yt[idx].to(device))
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                tr, va = net(xt.to(device)), net(xvt.to(device))
                history["train_loss"].append(float(loss_fn(tr, yt.to(device))))
                v = float(loss_fn(va, yvt.to(device)))
                history["val_loss"].append(v)
                if self._classification:
                    history["val_metric"].append(float((va.argmax(1).cpu() == yvt).float().mean()))
                else:
                    var = float(((yvt - yvt.mean()) ** 2).mean()) or 1.0
                    history["val_metric"].append(1.0 - v / var)
            if v < best - 1e-5:
                best, stale = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net.cpu().eval()
        self.history_ = history
        self.n_features_in_ = x.shape[1]
        return self

    def _raw(self, X: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.net_(torch.as_tensor(np.asarray(X, dtype="float32"))).numpy()


class TorchMLPClassifier(ClassifierMixin, _TorchMLPBase):
    """PyTorch MLP classifier (softmax output)."""

    _classification = True

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities."""
        z = self._raw(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        """Most probable class."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchMLPRegressor(RegressorMixin, _TorchMLPBase):
    """PyTorch MLP regressor (MSE)."""

    _classification = False

    def predict(self, X: Any) -> np.ndarray:
        """Predicted values."""
        return self._raw(X).ravel()


# --- TabNet -----------------------------------------------------------------------------------------------
class _TabNetBase(BaseEstimator):
    """Wrap ``pytorch_tabnet`` behind the sklearn API with early stopping and a loss history."""

    _classification = True

    def __init__(
        self, n_d: int = 8, n_a: int = 8, n_steps: int = 3, gamma: float = 1.3, lr: float = 0.02,
        max_epochs: int = 60, batch_size: int = 512, patience: int = 10, random_state: int = 42,
    ) -> None:
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X: Any, y: Any, eval_set: tuple[Any, Any] | None = None, sample_weight: Any = None) -> "_TabNetBase":
        """Train TabNet with early stopping on a validation set (internal 10 % split if none is given)."""
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

        x = np.asarray(X, dtype="float32")
        y = np.asarray(y)
        if eval_set is None:
            x, y, xv, yv = _holdout(x, y, self._classification, self.random_state)
        else:
            xv, yv = np.asarray(eval_set[0], dtype="float32"), np.asarray(eval_set[1])
        n = len(x)
        bs = max(2, min(self.batch_size, n))
        common = dict(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps, gamma=self.gamma,
            optimizer_params={"lr": self.lr}, seed=self.random_state, verbose=0, device_name=pick_device(),
        )
        fit_kwargs = dict(
            max_epochs=self.max_epochs, patience=self.patience, batch_size=bs, virtual_batch_size=min(128, bs),
            drop_last=bool(n % bs == 1), num_workers=0,
        )
        if self._classification:
            self.classes_ = np.unique(np.concatenate([y, yv]))
            model = TabNetClassifier(**common)  # eval_metric order matters: TabNet early-stops on the last one
            with contextlib.redirect_stdout(io.StringIO()):  # TabNet prints "Early stopping..." even with verbose=0
                model.fit(x, y, eval_set=[(xv, yv)], eval_metric=["accuracy", "logloss"], **fit_kwargs)
        else:
            model = TabNetRegressor(**common)
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(x, y.reshape(-1, 1), eval_set=[(xv, yv.reshape(-1, 1))], eval_metric=["rmse"], **fit_kwargs)
        self.model_ = model
        hist = model.history
        train_loss = [float(v) for v in hist["loss"]]
        if self._classification:
            val_loss = [float(v) for v in hist["val_0_logloss"]]
            val_metric = [float(v) for v in hist["val_0_accuracy"]]
        else:
            var = float(np.var(yv)) or 1.0
            rmse = [float(v) for v in hist["val_0_rmse"]]
            val_loss = [r * r for r in rmse]
            val_metric = [1.0 - r * r / var for r in rmse]
        self.history_ = {"train_loss": train_loss, "val_loss": val_loss, "val_metric": val_metric}
        self.feature_importances_ = np.asarray(model.feature_importances_, dtype="float64")
        self.n_features_in_ = x.shape[1]
        return self


class TabNetClassifierW(ClassifierMixin, _TabNetBase):
    """TabNet classifier."""

    _classification = True

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities."""
        return np.asarray(self.model_.predict_proba(np.asarray(X, dtype="float32")))

    def predict(self, X: Any) -> np.ndarray:
        """Most probable class."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TabNetRegressorW(RegressorMixin, _TabNetBase):
    """TabNet regressor."""

    _classification = False

    def predict(self, X: Any) -> np.ndarray:
        """Predicted values."""
        return np.asarray(self.model_.predict(np.asarray(X, dtype="float32"))).ravel()


def fit_torch_with_telemetry(
    estimator: Any, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, task: str,
) -> dict[str, Any]:
    """Fit a torch/TabNet estimator with a validation set and return per-epoch curves in the telemetry format."""
    estimator.fit(x_train, y_train, eval_set=(x_val, y_val))
    hist = estimator.history_
    regression = task != "classification"
    return {
        "train_loss": hist["train_loss"], "val_loss": hist["val_loss"], "val_accuracy": hist["val_metric"],
        "val_metric": "r2" if regression else "accuracy",
    }
