"""Training-only running normalization for centralized critic states."""

from __future__ import annotations

from typing import Any

import numpy as np


class RunningMeanStd:
    """Numerically stable per-feature running moments with serializable state."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4, clip: float = 10.0) -> None:
        self.shape = tuple(shape)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(epsilon)
        self.clip = float(clip)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-len(self.shape):] != self.shape:
            raise ValueError(f"expected trailing shape {self.shape}, got {array.shape}")
        array = array.reshape((-1,) + self.shape)
        if not array.size:
            return
        if not np.isfinite(array).all():
            raise ValueError("cannot update normalizer with NaN or Inf")
        batch_count = float(array.shape[0])
        batch_mean, batch_var = array.mean(axis=0), array.var(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a, m_b = self.var * self.count, batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean, self.var, self.count = new_mean, np.maximum(m2 / total, 1e-12), total

    def normalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-len(self.shape):] != self.shape:
            raise ValueError(f"expected trailing shape {self.shape}, got {array.shape}")
        normalized = (array - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"shape": self.shape, "mean": self.mean.copy(), "var": self.var.copy(), "count": self.count, "clip": self.clip}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if tuple(state["shape"]) != self.shape:
            raise ValueError(f"normalizer shape mismatch: {state['shape']} vs {self.shape}")
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count, self.clip = float(state["count"]), float(state["clip"])
        if not np.isfinite(self.mean).all() or not np.isfinite(self.var).all() or self.count <= 0:
            raise ValueError("invalid normalizer checkpoint state")


class RunningScalarMeanStd:
    """Serializable running scalar moments for optional Critic value targets."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean, self.var, self.count = 0.0, 1.0, float(epsilon)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not array.size:
            return
        if not np.isfinite(array).all():
            raise ValueError("cannot update value normalizer with NaN or Inf")
        batch_count = float(array.size)
        batch_mean, batch_var = float(array.mean()), float(array.var())
        delta, total = batch_mean - self.mean, self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = max((self.var * self.count + batch_var * batch_count
                        + delta * delta * self.count * batch_count / total) / total, 1e-12)
        self.count = total

    @staticmethod
    def _state_values(state: dict[str, Any]) -> tuple[float, float]:
        mean, var = float(state["mean"]), float(state["var"])
        if not np.isfinite((mean, var)).all() or var <= 0.0:
            raise ValueError("invalid value normalizer state")
        return mean, var

    def normalize(self, values: np.ndarray, *, state: dict[str, Any] | None = None) -> np.ndarray:
        mean, var = self._state_values(self.state_dict() if state is None else state)
        return ((np.asarray(values, dtype=np.float32) - mean) / np.sqrt(var + 1e-8)).astype(np.float32)

    def denormalize(self, values: np.ndarray, *, state: dict[str, Any] | None = None) -> np.ndarray:
        mean, var = self._state_values(self.state_dict() if state is None else state)
        return (np.asarray(values, dtype=np.float32) * np.sqrt(var + 1e-8) + mean).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        mean, var = self._state_values(state)
        count = float(state["count"])
        if not np.isfinite(count) or count <= 0.0:
            raise ValueError("invalid value normalizer count")
        self.mean, self.var, self.count = mean, var, count
