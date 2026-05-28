"""Chronos-Bolt (Chronos 2) solver for the TSFM benchmark.

Supports:
  - forecasting        : zero-shot via ChronosBoltPipeline
  - anomaly_detection  : forecast-residual (zero-shot)

Chronos-Bolt is the second-generation Chronos architecture. It outputs
quantile forecasts directly (shape: batch × num_quantiles × horizon) rather
than sample-based distributions, making inference significantly faster.

Model loading is done in ``set_objective`` (untimed).
Adaptation fitting is done in ``run`` (timed).
"""

import numpy as np
from benchopt import BaseSolver

from benchmark_utils.adapters.forecast_residual import ForecastResidualAdapter


SUPPORTED_TASKS = {"forecasting", "anomaly_detection"}


class _ChronosBoltForecaster:
    """Wraps ChronosBoltPipeline to expose predict(x: (T, C)) -> (H, C)."""

    def __init__(self, pipeline, prediction_length):
        self.pipeline = pipeline
        self.prediction_length = prediction_length
        quantiles = getattr(pipeline, "quantiles", [])
        self._median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(x, dtype=np.float32)  # (T, C)
        C = x.shape[1]

        preds = []
        for c in range(C):
            context = torch.from_numpy(x[:, c]).unsqueeze(0)  # (1, T)
            # Returns (1, num_quantiles, H) for ChronosBolt
            forecast = self.pipeline.predict(
                context,
                prediction_length=self.prediction_length,
            )
            f = forecast[0]  # (num_quantiles, H) or (H,) for point models
            if f.ndim == 2:
                f = f[self._median_idx]  # median quantile → (H,)
            preds.append(f.cpu().numpy())

        return np.stack(preds, axis=-1).astype(np.float32)  # (H, C)


class Solver(BaseSolver):
    """Chronos-Bolt (Chronos 2) zero-shot solver.

    Parameters
    ----------
    task_adaptation : str
        How to use Chronos-Bolt for each task:
          "zeroshot"          — direct forecasting API (forecasting only)
          "forecast_residual" — anomaly score = forecast error (AD only)
    """

    name = "Chronos2"

    requirements = ["pip::chronos-forecasting>=2.0", "pip::torch"]

    sampling_strategy = "run_once"

    parameters = {
        "task_adaptation": ["zeroshot"],
    }

    def skip(self, task, **kwargs):
        if task not in SUPPORTED_TASKS:
            return True, f"Chronos2 solver does not support task={task!r}"
        return False, None

    def set_objective(self, X_train, y_train, task, **meta):
        import os
        import sys
        if sys.platform == "win32":
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            if conda_prefix:
                lib_bin = os.path.join(conda_prefix, "Library", "bin")
                if os.path.isdir(lib_bin):
                    os.add_dll_directory(lib_bin)
        import torch
        from chronos.base import BaseChronosPipeline

        self.task = task
        self.X_train = X_train
        self.meta = meta

        model_name = "amazon/chronos-2"
        if not hasattr(self, "_pipeline") or self._loaded_model != model_name:
            self._pipeline = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map="cpu",
                dtype=torch.float32,
            )
            self._loaded_model = model_name

    def run(self, _):
        pred_len = self.meta.get("prediction_length", 1)
        forecaster = _ChronosBoltForecaster(self._pipeline, pred_len)

        if self.task == "forecasting":
            self._adapter = forecaster

        elif self.task == "anomaly_detection":
            self._adapter = ForecastResidualAdapter(
                forecaster, prediction_length=1
            )

    def get_result(self):
        return {"model": self._adapter}
