import numpy as np
from hierarchicalforecast.methods import BottomUp, MinTrace


class Reconciliation:
    """Official Nixtla reconcilers fitted exclusively from calibration forecasts."""

    def __init__(self, S, bottom_indices):
        self.S = np.asarray(S, dtype=np.float64)
        self.bottom = np.asarray(bottom_indices, dtype=int)
        self.order = np.r_[[i for i in range(len(S)) if i not in self.bottom], self.bottom]
        self.inverse = np.argsort(self.order)
        self.ordered_S = self.S[self.order]
        np.testing.assert_array_equal(self.ordered_S[-len(self.bottom):], np.eye(len(self.bottom)))
        self.reconcilers = {}

    def fit(self, calibration_y, calibration_predictions, *, role):
        if role != "CALIBRATION":
            raise ValueError("Reconciliation fit is CALIBRATION-only")
        y = np.asarray(calibration_y, dtype=np.float64)
        pred = np.asarray(calibration_predictions, dtype=np.float64)
        if y.shape != pred.shape or y.ndim != 3 or y.shape[1] != len(self.S):
            raise ValueError("Expected origins x nodes x horizon calibration arrays")
        if not np.isfinite(y).all() or not np.isfinite(pred).all():
            raise ValueError("Nonfinite calibration values")
        # Each rolling-origin/horizon pair is one residual observation; overlaps are retained.
        actual = y.transpose(1, 0, 2).reshape(len(self.S), -1)[self.order]
        forecast = pred.transpose(1, 0, 2).reshape(len(self.S), -1)[self.order]
        for name, reconciler in [("bottom_up", BottomUp()), ("ols", MinTrace(method="ols")),
                                 ("mint_shrink", MinTrace(method="mint_shrink"))]:
            reconciler.fit(S=self.ordered_S, y_hat=forecast[:, :12],
                           y_insample=actual, y_hat_insample=forecast)
            self.reconcilers[name] = reconciler
        self.audit = {"fit_role": role, "calibration_origins": len(y), "residual_observations": actual.shape[1],
                      "residual_layout": "nodes x (origin-major, then horizon); overlapping dates retained",
                      "implementation": "official hierarchicalforecast.methods; no custom covariance estimator",
                      "test_labels_used": False, "ridge": self.reconcilers["mint_shrink"].mint_shr_ridge}
        return self

    def transform(self, predictions, method):
        prediction = np.asarray(predictions, dtype=np.float64)
        if method == "unreconciled":
            return prediction.copy()
        if not self.reconcilers:
            raise ValueError("Reconciler is not fitted")
        rec = self.reconcilers[method]
        ordered = prediction[:, self.order, :]
        output = np.einsum("nb,bc,och->onh", self.ordered_S, rec.P, ordered)
        return output[:, self.inverse, :]
