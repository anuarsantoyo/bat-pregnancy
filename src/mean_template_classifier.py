"""MeanTemplateClassifier -- shift-invariant nearest-prototype classifier.

Design (Anuar, 2026-09-13), kept as specified
---------------------------------------------
* ``fit(X, y)`` does no learning: it computes two prototypes, the element-wise
  average of the rows of class 0 and of class 1 (``mean_0_``, ``mean_1_``).
  ("internally creates 2 datasets which are simply the average of the 1s and 0s")
* ``predict(X)`` zero-pads each series with ``n_features`` zeros on both sides and
  slides a fixed-length window over it (the "convolution" step). At every shift it
  computes the MSE against both prototypes over the whole window. The denominator is
  always ``n_features``, so partial overlaps automatically score worse: at zero overlap
  the MSE equals the mean of the squared prototype, its ceiling. No minimum-overlap
  hyper-parameter, no edge flukes.
* The label is the prototype with the **lowest minimum MSE** over all shifts.
* ``predict(X, onset=True)`` additionally returns the day at which the elevated phase
  starts, for the series predicted pregnant (``nan`` for the others).

Measured on the synthetic sandbox (``data/processed/dummy_data.csv``)
--------------------------------------------------------------------
* accuracy **0.893** -- all 107 errors are pregnant -> non-pregnant, zero false alarms
* jump-start recovery: median error +0 to +1 day, IQR +-1-2 days, 99% within +-7 days
* why the errors: for a truly pregnant bat the winning margin is only **0.25 MSE**
  (2.82 vs 3.07) because 145 of the 180 days are identical under both prototypes.

Deferred to the real dataset (see ``TODO.md``): the equivalent but better-behaved
decision via the projection onto ``mean_1_ - mean_0_`` (AUC 1.000 on the sandbox).

Caveat: ``predict(X, onset=True)`` returns a tuple, so it breaks the usual sklearn
estimator contract (``score``/``cross_val_score`` need the plain-label form).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["MeanTemplateClassifier"]


class MeanTemplateClassifier(ClassifierMixin, BaseEstimator):
    """Nearest-prototype classifier whose prototypes may sit anywhere in the series.

    Parameters
    ----------
    none -- the model is fully determined by the training data.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
    mean_0_, mean_1_ : ndarray of shape (n_features,) -- the two class prototypes.
    diff_template_ : ndarray -- ``mean_1_ - mean_0_`` (kept for inspection/plots).
    onset_anchor_ : int -- index inside the prototypes where the elevated phase is
        taken to start (50% of the rise of ``diff_template_``). Used to turn the
        best-matching shift into a day number.
    n_features_in_ : int
    """

    def __init__(self) -> None:  # no hyper-parameters (yet)
        pass

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_days), got {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        uniq = np.unique(y)
        if not np.array_equal(uniq, [0, 1]):
            raise ValueError(f"y must be binary 0/1, got classes {uniq}")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.mean_0_ = X[y == 0].mean(axis=0)
        self.mean_1_ = X[y == 1].mean(axis=0)
        self.diff_template_ = self.mean_1_ - self.mean_0_

        # where the elevated phase starts inside the prototype (robust to the
        # smoothing caused by averaging misaligned individuals)
        half = 0.5 * self.diff_template_.max()
        self.onset_anchor_ = int(np.argmax(self.diff_template_ >= half))
        return self

    # ------------------------------------------------------------- helpers
    def _mse_curves(self, x):
        """MSE of the sliding window against both prototypes, one value per shift."""
        n = self.n_features_in_
        zeros = np.zeros(n)
        padded = np.concatenate([zeros, x, zeros])
        windows = sliding_window_view(padded, n)          # (2n + 1, n)
        mse0 = ((windows - self.mean_0_) ** 2).mean(axis=1)
        mse1 = ((windows - self.mean_1_) ** 2).mean(axis=1)
        return mse0, mse1

    # ------------------------------------------------------------- predict
    def predict(self, X, onset=False):
        """Predict labels; optionally also the day the elevated phase starts.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_days)
        onset : bool, default False
            If True, return ``(labels, onsets)`` where ``onsets`` holds the 1-based day
            number at which the jump starts for series predicted pregnant, and ``nan``
            for the others.

        Returns
        -------
        labels : ndarray of shape (n_samples,) with values in {0, 1}
        onsets : ndarray of shape (n_samples,), only when ``onset=True``
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} columns, model trained on {self.n_features_in_}")

        n = self.n_features_in_
        labels = np.zeros(len(X), dtype=int)
        onsets = np.full(len(X), np.nan)
        for k, x in enumerate(X):
            mse0, mse1 = self._mse_curves(x)
            if mse1.min() < mse0.min():
                labels[k] = 1
                if onset:
                    shift = int(mse1.argmin())
                    # window element i maps to series index shift + i - n
                    # -> 1-based day number of the anchor day
                    onsets[k] = shift + self.onset_anchor_ - n + 1
        if onset:
            return labels, onsets
        return labels

    # ------------------------------------------------------------- extras
    def decision_margins(self, X):
        """min-MSE(mean_0) - min-MSE(mean_1) per sample. Positive -> predicted 1.

        Handy to see *how close* a call was: on the sandbox the true-pregnant samples
        sit around +0.25 and the true non-pregnant around -1.2.
        """
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        for k, x in enumerate(X):
            mse0, mse1 = self._mse_curves(x)
            out[k] = mse0.min() - mse1.min()
        return out
