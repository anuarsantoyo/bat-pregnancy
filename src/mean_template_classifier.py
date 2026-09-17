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
  the MSE equals the mean of the squared prototype.
* The label is the prototype with the **lowest minimum MSE** over the allowed shifts.

Why ``min_overlap`` exists (added 2026-09-17, real data)
--------------------------------------------------------
The original "no minimum-overlap hyper-parameter, no edge flukes" reasoning assumed that
zero overlap is the *worst* achievable score, so it would never be selected. That is only
true while every genuine alignment scores above the constant ``mean(P^2)``. On the real
data the series are empty for ~7 months and individual amplitude spans an order of
magnitude, so a real alignment often scores *worse* than the constant, and the zero-overlap
("all padding") window becomes the **argmin**. Its score depends only on the prototype, and
``mean(P0^2) < mean(P1^2)``, so the model then answers "not lactating" for that bat
regardless of its data. Measured: 479/658 bats fell back to it, accuracy 0.310.

There is a mirror-image failure if the overlap denominator is used instead of the fixed
``n_features``: the minimum is then driven to the *smallest* overlap (1 day), which also
scores ~0 for both prototypes. Any "min over shifts" rule needs the alignment range pinned
down; that is what ``min_overlap`` does.

``min_overlap`` is the minimum fraction of the prototype that must overlap the series for
a shift to be admissible.

* ``min_overlap=1.0`` (default) -- only the fully-overlapping alignment; the sweep is off.
  This is the right setting when every series spans the same fixed calendar window (real
  data), where there is nothing to align and sliding only creates ways to cheat.
* ``min_overlap=0.5`` -- allow shifts up to half the series length.
* ``min_overlap=0.0`` -- no restriction: all ``2*n_features + 1`` shifts, i.e. the original
  behaviour, kept for the synthetic sandbox where the elevated phase can start anywhere.

Measured on the real data (leave-one-out, 658 bat-years, sit 300 s):

===============  ======  ========  =======  ======  ======
min_overlap      acc     precision recall   F1      AUC
===============  ======  ========  =======  ======  ======
1.0 (default)    0.708   0.873     0.677    0.762   0.711
0.9              0.698   0.842     0.692    0.760   0.700
0.5              0.532   0.873     0.378    0.528   0.554
0.0 (original)   0.310   1.000     0.002    0.004   0.349
===============  ======  ========  =======  ======  ======

Majority baseline (always lactating): acc 0.691, precision 0.691, recall 1.000, F1 0.818.
Leave-one-*bat*-out at ``min_overlap=1.0`` gives 0.710 / 0.873 / 0.679 / 0.764 / 0.711 --
indistinguishable, so bat-identity leakage is not a factor.

The same fix also helps the synthetic sandbox (leave-one-out: 0.893 -> **0.992**), i.e. the
unrestricted sweep was never carrying signal, only escape hatches. **But onset localisation
needs the sweep**: with a single admissible shift the returned onset is the same constant for
every series (``nunique == 1``). Use ``min_overlap=1.0`` for the decision and, if the onset is
wanted, a restricted sweep (e.g. ``min_overlap=0.75``) for the series called lactating.
* ``predict(X, onset=True)`` additionally returns the day at which the elevated phase
  starts, for the series predicted pregnant (``nan`` for the others).

Measured on the synthetic sandbox (``data/processed/dummy_data.csv``)
--------------------------------------------------------------------
* ``min_overlap=0.0`` (original sweep): accuracy **0.893** -- all 107 errors are
  pregnant -> non-pregnant, zero false alarms
* ``min_overlap=1.0`` (new default): accuracy **0.992** -- so on the sandbox too, the
  unrestricted sweep was losing ~10 points. This is also the setting that reproduces the
  projection rule's performance (acc 0.996, AUC 1.000).
* jump-start recovery (needs a sweep): median error +0 to +1 day, IQR +-1-2 days, 99% within +-7 days
* why the errors at ``min_overlap=0.0``: for a truly pregnant bat the winning margin is only
  **0.25 MSE** (2.82 vs 3.07) because 145 of the 180 days are identical under both prototypes.

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
    min_overlap : float in [0, 1], default 1.0
        Minimum fraction of the prototype that must overlap the series for a shift to be
        admissible. ``1.0`` disables the sweep, ``0.0`` is the original unrestricted sweep.
        See the module docstring for why this exists.

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

    def __init__(self, min_overlap: float = 1.0) -> None:
        """
        Parameters
        ----------
        min_overlap : float in [0, 1], default 1.0
            Minimum fraction of the prototype that must overlap the series for a shift to
            be admissible. ``1.0`` disables the sweep (fully-overlapping alignment only,
            the right choice for calendar-aligned year series); ``0.0`` restores the
            original unrestricted sweep needed by the synthetic sandbox.
        """
        self.min_overlap = min_overlap

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

        if not 0.0 <= float(self.min_overlap) <= 1.0:
            raise ValueError(f"min_overlap must be in [0, 1], got {self.min_overlap}")

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
    def _shift_bounds(self):
        """Inclusive range of window positions allowed by ``min_overlap``.

        With the series occupying ``[n, 2n)`` of ``padded``, a window at position ``i``
        overlaps the series in ``n - |i - n|`` elements, so ``min_overlap`` bounds ``|i - n|``.
        """
        n = self.n_features_in_
        span = int(np.floor((1.0 - float(self.min_overlap)) * n))
        return n - span, n + span

    def _mse_curves(self, x):
        """MSE of the sliding window against both prototypes, one value per allowed shift."""
        n = self.n_features_in_
        zeros = np.zeros(n)
        padded = np.concatenate([zeros, x, zeros])
        windows = sliding_window_view(padded, n)          # (2n + 1, n)
        lo, hi = self._shift_bounds()
        windows = windows[lo:hi + 1]                      # drop the inadmissible alignments
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
        lo, hi = self._shift_bounds()
        labels = np.zeros(len(X), dtype=int)
        onsets = np.full(len(X), np.nan)
        for k, x in enumerate(X):
            mse0, mse1 = self._mse_curves(x)
            if mse1.min() < mse0.min():
                labels[k] = 1
                if onset:
                    shift = lo + int(mse1.argmin())
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
