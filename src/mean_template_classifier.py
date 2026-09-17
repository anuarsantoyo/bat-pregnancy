"""MeanTemplateClassifier -- shift-invariant nearest-prototype classifier.

Design (Anuar, 2026-09-13; sweep formulation corrected 2026-09-17)
------------------------------------------------------------------
* ``fit(X, y)`` does no learning: it computes two prototypes, the element-wise
  average of the rows of class 0 and of class 1 (``mean_0_``, ``mean_1_``).
  ("internally creates 2 datasets which are simply the average of the 1s and 0s")
* ``predict(X)`` pads **both** the series and the prototype with ``n_features`` zeros on
  each side, so each becomes ``pad | data | pad`` -- ``l1 l2 l3`` and ``r1 r2 r3`` -- and
  sweeps one against the other over the ``2n + 1`` relative offsets ``o``:

      start   o = +n :  r1 on l2, r2 on l3   (l1 and r3 stick out)
      ...
      o = 0         :  full alignment (l1 on r1, l2 on r2, l3 on r3)
      ...
      end     o = -n :  r2 on l1, r3 on l2

  At each offset the MSE is taken **over the values that cross**, i.e. over the
  overlapping region of the two padded series, which spans ``3n - |o|`` elements.
  The denominator is that crossing width.
* The label is the prototype with the **lowest minimum MSE** over the offsets.
* ``predict(X, onset=True)`` additionally returns the day at which the elevated phase
  starts, for the series predicted pregnant.
* ``min_overlap`` optionally restricts the sweep (see below); default ``0.0`` = all offsets.

Why the crossing region matters
-------------------------------
Because the crossing is taken over the two padded series, **real data is always in the
score**: at the end offsets the comparison is one series' data against the other's padding,
so a series can never be scored without itself being involved. On the real data the winning
offset is a median of 8 days from full alignment, the maximum is 68, and no bat ever selects
an end offset.

The first implementation did *not* do this. It padded only the series and slid a **fixed
length-``n`` window** over it, comparing the whole prototype against that window. At the end
offsets such a window is pure padding, so the score became prototype-vs-zeros = ``mean(P^2)``
-- a constant containing none of the bat. With ``mean(P0^2) = 0.599`` and
``mean(P1^2) = 1.782``, 479 of 658 real bats picked that window on prototype 0 and the
decision degenerated to comparing two constants: accuracy 0.310, every bat called
"not lactating".

``min_overlap`` (optional restriction, default 0.0)
---------------------------------------------------
Minimum fraction of the series' data that must overlap the prototype's data for an offset to
be admissible: ``|offset| <= (1 - min_overlap) * n``.

* ``min_overlap=0.0`` (default) -- all ``2n + 1`` offsets: the intended sweep.
* ``min_overlap=1.0`` -- only full alignment, sweep disabled.

Measured, real data (658 bat-years, sit 300 s), leave-one-out, corrected sweep:

===============  ======  ========  =======  ======  ======
min_overlap      acc     precision recall   F1      AUC
===============  ======  ========  =======  ======  ======
0.0 (default)    0.701   0.845     0.695    0.762   0.700
1.0              0.708   0.873     0.677    0.762   0.711
0.0, first impl. 0.310   1.000     0.002    0.004   0.349
===============  ======  ========  =======  ======  ======

Majority baseline (always lactating): acc 0.691, precision 0.691, recall 1.000, F1 0.818.
Leave-one-*bat*-out at ``min_overlap=0.0`` gives 0.701 / 0.845 / 0.695 / 0.762 / 0.700 --
identical to leave-one-row-out, so bat-identity leakage is not a factor.
Onset output: 374 onsets returned, 39 distinct values spanning day 160-205 (early June to
late July), so the sweep still localises the jump.

``smooth_window`` (rolling-window average of the class means)
------------------------------------------------------------
Optional simple centred rolling average applied to **both class-mean series** before the sweep,
width in days (even widths bumped to odd so the average stays centred; prototypes zero-padded at
the year edges, harmless because activity there is ~0). It is linear, so smoothing the training
series and then averaging is identical to averaging and then smoothing the means.

Measured on the real data, leave-one-out, with the sweep as specified:

=============  ======  ========  =======  ======  ======
smooth_window  acc     precision recall   F1      AUC
=============  ======  ========  =======  ======  ======
1 (none)       0.701   0.845     0.695    0.762   0.700
3              0.695   0.841     0.688    0.757   0.700
5              0.696   0.842     0.690    0.758   0.700
7              0.696   0.842     0.690    0.758   0.700
11             0.695   0.841     0.688    0.757   0.699
15             0.698   0.840     0.695    0.761   0.699
21             0.698   0.839     0.697    0.761   0.699
=============  ======  ========  =======  ======  ======

No gain: AUC is flat and accuracy is flat to marginally lower. Reason, quantified: the
prototypes' day-to-day wiggle is 0.155 with a daily SEM of 0.070, while their seasonal swing is
**5.61** -- the decision runs on that seasonal level, which a low-pass filter cannot change. The
smearing that does exist comes from onset/duration jitter across bats (a *low*-frequency
misalignment), so smoothing makes it marginally worse rather than better; the fix for that is
align-then-average or a parametric template. Smoothing the individual series instead is equally
flat. Kept as an option, default is off (``1``).

Measured, synthetic sandbox (``data/processed/dummy_data.csv``, 1000 x 180)
--------------------------------------------------------------------------
===============  ======  ========  =======  ======  ======
min_overlap      acc     precision recall   F1      AUC
===============  ======  ========  =======  ======  ======
0.0 (default)    0.999   1.000     0.998    0.999   1.000
1.0              0.992   1.000     0.984    0.992   1.000
0.0, first impl. 0.893   1.000     0.786    0.880   0.999
===============  ======  ========  =======  ======  ======

The sweep is worth ~7 points on the sandbox (where the elevated phase can start anywhere)
and is neutral on the real data (where every series shares the same calendar window). Under
the corrected formulation it helps in both; under the first implementation it hurt in both.

Deferred to the real dataset (see ``TODO.md``): the equivalent but better-behaved decision via
the projection onto ``mean_1_ - mean_0_``, and a scale-invariant score -- the residual error on
the real data is amplitude-driven (normalised cross-correlation reaches AUC ~0.76 vs 0.70).

Caveat: ``predict(X, onset=True)`` returns a tuple, so it breaks the usual sklearn estimator
contract (``score``/``cross_val_score`` need the plain-label form).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["MeanTemplateClassifier"]


class MeanTemplateClassifier(ClassifierMixin, BaseEstimator):
    """Nearest-prototype classifier whose prototypes may sit anywhere in the series.

    Parameters
    ----------
    min_overlap : float in [0, 1], default 0.0
        Minimum fraction of the series' data that must overlap the prototype's data for an
        offset to be admissible: ``|offset| <= (1 - min_overlap) * n``. ``0.0`` is the full
        sweep, ``1.0`` disables it (full alignment only).

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
    mean_0_, mean_1_ : ndarray of shape (n_features,) -- the two class prototypes (the smoothed
        per-day means of the two labels when ``smooth_window > 1``).
    diff_template_ : ndarray -- ``mean_1_ - mean_0_`` (kept for inspection/plots).
    onset_anchor_ : int -- index inside the prototypes where the elevated phase is taken to
        start (50% of the rise of ``diff_template_``). The onset day of a series is
        ``onset_anchor_ + winning_offset + 1``.
    n_features_in_ : int
    """

    def __init__(self, min_overlap: float = 0.0, smooth_window: int = 1) -> None:
        """
        Parameters
        ----------
        min_overlap : float in [0, 1], default 0.0
            Minimum fraction of the series' data that must overlap the prototype's data for an
            offset to be admissible: ``|offset| <= (1 - min_overlap) * n``. ``0.0`` is the full
            sweep, ``1.0`` disables it (full alignment only).
        smooth_window : int, default 1
            Window width (days) of a simple centred rolling average applied to **the mean time
            series of both labels** before the sweep. ``1`` means no smoothing. Even widths are
            bumped to the next odd number so the average stays centred. The prototypes are
            zero-padded at the year edges, which is harmless because activity there is ~0.
        """
        self.min_overlap = min_overlap
        self.smooth_window = smooth_window

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
        if int(self.smooth_window) < 1:
            raise ValueError(f"smooth_window must be >= 1, got {self.smooth_window}")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.mean_0_ = self._smooth(X[y == 0].mean(axis=0))
        self.mean_1_ = self._smooth(X[y == 1].mean(axis=0))
        self.diff_template_ = self.mean_1_ - self.mean_0_

        # zero-padded prototypes: pad | data | pad
        n = self.n_features_in_
        self._pad0_ = np.concatenate([np.zeros(n), self.mean_0_, np.zeros(n)])
        self._pad1_ = np.concatenate([np.zeros(n), self.mean_1_, np.zeros(n)])

        # where the elevated phase starts inside the prototype (robust to the smoothing
        # caused by averaging misaligned individuals)
        half = 0.5 * self.diff_template_.max()
        self.onset_anchor_ = int(np.argmax(self.diff_template_ >= half))
        return self

    # ------------------------------------------------------------- helpers
    def _smooth(self, series):
        """Simple centred rolling-window average of a time series (the class-mean series)."""
        w = int(self.smooth_window)
        if w <= 1:
            return np.asarray(series, dtype=float)
        if w % 2 == 0:                      # keep the average centred
            w += 1
        kernel = np.ones(w) / w
        return np.convolve(np.asarray(series, dtype=float), kernel, mode="same")

    def _shift_bounds(self):
        """Inclusive range of relative offsets allowed by ``min_overlap``.

        Offset ``o`` places prototype index ``j`` on series index ``j + o``, so ``|o| = n``
        means the two data segments no longer overlap at all.
        """
        n = self.n_features_in_
        span = int(np.floor((1.0 - float(self.min_overlap)) * n))
        return -span, span

    def _offsets(self):
        lo, hi = self._shift_bounds()
        return np.arange(lo, hi + 1)

    def _mse_curves(self, x):
        """MSE over the crossing region of the two padded series, one value per offset.

        At offset ``o`` the crossing covers the indices ``i`` with ``0 <= i < 3n`` and
        ``0 <= i - o < 3n``; the score is the mean squared difference over that region, so
        it is ``3n - |o|`` wide and always contains real data from at least one side.
        """
        n = self.n_features_in_
        xp = np.concatenate([np.zeros(n), x, np.zeros(n)])
        offsets = self._offsets()
        mse0 = np.empty(len(offsets))
        mse1 = np.empty(len(offsets))
        for k, o in enumerate(offsets):
            a = max(0, o)
            b = min(3 * n, 3 * n + o)
            mse0[k] = ((xp[a:b] - self._pad0_[a - o:b - o]) ** 2).mean()
            mse1[k] = ((xp[a:b] - self._pad1_[a - o:b - o]) ** 2).mean()
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

        offsets = self._offsets()
        labels = np.zeros(len(X), dtype=int)
        onsets = np.full(len(X), np.nan)
        for k, x in enumerate(X):
            mse0, mse1 = self._mse_curves(x)
            if mse1.min() < mse0.min():
                labels[k] = 1
                if onset:
                    o = offsets[int(mse1.argmin())]
                    onsets[k] = self.onset_anchor_ + o + 1
        if onset:
            return labels, onsets
        return labels

    # ------------------------------------------------------------- extras
    def decision_margins(self, X):
        """min-MSE(mean_0) - min-MSE(mean_1) per sample. Positive -> predicted 1.

        Handy to see *how close* a call was.
        """
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        for k, x in enumerate(X):
            mse0, mse1 = self._mse_curves(x)
            out[k] = mse0.min() - mse1.min()
        return out
