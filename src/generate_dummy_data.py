"""Generate the synthetic bat activity dataset used as a sandbox for the reproductive-status model.

Output: data/processed/dummy_data.csv  (git-tracked; see note in README below)

Layout
------
1000 rows (one per bat) x 180 day columns (`day_001` ... `day_180`) + a final `pregnant` label (1/0).
Each cell is an integer: the number of roost entries recorded for that bat on that day.

Generative model
----------------
Baseline : every bat makes ~2 entries/day  -> Poisson(2.0)     (days realistically range 0-4)
Pregnant : label == 1 bats additionally show a ~5-week window near the middle of the series
           where daily entries become Poisson(5.0)             (~5 entries/day)
Window   : start ~ Normal(72, 8)  clipped to [40, 110)
           duration ~ Normal(35, 4) days clipped to [28, 42]
Labels   : balanced, 500 / 500.

The window overlap with the baseline (Poisson 5 vs 2) is deliberate: a single day is not
diagnostic, the *shape over weeks* is. That mirrors the real PIT-tag data, where an individual
shifts from one emergence sequence per night (gestation) to several (lactation)
-- see Fontaine et al. 2024, J. Mammalogy 105:289-299, doi:10.1093/jmammal/gyad134.

Reproducibility
---------------
Fixed SEED => byte-identical output.
    python src/generate_dummy_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

# --- parameters (change here, not downstream) ---
SEED = 42
N_BATS = 1000
N_DAYS = 180
BASE_LAMBDA = 2.0                    # entries/day for every bat
PREG_LAMBDA = 5.0                    # entries/day for pregnant bats inside the window
START_MEAN, START_SD = 72.0, 8.0     # window start (keeps the ~35 d window mid-series)
DUR_MEAN, DUR_SD = 35.0, 4.0         # ~5 weeks
START_CLIP = (40, 110)
DUR_CLIP = (28, 42)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "processed" / "dummy_data.csv"


def generate(seed=SEED, n_bats=N_BATS, n_days=N_DAYS):
    rng = np.random.default_rng(seed)

    labels = np.array([0] * (n_bats // 2) + [1] * (n_bats // 2))
    rng.shuffle(labels)

    counts = rng.poisson(BASE_LAMBDA, size=(n_bats, n_days))       # baseline for everyone

    window = np.zeros((n_bats, 2), dtype=int)                      # [start, end) per pregnant bat
    for i in np.flatnonzero(labels == 1):
        start = int(np.clip(rng.normal(START_MEAN, START_SD), *START_CLIP))
        dur = int(np.clip(rng.normal(DUR_MEAN, DUR_SD), *DUR_CLIP))
        end = min(start + dur, n_days)
        window[i] = (start, end)
        counts[i, start:end] = rng.poisson(PREG_LAMBDA, size=end - start)

    cols = [f"day_{d:03d}" for d in range(1, n_days + 1)]
    df = pd.DataFrame(counts, columns=cols)
    df.insert(0, "bat_id", [f"bat_{i:04d}" for i in range(n_bats)])
    df["pregnant"] = labels
    return df, window, labels


def main():
    df, window, labels = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    day = [c for c in df.columns if c.startswith("day_")]
    vals = df[day].to_numpy()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"shape {df.shape} | labels {df.pregnant.value_counts().to_dict()}")
    print(f"non-pregnant mean/day {vals[labels == 0].mean():.2f} | "
          f"pregnant mean/day {vals[labels == 1].mean():.2f}")

    inside, outside = [], []
    for i in np.flatnonzero(labels == 1):
        s, e = window[i]
        inside.append(vals[i, s:e].mean())
        outside.append(np.r_[vals[i, :s], vals[i, e:]].mean())
    print(f"pregnant: mean inside window {np.mean(inside):.2f} vs outside {np.mean(outside):.2f}")
    starts = window[labels == 1, 0]
    durs = window[labels == 1, 1] - window[labels == 1, 0]
    print(f"windows: start {starts.min()}-{starts.max()} (mean {starts.mean():.1f}) | "
          f"duration {durs.min()}-{durs.max()} d (mean {durs.mean():.1f})")


if __name__ == "__main__":
    main()
