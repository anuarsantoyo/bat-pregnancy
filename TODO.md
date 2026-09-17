# TODO — bat-pregnancy

_Updated 2026-09-13_

## Done
- [x] Repo set up; `docs/gyad134.pdf` = Fontaine et al. 2024, *J. Mammalogy* 105:289–299
      (`10.1093/jmammal/gyad134`) — the method reference for stage-1 processing.
- [x] Sandbox dataset `data/processed/dummy_data.csv` (1000 bats x 180 days + `pregnant`
      label) with its seeded generator `src/generate_dummy_data.py` (reproducible, byte-identical).
- [x] `src/mean_template_classifier.py` — Anuar's rule implemented as specified
      (class-mean prototypes + zero-padded sliding-window MSE, lowest min-MSE wins,
      `predict(X, onset=True)` gives the jump start for label 1, `nan` otherwise).
- [x] First pass over the rescued PIT-tag CSVs: `notebooks/01-eda-detections.ipynb`
      (untracked so far).

## Done (2026-09-17) — the real dataset
- [x] **EDA 02** `notebooks/02-eda-daily-counts.ipynb` — one chapter per sit threshold; line plot of
      mean visits per bat-day ± 95 % CI for lactating / not lactating / unlabelled.
      **Lactating bats are the most active at every threshold** (Jul 5: 7.00 vs 2.40 at sit10s →
      4.02 vs 1.74 at sit600s), and the *ratio* is sit-robust (~2.3–2.9 everywhere).
      What is **not** robust is the separation: two-sample z on Jul 5 falls 4.7 (10s) → 10.2 (300s),
      because the extra events a fine threshold buys are ~equally spread over groups
      (chatter inflation at 10s = ×1.68 lactating / ×1.59 not lactating) — i.e. pure variance,
      no signal. ⇒ **prefer sit 300–600s**; sit10s is close to unusable as a visit count.
      Logger window ≈ Apr 20 – Sep 26 (160/365 days with any detection).
- [x] **Wide daily activity matrix built** — `src/build_daily_counts.py` →
      `data/processed/daily_counts/daily_counts_sit{10,30,60,120,300,600}s.csv`
      (2,070 rows each: one bat-year; `Year, Bat Id, Colonie, Day1..Day365, Lactating`).
      Bat-day = 12:00→12:00, Feb 29 dropped so 365 columns always.
      `Lactating` = `Reproduction State` joined on `ID4` (suffix match, see below).
- [x] **`ID4` is a partial tag id** ("last 4 digits, longer if not unique") → joins to the
      10-char activity `id` by **suffix**, not equality (only 3 exact matches).
      223/233 resolve; 4 are genuine last-4 collisions (`0832`, `9900`, `AA3C`, `F44A`) → left
      unlabelled. 7-digit activity tags are zero-padded and merged into their 10-char form.
- [x] **Cross-colony movers exist in the activity data** (68 (year,tag) combos at ≥2 colonies
      at every sit level; e.g. `8E602D0D0F` 2011 = BS 203 / GB2 394 / UA 42 at sit600s), but
      the label table assigns each individual **one colony for life** (0 of 233 `ID4`s move).

## Done (2026-09-17) — real-data test of the model
- [x] **05** `notebooks/05-corrected-sweep.ipynb` + **sweep implemented as specified** — the real fix.
      Pad both series, sweep the 2n+1 offsets, MSE over the **crossing region**. Real LOOCV
      **0.310 → 0.701** (P 0.845, R 0.695, F1 0.762, AUC 0.700), sandbox **0.893 → 0.999**, and
      onset localisation restored (374 onsets, 39 distinct days, day 160–205). `min_overlap` kept
      as an optional restriction, default 0.0 = full sweep. Notebooks 03/04 bannered as historical.
- [x] **03** `notebooks/03-loocv-mean-template.ipynb` (**historical record**) — the failure and its
      diagnosis against the misimplemented sweep: LOOCV acc 0.310, precision 1.000, recall 0.002,
      F1 0.004, below the 0.691 majority baseline; 479/658 bats scored by a constant.
- [x] **04** `notebooks/04-min-overlap-repair.ipynb` (**historical record**) — the interim
      `min_overlap` workaround, which reached 0.708 by forbidding the bad alignments instead of
      fixing the sweep.

## Next (post-dataset)

- [ ] Turn `Day1..Day365` into per-bat-year *features* (e.g. n_active_days, mean/max nightly
      count, first/last activity date, mid-night (23–03h) share) and try to reproduce the
      table's own `n_days_Tq` / `Box_*` columns as a sanity join before modelling.
- [ ] Decide sit threshold (10s vs 600s) as *the* representation — EDA 02 says **300–600s**
      (best group separation, least chatter); 10s adds ~68 % more events that are pure variance.
- [ ] **Remaining model error is amplitude, not shape.** Ranking by *normalised* cross-correlation
      reaches AUC ~0.76 vs 0.70 for the corrected min-MSE rule. Next lever = a scale-invariant score
      (or per-bat normalisation), not another alignment tweak.
- [ ] F1 0.762 is still under the "always lactating" F1 of 0.818 — the gain is in accuracy/precision/AUC.
      Recall 0.695 means ~139 lactating bat-years are still missed; look at what they have in common.
- [ ] The sweep is neutral on the real data (winning offset is a median 8 days from full alignment):
      worth asking whether the real series should be trimmed to the Apr–Sep window before modelling.

## Blocked on the real dataset (expected ~2026-09-15)
> Nothing below should be decided on the synthetic sandbox — the dummy data is
> unrealistically kind (identical per-bat baselines, no roost switching).

- [ ] **Decision rule.** Keep the MSE rule, or switch to the equivalent projection onto
      `mean_1_ - mean_0_` (sandbox: acc 0.893 -> 0.996, AUC 1.000; same shift search, one
      template, and it makes the threshold a tunable knob instead of a fixed one).
      Re-run both and compare before choosing.
- [ ] **Pooling strategy.** The real setup has several box antennas, so passes at two
      loggers can be roost switching rather than a foraging bout. Per box vs per individual
      across boxes changes what "5 entries/day" even means.
- [ ] **Onset output format.** Currently `nan` for label 0 (vectorised). Switch to `None`
      / `-1` if that suits downstream use better.
- [ ] **What is the training unit?** one row per individual-season (~20–60 per roost per
      year in the paper) — confirm the real dataset has enough rows to fit *and* validate.
- [ ] **Validation data.** Ask for the field sheet (tag -> species, sex, reproductive state
      from palpation/palpation of nipples) so accuracy can be checked against ground truth,
      like the paper does (92% agreement in their subset).
- [ ] **Minimum-overlap / edge rules.** Confirm the zero-padding behaviour is still the
      right call once real baselines differ between individuals.

## Known weaknesses of the current model (measured on the sandbox)
- **Error asymmetry:** 107/500 pregnancies missed, 0 false alarms. For a truly pregnant bat
  the winning margin is only **+0.24 MSE** (2.82 vs 3.07) because 145 of the 180 days are
  identical under both prototypes — the bump is diluted ~5x.
- **Fixed threshold:** "lowest MSE wins" bakes in the decision boundary (the constant
  `sum(mean_1^2 - mean_0^2)/2`); no sensitivity/specificity dial.
- **Baseline sensitivity (expected to bite on real data):** raw MSE is dominated by absolute
  activity level, so a generally more active bat drifts toward the pregnant prototype.
  Sandbox cannot show this because all baselines are 2.0 by construction.
- Onset recovery on the sandbox is actually good: median error +0 d, IQR [-2,+1], 99.7%
  within +/-7 d (an earlier "±10 d" claim of mine was a bug in a throwaway probe, retracted).

## On the shelf (deliberately not implemented yet)
- [ ] Decision by projection onto the difference template (see above).
- [ ] Per-series baseline removal (subtract each bat's median) before matching — fixes the
      activity-level confound. Note: if we ever centre, the zero padding must move to the
      baseline value, otherwise the edge penalty silently disappears.
- [ ] Sharpen the prototype: the class mean is the average of misaligned windows (onset
      jitters, duration jitters), so the template is smeared. Options: parametric boxcar
      template, or align-then-average (EM-ish).
- [ ] `predict_proba` / calibrated score curve (the matched-filter score is monotone in
      evidence and could be turned into a probability).
- [ ] Duration inference: the width of the correlation peak should encode the windows'
      length, not just its start.

## Data issues to raise with the provider
- **Excel-mangled dates** in the exported CSVs: 498/998 Date-column values are Excel serials
  / garbage (`6461581`, `15.03.7407`). Recoverable from the filename, but fix at export.
- **Mixed delimiters:** half the files are `;`-delimited with quoted codes, half are
  `,`-delimited without quotes (Excel round-trip). Parser is delimiter-agnostic but this
  should be standardised.

## Open questions about the real data
- Is `Mbec` *Myotis bechsteinii*? What is site `GB2`?
- What does the `empty` / `TQ` flag in the filenames mean (box state, logger config)?
- `K##` = box number and `LID##` = logger id? (observed: LID31 -> {K46,K89},
  LID37 -> {K9,K84,K81}, LID32 -> {K37,K44})
- Is tag `00079ABB95` a stationary/reference tag? It fires every morning 09:30–11:30 on all
  five nights logged so far.
- Does the real dataset cover the full reproductive season (paper needs the
  gestation -> lactation transition, i.e. June–July), or just the May pilot?

## Infra
- [ ] **Push access** — repo cloned read-only; commits are local until a token with
      Contents: read & write is provided.
- `data/` is in `.gitignore`; `data/processed/dummy_data.csv` is force-added.
  Decide the policy for the real data (likely keep raw data out of git).
