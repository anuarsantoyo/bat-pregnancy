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

## Next (post-dataset)

- [ ] Turn `Day1..Day365` into per-bat-year *features* (e.g. n_active_days, mean/max nightly
      count, first/last activity date, mid-night (23–03h) share) and try to reproduce the
      table's own `n_days_Tq` / `Box_*` columns as a sanity join before modelling.
- [ ] Decide sit threshold (10s vs 600s) as *the* representation — they differ ~1.6× in total
      events and near-raw 10s counts can be 4,865 in a single bat-year.

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
