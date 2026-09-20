# Loan Risk Assessment

A two-page questionnaire about a loan applicant returns a risk band, a
calibrated probability of default, and a readable account of what drove the
estimate. It is built for someone who wants to see how a credit risk
assessment is put together — which factors are learned from data, which are
asserted as policy, and where the two disagree. It is an assessment tool, not
a decision engine: every applicant is typed in by hand or loaded from a
synthetic demo profile, and there is no bureau or Account Aggregator
integration anywhere in it.

**Live app:** _deploy to Streamlit Community Cloud and put the URL here_
**Screenshots:** _add `docs/screenshot-result.png` and a short GIF of the
demo-profile flow_

---

## Status

The committed model artefact is a **placeholder scorecard** — hand-specified
coefficients, not a trained model. The app runs end to end on it, the
six demo profiles score, and 35 tests pass, but the probabilities are
illustrative until `notebooks/05_train.py` has been run against Lending Club
data. The app shows a warning banner whenever the placeholder is loaded.

The committed pincode reference is likewise a **24-row sample** covering the
tested pincodes. `python data/build_pincode_reference.py` exports the full
table (~20k pincodes) from the source Google Sheet.

---

## Quick start

```bash
git clone <this repo> && cd loan-risk-assessment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# build the committed artefacts
python data/build_pincode_reference.py --sample   # or without --sample for the full table
python models/build_placeholder_model.py

# run
streamlit run app/streamlit_app.py

# test
pytest                  # or: python tests/run.py  (no pytest required)
```

The sidebar carries six demo profiles. Any of them shows the app working in
about fifteen seconds without typing anything.

---

## The risk framework

Two groups of inputs, and the UI marks every factor as one or the other.

### Model features — learned

Mapped onto the training schema and fed to the PD calculation.

| Factor | What it measures | Why it matters |
| --- | --- | --- |
| Bureau score | Aggregate repayment history | The single strongest predictor in every consumer credit dataset, and the largest contribution for most applicants here |
| Annual income (log) | Capacity | Log scale because the marginal effect of income falls sharply; the difference between ₹30k and ₹60k a month matters far more than between ₹3L and ₹6L |
| DTI at application | Existing leverage | Obligations already committed before this loan |
| Years in current job | Income stability | Short tenure predicts default independently of income level |
| Residence type | Fixed-cost exposure and stability | Renting carries a cost the lender cannot see in the EMI |
| Revolving utilisation | Revolving stress | A borrower running cards near the limit is short of liquidity now, not historically |
| Open accounts | Credit footprint | Weak on its own; interacts with utilisation and enquiries |
| Delinquencies, 2 years | Recent repayment failure | Direct evidence of the behaviour being predicted |
| Enquiries, 6 months | Credit seeking | Rate-shopping and distress borrowing look the same to a bureau |
| Loan amount (log) and loan-to-income | Size of the ask relative to capacity | A large loan relative to income fails on a smaller shock |
| Tenure over 36 months | Exposure window | Longer terms carry more chances to default |
| Purpose bucket | Use of funds | Grouped rather than one-hot, to keep the baseline readable |

### Policy signals — asserted

Collected by the questionnaire but absent from the public training data, so
they cannot be learned. They run through a deterministic rule layer in
`engine/policy.py`, each flag naming the value that fired it so a reviewer
can disagree with the threshold rather than with the output.

Per-trade-line DPD, write-offs, settlements, suit-filed, NACH and cheque
bounces, co-applicant profile, cash-flow fields, and negative surplus.

**Policy can only ever raise severity, never lower it.** There is no path by
which an asserted rule improves a learned estimate. `tests/` asserts this
across every demo profile.

The downgrade rule is sub-additive and capped: worst single flag, plus one
step if three or more substantive flags stack up, capped at two bands. One
step per flag was the first attempt, and with five bands it sent anyone
carrying three minor flags to the worst band — which destroys the resolution
the bands exist to provide.

**Hard knockouts** are kept apart from all of the above, in the engine and on
the result screen. An active written-off account is a different kind of fact
from a slightly high FOIR, and averaging them into one score is the mistake
the separation exists to prevent. Knockouts force the worst band regardless
of the model's estimate.

### The expense floor, and why the dependent fields exist

Plain FOIR scores a single earner with no dependents identically to a
household of five at the same income. For education lending that is wrong in
a way that matters, because the co-applicant parent is often supporting
siblings on the same salary.

```
expense  = base cost for a single adult at the applicant's city tier
         + per dependent child
         + per dependent adult
         − discount when the residence is owned
surplus  = net income + other household income − expense − all EMIs
```

Children and dependent adults are counted separately: a child and an elderly
parent do not carry the same cost, and collapsing them into one "dependents"
number throws away the distinction the questionnaire collects. A negative
surplus is a hard policy flag whatever plain FOIR says.

All four numbers live in `config/config.yaml` with a dated provenance
comment. The `high_foir_clean` demo profile exists to exercise exactly this:
spotless repayment record, ₹81k net in Mumbai, four dependents, and a surplus
of about −₹15k a month. Plain FOIR alone would not have condemned it.

---

## Model card

**Dataset:** Lending Club accepted loans, 2007–2018.

**Why Lending Club over Home Credit.** Home Credit is the closer schema
match — it ships an actual bureau table with per-account amounts and DPD,
plus `CNT_CHILDREN` and `CNT_FAM_MEMBERS`, which would have made dependents
a legitimate model feature rather than a policy signal. It was still the
wrong choice here, for one decisive reason: **Home Credit's application table
carries no absolute origination date**, only `DAYS_*` fields measured
relative to each application. Temporal validation is therefore impossible
with it, and temporal validation is not optional in credit modelling. Lending
Club has `issue_d`. The cost of the choice is that dependents feed only the
expense model and the policy layer.

**Target:** binary. Charged-off or defaulted = 1, fully repaid = 0. Loans
still in progress are **dropped**, not treated as good — a loan eight months
into a sixty-month term has no known outcome, and coding it as repaid dilutes
the positive class with loans that will default later.

_Observed default rate: fill in from `notebooks/01_load_and_target.py`._

**Excluded features.** 46 columns, in four groups, each with a reason.
`python notebooks/leakage.py` prints the full table. In summary:

1. **Post-origination** — payment history, recoveries, principal received,
   last payment date, collection and settlement fields, hardship flags,
   post-loan bureau scores. Straightforward target leakage: consequences of
   repayment rather than predictors of it. `recoveries` is nonzero *only* for
   defaults.
2. **The lender's own assessment** — `grade`, `sub_grade`, `int_rate`,
   `installment`. Not leakage in the temporal sense, but circular: these
   *are* Lending Club's risk model output. Predicting default from them
   measures how well we reproduce their scorecard, which is not the question.
3. **Identifiers and free text** — `id`, `url`, `desc`, `emp_title`
   (high-cardinality and a proxy for protected attributes).
4. **Geographic** — `zip_code`, `addr_state`. A design decision, not a
   statistical one. See below.

**Validation:** split by time, on origination cohorts. Train on earlier
years, calibrate on a middle slice, test on later years. A random split leaks
the macro environment — loans originated in the same month share an
unemployment rate, a house-price trend and an underwriting policy, so a
randomly held-out loan has near-identical siblings in training, and reported
AUC comes out several points above anything achievable on future applicants.
The calibrator gets its own slice, earlier than the test set, or it is fitted
on the same data the metrics are reported on.

**Class imbalance:** class weights, never oversampling. Oversampling before
the split duplicates rows across train and validation and inflates every
metric; doing it after still distorts the predicted probabilities, and this
app displays a probability.

**Metrics:** _fill in from `notebooks/05_train.py`._

| | AUC | Gini | KS |
| --- | --- | --- | --- |
| Logistic (baseline) | | | |
| Gradient boosting | | | |

Both models are kept and the trade-off reported, because measuring it is
worth more than shipping only the stronger one. If the gap is under about
0.01 AUC, ship the logistic model: a scorecard whose coefficients can be read
off and argued with is worth more here than a marginal ranking gain, and the
app's attribution is then exact rather than approximated.

**Calibration:** isotonic, on the held-out middle slice. Required, because
the app displays PD as a percentage — if it says 7%, the observed rate in
that bucket should be near 7%. The calibration plot goes in
`notebooks/_work/figures/`.

**Bands:** cutoffs from PD quantiles on the held-out test set, not round
numbers. A 5% boundary implies someone decided 5% was meaningful; a boundary
of 0.0532 admits the bands are a partition of the observed distribution and
nothing more. Each band displays the default rate observed in it.

---

## Why location is excluded from the model

Pincode, city, state, region and tier are used for **expense estimation and
display only**. They are never model features and never one-hot encoded into
training.

Geographic risk pricing is the structural form of redlining. In the Indian
context this is sharper than in the US: a pincode proxies caste and religion
closely enough that a model given location would learn them from location
alone, without any protected attribute ever appearing in the training data.
The same reasoning removes `zip_code` and `addr_state` from the Lending Club
feature set.

What location *is* used for is the cost-of-living lookup, which is a
transparent arithmetic adjustment with four numbers in a config file, not a
learned coefficient. Anyone can disagree with the Metro base cost by reading
one line.

---

## Limitations

Stated plainly, because several of them are load-bearing.

**Trained on accepted applicants only.** Lending Club publishes the loans it
funded. The model therefore says nothing about applicants who were rejected,
and the rejection decision was itself made with a risk model. This is
reject inference and it is not addressed here at all.

**Trained on US consumer loans.** The coefficients are not transferable to
the Indian market, and two bridges paper over the gap:

- *Bureau score.* CIBIL's 300–900 is linearly rescaled onto FICO's 300–850.
  Monotone, which is what the model needs, but the two distributions are not
  the same shape. Flagged on the result screen.
- *Currency.* Amounts are scaled by a single income-anchored factor, not an
  FX rate. FX would be the wrong conversion: at 60 INR/USD every Indian
  income looks destitute relative to the training distribution. The anchor
  puts a median Indian salaried income at the median of the training
  distribution and preserves loan-to-income exactly. It preserves rank and
  ratio, not level.

**The population transfer is visible and only partly fixed.** Three features
behave structurally differently in the target population: `term_60` fires for
essentially *every* Indian education loan, and `log_loan_amnt` and
`loan_to_income` clamp at the top of the training support for most
applicants, because education loans are large relative to income compared
with US personal loans. Untreated, those three add a near-constant penalty to
every applicant and the whole portfolio lands in the worst band.

The response is intercept recalibration — a constant shift in log-odds,
derived by `notebooks/07_population_shift.py` against a reference grid, which
cannot change ranking or break monotonicity. It is the minimum correct step
and it is not sufficient: the coefficients are still US coefficients, and a
constant cannot fix a slope. The script warns when the shifted spread is too
wide, which it currently is under the placeholder. Features that clamp for
most of the population carry no information and should be dropped at training
time rather than absorbed by the shift — `term_60` is the obvious candidate.

The app clamps any feature outside the training support and reports the clamp
as a caveat on the result screen, rather than extrapolating silently. A
gradient booster asked about a region it never saw produces confident
nonsense.

**Weights are reasoned, not calibrated on local default data.** The expense
numbers, policy thresholds and band penalties are judgement calls with dated
provenance comments, not parameters fitted to observed Indian outcomes. Only
default outcomes on Indian education loans would validate them.

**Band separation is limited under the placeholder.** Three of the six demo
profiles currently land in band E because the placeholder's coefficients are
too steep for this population. This resolves when a real model replaces it.

**Illustrative, not a production credit decision engine.** No claim of
regulatory compliance or production readiness. No real personal data — every
applicant in the repo is synthetic, and the UI says so.

---

## Architecture

```
app/          Streamlit pages, view components, session-state mapping
engine/       pure scoring: feature mapping, model wrapper, policy rules,
              expense model, pincode lookup, bands, export, persistence
config/       every weight, cutoff, threshold and expense number — one file
notebooks/    EDA, feature work, training, calibration, band derivation
models/       serialised model + calibrator + feature schema, versioned
data/         committed pincode artefact + synthetic demo profiles only
tests/        35 tests, runnable with or without pytest
```

Three decisions worth knowing about:

**Scoring is a pure function.** `engine.score.assess(applicant, model=...,
reference=..., cfg=...)` does no file reads, no database writes, no clock
reads unless you pass `as_of`, and imports nothing from Streamlit. The model
and the pincode reference are loaded by the caller and injected. That is what
makes the counterfactual engine possible: every "what would change this"
suggestion is re-scored through the same function, so a suggestion can never
claim an improvement the engine would not actually produce.

**The engine does not depend on Pydantic.** Validation happens at the UI
boundary in `engine/schemas.py`, which emits plain dataclasses from
`engine/types.py`. The consequence is that the whole test suite runs in an
environment with neither Streamlit nor Pydantic installed.

**Config and model versions are stamped on every assessment.** A saved
assessment read back under different weights is a different assessment.

### The form-boundary trade-off

The spec asked for three things that pull against each other: wrap each page
in `st.form` so the app does not rerun on every keystroke, fire the pincode
lookup on entry rather than on submit, and show live derived values in the
footer as the user types. A widget inside `st.form` cannot trigger a rerun,
so the first is incompatible with the other two.

The resolution: the pincode input, the no-credit-history toggle and the
trade-line editor sit *outside* the form, and the twenty-odd scalar fields
sit inside it. The inline lookup and live footer work; the fields that would
cause a rerun per keystroke do not.

---

## Tests

```bash
pytest                 # normal
python tests/run.py    # no pytest needed
```

35 tests. The ones that matter most:

- **Monotonicity** — raising income, lowering obligations or raising the
  bureau score must never increase PD, checked across a sweep of values. A
  model that gets this wrong is broken in a way no AUC will reveal, and a
  reviewer who knows credit will try exactly this.
- **One per demo profile**, each asserting the band *and the mechanism that
  produced it*. Asserting the band alone passes for the wrong reason: a
  profile can reach E because the model scored it badly or because a knockout
  fired, and those are different claims.
- **Policy directionality** — policy never improves a band, on any profile.
- **Surplus** falls monotonically as dependents rise, goes negative, and
  fires the flag when it does.
- **Pincodes** — 400001 → Metro, 110001 → Metro, 504273 → Tier 3. `999999`
  and `012345` rejected as malformed rather than looked up (first digit 9 is
  Army Postal Service, 0 is unassigned). A valid-but-absent pincode takes the
  manual path and the assumption reaches the result screen. Unique index
  holds after load.
- **Feature schema** — a missing, out-of-range or unknown feature raises
  rather than silently imputing. A median-imputed bureau score is
  indistinguishable downstream from a real one.
- **Edge cases** — zero income, no trade lines, a single line at 90+ DPD,
  FOIR over 100%, applicant and co-applicant both thin-file.

---

## Non-goals

No real bureau or Account Aggregator integration. No real personal data. No
claim of regulatory compliance or production readiness.
