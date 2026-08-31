# Capstone: Clinical Entity Extraction and Negation Detection

**Phase 8, Weeks 35–39 — final deliverable of a 39-week self-taught AI engineering curriculum.**

## Why this direction

The curriculum's stated goal from the start was AI + medicine. Given a choice between deepening the EHR/time-series work from Phase 7 or branching into clinical NLP, I picked NLP — partly for breadth, but mainly because it connects directly to real work: Transcribe, the medical transcription product I maintain at Intron, is squarely in this domain. A capstone that reproduces that kind of problem, end to end, is worth more than one built purely for the portfolio.

## The dataset

MTSamples — ~5,000 real medical transcription reports across 40 specialties, CC0 public domain. Chosen specifically to avoid the MIMIC-style credentialing wall that shaped the Phase 7 dataset decision: no data use agreement, no CITI training, immediately usable. Same lesson, applied a second time.

## What was built

A full pipeline, from raw MTSamples text to evaluated, negation-aware clinical entities:

1. **Boilerplate stripping** — MTSamples' own site template (`Sample Name:`, `Medical Specialty:`, `Description:`) gets mixed into every report and needs removing before any real NLP happens.
2. **Dual-model entity extraction** — two separate scispaCy passes over the same text:
   - `en_core_sci_sm`: general biomedical entity detection, generic `ENTITY` label
   - `en_ner_bc5cdr_md`: typed extraction, `DISEASE` and `CHEMICAL` specifically
     Run as two independent passes rather than merged, since reconciling entity boundaries across two different models is a real, separate problem in its own right — a deliberate scoping decision, not an oversight.
3. **Negation detection, built from scratch** — a NegEx-style scope-window detector: a fixed list of trigger phrases (`no`, `not`, `without`, `denies`, `denied`, `negative`), checking whether a trigger appears within _N_ tokens before an entity, bounded to the entity's own sentence so negation can't leak across sentence boundaries. Extended with a second check for cases where the model bundles the trigger word directly into the entity span itself (e.g. `"no pallor"` extracted as one unit) — a real gap found only by inspecting genuine model output, not from the NegEx literature.

## Real decisions, not defaults

**Window size (6 → 10).** Real data surfaced a genuine trade-off: `window=6` missed list-style negations (`"without guarding, rigidity, tenderness, mass or organomegaly"` — the trigger sits too far from the last item in a long list). Widening to `window=10` fixed that, but introduced new false negatives — `niacin`, sitting in a clause unrelated to an earlier `"without any recent change"`, got incorrectly swept up as negated purely because it fell within the wider token window. There is no window size that cleanly solves both; the real fix would need actual clause-boundary awareness (dependency parsing), not just token counting. Kept `window=10`, documented the trade-off with concrete examples rather than picking silently.

**Evaluation, with no gold-standard labels.** MTSamples has no entity annotations at all. The standard move — and the same instinct as Phase 4's calibration work and Phase 7's benchmark-comparison habit — was to build ground truth by hand: five real reports across different specialties, hand-annotated, checked against the pipeline's actual output. `compute_precision` reports five distinct numbers rather than one aggregate score: text-extraction precision/recall, label-typing precision/recall, and typing-accuracy-conditioned-on-correct-extraction — because a single blended number hides _where_ a pipeline is actually failing.

## What went wrong along the way, and what it means

Every one of these was caught by testing against concrete, real cases — not by code review or assumption:

- A dictionary-based entity store silently collapsed duplicate mentions (`hemorrhoids` appearing twice in one document, collapsed to one) — fixed by switching to a list-of-dicts structure.
- A first negation-scope check only looked _before_ the entity, missing cases where the trigger word was the entity's own first token — fixed with a second, independent check.
- An early `compute_precision` mutated the caller's own ground-truth list in place via `.remove()` during iteration — the same class of bug as an early Phase 7 list-mutation error, showing up again in a new form. Fixed by working on a local copy.
- The same function initially allowed **recall greater than 1.0** — a mathematically impossible result — because matched ground-truth entries were never "consumed," letting multiple predictions match the same entry repeatedly. Fixed with proper one-to-one consumption.
- Exact-text matching initially conflated genuine misses with simple boundary artifacts (`"vitamin D."` vs `"vitamin D"`, a trailing period; `"immune thrombocytopenia"` vs `"thrombocytopenia"`, an extra word) — fixed with directional substring matching. The first attempt at this fix checked the substring relationship backwards (asking whether the _shorter_ ground-truth text contained the _longer_ predicted text, which is never possible), silently matching nothing; correcting the direction — does the predicted text contain the ground-truth text — resolved both real cases.
- My own hand-written ground truth was itself incomplete on a first pass (`prednisone`, mentioned four times in one report, initially logged only twice) — caught by programmatically counting real mention frequency rather than trusting a manual read-through.

## Results

Five documents, different specialties, evaluated with the fully corrected harness (proper substring direction, one-to-one match consumption, verified ground truth):

|                      | Untyped (`en_core_sci_sm`) | Typed (`en_ner_bc5cdr_md`) |
| -------------------- | -------------------------- | -------------------------- |
| Text precision (avg) | 0.089                      | 0.663                      |
| Text recall (avg)    | 0.817                      | 0.857                      |

Precision differs sharply between the two models; recall is much closer. Same conclusion as the earlier pass, now on cleaner numbers: `en_core_sci_sm` is built to catch _any_ biomedical-sounding span (anatomy, exam findings, general vocabulary), so most of what it finds simply isn't a clean `DISEASE`/`CHEMICAL` entity by the ground truth's narrower definition — a scope difference, not a quality difference. Both models are similarly capable of finding real entities somewhere in their output; they differ mainly in how much off-scope material comes bundled with it.

A second, smaller finding worth naming: the typed model's `extracted_typed` score (label accuracy, conditioned on correct extraction) was `1.0` on four of five documents but `0.944` on text1 — meaning at least one entity there was correctly extracted but assigned the wrong type. A genuine, if rare, `DISEASE`/`CHEMICAL` confusion, not a negation or extraction failure.

**One outlier investigated directly**, rather than left as a number. Before the substring-matching fix, text5's untyped recall looked like the weakest of the five, muddied by boundary artifacts (`"vitamin D."` vs `"vitamin D"`, `"immune thrombocytopenia"` vs `"thrombocytopenia"`) that made it look worse than it was. With those artifacts correctly resolved, the number lands at exactly `6/9` — and every one of the three real misses is `prednisone`: `en_core_sci_sm` correctly extracted a bare `"prednisone"` once, but consistently failed to extract it when immediately followed by dosage phrasing (`"prednisone to 60 mg"`, `"prednisone to 40 mg"`, twice). A single, precise, repeatable failure mode — not five scattered problems — and only visible once the evaluation methodology itself was trustworthy enough to isolate it.

## What this pipeline does not do

It extracts and negation-filters clinical entities. It does **not** de-identify text. Real Safe Harbor identifiers — a named hospital (`"St. Luke's Episcopal Hospital"`), a full physician name (`"Dr. Chip Winkel"`), exact dates (`"01/09/07"`) — sit untouched in the same documents this pipeline processes, right alongside correctly-extracted clinical entities. De-identification is a genuinely different NER task (finding names/dates/facilities, not diagnoses), out of scope for what was built here, and this pipeline would need that separate pass before it could ever touch real patient text. Worth stating as an explicit boundary, not an implicit gap.

## Where this leaves things

Real, working pipeline: boilerplate handling, dual-model extraction, a from-scratch negation detector with two documented failure modes, and an honest evaluation methodology built specifically because no ready-made one existed. Every design decision — window size, evaluation structure, matching logic — is stated with the trade-off attached, not presented as if it were the only reasonable choice.

That's the close of the curriculum: 39 weeks, from a from-scratch linear regression to this.
