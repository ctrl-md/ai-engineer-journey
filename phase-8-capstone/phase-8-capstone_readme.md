# Phase 8 — Capstone: AI + Medicine

The final phase of the curriculum. Landscape survey and direction-setting (Week 35), regulatory realities and HIPAA (Week 36), the capstone build (Weeks 37–38), and the final writeup (Week 39).

## Weeks 37–39: Clinical entity extraction and negation detection

A full clinical NLP pipeline built on MTSamples (~5,000 real medical transcription reports, CC0 public domain — chosen specifically to avoid the MIMIC-style credentialing wall from Phase 7):

- Boilerplate stripping for MTSamples' own site template
- Dual-model entity extraction (`en_core_sci_sm` for generic biomedical spans, `en_ner_bc5cdr_md` for typed `DISEASE`/`CHEMICAL` extraction), run as two independent passes
- A from-scratch NegEx-style negation detector — trigger-word scope windows, sentence-boundary aware, extended to catch cases where the trigger word is bundled directly into the entity span itself
- An evaluation harness built from scratch against hand-annotated ground truth, since MTSamples has no entity labels — precision, recall, and typing accuracy reported separately rather than as one blended score

Full writeup, including every real bug found and fixed along the way and the honest trade-offs behind each design decision: [`capstone-writeup.md`](./capstone-writeup.md).

**Explicit scope boundary**: this pipeline extracts and negation-filters clinical entities. It does not de-identify text — real HIPAA Safe Harbor identifiers (names, dates, facilities) are untouched in its output. See the writeup for detail.

Run it: `python weeks_37_38_clinical_entity_extraction.py`
