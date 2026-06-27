# taste_exemplars

`deai.py taste draft.md` measures how far a draft sits from **tasteful** writing — concreteness,
reader engagement, low metadiscourse, Writer's-Diet leanness, cohesion — using the same
neuro-symbolic core as the rest of the tool (Coh-Metrix / Hyland / Sword) plus an exemplar-anchored
LLM read. The exemplars are the *ceiling*, the way the venue corpus is the *floor* for `mirror`.

This folder ships empty because the best exemplars are usually copyrighted. Drop in a handful of
`*.md` pieces you consider beautifully written **in or near your genre** — the corpus is the standard.
Good starting points (fetch the text yourself; do not redistribute): Ha & Schmidhuber, *World Models*;
Chris Olah's Distill articles; Karpathy's essays; or the clearest-written papers in your field.

Then either keep them here, pass `--exemplars DIR`, or set `DEAI_TASTE_EXEMPLARS`. The metrics are
**diagnostic, not a target** — optimizing them yields lean, concrete, soulless prose. Use them to find
what reads machine-made; fix those things because they are real.
