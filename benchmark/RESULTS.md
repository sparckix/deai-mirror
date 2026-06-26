# Benchmark: does the deterministic mirror separate human-venue from AI-default prose?

Two runs. The first (deepseek-only) was optimistic; the second (three models) is the
honest number. Metric = separation AUC (P[AI score > human score]; 0.5 = chance, 1.0 = perfect).

## Run 1 — deepseek-only (optimistic)

Human = 14 venue corpus papers. AI = 10 deepseek-chat essays on matched
AI-governance topics, no style/de-AI instruction (AI-default prose).

| signal | AUC |
|---|---|
| function-word Burrows's Delta (leave-one-out) | 0.75 |
| best single tell (triadic) | 0.79 |
| deai_lib flag-count (shape + lexical composite) | 0.93 |
| **combined: function-word Delta + flag-count** | **0.96** |

## Run 2 — three models (HARDENED — the honest number)

Human = 14. AI = 26 (deepseek ×10, grok ×8, kimi ×8). Same corpus, same metric.

| signal | human med | AI med | AUC |
|---|---|---|---|
| function-word Burrows's Delta | 0.81 | 1.18 | 0.78 |
| deai_lib flag-count (composite) | 5.0 | 7.0 | 0.72 |
| triadic | 9.08 | 11.05 | 0.67 |
| ai_vocab / transition_open | — | — | 0.62 |
| that_density | 13.51 | 13.80 | 0.58 |
| compound_and | 9.79 | 10.00 | 0.55 |
| abstract_subj | 0.22 | 0.00 | 0.30 |
| what_cleft | 0.66 | 0.00 | 0.13 |
| **combined: function-word Delta + flag-count** | | | **0.84** |

## Findings

1. **Model diversity drops the combined signal from 0.96 to 0.84.** Deepseek's
   AI-default prose is the easiest to separate; grok and kimi essays sit closer to
   the human-venue distribution. The 0.96 was a single-model artifact — this is why
   you harden a benchmark with more than one generator.
2. **The composite still beats every single tell.** Function-word Delta (0.78) and
   the flag-count composite (0.72) are the durable cross-model discriminators; the
   combined neuro-symbolic signal (0.84) is the headline. No individual tell exceeds
   ~0.78 — the power is multivariate, which is exactly why single-pattern
   LLM-"humanizer" skills are weak.
3. **Some syntactic-shape tells are model/venue-specific, not universal.**
   abstract_subj (0.30) and what_cleft (0.13) *invert* on the three-model set:
   grok/kimi under-use them relative to the human venue corpus, so they were
   over-fit to deepseek/academic-AI. The universal-ish survivors are function-word
   Delta, the flag-count composite, and triadic (0.67). Keep the universal defaults
   hand-set tight; let the pluggable corpus carry the venue-specific calibration.

## Run 3 — Register 2: deliberative AI (the discourse-stance test)

Human = 14. AI = 20 essays from `generate.py --r2` (deepseek + grok), a DELIBERATIVE prompt ("argue but be rigorous
and self-aware about the argument's limits"). Tests whether the discourse-stance tells separate once the AI prose
is deliberative rather than one-shot.

| tell | naïve AUC (Run 2) | deliberative AUC |
|---|---|---|
| intensifier | 0.74 | **0.96** |
| abstract-subject openers | 0.43 (inverted) | **0.91** |
| antithesis (not-but) | 0.61 | 0.68 |
| adjective rate | 0.78 | 0.79 |
| `metanarr_synt` (syntactic meta-narration) | 0.24 (inverted) | 0.57 |
| `meta_narration` (regex) | 0.46 | 0.52 |

**Findings.** A deliberative prompt makes AI *more* detectable, via a DIFFERENT tell profile — intensifiers and
abstract-subject openers jump (the latter inverts from human-higher to strongly AI-higher). Each register has its
own signature, and the multivariate composite covers both. But **meta-narration proper only moves weakly**
(0.24→0.57; the regex idioms stay ~0): it is an *iterative-revision* artifact, not a deliberative-prompt one, and
fully manifests only after many self-critique passes (a heavily-revised draft hit `metanarr_synt` 4.6 vs corpus
0.0). So the deliberative set validates the *stance register*; meta-narration's strongest evidence remains the
revised draft itself. A multi-pass self-critique generator is the next step to validate meta-narration directly.

## Caveats (the honest frontier)

- Still small (26 AI, 14 human), and the flag thresholds are computed in-sample on
  the human set, so the flag-count/combined AUC is somewhat optimistic. A
  leave-one-out threshold pass is the next step. (The function-word Delta *is*
  leave-one-out already.)
- The human venue corpus may itself be partly AI-contaminated (recent papers),
  which would compress separation. A pre-2022 human baseline would sharpen the test.
- This is **directional validation, not a certified number.** It earns "the
  composite separates AI-default prose, weaker under model diversity (0.84)" — not
  "certified." Proving it wrong is a standing invitation.
