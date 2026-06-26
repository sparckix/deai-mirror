---
name: deai-conform
description: >
  Measure how far a draft sits from a target venue's real human prose, then rewrite it to conform — using deai.py as
  the deterministic oracle, never a "feels human" vibe. Dual-channel (is-it-AI + venue-conformance) and
  anti-over-correction (target the human band's floor AND ceiling, so you never strip below human into a new tell).
allowed-tools: [Read, Write, Edit, Bash, AskUserQuestion]
---

# deai-conform

A **measured** rewrite loop. The deterministic Python (`deai.py`) is the oracle; the agent (you) does the rewriting.
Do not edit to a feeling — edit to close the *measured* gap against a real venue corpus, and stop when the tells sit
**inside** the human band (not below it). The rewrite belongs here in the agent loop; the Python only measures.

## Inputs
- `draft` — the markdown file to conform.
- `corpus` — a folder of real `*.md` papers from the **target venue** (the human standard; bring your own).
- `ai_corpus` *(optional)* — AI-default samples, to also run the is-it-AI channel.

## The loop

0. **Prime, before writing a word (ex-ante — do not skip).** Run `python deai.py describe <corpus>` for the venue's
   **writing brief**: its voice, the diction it reaches for and the texture it keeps near-zero, sentence and heading
   structure, and a DO/DON'T list at the word-and-construction level (cached to `<corpus>/.register.md`). Read the
   brief and draft *in* that register from the first sentence. This is the half that *prevents* the tells instead of
   removing them after; skipping it is exactly why a rewrite reverts to AI-default.

1. **Measure (the oracle).**
   `python deai.py mirror <draft> <corpus>` → the fold: A distributional fingerprints (function-word + POS-bigram Δ),
   B flagged shape/lexical tells, C syntactic/POS, D function-word over-use, E headings, and the **ACTIONABLE** list.
   Then `python deai.py describe <corpus>` → the venue's tone in words + measured dims (cached to `.register.md`):
   this is the *target register*. Optionally `python deai.py verdict <draft> <corpus> <ai_corpus>` for the two
   channels — **(1) AI-likelihood** (neural style margin) and **(2) venue-conformance** (Δ) — kept separate.

2. **Fix ONLY what the ACTIONABLE list names.** Typical moves:
   - `pron_rate` high → replace `it / its / they / their / them` with the explicit referent ("the floor", "the
     verifier", "the architecture"); trim authorial `we` where the sentence stands without it.
   - `metanarr_synt` / `meta_narration` high → **make the move instead of narrating it** ("X is the contribution" →
     just make the claim; "we name it rather than bury it" → just name it).
   - `intensifier` high → delete `precisely / exactly / itself / indeed / merely / simply` unless load-bearing.
   - antithesis (`comma_not` / `not_but`) high → state the positive directly; keep only load-bearing scope contrasts.
   - headings (E) → match the venue's case + style from `describe` (usually sentence-case, topic-naming).

3. **ANTI-OVER-CORRECTION (critical).** Target the corpus **band — floor AND ceiling**. Never push a tell *below* the
   corpus median: zero em-dashes, zero hedging, zero `we`, a triadic rate of 0 — each is itself a machine-laundered
   tell. Match the human *distribution*; do not flee it. (This is why the oracle reports the corpus spread, not a "0 is
   best" target.)

4. **PRESERVE substance — non-negotiable.** Never change a claim, citation key, number/statistic, table cell, or any
   section-heading meaning. Keep the venue-appropriate information **density** (do not simplify or shorten the
   argument). Keep the framework's **core terms consistent** — do not elegantly-vary them, which is its own tell.

5. **Re-measure and iterate** until the ACTIONABLE list is empty or only voice/structural items remain. The
   function-word / POS Δ may stay **OUTLIER** when the venue is dense and the paper has recurring core terms — that is
   *not* an AI tell, and chasing it by gutting substance or renaming the framework is the failure mode, not the goal.

## The one rule
If you edit until the tool says CLEAN, you have made the tool your evaluation and you will game it — the exact
pathology the underlying research is about. Use it to **find** what reads machine-made; fix those because they are
real; and when you suppress a tell, re-measure — tells **displace** (suppress em-dashes and they reappear as
comma-appositives). Honest target: inside the human band, substance intact, the two channels read human-and-conforming.
