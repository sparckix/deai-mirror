# deai — the de-AI mirror (one algorithm)

Everything is in **`deai.py`**. This is the only doc. It supersedes the scattered notes
(`anti_neuralese_guide.md`, the old `RESULTS.md`, the per-module headers).

## The thesis

AI generation is a **lossy one-way function**: it smooths a human sentence toward the centre of everything it has
read, discarding the particular word that made the sentence that person's. You cannot invert it word-by-word — you
cannot recover the human who never wrote the thing. But the AI-default leaves a **finite set of stance/structure
categories** that survive every surface variation. Those categories — not the phrases — are the invariants of the
function. Name them, detect them, strip them, and the text **re-projects toward the human distribution of a chosen
venue**. De-AI editing is not decoding; it is re-projection. The corpus *is* the standard: change the corpus,
change what "human" means.

## Three levels of abstraction (why it is neuro-symbolic)

| level | where | catches | blind to |
|---|---|---|---|
| **cue words / regex** | `feats_regex()` | known surface forms, cheap, high recall | paraphrase |
| **syntactic structure** | `feats_syntax()` (spaCy dependency rules) | a construction regardless of words | semantics |
| **category (meta-label)** | `TAXONOMY` + `judge()` (LLM) | an *instance* of a category with no surface signature | (nothing — but noisy/expensive) |

The symbolic part is the **taxonomy of categories**; the neural part is an LLM detecting category instances at that
abstraction level *without us enumerating phrasings*. Deterministic where a category has a surface signature; LLM
where it is only semantic (4 categories are `LLM-only`: AABB frame repetition, elegant variation, frictionless
balance, false range).

## The two registers (the key 2026-06-23 finding — read this before trusting any verdict)

There are **two registers of AI text, and a naïve generation benchmark only labels the first.**

- **Register 1 — naïve one-shot generation.** Tells are **lexical / density**: vague-evaluative adjectives,
  intensifiers, triads, generic-AI vocabulary, nominalization, transition openers. These separate naïve AI from
  human at ~0.84 AUC (the multi-model benchmark). The deterministic regex + spaCy layers catch them well.
- **Register 2 — deliberative, heavily-revised argument prose** (what a careful AI *paper draft* is). Tells are
  **discourse-stance**: meta-narration, lockstep symmetric hedging, AABB frames, frictionless balance. Naïve AI
  essays score ~0 on these; real papers a little; an over-worked draft a lot (the v4 cognitive-firm draft:
  `metanarr_synt` 4.6% of sentences vs corpus 0.0). **A naïve AI-vs-human benchmark cannot validate these — they
  need a Register-2 labelled set (revised-AI academic prose vs human).** This is exactly why meta-narration slipped
  past the distributional gate yet a human eye caught it. **Meta-narration is a revision artifact, not a generation
  default.** Treat a CLEAR distributional verdict as covering Register 1 only.

**Register-2 validation** (2026-06-23, `generate.py --r2` deliberative prompt — "argue but be rigorous and
self-aware about limits" — 20 essays vs the human corpus). A deliberative prompt sharply shifts the tell profile:
**intensifier AUC 0.96, abstract-subject openers 0.91, antithesis 0.68** all *jump* (the first two were weak or
*inverted* on naïve AI). So Register-2 prose is in fact *more* detectable, just via different tells (stance /
intensifier / abstract-subject, not vague adjectives) — and the multivariate composite covers both registers. **But
meta-narration proper only moves weakly under a deliberative prompt** (`metanarr_synt` 0.24→0.57; the regex idioms
stay ~0): it is specifically an *iterative-revision* artifact, not a deliberative-prompt one, and fully manifests
only after many self-critique passes (the v4 draft's 4.6). A deliberative prompt validates the *stance register*;
meta-narration's strongest evidence remains the heavily-revised draft itself.

## Meta-narration (the deepest tell), done three ways

Announcing the argument's own move / status / honesty instead of making it ("the contribution is", "we name it
rather than bury it", "whose status is modest", "the model organizes the argument"). Calibration finding: **broad
metadiscourse is normal** — real papers narrate themselves in 8–20% of sentences ("we argue", "the paper shows").
The AI tell is the **stance/status flavor only**, so `feats_syntax`'s `metanarr_synt` targets just that (copula +
status-adjective; self-mention/discourse-noun + a *stance* verb like concede/decline/name; negated claim of
strength), not all metadiscourse. Covered by: `meta_narration` (regex), `metanarr_synt` (syntactic), and the
`META_NARRATION`/judge category (LLM). Fix = **make the move instead of announcing it.**

## The algorithm (`deai.py mirror`)

Four deterministic layers, folded into one verdict, plus the optional LLM judge:

- **A. Distributional fingerprints (the trustworthy headline).** Function-word **Burrows's Delta** (lexical) +
  **POS-bigram Delta** (syntactic), each compared to the corpus's **own leave-one-out distances** → *OUTLIER* if the
  draft sits past a robust cutoff (median + 3·MAD) of how far real corpus papers fall from the centroid, *MIRRORS*
  otherwise. The robust cutoff replaces a brittle `max()` that a single stylistically-odd corpus paper could push out
  of reach; on the benchmark it flags ~100% of AI essays at ~8% human false-positive, where `max()` flagged ~5%.
- **B. Shape / lexical tells** (`feats_regex`, ~30 features) flagged vs corpus p80.
- **C. Syntactic / POS** (`feats_syntax`, spaCy) flagged vs corpus p15/p85.
- **D. Function-word over-use** (z-scores vs corpus).
- **LLM judge** (`deai.py judge`): deepseek + grok + kimi, prompted **by the taxonomy** for category instances —
  named tells only, never the saturated/noisy probability number.

## Empirical grounding (`deai.py infer` — taxonomy from data, not priors)

We have a labelled split (human venue corpus vs AI-default essays), so the taxonomy is **measured, not asserted**.
On a human venue corpus (n=14) vs 3-model AI-default (n=26):

- **Validated AI-higher tells (AUC):** adjective rate **0.78** (strongest), intensifier 0.74, triadic 0.67, noun
  rate 0.65, generic-AI vocab 0.64, transition openers 0.62, not-but antithesis 0.61, nominalization 0.60.
- **Inverted (human-higher → NOT generic-AI tells):** clefts, `which-is` tails, `comma,not`, hollow hedges, and
  `metanarr_synt` — because naïve AI doesn't meta-narrate (the Register-2 point above).
- **Discovery (open-ended excess words):** topic-confounded, but the genuine style residue is `whose` (15×),
  `merely` (6×), and a vague-evaluative-adjective cluster (profound/meaningful/structural/primary) — now the
  `vague_eval_adj` feature and category.

## Lexical over-use, the world-class way (`deai.py excess`)

`infer`'s open-ended discovery (above) finds excess words by a raw rate ratio: topic-confounded and noisy on a
single paper. `excess` is the principled upgrade, three research-backed pieces:
- **Monroe, Colaresi & Quinn (2008) "Fightin' Words"** — weighted log-odds-ratio with an informative Dirichlet
  prior, a *variance-adjusted* z for "which words does the draft over-use vs the corpus." A word seen 3× by chance
  is shrunk by the prior, where a raw ratio over-weights it. This is why `excess` correctly clears `real` (6×) that
  the naive ratio flags.
- **Topic control by MATCHING (Wegmann 2022; Kobak 2024)** — the AI reference is generated *on the same topics as
  the human corpus papers*, so the words AI over-uses vs human are pure **style**, not topic. That is what separates
  generic-AI slop (`harden`) from a paper's legitimate topic vocabulary.
- **wordfreq** splits common-word slop (`harden`, zipf 3.5) from rare domain jargon (`multidivisional`, zipf 0).

Two channels. **generic_ai_slop** (draft over-uses AND AI over-uses on matched topics AND common word) is a
*candidate* de-slop list, not an auto-cut: the LLM triages gratuitous vs load-bearing use. This is the neuro-symbolic
split at the lexical layer, the count names the word the LLM read past, the LLM judges the use. **draft_overrepetition**
(the draft over-uses even vs AI on its *own* topic) flags terms the author leans on, vary them unless they are core
framework terms, since elegantly varying a core term is itself a tell.

**Honest limit.** When a paper's *subject* vocabulary IS the AI-abstraction register (architecture, structural,
institutional, protocol), the slop channel cannot separate the two, both are over-used by AI on any topic. The
over-repetition channel stays reliable; treat the slop channel as candidates for the LLM, not a verdict. `excess`
needs `wordfreq` and generates a small cached AI reference (deepseek/kimi/grok) on first run.

## Heading tells, including the one the surface rules miss

`heading_feats()` differentials the draft's headings against the corpus's: Title-Case (venues are sentence-case),
`toward` cliché, comma-splice, `not`-antithesis, colon/quoted/compound. Those are surface rules. The tell they
STRUCTURALLY miss is the **vague abstract nominalization** ("The governance of the residual", "Toward an audit
institution") which is sentence-case, comma-free, puffery-word-free, yet a pure abstract-noun pile. `abstract_headings()`
catches it with **nominalization density** (Biber's nominal style; Brysbaert concreteness is the gold lexicon but needs
40k words, so the suffix proxy is corpus-free): the fraction of a heading's content words that are nominalizations
(-tion/-ment/-ance/-ity/-al...). Gerunds (-ing) do NOT count, because "Governing X" is the concrete fix for "The
governance of X". Like the lexical slop channel it returns a **candidate** list: a defined-term heading ("Attestation
and liability") or the framework's own name scores high too, so the author/LLM triages vague vs technical; single-word
and standard headings (Introduction, Conclusion, Independence) are dropped.

## Usage

```bash
python deai.py mirror   draft.md corpus_dir/             # full deterministic fold + verdict (A fingerprints, B tells, C syntax, D lexicon, E headings)
python deai.py check    draft.md [corpus_dir/]           # regex-only, fast, no spaCy (Layer-1 or -2)
python deai.py infer    human_dir/ ai_dir/               # EMPIRICALLY rank tells AI-vs-human (AUC) + open-ended discovery
python deai.py taxonomy [--md]                            # the meta-label taxonomy / coverage audit
python deai.py profile  corpus_dir/ [corpus2/]           # register card (Biber/Hyland dims) — a corpus's "tone" one level up
python deai.py describe corpus_dir/                       # LLM register meta-attributes, grounded on the dims; caches .register.md
python deai.py style    draft.md human_corpus/ ai_corpus/ # NEURAL style verdict (chunk-agg + few-shot 2-prototype)
python deai.py probe    human_corpus/ ai_corpus/         # EXPLAIN the style embedding (probe R² per dim + AI-direction correlations)
python deai.py judge    draft.md "Venue Name"            # LLM stance + meta-narration named-tell panel
python deai.py excess   draft.md corpus_dir/ "topic"     # topic-controlled lexical over-use (Monroe log-odds + matched-topic AI style ref + wordfreq)
```
Bring your own venue: a folder of `*.md` papers is the corpus. `check`/`taxonomy` run on plain `python3`;
`mirror`/`infer`/`profile` need the venv (spaCy + nltk); `style`/`probe` also need `sentence-transformers`;
`excess` needs `wordfreq`. `describe`/`judge`/`excess` read API keys portably (`$DEAI_ENV`, then `./.env`, `~/.env`,
or the repo's `.env`, no hardcoded paths). The LLM judges default to deepseek + grok + kimi; pick the set with
`DEAI_JUDGES=deepseek,grok` and override any model or endpoint via `DEEPSEEK_MODEL` / `XAI_MODEL` / `KIMI_MODEL`
(and the matching `*_URL`) — no model ID is hardcoded beyond the defaults. Swap the neural style model with
`DEAI_STYLE_MODEL=...` (default Wegmann; `gabrielloiseau/LUAR-MUD-sentence-transformers` for the exact-SOTA authorship rep).

## The neural style layer + explainability (SOTA, and how it is read out)

The neural layer is an **added verdict input**, not a replacement — it answers "how far from the human style
manifold," the deterministic layers answer "which tell to fix." Two implementation lessons, both learned the hard way:

- **The readout matters more than the model.** Naive sentence-mean + cosine-to-global-centroid gave AUC 0.75 and
  looked like "neural fails on academic prose." Done the SOTA way (Soto et al. ICLR-2024: **chunk-aggregation +
  few-shot 2-prototype**, comparing to *human and machine* style prototypes), the **same** Wegmann embedding hits
  **AUC 0.93–0.96** — above the deterministic 0.84. The model (Wegmann/LUAR/UAR) is pretrained on human authorship;
  inference-only, **no GPU, no training** — model-drift is handled by refreshing the few-shot *exemplars* (the AI
  reference set), not the embedding (that is *why* the SOTA is few-shot — robust to unseen LLMs).
- **The opaque vector is interpretable** (`probe`). Probing classifiers (Alain & Bengio 2017; Hewitt & Manning 2019)
  + concept directions (TCAV, Kim et al. 2018) decode the AI-axis into named tells: in our run, **AI = +stance/eval
  (r 0.69), −burstiness (more uniform), −involvement, +density** — which *triangulates with the deterministic top
  tell* (vague-adjectives/stance, AUC 0.86). Two independent methods agreeing on what "AI" means is the validation.

## Corpus meta-attributes — "describe language one level up", three routes

Each corpus has a **meta-definition** (its register, computed once at ingestion and cached to `.register.md`):
1. **Known algorithms** (`profile`) — Biber multidimensional analysis + Hyland metadiscourse: count features →
   interpretable dimensions (info-density, involvement, metadiscourse, stance, rhythm, heading style). Deterministic.
2. **The pretrained model** (`style_vec`) — a holistic but **opaque** 512-d style vector; interpret it with `probe`.
3. **An LLM** (`describe`) — human-readable meta-*labels*, **grounded on the measured dimensions** so it describes
   what is measured, not what it imagines. The richest "describe language using language", neuro-symbolic.

## Honest status / how it is hardened

- **Benchmark.** Combined deterministic signal separates AI-default from human at **AUC 0.84** on three generators
  (deepseek + grok + kimi). The earlier **0.96 was a single-model (deepseek) artifact** — hardening with model
  diversity dropped it, and the durable cross-model discriminators are function-word Δ (0.78) + the flag-count
  composite (0.72). No single tell clears ~0.78 — the power is the **multivariate composite**, which is why
  single-pattern "humanizers" are weak.
- **The one rule (Goodhart your own gate).** If you edit until this says *clean*, you have made the tool your
  evaluation and you will game it — the exact pathology of the research this grew from. Use it to *find* what reads
  machine-made; fix those because they are real; and **never trust CLEAR** — a clean distributional verdict only
  covers Register 1. When you suppress a tell, re-measure: tells **displace** (suppress em-dashes → comma-appositives).
- **Open frontier.** (1) a **Register-2 labelled set** (revised-AI academic prose vs human) to validate the
  discourse-stance categories empirically — the current benchmark can't; (2) **leave-one-out flag thresholds**
  (the distributional Δ verdict is LOO with a robust median+3·MAD cutoff, but the regex flag thresholds are still
  in-sample, so the composite AUC is optimistic);
  (3) a **pre-2022 human baseline** (the recent corpus may be AI-contaminated, compressing separation).

## Research context (what exists, and the gap)

Pieces exist; no single empirically-validated category taxonomy does — hence this hybrid (priors validated by our
corpus). **Hyland's metadiscourse taxonomy** (hedges/boosters/self-mentions/frame-markers) is the citable framework
for the discourse-stance family. The **AI-text-detection feature literature** reports LLM text as lower-burstiness,
deeper/longer dependency structures, lower semantic diversity (features, not a stance taxonomy). The Wikipedia
**"Signs of AI writing"** community catalogue and **Kobak et al. "excess words"** (data-driven, lexical-only) are
the closest catalogues. Our contribution is the **neuro-symbolic recombination**: deterministic corpus-calibrated
stylometry wired to an LLM working at the category level, validated on a labelled benchmark.

## Corpus alignment, self-explaining (`register` / `styleprofile`)

`check`/`profile` tell you the numbers; `register` tells you what they MEAN, and adds a semantic layer the
deterministic features cannot reach. Two layers, with deliberately opposite label policies:

- **Layer 1 — deterministic, FIXED axes (comparable across venues).** Every surface/syntactic feature is glossed
  in plain language (what it measures + why it reads AI) and read against the venue's own distribution (median +
  p25–p75), with a one-line interpretation. Same axes for every corpus, so venues are comparable and the benefit
  compounds across drafts. No API; `register draft corpus/ --no-llm` runs this alone.
- **Layer 2 — semantic, EMERGENT per corpus (labels need NOT match across venues).** An LLM ingests the corpus and
  *induces* the 6–9 categories that characterize THIS venue's tone and style — stance, hedging, rhetorical
  structure, diction, evidence use, framing, whatever is actually distinctive — each with prose on what the venue
  **does** and **does not** do and an intensity score *where meaningful* (else `null`). The draft is then checked
  against the venue's OWN categories (alignment 0–100 + verdict + evidence). Cached to `<corpus>/.style_profile.{json,md}`,
  so the fingerprint is a durable, reusable property of the corpus.

Why the split: the research consensus (Biber MDA; CORE multilingual register, arXiv 2406.19892; StyleDistance,
2410.12757) is "fixed labels + per-venue thresholds" for *comparability* — that is Layer 1. But a fixed taxonomy
cannot say what a *particular* venue's voice is; for that you want emergent, corpus-specific labels — that is
Layer 2.

**Reliability (ensemble + agreement).** LLM style-labeling is volatile (Rating Roulette, EMNLP 2025; Nature 2025),
so Layer 2 scores the draft with a **3-judge ensemble** (deepseek + grok + kimi) and reports the **median** alignment
per category plus the **inter-judge Krippendorff alpha** (interval): >=0.8 reliable, 0.5-0.8 moderate, <0.5 directional.
A high per-category `spread` flags exactly which categories the judges disagree on. The induced profile stays single-model
(the emergent categories are stable); only the *scoring* is ensembled. A few "evidence" spans still come back paraphrased
rather than verbatim — the verdicts are the signal, not the quotes. (kimi needs thinking disabled + temp 0.6 or it 400s.)

```
python deai.py styleprofile corpus_dir/ [--refresh]      # induce + cache the venue's emergent tone/style categories
python deai.py register     draft.md corpus_dir/ [--no-llm]   # Layer 1 (glossed numbers) + Layer 2 (draft vs emergent categories)
```
