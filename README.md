# deai-mirror

### Is neuralese a lossy, one-way function? And if it is, are we doomed to read slop forever?

That is not a rhetorical flourish. It is the question this repository tries to answer with measurement instead of vibes.

---

## The problem

When a language model turns an idea into prose, it does not write the sentence you would have written. It writes the *likeliest* one: the sentence nearest the center of everything it has ever read. Run that at scale and the center fills up. The same balanced triads. The same "it is important to note." The same frictionless, faintly corporate calm where every paragraph lands cleanly and nothing is ever at risk. Call it neuralese. It isn't wrong. It's *average*, it's everywhere, and it is converging, because every model is averaging the same internet.

Here is the uncomfortable part: averaging is lossy. The thing that made a human sentence *that person's* sentence (the odd word, the buried clause, the choice that could have failed) is the very information the model discards to land in the middle. So the real question is whether you can run the map backwards. Once the human signal has been smoothed out, can you get it back?

Honestly? Not by inversion. You cannot reconstruct the particular person who never wrote the thing. But you can do the next best thing, and it turns out to be enough. Measure how far a text sits from how real people write *in a particular place*, then push it back toward them. De-AI editing is re-projection onto a human distribution, a different operation from decoding. That one distinction is the whole tool.

## Why the tools we already have don't help

Two industries grew up around the slop. Both share a flaw.

Detectors (GPTZero, Pangram, Binoculars) answer "is this AI?" with a probability. Handy for a gatekeeper, useless for a writer: a number won't tell you which sentence betrays you, and the strong ones collapse on lightly-edited text anyway.

Humanizers (the open-source `SKILL.md` skills, the "nine levers," the QuillBots) ask a language model to find and rewrite its own tells. Read that twice. They hand the smoothing function the job of detecting smoothing. And the score *is* the model's opinion, which we clocked three times on one unchanged paragraph: 95, 85, 95. Once, 15. A ruler that reads differently each time you look is not a ruler.

And the actual science (Burrows's Delta, function-word fingerprints, dependency parsing, half a century of stylometry that separates Shakespeare from Marlowe) sits in a lab, wired to nobody's draft.

## The idea: mirror the corpus, don't ask the oracle

`deai-mirror` is neuro-symbolic. It counts what a machine can count, deterministically, and asks a language model only the part that genuinely needs meaning. One command folds four layers:

| layer | what it sees | how |
|---|---|---|
| distributional fingerprint *(the verdict)* | function-word Burrows's Delta + POS-bigram Delta vs the corpus's own internal spread, giving *mirrors / outlier* | pure Python + spaCy |
| shape | triads, abstract-subject openers, clefts, comma-appositives, antithesis, that-density | regex |
| syntax | dependency distance, parse-tree depth, subordination, POS rates | spaCy |
| lexicon | per-word z-scores; harvested generic-AI vocabulary | counts |

The verdict is not a judgment call. It is a distance: how far your draft's fingerprint sits from where real papers in the venue actually land. Inside their spread, you mirror them. Outside it, you're an outlier, and the tool names the exact features that put you there, down to the single over-used word.

```bash
./setup.sh                                                        # venv + spaCy + nltk (once)
.venv/bin/python deai.py mirror draft.md corpora/emerson_essays   # full fold + verdict
.venv/bin/python deai.py infer corpora/emerson_essays benchmark/_generated_r2   # rank tells AI-vs-human (AUC)
```

Everything is in one file, `deai.py`: the deterministic commands (`mirror` / `check` / `infer`), the meta-label `taxonomy`,
the corpus register meta-attributes (`profile` / `describe`), the neural SOTA style verdict (`style`) and its
explainability (`probe`), and the LLM `judge`. The algorithm, the tell taxonomy, and the honest status are
documented in one doc, `deai.md`.

Bring your own venue: drop its papers into `corpora/<venue>/`. The corpus *is* the standard. Change the corpus, change what "human" means. (A small public-domain sample, Emerson's *Essays*, ships in `corpora/emerson_essays/` so the commands run out of the box.)

The deterministic commands (`mirror` / `check` / `infer`) need no API keys. The LLM layers (`judge` / `describe` / `style`) read keys from a `.env` (`DEEPSEEK_API_KEY`, `XAI_API_KEY`, `KIMI_API_KEY`); the model IDs, endpoints, and active judge set are all env-overridable (`DEAI_JUDGES`, `DEEPSEEK_MODEL`, …), so nothing about the models is hardcoded.


## The one rule

If you edit until this tool says *clean*, you have made the tool your evaluation, and you will game it, the way every optimizer games the metric it is scored against. (The research this grew out of is *about* exactly that failure.) Use it to find what reads machine-made. Fix those things because they are real. And when you flatten one tell, measure again: tells do not die, they move. Strip the em-dashes and they reappear as comma-appositives, every single time.

## Honest status

A sharp instrument, getting validated, and the validation just got more honest. On a first benchmark (14 human venue papers vs 10 deepseek essays) the combined deterministic signal separated AI from human at AUC ~0.96. Hardening it with three models (deepseek + grok + kimi, 26 AI-default essays) dropped that to AUC 0.84. The 0.96 was a single-model artifact, and grok/kimi prose sits closer to the human distribution. The durable cross-model discriminators are the function-word Burrows's Delta (0.78) and the flag-count composite (0.72); the combined signal (0.84) still beats every single tell (no individual one clears ~0.78), which is why single-pattern humanizers are weak and a multivariate composite wins. Diversity also exposed a couple of syntactic-shape tells over-fit to academic-AI: abstract-subject openers and what-clefts *invert* on grok/kimi, so the universal defaults stay hand-set tight while the pluggable corpus carries the venue-specific calibration. The regex-flag thresholds are still in-sample and the corpus is small, so the honest claim is that it separates AI-default prose and weakens under model diversity (0.84). It is not certified. The distributional verdict now uses a robust leave-one-out cutoff (median + 3·MAD); the old `max()` was brittle. It flags AI essays while held-out human papers still mirror, and a pre-AI human baseline remains the open frontier (`benchmark/RESULTS.md` has both runs). The stylometry is old and the humanizers are many; the one new thing here is wiring deterministic, corpus-calibrated measurement to the model.

So, are we doomed to read slop? Not if we stop asking the slop-machine to grade itself, and start measuring the distance to the people who write the way we wish we could.

---

MIT. No warranty, and a standing invitation to prove the benchmark wrong.
