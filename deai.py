"""deai.py — ONE file. The whole neuro-symbolic de-AI mirror.

Reverse a lossy one-way function: AI generation smooths a human sentence toward the centre, discarding the
particular words. You cannot invert it word-by-word, but the AI-default has a finite set of STANCE/STRUCTURE
categories (the taxonomy below) that survive every surface variation. Detect them — deterministically where they
have a surface signature, with an LLM judge where they are only semantic — then strip them to re-project toward
the human distribution. The corpus IS the standard: change the corpus, change what "human" means.

Three levels of abstraction, all in this file:
  cue words / regex        -> feats_regex()    cheap, high-recall on known forms, brittle to paraphrase
  syntactic structure      -> feats_syntax()   spaCy dependency rules; generalizes within a construction
  CATEGORY (the meta-label)-> TAXONOMY + judge() the LLM detects an instance without us enumerating its phrasings

CLI:
  python deai.py mirror   draft.md corpus_dir/ [corpus2/ ...]   # full deterministic fold + verdict
  python deai.py check    draft.md [corpus_dir/]                # regex-only (fast, no spaCy); Layer-1 or -2
  python deai.py infer    human_dir/ ai_dir/                    # EMPIRICALLY rank tells AI-vs-human (AUC) + discovery
  python deai.py taxonomy [--md]                                # print the meta-label taxonomy / coverage audit
  python deai.py profile  corpus_dir/ [corpus2/]               # register card (Biber/Hyland dims) of a corpus's "tone"
  python deai.py describe corpus_dir/                           # LLM register meta-attributes, grounded on the dims; caches .register.md
  python deai.py style    draft.md human_corpus/ ai_corpus/   # NEURAL style verdict (chunk-agg + few-shot 2-prototype; needs sentence-transformers)
  python deai.py probe    human_corpus/ ai_corpus/             # EXPLAIN the style embedding: linear-probe R² per dim + AI-direction correlations
  python deai.py verdict  draft.md human_corpus/ ai_corpus/    # FUSE all signals the world-class way: stacked ensemble -> one calibrated P(AI)
  python deai.py judge    draft.md "Venue Name"                 # LLM stance + meta-narration named-tell panel
  python deai.py excess   draft.md corpus_dir/ "topic"          # topic-controlled lexical over-use (Monroe + wordfreq)
  python deai.py styleprofile corpus_dir/ [--refresh]           # EMERGENT per-corpus tone/style categories (does/does-not + score); caches .style_profile.{json,md}
  python deai.py baseline corpus_dir/ [--draft X] [--only m,m]  # HUMAN PROB_AI baseline for a venue (caches .prob_ai_baseline.jsonl); --draft places a draft against the band
  python deai.py register draft.md corpus_dir/ [--no-llm]       # SELF-EXPLAINING alignment: surface & syntactic axes (glossed) + semantic register (draft-vs-emergent-categories)
  python deai.py excess   draft.md corpus_dir/ "topic"          # topic-controlled lexical over-use (Monroe log-odds + AI-matched style ref + wordfreq)
Run with the venv python (spaCy/nltk) for mirror/infer/judge; check works on plain python3.
"""
import re, os, sys, json, time, pathlib, statistics as st, collections, urllib.request

# ════════════════════════════════ shared ════════════════════════════════
def _env_paths():
    """Portable .env locations (no hardcoded user paths): $DEAI_ENV, then ./.env, ~/.env, and this repo's .env."""
    return [p for p in (os.environ.get("DEAI_ENV"), str(pathlib.Path.cwd() / ".env"),
            str(pathlib.Path.home() / ".env"), str(pathlib.Path(__file__).resolve().parent / ".env")) if p]

_ENVCACHE = None
def _load_env():
    """Merge .env files (first match wins) then process env (os.environ overrides). Cached."""
    global _ENVCACHE
    if _ENVCACHE is None:
        e = {}
        for ep in _env_paths():
            if ep and pathlib.Path(ep).exists():
                for line in pathlib.Path(ep).read_text(errors="ignore").splitlines():
                    m = re.match(r'^([A-Z0-9_]+)=(.*)$', line.strip())
                    if m: e.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
        e.update({k: v for k, v in os.environ.items() if k not in e or k.endswith(("_MODEL", "_URL")) or k == "DEAI_JUDGES"})
        _ENVCACHE = e
    return _ENVCACHE

# Provider registry: endpoint, API-key env var, default model, per-provider payload extras.
# Model IDs and endpoints are overridable via env (<PROVIDER>_MODEL / <PROVIDER>_URL); the active
# judge set is overridable via DEAI_JUDGES (comma-separated). Nothing about the models is hardcoded
# beyond these defaults.
_PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY", "deepseek-chat", {}),
    "grok":     ("https://api.x.ai/v1/chat/completions",      "XAI_API_KEY",      "grok-4.3",     {}),
    "kimi":     ("https://api.moonshot.ai/v1/chat/completions","KIMI_API_KEY",    "kimi-k2.6",    {"thinking": {"type": "disabled"}}),
}
def _roster(names=None):
    """Active judge roster as (name, url, keyvar, model, extra) tuples, resolved from env."""
    env = _load_env()
    names = names or [s.strip() for s in env.get("DEAI_JUDGES", "deepseek,grok,kimi").split(",") if s.strip()]
    out = []
    for n in names:
        if n not in _PROVIDERS: continue
        url, keyvar, mdef, extra = _PROVIDERS[n]
        out.append((n, env.get(n.upper()+"_URL", url), keyvar, env.get(n.upper()+"_MODEL", mdef), dict(extra)))
    return out

def prose(t):
    t = re.split(r"(?im)^#{1,3}\s*(references|bibliography|appendix)\b", t)[0]
    t = re.sub(r"<!--.*?-->", "", t, flags=re.DOTALL)
    return " ".join(ln for ln in t.splitlines()
                    if ln.strip() and ln.strip()[0] not in "#-*|>" and not ln.strip().startswith("[^"))

def _read_corpus(dirs, minw=400):
    out = []
    for d in dirs:
        for f in sorted(pathlib.Path(d).glob("*.md")):
            if f.name.startswith(".") or f.name.lower() in ("readme.md", "index.md"): continue
            ct = f.read_text(errors="ignore")
            if len(re.findall(r"\b\w+\b", prose(ct))) > minw: out.append((f.stem, ct))
    return out

# ════════════════════════════ the taxonomy (meta-labels) ════════════════════════════
# family ∈ {lexical, punctuation, rhythm, syntactic-shape, discourse-stance, semantic}. `det` = the deterministic
# feature(s) that partially cover the category (so we can audit which categories are LLM-ONLY). `probe` = judge text.
TAXONOMY = [
    dict(id="meta_narration", family="discourse-stance", label="Meta-narration (body + heading level)",
         definition="Sentences — OR headings — that narrate the argument's own move, structure, epistemic status, or "
                    "honesty instead of making the move. Sub-types: (a) BODY ('the contribution is', 'we name it "
                    "rather than bury it'); (b) HEADING announcing the section's CONTENTS ('Two predictions and a "
                    "minimal model') instead of naming its topic ('Regulatory capture of AI safety').",
         why_ai="The model describes its own reasoning and writes headings as a table of contents of its own moves; a "
                 "scholar just makes the point and names the subject.",
         det=["meta_narration", "metanarr_synt", "compound", "mean_words"],
         probe="(a) body spans that describe the argument/claim/model rather than advance it ('the contribution is', "
               "'whose status is modest', 'the model organizes the argument'); (b) headings that list what the section "
               "contains/does rather than naming its topic"),
    dict(id="hedging_miscalibration", family="discourse-stance", label="Hedging miscalibration (lockstep + over-honesty)",
         definition="Stance miscalibration toward over-qualification, two sub-types: (a) RHYTHMIC lockstep — every "
                    "claim concedes-then-qualifies and every objection is answered the instant it is raised; (b) "
                    "SEMANTIC over-honesty — repeated scope-disclaimers / self-deprecation that talk the result down "
                    "(the opposite-direction tell to puffery).",
         why_ai="Averaging the corpus yields a frictionless, maximally-safe register that re-hedges and re-states its "
                 "own limits; a confident scholar concedes ONCE, plainly, then makes the claim.",
         det=["hedge_modal", "disclaimer", "hedge_phrase"],
         probe="(a) runs where each sentence concedes-then-qualifies in lockstep; (b) repeated scope-restatement / "
               "self-deprecation ('an existence proof, not a validation' more than once, 'we make no claim', 'we are "
               "careful about', a conclusion ending on its own modesty) — a SINGLE plain concession is rigor, NOT a tell"),
    dict(id="antithesis", family="rhythm", label="Antithetical parallelism (not-X-but-Y + AABB)",
         definition="Balanced antithetical constructions, two sub-types: (a) single-sentence 'not X but Y' / 'rather "
                    "than' aphoristic kicker, especially a closing punch-line; (b) AABB — adjacent sentences that state "
                    "a frame then negate/mirror it in parallel syntax ('The parallel is structural. The parallel is "
                    "also partial.').",
         why_ai="The model loves the balanced aphorism and the even-handed mirror; human prose uses them sparingly.",
         det=["comma_not", "rather_than", "not_but"],
         probe="(a) aphoristic 'not X but Y' kickers, especially closing sentences; (b) adjacent sentences repeating a "
               "frame then mirroring/negating it in parallel structure"),
    dict(id="over_triadic", family="syntactic-shape", label="Over-triadic structure",
         definition="The rule of three everywhere: three-item lists and three parallel clauses as a default rhythm.",
         why_ai="Triads are the model's comfortable cadence.", det=["triadic"],
         probe="three-part parallel lists/clauses used as a recurring rhythm rather than because the content is triple"),
    dict(id="abstract_subject", family="syntactic-shape", label="Abstract-subject / nominalization",
         definition="Openers like 'The obstruction is…', dense nominal subjects standing in for agents.",
         why_ai="Nominalized, agentless density is the model's formal default.",
         det=["abstract_subj", "nominal_rate", "para_open_The"],
         probe="abstract nominal subjects ('The tension is…', 'The obstruction lies…') as a default opener"),
    dict(id="importance_inflation", family="lexical", label="Importance inflation (adjective + phrase + trailing clause)",
         definition="Asserting unearned importance instead of demonstrating it, three sub-types: (a) single vague "
                    "evaluative ADJECTIVES (profound/meaningful/crucial/structural/primary); (b) PHRASAL assertion "
                    "('plays a crucial role', 'is a testament to', 'underscores the importance of'); (c) trailing "
                    "SIGNIFICANCE clause or historical-weight framing ('…, reflecting the growing importance of…', "
                    "'marked a turning point', 'cemented its legacy') — Wikipedia WP:AIPEACOCK / SUPERFICIAL / AILEGACY.",
         why_ai="Sub-type (a) is the STRONGEST empirical discriminator in our labelled set (adj-rate AUC 0.78): the "
                 "model reaches for an importance word, phrase, or significance flourish where a human gives a specific "
                 "one or none.",
         det=["vague_eval_adj", "adj_rate", "puffery"],
         probe="(a) contentless evaluative adjectives as default emphasis; (b) phrases asserting importance rather than "
               "showing it; (c) trailing '-ing' significance clauses and 'marked a turning point' historical inflation "
               "that assert weight the evidence does not earn"),
    dict(id="appositive_restate", family="syntactic-shape", label="Comma-appositive restatement",
         definition="A noun immediately restated by a comma-appositive — the em-dash's quiet replacement.",
         why_ai="When em-dashes are suppressed the restating impulse displaces to comma-appositives.",
         det=["appositive"], probe="comma-appositive restatements that re-explain a noun just named"),
    dict(id="cleft", family="syntactic-shape", label="It-/what-cleft fronting",
         definition="'What matters is…', 'It is X that…' used to front emphasis.",
         why_ai="A favoured emphasis device, over-used as a stance marker.",
         det=["what_cleft"], probe="'what X is Y' and 'it is X that' clefts used for emphasis"),
    dict(id="elegant_variation", family="lexical", label="Elegant variation",
         definition="Renaming one concept several ways to avoid repetition, which scatters a key term and confuses the reader.",
         why_ai="Anti-human: scholars keep key terms stable; the model varies the label to seem fluent.", det=["LLM-only"],
         probe="a single concept/referent given several different labels where one stable term belongs"),
    dict(id="template_repetition", family="syntactic-shape", label="Template repetition",
         definition="Reusing one sentence/phrase frame mechanically across adjacent items, list rows, or paragraph openers.",
         why_ai="The model reaches for a fill-in-the-blank frame; human prose varies structure.", det=["LLM-only"],
         probe="an identical sentence template reused across list items or adjacent paragraphs, incl. repeated 'This X…' / 'These Y…' paragraph openers"),
    dict(id="transition_signpost", family="lexical", label="Transition over-signposting",
         definition="'Furthermore', 'Moreover', 'Additionally', 'Importantly' as sentence openers.",
         why_ai="Connective scaffolding denser than real prose.",
         det=["transition_open"], probe="explicit transition-word openers used to scaffold flow"),
    dict(id="generic_ai_vocab", family="lexical", label="Generic-AI vocabulary",
         definition="delve, leverage, robust, comprehensive, tapestry, nuanced, intricate, multifaceted, realm.",
         why_ai="The lexical centre of mass of internet-scale training.",
         det=["ai_vocab"], probe="the generic-AI lexicon (delve/leverage/robust/tapestry/nuanced/realm…)"),
    dict(id="hollow_hedge", family="lexical", label="Hollow hedge phrase",
         definition="'it is important to note', 'it is worth mentioning', 'needless to say'.",
         why_ai="Filler that announces salience without adding content.",
         det=["hedge_phrase"], probe="content-free hedges that announce salience ('it is important to note…')"),
    dict(id="frictionless_balance", family="semantic", label="Frictionless balance / no risk",
         definition="Every paragraph lands cleanly; nothing is genuinely at stake; an even, faintly corporate calm.",
         why_ai="The defining felt quality of averaged prose — the hardest tell to quantify, the one the judge is most needed for.",
         det=["LLM-only"],
         probe="passages where (a) every claim is hedged or qualified, (b) no objection stands unanswered beyond a "
               "sentence, (c) the close restates rather than extends, and (d) no specific checkable prediction or "
               "commitment appears — require at least THREE of the four and quote them; do NOT flag on 'feel' alone"),
    dict(id="false_range", family="semantic", label="False range / false precision",
         definition="Empty 'ranges from X to Y' spans and fake specificity that adds no information.",
         why_ai="A fluency reflex that simulates rigor.", det=["LLM-only"],
         probe="'ranges from … to …' or pseudo-precise enumerations that carry no real information"),
    dict(id="uniform_rhythm", family="rhythm", label="Uniform sentence rhythm",
         definition="Low burstiness: sentence lengths cluster, the cadence is metronomic.",
         why_ai="Human prose varies sentence length sharply; the model regresses to a uniform mean.",
         det=["burstiness_CV", "mean_sent"], probe="long stretches of similar-length sentences with no short punch or long breath"),
    dict(id="titlecase_headings", family="headings", label="Title-Case headings",
         definition="Section headings (and the title) in Title Case where the venue uses sentence case.",
         why_ai="The model's default heading style; real journals — and their corpora — are overwhelmingly sentence-case.",
         det=["title_case"], probe="headings/title capitalised Like This rather than Like this"),
    dict(id="nominalization_headings", family="headings", label="Vague abstract-nominalization headings",
         definition="Headings built from abstract nominalizations with no concrete anchor ('The governance of the "
                    "residual', 'Toward an audit institution') — sentence-case and comma-free, so the surface heading "
                    "rules (Title-Case, comma, 'not Y') structurally miss them.",
         why_ai="The model reaches for an abstract noun-pile where a scholar names a concrete subject or asks a concrete question.",
         det=["nominalization"],
         probe="headings that are pure abstract nominalizations ('The X of the Y', 'Toward an X') rather than concrete or "
               "verbal ('A checker that tested the wrong property', 'Who answers for the judgment a rule cannot check')"),
    dict(id="clever_headings", family="headings", label="Clever / quoted / self-referential headings",
         definition="Scare-quoted, aphoristic, 'X, not Y', or self-congratulatory headings.",
         why_ai="The model reaches for the clever heading; the venue is plain.",
         det=["quoted", "antithesis", "selfcong"], probe="scare-quotes, aphorisms, antithesis, or self-praise in a heading"),
    dict(id="coined_jargon", family="lexical", label="Coined / unsignaled neologisms",
         definition="Capitalized multi-word terms the author mints and emphasizes or scare-quotes, then uses as if "
                    "established (e.g. 'Blame Shield', 'Float Masking', 'Audited Multidivisional Architecture').",
         why_ai="The model freely coins compound jargon to sound rigorous; a human scholar reuses an established "
                 "term, or signals plainly when coining one.",
         det=["coined_jargon"],
         probe="the cue lists emphasized/quoted Title-Case multi-word candidates; flag one when (a) it first appears "
               "WITHOUT a coinage signal ('we call/term/introduce/define'), (b) it is not an established term in the "
               "venue's literature, and (c) it then recurs as if established — judge those observable criteria, not 'tone'"),
    dict(id="template_placeholder", family="structure", label="Placeholder / template residue",
         definition="Unfilled template markers left in the text: xxx, [N], [cite], TBD, <placeholder>, lorem ipsum.",
         why_ai="The model leaves scaffolding it never filled; any instance is a hygiene failure a careful author removes.",
         det=["placeholder"], probe="any unfilled placeholder or template residue (xxx, [N], [cite], TBD, <...>)"),
    dict(id="false_scale", family="semantic", label="False-scale / over-enumeration",
         definition="Claims of excessive structure that inflate apparent rigor: '30 tables', 'dozens of experimental "
                    "conditions', 'numerous configurations'.",
         why_ai="The model pads with apparent scale; the count is round, large, or vague rather than load-bearing.",
         det=["false_scale"],
         probe="over-claimed enumeration of tables/figures/conditions/experiments that inflates apparent scale rather "
               "than carrying content"),
    dict(id="listification", family="structure", label="Listification / inline-header lists",
         definition="Argued prose replaced by vertical lists — especially the inline-bold-header list item "
                    "('- **Term**: explanation', '1. **Step**: ...') — or a rigid numbered procedure imposed on a "
                    "task that is really a matter of judgment.",
         why_ai="A documented chatbot default (Wikipedia WP:AILIST, 'inline-header vertical lists'): the model sorts "
                 "everything into evenly-sized chunks, so the page reads like a how-to / listicle rather than a "
                 "developed argument. A scholar develops the idea in prose and reserves lists for genuinely parallel items.",
         det=["inline_bold_list"],
         probe="prose broken into vertical lists / inline-bold-header items where a paragraph belongs, or a numbered "
               "step-procedure imposed on a judgment task; do NOT flag legitimate reference lists, data tables, or "
               "genuinely enumerable parallel items"),
    dict(id="bold_overuse", family="structure", label="Boldface overuse",
         definition="Mechanical **boldface** emphasis scattered through running prose to mark key terms, beyond a "
                    "sparing, meaningful use.",
         why_ai="Wikipedia WP:AIBOLD: the model bolds phrases throughout to look organized and emphatic; human "
                 "academic prose almost never bolds inside running text.",
         det=["bold_run"],
         probe="**bold** emphasis sprinkled through running prose to mark terms (not the rare first-use of a defined "
               "term); the page looks emphatic rather than argued"),
    dict(id="emdash_punct", family="punctuation", label="Em-dash / punctuation overuse",
         definition="Em-dashes (—) used as an all-purpose connector several times a page, plus curly quotes/apostrophes "
                    "pasted from a chat model into an otherwise plaintext document.",
         why_ai="Wikipedia WP:AIDASH/AICURLY: the em-dash is the single most-cited surface tell — the model reaches "
                 "for it as a default connective where a comma, colon, or full stop would serve.",
         det=["emdash"],
         probe="em-dashes used as a default connector multiple times per page, and curly “ ” ‘ ’ in a plaintext doc; "
               "judge overuse against the author's own baseline, not absolute count"),
    dict(id="weasel_attribution", family="discourse-stance", label="Weasel-word attribution",
         definition="Vague, sourceless attributions standing in for a citation: 'some critics argue', 'many experts "
                    "believe', 'studies suggest', 'it is widely regarded as'.",
         why_ai="Wikipedia WP:AIWEASEL: the model attributes opinions to an unnamed crowd to sound balanced and "
                 "authoritative without committing to a source a reader could check.",
         det=["LLM-only"],
         probe="unsourced crowd-attribution ('some argue', 'many experts believe', 'studies suggest', 'is widely "
               "regarded as', 'it is often said') used where a specific citation or named actor belongs"),
    dict(id="this_fronting", family="syntactic-shape", label="Demonstrative 'This/These' fronting",
         definition="Sentences repeatedly opened with a bare 'This/These/It' + verb ('This suggests…', 'This "
                    "highlights…', 'This underscores…'), often with an under-specified referent.",
         why_ai="AI-detection work (Liang et al. 2024) finds 'this'-fronted sentences a strong discriminator; the "
                 "model chains forward with weak anaphora where a human re-specifies the referent.",
         det=["LLM-only"],
         probe="bare 'This/These/It' + verb sentence-openers used more than once or twice per paragraph, especially "
               "where the referent is distant or vague"),
    dict(id="reformulation_overuse", family="discourse-stance", label="Reformulation-marker overuse",
         definition="Restating the same point with 'in other words', 'that is', 'i.e.', 'put differently' instead of "
                    "trusting the reader.",
         why_ai="The model defaults to restating for clarity; a human writer trusts the reader and moves on.",
         det=["reformulation"],
         probe="the same point restated via 'in other words / that is / i.e. / put differently' markers"),
    dict(id="explicit_crossref", family="lexical", label="Explicit cross-reference over-signaling",
         definition="Over-signalled internal navigation: 'as discussed above', 'as will be shown below', 'returning to "
                    "the earlier point'.",
         why_ai="Editor lore (WP:AICROSSREF): the model over-signals its own structure; scholars cross-reference sparingly.",
         det=["explicit_crossref"],
         probe="repeated 'as discussed above / as shown below / returning to' internal-navigation signals"),
    dict(id="conclusion_summary", family="discourse-stance", label="Conclusion opened as structure-summary",
         definition="A conclusion or section close that opens 'In this paper we have…' and restates the structure "
                    "rather than stating what was found or argued.",
         why_ai="Wikipedia WP:AICONCLUSION: the model's default close summarizes its own moves; a human states the "
                 "finding. Distinct from meta-narration in being position-specific and summative.",
         det=["LLM-only"],
         probe="a conclusion/close that recaps 'in this paper we have X, Y, Z' instead of stating the result or its consequence"),
]
FAMILIES = ["lexical", "punctuation", "rhythm", "syntactic-shape", "discourse-stance", "semantic", "headings", "structure"]
# EMPIRICAL GROUNDING (deai.py infer, 2026-06-23, human venue corpus n=14 vs 3-model AI-default n=26):
#   strong AI-higher: adj_rate 0.78, intensifier 0.74, triadic 0.67, noun_rate 0.65, ai_vocab 0.64,
#       transition_open 0.62, not_but 0.61, nominal_rate 0.60, para_open_The 0.59.
#   INVERTED (human-higher, NOT a generic-AI tell here): first_person, what_cleft, which_is_tail, metanarr_synt(!),
#       comma_not, abstract_subj, hedge_phrase.
# TWO REGISTERS, and the naive benchmark only labels Register 1:
#   Register 1 (one-shot generation): LEXICAL/DENSITY tells (vague adjectives, intensifiers, triads, vocab) — these
#       separate naive AI from human (~0.84 combined).
#   Register 2 (deliberative / heavily-revised argument prose, e.g. a careful AI paper draft): DISCOURSE-STANCE tells
#       (meta-narration, lockstep hedging, AABB). Naive AI ~0, real papers a little, an over-worked draft a lot. So
#       the naive benchmark CANNOT validate the discourse-stance categories — they need a Register-2 labelled set.
REGISTER = {"naive-generation": ["importance_inflation", "generic_ai_vocab", "over_triadic", "abstract_subject",
                                  "transition_signpost", "antithesis", "uniform_rhythm", "hollow_hedge", "this_fronting",
                                  "listification", "bold_overuse", "emdash_punct", "weasel_attribution",
                                  "reformulation_overuse", "explicit_crossref"],
            "deliberative-revision": ["meta_narration", "hedging_miscalibration", "frictionless_balance",
                                      "appositive_restate", "cleft", "false_range", "elegant_variation",
                                      "template_repetition", "conclusion_summary"]}

def judge_block(only_llm=False):
    items = [t for t in TAXONOMY if (not only_llm or t["det"] == ["LLM-only"])]
    return "\n".join(f"{i}. {t['label']} [{t['family']}] — {t['probe']}." for i, t in enumerate(items, 1))

# ════════════════════════ Layer 1: regex features ════════════════════════
def feats_regex(t):
    """Per-1000-word tell rates (+ mean_sent/burstiness). The universal cue-word library."""
    p = prose(t); pc = re.sub(r"\([^)]*\)", "", p).replace("et al.", "et al"); tl = pc.lower()
    w = len(re.findall(r"\b\w+\b", pc)) or 1
    S = [x.strip() for x in re.split(r"(?<=[.!?])\s+", pc) if len(x.split()) >= 4]
    lens = [len(x.split()) for x in S] or [1]
    P = lambda n: round(1000 * n / w, 2)
    R = lambda pat, s=tl, fl=0: len(re.findall(pat, s, fl))
    return dict(
        mean_sent=round(st.mean(lens), 1), burstiness_CV=round(st.pstdev(lens) / max(st.mean(lens), 1), 2),
        pct_short=round(100 * sum(1 for L in lens if L < 8) / max(len(S), 1)),
        emdash=P(p.count("—")), semicolon=P(pc.count(";")), comma_not=P(R(r",\s+not\s+", pc, 0)),
        rather_than=P(R(r"\brather than\b")), not_but=P(R(r"\bnot\b[^.,]{1,40}\bbut\b")),
        triadic=P(len(re.findall(r"[^,.;:()]+,\s+[^,.;:()]+,\s+(?:and|or)\s+[^,.;:()]+", pc))),
        compound_and=P(R(r",\s+and\s+", pc, 0)),
        appositive=P(R(r",\s+(?:the|a|an)\s+\w+(?:\s+\w+){0,3},", pc, 0)),
        what_cleft=P(R(r"\b[Ww]hat\b[^.,]{1,45}\bis\b", pc, 0) + R(r"\b[Ii]t is\b[^.,]{1,45}\bthat\b", pc, 0)),
        abstract_subj=P(R(r"(?:^|\.\s+)The\s+[a-z]+\s+(?:is|was|are|has|lies|comes|turns)\b", pc, 0)),
        para_open_The=P(sum(1 for s in S if s.startswith("The "))),
        demonstr_abstract=P(R(r"\b(this|that|these|those)\s+(failure|coupling|obstruction|tension|distinction|residue|question|problem|move|point|account|gap|regress|pattern|practice|claim|argument|view|reading)\b")),
        which_is_tail=P(R(r",\s+which\s+(is|are|was|means|gives|makes|leaves|amounts)\b")),
        the_way_each=P(R(r"\bthe way (each|in which|that)\b") + R(r"\bthe (one|kind|sort|thing) (that|which)\b")),
        to_inf_purpose=P(R(r",\s+(?:so as |in order )?to\s+\w+,")),
        intensifier=P(R(r"\b(precisely|exactly|itself|the very|in fact|indeed|merely|simply)\b")),
        that_density=P(R(r"\bthat\b")), first_person=P(R(r"\b(we|our|us)\b")),
        # meta_narration (the deepest tell): announcing the move/status/honesty instead of making it. Covers the
        # stance-narration family (status-of-claim, epistemic-status hedging, honesty-announcing, self-placement).
        meta_narration=P(R(
            r"the (contribution|point|crux) is|(is|are) the (contribution|point|crux|upshot|lesson)\b"
            r"|\b(all|what) we claim\b|the contribution we claim"
            r"|what (this|the|our|the present) (paper|argument|account|reading|analysis) (adds|offers|does|claims|shows)"
            r"|what is new is|the key insight|turns out that|the gap (this|the) (paper|argument)"
            r"|the obvious (move|answer)|the (standard|common) (fix|approach) is"
            r"|status is modest|supplies no .{0,14}rigor|organizes the argument|earns its keep"
            r"|does analytic work|is a consistency (device|check)|whose status is"
            r"|rather than bury|read honestly|to be (honest|clear|precise|fair|blunt)\b"
            r"|hold(s)? a line|in the foreground|keep\w* [^.,]{1,25} in (the foreground|view)"
            r"|belongs here|which is why it sits|sits here\b|deserves a \w+, not a|we are careful|with care, since"
            # first-person contribution-announcing (we recast/carry/adapt/build on the X) — the regex missed this; 3 judges caught it
            r"|\bwe (?:recast|carr(?:y|ies|ied)|adapt|build on|reframe|situate|position|advance) (?:the|a|an|our|this|that)\b")),
        ai_vocab=P(R(r"\b(delv\w+|leverag\w+|utiliz\w+|utilis\w+|robust|comprehensive|streamlin\w+|foster\w*|facilitat\w+|pivotal|nuanced|notably|garner\w*|tapestry|landscape|intricat\w+|interplay|meticulous\w*|showcas\w+|underscor\w+|myriad|multifaceted|boasts|realm)\b")),
        vague_eval_adj=P(R(r"\b(profound|meaningful|significant|crucial|vital|essential|fundamental|paramount|substantial|considerable|notable|central|salient|structural|primary|critical|deeper|broader|richer)\b")),
        transition_open=P(sum(1 for s in S if re.match(r"(Furthermore|Moreover|Additionally|However|Therefore|Thus|Consequently|Indeed|Notably|Importantly|In addition|In conclusion|To summari[sz]e|In summary)\b", s))),
        hedge_phrase=P(R(r"\b(it is important to note|it is worth noting|it is worth mentioning|it should be noted|in many cases|more often than not|needless to say|it goes without saying)\b")),
        hedge_modal=P(R(r"\b(may|might|could|would)\b")),
        puffery=P(R(r"\b(serves as a|stands as a|is a testament|plays a (vital|crucial|pivotal|significant|key) role|under(scor|lin)es (its|the) importance|highlights (its|the) importance|key turning point)\b")),
        # OVER-HONESTY / underclaiming (the OPPOSITE-direction tell to puffery): repeated scope-disclaimers and
        # self-deprecation, the "modest, self-aware, concede-everything" AI register. Corpus-calibrated: a HUMAN
        # paper concedes ONCE; AI re-hedges. Flag HIGH vs corpus (over-hedged) — and note the band, since ZERO
        # disclaimers is the overclaiming tell. Pairs with hedge_modal/hedge_phrase/meta_narration for the stance read.
        disclaimer=P(R(
            r"\bexistence proof\b|\bnot a validation\b|\bbounds (?:the|rather)\b|\bn=1\b|one system, one principal"
            r"|\b(?:we )?(?:do not|don't|cannot|can't|make no|claim no) claim\b|\bwe concede\b|\bwe acknowledge\b"
            r"|\bshort of (?:a )?(?:validation|proof|guarantee|claim)\b|\bwithout (?:establishing|claiming|proving)\b"
            r"|\b(?:an? )?(?:modest|preliminary|illustrative|tentative) (?:claim|case|step|illustration|reading)\b"
            r"|\bholding[^.,]{0,20}(?:scope|view)\b|\bis itself (?:a )?(?:limit|caveat|tell)\b"
            r"|\bconsistent with[^.,]{0,30}(?:not|rather than)\b|\bnot (?:proof|a proof|validation|evidence) of\b"
            r"|\bwe are careful\b|\bto be (?:honest|clear|fair) about\b|\billustration (?:rather than|, not|not a)\b")),
        # COINED / tone-deaf neologisms: Title-Case multi-word terms the author EMPHASIZES or scare-quotes and uses
        # as if established (the "Blame Shield" / "Audited Multidivisional Architecture" habit). A human coins
        # sparingly and signals it; the model mints capitalized compound jargon freely. The feature lists emphasized
        # /quoted candidates; the LLM triages tone-deaf coinage from a legitimate defined term.
        coined_jargon=P(len(re.findall(
            r"\\(?:textbf|emph|textit)\{[A-Z][A-Za-z][^}]{2,46}\}|\*\*[A-Z][A-Za-z][^*]{2,46}\*\*"
            r"|[`'\"][A-Z][a-z]+(?: [A-Z][a-z]+){1,3}[`'\"]", p))),
        # PLACEHOLDER / template residue (a hygiene tell; ANY instance is a fail): xxx, [N], [cite], TBD, <placeholder>,
        # plus AI-tool markup artifacts pasted verbatim (oaicite/contentReference/turn0search/:::), a literal-string CUE.
        placeholder=P(len(re.findall(
            r"\b[xX]{3,}\b|\[(?:TODO|TBD|cite|citation|ref|N|x+|\.\.\.|placeholder)\]|<[a-z][a-z _/-]{2,22}>"
            r"|\bTBD\b|\bFIXME\b|\bplaceholder\b|lorem ipsum"
            r"|oaicite|contentReference|attributableIndex|turn\d+(?:search|view|news|forecast|image)\d+|:::\s*(?:writing|note|task)", t, re.I))),
        # FALSE SCALE / over-enumeration: claims of excessive structure that inflate apparent rigor ("30 tables",
        # "dozens of experimental conditions"). Not data counts (projects/iterations) -- structural padding only.
        false_scale=P(len(re.findall(
            r"\b(?:\d{2,}|dozens|hundreds|thousands|scores|myriad|numerous|countless)\s+(?:of\s+)?"
            r"(?:tables?|figures?|experiments?|conditions?|variants?|configurations?|ablations?|baselines?|settings?|scenarios?)\b",
            pc, re.I))),
        # LISTIFICATION (Wikipedia WP:AILIST "inline-header vertical lists"): prose replaced by vertical lists, the
        # tell being the inline-BOLD-header list item ('- **Term**: ...', '1. **Step**: ...'). Counted from RAW
        # markdown (prose() strips list markers/bold). CUE only — the LLM judge owns list-as-argument vs reference list.
        inline_bold_list=P(len(re.findall(r"(?m)^[ \t>]*(?:[-*•‣]|\d+[.)])\s+\*\*[^*\n]{1,70}\*\*", t))),
        # BOLDFACE overuse (WP:AIBOLD): **bold** in RUNNING prose / as bold lead-in labels (NOT '- ' list items). The
        # list-marker exclusion needs a trailing \s so a line opening with '**bold**' is not mistaken for a '*' bullet. CUE only.
        bold_run=P(len(re.findall(r"(?m)^(?![ \t>]*(?:[-•‣*]|\d+[.)])\s)[^\n]*?(?<!\*)\*\*[^*\n]{1,60}\*\*", t))),
        # REFORMULATION-marker overuse: restating the same point ('in other words', 'i.e.', 'put differently'). CUE only.
        reformulation=P(R(r"\b(in other words|that is to say|put (?:it )?differently|to put it another way|i\.e\.|stated differently|simply put|or rather)\b")),
        # EXPLICIT CROSS-REFERENCE over-signaling (WP:AICROSSREF): 'as discussed above', 'as we will show'. CUE only.
        explicit_crossref=P(R(r"\bas (?:discussed|noted|shown|mentioned|described|seen|explained|stated) (?:above|below|earlier|previously)\b|\bas we (?:will|shall) (?:see|show|discuss)\b|\breturning to (?:the|our|this)\b|\bas noted (?:above|earlier)\b")),
    )

DEFAULTS = dict(pct_short=9, emdash=3.0, semicolon=2.5, comma_not=0.7, rather_than=1.3, not_but=0.8, triadic=11,
    compound_and=13, appositive=0.6, what_cleft=1.6, abstract_subj=1.0, para_open_The=7.0, demonstr_abstract=0.5,
    which_is_tail=0.9, the_way_each=0.3, to_inf_purpose=1.6, intensifier=1.8, that_density=17, first_person=99,
    meta_narration=1.2, ai_vocab=4.0, vague_eval_adj=5.0, transition_open=4.0, hedge_phrase=0.6, hedge_modal=99, puffery=0.5,
    coined_jargon=4.0, placeholder=0.1, false_scale=0.2, inline_bold_list=0.4, bold_run=0.6, reformulation=1.0, explicit_crossref=0.6)
BAND = dict(mean_sent=(15, 33), burstiness_CV=(0.45, 1.2))

def pctl(vals, q):
    vals = sorted(vals)
    return None if not vals else vals[min(len(vals) - 1, max(0, int(round((q / 100) * (len(vals) - 1)))))]

def check(text, corpus_dirs=None, q=80):
    d = feats_regex(text)
    corp = [feats_regex(c) for _, c in (_read_corpus(corpus_dirs) if corpus_dirs else [])]
    mode = f"Layer-2 (corpus p{q}, n={len(corp)})" if len(corp) >= 8 else "Layer-1 (universal defaults)"
    rows, flags = [], 0
    for k, v in d.items():
        if k in BAND:
            lo, hi = BAND[k]; flag = "OUT" if (v < lo or v > hi) else "ok"; thr = f"{lo}-{hi}"
        else:
            thr = pctl([c[k] for c in corp], q) if len(corp) >= 8 else DEFAULTS.get(k, 1.0)
            flag = "HIGH" if (thr is not None and v > max(thr, 0.1)) else "ok"
        if flag != "ok": flags += 1
        rows.append((k, round(thr, 2) if isinstance(thr, float) else thr, v, flag))
    return mode, rows, flags

# ════════════════════ Layer A: function-word Burrows's Delta ════════════════════
_FW_RAW = ("the of and to a in that is was it for on are as with his they at be this from i you or had not but what "
    "all were we when your can said there use an each which she do how their if will up other about out many then them "
    "these so some her would make like him into time has look two more write go see number no way could people my than "
    "first water been call who down side now find any new work part take get place made live where after back little "
    "only round man year came show every good me give our under name very through just form much great think say help "
    "low line before turn cause same mean differ move right also around another come three word must because does even "
    "here old too may such still between should never while might shall upon whether toward against among nor yet either "
    "neither its their our your my whom whose itself himself herself themselves")
FW = sorted(set(_FW_RAW.split()))
_SUBORD = re.compile(r"\b(because|although|though|while|whereas|since|if|unless|until|when|whenever|where|that|which|who|whom|whose|after|before|as|so that|in order)\b")

def _mtld(tokens, thr=0.72):
    def one(toks):
        fac, types, n = 0.0, set(), 0
        for x in toks:
            n += 1; types.add(x)
            if len(types) / n <= thr: fac += 1; types, n = set(), 0
        if n > 0: fac += (1 - len(set(toks[-n:])) / n) / (1 - thr)
        return len(toks) / fac if fac else len(toks)
    return 0.0 if len(tokens) < 50 else round((one(tokens) + one(tokens[::-1])) / 2, 1)

def profile(t):
    pc = re.sub(r"\([^)]*\)", "", prose(t)); tl = pc.lower()
    toks = re.findall(r"[a-z]+", tl); w = len(toks) or 1
    S = [x for x in re.split(r"(?<=[.!?])\s+", pc) if len(x.split()) >= 4] or [pc]
    counts = collections.Counter(x for x in toks if x in set(FW))
    fwfreq = {fw: 1000 * counts.get(fw, 0) / w for fw in FW}
    nsub = len(_SUBORD.findall(tl)); ncom = tl.count(",")
    glob = dict(mtld=_mtld(toks), clauses_per_s=round((ncom + nsub + len(S)) / len(S), 2),
                subord_ratio=round(1000 * nsub / w, 1), mean_clause=round(w / max(ncom + nsub + len(S), 1), 1),
                fw_rate=round(1000 * sum(counts.values()) / w, 1))
    return fwfreq, glob

def delta(target_fw, mean, sd):
    return (round(sum(abs((target_fw[fw] - mean[fw]) / sd[fw]) for fw in FW if sd[fw] > 1e-9) / len(FW), 3),
            sorted(((target_fw[fw] - mean[fw]) / sd[fw], fw) for fw in FW if sd[fw] > 1e-9))

# ════════════════════ Layer C: spaCy syntactic + syntactic meta-narration ════════════════════
_NLP = None
def _nlp():
    global _NLP
    if _NLP is None:
        import spacy; _NLP = spacy.load("en_core_web_sm", disable=["ner"])  # lemmatizer ON for verb matching
    return _NLP

# Meta-narration done STRUCTURALLY, not by phrase. Calibration finding: broad metadiscourse ("we argue", "the paper
# shows") is NORMAL (corpus 8-20% of sentences). The AI tell is the STANCE/STATUS flavor only: announcing the
# claim's modesty/honesty/partialness, conceding in self-aware hedge-rhythm, naming-rather-than-burying, X-is-the-
# contribution. So we target that — copula+status-adjective, self-mention/discourse-noun + stance verb, negated claim.
_DISC_NOUN = {"paper", "argument", "section", "claim", "contribution", "model", "analysis", "point", "framework",
    "thesis", "reading", "account", "table", "chapter", "essay", "study", "work", "discussion", "approach", "aim",
    "purpose", "parallel", "analogy", "homology", "prediction", "proposition", "lesson", "upshot", "crux", "ledger"}
_STANCE_V = {"concede", "grant", "acknowledge", "decline", "disclaim", "qualify", "hedge", "stress", "emphasize",
    "emphasise", "name", "note", "hold", "bury", "admit", "caution", "disavow", "own"}
_STATUS_ADJ = {"modest", "honest", "careful", "candid", "partial", "incomplete", "limited", "deliberate", "explicit",
    "tentative", "provisional", "narrow", "thin", "precise", "cautious", "conscious", "removable"}
_PRED_N = {"contribution", "point", "crux", "claim", "thesis", "lesson", "upshot", "aim", "purpose", "move"}
_CLAIM_V = {"claim", "prove", "derive", "establish", "assert", "guarantee"}

def _is_meta_sent(s):
    root = s.root; rl = root.lemma_.lower(); kids = list(root.children)
    subj = [t for t in kids if t.dep_ in ("nsubj", "nsubjpass", "csubj")]
    fp = any(t.lower_ in ("we", "i") for t in subj); disc = any(t.lemma_.lower() in _DISC_NOUN for t in subj)
    if rl == "be":
        if any(c.dep_ in ("acomp", "attr") and c.lemma_.lower() in _STATUS_ADJ for c in kids): return True
        if any(c.dep_ in ("attr", "acomp") and c.lemma_.lower() in _PRED_N for c in kids): return True
    if (fp or disc) and rl in _STANCE_V: return True
    if fp and rl in _CLAIM_V and any(c.dep_ == "neg" for c in kids): return True
    return False

_NOMINAL = ("tion", "ment", "ence", "ance", "ity", "ness", "ism", "sion")
def _depth(tok):
    d, cur = 0, tok
    while cur.head != cur and d < 100: d += 1; cur = cur.head
    return d

def feats_syntax(text):
    doc = _nlp()(re.sub(r"\([^)]*\)", "", prose(text))[:120000])
    toks = [t for t in doc if not t.is_space and not t.is_punct]
    sents = [s for s in doc.sents if len(s) > 3]
    n = len(toks) or 1; ns = len(sents) or 1
    pos = collections.Counter(t.pos_ for t in toks); Pr = lambda x: round(1000 * pos.get(x, 0) / n, 1)
    posbg = collections.Counter()
    for s in sents:
        ps = [t.pos_ for t in s if not t.is_space and not t.is_punct]; posbg.update(zip(ps, ps[1:]))
    g = dict(
        mean_dep_dist=round(sum(abs(t.i - t.head.i) for t in toks if t.head != t) / n, 2),
        parse_depth=round(st.mean([max((_depth(t) for t in s if not t.is_space), default=0) for s in sents]), 2) if sents else 0,
        noun_rate=Pr("NOUN"), adj_rate=Pr("ADJ"), pron_rate=Pr("PRON"), adp_rate=Pr("ADP"),
        nominal_rate=round(1000 * sum(1 for t in toks if t.pos_ == "NOUN" and t.text.lower().endswith(_NOMINAL)) / n, 1),
        subcl_ratio=round(sum(1 for t in toks if t.dep_ in ("advcl", "ccomp", "relcl", "acl", "csubj")) / ns, 2),
        that_subj_pm=round(1000 * sum(1 for t in doc if t.text.lower() == "that" and t.head.dep_ in ("csubj", "ccomp", "acl")) / n, 2),
        metanarr_synt=round(100 * sum(1 for s in sents if _is_meta_sent(s)) / ns, 1),
    )
    return g, posbg, n

def posbg_delta(target, corpus_bgs):
    keys = set().union(*[set(b) for b in corpus_bgs]) | set(target)
    norm = lambda b: {k: b.get(k, 0) / (sum(b.values()) or 1) for k in keys}
    cn = [norm(b) for b in corpus_bgs]; tn = norm(target)
    mean = {k: st.mean(c[k] for c in cn) for k in keys}; sd = {k: (st.pstdev([c[k] for c in cn]) or 1e-9) for k in keys}
    zs = [abs((tn[k] - mean[k]) / sd[k]) for k in keys if sd[k] > 1e-9]
    return (round(sum(zs) / len(zs), 3) if zs else 0.0, sorted(((tn[k] - mean[k]) / sd[k], k) for k in keys if sd[k] > 1e-9))

# ════════════════════ headings (title + section headings carry tells prose() strips out) ════════════════════
# Academic headings are PLAIN/descriptive ("Methods", "Related work"). AI headings are clever/compound/Title-Case
# ("Two Predictions and a Minimal Model", "Toward an Audit Institution", "Disanalogies and the 'Just Good
# Engineering' Objection"). Differential vs the corpus's own headings, since some venues do use Title Case.
def headings(text):
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), "")
    if not title:  # LaTeX
        m = re.search(r"\\title\{([^}]*)\}", text)
        title = re.sub(r"\\[a-zA-Z]+\*?|[{}]", "", m.group(1)).strip() if m else ""
    heads = [re.sub(r"^#+\s*\d+(?:\.\d+)*\.?\s*", "", l).strip() for l in text.splitlines() if re.match(r"^###?\s", l)]
    heads += [re.sub(r"\\[a-zA-Z]+\*?|[{}]", "", m.group(1)).strip()  # LaTeX \section/\subsection titles
              for m in re.finditer(r"\\(?:sub)?section\*?\{([^}]*)\}", text)]
    return title, [h for h in heads if h and h.lower() != "abstract"]

# Interrogative / wh-nominal headings & captions — a tell the antithesis/nominalization rules miss. Human academic
# headings are declarative ("The rival-framework ledger"); AI reaches for interrogative or wh-nominal titles ("What the
# rival frameworks predict", "What survives the floor", "How the floor works"). Flagged when a heading/caption OPENS on a
# wh-word or a yes/no auxiliary, or ENDS in '?'. Generalized to the PATTERN, not any single hardcoded phrase.
_INTERROG = re.compile(r"^(?:what|how|why|when|where|who|whom|whose|which|whether|is|are|was|were|do|does|did|can|"
                       r"could|should|would|will|shall|has|have|had|may|might)\b", re.I)

def _titlecase_ratio(s):
    nf = re.findall(r"[A-Za-z][A-Za-z'\-]+", s)[1:]  # skip first word (always capitalized)
    keep = {"ai", "eu", "gaap", "sox", "us", "uk", "rlhf", "llm"}
    return round(sum(1 for w in nf if w[0].isupper() and w.lower() not in keep) / len(nf), 2) if nf else 0.0

# Abstract-NOMINALIZATION headings — the tell the Title-Case/comma/puffery/antithesis rules structurally miss. AI piles
# abstract nominalizations ("The governance of the residual", "Toward an audit institution") where human headings stay
# concrete/verbal ("A checker that tested the wrong property", "Where the analogy stops"). Biber's "nominal style";
# Brysbaert concreteness norms are the gold reference but need a 40k lexicon, so we use the nominalization-suffix proxy
# (corpus-free). GERUNDS (-ing) are NOT counted — "Governing X" is the concrete fix for "The governance of X".
_NOM_SUF = re.compile(r"(?:tion|sion|ment|ance|ence|ity|ety|ism|ness|ure|acy|ship|hood|al|cy)$")
_NOM_STOP = set("the a an of to in on for and or over with as is are was were be that this these those from one its "
                "their our we who whom whose what when where why how no not".split())
def _nom_rate(h):
    """Fraction of a heading's content words that are abstract nominalizations. 1.0 = pure abstract nominalization."""
    cw = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", h.lower()) if w not in _NOM_STOP]
    return round(sum(1 for w in cw if _NOM_SUF.search(w)) / len(cw), 2) if cw else 0.0
_STD_HEADINGS = {"introduction", "conclusion", "conclusions", "abstract", "background", "discussion", "results",
                 "method", "methods", "related work", "references", "acknowledgments", "acknowledgements",
                 "limitations", "appendix", "preliminaries", "notation", "overview", "summary"}
def abstract_headings(text, thresh=0.6, min_content=2):
    """Vague abstract-nominalization headings, CANDIDATES (a defined-term heading like 'Attestation and liability' or
    the framework's own name legitimately scores high; the author/LLM triages which are vague vs technical). Drops
    standard section names and single-content-word labels (one-word nominalizations like 'Independence' are technical,
    not vague PHRASES). Returns [(rate, heading)] >= thresh, worst first."""
    _, hs = headings(text)
    out = []
    for h in hs:
        if h.lower().strip() in _STD_HEADINGS:
            continue
        cw = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", h.lower()) if w not in _NOM_STOP]
        if len(cw) >= min_content and _nom_rate(h) >= thresh:
            out.append((_nom_rate(h), h))
    return sorted(out, reverse=True)

def heading_feats(text):
    title, heads = headings(text)
    allh = ([title] if title else []) + heads
    if not allh: return {}
    nh = len(heads) or 1; na = len(allh); R = lambda c: round(c / na, 2)
    return dict(
        title_case=round(sum(_titlecase_ratio(h) for h in allh) / na, 2),     # 1.0 = Title Case, 0 = sentence case
        mean_words=round(sum(len(h.split()) for h in allh) / na, 1),
        nominalization=round(sum(_nom_rate(h) for h in allh) / na, 2),  # abstract-noun density (Biber nominal style); AI piles nominalizations
        compound=round(sum(1 for h in heads if re.search(r"\b(and|or)\b", h.lower())) / nh, 2),  # "X and Y" promise
        toward=R(sum(1 for h in allh if re.match(r"(?i)towards?\b", h))),       # aspirational cliché
        antithesis=R(sum(1 for h in allh if re.search(r"\bnot\b", h.lower()))), # "X, not Y" / "why this is not"
        colon=R(sum(1 for h in allh if ":" in h)), quoted=R(sum(1 for h in allh if '"' in h or '“' in h)),  # not the possessive apostrophe
        comma=R(sum(1 for h in heads if "," in h)),
        selfcong=R(sum(1 for h in allh if re.search(r"(?i)\b(honest|candid|careful|rigorous|novel|elegant|crucial|just good)\b", h))),
        interrogative=R(sum(1 for h in allh if _INTERROG.match(h) or h.rstrip().endswith("?"))),  # wh-/aux-opener or '?' heading
    )

def _all_heading_strings(text):
    """Section titles AND figure/table captions, from BOTH LaTeX (\\section/\\caption) and markdown (#) source, markup stripped."""
    out  = [m.group(1) for m in re.finditer(r"\\(?:sub){0,2}section\*?\{([^}]*)\}", text)]
    out += [m.group(1) for m in re.finditer(r"\\caption\{([^}]*)\}", text)]
    out += [re.sub(r"^#+\s*\d+(?:\.\d+)*\.?\s*", "", l).strip() for l in text.splitlines() if re.match(r"^#{1,4}\s", l)]
    t = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), "")
    if t: out.append(t)
    return [h2 for h2 in (re.sub(r"\\[a-zA-Z]+\*?|[{}$]", "", h).strip() for h in out) if h2]

def interrogative_headings(text):
    """Headings/captions reading as interrogative or wh-nominal (an AI tell; human academic headings are declarative).
    Trailing-'?' worst, then wh-/aux-openers. Drops standard section names. Returns offending strings, worst-first."""
    out = []
    for h in _all_heading_strings(text):
        if h.lower() in _STD_HEADINGS: continue
        if h.rstrip().endswith("?"): out.append((2, h))
        elif _INTERROG.match(h):      out.append((1, h))
    return [h for _, h in sorted(out, key=lambda x: -x[0])]

# ════════════════════════════ the fold (verdict) ════════════════════════════
def _robust_hi(vals, k=3.0):
    """Robust upper bound of a small sample: median + k·(scaled MAD). Replaces a brittle max() outlier cutoff,
    so one extreme leave-one-out document no longer sets an impossibly high bar. Validated on the benchmark
    (held-out human vs AI essays): the old max() flagged ~5% of AI drafts as outliers; this catches ~100% at
    ~8% human false-positive, with the metric (Burrows's Delta) unchanged."""
    if not vals: return 0.0
    m = st.median(vals); mad = 1.4826 * (st.median([abs(v - m) for v in vals]) or 1e-9)
    return m + k * mad

def mirror(draft_path, corpus_dirs):
    draft = pathlib.Path(draft_path).read_text(errors="ignore")
    corpus = [c for _, c in _read_corpus(corpus_dirs)]
    if len(corpus) < 3:
        print(f"mirror needs >=3 corpus documents (got {len(corpus)})"); return
    print(f"# DEAI MIRROR — {pathlib.Path(draft_path).name} — corpus n={len(corpus)} ({', '.join(corpus_dirs)})\n", flush=True)
    # A1. function-word Delta vs corpus leave-one-out spread
    cfw = [profile(c)[0] for c in corpus]
    mean = {fw: st.mean(p[fw] for p in cfw) for fw in FW}; sd = {fw: (st.pstdev([p[fw] for p in cfw]) or 1e-9) for fw in FW}
    fw_int = []
    for i in range(len(cfw)):
        rest = [cfw[j] for j in range(len(cfw)) if j != i]
        m = {fw: st.mean(p[fw] for p in rest) for fw in FW}; s = {fw: (st.pstdev([p[fw] for p in rest]) or 1e-9) for fw in FW}
        fw_int.append(delta(cfw[i], m, s)[0])
    dfw, fw_devs = delta(profile(draft)[0], mean, sd)
    # A2. POS-bigram Delta (spaCy)
    parsed = [feats_syntax(c) for c in corpus]; cg = [p[0] for p in parsed]; cbg = [p[1] for p in parsed]
    dg, dbg, _ = feats_syntax(draft)
    posd, pos_top = posbg_delta(dbg, cbg)
    pos_int = [posbg_delta(cbg[i], [cbg[j] for j in range(len(cbg)) if j != i])[0] for i in range(len(cbg))]
    fw_hi, pos_hi = _robust_hi(fw_int), _robust_hi(pos_int)
    fw_ok, pos_ok = dfw <= fw_hi, posd <= pos_hi
    print("== A. DISTRIBUTIONAL FINGERPRINTS (the verdict) ==")
    print(f"  function-word Delta  {dfw:>6}   corpus LOO med {round(st.median(fw_int),2)}, outlier > {round(fw_hi,2)}   {'MIRRORS' if fw_ok else 'OUTLIER'}")
    print(f"  POS-bigram   Delta   {posd:>6}   corpus LOO med {round(st.median(pos_int),2)}, outlier > {round(pos_hi,2)}   {'MIRRORS' if pos_ok else 'OUTLIER'}")
    print(f"  >>> OVERALL: {'MIRRORS on both fingerprints' if (fw_ok and pos_ok) else 'STYLOMETRIC OUTLIER — see below'}\n")
    actions = []
    # B. regex shape tells
    _, rows, flags = check(draft, corpus_dirs)
    bad = [(k, v) for k, thr, v, fl in rows if fl != "ok"]
    print(f"== B. shape/lexical tells (regex, {flags} flags) ==\n  " + (", ".join(f"{k}={v}" for k, v in bad) or "clean"))
    if bad: actions.append("regex: " + ", ".join(k for k, _ in bad[:8]))
    # C. spaCy syntactic
    print("\n== C. syntactic / POS (draft vs corpus med [min,max]) ==")
    for k in dg:
        vals = sorted(g[k] for g in cg); hi = vals[int(0.85 * (len(vals) - 1))]; lo = vals[int(0.15 * (len(vals) - 1))]
        fl = "HIGH" if dg[k] > hi else "LOW" if dg[k] < lo else "ok"
        if fl != "ok":
            print(f"  {k:14}{dg[k]:>8}   corpus {round(st.median(vals),1)} [{vals[0]}, {vals[-1]}]   <<< {fl}")
            actions.append(f"{k} {fl} ({dg[k]} vs {round(st.median(vals),1)})")
    # D. lexical over-use (function-word z-scores)
    over = [(z, fw) for z, fw in fw_devs[::-1] if z > 3.0][:8]
    print("\n== D. function-word over-use (z-score) ==\n  " + ", ".join(f"{fw}(+{z:.1f})" for z, fw in over))
    if over: actions.append("over-used words: " + ", ".join(fw for _, fw in over[:6]))
    # E. headings (prose() stripped them; they carry their own tells)
    dh = heading_feats(draft)
    if dh:
        ch = [h for h in (heading_feats(c) for c in corpus) if h]
        cmed = {k: round(st.median([h[k] for h in ch]), 2) for k in dh} if ch else {k: 0 for k in dh}
        print("\n== E. headings (draft vs corpus med) ==")
        hbad = []
        for k in dh:
            up = dh[k] > cmed[k] * 1.3 + 0.05 if k in ("title_case", "mean_words", "compound") else dh[k] > cmed[k]
            if up:
                print(f"  {k:11}{dh[k]:>7}   corpus {cmed[k]}   <<<"); hbad.append(k)
        print("  (headings match the corpus)" if not hbad else "")
        if hbad: actions.append("headings: " + ", ".join(hbad))
    print("\n== ACTIONABLE ==")
    for a in actions: print("  - " + a)
    if fw_ok and pos_ok and not bad: print("  (fingerprints in-range and no flags — only fine-tuning left)")

# ════════════════════════ empirical inference (taxonomy from data) ════════════════════════
def infer(human_dir, ai_dir):
    human = [c for _, c in _read_corpus([human_dir])]
    ai = [p.read_text(errors="ignore") for p in sorted(pathlib.Path(ai_dir).glob("ai_*.md"))
          if len(prose(p.read_text(errors="ignore")).split()) > 300]
    print(f"human={len(human)} AI={len(ai)} — parsing (spaCy) ...", flush=True)
    allf = lambda t: {**feats_regex(t), **{"synt_" + k: v for k, v in feats_syntax(t)[0].items()}}
    HF = [allf(t) for t in human]; AF = [allf(t) for t in ai]
    if not HF or not AF:
        print(f"infer needs at least one human and one AI document (found {len(HF)} human, {len(AF)} AI)"); return
    auc = lambda a, h: 0.5 if not a or not h else sum((x > y) + 0.5 * (x == y) for x in a for y in h) / (len(a) * len(h))
    feat2cat = {}
    for cat in TAXONOMY:
        for d in cat["det"]:
            if d != "LLM-only": feat2cat[d] = cat["label"]; feat2cat["synt_" + d] = cat["label"]
    rows = []
    for k in HF[0]:
        a = auc([f[k] for f in AF], [f[k] for f in HF])
        cat = feat2cat.get(k) or feat2cat.get(k.replace("synt_", "")) or "—"
        rows.append((abs(a - 0.5), a, k, round(st.median([f[k] for f in HF]), 2), round(st.median([f[k] for f in AF]), 2), cat))
    rows.sort(reverse=True)
    print("\n== PER-FEATURE AUC = P[AI>human] (0.5 none, >0.5 AI-higher, <0.5 human-higher) ==")
    print(f"{'feature':18}{'AUC':>6}{'h_med':>9}{'ai_med':>9}  category")
    for disc, a, k, hm, am, cat in rows:
        mark = "  <<< strong" if disc >= 0.25 else ("  < invert" if a < 0.42 else "")
        print(f"{k:18}{a:>6.2f}{hm:>9}{am:>9}  {cat}{mark}")
    print("\n== PER-CATEGORY mean |AUC-.5| (does the data support the meta-label?) ==")
    cd = collections.defaultdict(list)
    for disc, a, k, hm, am, cat in rows:
        if cat != "—": cd[cat].append(disc)
    for cat, ds in sorted(cd.items(), key=lambda x: -st.mean(x[1])):
        print(f"  {cat:34} {st.mean(ds):.2f}  (n={len(ds)})")
    # open-ended discovery: AI-excess words (Kobak-style; topic-confounded, keep the style residue)
    def wc(texts):
        c = collections.Counter(); tot = 0
        for t in texts:
            ws = re.findall(r"[a-z][a-z'-]+", prose(t).lower()); c.update(ws); tot += len(ws)
        return c, tot
    hc, ht = wc(human); ac, at = wc(ai)
    cand = sorted(((1000 * ac[k] / at) / max(1000 * hc.get(k, 0) / ht, 0.05), 1000 * hc.get(k, 0) / ht, 1000 * ac[k] / at, k)
                  for k in ac if ac[k] >= max(6, 0.4 * len(ai)) and len(k) > 3)
    print("\n== DISCOVERY: AI-excess words (rate ratio AI/human; topic-confounded — keep the style residue) ==")
    for ratio, hr, ar, k in [c for c in cand if c[0] >= 2.0][::-1][:24][::-1]:
        print(f"  {k:22} {ratio:>5.1f}x   human {hr:.2f}/1k  AI {ar:.2f}/1k")

# ════════════════════════════ LLM judge (semantic categories) ════════════════════════════
# model context budgets (tokens) -- the REAL window, so we never truncate at an arbitrary char count
_MODEL_CTX = {"deepseek": 64000, "grok": 128000, "kimi": 256000, "claude": 200000, "codex": 200000, "gpt": 128000, "gemini": 1000000}
def _ctx_tokens(model):
    m = (model or "").lower()
    return next((v for k, v in _MODEL_CTX.items() if k in m), 32000)
def _fit_or_chunk(text, model, reserve_tok=12000):
    """Split text ONLY if it exceeds the model's real context window (no arbitrary char cap). Paragraph-boundary
    chunks, each sized to fit the model with room reserved for the prompt + reply. A doc that fits is sent whole,
    so the panel reads EVERY word; a doc that doesn't is read in full across consecutive parts."""
    budget = max(4000, _ctx_tokens(model) - reserve_tok) * 4   # ~4 chars per token
    if len(text) <= budget: return [text]
    out, cur = [], ""
    for para in text.split("\n\n"):
        while len(para) > budget:                              # a single oversized paragraph: hard-split
            out.append(para[:budget]); para = para[budget:]
        if cur and len(cur) + len(para) + 2 > budget:
            out.append(cur.strip()); cur = ""
        cur += para + "\n\n"
    if cur.strip(): out.append(cur.strip())
    return out

# ── self-improving taxonomy: harvest the tells the 25 categories MISS, promote by recurrence (human-gated) ──
# A tell the taxonomy does not cover is a RESIDUAL. The judge is given an open slot to name it; we persist it as a CANDIDATE
# with provenance. A candidate earns promotion into TAXONOMY only after it (1) recurs across multiple docs/judges
# and (2) survives a nearest-confuser check against the existing categories — and promotion is a HUMAN edit to the
# TAXONOMY list, never an automatic append. A proposal does not grade itself; the same discipline the repo preaches.
_CAND_PATH = pathlib.Path(__file__).with_name("tell_candidates.jsonl")
def _cand_key(label):
    return re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")
def _harvest_candidates(text, doc, judge_name, venue, marker="UNCATALOGUED"):
    """Parse '<marker>: <label> — <span> — <why/nearest>' lines from a judge reply; append with provenance.
    marker is UNCATALOGUED (bottom-up, from a real doc) or MISSING (top-down, from adversarial taxonomy review).
    Returns the count harvested. Existing TAXONOMY labels are skipped (the judge mislabelled a known category)."""
    known = {_cand_key(t["label"]) for t in TAXONOMY} | {_cand_key(t["id"]) for t in TAXONOMY}
    n = 0
    for m in re.finditer(rf"^\s*[-*\d.]*\s*{marker}:\s*(.+)$", text, re.M | re.I):
        raw = m.group(1).strip()
        if not raw or raw.lower().rstrip(".") in ("none", "n/a", "na"): continue
        parts = re.split(r"\s+(?:—|–|--|-)\s+", raw, maxsplit=2)
        label = parts[0].strip().strip("*`\"'.")[:80]
        span  = parts[1].strip()[:240] if len(parts) > 1 else ""
        why   = parts[2].strip()[:400] if len(parts) > 2 else ""
        key = _cand_key(label)
        if not key or key in known: continue
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": key, "label": label, "source": marker.lower(),
               "doc": str(doc), "judge": judge_name, "venue": venue, "span": span, "why": why}
        with _CAND_PATH.open("a") as f: f.write(json.dumps(rec) + "\n")
        n += 1
    return n

def candidates(min_docs=1):
    """Review harvested uncatalogued-tell candidates: group by label, show recurrence across docs/judges (the
    promotion signal). Promotion into TAXONOMY stays a human edit. Pass min_docs=2 to see only recurring residuals."""
    if not _CAND_PATH.exists():
        print(f"# no tell candidates yet ({_CAND_PATH.name} absent) — run `judge` over more docs to harvest residuals"); return
    recs = [json.loads(l) for l in _CAND_PATH.read_text().splitlines() if l.strip()]
    groups = collections.defaultdict(list)
    for r in recs: groups[r["key"]].append(r)
    rows = sorted(groups.items(), key=lambda kv: (len({r["doc"] for r in kv[1]}), len(kv[1])), reverse=True)
    print(f"# tell-candidate review — {len(recs)} raw harvests, {len(groups)} distinct residual tells "
          f"({_CAND_PATH.name})\n")
    print("Promotion signal = recurs across MANY docs AND judges, and is not already a TAXONOMY category.\n")
    shown = 0
    for key, rs in rows:
        ndoc = len({r["doc"] for r in rs}); njudge = len({r["judge"] for r in rs})
        if ndoc < min_docs: continue
        shown += 1
        flag = "  <-- PROMOTION CANDIDATE" if (ndoc >= 2 and njudge >= 2) else ""
        print(f"## {rs[0]['label']}   [{ndoc} docs, {njudge} judges, {len(rs)} hits]{flag}")
        print(f"   nearest/why: {rs[0]['why'][:160] or '(none given)'}")
        print(f"   e.g.: \"{rs[0]['span'][:120]}\"  ({pathlib.Path(rs[0]['doc']).name}, {rs[0]['judge']})\n")
    if not shown: print(f"(no candidates with >= {min_docs} docs)")
    print("To promote: add a dict to TAXONOMY (label/family/definition/why_ai/det/probe) and, if detectable, a regex feat.")

def judge(draft_path, venue, only=None):
    env = _load_env()
    _raw = re.sub(r"<!--.*?-->", "", pathlib.Path(draft_path).read_text(errors="ignore"), flags=re.DOTALL)
    # prose() strips headings, so the judge would CONFABULATE heading tells (the taxonomy has heading categories).
    # Feed the real title + headings as a labeled block so heading flags are grounded in the actual document.
    _title = next((l[2:].strip() for l in _raw.splitlines() if l.startswith("# ")), "")
    _heads = [re.sub(r"^#+\s*\d*\.?\s*", "", l).strip() for l in _raw.splitlines() if re.match(r"^#{2,}\s", l)]
    _heads = [h for h in _heads if h and h.lower() != "abstract"]
    _hblock = (f"=== TITLE ===\n{_title}\n=== HEADINGS ===\n" + " | ".join(_heads) + "\n\n") if (_title or _heads) else ""
    body = _hblock + prose(_raw)
    instr = (f"You are a forensic linguist checking whether ACADEMIC prose reads AI-generated, for a QUALITY purpose "
              f"(the author wants to read like a human scholar in {venue}). The absolute probability saturates for "
              "dense academic prose, so DO NOT chase the number — find INSTANCES of these tell CATEGORIES (the "
              "abstraction above cue words; an instance need not match any fixed phrase). For EACH category quote up "
              "to 3 exact spans, or write 'none':\n\n" + judge_block() +
              "\n\nFormat: for each category number, '<n>. <label>: <span> — <rewrite that fixes it>' or '<n>. none'. "
              "\n\nTHEN, the taxonomy is incomplete by design: if you find a RECURRING AI tell that NONE of the "
              "categories above captures, surface it as a candidate for the catalog. One per line, up to 3:\n"
              "'UNCATALOGUED: <short label for the pattern> — <exact span> — <why it reads AI, and which numbered "
              "category is closest but insufficient>'. Only for genuinely new, recurring patterns — if every tell you "
              "found fits an existing category, write 'UNCATALOGUED: none'.\n\n"
              "End with GLOBAL: <one sentence: biggest reason it reads AI, or 'reads human'>.\n\n")
    judges = _roster()
    if only: judges = [j for j in judges if j[0] in only] or judges
    for name, url, keyvar, model, extra in judges:
        print(f"\n## {name} — taxonomy-driven tell judge\n", flush=True)
        chunks = _fit_or_chunk(body, model)
        for ci, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"\n### document part {ci+1}/{len(chunks)} (chunked to fit {model}; the panel still reads every word)\n", flush=True)
            tag = f" (part {ci+1} of {len(chunks)})" if len(chunks) > 1 else ""
            prompt = instr + f"=== PROSE{tag} ===\n" + chunk
            try:
                payload = {"model": model, "temperature": 0.2, "max_tokens": 3000, "messages": [{"role": "user", "content": prompt}]}  # 25 categories + UNCATALOGUED + GLOBAL: 1600 truncated the tail
                payload.update(extra)   # per-provider extras (e.g. kimi thinking-disabled)
                if name == "kimi":   # kimi-k2.6: temp 0.2 + default thinking → reasoning_content eats the budget / 400s
                    payload["temperature"] = 0.6; payload["thinking"] = {"type": "disabled"}
                req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                    headers={"Authorization": f"Bearer {env.get(keyvar,'')}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    reply = json.loads(r.read().decode())["choices"][0]["message"]["content"]
                print(reply, flush=True)
                nc = _harvest_candidates(reply, pathlib.Path(draft_path).name, name, venue)   # self-improving loop
                if nc: print(f"\n[harvested {nc} uncatalogued-tell candidate(s) -> {_CAND_PATH.name}; review with `deai.py candidates`]", flush=True)
            except Exception as e:
                print(f"[ERROR {type(e).__name__}: {str(e)[:200]}]", flush=True)

def taxonomy_review(only=None):
    """ADVERSARIAL review of the TAXONOMY itself by the judge panel — audit the meta-language, not any document.
    Finds non-MECE overlaps, missing recognized tells, weak/confabulation-prone probes, wrong family, merge/split.
    The TOP-DOWN complement to `judge`'s bottom-up UNCATALOGUED harvest; 'MISSING:' proposals flow into the same
    tell_candidates.jsonl (review with `candidates`). Default panel is deepseek + kimi (two adversarial voices)."""
    env = _load_env()
    dump = "\n".join(f"{i}. [{t['family']}] {t['label']} — {t['definition']} WHY: {t['why_ai']} PROBE: {t['probe']}"
                     for i, t in enumerate(TAXONOMY, 1))
    prompt = (
        "You are a skeptical computational-linguistics referee auditing a TAXONOMY of 'tells' that a panel uses to "
        "decide whether academic/technical prose reads AI-generated. AUDIT THE TAXONOMY ITSELF — do NOT apply it to "
        "any text. Be specific and cite category NUMBERS. Assess:\n"
        "1. NON-MECE / OVERLAP: pairs of categories that overlap or are not mutually exclusive — name the numbers and "
        "say exactly where to draw the boundary or which should absorb which.\n"
        "2. MISSING: recognized AI-writing tells from the documented literature (Wikipedia 'Signs of AI writing', "
        "AI-detection research, editor lore) that are NOT represented. Format each EXACTLY as:\n"
        "   'MISSING: <short label> — <one-line probe a judge could apply> — <why it is a recognized tell>'\n"
        "3. WEAK PROBE: probes vague enough to cause false positives, or that a judge could confabulate — name the "
        "number, give a tighter probe.\n"
        f"4. WRONG FAMILY: any category filed under the wrong family (families: {', '.join(FAMILIES)}).\n"
        "5. MERGE / SPLIT: categories to merge, or one category hiding two distinct tells that should be split.\n"
        "End with: TOP_FIX: <the single highest-leverage change to the taxonomy>.\n\n"
        f"=== TAXONOMY ({len(TAXONOMY)} categories) ===\n" + dump)
    panel = _roster()
    if only: panel = [j for j in panel if j[0] in only] or panel
    for name, url, keyvar, model, extra in panel:
        print(f"\n## {name} — adversarial taxonomy review\n", flush=True)
        try:
            payload = {"model": model, "temperature": 0.3, "max_tokens": 3000, "messages": [{"role": "user", "content": prompt}]}
            payload.update(extra)   # per-provider extras (e.g. kimi thinking-disabled)
            if name == "kimi":
                payload["temperature"] = 0.6; payload["thinking"] = {"type": "disabled"}
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {env.get(keyvar,'')}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                reply = json.loads(r.read().decode())["choices"][0]["message"]["content"]
            print(reply, flush=True)
            nc = _harvest_candidates(reply, "<taxonomy-review>", name, "taxonomy", marker="MISSING")
            if nc: print(f"\n[harvested {nc} MISSING-tell candidate(s) -> {_CAND_PATH.name}; review with `deai.py candidates`]", flush=True)
        except Exception as e:
            print(f"[ERROR {type(e).__name__}: {str(e)[:200]}]", flush=True)

# ════════════════ register profile — describe a corpus's "tone" one level up (Biber/Hyland) ════════════════
# Biber's Multi-Dimensional analysis (1988): reduce many features to a few INTERPRETABLE dimensions (informational
# vs involved, abstraction, persuasion, formality) and read a register's profile off them. Hyland adds the
# metadiscourse dimension. This computes those dimensions over a pluggable corpus — its "tone" as meta-attributes —
# so a corpus is both (1) a way to learn what generalizes across registers and (2) a target tone to conform to.
def register_profile(targets):  # renamed: must NOT shadow the stylometric profile(text) used by mirror/infer
    def card(d):
        docs = [c for _, c in _read_corpus([d])]
        if not docs: return None
        rf = [feats_regex(x) for x in docs]; sf = [feats_syntax(x)[0] for x in docs]
        hf = [h for h in (heading_feats(x) for x in docs) if h]
        M = lambda L, k: round(st.median([x[k] for x in L]), 2) if L else 0.0
        return dict(n=len(docs),
            h_case=M(hf, "title_case"), h_colon=M(hf, "colon"), h_words=M(hf, "mean_words"), h_compound=M(hf, "compound"),
            info_density=round(M(sf, "nominal_rate") + M(sf, "noun_rate") / 5 + M(rf, "that_density"), 1),
            involvement=round(M(rf, "first_person") + M(sf, "pron_rate") / 10, 1),
            metadiscourse=round(M(rf, "meta_narration") + M(sf, "metanarr_synt") + M(rf, "hedge_modal") / 5, 1),
            stance=round(M(rf, "vague_eval_adj") + M(rf, "intensifier") + M(rf, "not_but"), 1),
            abstraction=round(M(rf, "abstract_subj") + M(sf, "nominal_rate") / 10, 1),
            rhythm_cv=M(rf, "burstiness_CV"), mean_sent=M(rf, "mean_sent"))
    cards = [(t, card(t)) for t in targets]; cards = [(t, c) for t, c in cards if c]
    for t, c in cards:
        hc = "sentence-case" if c["h_case"] < 0.3 else ("Title Case" if c["h_case"] > 0.6 else "mixed-case")
        hs = "content-announcing" if c["h_compound"] > 0.3 else "topic-naming"
        rh = "varied" if c["rhythm_cv"] > 0.4 else "uniform"
        print(f"\n# REGISTER PROFILE — {t}  (n={c['n']})   [Biber/Hyland dimensions — the corpus 'one level up']")
        print(f"  headings        : {hc}, {hs}, colon {c['h_colon']}/heading, ~{c['h_words']} words")
        print(f"  info-density    : {c['info_density']}   (nominalization + noun rate + that-density)")
        print(f"  involvement     : {c['involvement']}   (first-person + pronouns)")
        print(f"  metadiscourse   : {c['metadiscourse']}   (meta-narration + hedging + self-mention)")
        print(f"  stance/eval     : {c['stance']}   (vague adjectives + intensifiers + antithesis)")
        print(f"  abstraction     : {c['abstraction']}   (abstract subjects + nominalization)")
        print(f"  rhythm          : {rh}  (burstiness {c['rhythm_cv']}, mean-sentence {c['mean_sent']})")
    if len(cards) == 2:
        (ta, a), (tb, b) = cards
        print(f"\n# DIFFERENTIAL — {ta.split('/')[-1]} vs {tb.split('/')[-1]} (Δ = first minus second)")
        for k in ("h_case", "info_density", "involvement", "metadiscourse", "stance", "abstraction"):
            print(f"  {k:13} {a[k]:>7} vs {b[k]:>7}   Δ {a[k]-b[k]:+.2f}")

# ════════════════ neural style layer (SOTA: learned contrastive STYLE embedding, optional) ════════════════
# Hand-features are interpretable but only capture what we thought to count. A contrastively-trained STYLE embedding
# (Wegmann et al. 2022, "Same Author or Just Same Topic?") learns the style manifold we did NOT hand-engineer —
# style, NOT topic (a generic/semantic embedding would capture the subject instead). This is the holistic SOTA
# VERDICT ("how far from the human style centroid"); the deterministic layers stay for the actionable fix. It is an
# ADDED decision input, not a replacement. Optional: lazy-imported, the rest of the tool runs without it.
_STYLE = None
def _style_model():
    global _STYLE
    if _STYLE is None:
        from sentence_transformers import SentenceTransformer
        # default = Wegmann Style-Embedding; export DEAI_STYLE_MODEL=gabrielloiseau/LUAR-MUD-sentence-transformers
        # for the exact-SOTA authorship rep (Soto et al. 2024). Both are PRETRAINED — inference only, no training.
        _STYLE = SentenceTransformer(os.environ.get("DEAI_STYLE_MODEL", "AnnaWegmann/Style-Embedding"))
    return _STYLE

def style_vec(text):
    # Aggregate AUTHOR-REPRESENTATIVE chunks (120-word windows), NOT single sentences. Averaging hundreds of
    # sentence vectors collapses every doc onto the global "academic" direction (the naive-impl bug: AUC 0.75 ->
    # 0.93 just by switching to chunks). Style/authorship models (UAR/LUAR, Soto et al. 2024) see "posts", not sentences.
    import numpy as np
    ws = re.sub(r"\([^)]*\)", "", prose(text)).split()
    chunks = [" ".join(ws[i:i + 120]) for i in range(0, len(ws), 120)][:48]
    if not chunks: return None
    embs = np.asarray(_style_model().encode(chunks, normalize_embeddings=True, show_progress_bar=False, batch_size=32))
    v = embs.mean(axis=0); n = float(np.linalg.norm(v))
    return v / n if n else v

def style_score(draft_path, human_dir, ai_dir):
    # SOTA readout (Soto et al. ICLR-2024): few-shot in style space — compare the draft to HUMAN and MACHINE style
    # prototypes, not to a single global mean (that naive 1-class cosine threw away the signal: AUC 0.75 vs 0.96).
    import numpy as np
    hv = [v for v in (style_vec(c) for _, c in _read_corpus([human_dir])) if v is not None]
    av = [v for v in (style_vec(c) for _, c in _read_corpus([ai_dir], minw=200)) if v is not None]
    if not hv or not av:
        print(f"style needs at least one human and one AI document (found {len(hv)} human, {len(av)} AI)"); return
    cos = lambda a, b: float(np.dot(a, b))
    cen = lambda V: (lambda m: m / (np.linalg.norm(m) or 1))(np.mean(V, axis=0))
    # reference separation (leave-one-out 2-prototype) so the verdict comes with its own measured signal strength
    sh = [cos(hv[i], cen(av)) - cos(hv[i], cen([hv[j] for j in range(len(hv)) if j != i])) for i in range(len(hv))]
    sa = [cos(av[i], cen([av[j] for j in range(len(av)) if j != i])) - cos(av[i], cen(hv)) for i in range(len(av))]
    auc = sum((a > h) + 0.5 * (a == h) for a in sa for h in sh) / (len(sa) * len(sh))
    dv = style_vec(pathlib.Path(draft_path).read_text(errors="ignore"))
    margin = cos(dv, cen(av)) - cos(dv, cen(hv))  # >threshold leans AI, <threshold leans human
    thresh = (st.median(sh) + st.median(sa)) / 2
    return margin, thresh, auc, len(hv), len(av)

# ════════════ corpus meta-attributes: "describe language using language" (neuro-symbolic) ════════════
# Three routes to a corpus's meta-definition: (1) KNOWN ALGORITHMS -> register_profile() Biber/Hyland dims
# (deterministic, interpretable); (2) the PRETRAINED model -> style_vec() (a holistic but OPAQUE 512-d vector, not
# labels); (3) an LLM -> human-readable meta-LABELS. The best is neuro-symbolic: feed the LLM the MEASURED dims +
# samples so it describes what is measured, not what it imagines. Computed once at ingestion and cached to
# <corpus>/.register.md, so the meta-attributes become a property of the corpus.
def _llm(prompt, model=None, url=None, keyvar=None, mt=900):
    env = _load_env()
    _du, _dkv, _dm, _ = _PROVIDERS["deepseek"]
    model = model or env.get("DEEPSEEK_MODEL", _dm)
    url = url or env.get("DEEPSEEK_URL", _du); keyvar = keyvar or _dkv
    req = urllib.request.Request(url, data=json.dumps({"model": model, "temperature": 0.3, "max_tokens": mt,
        "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": f"Bearer {env.get(keyvar, '')}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=200) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]

def describe(corpus_dir, cache=True):
    # Produce a WRITING BRIEF, not abstract tells: the macro register + MICRO pointers (the venue's own diction, what
    # it keeps near-zero, concrete structure) + real samples, synthesized by an LLM into an actionable guide an author
    # reads BEFORE drafting (ex-ante priming). Cached to <corpus>/.register.md; the deai-conform SKILL reads it first.
    import collections
    docs = [c for _, c in _read_corpus([corpus_dir])]
    rf = [feats_regex(x) for x in docs]; sf = [feats_syntax(x)[0] for x in docs]
    hf = [h for h in (heading_feats(x) for x in docs) if h]
    M = lambda L, k: round(st.median([x[k] for x in L]), 2) if L else 0.0
    dims = (f"headings: {'sentence-case' if M(hf, 'title_case') < 0.3 else 'Title Case'}, "
            f"{'content-announcing' if M(hf, 'compound') > 0.3 else 'topic-naming'}, ~{M(hf, 'mean_words')} words, colon {M(hf, 'colon')}/heading\n"
            f"info-density: nominalization {M(sf, 'nominal_rate')}, noun-rate {M(sf, 'noun_rate')}, that-density {M(rf, 'that_density')}\n"
            f"involvement: first-person {M(rf, 'first_person')}/1k, pronoun-rate {M(sf, 'pron_rate')}\n"
            f"sentence rhythm: mean {M(rf, 'mean_sent')} words, burstiness(CV) {M(rf, 'burstiness_CV')}, subordination {M(sf, 'subcl_ratio')}\n"
            f"metadiscourse: meta-narration {M(rf, 'meta_narration')}, hedging {M(rf, 'hedge_modal')}")
    # MICRO 1 — diction the venue USES (its own distinctive content words)
    cc = collections.Counter(); ct = 0
    for c in docs:
        ws = re.findall(r"[a-z][a-z-]{4,}", prose(c).lower()); cc.update(ws); ct += len(ws)
    stop = set("which their there about would these other where while have been that this from with into more than then them such your only over most some what when also between because through against during before after under above their being where".split())
    vocab = [w for w, _ in cc.most_common(140) if w not in stop][:25]
    # MICRO 2 — generic-AI texture the venue keeps NEAR-ZERO (so a writer should too)
    avoid = ", ".join(f"{k} {M(rf, k)}" for k in ("emdash", "intensifier", "ai_vocab", "vague_eval_adj", "puffery",
                      "meta_narration", "transition_open", "hedge_phrase", "not_but", "comma_not") if M(rf, k) <= 1.0)
    micro = (f"diction the venue reaches for (its own content words): {', '.join(vocab)}\n"
             f"texture the venue keeps near-zero (match it): {avoid}")
    samp = []
    for c in docs[:6]:
        samp += [x for x in re.split(r"(?<=[.!?])\s+", prose(c)) if 12 <= len(x.split()) <= 38][:2]
    prompt = ("You are a writing coach. Below are MEASURED register statistics for a target venue's corpus (Biber "
              "multidimensional + Hyland metadiscourse), the venue's own characteristic vocabulary, the texture it keeps "
              "near-zero, and real sample sentences. Write a concrete WRITING BRIEF an author follows to draft IN this "
              "venue's voice from the first sentence, MICRO and MACRO, not abstract 'avoid AI tells'. Structure it: "
              "(1) VOICE in one line; (2) DICTION, the kind of words to reach for and to avoid, with examples; "
              "(3) SENTENCE STRUCTURE, length, rhythm, subordination, how citations sit in the syntax; (4) HEADINGS, "
              "case and phrasing; (5) a tight DO and DON'T list at the word-and-construction level. Ground every point "
              f"in the numbers and the samples; be specific.\n\nMEASURED (medians over {len(docs)} texts):\n{dims}\n\n{micro}\n\n"
              "SAMPLE SENTENCES:\n- " + "\n- ".join(samp[:12]))
    try:
        out = _llm(prompt, mt=1400)
    except Exception as e:
        out = f"[LLM unavailable: {type(e).__name__}; the measured register + vocabulary above are the deterministic brief]"
    body = (f"# WRITING BRIEF — {corpus_dir}  (n={len(docs)})\n\n## Measured register (deterministic)\n{dims}\n\n"
            f"## Micro pointers (deterministic)\n{micro}\n\n## Writing brief (LLM, grounded on the measurements)\n{out}\n")
    print(body)
    if cache:
        pathlib.Path(corpus_dir, ".register.md").write_text(body)

# ════════════ probe: EXPLAIN the opaque style embedding via the interpretable dims (neuro-symbolic) ════════════
# The 512-d style vector is opaque, but interpretable. Two standard methods: (1) PROBING CLASSIFIERS (Alain & Bengio
# 2017; Hewitt & Manning 2019) — linear-probe the embedding for each known dimension; R^2 = how much of that
# human-readable property is linearly encoded. (2) CONCEPT DIRECTIONS (TCAV, Kim et al. 2018) — the AI-vs-human
# direction is mean(AI)-mean(human); correlating the projection on it with each dim says what "reads AI" MEANS.
def probe(human_dir, ai_dir):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold
    H = [c for _, c in _read_corpus([human_dir])]; A = [c for _, c in _read_corpus([ai_dir], minw=200)]
    docs = H + A; nh = len(H)
    print(f"# PROBING the opaque style embedding  (n={len(docs)}: {nh} human + {len(A)} AI)\n", flush=True)
    X = np.array([style_vec(d) for d in docs])
    def dims(d):
        rf = feats_regex(d); sf = feats_syntax(d)[0]
        return {"info_density": sf["nominal_rate"] + sf["noun_rate"] / 5 + rf["that_density"],
                "involvement": rf["first_person"] + sf["pron_rate"] / 10,
                "metadiscourse": rf["meta_narration"] + sf["metanarr_synt"] + rf["hedge_modal"] / 5,
                "stance_eval": rf["vague_eval_adj"] + rf["intensifier"] + rf["not_but"],
                "nominalization": sf["nominal_rate"], "pron_rate": sf["pron_rate"],
                "mean_sentence": rf["mean_sent"], "burstiness": rf["burstiness_CV"]}
    D = [dims(d) for d in docs]; keys = list(D[0])
    print("## 1. Linear-probe R² — how much of each interpretable dimension the embedding ENCODES (5-fold ridge)")
    for k in keys:
        y = np.array([d[k] for d in D])
        r2 = cross_val_score(Ridge(alpha=1.0), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
        print(f"  {k:15} R² = {max(r2, 0.0):.2f}")
    w = X[nh:].mean(0) - X[:nh].mean(0); w = w / (np.linalg.norm(w) or 1)
    proj = X @ w
    print("\n## 2. The AI-vs-human direction, explained — Pearson r of (projection on the AI-axis) with each dim")
    print("   (positive r = this dimension rises as the text moves toward the AI side)")
    rows = sorted(((float(np.corrcoef(proj, [d[k] for d in D])[0, 1]), k) for k in keys), reverse=True)
    for r, k in rows:
        print(f"  {k:15} r = {r:+.2f}")

# ════════════ verdict: fuse the signals the WORLD-CLASS way — supervised STACKING, not PCA ════════════
# PCA is wrong for the verdict (unsupervised: maximizes variance incl. topic/length, not AI-discriminativeness).
# The right fusion is STACKING (Wolpert 1992): each signal is a feature, a cross-validated meta-logistic learns the
# weighting and HANDLES REDUNDANCY (correlated signals share weight, no double-counting). The probe is NOT an input
# here — it is the explainer. The deterministic tells + register dims stay SEPARATE for the FIX; this is the one
# calibrated verdict. Signals fused: function-word Δ, POS-bigram Δ, deterministic flag-count, neural style margin.
def verdict(draft_path, human_dir, ai_dir):
    # ROLE-SEPARATED, not one blended number. The naive stack conflates two DIFFERENT questions and inherits the
    # benchmark's "AI = off-corpus" confound (it called a human off-venue paper 99% AI). So we report TWO things,
    # each with leave-one-out (no centroid self-leakage):
    #   (1) AI-LIKELIHOOD = the neural margin (contrastively trained on AI-vs-human; venue-ROBUST).
    #   (2) VENUE-CONFORMANCE = function-word + POS-bigram Δ (distance from THIS corpus; venue/topic-CONFOUNDED, so a
    #       human paper from another venue also scores far). This is the de-AI EDITING target, NOT an AI verdict.
    import numpy as np
    H = [c for _, c in _read_corpus([human_dir])]; A = [c for _, c in _read_corpus([ai_dir], minw=200)]
    print(f"# VERDICT  (n={len(H) + len(A)}: {len(H)} human + {len(A)} AI)\n", flush=True)
    hv = [style_vec(c) for c in H]; av = [style_vec(c) for c in A]
    if not hv or not av:
        print(f"verdict needs at least one human and one AI document (found {len(hv)} human, {len(av)} AI)"); return
    cos = lambda a, b: float(np.dot(a, b)); cen = lambda V: (lambda m: m / (np.linalg.norm(m) or 1))(np.mean(V, 0))
    sh = [cos(hv[i], cen(av)) - cos(hv[i], cen([hv[j] for j in range(len(hv)) if j != i])) for i in range(len(hv))]
    sa = [cos(av[i], cen([av[j] for j in range(len(av)) if j != i])) - cos(av[i], cen(hv)) for i in range(len(av))]
    nauc = sum((a > h) + 0.5 * (a == h) for a in sa for h in sh) / (len(sa) * len(sh))
    thr = (st.median(sh) + st.median(sa)) / 2
    cfw = [profile(c)[0] for c in H]; cbg = [feats_syntax(c)[1] for c in H]
    h_fwd, h_posd = [], []  # human leave-one-out conformance distances
    for i in range(len(H)):
        rest = [cfw[j] for j in range(len(H)) if j != i]
        m = {fw: st.mean(p[fw] for p in rest) for fw in FW}; s = {fw: (st.pstdev([p[fw] for p in rest]) or 1e-9) for fw in FW}
        h_fwd.append(delta(cfw[i], m, s)[0]); h_posd.append(posbg_delta(cbg[i], [cbg[j] for j in range(len(H)) if j != i])[0])
    mean = {fw: st.mean(p[fw] for p in cfw) for fw in FW}; sd = {fw: (st.pstdev([p[fw] for p in cfw]) or 1e-9) for fw in FW}
    d = pathlib.Path(draft_path).read_text(errors="ignore")
    d_marg = cos(style_vec(d), cen(av)) - cos(style_vec(d), cen(hv))
    d_fwd = delta(profile(d)[0], mean, sd)[0]; d_posd = posbg_delta(feats_syntax(d)[1], cbg)[0]
    print(f"## (1) AI-LIKELIHOOD  — neural style margin, venue-robust  [reference AUC {nauc:.2f}]")
    print(f"  draft margin {d_marg:+.3f}  (AI threshold {thr:+.3f})  ->  reads {'AI' if d_marg > thr else 'HUMAN'}")
    print(f"\n## (2) VENUE-CONFORMANCE  — distance from THIS corpus (confounded by venue/topic; NOT an AI verdict)")
    fw_hi, pos_hi = _robust_hi(h_fwd), _robust_hi(h_posd)
    print(f"  function-word Δ {d_fwd:.2f}  corpus LOO cutoff {fw_hi:.2f}   {'CONFORMS' if d_fwd <= fw_hi else 'OFF-VENUE'}")
    print(f"  POS-bigram   Δ {d_posd:.2f}  corpus LOO cutoff {pos_hi:.2f}   {'CONFORMS' if d_posd <= pos_hi else 'OFF-VENUE'}")
    print(f"\n  >>> the two questions are independent: a draft can read HUMAN yet be OFF-VENUE (wrong corpus), or read AI yet conform.")

# ════════════════════════════ topic-controlled lexical over-use (excess) ════════════════════════════
# The principled upgrade of infer()'s open-ended discovery. Monroe, Colaresi & Quinn (2008) weighted log-odds with
# an informative Dirichlet prior -> a VARIANCE-ADJUSTED z for "which words does A over-use vs B" (a word seen 3x by
# chance is shrunk, unlike a raw ratio). Topic is controlled the world-class way (Wegmann 2022; Kobak 2024 excess
# vocabulary): the AI reference is generated ON THE SAME TOPICS as the human corpus, so generic-AI slop ("harden")
# separates from topic words. wordfreq splits common-word slop from rare jargon. Two channels:
#   generic_ai_slop      candidate de-slop list (draft over-uses vs human AND AI over-uses vs human on MATCHED topics
#                        = pure style AND common word). NOT auto-cut: the LLM triages gratuitous vs load-bearing use.
#   draft_overrepetition terms the draft leans on even vs AI on the SAME topic (vary if not a core term).
# Honest limit: when the paper's SUBJECT vocabulary IS the AI-abstraction register (architecture/structural/
# institutional/protocol), the slop channel cannot separate the two; the over-repetition channel stays reliable.
_LEX_STOP = set(("the a an of to in and or for on with as is are was were be been being by that this these those it its "
    "their they them we our us you he she his her at from into than then so but not no nor only also both each more most "
    "other some such which who whom whose what when where why how all any can could may might must shall should will would "
    "do does did done has have had having if because while between within without about over under above below up down out "
    "off again further once here there very just i me my mine your yours one two three first second").split())
def _ctoks(t):
    t = re.sub(r"<!--.*?-->", "", t, flags=re.DOTALL)
    t = re.sub(r"[*`#>\[\]()]|\\[a-z]+", " ", t)
    out = []
    for w in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", t.lower()):
        if w in _LEX_STOP: continue
        for suf in ("ization", "izing", "ized", "izes", "ness", "ing", "ed", "es", "ly", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3: w = w[:-len(suf)]; break
        out.append(w)
    return out
def _ccounts(texts):
    c = collections.Counter()
    for t in texts: c.update(_ctoks(t))
    return c
def _monroe(yi, yj, a0=1000.0):
    """Fightin' Words weighted log-odds w/ informative Dirichlet prior. z>0 => over-used in yi vs yj."""
    import math as _m
    vocab = set(yi) | set(yj); ni, nj = sum(yi.values()), sum(yj.values()); nbg = ni + nj
    z = {}
    if not nbg: return z
    for w in vocab:
        aw = a0 * (yi.get(w, 0) + yj.get(w, 0)) / nbg
        if aw <= 0: continue
        di = _m.log(yi.get(w, 0) + aw) - _m.log(ni + a0 - yi.get(w, 0) - aw)
        dj = _m.log(yj.get(w, 0) + aw) - _m.log(nj + a0 - yj.get(w, 0) - aw)
        z[w] = (di - dj) / _m.sqrt(1.0 / (yi.get(w, 0) + aw) + 1.0 / (yj.get(w, 0) + aw))
    return z
try:
    from wordfreq import zipf_frequency as _zipf_freq
    def _zipf(w): return _zipf_freq(w, "en")
    _HAVE_WF = True
except Exception:
    def _zipf(w): return 3.5
    _HAVE_WF = False
def _lex_env():
    return _load_env()
_LEX_MODELS = _roster()
def _lex_prompt(topic):
    return ("Write a single ~450-word passage from the body of a scholarly journal article on the following topic. "
            "Use the measured register of an academic paper: continuous prose, no headings, no bullet lists, no title. "
            f"Topic: {topic}.")
def _lex_call(env, m, prompt):
    name, url, kv, model, extra = m
    if not env.get(kv): return None
    payload = {"model": model, "max_tokens": 800, "messages": [{"role": "user", "content": prompt}]}; payload.update(extra)
    if name == "kimi": payload["temperature"] = 0.6   # kimi spends the budget on reasoning_content at default temp
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {env[kv]}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=150) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]
    except Exception: return None
def _lex_topic_of(text):
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            h = line.lstrip("#").strip()
            if h and h.lower() != "abstract": return h[:200]
    return " ".join(re.findall(r"\S+", re.sub(r"\s+", " ", re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)))[:25])[:200]
def lex_gen_reference(env, draft_topic, corpus_dir, cache_dir):
    """(ai_on_draft_topic, ai_matched_to_corpus_topics), cached. on=all models (over-repetition ref); matched=deepseek
    per corpus paper for speed (the topic-MATCHED AI-style ref)."""
    cd = pathlib.Path(cache_dir); cd.mkdir(parents=True, exist_ok=True)
    def slot(key, prompt, models):
        f = cd / (key + ".json")
        if f.exists():
            try: return json.loads(f.read_text())
            except Exception: pass
        ess = [e for e in (_lex_call(env, m, prompt) for m in models) if e]
        if ess: f.write_text(json.dumps(ess))   # don't cache a failed (empty) generation
        return ess
    on = []
    for i in range(2): on += slot(f"on_{i}", _lex_prompt(draft_topic) + f" (variation {i + 1})", _LEX_MODELS)
    ds = [m for m in _LEX_MODELS if m[0] == "deepseek"] or _LEX_MODELS[:1]
    matched = []
    for i, p in enumerate(sorted(pathlib.Path(corpus_dir).glob("*.md"))):
        matched += slot(f"matched_{i}", _lex_prompt(_lex_topic_of(p.read_text(errors="ignore"))), ds)
    return on, matched
def lexical_overuse(draft_text, human_texts, ai_on, ai_matched, min_draft=3, z_hi=1.96, z_med=1.5, common_zipf=3.0):
    D, H, Ad, Am = _ccounts([draft_text]), _ccounts(human_texts), _ccounts(ai_on), _ccounts(ai_matched)
    z_DH = _monroe(D, H); z_AmH = _monroe(Am, H) if Am else {}; z_DAd = _monroe(D, Ad) if Ad else {}
    slop, overrep = [], []
    for w, dc in D.items():
        if dc < min_draft: continue
        zdh, zamh, zdad = z_DH.get(w, 0), z_AmH.get(w, 0), z_DAd.get(w, 0)
        if Am and zdh > z_hi and zamh > z_med and _zipf(w) >= common_zipf:
            slop.append((round(zdh, 1), round(zamh, 1), w, dc))
        elif Ad and zdad > z_hi and dc >= 5:
            overrep.append((round(zdad, 1), w, dc))
    slop.sort(reverse=True); overrep.sort(reverse=True)
    return {"generic_ai_slop": slop, "draft_overrepetition": overrep, "wordfreq": _HAVE_WF,
            "n_ai_on": len(ai_on), "n_ai_matched": len(ai_matched)}
def excess(draft_path, corpus_dir, topic, cache_dir=None):
    cache_dir = cache_dir or str(pathlib.Path(__file__).resolve().parent / ".lexcache")
    env = _lex_env()
    human = [p.read_text(errors="ignore") for p in pathlib.Path(corpus_dir).glob("*.md")]
    draft = pathlib.Path(draft_path).read_text(errors="ignore")
    on, matched = lex_gen_reference(env, topic, corpus_dir, cache_dir)
    r = lexical_overuse(draft, human, on, matched)
    print(f"# EXCESS — {pathlib.Path(draft_path).name}  (Monroe weighted log-odds; topic-MATCHED AI style ref; wordfreq={r['wordfreq']})")
    print(f"  refs: {r['n_ai_on']} AI-on-draft-topic + {r['n_ai_matched']} AI-matched-to-corpus-topics essays")
    print("\n  GENERIC-AI SLOP CANDIDATES (LLM triages gratuitous vs load-bearing) [zDraft, zAIstyle, word, draft#]:")
    for row in r["generic_ai_slop"][:20]: print("   ", row)
    print("\n  DRAFT OVER-REPETITION (vs AI on the SAME topic; vary if not a core term) [z, word, draft#]:")
    for row in r["draft_overrepetition"][:15]: print("   ", row)
    return r

# ════════════ corpus alignment, SELF-EXPLAINING: fixed numeric axes + EMERGENT semantic profile ════════════
# Design (per the 2026-06-24 spec): two layers with OPPOSITE label policies, on purpose.
#  Layer 1 — DETERMINISTIC, fixed axes: every surface/syntactic feature is GLOSSED in plain language (what it
#    measures + why it reads AI) and read against the VENUE's own distribution. Same axes for every corpus, so
#    venues are comparable and the benefit compounds across drafts.
#  Layer 2 — SEMANTIC, EMERGENT per-corpus: an LLM INGESTS the corpus and INDUCES the categories that characterize
#    THIS venue's tone and style (labels need NOT match across corpora), each with prose on what the venue DOES and
#    DOES NOT do, and an intensity score WHERE meaningful (else null). Then the draft is checked against the venue's
#    OWN categories. Cached to <corpus>/.style_profile.{json,md}, so the fingerprint is a durable corpus property.

FEATURE_GLOSS = {  # feature -> (display name, what it measures, in plain language)
    "mean_sent": ("Average sentence length", "mean words per sentence"),
    "burstiness_CV": ("Burstiness", "how sharply sentence length swings short-to-long; humans vary a lot and averaged model prose flattens, so a LOW value is the tell"),
    "pct_short": ("Short-sentence share", "percent of sentences under 8 words (the staccato a model rarely uses)"),
    "emdash": ("Em-dashes", "em-dash rate per 1000 words"),
    "semicolon": ("Semicolons", "semicolon rate per 1000 words"),
    "comma_not": ("Comma-'not' antithesis", "'X, not Y' balanced contrasts"),
    "rather_than": ("'rather than' antithesis", "'rather than' contrasts"),
    "not_but": ("'not X but Y' antithesis", "the full balanced-antithesis frame"),
    "triadic": ("Triadic lists", "three-item 'A, B, and C' lists used as a default rhythm"),
    "compound_and": ("Compound-clause density", "', and' clause joins per 1000 words"),
    "appositive": ("Comma-appositive restatement", "a noun restated by a comma phrase, the em-dash's quiet replacement"),
    "what_cleft": ("It-/what-clefts", "'what matters is', 'it is X that' emphasis-fronting"),
    "abstract_subj": ("Abstract-subject openers", "sentences opening on an abstract nominal subject ('The tension is...')"),
    "para_open_The": ("'The'-opener density", "sentences that start with 'The'"),
    "demonstr_abstract": ("Demonstrative + abstract noun", "'this failure', 'that tension' pointing at an abstraction"),
    "which_is_tail": ("'which is/means' tails", "trailing relative clauses that re-explain a noun"),
    "the_way_each": ("'the way each' framing", "'the way that', 'the kind that' framing"),
    "to_inf_purpose": ("Purpose-infinitive insets", "', to X,' purpose clauses set off by commas"),
    "intensifier": ("Intensifiers", "precisely/exactly/itself/indeed/merely as emphasis"),
    "that_density": ("'that' density", "frequency of 'that', a marker of complement-heavy formal syntax"),
    "first_person": ("First-person (we/our/us)", "author presence; venues vary and a LOW value can read detached"),
    "meta_narration": ("Meta-narration", "narrating the argument's own moves/status instead of making them"),
    "ai_vocab": ("Generic-AI vocabulary", "delve/leverage/robust/tapestry/nuanced/realm"),
    "vague_eval_adj": ("Vague evaluative adjectives", "profound/meaningful/crucial/significant: importance asserted, not shown (the single strongest discriminator)"),
    "transition_open": ("Transition signposting", "Furthermore/Moreover/However as sentence openers"),
    "hedge_phrase": ("Hollow hedge phrases", "'it is important to note', 'needless to say'"),
    "hedge_modal": ("Modal hedging", "may/might/could/would density"),
    "puffery": ("Puffery", "'plays a crucial role', 'is a testament to'"),
    "disclaimer": ("Scope-disclaimer / underclaiming", "repeated 'existence proof not validation', 'we make no claim' (over-honesty, the opposite tell to puffery)"),
}
_LOWER_IS_TELL = {"burstiness_CV", "mean_sent", "first_person"}  # for these, a value BELOW the venue is the flag

def describe_alignment(draft_path, corpus_dir):
    """Layer 1: fixed deterministic axes, each glossed and read against the venue's own distribution (p25-p75)."""
    draft = feats_regex(re.sub(r"<!--.*?-->", "", pathlib.Path(draft_path).read_text(errors="ignore"), flags=re.DOTALL))
    corp = [feats_regex(c) for _, c in _read_corpus([corpus_dir])]
    if not corp:
        print("corpus empty or no documents over the word threshold"); return []
    n = len(corp)
    print(f"\n## Surface & syntactic alignment (fixed axes, read against this venue; n={n})\n")
    flagged, inband = [], []
    for k, (name, what) in FEATURE_GLOSS.items():
        if k not in draft: continue
        vals = sorted(c[k] for c in corp); med = st.median(vals)
        p25, p75 = pctl(vals, 25), pctl(vals, 75); v = draft[k]
        if k in _LOWER_IS_TELL: tell = v < 0.7 * med; rel = "below"
        else: tell = v > 1.6 * med + 0.2; rel = "above"
        (flagged if tell else inband).append((k, name, what, med, p25, p75, v, rel))
    if not flagged:
        print("  All axes sit inside the venue's range. The draft's surface reads in-register.\n")
    for k, name, what, med, p25, p75, v, rel in flagged:
        x = f"{round(v/med,1)}x" if med else "much"
        print(f"  • {name} ({what}). Venue runs ~{round(med,2)} (typical {round(p25,2)}-{round(p75,2)}); "
              f"the draft is {v}, {x} the venue median, {rel} the band. "
              + ("This axis being below the venue is the AI read here." if k in _LOWER_IS_TELL
                 else "Denser than this venue tends to write, which reads more machine-made on this axis.") + "\n")
    if inband:
        print("  In-register (matches the venue): " + ", ".join(f"{FEATURE_GLOSS[k][0]} {v}" for k, _, _, _, _, _, v, _ in inband) + ".\n")
    return flagged

def _json_from(txt):
    """Pull the first JSON object/array out of an LLM reply (handles ```json fences and prose wrappers)."""
    txt = re.sub(r"```(?:json)?", "", txt).strip().strip("`")
    m = re.search(r"(\{.*\}|\[.*\])", txt, re.DOTALL)
    if not m: raise ValueError("no JSON found in model reply")
    return json.loads(m.group(1))

def _corpus_samples(corpus_dir, n_docs=10, per=3, lo=12, hi=40):
    docs = [c for _, c in _read_corpus([corpus_dir])]
    samp = []
    for c in docs[:n_docs]:
        ss = [x.strip() for x in re.split(r"(?<=[.!?])\s+", prose(c)) if lo <= len(x.split()) <= hi]
        samp += ss[:per]
    return docs, samp

def style_profile(corpus_dir, model=None, cache=True, refresh=False):
    """EMERGENT per-corpus semantic fingerprint: induce the categories that characterize THIS venue's tone/style,
    each with does / does-not prose + an optional intensity. Cached as a durable property of the corpus."""
    model = model or _load_env().get("DEEPSEEK_MODEL", "deepseek-chat")
    cf = pathlib.Path(corpus_dir, ".style_profile.json")
    if cache and cf.exists() and not refresh:
        return json.loads(cf.read_text())
    docs, samp = _corpus_samples(corpus_dir)
    rf = [feats_regex(x) for x in docs]; M = lambda k: round(st.median([x[k] for x in rf]), 2)
    measured = (f"mean-sentence {M('mean_sent')}w, burstiness {M('burstiness_CV')}, first-person {M('first_person')}/1k, "
                f"modal-hedging {M('hedge_modal')}, meta-narration {M('meta_narration')}, vague-adj {M('vague_eval_adj')}, "
                f"generic-AI-vocab {M('ai_vocab')}, em-dash {M('emdash')}, semicolon {M('semicolon')}")
    prompt = ("You are a register and style analyst. Below are real sample sentences from ONE academic venue's corpus, "
              "plus a few measured statistics. INDUCE the 6 to 9 CATEGORIES that best characterize THIS venue's tone and "
              "style. Choose whatever dimensions are genuinely distinctive here (e.g. stance and certainty, hedging, "
              "rhetorical/argument structure, diction, sentence rhythm, how evidence and citations are used, normative vs "
              "descriptive framing, reader engagement, affect). Categories are SPECIFIC to this corpus and need NOT be a "
              "fixed or standard list. For EACH category give: name; does (what this venue DOES on it, 1-2 concrete "
              "sentences); does_not (what it avoids); intensity (0-100 for how strongly the venue exhibits it, or null if a "
              "number is not meaningful); example (a short real quote, or empty). Ground every claim in the samples and "
              "stats. Reply ONLY with JSON: {\"venue_voice\":\"<one sentence>\",\"categories\":[{\"name\":...,\"does\":...,"
              "\"does_not\":...,\"intensity\":<0-100 or null>,\"example\":\"...\"}]}.\n\n"
              f"MEASURED (medians, n={len(docs)}): {measured}\n\nSAMPLE SENTENCES:\n- " + "\n- ".join(samp[:18]))
    prof = _json_from(_llm(prompt, model=model, mt=2000))
    prof["_n_docs"] = len(docs); prof["_corpus"] = str(corpus_dir)
    try:   # PROB_AI human baseline is a corpus metric too: compute once at ingestion, cache, fold into the profile
        prof["prob_ai_baseline"] = prob_ai_baseline(corpus_dir, quiet=True)["band"]
    except Exception as e:
        prof["prob_ai_baseline"] = None
    if cache:
        cf.write_text(json.dumps(prof, indent=2))
        md = [f"# STYLE PROFILE (emergent) — {corpus_dir}  (n={len(docs)})\n", f"**Venue voice:** {prof.get('venue_voice','')}\n"]
        for c in prof.get("categories", []):
            sc = f"  _(intensity {c['intensity']}/100)_" if c.get("intensity") is not None else "  _(no score)_"
            md.append(f"### {c['name']}{sc}\n- **Does:** {c.get('does','')}\n- **Does not:** {c.get('does_not','')}"
                      + (f"\n- _e.g._ \"{c['example']}\"" if c.get("example") else ""))
        pathlib.Path(corpus_dir, ".style_profile.md").write_text("\n".join(md) + "\n")
    return prof

# --- PROB_AI human baseline: a venue's absolute AI-probability SATURATES for dense academic prose (judge() says so),
#     so the bare number is uninterpretable. It only means something against the venue's OWN human corpus. We score
#     each human doc once at ingestion, cache to <corpus>/.prob_ai_baseline.jsonl, and report the band — a draft is
#     "above the humans" iff it scores past the band. Computed mechanically, persisted, never re-derived by hand. ---
def _deai_env():
    return _load_env()

def _band(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals: return None
    q = lambda p: vals[min(int(round(p * (len(vals) - 1))), len(vals) - 1)]
    return {"n": len(vals), "min": vals[0], "p25": q(.25), "median": q(.5), "p75": q(.75), "max": vals[-1]}

def _ai_prob_doc(env, text, only=None):
    """Holistic PROB_AI (0-100) for ONE document from each judge (deepseek/grok/kimi)."""
    judges = _roster()
    if only: judges = [j for j in judges if j[0] in only] or judges
    body = prose(re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL))
    out = {}
    for name, url, keyvar, model, extra in judges:
        chunk = _fit_or_chunk(body, model)[0]   # first chunk: a holistic read that bounds cost
        prompt = ("You are a forensic linguist scoring whether ACADEMIC prose reads AI-generated. Give ONE integer "
                  "0-100: the probability this was AI-generated (0 = clearly a human scholar, 100 = clearly AI). "
                  "Judge holistically. Reply with ONLY 'PROB_AI: <0-100>'.\n\n" + chunk)
        try:
            payload = {"model": model, "temperature": 0.2, "max_tokens": 24, "messages": [{"role": "user", "content": prompt}]}
            payload.update(extra)   # per-provider extras (e.g. kimi thinking-disabled)
            if name == "kimi": payload["temperature"] = 0.6; payload["thinking"] = {"type": "disabled"}; payload["max_tokens"] = 64
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {env.get(keyvar, '')}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                reply = json.loads(r.read().decode())["choices"][0]["message"]["content"]
            m = re.search(r"PROB_AI:\s*(\d+)", reply) or re.search(r"\b(\d{1,3})\b", reply)
            out[name] = max(0, min(100, int(m.group(1)))) if m else None
        except Exception:
            out[name] = None
    return out

def prob_ai_baseline(corpus_dir, only=None, refresh=False, env=None, quiet=False):
    """Score each human doc in a venue corpus once, cache to <corpus>/.prob_ai_baseline.jsonl, return the per-judge
    band. Resumable: only un-cached docs are scored, so re-running costs nothing."""
    env = env if env is not None else _deai_env()
    cache = pathlib.Path(corpus_dir, ".prob_ai_baseline.jsonl")
    done = {}
    if cache.exists() and not refresh:
        for ln in cache.read_text(errors="ignore").splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r["file"].startswith(".") or r["file"].lower() in ("readme.md", "index.md"): continue
                done[r["file"]] = r
    for p in sorted(pathlib.Path(corpus_dir).glob("*.md")):
        if p.name.startswith(".") or p.name.lower() in ("readme.md", "index.md"): continue  # skip caches/meta, not corpus docs
        if p.name in done: continue
        sc = _ai_prob_doc(env, p.read_text(errors="ignore"), only)
        if not any(v is not None for v in sc.values()):   # every judge failed: don't cache, leave it to be re-scored
            if not quiet: print(f"  baseline {p.name}: all judges failed, not cached", flush=True)
            continue
        row = {"file": p.name, **sc}
        with cache.open("a") as f: f.write(json.dumps(row) + "\n")
        done[p.name] = row
        if not quiet: print(f"  baseline {p.name}: {sc}", flush=True)
    band = {k: _band([r.get(k) for r in done.values()]) for k in ("deepseek", "grok", "kimi")}
    return {"corpus": str(corpus_dir), "n": len(done), "band": band}

def place_in_baseline(draft_path, corpus_dir, only=None):
    """Score a draft and place it against the venue's cached human band."""
    env = _deai_env()
    base = prob_ai_baseline(corpus_dir, only=only, env=env, quiet=True)["band"]
    sc = _ai_prob_doc(env, pathlib.Path(draft_path).read_text(errors="ignore"), only)
    return sc, base

# Multi-judge panel for ENSEMBLE scoring (reliability): LLM style-labeling is volatile (Rating Roulette, EMNLP 2025;
# Nature 2025), so the draft is scored by 3 independent judges and we report Krippendorff's alpha across them. kimi
# needs thinking DISABLED + temp 0.6 or it spends the budget on reasoning_content and returns empty (known bug).
_JUDGES = _roster()

def _judge_call(prompt, spec, mt=1700, temp=None):
    name, url, keyvar, model, extra = spec
    env = _load_env()
    payload = {"model": model, "temperature": (0.6 if name == "kimi" else 0.3) if temp is None else temp, "max_tokens": mt,
               "messages": [{"role": "user", "content": prompt}], **extra}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {env.get(keyvar, '')}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]

def _kripp_interval(rows):
    """Krippendorff's alpha, interval metric, across judges. rows: per-unit lists of ratings (None = missing).
    alpha=1 perfect agreement, 0 chance, <0 systematic disagreement. Pairwise within-unit vs global spread."""
    import itertools
    do = dn = 0.0; allv = []
    for r in rows:
        vals = [v for v in r if v is not None]; allv += vals
        for a, b in itertools.combinations(vals, 2): do += (a - b) ** 2; dn += 1
    if dn == 0 or len(allv) < 2: return None
    Do = do / dn
    de = dd = 0.0
    for a, b in itertools.combinations(allv, 2): de += (a - b) ** 2; dd += 1
    De = de / dd if dd else 0.0
    return None if De == 0 else round(1 - Do / De, 3)

def style_align(draft_path, corpus_dir, refresh=False):
    """ENSEMBLE Layer-2 check: induce the venue profile once, then score the draft against its categories with EACH of
    the 3 judges, returning per-category median alignment + inter-judge Krippendorff alpha (the reliability number)."""
    prof = style_profile(corpus_dir, refresh=refresh)
    body = prose(re.sub(r"<!--.*?-->", "", pathlib.Path(draft_path).read_text(errors="ignore"), flags=re.DOTALL))
    cats = "\n".join(f"- {c['name']}: DOES {c.get('does','')} || DOES NOT {c.get('does_not','')}" for c in prof.get("categories", []))
    prompt = ("You are checking whether a DRAFT matches a target venue's style, using the venue's OWN induced style "
              "categories below (each says what the venue DOES and DOES NOT do). For EACH category, judge how the draft "
              "compares: does it do what the venue does, or what the venue avoids? Give alignment (0-100, where 100 = fully "
              "in the venue's register on that category, or null if not applicable), one sentence of explanation, and a short "
              "quoted draft span as evidence. Reply ONLY with JSON: {\"categories\":[{\"name\":...,\"alignment\":<0-100 or "
              "null>,\"verdict\":\"<one sentence>\",\"evidence\":\"<draft span>\"}],\"summary\":\"<2 sentences: where the draft "
              "is in-register and where it drifts>\"}.\n\nVENUE STYLE CATEGORIES:\n" + cats + "\n\n=== DRAFT PROSE ===\n" + body[:22000])
    per_judge, verdicts, summaries = {}, {}, []
    for spec in _JUDGES:
        try:
            out = _json_from(_judge_call(prompt, spec))
        except Exception as e:
            print(f"  [judge {spec[0]} unavailable: {type(e).__name__}: {str(e)[:80]}]", flush=True); continue
        per_judge[spec[0]] = {c["name"]: c.get("alignment") for c in out.get("categories", [])}
        for c in out.get("categories", []):
            verdicts.setdefault(c["name"], []).append((spec[0], c.get("verdict", ""), c.get("evidence", "")))
        if out.get("summary"): summaries.append((spec[0], out["summary"]))
    names = [c["name"] for c in prof.get("categories", [])]
    agg, rows = {}, []
    for n in names:
        row = [per_judge[j].get(n) for j in per_judge]; rows.append(row)
        sc = [v for v in row if v is not None]
        agg[n] = {"scores": {j: per_judge[j].get(n) for j in per_judge},
                  "median": round(st.median(sc), 1) if sc else None,
                  "spread": (max(sc) - min(sc)) if len(sc) > 1 else 0, "verdicts": verdicts.get(n, [])}
    return prof, {"per_category": agg, "alpha": _kripp_interval(rows),
                  "n_judges": len(per_judge), "judges": list(per_judge), "summaries": summaries}

def register(draft_path, corpus_dir, no_llm=False):
    """Full self-explaining corpus-alignment report: Layer 1 (deterministic, glossed) + Layer 2 (emergent, 3-judge ensemble)."""
    print(f"# CORPUS ALIGNMENT — {pathlib.Path(draft_path).name}  vs  {corpus_dir}")
    describe_alignment(draft_path, corpus_dir)
    if no_llm:
        print("## Semantic register — profile skipped (--no-llm)\n"); return
    try:
        prof, al = style_align(draft_path, corpus_dir)
    except Exception as e:
        print(f"## Semantic register — profile unavailable [{type(e).__name__}: {str(e)[:160]}]\n"); return
    nj = al["n_judges"]; a = al["alpha"]
    print(f"\n## Semantic register — emergent tone & style categories ({nj}-judge ensemble: {', '.join(al['judges'])})\n")
    print(f"  Venue voice: {prof.get('venue_voice','')}\n")
    band = ("high agreement, scores reliable" if a is not None and a >= 0.8 else
            "moderate agreement" if a is not None and a >= 0.5 else
            "LOW agreement, treat scores as directional" if a is not None else "n/a (need >=2 judges)")
    print(f"  Inter-judge agreement: Krippendorff alpha = {a if a is not None else 'n/a'}  ({band})\n")
    for c in prof.get("categories", []):
        g = al["per_category"].get(c["name"], {})
        sc = "  ".join(f"{j}:{v}" for j, v in g.get("scores", {}).items() if v is not None) or "no scores"
        print(f"  ▸ {c['name']}  [median {g.get('median')}/100 | {sc} | spread {g.get('spread')}]")
        print(f"      venue does:     {c.get('does','')}")
        print(f"      venue does not: {c.get('does_not','')}")
        for j, verdict, ev in g.get("verdicts", [])[:1]:
            if verdict: print(f"      draft ({j}): {verdict}" + (f"  (e.g. \"{ev}\")" if ev else ""))
        print()
    for j, s in al.get("summaries", [])[:1]:
        print(f"  SUMMARY ({j}): {s}\n")

# ════════════ DENSITY & CLARITY (neurosymbolic): the AI-slop axis the register checks miss ════════════
# Dense + repetitive prose is a documented machine/low-quality tell and a clarity problem. Deterministic surface
# proxies + per-section readability + inter-section redundancy; an LLM layer (added separately) judges whether the
# density is load-bearing or gratuitous. Research basis: Flesch (1948) reading ease; Kincaid et al. (1975) grade;
# Gunning (1952) Fog; Halliday & Ure lexical density (content-word ratio); Biber nominalization; Kintsch-Turner /
# CPIDR idea density; Coh-Metrix (McNamara & Graesser) cohesion; Kobak et al. (2024) excess words.
def _syll(w):
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w: return 1
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1: n -= 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy": n += 1
    return max(1, n)

def _dsents(p):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\([^)]*\)", "", p)) if len(s.split()) >= 3]

def readability(text):
    """Flesch reading ease / Flesch-Kincaid grade / Gunning Fog from syllable + sentence-length counts."""
    p = prose(text); S = _dsents(p); W = re.findall(r"[A-Za-z]+", p)
    nw, ns = max(len(W), 1), max(len(S), 1)
    syl = sum(_syll(w) for w in W); cx = sum(1 for w in W if _syll(w) >= 3)
    wps, spw = nw / ns, syl / nw
    return dict(flesch_ease=round(206.835 - 1.015 * wps - 84.6 * spw, 1),
                fk_grade=round(0.39 * wps + 11.8 * spw - 15.59, 1),
                gunning_fog=round(0.4 * (wps + 100 * cx / nw), 1),
                words_per_sentence=round(wps, 1), pct_complex=round(100 * cx / nw, 1))

def _lex_density(p):
    toks = [w for w in re.findall(r"[a-z']+", p.lower()) if w]
    fw = set(FW); content = sum(1 for w in toks if w not in fw)
    return round(100 * content / max(len(toks), 1), 1)            # Halliday/Ure content-word %

def _nom_density(p):
    toks = [w for w in re.findall(r"[a-z]+", p.lower()) if len(w) > 4]
    nom = sum(1 for w in toks if _NOM_SUF.search(w) and w not in _NOM_STOP)
    return round(1000 * nom / max(len(toks), 1), 1)

def _trigram_rep(p):
    toks = _ctoks(p); tri = list(zip(toks, toks[1:], toks[2:]))
    c = collections.Counter(tri); rep = sum(v for v in c.values() if v > 1)
    return round(100 * rep / max(len(tri), 1), 1)

def _split_sections(text):
    out, last, pos = [], None, 0
    for m in re.finditer(r"(?m)^(?:\\(?:sub)?section\*?\{([^}]*)\}|#{2,3}\s+(.+))$", text):
        title = (m.group(1) or m.group(2) or "").strip()
        if last is not None: out.append((last, text[pos:m.start()]))
        last, pos = title, m.end()
    if last is not None: out.append((last, text[pos:]))
    return out

def _cos_counter(a, b):
    """Cosine between two token Counters -- the ONE shared primitive for every section-overlap feature (density's
    pairwise redundancy and conventions' conclusion-vs-body), so the two read the same metric and never re-define it."""
    import math as _m
    if not a or not b: return 0.0
    dot = sum(a[w] * b[w] for w in (set(a) & set(b)))
    na = _m.sqrt(sum(v * v for v in a.values())); nb = _m.sqrt(sum(v * v for v in b.values()))
    return round(dot / (na * nb), 2) if na and nb else 0.0

def _section_redundancy(text, topn=6):
    secs = [(re.sub(r"^\d+\.?\s*", "", t), prose(b)) for t, b in _split_sections(text)]
    vecs = [(t, collections.Counter(_ctoks(b))) for t, b in secs if len(b.split()) > 120]
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            pairs.append((_cos_counter(vecs[i][1], vecs[j][1]), vecs[i][0], vecs[j][0]))
    return sorted(pairs, reverse=True)[:topn]

# ════════ conventions: venue STRUCTURAL/RHETORICAL conformance (the meta-descriptor above register's style axes) ════════
# register/styleprofile model the venue's SENTENCE-level voice; conventions models its SECTION-level moves: heading style,
# and whether the conclusion ADVANCES (synthesize -> recommend -> bound scope -> outlook, the move structure Swales' CARS
# and academic-writing research describe) or merely RESTATES the body, benchmarked against the venue corpus' own papers.
_CONCL_RE = re.compile(r"(?i)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:conclusion|conclusions|concluding\s+remarks|"
                       r"discussion\s+and\s+conclusion|final\s+remarks?|closing\s+remarks?|outlook)\b")
def conclusion_restatement(text):
    """Cosine(conclusion section, rest of body). HIGH = the conclusion re-says earlier sections instead of synthesizing,
    recommending, and bounding scope -- the academic anti-pattern of a restatement conclusion. Returns (cosine, title)|None."""
    secs = [(re.sub(r"^\d+\.?\s*", "", t), prose(b)) for t, b in _split_sections(text)]
    ci = next((i for i, (t, _) in enumerate(secs) if _CONCL_RE.match(t)), None)
    if ci is None: return None
    cvec = collections.Counter(_ctoks(secs[ci][1])); body = collections.Counter()
    for i, (t, b) in enumerate(secs):
        if i != ci and len(b.split()) > 80: body.update(_ctoks(b))
    if not cvec or not body: return None
    return (_cos_counter(cvec, body), secs[ci][0])

# Coarse section-type classifier for the SYMBOLIC structural signal (the measured complement to the neuro move panel):
# where does each rhetorical block sit, and does the draft's section ORDER match the venue's (e.g. related-work placement)?
_SECTYPE = [
    ("intro",      r"\bintroduction\b|\bmotivation\b"),
    ("related",    r"related work|prior work|\bliterature\b|state of the art|\bbackground\b"),
    ("framework",  r"definition|\bscope\b|preliminar|notation|framework|architecture|\bmodel\b|approach|\bdesign\b|method|theory|conceptuali"),
    ("evidence",   r"result|finding|evaluation|experiment|case study|measur|\banalysis\b|empiric|\bstudy\b"),
    ("discussion", r"discussion|objection|disanalog|limitation|threat|implication|governance|recommendation|polic"),
    ("conclusion", r"conclusion|concluding|final remark|closing remark|\boutlook\b"),
    ("appendix",   r"appendix|implementation mechanism|supplementary|declaration|statement"),
]
def _sectype(title):
    t = title.lower()
    return next((nm for nm, pat in _SECTYPE if re.search(pat, t)), "other")

def section_order(text):
    """Ordered [(sectype, position_fraction)] for the document's sections + the related-work position (or None).
    Symbolic; reads LaTeX \\section and markdown ## alike."""
    secs = [re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", t).strip() for t, _ in _split_sections(text)]
    secs = [s for s in secs if s and s.lower() != "abstract"]
    n = len(secs) or 1
    seq = [(_sectype(s), round(i / n, 2)) for i, s in enumerate(secs)]
    rel = next((p for ty, p in seq if ty == "related"), None)
    return seq, rel

# ── Pluggable genre modules ────────────────────────────────────────────────────────────────────────────────────────
# The conventions ENGINE is genre-neutral: heading style, interrogative tells, conclusion-vs-body restatement, section
# count + length vs corpus, and the corpus-inferred move sequence -- all of which apply to ANY corpus/genre. A GENRE
# MODULE is an optional plugin adding domain section-typing, move PRIORS for the neuro prompt, and specialized structural
# metrics. The 'academic' module ships by default; third parties add their own via register_genre_module(). Contract
# (name/applies required, rest optional):
#   name:str | applies:(docs)->bool | sectype:(title)->str|None | priors:str | struct:(docs)->dict | lines:(text,prof)->[str]
GENRE_MODULES = {}
def register_genre_module(mod): GENRE_MODULES[mod["name"]] = mod
def _route_genre(docs):
    """LLM ROUTER: classify the corpus genre and pick the best-fit registered module (genre routing is judgment, not a
    keyword match). Returns a module name, 'none' if the judge says none fit, or None if no LLM is reachable (the caller
    then falls back to the deterministic applies())."""
    if not GENRE_MODULES or not docs: return None
    blob = " ".join(c for _, c in docs)
    # deterministic README / how-to guard (clear install/usage markers; there is no README module, so route genre-neutral)
    if re.search(r"pip install|npm install|git clone|##\s*(installation|usage|getting started|quick\s*start)", blob, re.I):
        return "none"
    # deterministic ESSAY signal: argument-driven, personal voice, no code blocks (the LLM under-recognizes this register)
    _b = prose(blob); _bw = max(1, len(_b.split()))
    _fp1k = len(re.findall(r"\b(i|we|you|my|our)\b", _b.lower())) / _bw * 1000
    _argc = len(re.findall(r"\b(because|therefore|however|suppose|consider|imagine|i argue|i think|the point is)\b", _b.lower()))
    if blob.count("```") == 0 and _fp1k >= 12 and _argc >= 4 and "essay" in GENRE_MODULES:
        return "essay"
    opts = "\n".join(f"- {nm}: {(m.get('desc') or m.get('priors',''))[:150]}" for nm, m in GENRE_MODULES.items())
    opts += "\n- none: fits no specialized module; use the genre-neutral engine only (this is a real, valid choice -- "
    opts += "pick it when the corpus is a genre we have no module for, do not force a poor fit)"
    def _sample(c):  # a SLICE of the text (opening + a middle window) + register signals -- not full text, not headings only
        body = prose(c); w = max(1, len(body.split()))
        head = " | ".join(h[:48] for h in _all_heading_strings(c)[:8])
        mid = body[len(body) // 2: len(body) // 2 + 320] if len(body) > 1400 else ""
        fp = round(len(re.findall(r"\b(i|we|you|my|our)\b", body.lower())) / w * 1000)
        argu = len(re.findall(r"\b(because|therefore|however|suppose|consider|i argue|i think|the point is)\b", body.lower()))
        return (f"headings: {head}\n  opening: {body[:600]}\n  middle: {mid}\n"
                f"  signals: first-person-per-1k~{fp}, argument-connectives={argu}, code-fences={c.count('```')}")
    sample = "\n\n".join(f"[doc {i+1}]\n  " + _sample(c) for i, (_, c) in enumerate(docs[:3]))
    prompt = ("Route a document to its genre. Choose the SINGLE best-fit option below (including 'none'). Decide from the "
              "OPENING PROSE and the SIGNALS, not the headings alone: high first-person + argument-connectives + few code "
              "fences reads as an essay; decision/tradeoff/architecture prose reads as a design-doc; an abstract + related-work "
              "reads as academic. Reply with ONLY that one word.\n\nOPTIONS:\n" + opts + "\n\nDOCUMENT SAMPLE:\n" + sample)
    try:
        ans = _judge_call(prompt, _JUDGES[0], mt=12, temp=0).strip().lower()
        return next((nm for nm in GENRE_MODULES if nm in ans), "none")
    except Exception:
        return None
def _active_genre(docs, genre=None, route=True):
    """Resolve the genre module: forced name > LLM router > deterministic applies() fallback (offline)."""
    if genre == "none": return None
    if genre and genre in GENRE_MODULES: return GENRE_MODULES[genre]
    if route:
        r = _route_genre(docs)
        if r in GENRE_MODULES: return GENRE_MODULES[r]
        if r == "none": return None
    return next((m for m in GENRE_MODULES.values() if m.get("applies", lambda d: False)(docs)), None)

def _struct_neutral(docs):
    """GENRE-NEUTRAL structural profile -- median section count + section length. Applies to any corpus/genre."""
    nsecs, seclens = [], []
    for _, c in docs:
        bodies = [b for _, b in _split_sections(c) if len(b.split()) >= 30]
        nsecs.append(len(bodies)); seclens += [len(prose(b).split()) for b in bodies]
    return {"n": len(docs), "n_sections_median": int(st.median(nsecs)) if nsecs else None,
            "section_len_median": int(st.median(seclens)) if seclens else None}

def _struct_profile(corpus_dir, genre=None, route=True):
    """GENRE-NEUTRAL structural profile + the active genre module's specialized metrics (genre via LLM router)."""
    docs = _read_corpus([corpus_dir]); prof = _struct_neutral(docs)
    mod = _active_genre(docs, genre, route)
    if mod and mod.get("struct"):
        prof.update(mod["struct"](docs)); prof["genre"] = mod["name"]
    return prof

# ── the 'academic' genre module (ships by default) ──
def _academic_applies(docs):  # require GENUINE academic markers (not just numbered headings, which design docs also use)
    hits = sum(1 for _, c in docs if re.search(r"(?im)\\section|\\cite|^#+\s*(abstract|related work)\b|\bliterature review\b", c))
    return bool(docs) and hits >= max(2, len(docs) // 2)
def _academic_struct(docs):
    rels, hasconcl = [], 0
    for _, c in docs:
        seq, rel = section_order(c)
        if rel is not None: rels.append(rel)
        if any(ty == "conclusion" for ty, _ in seq): hasconcl += 1
    nd = len(docs) or 1
    return {"related_pos_median": round(st.median(rels), 2) if rels else None,
            "related_present_rate": round(len(rels) / nd, 2), "conclusion_rate": round(hasconcl / nd, 2)}
def _academic_lines(text, prof):
    seq, rel = section_order(text); out = []
    out.append("  [academic] section flow: " + " -> ".join(dict.fromkeys(ty for ty, _ in seq if ty != "other")))
    if prof.get("related_pos_median") is not None and rel is not None:
        out.append(f"  [academic] related-work position: draft {rel} vs venue median {prof['related_pos_median']}"
                   + ("  <<< the venue places related work EARLIER" if rel > prof["related_pos_median"] + 0.2 else "  (in band)"))
    elif rel is not None:
        out.append(f"  [academic] related-work position: draft {rel}; a dedicated related-work section appears in only "
                   f"{prof.get('related_present_rate')} of venue papers (often woven in)")
    out.append(f"  [academic] conclusion present: {'yes' if any(ty == 'conclusion' for ty, _ in seq) else 'no'}"
               f"  (venue rate {prof.get('conclusion_rate')})")
    return out
_ACADEMIC_PRIORS = (
    "Academic-genre priors, apply ONLY if the corpus confirms them: research-article Introductions often run Swales CARS "
    "(establish-territory -> establish-niche/gap -> occupy-niche); research-article Conclusions often consolidate, then "
    "evaluate with LIMITATIONS, then draw deductions/future-work (Yang & Allison 2003), so a conclusion that only "
    "re-summarizes the body is a restatement.")
register_genre_module({"name": "academic", "applies": _academic_applies, "sectype": _sectype,
                       "desc": "peer-reviewed research articles / scholarly papers: abstract, numbered sections, "
                               "citations, related work, conclusions",
                       "priors": _ACADEMIC_PRIORS, "struct": _academic_struct, "lines": _academic_lines})

# ── the 'design-doc' genre module (technical design / concept docs, e.g. a repo's docs/concepts/) -- shipped non-academic plugin ──
_DD_SECTYPE = [
    ("motivation", r"motivation|problem|\bwhy\b|non-?goals?|\bgoals?\b|objective|background|context|tl;?dr|summary|overview|where .* (lives|matters)"),
    ("concept",    r"concept|definition|\bmodel\b|abstraction|\bidea\b|principle|approach|design|architecture|mechanism|how (it|this|the)|what (is|it)|choosing|unit of"),
    ("consequence",r"consequence|implication|\bresult|effect|what changes|what stays|impact|outcome|so what"),
    ("tradeoff",   r"trade-?off|alternative|\boption|comparison|\bversus\b|\bvs\b|considered|rejected|limitation|\brisk|drawback|caveat|failure mode"),
    ("migration",  r"migration|rollout|adoption|maintainer|next steps?|\bfuture\b|\bplan\b|roadmap|\btodo\b|what to do"),
    ("example",    r"example|\bcase\b|scenario|walkthrough|in practice|concretely|flywheel|recipe"),
    ("meta",       r"reference|appendix|glossary|see also|related|acknowledg|\bindex\b"),
]
def _dd_sectype(title):
    t = title.lower()
    return next((nm for nm, pat in _DD_SECTYPE if re.search(pat, t)), "other")
def _dd_applies(docs):  # catch-all for technical markdown prose docs that are NOT academic papers (academic is checked first)
    if not docs: return False
    md = sum(1 for _, c in docs if re.search(r"(?m)^#{1,4}\s", c))
    academic = sum(1 for _, c in docs if re.search(r"(?im)\\section|\\cite|^#+\s*abstract\b|related work", c))
    return md >= max(1, len(docs) // 2) and academic < max(2, len(docs) // 2)
def _dd_struct(docs):
    rate = lambda ty: round(sum(1 for _, c in docs if any(_dd_sectype(re.sub(r"^#*\s*\d*\.?\s*", "", t)) == ty
                            for t, _ in _split_sections(c))) / (len(docs) or 1), 2)
    return {"motivation_rate": rate("motivation"), "tradeoff_rate": rate("tradeoff"), "consequence_rate": rate("consequence")}
def _dd_lines(text, prof):
    secs = [re.sub(r"^#*\s*\d*\.?\s*", "", t).strip() for t, _ in _split_sections(text)]
    types = [_dd_sectype(s) for s in secs]
    flow = " -> ".join(dict.fromkeys(t for t in types if t != "other"))
    out = [f"  [design-doc] move flow: {flow or '(no recognized design-doc moves)'}"]
    for key, label in [("motivation", "states a motivation/problem"), ("tradeoff", "weighs tradeoffs/alternatives"),
                       ("consequence", "draws consequences")]:
        out.append(f"  [design-doc] {label}: {'yes' if key in types else 'NO'}  (corpus rate {prof.get(key + '_rate')})")
    return out
_DD_PRIORS = (
    "Technical design-doc / concept-doc priors: a good design doc LEADS with the problem or motivation (often with "
    "explicit non-goals), then states the concept/design, then its consequences, the tradeoffs and alternatives "
    "considered, and what changes for adopters. It is decision-oriented and concrete -- NOT a README (no install/usage), "
    "NOT an academic paper (no abstract / related-work / literature review). AI tells in this genre: hand-wavy "
    "abstraction with no concrete mechanism, hedging that dodges the decision, false precision, 'it is worth noting', and "
    "enumerations ('Three consequences', 'Key benefits') padded with restatement instead of substance.")
# ── the 'essay' genre module (argumentative / expository essays: alignment-forum / lesswrong / rationalist / technical-blog) ──
_ESSAY_SECTYPE = [
    ("hook",        r"tl;?dr|\bintro|motivation|the problem|puzzle|\bwhy\b|setup|premise|starting point|the question"),
    ("thesis",      r"thesis|claim|the (idea|point|argument)|position|in short|the core|what i('?m| am) arguing"),
    ("development", r"because|mechanism|how (it|this|the)|in detail|unpack|the case for|reasoning|derivation"),
    ("example",     r"example|\bcase\b|concretely|in practice|illustration|story|scenario|consider|suppose|imagine"),
    ("objection",   r"objection|\bbut\b|counter|worry|pushback|response|rebuttal|caveat|to be fair|one might"),
    ("implication", r"implication|upshot|so what|consequence|takeaway|why it matters|what this means|\bmoral\b"),
    ("close",       r"conclu|closing|\bcoda\b|where this leaves|in sum|wrapping up|\bfinal"),
]
def _essay_sectype(title):
    t = title.lower()
    return next((nm for nm, pat in _ESSAY_SECTYPE if re.search(pat, t)), "other")
def _essay_applies(docs):  # argumentative first-person prose that is NOT a paper, README, or decision-oriented design doc
    if not docs: return False
    blob = " ".join(c for _, c in docs).lower()
    if re.search(r"\\section|\\cite|#+\s*abstract\b|related work", blob): return False      # academic
    if re.search(r"```bash|pip install|npm install|## (installation|usage|getting started)|make hello", blob): return False  # README
    md = sum(1 for _, c in docs if re.search(r"(?m)^#{1,4}\s", c))
    argu = len(re.findall(r"\b(because|therefore|however|suppose|consider|imagine|i think|i argue|we argue|the point is)\b", blob))
    fp = len(re.findall(r"\b(i|we|you|my|our)\b", blob))
    return md >= 1 and argu >= 3 and fp >= 10
def _essay_struct(docs):
    rate = lambda ty: round(sum(1 for _, c in docs if any(_essay_sectype(re.sub(r"^#*\s*\d*\.?\s*", "", t)) == ty
                            for t, _ in _split_sections(c))) / (len(docs) or 1), 2)
    return {"hook_rate": rate("hook"), "example_rate": rate("example"), "objection_rate": rate("objection")}
def _essay_lines(text, prof):
    secs = [re.sub(r"^#*\s*\d*\.?\s*", "", t).strip() for t, _ in _split_sections(text)]
    types = [_essay_sectype(s) for s in secs]
    flow = " -> ".join(dict.fromkeys(t for t in types if t != "other"))
    out = [f"  [essay] move flow: {flow or '(no recognized essay moves)'}"]
    for key, label in [("hook", "opens with a hook/problem"), ("example", "grounds claims in concrete examples"),
                       ("objection", "anticipates an objection")]:
        out.append(f"  [essay] {label}: {'yes' if key in types else 'NO'}  (corpus rate {prof.get(key + '_rate')})")
    return out
_ESSAY_PRIORS = (
    "Argumentative / expository essay (alignment-forum / rationalist / technical-blog register): a good essay OPENS with a "
    "concrete hook, a specific example, a puzzle, or a sharp claim, states its thesis early and plainly, DEVELOPS the argument "
    "through named mechanisms and concrete examples instead of abstraction, ANTICIPATES the strongest objection and answers it, "
    "and CLOSES on the implication or what changes. It is written in a direct, personal voice (I / we / you), a person reasoning "
    "on the page. AI tells in this genre: abstract throat-clearing before the point, hedging that dodges the claim, signposting "
    "('In this essay I will argue'), listicle padding ('There are three reasons') whose items restate rather than build, vague "
    "profundity and importance-assertion with no concrete instance, false balance, and an impersonal encyclopedic register "
    "instead of a person thinking.")
register_genre_module({"name": "essay", "applies": _essay_applies, "sectype": _essay_sectype,
                       "desc": "argumentative / expository essays and concept notes (alignment-forum / rationalist / "
                               "technical-blog register): a hook, a thesis, argument developed through concrete examples, "
                               "objections answered, implications; personal voice; NOT an academic paper, NOT a README, "
                               "NOT a decision-oriented design doc",
                       "priors": _ESSAY_PRIORS, "struct": _essay_struct, "lines": _essay_lines})

register_genre_module({"name": "design-doc", "applies": _dd_applies, "sectype": _dd_sectype,
                       "desc": "technical design docs / concept notes in a code repo: motivation, design/approach, "
                               "tradeoffs, consequences, migration; markdown prose, NOT a README and NOT a paper",
                       "priors": _DD_PRIORS, "struct": _dd_struct, "lines": _dd_lines})

def conventions_profile(corpus_dir, refresh=False, genre=None, route=True):
    """Infer + CACHE the venue's structural conventions -- the section-level analog of styleprofile's emergent voice.
    Genre-neutral structural profile + the active genre module's metrics + a one-shot neuro synthesis of the venue's
    section/move arc and conclusion convention, cached to <corpus>/.conventions_profile.{json,md}."""
    cache = pathlib.Path(corpus_dir) / ".conventions_profile.json"
    if cache.exists() and not refresh:
        try: return json.loads(cache.read_text())
        except Exception: pass
    sp = _struct_profile(corpus_dir, genre, route)
    outlines = []
    for _, c in _read_corpus([corpus_dir]):
        ss = [re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", t).strip() for t, b in _split_sections(c) if len(b.split()) >= 30]
        if ss: outlines.append(" | ".join(ss[:14]))
    prompt = ("From these section sequences of accepted papers at one venue, infer the venue's STRUCTURAL conventions. "
              "Be concise (<=140 words):\n- the typical section arc (move order)\n- where RELATED WORK / prior literature "
              "sits (early / late / woven in)\n- how papers CONCLUDE: which moves the conclusion makes (consolidate, "
              "evaluate + limitations, implications, future work, call-to-action) and whether it restates or advances\n"
              "- any heading-style norm.\n\nSECTION SEQUENCES:\n- " + "\n- ".join(outlines[:12]))
    try: conv = _judge_call(prompt, _JUDGES[0], mt=800).strip()
    except Exception as e: conv = f"[neuro synthesis unavailable: {type(e).__name__}]"
    prof = {**sp, "corpus": pathlib.Path(corpus_dir).name, "convention": conv}
    try:
        if conv.startswith("["): return prof   # synthesis failed (e.g. no keys) -> don't cache a broken convention
        cache.write_text(json.dumps(prof, indent=2))
        (pathlib.Path(corpus_dir) / ".conventions_profile.md").write_text(
            f"# CONVENTIONS PROFILE (structural) — {prof['corpus']} (n={prof['n']})\n\n"
            f"- related-work position, median: {prof['related_pos_median']} (a dedicated related-work section appears in "
            f"{prof['related_present_rate']} of papers)\n- conclusion present rate: {prof['conclusion_rate']}\n\n"
            f"## Venue convention (inferred)\n\n{conv}\n")
    except Exception: pass
    return prof

# GENRE-NEUTRAL discourse-move palette -- a consistent labeling vocabulary only. The convention a draft is judged against
# is INFERRED from the target corpus (like styleprofile infers the emergent voice), NOT imposed from here, so conventions
# works for any corpus/tone/genre, not just research articles. The academic move models (Swales CARS for Introductions;
# Yang & Allison 2003 / Bunton 2005 for Conclusions = consolidate -> evaluate+limitations -> deductions) are priors the
# judge applies ONLY where the corpus confirms them.
_MOVES = (
    "Genre-neutral move palette (labels only; the venue's ACTUAL convention is whatever its corpus does -- infer it from "
    "the corpus examples, do not assume a fixed template):\n"
    "  orient/contextualize, state-problem-or-gap, state-purpose-or-thesis, define-terms, situate-in-prior-work, "
    "describe-method-or-approach, present-evidence-or-findings, interpret/comment, evaluate/weigh-significance, "
    "address-objection, recommend/prescribe, draw-implications, consolidate/close, point-forward/future-work."
)  # genre-specific priors (e.g. academic Swales/CARS) come from the active GENRE MODULE, not from here.

def conventions(path, corpus_dir=None, neuro=True, genre=None):
    """Venue structural/rhetorical conventions conformance. The ENGINE is genre-neutral (heading style, interrogative
    tells, conclusion-vs-body restatement, section count/length, corpus-inferred move sequence). A pluggable GENRE MODULE
    (register_genre_module(); 'academic' ships by default) adds domain section-typing, move priors, and structural metrics.
    Pass genre='<name>' to force a module, genre='none' to disable; default auto-detects from the corpus."""
    text = pathlib.Path(path).read_text(errors="ignore")
    print(f"# CONVENTIONS — {pathlib.Path(path).name}" + (f"  vs {pathlib.Path(corpus_dir).name}" if corpus_dir else "") + "\n")
    hf = heading_feats(text)
    if corpus_dir:
        chf = [h for h in (heading_feats(c) for _, c in _read_corpus([corpus_dir])) if h]
        if hf and chf:
            print("## Heading conventions (draft vs venue median; <<< = above venue)")
            for k in ("title_case", "mean_words", "nominalization", "antithesis", "interrogative", "colon", "selfcong"):
                med = round(st.median([h.get(k, 0) for h in chf]), 2); dv = hf.get(k, 0)
                print(f"  {k:15} draft {dv:<6} venue {med}" + ("  <<<" if dv > med + max(0.1, 0.5 * med) else ""))
    iq = interrogative_headings(text)
    print("\n## Interrogative / wh-nominal headings & captions (declarative is the academic norm)")
    print("  " + ("none" if not iq else "\n  ".join(f"?  {h[:72]}" for h in iq[:10])))
    concl = conclusion_restatement(text)
    print("\n## Conclusion: advances or restates? (cosine of conclusion vs body; high = restatement)")
    if not concl:
        print("  no conclusion section detected")
    else:
        cos, ctitle = concl; tag = ""
        if corpus_dir:
            cvals = [v[0] for v in (conclusion_restatement(c) for _, c in _read_corpus([corpus_dir])) if v]
            if cvals:
                med = round(st.median(cvals), 2)
                tag = f"; venue median {med}" + ("  <<< restates more than the venue" if cos > med + 0.08 else "  (within venue norm)")
        print(f"  cosine {cos}  [\"{ctitle[:42]}\"]{tag}")
    prof, mod = None, None
    if corpus_dir:
        prof = conventions_profile(corpus_dir, genre=genre, route=neuro)   # cached; LLM router (or deterministic if --no-llm)
        mod = GENRE_MODULES.get(prof.get("genre")) or _active_genre(_read_corpus([corpus_dir]), genre, route=neuro)
        dbodies = [b for _, b in _split_sections(text) if len(b.split()) >= 30]
        dn = len(dbodies); dlen = int(st.median([len(prose(b).split()) for b in dbodies])) if dbodies else 0
        print("\n## Structure (measured, draft vs venue)")
        print(f"  [genre-neutral] sections: draft {dn} vs venue median {prof.get('n_sections_median')}    "
              f"median section length: draft {dlen}w vs venue {prof.get('section_len_median')}w")
        if mod and mod.get("lines"):
            print(f"  [genre module: {mod['name']}]")
            for ln in mod["lines"](text, prof): print(ln)
        else:
            print("  [no genre module matched this corpus -- genre-neutral metrics only]")
    if neuro:
        secs = [(re.sub(r"^\d+\.?\s*", "", t), prose(b)) for t, b in _split_sections(text)]
        secs = [(t, b) for t, b in secs if len(b.split()) >= 40]
        outline = "\n".join(f"[{i+1}] {t} :: {' '.join(b.split()[:50])}" for i, (t, b) in enumerate(secs))
        venue = ""
        if corpus_dir:
            outlines, concls = [], []
            for _, c in _read_corpus([corpus_dir]):
                cs = [(re.sub(r"^\d+\.?\s*", "", t), prose(b)) for t, b in _split_sections(c)]
                cs = [(t, b) for t, b in cs if len(b.split()) >= 30]
                if cs:
                    outlines.append(" | ".join(t[:34] for t, _ in cs[:12]))
                    ct = next((b for t, b in cs if _CONCL_RE.match(t)), "")
                    if ct: concls.append(" ".join(ct.split()[:80]))
            if outlines:
                venue = ("\n\nINFER this venue's conventions from its OWN papers below (do not assume a fixed template).\n"
                         "Section sequences of accepted papers:\n- " + "\n- ".join(outlines[:8]))
                if concls:
                    venue += "\nHow its conclusions actually read (first lines):\n--- " + "\n--- ".join(concls[:5])
        if prof and isinstance(prof.get("convention"), str) and not prof["convention"].startswith("["):
            venue += "\n\nA cached inference of this venue's convention (use as the reference):\n" + prof["convention"]
        priors = ("\n\n" + mod["priors"]) if (mod and mod.get("priors")) else ""
        prompt = ("Audit the SECTION-LEVEL rhetorical structure of this manuscript. FIRST infer the target venue's "
                  "conventions from its own corpus below; the move palette is only a labeling aid, not a template to impose.\n\n"
                  "MOVE PALETTE:\n" + _MOVES + priors + "\n\nTASKS:\n"
                  "1. VENUE CONVENTION: in 2-3 lines, what section/move structure and what conclusion style does this venue use "
                  "(inferred from its papers)? If no corpus is given, say so and fall back to the genre priors.\n"
                  "2. MOVES: one line per DRAFT section, '[n] title -> move(s)'.\n"
                  "3. DIVERGENCE: where does the draft depart from the inferred venue convention (missing / extra / mis-ordered moves)?\n"
                  "4. CONCLUSION_VERDICT: does the draft's conclusion match how THIS venue concludes, or does it merely RESTATE "
                  "the body? State ADVANCES | RESTATES | MIXED + which moves are present vs missing, in 1-2 sentences.\n"
                  "5. FIX: if it diverges or restates, one concrete instruction.\n\nDRAFT SECTIONS:\n" + outline + venue)
        print("\n## Section roles (neuro panel)")
        for spec in _JUDGES:
            try: print(f"\n### {spec[0]}\n{_judge_call(prompt, spec, mt=1300)}")
            except Exception as e: print(f"\n### {spec[0]}: unavailable ({type(e).__name__}: {str(e)[:80]})")

def density(path, corpus_dir=None):
    text = pathlib.Path(path).read_text(errors="ignore"); p = prose(text)
    r = readability(text)
    print(f"# DENSITY & CLARITY — {pathlib.Path(path).name}\n")
    print("## Whole document (deterministic surface proxies)")
    print(f"  Flesch reading ease    {r['flesch_ease']:>6}   higher = clearer; solid academic ~30-50, <25 punishing")
    print(f"  Flesch-Kincaid grade   {r['fk_grade']:>6}   US grade-level; >16 reads grad-dense")
    print(f"  Gunning Fog            {r['gunning_fog']:>6}   years of schooling to parse on first read")
    print(f"  words / sentence       {r['words_per_sentence']:>6}")
    print(f"  complex words (3+ syl) {r['pct_complex']:>6}%")
    print(f"  lexical density        {_lex_density(p):>6}%   content-word ratio (Halliday/Ure); >55 densely packed")
    print(f"  nominalizations        {_nom_density(p):>6}/1k  abstract-noun packing (Biber)")
    print(f"  trigram repetition     {_trigram_rep(p):>6}%   recurring content 3-grams")
    print(f"  lexical diversity MTLD {_mtld(re.findall(r'[a-z]+', p.lower())):>6}")
    if corpus_dir:
        cr = [readability(c) for _, c in _read_corpus([corpus_dir])]
        cp = [prose(c) for _, c in _read_corpus([corpus_dir])]
        if cr:
            M = lambda L, k: round(st.median([x[k] for x in L]), 1)
            print(f"\n  venue median (n={len(cr)}): ease {M(cr,'flesch_ease')}, FK {M(cr,'fk_grade')}, fog {M(cr,'gunning_fog')}, "
                  f"lexdens {round(st.median([_lex_density(x) for x in cp]),1)}%, "
                  f"trigram-rep {round(st.median([_trigram_rep(x) for x in cp]),1)}%")
    print("\n## Densest sections (Flesch-Kincaid grade; clarity drops as FK rises)")
    rows = [(readability(b)['fk_grade'], readability(b)['flesch_ease'], _lex_density(prose(b)), t)
            for t, b in _split_sections(text) if len(b.split()) >= 80]
    for g, e, ld, t in sorted(rows, reverse=True)[:8]:
        print(f"  FK {g:>5}  ease {e:>6}  lexdens {ld:>4}%   {re.sub(chr(92)+'[a-z]+|[{}]','',t)[:58]}")
    print("\n## Most REDUNDANT section pairs (content cosine; high = the same argument restated)")
    for cos, ti, tj in _section_redundancy(text):
        flag = "  <<< restated" if cos >= 0.6 else ""
        print(f"  {cos}   {ti[:32]:32} <-> {tj[:30]}{flag}")

# ════════ cites: citation ATTRIBUTION audit (a distinct failure mode from prose tells) ════════
# A known LLM failure mode: a claim hung on a source that does not support it; ANACHRONISM (a modern/coined construct
# pinned on an old source); OVERCLAIM verbs (a source said to predict/show/prove/cast/frame/classify what it did not);
# and the silent one -- a plain-text "Author (year)" that never enters the bibliography (LaTeX only warns on \cite, not
# on prose mentions). Symbolic pass flags SUSPECTS; the neuro judge verifies each against what the source actually says.
_OVERCLAIM_VERB = (r"predicts?|shows?|proves?|demonstrat\w+|casts?|frames?|classif\w+|finds?|establish\w+|argues?|"
                   r"derives?|coin\w+|introduc\w+|defin\w+|reveals?")
def _deaccent(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()
def _textual_cites(text):
    """Plain-text 'Author (year)' / 'A and B (year)' / 'A et al. (year)' mentions -> (author_phrase, year, pos)."""
    return [(m.group(1).strip(), m.group(2), m.start()) for m in re.finditer(
        r"([A-Z][\wÀ-ÿ'’.-]+(?:\s+(?:and|&)\s+[A-Z][\wÀ-ÿ'’.-]+|\s+et\s+al\.?|\s+and\s+colleagues)?)\s*\(?((?:19|20)\d\d)\b", text)]
def cite_suspects(text, bib_text=""):
    """Symbolic attribution suspects: (1) plain-text mentions whose surname is in NO \\cite key or bib author (may be
    missing from the bibliography); (2) a citation/author-year immediately followed by an overclaim verb."""
    keys = _deaccent(" ".join(re.findall(r"\\cite[a-z]*\*?(?:\[[^]]*\])*\{([^}]+)\}", text)))
    bibblob = _deaccent(bib_text)
    missing = []
    for auth, yr, _ in _textual_cites(text):
        first = re.sub(r"['’]s\b", "", re.split(r"\s+(?:and|&|et)\b", auth, maxsplit=1)[0])  # strip possessive
        sur = re.sub(r"[^a-z]", "", _deaccent(first))
        if len(sur) < 3 or sur in {"the", "section", "table", "figure", "appendix", "enron", "sarbanes", "oxley"}: continue
        if sur in keys or (bib_text and sur in bibblob): continue
        missing.append(f"{auth} ({yr})")
    pat = re.compile(r"(\\cite[a-z]*\{[^}]+\}|[A-Z][\wÀ-ÿ'’.-]+(?:\s+(?:and|&)\s+[A-Z][\wÀ-ÿ'’.-]+|\s+et\s+al\.?)?\s*"
                     r"\((?:19|20)\d\d\))((?:\s+\w+){0,3}?\s+(?:" + _OVERCLAIM_VERB + r"))\b", re.I)
    over = [re.sub(r"\s+", " ", m.group(0))[:120] for m in pat.finditer(text)]
    return {"missing_from_bib": sorted(set(missing)), "overclaim": over[:25]}
def cites(path, bib=None, neuro=True):
    text = pathlib.Path(path).read_text(errors="ignore")
    if bib and pathlib.Path(bib).exists():
        bib_text = pathlib.Path(bib).read_text(errors="ignore")
    else:
        bib_text = "\n".join(p.read_text(errors="ignore") for p in pathlib.Path(path).resolve().parent.glob("*.bib"))
    sus = cite_suspects(text, bib_text)
    print(f"# CITES (attribution audit) — {pathlib.Path(path).name}\n")
    print("## Symbolic suspects (candidates; the judge verifies)\n")
    print("- plain-text 'Author (year)' with NO matching \\cite key or bib author (may be MISSING from the bibliography):")
    print("    " + (", ".join(sus["missing_from_bib"]) or "none"))
    print("\n- citation + an OVERCLAIM verb nearby (attribution-drift / anachronism suspects):")
    print("\n".join("    " + o for o in sus["overclaim"]) or "    none")
    if neuro:
        sents = re.split(r"(?<=[.!?])\s+", prose(text))
        cited = [s.strip() for s in sents if _textual_cites(s) or "\\cite" in s]
        blob = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(cited))
        prompt = ("Audit CITATION ATTRIBUTION for the failure mode of a claim hung on a source that does not support it, "
                  "ANACHRONISM (a modern/coined construct attributed to an old source), or OVERCLAIM verbs (a source said "
                  "to predict/show/prove/cast/frame/classify what it did not). Use your knowledge of the cited works. For "
                  "each numbered sentence that misattributes, output:\n  [n] MISATTRIBUTION|OVERCLAIM|ANACHRONISM - what the "
                  "source actually says vs the claim - a one-line fix.\nList ONLY problems; end with 'OK: [n,...]'.\n\n"
                  "SENTENCES:\n" + blob)
        print("\n## Neuro verification (judge panel)")
        for spec in _JUDGES:
            try: print(f"\n### {spec[0]}\n{_judge_call(prompt, spec, mt=1500)}")
            except Exception as e: print(f"\n### {spec[0]}: unavailable ({type(e).__name__}: {str(e)[:80]})")

# ════════════════════════════════ CLI ════════════════════════════════
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "mirror":
        mirror(sys.argv[2], sys.argv[3:])
    elif cmd == "check":
        mode, rows, flags = check(pathlib.Path(sys.argv[2]).read_text(errors="ignore"), sys.argv[3:] or None)
        print(f"# regex check — {pathlib.Path(sys.argv[2]).name} — {mode}\n")
        for k, thr, v, fl in rows:
            print(f"  {k:18}{str(thr):>8}{v:>8}  {'<<< ' + fl if fl != 'ok' else ''}")
        print(f"\n{flags} flags. {'CLEAN' if flags <= 2 else 'reads AI — see flags'}")
    elif cmd == "infer":
        infer(sys.argv[2], sys.argv[3])
    elif cmd == "taxonomy":
        if "--md" in sys.argv:
            print("# AI-tell taxonomy — the meta-labels (abstraction above cue words)\n")
            for fam in FAMILIES:
                fi = [t for t in TAXONOMY if t["family"] == fam]
                if not fi: continue
                print(f"## {fam}\n")
                for t in fi:
                    cov = "deterministic: " + ", ".join(t["det"]) if t["det"] != ["LLM-only"] else "**LLM-only**"
                    print(f"- **{t['label']}** — {t['definition']} *Why AI:* {t['why_ai']} ({cov})")
                print()
        else:
            llm = [t for t in TAXONOMY if t["det"] == ["LLM-only"]]
            print(f"{len(TAXONOMY)} categories, {len(FAMILIES)} families. LLM-only (no surface signature): "
                  + ", ".join(t["label"] for t in llm) + "\n\n--- judge block ---\n" + judge_block())
    elif cmd == "profile":
        register_profile(sys.argv[2:])
    elif cmd == "describe":
        describe(sys.argv[2])
    elif cmd == "probe":
        probe(sys.argv[2], sys.argv[3])
    elif cmd == "verdict":
        verdict(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "style":
        res = style_score(sys.argv[2], sys.argv[3], sys.argv[4])
        if res is not None:
            margin, thresh, auc, nh, na = res
            print(f"# NEURAL STYLE — {pathlib.Path(sys.argv[2]).name}  (Wegmann emb, chunk-agg, few-shot 2-prototype)")
            print(f"  reference: {nh} human vs {na} AI  ->  style-space separation AUC {auc:.2f}")
            print(f"  draft AI-vs-human margin: {margin:+.3f}  (threshold {thresh:+.3f})  ->  leans {'AI' if margin > thresh else 'HUMAN'}")
    elif cmd == "judge":
        _only, _ven, _i = None, "an academic journal", 3
        while _i < len(sys.argv):
            if sys.argv[_i] == "--only" and _i + 1 < len(sys.argv):
                _only = sys.argv[_i + 1].split(","); _i += 2
            else:
                _ven = sys.argv[_i]; _i += 1
        judge(sys.argv[2], _ven, only=_only)
    elif cmd == "candidates":
        _md = next((int(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--min-docs" and i + 1 < len(sys.argv)), 1)
        candidates(min_docs=_md)
    elif cmd == "taxreview":
        _only = next((sys.argv[i + 1].split(",") for i, a in enumerate(sys.argv) if a == "--only" and i + 1 < len(sys.argv)), ["deepseek", "kimi"])
        taxonomy_review(only=_only)
    elif cmd == "excess":
        excess(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif cmd == "register":
        register(sys.argv[2], sys.argv[3], no_llm="--no-llm" in sys.argv)
    elif cmd == "density":
        density(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else None)
    elif cmd == "conventions":
        _g = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--genre" and i + 1 < len(sys.argv)), None)
        conventions(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else None,
                    neuro="--no-llm" not in sys.argv, genre=_g)
    elif cmd == "cites":
        _b = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--bib" and i + 1 < len(sys.argv)), None)
        cites(sys.argv[2], bib=_b, neuro="--no-llm" not in sys.argv)
    elif cmd == "conventionsprofile":
        _g = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--genre" and i + 1 < len(sys.argv)), None)
        prof = conventions_profile(sys.argv[2], refresh="--refresh" in sys.argv, genre=_g)
        print(f"# CONVENTIONS PROFILE (structural) — {prof['corpus']} (n={prof['n']})\n")
        print(f"  related-work position (median): {prof['related_pos_median']}  "
              f"(dedicated section in {prof['related_present_rate']} of papers)")
        print(f"  conclusion present rate: {prof['conclusion_rate']}\n\n## venue convention (inferred)\n\n{prof.get('convention','')}")
    elif cmd == "styleprofile":
        prof = style_profile(sys.argv[2], refresh="--refresh" in sys.argv)
        print(f"# STYLE PROFILE (emergent) — {sys.argv[2]}  (n={prof.get('_n_docs','?')})\n\nVenue voice: {prof.get('venue_voice','')}\n")
        for c in prof.get("categories", []):
            sc = f"  [intensity {c['intensity']}/100]" if c.get("intensity") is not None else "  [no score]"
            print(f"▸ {c['name']}{sc}\n    does:     {c.get('does','')}\n    does not: {c.get('does_not','')}\n")
        b = prof.get("prob_ai_baseline")
        if b: print("PROB_AI human baseline: " + ", ".join(f"{k} median {v['median']} [{v['min']}-{v['max']}]" for k, v in b.items() if v))
    elif cmd == "baseline":
        corpus = sys.argv[2]
        only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
        draft = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--draft" and i + 1 < len(sys.argv)), None)
        res = prob_ai_baseline(corpus, only=only, refresh="--refresh" in sys.argv)
        print(f"\n=== PROB_AI human baseline — {corpus}  (n={res['n']}) ===")
        for k, v in res["band"].items():
            if v: print(f"  {k:9} n={v['n']:2}  min {v['min']:>3}  p25 {v['p25']:>3}  median {v['median']:>3}  p75 {v['p75']:>3}  max {v['max']:>3}")
        if draft:
            sc, base = place_in_baseline(draft, corpus, only=only)
            print(f"\n=== draft: {draft} ===")
            for k in ("deepseek", "grok", "kimi"):
                v, bb = sc.get(k), base.get(k)
                if v is not None and bb:
                    verdict = "IN band" if v <= bb["max"] else f"ABOVE the humans (max {bb['max']})"
                    print(f"  {k:9} {v:>3}   {verdict}  (human median {bb['median']})")
    else:
        print(__doc__)
