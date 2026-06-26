"""generate.py — (re)build the AI-default essay set across three model families.

Writes _generated/ai_NN.md (deepseek), ai_grok_NN.md, ai_kimi_NN.md on matched AI-governance topics, with NO
style/de-AI instruction: AI-DEFAULT prose, the negative class for `python ../deai.py infer`. Self-contained;
reads API keys from a .env in the repo root or its parent. Skips files that already exist.

Run: python generate.py
"""
import os, re, json, pathlib, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
GEN = HERE / "_generated"; GEN.mkdir(exist_ok=True)
env = {}
for ep in (HERE.parents[0] / ".env", HERE.parents[1] / ".env"):
    if pathlib.Path(ep).exists():
        for line in pathlib.Path(ep).read_text(errors="ignore").splitlines():
            m = re.match(r'^([A-Z0-9_]+)=(.*)$', line.strip())
            if m: env.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))

TOPICS = [
    "accountability in AI decision-making", "the limits of AI auditing", "human oversight of automated systems",
    "regulatory capture in AI safety", "transparency and explainability in AI governance",
    "AI and the future of professional judgment", "public trust in algorithmic decision-making",
    "the ethics of self-evaluating AI systems", "institutional design for AI accountability",
    "correlated failure in AI monitoring systems",
]
# (prefix, endpoint, key-var, model). deepseek = the original 10; grok/kimi = the multi-model hardening set.
# Endpoints and model IDs are overridable via env (<PROVIDER>_URL / <PROVIDER>_MODEL); defaults below.
MODELS = [("",      env.get("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions"), "DEEPSEEK_API_KEY", env.get("DEEPSEEK_MODEL", "deepseek-chat")),
          ("grok_", env.get("XAI_URL",      "https://api.x.ai/v1/chat/completions"),      "XAI_API_KEY",      env.get("XAI_MODEL",      "grok-4.3")),
          ("kimi_", env.get("KIMI_URL",     "https://api.moonshot.ai/v1/chat/completions"),"KIMI_API_KEY",     env.get("KIMI_MODEL",     "kimi-k2.6"))]

# Register 1 = naive one-shot. Register 2 = DELIBERATIVE: ask the model to argue AND be self-aware about its own
# limits/objections, which is exactly what induces the discourse-stance register (meta-narration, lockstep hedging)
# that a careful AI paper draft has. The Register-2 set is how we EMPIRICALLY validate the discourse-stance tells —
# the naive set cannot (see deai.md, "two registers").
PROMPT = {
    "r1": "Write an ~800-word academic essay on '{t}' for a scholarly journal. Just the essay prose, no headings or references.",
    "r2": ("Write an ~900-word scholarly essay on '{t}'. Argue a clear thesis, but be rigorous and self-aware: state "
           "precisely what your claim does and does not establish, acknowledge the limits of your argument, and "
           "pre-empt the strongest objection before it is raised. Measured academic register, no headings or references."),
}

def gen(url, key, model, prompt):
    body = json.dumps({"model": model, "temperature": 0.7, "max_tokens": 1500, "messages": [{"role": "user",
        "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=200) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]

if __name__ == "__main__":
    import sys
    reg = "r2" if "--r2" in sys.argv else "r1"
    tag = "r2_" if reg == "r2" else ""
    for prefix, url, keyvar, model in MODELS:
        key = env.get(keyvar, "")
        if not key:
            print(f"  skip {model}: no {keyvar}"); continue
        for i, topic in enumerate(TOPICS):
            f = GEN / f"ai_{tag}{prefix}{i:02d}.md"
            if f.exists() and len(f.read_text()) > 800: continue
            try:
                f.write_text(gen(url, key, model, PROMPT[reg].format(t=topic)))
                print(f"  {reg} {model} ai_{tag}{prefix}{i:02d}: {topic[:40]}", flush=True)
            except Exception as e:
                print(f"  FAIL {model} ai_{tag}{prefix}{i:02d}: {str(e)[:120]}", flush=True)
