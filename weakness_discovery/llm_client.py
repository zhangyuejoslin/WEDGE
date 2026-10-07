"""Step 4: LLM call via Gemini's OpenAI-compatible REST endpoint (stdlib only).

Key resolution order: $GEMINI_API_KEY, then the file gemini.key next to this
module (one line, plain key; that file is for the user to drop in later).
`--mock <dir>` in discover.py bypasses this entirely.
"""
import json
import os
import urllib.request

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
KEYFILE = os.path.join(HERE, "gemini.key")


def get_key():
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k and os.path.exists(KEYFILE):
        k = open(KEYFILE).read().strip()
    if not k:
        raise RuntimeError(
            "no Gemini key: set $GEMINI_API_KEY or write it to " + KEYFILE)
    return k


def call_llm(prompt, model="gemini-2.5-pro", temperature=0.8, timeout=180):
    body = json.dumps({
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + get_key(),
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    return resp["choices"][0]["message"]["content"]
