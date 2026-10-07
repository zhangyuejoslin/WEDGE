#!/usr/bin/env python3
"""LM-assisted weakness discovery: main loop (stages 1-3).

Per round: sample success/failure exemplars -> prompt LLM for N candidate feature
functions -> parse & sandbox-exec -> run the 4 gates -> accepted features join the
library, rejections (with reasons) feed back into the next round's prompt.

Usage:
    python discover.py --episodes ../data/uninavid_pretrained_655.json --tag demo \
        --from-scratch --rounds 5 --save-prompts          # real LLM (needs $GEMINI_API_KEY)
    python discover.py --episodes ../data/uninavid_pretrained_655.json --tag mocktest \
        --from-scratch --rounds 1 --mock mock_responses   # plumbing test, no API

Outputs (in --out-dir, default: next to this file):
    ledger.json       every candidate's full verdict record   (paper appendix)
    feature_lib.py    accepted feature functions, importable  (generation knobs)
    report.md         human-readable verdicts per round
    weakness_table.md tertile SR table: baseline 6 + accepted (updated heatmap)

Design note (deliberate, beyond the written spec): candidate functions receive
ONLY {targetCategory, baseline, scene, planner} — the `agent` rollout stats are
shown in the prompt as diagnostic context but stripped from the record at
feature-eval time. Outcome statistics trivially predict the outcome and cannot
be layout-generation knobs, so letting them into features would poison the
library while sailing through every gate.
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np

from gates import gate1_coverage, gate2_redundancy, gate3_predictive, gate4_incremental
from sandbox import extract_functions
from summarize import summarize

HERE = os.path.dirname(os.path.abspath(__file__))
EPISODES = os.path.join(HERE, "..", "data", "uninavid_pretrained_655.json")
OUT_DIR = HERE
AGENT_DESC = ("an ObjectNav agent (a vision-language model that receives first-person video "
              "and outputs discrete actions: forward / left / right / stop)")

# Two run modes, kept in separate files via --tag:
#   seeded (default): library starts with the hand-written six; LLM hunts BEYOND them.
#   --from-scratch  : library starts EMPTY and the prompt never mentions the six.
#     The validation question is whether the loop REDISCOVERS the hand-filtered
#     features on its own (compare_to_baseline.py measures the overlap).
LEDGER = FEATURE_LIB = REPORT = WEAKNESS_TABLE = None


def set_paths(tag, out_dir=None):
    global LEDGER, FEATURE_LIB, REPORT, WEAKNESS_TABLE, OUT_DIR
    OUT_DIR = out_dir or HERE
    os.makedirs(OUT_DIR, exist_ok=True)
    LEDGER = os.path.join(OUT_DIR, f"ledger_{tag}.json")
    FEATURE_LIB = os.path.join(OUT_DIR, f"feature_lib_{tag}.py")
    REPORT = os.path.join(OUT_DIR, f"report_{tag}.md")
    WEAKNESS_TABLE = os.path.join(OUT_DIR, f"weakness_table_{tag}.md")

N_EXEMPLARS = 15
N_CANDIDATES = 5

BASELINE_DESC = {
    "room_hops_to_target": "rooms crossed on the shortest path (0 = same room)",
    "geodesic_distance": "shortest-path length in meters",
    "target_room_area": "area (m2) of the room containing the target",
    "rooms_in_house": "total rooms in the house",
    "junctions_on_path": "branching junctions on the path before the target",
    "doorway_sees_target": "is the target visible from its room's doorway (bool)",
}


def eval_record(ep, scratch=False):
    """What a candidate feature function is allowed to see.
    from-scratch mode also drops the baseline dict (its values still exist as raw
    scene/planner fields, so the LLM can rediscover them as candidates)."""
    rec = {"targetCategory": ep["targetCategory"],
           "scene": ep["scene"], "planner": ep["planner"]}
    if not scratch:
        rec["baseline"] = ep["baseline"]
    return rec


def baseline_library(records):
    lib = {}
    for name in BASELINE_DESC:
        raw = [ep["baseline"].get(name) for ep in records]
        med = float(np.median([float(v) for v in raw if v is not None]))
        lib[name] = np.array([med if v is None else float(v) for v in raw])
    return lib


def load_ledger():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else []


def accepted_from(ledger):
    return [e for e in ledger if e["verdict"] == "accepted"]


def rebuild_library(records, ledger, scratch=False):
    """(Baseline six unless from-scratch) + every accepted feature, as value vectors."""
    lib = {} if scratch else baseline_library(records)
    evrecs = [eval_record(ep, scratch) for ep in records]
    for e in accepted_from(ledger):
        fns = extract_functions(e["source"])
        fn = fns[e["name"]]
        ok, _, vals = gate1_coverage(fn, evrecs)
        assert ok, f"accepted feature {e['name']} no longer computes"
        lib[e["name"]] = vals
    return lib


def build_prompt(records, ledger, rnd, scratch=False, agent_desc=None):
    rng = random.Random(1000 + rnd)
    agent_desc = agent_desc or AGENT_DESC
    sr = 100.0 * sum(bool(ep["success"]) for ep in records) / len(records)
    succ = [ep for ep in records if ep["success"]]
    fail = [ep for ep in records if not ep["success"]]
    ex_s = rng.sample(succ, N_EXEMPLARS)
    ex_f = rng.sample(fail, N_EXEMPLARS)

    accepted = accepted_from(ledger)
    rejected = [e for e in ledger if e["verdict"] == "rejected"]

    lines = []
    lines.append(
        f"You are analyzing failures of {agent_desc} "
        f"in procedurally generated indoor houses. Overall success rate is ~{sr:.0f}%. "
        "Your job: propose NEW numeric episode features that predict whether the agent will FAIL, "
        "to be used later as knobs when generating training layouts.")
    lines.append("\n## Successful episodes (sample)")
    lines += [f"{i+1}. {summarize(ep)}" for i, ep in enumerate(ex_s)]
    lines.append("\n## Failed episodes (sample)")
    lines += [f"{i+1}. {summarize(ep)}" for i, ep in enumerate(ex_f)]

    if not scratch or accepted:
        lines.append("\n## Features already validated (do NOT propose anything similar)")
        if not scratch:
            for n, d in BASELINE_DESC.items():
                lines.append(f"- {n}: {d}")
        for e in accepted:
            lines.append(f"- {e['name']}: {e['docstring']}")

    if rejected:
        lines.append("\n## Previously REJECTED candidates (do not repeat these ideas)")
        for e in rejected[-40:]:
            lines.append(f"- {e['name']}: {e.get('docstring','')} -- rejected: {e['reason']}")

    scene_keys = ", ".join(records[0]["scene"])
    planner_keys = ", ".join(records[0]["planner"])
    baseline_key_line = ("" if scratch else
                         f'    ep["baseline"]        dict with keys: {", ".join(BASELINE_DESC)}\n')
    novelty_line = ("simple raw fields and combinations/ratios/interactions are all fair game."
                    if scratch else
                    "combinations/ratios/interactions of raw fields are encouraged, trivial "
                    "rescalings of a single validated feature are not.")
    lines.append(f"""
## Output requirements
Propose exactly {N_CANDIDATES} NEW candidate features. For each, write a Python function:
- name starts with `feature_`, takes one argument `ep` and returns a single number;
- first line is a one-sentence docstring stating the hypothesis;
- `ep` is a dict with EXACTLY these keys (nothing else exists at feature time):
    ep["targetCategory"]  one of Bed/Chair/HousePlant/Sofa/Television/Toilet
{baseline_key_line}    ep["scene"]           dict with EXACTLY these keys: {scene_keys}
    ep["planner"]         dict with EXACTLY these keys: {planner_keys}
  Use these key names LITERALLY — any other key returns None and the feature is
  rejected for coverage. The agent(context only) fields are rollout OUTCOMES — they
  are NOT in `ep` and must not be used; they are shown only to help you reason about
  failure modes.
- only `math` and basic builtins are available; no imports, no I/O;
- fields may be None — guard against it (return float('nan') is acceptable, but keep it rare);
- prefer properties CONTROLLABLE at generation time (geometry, layout, object placement);
  {novelty_line}
Wrap all {N_CANDIDATES} functions in ONE ```python code block.""")
    return "\n".join(lines)


def parse_candidates(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no code block in reply")
    fns = {}
    for b in blocks:
        fns.update(extract_functions(b))
    if not fns:
        raise ValueError("no feature_* functions found")
    src = {}
    for b in blocks:
        for m in re.finditer(r"(def (feature_\w+)\(.*?)(?=\ndef |\Z)", b, re.S):
            src[m.group(2)] = m.group(1).rstrip() + "\n"
    return fns, src


def judge(name, fn, source, records, succ, lib, rnd, scratch=False):
    entry = {"round": rnd, "name": name, "source": source,
             "docstring": (fn.__doc__ or "").strip().split("\n")[0],
             "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gates": {}}
    if name in lib:
        entry.update(verdict="rejected", reason="duplicate name / already in library")
        return entry, None
    evrecs = [eval_record(ep, scratch) for ep in records]

    ok, info, vals = gate1_coverage(fn, evrecs)
    entry["gates"]["coverage"] = info
    if not ok:
        entry.update(verdict="rejected", reason=info["reason"])
        return entry, None
    ok, info = gate2_redundancy(vals, lib)
    entry["gates"]["redundancy"] = info
    if not ok:
        entry.update(verdict="rejected", reason=info["reason"])
        return entry, None
    ok, info = gate3_predictive(vals, succ)
    entry["gates"]["predictive"] = info
    if not ok:
        entry.update(verdict="rejected", reason=info["reason"])
        return entry, None
    ok, info = gate4_incremental(vals, lib, succ)
    entry["gates"]["incremental"] = info
    if not ok:
        entry.update(verdict="rejected", reason=info["reason"])
        return entry, None
    entry.update(verdict="accepted", reason="")
    return entry, vals


def write_outputs(ledger, records, lib, succ):
    json.dump(ledger, open(LEDGER, "w"), indent=1, ensure_ascii=False)

    accepted = accepted_from(ledger)
    with open(FEATURE_LIB, "w") as f:
        f.write('"""Auto-discovered weakness features (all passed the 4 gates).\n'
                'Generated by discover.py -- do not edit by hand. Each takes the\n'
                'eval_record dict (targetCategory/baseline/scene/planner).\n"""\n'
                "import math\n\n")
        for e in accepted:
            f.write(e["source"] + "\n\n")
        f.write("ACCEPTED = [%s]\n" % ", ".join(repr(e["name"]) for e in accepted))

    with open(REPORT, "w") as f:
        f.write("# LM-assisted weakness discovery — verdict report\n\n")
        f.write(f"episodes: {len(records)}, overall SR {100*succ.mean():.1f}%. "
                f"candidates so far: {len(ledger)}, accepted: {len(accepted)}.\n\n")
        for e in ledger:
            f.write(f"## round {e['round']} — `{e['name']}` — **{e['verdict'].upper()}**\n")
            if e.get("docstring"):
                f.write(f"hypothesis: {e['docstring']}\n\n")
            if e["reason"]:
                f.write(f"reason: {e['reason']}\n\n")
            if e.get("gates"):
                f.write("```json\n" + json.dumps(e["gates"], ensure_ascii=False, indent=1)
                        + "\n```\n\n")

    with open(WEAKNESS_TABLE, "w") as f:
        f.write("# Weakness heatmap — baseline 6 + auto-discovered (tertile SR, all %d eps)\n\n"
                % len(records))
        f.write("| feature | SR@low tertile | SR@high tertile | gap (pp) | origin |\n")
        f.write("|---|---|---|---|---|\n")
        acc_names = {e["name"] for e in accepted}
        for name, vals in lib.items():
            q33, q66 = np.percentile(vals, [100 / 3, 200 / 3])
            if q33 == q66:  # boolean / heavily-tied feature: binary split instead
                lo, hi = succ[vals <= q33], succ[vals > q33]
                if not len(lo) or not len(hi):
                    continue
            else:
                lo, hi = succ[vals <= q33], succ[vals >= q66]
            f.write("| %s | %.1f%% (n=%d) | %.1f%% (n=%d) | %+.1f | %s |\n" % (
                name, 100 * lo.mean(), len(lo), 100 * hi.mean(), len(hi),
                100 * (hi.mean() - lo.mean()),
                "auto" if name in acc_names else "baseline"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--mock", default="", help="dir with round<i>.md canned replies (no API)")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--save-prompts", action="store_true")
    ap.add_argument("--from-scratch", action="store_true",
                    help="empty library, prompt never mentions the hand-written six; "
                         "validates whether the loop REDISCOVERS them")
    ap.add_argument("--tag", default=None,
                    help="output-file suffix (default: 'scratch' if --from-scratch else 'seeded')")
    ap.add_argument("--episodes", default=EPISODES,
                    help="episode corpus to mine (see data/). Always pair with a distinct "
                         "--tag so ledgers of different agents do not mix.")
    ap.add_argument("--out-dir", default=None,
                    help="where ledger/feature_lib/report/table/prompts go (default: next to this file)")
    ap.add_argument("--agent-desc", default=None,
                    help="one-phrase description of the agent for the prompt header, e.g. "
                         "'an RL ObjectNav policy (DD-PPO, RGB only)'")
    args = ap.parse_args()
    tag = args.tag or ("scratch" if args.from_scratch else "seeded")
    set_paths(tag, args.out_dir)

    records = json.load(open(args.episodes))
    succ = np.array([ep["success"] for ep in records], dtype=float)
    ledger = load_ledger()
    lib = rebuild_library(records, ledger, args.from_scratch)
    start_round = max((e["round"] for e in ledger), default=-1) + 1

    for rnd in range(start_round, start_round + args.rounds):
        prompt = build_prompt(records, ledger, rnd, args.from_scratch, args.agent_desc)
        if args.save_prompts:
            open(os.path.join(OUT_DIR, f"prompt_{tag}_round{rnd}.md"), "w").write(prompt)
        if args.mock:
            path = os.path.join(args.mock, f"round{rnd}.md")
            if not os.path.exists(path):
                print(f"round {rnd}: no mock reply at {path}, stopping")
                break
            reply = open(path).read()
        else:
            from llm_client import call_llm
            reply = call_llm(prompt, model=args.model, temperature=args.temperature)
        try:
            fns, srcs = parse_candidates(reply)
        except Exception as e:
            ledger.append({"round": rnd, "name": f"<round{rnd}-parse>", "source": "",
                           "docstring": "", "gates": {}, "verdict": "rejected",
                           "reason": f"code failed to execute: {e}",
                           "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            write_outputs(ledger, records, lib, succ)
            continue

        n_acc = 0
        for name, fn in fns.items():
            entry, vals = judge(name, fn, srcs.get(name, ""), records, succ, lib, rnd,
                                args.from_scratch)
            ledger.append(entry)
            print(f"round {rnd}: {name} -> {entry['verdict']}"
                  + (f" ({entry['reason']})" if entry["reason"] else ""))
            if entry["verdict"] == "accepted":
                lib[name] = vals
                n_acc += 1
        write_outputs(ledger, records, lib, succ)
        print(f"round {rnd} done: {n_acc} accepted, library size {len(lib)}")
        if n_acc == 0 and rnd > start_round:
            print("a full round with no new feature -> stop")
            break


if __name__ == "__main__":
    main()
