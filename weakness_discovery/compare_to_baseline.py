#!/usr/bin/env python3
"""The from-scratch validation readout: did the LLM loop REDISCOVER the six
hand-filtered features?

For every accepted feature in a ledger, compute Spearman |rho| against each of
the hand-written six; report the best match per auto feature, and per baseline
feature whether anything recovered it (|rho| >= 0.7 counts as rediscovered).

Usage:  python compare_to_baseline.py --episodes ../data/uninavid_pretrained_655.json \
            --ledger ../runs/demo/ledger_demo.json
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

import discover
from discover import BASELINE_DESC, baseline_library, eval_record
from gates import gate1_coverage
from sandbox import extract_functions

HERE = os.path.dirname(os.path.abspath(__file__))
RECOVER_RHO = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--seeded", action="store_true",
                    help="ledger came from a seeded run (features saw ep[\"baseline\"])")
    args = ap.parse_args()

    records = json.load(open(args.episodes))
    ledger = json.load(open(args.ledger))
    accepted = [e for e in ledger if e["verdict"] == "accepted"]
    if not accepted:
        print("no accepted features in %s" % args.ledger)
        return

    base = baseline_library(records)
    scratch = not args.seeded
    evrecs = [eval_record(ep, scratch) for ep in records]

    print(f"# rediscovery check — {len(accepted)} auto features vs the hand-written six\n")
    print("| auto feature | best-matching baseline | |rho| | verdict |")
    print("|---|---|---|---|")
    best_per_base = {b: 0.0 for b in base}
    for e in accepted:
        fn = extract_functions(e["source"])[e["name"]]
        ok, _, vals = gate1_coverage(fn, evrecs)
        if not ok:
            print(f"| {e['name']} | (no longer computes) | - | - |")
            continue
        rhos = {b: abs(spearmanr(vals, bv).statistic) for b, bv in base.items()}
        bb = max(rhos, key=rhos.get)
        for b, r in rhos.items():
            if np.isfinite(r):
                best_per_base[b] = max(best_per_base[b], r)
        tagv = "rediscovers " + bb if rhos[bb] >= RECOVER_RHO else "novel"
        print(f"| {e['name']} | {bb} | {rhos[bb]:.2f} | {tagv} |")

    print("\n| hand-written feature | best |rho| from auto set | recovered? |")
    print("|---|---|---|")
    rec = 0
    for b, r in best_per_base.items():
        got = r >= RECOVER_RHO
        rec += got
        print(f"| {b} | {r:.2f} | {'YES' if got else 'no'} |")
    print(f"\nrecovered {rec}/{len(base)} hand-written features at |rho|>={RECOVER_RHO}")


if __name__ == "__main__":
    main()
