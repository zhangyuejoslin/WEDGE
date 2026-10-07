#!/usr/bin/env python3
"""Stage 0 of LM-assisted weakness discovery: build episodes.json.

One record per evaluated ProcTHOR val episode (hm3d cats, n=655), three sections:
  result   : success (the label the gates predict)
  baseline : the six hand-written features (same definitions as feature_table.py)
  scene    : layout/object properties -- controllable at generation time
  planner  : shortest-path properties -- derivable from layout before any rollout
  agent    : rollout statistics -- CONTEXT ONLY. Stripped from the record before
             candidate feature functions run (outcome vars trivially predict the
             outcome and are not generation knobs), but shown in the LLM prompt
             so it can reason about failure modes.

Sources: failure_labels.json (metrics from the los_oracle/analysis pipeline) +
output_uninavid_val/*.json (base rollout takenActions -> action-pattern stats).

Usage: python3 build_dataset.py --labels <failure_labels.json> --results '<results_*.json>' --out episodes.json
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "episodes.json")

SCENE_KEYS = [
    "houseArea", "roomsTotal", "numDoorsHouse", "numTargetObjs", "numSameTypeRooms",
    "numPlausibleDecoyRooms", "targetRoomArea", "targetRoomDoors", "targetRoomIsLeaf",
    "targetInCorner", "targetDistToNearestDoor", "occlusionDegree", "occlusionDegreeMax",
    "occlusionDegreeMean", "occlusion", "losOnsetRange", "visibilityCoverage",
    "startRoomType", "startRoom", "targetRoom", "roomPair",
]
PLANNER_KEYS = [
    "shortestPath", "roomHopsStartToTarget", "doorsOnPath", "branchingBeforeTarget",
    "decoyBranchesBeforeTarget", "doorwaySeesTarget", "doorwaySeeingFrac",
    "pathBeforeFirstVisible",
]
AGENT_KEYS = [
    "envSteps", "minDistToTarget", "roomsVisited", "revisitRatio", "stuckFrac",
    "houseCoverage", "wrongRoomEntries", "stayedInStartRoomSteps",
    "everReachedTargetDoorway", "reachedTargetRoom",
]


# allenact's RL policy emits THOR action names; Uni-NaVid emits its own words.
# Normalise so action_stats() reads the same for both agents.
RL_ACTION_ALIAS = {"MoveAhead": "forward", "RotateLeft": "left", "RotateRight": "right",
                   "End": "stop", "LookUp": "up", "LookDown": "down"}


def action_stats(taken):
    """Action-pattern stats from the rollout's emitted chunks."""
    flat = [a for chunk in taken for a in (chunk if isinstance(chunk, list) else [chunk])]
    flat = [RL_ACTION_ALIAS.get(a, a) for a in flat]
    n = len(flat) or 1
    fwd = sum(a == "forward" for a in flat)
    left = sum(a == "left" for a in flat)
    right = sum(a == "right" for a in flat)
    # longest same-direction turn run (>=12 consecutive 30-degree turns = a full spin)
    spins, run, prev = 0, 0, None
    for a in flat:
        if a in ("left", "right") and a == prev:
            run += 1
        elif a in ("left", "right"):
            run = 1
        else:
            if run >= 12:
                spins += 1
            run = 0
        prev = a
    if run >= 12:
        spins += 1
    return {
        "forwardShare": round(fwd / n, 3),
        "turnShare": round((left + right) / n, 3),
        "fullSpins": spins,
        "numActions": len(flat),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # Defaults reproduce the Uni-NaVid corpus. Point them elsewhere to build the
    # same record schema for a different agent -- e.g. the procthor-rl policy:
    #   label_failures.py --results <RLDIR> --out eval_objnav/failure_labels_rl
    #   merge_los.py --labels eval_objnav/failure_labels_rl/failure_labels.json
    #   build_dataset.py --labels ... --results '<RLDIR>/results.json' --out episodes_rl.json
    # The scene/planner blocks are agent-independent geometry, so only `success`
    # and the `agent` block actually change between agents.
    ap.add_argument("--labels", required=True,
                    help="failure_labels.json from the episode-analysis pipeline (scene/planner metrics + success)")
    ap.add_argument("--results", required=True,
                    help="glob of per-episode rollout results json (idx, calledEnd, takenActions)")
    ap.add_argument("--out", default=OUT)
    # Uni-NaVid is only evaluated on the 6 HM3D-overlapping categories, so hm3d is the
    # default. The RL policy trained on all 16 and is saturated on the 6 (SR 95.0), so
    # its corpus needs --cats all -- that is where its failures actually live.
    ap.add_argument("--cats", choices=["hm3d", "extra", "all"], default="hm3d")
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.labels))
            if args.cats == "all" or r["catGroup"] == args.cats]
    val = {}
    for f in glob.glob(args.results):
        for e in json.load(open(f)):
            val[e["idx"]] = e

    out = []
    for r in rows:
        m = r["metrics"]
        v = val.get(r["idx"], {})
        rec = {
            "id": r["id"],
            "idx": r["idx"],
            "success": bool(r["success"]),
            "targetCategory": r["cat"],
            "baseline": {
                "room_hops_to_target": m.get("roomHopsStartToTarget"),
                "geodesic_distance": m.get("shortestPath"),
                "target_room_area": m.get("targetRoomArea"),
                "rooms_in_house": m.get("roomsTotal"),
                "junctions_on_path": m.get("branchingBeforeTarget"),
                "doorway_sees_target": m.get("doorwaySeesTarget"),
            },
            "scene": {k: m.get(k) for k in SCENE_KEYS if k in m},
            "planner": {k: m.get(k) for k in PLANNER_KEYS if k in m},
            "agent": {k: m.get(k) for k in AGENT_KEYS if k in m},
        }
        rec["agent"]["everVisible"] = bool(r["everVisible"])
        rec["agent"]["spl"] = r.get("spl")
        if v:
            rec["agent"]["calledEnd"] = bool(v.get("calledEnd"))
            rec["agent"].update(action_stats(v.get("takenActions", [])))
        rec["failureLabels"] = [k for k, on in r.get("labels", {}).items()
                                if on and k != "success"]
        out.append(rec)

    json.dump(out, open(args.out, "w"), indent=1)
    n = len(out)
    sr = 100 * sum(e["success"] for e in out) / n
    cov = 100 * sum("calledEnd" in e["agent"] for e in out) / n
    print(f"wrote {args.out}: n={n} SR={sr:.1f}% valResultJoin={cov:.0f}%")


if __name__ == "__main__":
    main()
