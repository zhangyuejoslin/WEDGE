"""Step 2: compress one episode record into a compact structured summary for the prompt.

CRITICAL: labels must be the EXACT dict keys — the LLM writes ep["scene"]["<label>"]
straight from these summaries. The first real run lost ~8 candidates to nan-rejects
because pretty labels (decoyRooms, junctions, geodesic) didn't match the real keys
(numPlausibleDecoyRooms, branchingBeforeTarget, shortestPath).
"""

SCENE_SHOW = [
    "houseArea", "roomsTotal", "numDoorsHouse", "numTargetObjs", "numSameTypeRooms",
    "numPlausibleDecoyRooms", "targetRoomArea", "targetRoomDoors", "targetRoomIsLeaf",
    "targetInCorner", "targetDistToNearestDoor", "occlusionDegree", "losOnsetRange",
    "visibilityCoverage", "roomPair",
]
PLANNER_SHOW = [
    "shortestPath", "roomHopsStartToTarget", "doorsOnPath", "branchingBeforeTarget",
    "decoyBranchesBeforeTarget", "doorwaySeesTarget", "doorwaySeeingFrac",
    "pathBeforeFirstVisible",
]
AGENT_SHOW = [
    "envSteps", "everVisible", "minDistToTarget", "roomsVisited", "revisitRatio",
    "calledEnd", "forwardShare", "fullSpins",
]


def _f(v, nd=2):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)


def summarize(ep):
    tag = "SUCCESS" if ep["success"] else "FAIL"
    parts = [
        f"[{tag}] targetCategory={ep['targetCategory']}",
        "scene: " + " ".join(f"{k}={_f(ep['scene'].get(k))}" for k in SCENE_SHOW),
        "planner: " + " ".join(f"{k}={_f(ep['planner'].get(k))}" for k in PLANNER_SHOW),
        "agent(context only): " + " ".join(f"{k}={_f(ep['agent'].get(k))}" for k in AGENT_SHOW),
    ]
    if ep.get("failureLabels"):
        parts.append("failureLabels: " + ",".join(ep["failureLabels"][:6]))
    return " | ".join(parts)
