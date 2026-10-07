Here are my candidate features:

```python
def feature_geodesic_squared(ep):
    """Squared shortest-path distance amplifies the long-range difficulty."""
    d = ep["planner"]["shortestPath"]
    return d * d if d is not None else float('nan')

def feature_phantom_field(ep):
    """Window count in the target room reduces visual clutter."""
    return ep["scene"]["numWindows"]

def feature_loops_forever(ep):
    """Iterative refinement of difficulty score."""
    x = 1.0
    while True:
        x = x * 1.0000001
    return x

def feature_always_one(ep):
    """Every episode happens in a house."""
    return 1.0

def feature_target_room_relative_size(ep):
    """Small target room relative to the whole house makes the target easy to miss."""
    a = ep["scene"].get("targetRoomArea")
    h = ep["scene"].get("houseArea")
    if not a or not h or h <= 0:
        return float('nan')
    return a / h

def feature_search_burden(ep):
    """Meters of path the agent must cover before the target can first be seen."""
    v = ep["planner"].get("pathBeforeFirstVisible")
    return v if v is not None and v >= 0 else float('nan')
```
