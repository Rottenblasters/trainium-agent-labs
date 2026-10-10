#!/usr/bin/env python3
"""Deterministic checker for the last-mile delivery challenge.

The model proposes a JSON route plan.  This file is deliberately dependency-free: the
same checker runs on a laptop and in a Trainium seat.  It is the source of truth for
the score; do not infer a score from an attractive-looking route.
"""

import argparse
import json
import math
import re
import sys


DEPOT = (0.0, 0.0)
SPEED_KM_PER_MIN = 0.55


def _order(order_id, x, y, demand, earliest, latest, service=5, zone="north"):
    return {"id": order_id, "location": [x, y], "demand": demand,
            "window": [earliest, latest], "service_min": service, "zone": zone}


def scenario(level):
    """Small, medium, and context-limited instances with fixed, reproducible data."""
    if level == 0:
        orders = [
            _order("D01", 2, 1, 2, 0, 80), _order("D02", 5, 2, 2, 15, 110),
            _order("D03", 7, 1, 1, 25, 135), _order("D04", 4, 6, 2, 20, 145),
            _order("D05", 1, 7, 1, 45, 170), _order("D06", 8, 5, 2, 30, 180),
        ]
        vehicles = [{"id": "van-1", "capacity": 12, "shift": [0, 220]}]
    elif level in (1, 2):
        per_zone = 6 if level == 1 else 8
        centers = [(8, 4, "east"), (4, 10, "north"), (-7, 5, "west")]
        orders = []
        for z, (cx, cy, zone) in enumerate(centers[:2 if level == 1 else 3]):
            for i in range(per_zone):
                # A stable cloud around each neighborhood.  Windows overlap but later stops
                # become progressively less forgiving, which makes route order matter.
                x = cx + ((i * 3 + z) % 5) - 2
                y = cy + ((i * 2 + z) % 5) - 2
                earliest = 10 * (i % 3)
                latest = 190 + 12 * i + 15 * z
                orders.append(_order(f"D{len(orders) + 1:02d}", x, y,
                                     1 + ((i + z) % 3), earliest, latest, 4, zone))
        vehicles = [
            {"id": "van-1", "capacity": 14 if level == 1 else 16, "shift": [0, 360]},
            {"id": "van-2", "capacity": 14 if level == 1 else 16, "shift": [0, 360]},
        ]
        if level == 2:
            vehicles.append({"id": "van-3", "capacity": 16, "shift": [0, 360]})
    else:
        raise ValueError("level must be 0, 1, or 2")
    return {"name": f"delivery-level-{level}", "depot": list(DEPOT),
            "speed_km_per_min": SPEED_KM_PER_MIN, "vehicles": vehicles, "orders": orders}


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compact_brief(data):
    """The small context given first to the model. Locations require a lookup on level 2."""
    lines = ["Vehicles: " + ", ".join(
        f"{v['id']} capacity={v['capacity']} shift={v['shift'][0]}-{v['shift'][1]}"
        for v in data["vehicles"])]
    for o in data["orders"]:
        lines.append(f"{o['id']}: demand={o['demand']}, window={o['window'][0]}-{o['window'][1]}, "
                     f"zone={o['zone']}")
    return "\n".join(lines)


def lookup(data, request):
    """Return details for at most eight named stops; this models bounded retrieval."""
    ids = re.findall(r"D\d{2}", request.upper())[:8]
    by_id = {o["id"]: o for o in data["orders"]}
    found = [by_id[x] for x in ids if x in by_id]
    if not found:
        return "No known delivery IDs were requested. Use e.g. LOOKUP: D01,D02."
    return json.dumps(found, separators=(",", ":"))


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Reply with one JSON object, not prose or a code fragment.")
    return json.loads(text[start:end + 1])


def simulate(data, plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("routes"), list):
        raise ValueError('The JSON must have a "routes" list.')
    orders = {o["id"]: o for o in data["orders"]}
    vehicles = {v["id"]: v for v in data["vehicles"]}
    assigned, reports, violations = set(), [], []
    total_distance, on_time = 0.0, 0

    seen_vehicles = set()
    for route in plan["routes"]:
        if not isinstance(route, dict):
            violations.append("Each route must be an object with vehicle and stops.")
            continue
        vid, stops = route.get("vehicle"), route.get("stops")
        if vid not in vehicles:
            violations.append(f"Unknown vehicle {vid!r}; use one of {sorted(vehicles)}.")
            continue
        if vid in seen_vehicles:
            violations.append(f"Vehicle {vid} appears twice; combine its stops into one route.")
            continue
        seen_vehicles.add(vid)
        if not isinstance(stops, list):
            violations.append(f"Route for {vid} needs a list called stops.")
            continue
        v, pos, minute, load = vehicles[vid], DEPOT, float(vehicles[vid]["shift"][0]), 0
        route_stops, route_distance = [], 0.0
        for oid in stops:
            if oid not in orders:
                violations.append(f"{vid} includes unknown stop {oid!r}.")
                continue
            if oid in assigned:
                violations.append(f"{oid} is delivered more than once; keep it on one route.")
                continue
            o = orders[oid]
            assigned.add(oid)
            load += o["demand"]
            leg = distance(pos, o["location"])
            minute += leg / SPEED_KM_PER_MIN
            route_distance += leg
            arrival = minute
            if minute < o["window"][0]:
                minute = float(o["window"][0])
            late = max(0.0, minute - o["window"][1])
            if late <= 1e-9:
                on_time += 1
            route_stops.append({"id": oid, "arrival": round(arrival, 1),
                                "service_start": round(minute, 1), "late_min": round(late, 1)})
            minute += o["service_min"]
            pos = o["location"]
        return_leg = distance(pos, DEPOT) if route_stops else 0.0
        finish = minute + return_leg / SPEED_KM_PER_MIN
        route_distance += return_leg
        total_distance += route_distance
        if load > v["capacity"]:
            violations.append(f"{vid} carries {load} units but its capacity is {v['capacity']}; "
                              "move one or more stops to another van.")
        if finish > v["shift"][1] + 1e-9:
            violations.append(f"{vid} returns at minute {finish:.1f}, after its shift ends at "
                              f"{v['shift'][1]}; shorten or split this route.")
        reports.append({"vehicle": vid, "load": load, "finish": round(finish, 1),
                        "distance_km": round(route_distance, 2), "stops": route_stops})

    missed = [oid for oid in orders if oid not in assigned]
    return {"reports": reports, "violations": violations, "missing": missed,
            "delivered": len(assigned), "on_time": on_time, "total_distance": total_distance}


def score(data, plan):
    """Graded reward: syntax 0.05, feasible routes 0.25, coverage 0.30, punctuality 0.25, efficiency 0.15."""
    result = simulate(data, plan)
    n = len(data["orders"])
    delivered = result["delivered"]
    feasible = not result["violations"]
    coverage = delivered / n
    punctuality = result["on_time"] / delivered if delivered else 0.0
    # Six km/order is a deliberately attainable service envelope for these compact synthetic
    # neighborhoods. The term still differentiates wildly wasteful routes, but a fully valid
    # baseline can reach 1.0 rather than making "solved" impossible by construction.
    efficiency = min(1.0, (6.0 * n) / max(6.0 * n, result["total_distance"])) if delivered else 0.0
    reward = 0.05 + 0.25 * feasible + 0.30 * coverage + 0.25 * punctuality + 0.15 * efficiency
    if result["violations"]:
        feedback = result["violations"][0]
    elif result["missing"]:
        feedback = (f"The routes are feasible but {len(result['missing'])} deliveries are missing: "
                    f"{', '.join(result['missing'][:8])}. Assign them to a van with spare capacity.")
    else:
        late = [s for r in result["reports"] for s in r["stops"] if s["late_min"] > 0]
        if late:
            s = late[0]
            feedback = (f"All deliveries are assigned, but {s['id']} starts {s['late_min']:.1f} minutes "
                        "late. Visit it earlier or give it another vehicle.")
        else:
            feedback = "Valid plan: every delivery is assigned and on time."
    return {"reward": round(reward, 4), "feedback": feedback, "result": result,
            "parts": {"valid_routes": feasible, "coverage": round(coverage, 3),
                      "punctuality": round(punctuality, 3), "efficiency": round(efficiency, 3)}}


def is_operationally_solved(data, verdict):
    """Success is feasibility, complete coverage, and punctual service; distance is a tie-breaker."""
    result = verdict.get("result", {})
    return (not result.get("violations") and not result.get("missing")
            and result.get("on_time") == len(data["orders"]))


def greedy_plan(data):
    """A transparent feasible baseline, not an optimal solver.

    It preserves neighborhood order when capacity permits and otherwise moves an order to the
    least-loaded compatible van.  It exists to prove the checker, not to be a target solution.
    """
    routes = [{"vehicle": v["id"], "stops": []} for v in data["vehicles"]]
    capacity = {v["id"]: v["capacity"] for v in data["vehicles"]}
    used = {v["id"]: 0 for v in data["vehicles"]}
    by_zone = {}
    for o in data["orders"]:
        by_zone.setdefault(o["zone"], []).append(o)
    preferred = list(data["vehicles"])
    for i, group in enumerate(by_zone.values()):
        preferred_id = preferred[i % len(preferred)]["id"]
        # Largest parcels first avoids stranding capacity; append them back in deadline order later.
        for o in sorted(group, key=lambda item: -item["demand"]):
            candidates = [r for r in routes if used[r["vehicle"]] + o["demand"] <= capacity[r["vehicle"]]]
            chosen = next((r for r in candidates if r["vehicle"] == preferred_id),
                          min(candidates, key=lambda r: used[r["vehicle"]]))
            chosen["stops"].append(o["id"])
            used[chosen["vehicle"]] += o["demand"]
    deadlines = {o["id"]: o["window"][1] for o in data["orders"]}
    for route in routes:
        route["stops"].sort(key=lambda oid: deadlines[oid])
    return {"routes": routes}


def selftest():
    for level in range(3):
        data = scenario(level)
        good = score(data, greedy_plan(data))
        assert good["parts"]["coverage"] == 1.0, good
        assert good["parts"]["valid_routes"], good
        assert is_operationally_solved(data, good), good
        bad = score(data, {"routes": [{"vehicle": "van-1", "stops": ["D01", "D01"]}]})
        assert bad["result"]["violations"], bad
    print("selftest passed: checker catches duplicate stops and scores a baseline route")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--show", action="store_true", help="print the full scenario JSON")
    ap.add_argument("--brief", action="store_true", help="print the context-limited catalog")
    ap.add_argument("--reference", action="store_true", help="print the baseline heuristic plan")
    ap.add_argument("--check", metavar="PATH", help="score a candidate JSON file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    data = scenario(a.level)
    if a.show:
        print(json.dumps(data, indent=2))
    elif a.brief:
        print(compact_brief(data))
    elif a.reference:
        print(json.dumps(greedy_plan(data), indent=2))
    elif a.check:
        with open(a.check) as f:
            plan = json.load(f)
        print(json.dumps(score(data, plan), indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
