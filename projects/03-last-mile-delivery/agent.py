#!/usr/bin/env python3
"""LLM/checker repair loop for RouteBench.

The model first sees a compact delivery catalog.  It can issue `LOOKUP: D01,D02`
for exact stop locations, then must return a JSON route plan.  Every candidate and
checker result goes to JSONL so the trajectory is the experiment artifact.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import routebench


def ask(a, prompt):
    import httpx
    estimated_prompt_tokens = len(prompt) // 4
    budget = min(a.max_tokens, max(256, a.context - estimated_prompt_tokens - 64))
    payload = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": budget, "temperature": 0.6, "top_p": 0.95,
               "chat_template_kwargs": {"enable_thinking": False}}
    response = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=payload,
                          timeout=900, verify=False)
    if response.status_code != 200:
        raise RuntimeError(f"model returned HTTP {response.status_code}: {response.text[:500]}")
    choice = response.json()["choices"][0]
    text = choice.get("message", {}).get("content") or ""
    if choice.get("finish_reason") == "length":
        print("    truncated answer: shorten the repair prompt or increase server context")
    return text


def initial_prompt(data, level):
    catalog = routebench.compact_brief(data)
    if level == 0:
        details = json.dumps(data["orders"], separators=(",", ":"))
        catalog += "\n\nExact stop details:\n" + details
    return f"""You are dispatching last-mile delivery vans from depot {data['depot']}.
Travel speed is {data['speed_km_per_min']} km/minute.  Every delivery must be made exactly once.
Return ONLY this JSON shape: {{\"routes\":[{{\"vehicle\":\"van-1\",\"stops\":[\"D01\"]}}]}}.
Time windows are [earliest, latest] service-start minutes; vehicles must return before shift end.

Compact catalog:
{catalog}

For levels 1 and 2, locations are intentionally omitted. Before creating the final plan, you may
reply exactly `LOOKUP: D01,D02` (at most eight IDs) to retrieve their locations. Otherwise reply
with the route-plan JSON only."""


def complete_attempt(a, data, prompt):
    """Allow a bounded amount of selective retrieval before the final structured answer."""
    used = 0
    for _ in range(a.tool_steps + 1):
        reply = ask(a, prompt)
        if reply.strip().upper().startswith("LOOKUP:") and used < a.tool_steps:
            details = routebench.lookup(data, reply)
            prompt += f"\n\nRequested stop details:\n{details}\nNow reply with another LOOKUP or the final JSON plan."
            used += 1
            continue
        return reply, used
    return reply, used


def grade_text(data, text):
    try:
        plan = routebench.extract_json(text)
        verdict = routebench.score(data, plan)
        return verdict, plan
    except (ValueError, json.JSONDecodeError) as e:
        return ({"reward": 0.0, "parts": {}, "result": {},
                 "feedback": f"The plan is not valid JSON: {e} Send only one complete JSON object."}, None)


def repair_prompt(plan, feedback):
    return f"""Repair this delivery route plan:
```json
{json.dumps(plan, separators=(',', ':'))}
```
The simulator says: {feedback}
Make the smallest change that fixes that specific problem. Reply ONLY with the complete JSON plan."""


def offline_reply(data, round_number):
    plan = routebench.greedy_plan(data)
    if round_number == 0:
        # Deliberately omit a stop so the offline path proves that feedback travels into repair.
        plan["routes"][0]["stops"] = plan["routes"][0]["stops"][:-1]
    return json.dumps(plan)


def solve(a, level, log):
    data = routebench.scenario(level)
    prompt, best = initial_prompt(data, level), (0.0, None, "")
    feedback_history = []
    print(f"\n=========== delivery level {level}: {len(data['orders'])} orders ===========")
    for rnd in range(a.rounds):
        start = time.perf_counter()
        if a.offline:
            replies = [offline_reply(data, rnd)] * a.samples
            used_tools = [0] * a.samples
        else:
            with cf.ThreadPoolExecutor(max_workers=a.samples) as pool:
                futures = [pool.submit(complete_attempt, a, data, prompt) for _ in range(a.samples)]
                pairs = [f.result() for f in futures]
            replies, used_tools = zip(*pairs)
        graded = []
        for text, tools_used in zip(replies, used_tools):
            verdict, plan = grade_text(data, text)
            graded.append((verdict, plan))
            log.write(json.dumps({"level": level, "round": rnd, "prompt_chars": len(prompt),
                                  "reply": text, "plan": plan, "tool_calls": tools_used,
                                  "reward": verdict["reward"], "parts": verdict["parts"],
                                  "feedback": verdict["feedback"]}) + "\n")
        log.flush()
        verdict, plan = max(graded, key=lambda pair: pair[0]["reward"])
        if verdict["reward"] > best[0]:
            best = (verdict["reward"], plan, verdict["feedback"])
        print(f"round {rnd}: best {verdict['reward']:.3f}; best-ever {best[0]:.3f}; "
              f"{sum(used_tools)} lookup(s); {time.perf_counter() - start:.1f}s")
        print(f"  checker: {verdict['feedback']}")
        if routebench.is_operationally_solved(data, verdict):
            print("  SOLVED: valid, complete, on-time route plan")
            return best[0], rnd + 1
        if plan is None:
            prompt += "\n\nYour previous response was invalid. Return the required JSON object now."
        else:
            feedback_history.append(verdict["feedback"])
            ledger = "\n".join(f"- {x[:180]}" for x in dict.fromkeys(feedback_history[-3:]))
            prompt = repair_prompt(plan, verdict["feedback"]) + "\nEarlier failed checks:\n" + ledger
    return best[0], a.rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--tool-steps", type=int, default=4)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--model", default=os.environ.get("DELIVERY_MODEL", os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")))
    ap.add_argument("--base", default=os.environ.get("DELIVERY_BASE_URL", os.environ.get("KERNEL_AGENT_BASE_URL")))
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    if not a.offline and not a.base:
        sys.exit("Set DELIVERY_BASE_URL (or KERNEL_AGENT_BASE_URL) to the local vLLM /v1 endpoint, or pass --offline.")
    levels = (0, 1, 2) if a.all else (a.level,)
    if a.offline:
        print("*** OFFLINE: replaying a heuristic; do not report its score as model performance. ***")
    with open(a.log, "a") as log:
        results = [(level,) + solve(a, level, log) for level in levels]
    print("\n=========== summary ===========")
    for level, reward, rounds in results:
        print(f"level {level}: reward {reward:.3f} after {rounds} round(s)")
    print(f"attempts logged to {a.log}")


if __name__ == "__main__":
    main()
