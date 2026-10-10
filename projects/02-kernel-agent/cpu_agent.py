#!/usr/bin/env python3
"""Generate and repair NumPy kernel candidates against kernelbench.py.

This is the fast development path for the kernel-agent challenge. It shares the
agent architecture with agent.py, but validates ordinary Python/NumPy kernels in
milliseconds before committing effort to NKI compilation and simulation.
"""

import argparse
import concurrent.futures as cf
import inspect
import json
import os
import re
import sys
import textwrap
import time

import httpx

import kernelbench as kb


MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")
WEIGHTS = {"parses": 0.1, "rules": 0.2, "loads": 0.1, "correct": 0.6}
CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(reply):
    """Return a code block only; prose must not become a fake syntax failure."""
    blocks = CODE_BLOCK.findall(reply or "")
    if blocks:
        return max(blocks, key=len).strip()
    lines = (reply or "").splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |def )", line):
            return "\n".join(lines[i:]).strip()
    return ""


def repair_instruction(feedback):
    """Turn common checker verdicts into one concrete next change."""
    if "calls banned" in feedback:
        name = re.search(r"`([^`]+)`", feedback)
        return (f"Replace the banned `{name.group(1) if name else 'operation'}` call with explicit "
                "loops over tiles. Keep the function signature and all other working code unchanged.")
    if "fancy/boolean indexing" in feedback:
        return ("Replace the fancy or boolean indexing with ordinary slice ranges and scalar index "
                "arithmetic inside the existing tile loops.")
    if "final PARTIAL" in feedback:
        return ("Keep the full-tile logic, but derive each slice end from min(tile_start + tile_size, "
                "actual_dimension) so the final partial tile is processed and written at its real size.")
    if "NON-FINITE OUTPUT" in feedback:
        return ("Make the computation numerically stable. For softmax, compute the row maximum first, "
                "subtract it before exp, and accumulate the denominator with explicit loops.")
    if "WRONG SHAPE" in feedback:
        return "Fix only the output allocation and slice bounds so the returned array has the reference shape."
    if "RAISED:" in feedback:
        return "Fix the exception named by the checker while preserving the required kernel signature."
    if "NUMERICAL MISMATCH" in feedback:
        return ("Fix the arithmetic identified by the mismatch. Preserve tiling and compare intermediate "
                "accumulators with the reference operation before changing unrelated code.")
    return "Fix exactly the checker failure and keep the required kernel signature unchanged."


def grade(source, level, tol):
    """Grade one candidate without subprocesses or lossy CLI-output parsing."""
    parts = {name: False for name in WEIGHTS}
    if not source.strip():
        return 0.0, parts, "No code came back. Reply with one complete Python code block defining `kernel`.", "", None
    try:
        compiled = compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as exc:
        return 0.0, parts, f"The code does not parse: {exc.msg} on line {exc.lineno}.", "", None

    violations = kb.check_rules(source, level)
    if violations:
        feedback = " ".join(violations)
        return WEIGHTS["parses"], parts, feedback, repair_instruction(feedback), None
    parts["rules"] = True

    namespace = {"__name__": "candidate"}
    try:
        exec(compiled, namespace)
    except Exception as exc:
        feedback = f"The file fails while importing: {type(exc).__name__}: {exc}"
        return sum(WEIGHTS[k] for k, ok in parts.items() if ok), parts, feedback, repair_instruction(feedback), None
    kernel = namespace.get("kernel")
    if not callable(kernel):
        feedback = "The file defines no callable function named `kernel`."
        return sum(WEIGHTS[k] for k, ok in parts.items() if ok), parts, feedback, repair_instruction(feedback), None
    parts["loads"] = True

    result = kb.verify(kernel, level, tol=tol, stop_early=False, source=source)
    passed, total = result["passed"], result["total"]
    if result["ok"]:
        parts["correct"] = True
        return sum(WEIGHTS.values()), parts, "Correct on every hostile test case.", "", None
    label, feedback = result["failures"][0]
    feedback = f"{passed} of {total} cases passed. First failure ({label}): {feedback}"
    diagnosis = result["diagnoses"][0]
    reward = sum(WEIGHTS[k] for k, ok in parts.items() if ok)
    reward += WEIGHTS["correct"] * passed / total
    return reward, parts, feedback, diagnosis["instruction"], diagnosis


def first_prompt(level):
    spec = kb.LEVELS[level]
    return (
        f"Write a NumPy function named `kernel` with exactly the same signature as this reference. "
        f"It implements {spec['name']}.\n\n"
        f"```python\n{inspect.getsource(spec['ref']).strip()}\n```\n\n"
        "Use explicit Python loops over tiles no larger than 128 rows by 512 columns. "
        "Reply with one Python code block and no prose."
    )


def repair_prompt(source, feedback, instruction):
    return (
        "This NumPy kernel failed verification.\n\n"
        f"```python\n{source}\n```\n\n"
        f"Checker result:\n{feedback}\n\n"
        f"Required repair: {instruction}\n"
        "Change only what is needed. Reply with one complete Python code block and no prose."
    )


def ask(args, prompt):
    budget = min(args.max_tokens, max(256, args.context - len(prompt) // 4 - 64))
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": budget,
        "temperature": 0.6,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = httpx.post(f"{args.base.rstrip('/')}/chat/completions", json=body, timeout=900, verify=False)
    if response.status_code != 200:
        raise SystemExit(f"Model endpoint returned HTTP {response.status_code}: {response.text[:600]}")
    choice = response.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        print("    response was truncated; reduce prompt size before treating this as a code failure")
    return choice.get("message", {}).get("content") or ""


def offline_reply(level, round_number):
    """Exercise extraction, feedback, repair, and logging without a model; level 1 only."""
    if level != 1:
        return ""
    if round_number == 0:
        return """```python
import numpy as np
def kernel(x, a, b):
    return np.einsum('ij,->ij', a * x + b, 1.0).clip(0)
```"""
    return """```python
import numpy as np
def kernel(x, a, b):
    out = np.empty_like(x)
    rows, cols = x.shape
    for r0 in range(0, rows, 128):
        for c0 in range(0, cols, 512):
            tile = x[r0:r0 + 128, c0:c0 + 512]
            value = a * tile + b
            out[r0:r0 + tile.shape[0], c0:c0 + tile.shape[1]] = np.where(value > 0, value, 0.0)
    return out
```"""


def solve(args, level, log):
    prompt, best, failures = first_prompt(level), (0.0, ""), []
    for round_number in range(args.rounds):
        started = time.perf_counter()
        if args.offline:
            replies = [offline_reply(level, round_number)] * args.samples
        else:
            with cf.ThreadPoolExecutor(max_workers=args.samples) as pool:
                replies = list(pool.map(lambda _: ask(args, prompt), range(args.samples)))
        graded = []
        for reply in replies:
            source = extract_code(reply)
            reward, parts, feedback, instruction, diagnosis = grade(source, level, args.tol)
            graded.append((reward, source, feedback, instruction, parts, diagnosis))
            log.write(json.dumps({"level": level, "round": round_number, "reward": reward,
                                  "parts": parts, "prompt_chars": len(prompt), "reply_chars": len(reply),
                                  "code": source, "feedback": feedback, "instruction": instruction,
                                  "diagnosis": diagnosis}) + "\n")
        log.flush()
        top = max(graded, key=lambda item: item[0])
        best = max(best, (top[0], top[1]), key=lambda item: item[0])
        print(f"round {round_number}: reward {top[0]:.2f}, best {best[0]:.2f} ({time.perf_counter() - started:.2f}s)")
        print(textwrap.indent(top[2][:700], "  "))
        if top[0] >= sum(WEIGHTS.values()) - 1e-9:
            return top[0], round_number + 1
        failures.append(top[2])
        if failures.count(top[2]) >= args.give_up_after:
            print("  stopping: the same checker failure repeated; improve the repair instruction or checker")
            return best[0], round_number + 1
        if not top[1]:
            prompt = first_prompt(level)
        else:
            ledger = "\n".join(f"- {item[:180]}" for item in dict.fromkeys(failures[-3:]))
            prompt = repair_prompt(top[1], top[2], top[3]) + f"\n\nAlready failed:\n{ledger}"
    return best[0], args.rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=sorted(kb.LEVELS), default=1)
    parser.add_argument("--all", action="store_true", help="run all ten CPU challenge levels")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL") or os.environ.get("GPTOSS_BASE_URL"))
    parser.add_argument("--log", default="cpu_attempts.jsonl")
    parser.add_argument("--give-up-after", type=int, default=3)
    parser.add_argument("--offline", action="store_true", help="self-test the loop on level 1 without a model")
    args = parser.parse_args()
    if not args.offline and not args.base:
        sys.exit("Set KERNEL_AGENT_BASE_URL to an OpenAI-compatible /v1 endpoint, or use --offline.")
    levels = sorted(kb.LEVELS) if args.all else [args.level]
    history = {level: [] for level in levels}
    with open(args.log, "a", encoding="utf-8") as log:
        for run in range(args.repeat):
            if args.repeat > 1:
                print(f"\n===== run {run + 1}/{args.repeat} =====")
            for level in levels:
                print(f"\n===== level {level}: {kb.LEVELS[level]['name']} =====")
                score, rounds = solve(args, level, log)
                history[level].append(score)
                print(f"  result: {score:.2f} after {rounds} round(s)")
    if args.repeat > 1:
        for level, scores in history.items():
            solved = sum(score >= 1.0 - 1e-9 for score in scores)
            print(f"level {level}: solved {solved}/{len(scores)}, best {max(scores):.2f}, "
                  f"worst {min(scores):.2f}, mean {sum(scores) / len(scores):.2f}")
    print(f"attempts logged to {args.log}")


if __name__ == "__main__":
    main()
