# Kernel Agent: Beginner Guide

## The project in one sentence

This project is trying to build an AI assistant that can write a tiny, low-level program for an AWS AI chip, test whether that program is legal and correct, explain any failure, and try again.

## What is a kernel?

When you write a familiar operation such as matrix multiplication, softmax, or transpose in PyTorch or NumPy, a lower-level program eventually does the real work on the CPU, GPU, or AI accelerator. That lower-level program is a **kernel**.

A kernel author must decide details that normal Python hides:

- which small piece, or **tile**, of a large array to work on;
- when to copy that tile from slower external memory into fast on-chip memory;
- where temporary results live;
- how to handle the final tile when the input size is not an exact multiple of the tile size; and
- how to avoid repeatedly moving the same data, because moving data can be the slow part.

NKI (Neuron Kernel Interface) is AWS's Python-based language for writing these kernels for Trainium/Neuron hardware.

## The actual problem the team is solving

The task is not simply "ask a model to write code." A generated kernel can look plausible, run without crashing, and still return wrong numbers. It can also be numerically correct while breaking kernel rules or moving too much data.

The intended agent loop is:

1. Give a model an operation to implement.
2. Extract the kernel code from the model's reply.
3. Check its source for obvious cheating or invalid patterns.
4. Run it in the NKI simulator and compare its output with a trusted NumPy answer.
5. Diagnose the failure in a way that tells the model what to change.
6. Give the code and that diagnosis back to the model for a repair attempt.
7. Save every attempt and score, then measure success over repeated runs.

The difficult research/design question is: **what feedback and context should the agent give the model so it fixes one precise problem rather than making a different mistake?** The model has a limited context window, so it cannot be shown every API document, rule, old attempt, error, and reference implementation on every call.

## Two related tracks

### 1. The laptop track: `kernelbench.py`

This is the recommended starting point. Kernels are ordinary NumPy/Python functions, but they must follow kernel-like rules: fixed-size tiles, explicit loops, no whole-array shortcut for the operation, and correct handling of partial tiles.

It runs quickly on a CPU and has ten exercises, from an elementwise ReLU through softmax, tiled matrix multiplication, attention, and convolution. It lets a team improve the agent and checker without needing the Neuron SDK or a chip.

### 2. The NKI track: `nkibench.py` and `agent.py`

This is the Trainium-specific version. The model writes real NKI code. The harness checks the source, uses the NKI CPU simulator to execute it, compares the result with NumPy, and counts external-memory transfers to estimate whether a matrix multiplication wastes memory bandwidth.

The shipped NKI ladder is:

1. Average pooling
2. Transpose
3. One-tile matrix multiplication
4. Tiled matrix multiplication
5. Matrix multiplication with loads hoisted out of the inner loop
6. Matrix multiplication with M and N blocking
7. Fully blocked matrix multiplication
8. Single-head attention, as an example of adding a new operation

The first four have reference NKI kernels in this folder. Levels 5 to 7 define progressively stricter memory-traffic targets but do not ship reference kernels.

## What "success" means

For a candidate kernel to be trusted, it must pass each gate:

1. **Parse:** it is valid Python.
2. **Rules:** it has the required entry-point name and `@nki.jit`, does not call a high-level operation that solves the task for it, and follows visible tile constraints.
3. **Run:** the simulator can load and execute it.
4. **Numerics:** it matches the trusted NumPy reference on every test shape, including awkward edge sizes.
5. **Efficiency, for advanced levels:** it moves no more than the allowed amount of HBM memory.

A partial score distinguishes an unusable answer from one that runs but calculates the wrong result. A full score is `1.0`.

One successful run is not enough when model sampling is enabled. The agent supports `--repeat N` so results can be reported as a solve rate, best/worst score, and mean score.

## Why the checker matters more than the prompt

This repository's central lesson is that a verdict alone is weak feedback. For example:

- Weak: "Numerical mismatch."
- Useful: "The error is in the final partial tile, so derive the final tile's bounds from the input shape instead of assuming a full tile."

The project also found that giving a model a long list of rules can cause it to spend all its output budget reasoning about the rules and return no code. The preferred strategy is therefore:

1. ask for a simple first attempt;
2. let the verifier find the most important violation;
3. give one specific repair instruction; and
4. keep a compact ledger of failures so the agent does not loop on the same mistake.

## Practical starting commands

The CPU-only path needs NumPy:

```bash
cd projects/02-kernel-agent
python kernelbench.py --selftest
python kernelbench.py --list
python kernelbench.py --level 1 --show
```

The NKI path needs the AWS Neuron SDK in addition to the repository dependencies:

```bash
python nkibench.py --selftest
python nkibench.py --level 4 --check reference_level4.py
python agent.py --offline --level 1
```

`--offline` replays a deliberately broken then correct reference kernel. It tests the loop and logging, not the model's capability. To use a live model, the workshop pod provides `KERNEL_AGENT_BASE_URL` and `KERNEL_AGENT_MODEL`; then run, for example:

```bash
python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5
```

This writes every candidate, score, and feedback message to `attempts.jsonl` by default.

## Current project state

The verification harness, reference kernels for NKI levels 1 to 4, and the repair loop are present and runnable. The project is intentionally unfinished: generated NKI code commonly reaches the point where it parses, obeys the static rules, and runs, but produces incorrect numbers. The useful hackathon work is to improve the checker feedback, context selection, repair policy, test coverage, or kernel tasks, then show measured results honestly.

The CPU-only challenge is the fastest place to begin. It uses the same agent/checker idea while making failures much quicker to reproduce and understand.
