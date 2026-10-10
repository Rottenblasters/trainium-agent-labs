# Project 3 — Context-limited last-mile delivery

An LLM dispatches delivery vans, a deterministic simulator grades the plan, and the simulator's
specific reason becomes the next repair prompt. The task is a simplified version of the route
planning used by courier, food-delivery, and field-service fleets.

The model must assign every delivery exactly once while respecting vehicle capacity, shifts, travel
time, and delivery windows. Its output is one JSON object:

```json
{"routes":[{"vehicle":"van-1","stops":["D01","D02"]}]}
```

## What is measured

`routebench.py` is the judge. It calculates a graded reward:

- 0.05: valid JSON
- 0.25: no duplicate stops, unknown vehicles, capacity overflow, or missed shift
- 0.30: fraction of deliveries assigned
- 0.25: fraction of assigned deliveries served within their time window
- 0.15: route-distance efficiency

The score comes only from the deterministic simulator. A plausible route is not a result until the
checker accepts it.

Three levels make the effect of context visible:

- Level 0: six stops and one vehicle; all coordinates are in the initial prompt.
- Level 1: twelve stops and two vehicles; the model sees a compact catalog and looks up locations.
- Level 2: twenty-four stops and three vehicles; selective retrieval and compact failure memory matter.

The agent can reply `LOOKUP: D01,D02` before supplying a final plan. Each lookup returns details for
at most eight stops. That is the controlled context constraint: the complete world is not repeatedly
copied into each prompt.

## Validate the checker

```bash
cd /workspace/projects/03-last-mile-delivery
python3 routebench.py --selftest
python3 routebench.py --level 1 --show
python3 routebench.py --level 1 --reference
```

The reference is a transparent zone-based heuristic, not an optimal solution. It is useful to prove
the judge and to create a baseline; do not present it as model performance.

## Run the agent on Trainium

Start the local model in one pod shell:

```bash
cd /workspace
MAX_MODEL_LEN=8192 ./serve.sh
```

Then, in a second shell in the same pod:

```bash
cd /workspace/projects/03-last-mile-delivery
export DELIVERY_BASE_URL=http://localhost:8000/v1
export DELIVERY_MODEL=Qwen/Qwen3-8B
nohup python3 agent.py --all --rounds 6 --samples 4 --context 8192 > run.log 2>&1 < /dev/null &
tail -f run.log
```

`attempts.jsonl` records prompt size, selective lookups, candidate plan, score components, and
checker feedback. It is the evidence for a presentation: compare a one-shot attempt (`--rounds 1`)
against the repair loop, and report a rate over repeated runs rather than a lucky best run.

## What to show

Show a failed first route, the checker's named conflict, and the repair that fixes it. Then plot
reward by round, coverage/on-time rate, lookup count, and prompt size. The useful claim is not that
the model can draw routes; it is that under a bounded context it retrieves evidence, uses simulation
feedback, and improves a measurable operational plan.
