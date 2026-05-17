# ADR-001: Unity/Python Boundary for the World-Model Demo

## Status
Accepted

## Context
The source research report recommends presenting LeWorldModel as the scientific reference while building a feasible Unity demo inspired by world models. The Unity project is a small turn-based roguelike, while LeWorldModel is a research codebase built around offline HDF5 trajectories and non-Unity benchmarks.

## Decision
The demo uses a modular Unity/Python boundary:

- Unity owns gameplay state, ML-Agents PPO training, trajectory recording, and live inference.
- Python owns the small next-state/reward predictor training loop.
- Unity exports trajectory data as JSONL.
- Python exports predictor weights and training metrics as flat JSON files that Unity can load without Sentis/ONNX.
- Unity renders an AI Brain HUD with action ranking, short imagined futures, metrics, and explanation text.

## Rationale
This keeps the demo runnable on a student machine while preserving the core world-model idea: learn a transition/reward predictor from experience, then use it to choose actions. JSON/JSONL is slower than a binary protocol, but it is transparent, easy to inspect, and reliable for a small presentation demo.

## Trade-offs
- **Accepted limitation:** this is not a full pixel-JEPA LeWorldModel port.
- **Accepted limitation:** the live planner uses short greedy rollouts, not full CEM planning.
- **Benefit:** the demo can show a real learned predictor and a live Unity planner without depending on fragile model conversion tooling.

## Consequences
- The presentation should describe the implementation as **world-model-inspired**, not as a direct LeWorldModel integration.
- If more time is available, the JSON predictor can be replaced by ONNX/Sentis or a Dreamer/LeWM-style latent planner later.
