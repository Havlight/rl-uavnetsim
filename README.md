# rl-uavnetsim

`rl-uavnetsim` is a standalone, step-based UAV network simulator for MARL research.

The current `main` branch includes:
- multi-UAV trajectory control
- moving ground users
- user-UAV access, UAV-UAV relay, and satellite backhaul
- backlog and relay-queue dynamics
- heuristic association + PF scheduling
- compact-observation MAPPO training integration with `compact_v1` / `compact_v2`
- episode-aware evaluation artifacts with static-baseline comparison
- metrics plots and trajectory GIF visualization

This repository is intentionally a **step-based simulator**, not a packet-level or SimPy event-driven network simulator.

## Branches

- `main`: latest codebase, including MAPPO training/evaluation integration
- `v2`: pre-MAPPO stable version before the training stack was added
- `v1`: initial standalone simulator baseline

## Scope

- Step-based environment dynamics
- Continuous UAV movement actions (`rho`, `psi`)
- Heuristic inner networking stack:
  - user association
  - access PF scheduling
  - relay routing
  - satellite backhaul
- MARL currently controls **UAV trajectory only**

## Project Layout

```text
rl_uavnetsim/
  allocation/
  channel/
  entities/
  environment/
  metrics/
  mobility/
  network/
  rl_interface/
  training/
  visualization/
configs/
  marl/
tests/
```

## Installation

Core simulator only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

With MARL training dependencies:

```bash
pip install -e '.[marl]'
```

## Run Tests

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m pytest -q tests
```

## Run Demo

Default satellite demo:

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.main --steps 12 --backhaul-type satellite --output-dir demo_outputs
```

Stress satellite demo:

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.main --demo-mode stress --backhaul-type satellite --steps 50 --num-users 100 --output-dir demo_stress_satellite
```

## Train MAPPO

Training uses:
- `PettingZoo ParallelEnv`
- `TorchRL`
- compact geometry-first observation presets
- centralized critic over concatenated agent observations
- static movement baseline diagnostics during evaluation

Start training:

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite.yaml
```

Harder 3-UAV scenarios:

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite_3uav_hotspot_hard.yaml
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite_3uav_large_map.yaml
```

Run evaluation from a checkpoint:

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.evaluate --config configs/marl/eval_satellite.yaml --checkpoint runs/mappo_satellite_v1/<timestamp>/checkpoints/best.pt
```

Analyze a training run:

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.analyze_run --run-dir runs/<experiment>/<timestamp>
```

## Outputs

The demo runner writes artifacts such as:
- `summary.json`
- `metrics_history.json`
- `plots/*.png`
- `trajectory_final.png`
- `trajectory.gif`

The evaluator writes schema-v2 artifacts under `eval/latest/` and `eval/best/` during training:
- nested `summary.json` with `policies.trained`, `policies.static`, and `comparison`
- `episodes/episode_000/summary.json`
- `episodes/episode_000/metrics_history.json`
- `episodes/episode_000/plots/*.png`
- `episodes/episode_000/trajectory_final.png`
- `episodes/episode_000/trajectory.gif`

The MAPPO trainer also writes:
- `config_resolved.yaml`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `tensorboard/`
- `train_metrics.jsonl`

## Notes

- MAPPO controls only UAV movement; it does **not** replace association, PF scheduling, routing, or backhaul logic.
- `compact_v1` is kept for checkpoint compatibility. `compact_v2` adds local backlog, local association flags, and self associated-user count.
- `runs/` is ignored by git and is intended for local experiment outputs.
