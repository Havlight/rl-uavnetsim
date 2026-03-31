# rl-uavnetsim

`rl-uavnetsim` is a standalone, step-based UAV network simulator for MARL research.

This v1 follows `implementation_plan3_5.md` and focuses on:
- multi-UAV trajectory control
- moving ground users
- user-UAV access, UAV-UAV relay, and anchor backhaul
- bit-backlog and relay-queue dynamics
- heuristic PF scheduling with future RL hooks
- metrics, plots, and GIF visualization

This repository intentionally does not implement packet-level or SimPy event-driven logic.

## Scope

- Step-based simulator
- User demand represented by `_bits` backlog
- Multi-agent control currently only for UAV trajectory
- Subchannel allocation currently uses heuristic PF scheduling
- RL hooks reserved for future MAPPO and LinUCB integration

## Project Layout

```text
rl_uavnetsim/
tests/
implementation_plan3_5.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you prefer explicit test and plotting dependencies:

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
source ~/.venv/bin/activate
pytest -q tests
```

## Run Demo

Default demo:

```bash
source ~/.venv/bin/activate
python -m rl_uavnetsim.main --steps 12 --output-dir demo_outputs
```

Stress demo:

```bash
source ~/.venv/bin/activate
python -m rl_uavnetsim.main --demo-mode stress --steps 12 --output-dir demo_outputs_stress
```

## Outputs

The demo runner writes:
- `summary.json`
- `metrics_history.json`
- `plots/*.png`
- `trajectory_final.png`
- `trajectory.gif`

## GitHub Publishing

Create a new empty repository at:

`https://github.com/Havlight/rl-uavnetsim`

Then run:

```bash
git init
git add .
git commit -m "Initial standalone rl_uavnetsim"
git branch -M main
git remote add origin https://github.com/Havlight/rl-uavnetsim.git
git push -u origin main
```

If you use GitHub CLI:

```bash
gh repo create Havlight/rl-uavnetsim --public --source=. --remote=origin --push
```
