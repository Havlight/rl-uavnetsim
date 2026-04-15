# Coverage-Challenging Scenario Design

**Date:** 2026-04-15  
**Repo:** `standalone_rl_uavnetsim`

## Goal
Create a satellite scenario where `coverage_ratio` is meaningfully below `1.0` without collapsing the whole network, so UAV placement and movement have research value instead of being trivially feasible everywhere on the map.

## Problem Statement
The current simulator almost always reports `coverage_ratio = 1.0` in the satellite demos and training runs. In this codebase, coverage is defined as the fraction of ground users whose `associated_uav_id >= 0`. That means coverage only drops when the association stage decides a user has no feasible UAV.

Under the current A2G link budget and association rule, feasibility is too permissive:

- association uses `proxy_rate_bps = upper_bound_rate_bps / projected_load`
- feasibility requires only `proxy_rate_bps >= min_rate_bps`
- `upper_bound_rate_bps` is interference-free and uses the current A2G RF parameters
- `min_rate_bps` is currently tied to the global `config.R_MIN`

With the existing RF model, simply moving users toward map edges is not enough to break feasibility. Even users near the map corners can still satisfy the current association gate. As a result, changing user distributions alone will not produce `coverage_ratio < 1.0`.

## Key Findings
### 1. Geometry alone is insufficient
Adding new user distributions such as edge-biased or corner-hotspot placement does not solve the problem by itself. The current RF and association settings make most of the 2000m x 2000m map feasible even under moderate load.

### 2. The main lever is association feasibility
If the goal is to reduce coverage, the simulator needs a stronger or more explicit association gate. The cleanest way to do that is to make the minimum acceptable association rate configurable per scenario rather than globally fixed.

### 3. Outage and association should not share the same threshold
`config.R_MIN` is currently used both for outage reporting and association feasibility. Those are different concepts:

- outage threshold: whether a served user is getting acceptable service
- association threshold: whether a user is allowed to attach at all

If coverage needs to drop while outage semantics stay stable, these thresholds should be separated.

### 4. Association ordering amplifies coverage sensitivity
`associate_users_to_uavs()` processes users in descending backlog order. That means high-backlog users get first access to UAV capacity, while lower-backlog and farther users are evaluated later against already-loaded UAVs.

This is important for the design:

- coverage failure will not be random
- users that are both farther away and evaluated later are more likely to lose feasibility
- UAV movement can therefore matter in a useful way, because geometry affects which users remain feasible once load grows

This interaction is desirable, but it should be documented and preserved intentionally rather than treated as incidental behavior.

## Evaluated Approaches
### Approach A: Add more user distribution modes only
Examples: `full_map_uniform`, `edge_biased`, `corner_hotspots`

**Pros**
- intuitive
- easy to visualize

**Cons**
- does not actually reduce coverage under current RF parameters
- adds scenario surface area without addressing the real bottleneck

**Decision**
- reject as the primary solution
- keep only the minimal geometry control that is still useful for experiments

### Approach B: Hard distance cap for association
Add a `max_association_range_m` gate to association.

**Pros**
- very simple
- guarantees coverage can drop below 1
- easy to tune

**Cons**
- duplicates information already implicit in the RF model
- more heuristic than the current rate-based association logic

**Decision**
- keep as a fallback or ablation option
- not the preferred mainline design

### Approach C: Scenario-specific association QoS gate plus wider spawn support
Introduce a scenario-configurable `association_min_rate_bps` and allow users to spawn closer to the full map extent via `spawn_margin`.

**Pros**
- directly targets the real cause of always-on coverage
- preserves the existing rate-based association model
- more defensible as a research parameter: minimum QoS requirement for admission
- pairs naturally with scenario configuration and training YAMLs

**Cons**
- requires plumbing an extra parameter through the simulator path
- still needs calibration

**Decision**
- recommended

## Chosen Design
### 1. Add `spawn_margin` as a geometry control
`build_demo_entities()` should allow configuring how close users may spawn to the map edges.

Current uniform sampling uses a hard-coded `10%..90%` interior band. The new design makes this explicit:

- `spawn_margin = 0.1` reproduces the current behavior
- `spawn_margin = 0.02` is the recommended first calibration value, allowing near-full-map placement while avoiding exact edge placements

This parameter is useful, but it is not expected to reduce coverage by itself. Its role is to create more spatially demanding scenarios once the association gate is made stricter.

### 2. Split association admission threshold from outage threshold
Introduce a new configurable threshold for association feasibility, conceptually:

- `association_min_rate_bps`: used only in `associate_users_to_uavs()`
- `R_MIN` remains the outage/reporting threshold unless explicitly changed later

This keeps metric semantics stable while allowing the simulator to represent a stronger minimum admission QoS requirement.

The default should preserve current behavior:

- `association_min_rate_bps` defaults to `config.R_MIN`

Coverage-challenging scenarios can then raise it without changing the outage definition.

The recommended first calibration value is:

- `association_min_rate_bps = 3.0e6`

For the current satellite experiments, this should be strong enough to make farther users lose feasibility once UAV load rises, without collapsing the whole scenario.

### 3. Prefer explicit scenario config over more presets
The main path should be explicit config fields in YAML / training config rather than adding many new `demo_mode` variants. That keeps scenarios inspectable and easier to reason about in experiments.

The important new knobs are:

- `spawn_margin`
- `association_min_rate_bps`

These should be YAML-facing and configurable in both demo and training paths. The first implementation should make at least the following scenario controls explicit and overridable:

- `user_demand_rate_bps`
- `orbit_radius_m`
- `user_speed_mean_mps`
- `user_distribution`
- `spawn_margin`
- `association_min_rate_bps`

These should be added to:

- demo/scenario generation for demo runs
- training env config for MAPPO experiments

### 4. Keep hard max association range as optional future fallback
If rate-based admission plus `spawn_margin` still cannot reliably produce the target coverage range, a future optional `max_association_range_m` may be added. This should be treated as a secondary, explicit approximation rather than the first-line solution.

## Target Experimental Outcome
The first calibrated coverage-challenging scenario should aim for:

- `mean_coverage_ratio` clearly below `1.0`
- not catastrophically low; target roughly `0.8 .. 0.95`
- nontrivial spatial dependence, so UAV movement can improve or worsen coverage
- no collapse into a uniformly disconnected or uniformly overloaded scenario

The initial characterization target should use:

- `num_uavs = 3`
- `num_users = 30`
- `spawn_margin = 0.02`
- `association_min_rate_bps = 3.0e6`

That combination is the starting point for calibration and should be explicitly tested rather than left implicit.

## Scope
### In scope
- scenario generation controls that expand user placement toward map edges
- association feasibility controls that are scenario-configurable
- calibration tests to show coverage can drop below 1
- YAML-facing knobs for training/demos

### Out of scope
- changing the definition of `coverage_ratio`
- redesigning the entire A2G channel model
- adding multiple new user distribution families unless calibration proves they are needed
- changing compact MAPPO observations in this round

## Validation Strategy
The implementation should prove three things:

1. user generation can place users nearer the map edges than before
2. association feasibility is now tunable per scenario
3. at least one calibrated satellite scenario yields `coverage_ratio < 1.0` without making the scenario degenerate

This should be locked down with characterization tests rather than only visual inspection.

At minimum, characterization should confirm that the initial calibration scenario above produces `mean_coverage_ratio` in the target band `0.8 .. 0.95`.

## Plumbing Scope
The implementation should not invent a new path if the parameter plumbing already exists in pieces. The new association threshold needs to travel through the existing call chain:

- `SimEnv.__init__()` should own a scenario-level `association_min_rate_bps`
- `SimEnv.step()` should pass that value into `associate_users_to_uavs(...)`
- `MultiAgentUavNetEnv.reset()` also performs an initial association pass and must use the same threshold
- `build_training_env()` should read the threshold from `RunConfig.env`
- `build_demo_entities()` should read `spawn_margin`
- training/demo YAMLs should surface the new scenario knobs

The design goal is to make most scenario-facing parameters configurable without exploding the simulator surface. Scenario configuration should be broad; deep RF/model internals remain out of scope for this round.

## Baseline Note
While preparing this design, the current `main` branch baseline in the isolated worktree showed one pre-existing failing test:

- `tests/test_training_features.py::test_load_run_config_accepts_explicit_training_scenario_fields`

The failure was caused by a mismatch between the expected explicit training scenario values and the checked-in `mappo_satellite_3uav_medium.yaml`. The implementation plan should treat restoration of a clean baseline as an early prerequisite before adding the new coverage-calibration changes.
