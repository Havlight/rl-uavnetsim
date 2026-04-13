---
marp: true
theme: default
paginate: true
---

# `rl-uavnetsim` Satellite Demo Report

Professor-facing research briefing  
Case study: `demo_stress_satellite`

`python -m rl_uavnetsim.main --demo-mode stress --backhaul-type satellite --steps 50 --num-users 100 --output-dir ./demo_stress_satellite`

---

# 1. Problem and Goal

- This system aims to provide a `step-based` UAV network simulator for multi-UAV coordination and RL research.
- The current target problem is: how multiple UAVs move in continuous space, serve mobile ground users, and forward data through UAV relay and `satellite backhaul`.
- The current simulator explicitly models:
  - UAV trajectory control
  - user-UAV access
  - UAV-UAV relay
  - gateway-to-satellite backhaul
  - backlog / queue dynamics
- It is not a packet-level simulator and not a SimPy / event-driven platform; it is a step-and-slot-based research abstraction.

---

# 2. System Architecture

```text
Ground Users
    |
    v
Access UAVs / Relay UAVs
    |
    v
gateway-capable UAV
    |
    v
Satellite
```

- Ground users upload data to UAVs through access links.
- Non-gateway UAVs only relay traffic; forwarding can be multi-hop.
- The `gateway-capable UAV` is the egress node. In the current demo there is only one such UAV, located near the map center.
- The current RL interface only controls `trajectory`; association, PF scheduling, routing, relay, and backhaul are rule-based environment logic.

---

# 3. Simulation Pipeline

Each step follows the pipeline below:

`move UAV -> move users -> demand arrival -> association -> access scheduling -> routing -> relay -> satellite backhaul -> metrics`

```python
association_result = associate_users_to_uavs(self.users, self.uavs)
access_step_result = run_access_pf_step(
    uavs=self.uavs,
    users=self.users,
    alpha_by_uav=alpha_by_uav,
    linucb_controllers=linucb_controllers,
    context_by_uav=context_by_uav,
)
```

```python
routing_table = compute_routing_table(...)
relay_service_result = execute_relay_service(...)
backhaul_service_result = execute_backhaul_service(...)
```

Evidence: key logic excerpt from `sim_env.py`

---

# 4. Map and Scenario Setting

- Map size: `2000 m x 2000 m`
- UAV altitude: `100 m`
- Time step: `DELTA_T = 1 s`
- Each step is divided into `10` slots
- Backhaul node: a satellite positioned high above the map center

Stress demo setting:

- `4` UAVs
- `100` users
- `2 Mbps` demand per user
- hotspot-based user distribution
- initial deployment = center gateway + outer ring UAVs

```python
MAP_LENGTH = 2000.0
MAP_WIDTH = 2000.0
UAV_HEIGHT = 100.0
DELTA_T = 1.0
NUM_SLOTS_PER_STEP = 10
SAT_POSITION = [1000.0, 1000.0, SAT_ALTITUDE]
```

```python
if normalized_mode == "stress":
    return DemoModeConfig(
        num_uavs=4,
        num_users=config.NUM_USERS,
        user_demand_rate_bps=2.0e6,
        orbit_radius_m=600.0,
        user_speed_mean_mps=3.5,
        user_distribution="hotspots",
    )
```

Evidence: key settings from `config.py` and `main.py`

---

# 5. Current Mobility Design

UAV mobility:

- Each UAV action is defined by `rho` and `psi`
- `rho` controls the movement distance ratio within the step
- `psi` controls the movement direction
- The current demo uses `MAPPOStub`, so UAV motion is random rather than learned

Ground user mobility:

- random walk
- mean speed is `3.5 m/s` in stress mode

```python
actions_by_agent = policy.act(observations_by_agent, deterministic=deterministic_policy)
```

```python
if deterministic:
    rho_norm = 0.0
    psi_rad = 0.0
else:
    rho_norm = float(self.rng.uniform(0.0, 1.0))
    psi_rad = float(self.rng.uniform(-math.pi, math.pi))
```

```python
mobility_model=RandomWalkMobility(speed_mean_mps=user_speed_mean_mps)
```

Evidence: key logic excerpt from `main.py` and `mappo_stub.py`

---

# 6. Access and Resource Allocation

Current resource allocation is rule-based rather than RL-controlled:

- `association`: users are processed in backlog-descending order
- `association`: the selected UAV maximizes `proxy rate = upper bound rate / projected load`
- `access`: PF scheduling is applied
- `access`: the system uses `full frequency reuse`
- `access`: cross-UAV co-channel interference is explicitly included for the same slot and subchannel

```python
ordered_users = sorted(
    users,
    key=lambda user: (-float(user.user_access_backlog_bits), user.id),
)
proxy_rate_bps = upper_bound_rate_bps / max(projected_load, 1)
if proxy_rate_bps < float(min_rate_bps):
    continue
```

```python
tentative_assignments_by_uav_id = {uav.id: {} for uav in uavs}
ordered_uavs = _processing_order_for_slot(uavs, slot_index)
...
interfering_uavs = [
    uavs_by_id[other_uav_id]
    for other_uav_id, assignment_by_subchannel in tentative_assignments_by_uav_id.items()
    if other_uav_id != uav.id and subchannel_index in assignment_by_subchannel
]
```

Evidence: key logic excerpt from `user_association.py` and `resource_manager.py`

---

# 7. Relay and Satellite Backhaul

Routing / relay:

- the route is selected toward the best gateway path
- the decision criterion is `effective path capacity`
- relay service is `one-hop-per-step`
- queue updates use a staging buffer so that the same bits cannot be forwarded multiple hops within one step

Satellite backhaul:

- the active gateway UAV sends traffic to the satellite
- the current demo is a `single-gateway satellite` case

```python
effective_path_capacity_bps = min(
    path_bottleneck_capacity_bps,
    gateway_backhaul_capacity_bps,
)
```

```python
snapshot_queue_by_uav = {...}
staging_buffer_by_uav = {uav.id: {} for uav in uavs}
...
staging_buffer[user_id] = staging_buffer.get(user_id, 0.0) + forwarded_bits
```

Evidence: key logic excerpt from `routing.py` and `relay.py`

---

# 8. Trajectory Example

![w:950](../../demo_stress_satellite/trajectory_final.png)

- center red UAV: `active gateway`
- outer UAVs: relay / service UAVs
- blue lines: `user-to-UAV access links`
- green lines: `UAV-to-UAV adjacency / relay connectivity`
- redder users indicate larger backlog

This figure shows the expected single-gateway satellite topology: outer UAVs collect traffic and progressively funnel it toward the center gateway.

---

# 9. Metrics I: Throughput and Queue Growth

<table>
<tr>
<td><img src="../../demo_stress_satellite/plots/throughput.png" width="100%"></td>
<td><img src="../../demo_stress_satellite/plots/backlog_queue.png" width="100%"></td>
</tr>
</table>

- Mean throughput is about `76.0 Mbps`
- The arrival traffic in this case is about `200 Mbps` (`100 users x 2 Mbps`)
- Therefore, both access backlog and relay queue keep growing over time
- Final values:
  - user access backlog = `4.69 Gbits`
  - UAV relay queue = `1.50 Gbits`

Key interpretation: this indicates that the system is overloaded in the stress case. The queue growth is the expected behavior of a capacity-limited network model, not a simulator failure.

---

# 10. Metrics II: Service Quality

<table>
<tr>
<td><img src="../../demo_stress_satellite/plots/outage_ratio.png" width="100%"></td>
<td><img src="../../demo_stress_satellite/plots/jain_fairness.png" width="100%"></td>
<td><img src="../../demo_stress_satellite/plots/demand_satisfaction_ratio.png" width="100%"></td>
</tr>
</table>

| Metric | Value |
|---|---:|
| Coverage ratio | `1.00` |
| Outage ratio | `0.365` |
| Jain fairness | `0.709` |
| Demand satisfaction ratio | `0.380` |
| Lambda2 | `1.0` |
| Cumulative delivered | `3.80 Gbits / 10.0 Gbits arrived` |

- `coverage = 1.0` only means every user is associated with some UAV; it does not mean every user receives high-quality service.
- `outage ratio = 0.365` shows that a substantial fraction of users still fall below the minimum rate threshold.
- `lambda2 = 1.0` indicates that the UAV relay graph stays connected in this case.
- Additional episode summary: total energy is about `49.4 kJ`, and energy efficiency is about `76.9 kbits/J`.

---

# 11. Current Capability

The current prototype can already:

- run `step-based satellite UAV network simulation`
- jointly expose:
  - UAV motion
  - user association
  - PF scheduling
  - multi-hop relay
  - satellite backhaul
  - backlog / queue dynamics
- generate:
  - trajectory visualization
  - metric time-series plots
  - episode summary outputs

Therefore, it is already suitable as a baseline experimental environment for subsequent UAV-network and RL studies.

---

# 12. Current Limitation and Next Step

Current limitations:

- the current demo uses `MAPPOStub` random motion instead of a trained policy
- the map is still an abstract rectangular scenario
- the simulator is `step-based` and queue-based, not packet-level

Next steps:

1. replace the current stub policy with a trained RL policy
2. enrich the scenario with more realistic map semantics
3. extend the framework toward learning-based resource allocation / scheduling

The code excerpts in this report are included only as implementation evidence showing how the current system works; they are not presented as formal verification claims.
