---
marp: true
theme: default
paginate: true

---

# 1. What problem is this system trying to address?

- how multiple UAVs move in continuous space
- how they serve mobile ground users
- how traffic is relayed among UAVs
- and how the aggregated traffic is finally sent out through a satellite backhaul

So the focus is not only “where should the UAV fly,” but rather the joint behavior of:

- UAV motion
- user access
- UAV relay
- backhaul capacity

The current system is a `step-based` research simulator, not a packet-level simulator.

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
---

This can be understood as a simplified but explicit traffic flow:

- ground users first upload traffic to UAVs
- non-gateway UAVs relay the traffic
- the `gateway-capable UAV` forwards traffic to the satellite

In the current demo:

- there is only one `gateway-capable UAV`
- RL only controls `trajectory`
- association, PF scheduling, routing, relay, and backhaul are all handled by environment logic

---

# 3. What happens inside one simulation step?

Each step follows the same pipeline:

`move UAV -> move users -> demand arrival -> association -> access scheduling -> routing -> relay -> satellite backhaul -> metrics`

This ordering matters because it keeps the cause-and-effect relationship clear:

- positions are updated first
- new demand is then generated
- access decisions are made on the updated state
- relay and backhaul are executed afterward
- and metrics are recorded at the end
 ---
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

from `sim_env.py`

---

# 4. Map and Scenario Setting

- map size: `2000 m x 2000 m`
- UAV altitude: `100 m`
- time step: `DELTA_T = 1 s`
- each step is divided into `10` slots
- the satellite is positioned high above the map center

The stress demo uses the following setup:

- `4` UAVs
- `100` users
- `2 Mbps` demand per user

---

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

from `config.py` and `main.py`

---

# 5. Current Mobility Design

UAV motion is controlled by two action variables:

- `rho`: how far the UAV moves within the step
- `psi`: the movement direction

- the UAV behavior shown here is random

---

Ground users follow a `random walk` model, with mean speed `3.5 m/s` in the stress case.

```python
actions_by_agent = policy.act(observations_by_agent, deterministic=deterministic_policy)
```

```python
    rho_norm = float(self.rng.uniform(0.0, 1.0))
    psi_rad = float(self.rng.uniform(-math.pi, math.pi))
```

```python
mobility_model=RandomWalkMobility(speed_mean_mps=user_speed_mean_mps)
```

from `main.py` and `mappo_stub.py`

---

# 6. Current Access and Resource Allocation

At the moment, resource allocation is not RL-driven. It is rule-based.

During association:

- users with larger backlog are handled first
- each user is assigned based on
  `proxy rate = upper bound rate / projected load`

```python
ordered_users = sorted(
    users,
    key=lambda user: (-float(user.user_access_backlog_bits), user.id),
)
proxy_rate_bps = upper_bound_rate_bps / max(projected_load, 1)
if proxy_rate_bps < float(min_rate_bps):
    continue
```

---

# 7. Relay and Satellite Backhaul

At the relay layer, the current system works as follows:

- each UAV looks for a feasible path toward the gateway
- route selection is based on `effective path capacity`
- relay service is restricted to `one-hop-per-step`

At the backhaul layer:
- the active gateway UAV forwards traffic to the satellite
- the current demo is a single-gateway satellite case
---
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

![w:950](../../demo_stress_satellite/trajectory.gif)

---

This figure is useful because it gives an intuitive picture of the current system behavior:

- center red UAV: `active gateway`
- outer UAVs: relay / service UAVs
- blue lines: `user-to-UAV access links`
- green lines: `UAV-to-UAV adjacency / relay connectivity`
- redder users indicate larger backlog

---

# 9. Metrics I: Throughput and Queue Growth

<table>
<tr>
<td><img src="../../demo_stress_satellite/plots/throughput.png" width="100%"></td>
<td><img src="../../demo_stress_satellite/plots/backlog_queue.png" width="100%"></td>
</tr>
</table>

The key message from this pair of figures is straightforward:

- mean throughput is about `76.0 Mbps`
- but new arrival traffic is about `200 Mbps`
