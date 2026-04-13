---
marp: true
theme: default
paginate: true
---

# `rl-uavnetsim` Satellite Demo 報告

教授版研究簡報  
案例：`demo_stress_satellite`

`python -m rl_uavnetsim.main --demo-mode stress --backhaul-type satellite --steps 50 --num-users 100 --output-dir ./demo_stress_satellite`

---

# 1. 問題與目標

- 本系統的目標是建立一個 `step-based` 的 UAV 網路模擬器，用於多無人機協作與 RL 研究。
- 目前聚焦的問題是：多台 UAV 如何在連續平面中移動，服務移動中的地面使用者，並透過 UAV relay 與 `satellite backhaul` 把資料送出。
- 因此本系統同時建模：
  - UAV trajectory control
  - user-UAV access
  - UAV-UAV relay
  - gateway-to-satellite backhaul
  - backlog / queue dynamics
- 它不是 packet-level simulator，也不是 SimPy / event-driven 模型；目前採用的是以 step 與 slot 為單位的研究型抽象環境。

---

# 2. 系統架構

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

- `Ground Users` 先透過 access link 上傳資料到 UAV。
- 非 gateway UAV 只負責 relay；資料可逐跳轉送。
- `gateway-capable UAV` 是可出網節點；在本 demo 中只有一台，位於地圖中心附近。
- 當前 RL 介面只控制 `trajectory`；association、PF scheduling、routing、relay、backhaul 都由環境規則決定。

---

# 3. 單步模擬流程

每個 step 的執行順序如下：

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

證據：`sim_env.py` 中的關鍵邏輯

---

# 4. 地圖與案例設定

- 地圖大小：`2000 m x 2000 m`
- UAV 高度：`100 m`
- 時間步長：`DELTA_T = 1 s`
- 每個 step 內再切成 `10` 個 slots
- 回傳節點：位於地圖中心上方高空的 satellite

Stress demo 設定：

- `4` 架 UAV
- `100` 位 users
- 每位 user 需求率 `2 Mbps`
- user 分布採 `hotspots`
- UAV 初始部署為「中心 gateway + 外圍環狀 UAV」

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

證據：`config.py` 與 `main.py` 中的關鍵設定

---

# 5. 當前移動設計

UAV 移動：

- 每台 UAV 的動作為 `rho` 與 `psi`
- `rho` 決定本步移動距離比例
- `psi` 決定移動方向
- 當前 demo 使用 `MAPPOStub`，所以 UAV 動作是隨機產生，不是訓練後策略

Ground user 移動：

- 採 `random walk`
- stress 模式下平均速度為 `3.5 m/s`

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

證據：`main.py` 與 `mappo_stub.py` 中的關鍵邏輯

---

# 6. 當前接入與資源分配方式

目前資源分配不是 RL 控制，而是規則式：

- `association`：依 backlog 大小排序，優先處理 queue 壓力大的 user
- `association`：以 `proxy rate = upper bound rate / projected load` 選最適合的 UAV
- `access`：使用 PF scheduling
- `access`：採 `full frequency reuse`
- `access`：同一 slot、同一 subchannel 的跨 UAV 干擾會被顯式納入

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

證據：`user_association.py` 與 `resource_manager.py` 中的關鍵邏輯

---

# 7. Relay 與 Satellite Backhaul

Routing / relay：

- route 會選擇到 gateway 的最佳路徑
- 判準是 `effective path capacity`
- relay 採 `one-hop-per-step`
- queue 使用 staging buffer，避免同一步內連續多跳，維持 accounting 正確

Satellite backhaul：

- 由 active gateway UAV 對 satellite 回傳
- 當前 demo 是 `single-gateway satellite` 情境

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

證據：`routing.py` 與 `relay.py` 中的關鍵邏輯

---

# 8. Trajectory Example

![w:950](../../demo_stress_satellite/trajectory_final.png)

- 中央紅色 UAV：`active gateway`
- 外圍 UAV：relay / service UAV
- 藍線：`user-to-UAV access links`
- 綠線：`UAV-to-UAV adjacency / relay connectivity`
- user 顏色越紅：代表該 user backlog 越高

這張圖反映的是單 gateway satellite 拓樸下，外圍 UAV 收集資料並往中心 gateway 匯聚的行為。

---

# 9. Metrics I: Throughput 與 Queue Growth

<table>
<tr>
<td><img src="../../demo_stress_satellite/plots/throughput.png" width="100%"></td>
<td><img src="../../demo_stress_satellite/plots/backlog_queue.png" width="100%"></td>
</tr>
</table>

- 平均 throughput 約為 `76.0 Mbps`
- 本案例新到達流量約為 `200 Mbps`（`100 users x 2 Mbps`）
- 因此 access backlog 與 relay queue 都持續上升
- 最終：
  - user access backlog = `4.69 Gbits`
  - UAV relay queue = `1.50 Gbits`

重點：這代表系統在 stress case 下`容量不足`，所以 queue 持續累積。這是模型正常反映「容量受限」的結果，不是 simulator 壞掉。

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

- `coverage = 1.0` 只代表所有 users 都有關聯到某台 UAV，不代表都得到足夠品質的服務。
- `outage ratio = 0.365` 表示仍有相當比例的 users 未達最低速率門檻。
- `lambda2 = 1.0` 表示本案例的 UAV relay graph 保持連通。
- 另外，本 episode 的總能耗約 `49.4 kJ`，能量效率約 `76.9 kbits/J`。

---

# 11. 當前能力

目前這個 prototype 已能：

- 進行 `step-based satellite UAV network simulation`
- 同時觀察：
  - UAV 移動
  - user association
  - PF scheduling
  - multi-hop relay
  - satellite backhaul
  - backlog / queue dynamics
- 產出：
  - trajectory 圖
  - 指標時間序列圖
  - episode summary

因此，它已經足以作為後續 UAV network 與 RL 研究的基礎實驗環境。

---

# 12. 目前限制與下一步

目前限制：

- 目前 demo 使用的是 `MAPPOStub` 隨機移動，尚不是已訓練策略
- 地圖仍是抽象矩形場景，尚未包含更高擬真度的語義地圖
- 目前是 `step-based` queue simulator，不是 packet-level simulator

下一步：

1. 用已訓練的 RL policy 取代當前 stub policy
2. 增加更真實的場景與地圖語義
3. 擴展為可學習的 resource allocation / scheduling 方法

本報告中的程式碼摘錄僅作為「當前實作邏輯證據」，用來說明系統如何運作，而非形式化驗證證明。
