---
marp: true
theme: default
paginate: true
---

# `rl-uavnetsim` Satellite Demo 報告

給教授的簡報版本
案例：`demo_stress_satellite`

`python -m rl_uavnetsim.main --demo-mode stress --backhaul-type satellite --steps 50 --num-users 100 --output-dir ./demo_stress_satellite`

---

# 1. 這個系統想回答什麼問題？

這個專案想做的事情很直接：

- 讓多台 UAV 在連續平面中移動
- 服務地面上的移動使用者
- 再把資料經由 UAV relay 匯聚到 gateway UAV
- 最後透過 satellite backhaul 送出

換句話說，我們關心的不是單純「飛到哪裡比較好」，而是：

- UAV 的移動
- 使用者的接入
- UAV 之間的 relay
- 回傳鏈路的能力限制
- 以及這些因素如何一起影響 queue、throughput 與服務品質

這個系統目前採用的是 `step-based` 模型，所以它是一個研究型模擬環境，而不是 packet-level simulator。

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

可以把這個架構理解成一個簡化但清楚的資料流：

- 地面使用者先把資料上傳到 UAV
- 非 gateway UAV 負責中繼與匯聚
- `gateway-capable UAV` 負責作為出口
- 最後由 gateway UAV 對 satellite 回傳

在目前這個 demo 中：

- 只有一台 `gateway-capable UAV`
- RL 只控制 UAV 的移動
- association、resource allocation、routing、relay 與 backhaul 都由環境規則負責

---

# 3. 每個 step 內發生了什麼？

目前模擬器每一步都依照同一個固定流程往前推進：

`move UAV -> move users -> demand arrival -> association -> access scheduling -> routing -> relay -> satellite backhaul -> metrics`

這樣做的好處是，整個系統的因果關係很清楚：

- 先移動
- 再產生需求
- 再決定誰連到誰
- 接著做 access 與 relay
- 最後才做 backhaul 與統計

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

這次展示的案例是 `stress satellite demo`，使用的是一個抽象但可控的連續平面場景：

- 地圖大小：`2000 m x 2000 m`
- UAV 高度：`100 m`
- 時間步長：`DELTA_T = 1 s`
- 每個 step 再細分成 `10` 個 slots
- satellite 位在地圖中心上方高空

Stress demo 的設定如下：

- `4` 架 UAV
- `100` 位 users
- 每位 user 需求率 `2 Mbps`
- user 採 `hotspots` 分布
- 初始 UAV 部署為「中心 gateway + 外圍環狀 UAV」

這個設定的目的很明確：不是追求漂亮結果，而是刻意把系統推到高負載，觀察它在壓力下的行為。

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

# 5. 目前的移動方式

UAV 的移動是用兩個動作量來控制：

- `rho`：決定這一步要移動多遠
- `psi`：決定移動方向

所以從建模角度來看，UAV 並不是在離散格點上跳躍，而是在連續平面中移動。

不過要特別強調的是：

- 目前 demo 用的是 `MAPPOStub`
- 也就是說，這裡展示的是「系統能不能跑、行為是否合理」
- 不是「訓練後策略有多強」

Ground user 則採 `random walk`，在 stress 模式下平均速度為 `3.5 m/s`。

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

# 6. 目前的接入與資源分配方式

目前的資源分配不是 RL 決定，而是明確、可解釋的規則式邏輯。

在 association 階段：

- 先優先處理 backlog 較大的 user
- 再根據 `proxy rate = upper bound rate / projected load`
- 把 user 分配給相對更合適的 UAV

在 access 階段：

- 採用 PF scheduling
- 使用 `full frequency reuse`
- 若多台 UAV 在同一個 slot、同一個 subchannel 同時傳輸，會把跨 UAV interference 算進去

這表示目前系統不是只做理想化的 strongest-link 選擇，而是已經開始考慮負載與干擾。

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

在 relay 這一層，目前系統的想法是：

- 每台 UAV 先找一條到 gateway 的可行路徑
- 以 `effective path capacity` 作為主要選擇標準
- relay 採 `one-hop-per-step`

這裡 `one-hop-per-step` 很重要，因為它可以避免同一筆資料在同一步裡被連續轉送多次，讓 queue accounting 維持一致。

在 backhaul 這一層：

- active gateway UAV 再把資料送到 satellite
- 本 demo 是單一 gateway 的 satellite 情境

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

# 8. 軌跡圖範例

![w:950](../../demo_stress_satellite/trajectory_final.png)

這張圖很適合直接用來理解目前案例的系統行為：

- 中央紅色 UAV：`active gateway`
- 外圍 UAV：relay / service UAV
- 藍線：`user-to-UAV access links`
- 綠線：`UAV-to-UAV adjacency / relay connectivity`
- user 顏色越紅：表示 backlog 越高

從這張圖可以直觀看到，目前的資料流大致是由外圍 UAV 收集，再往中心 gateway 匯聚，最後由 gateway 對 satellite 回傳。

---

# 9. 指標一：Throughput 與 Queue Growth

<table>
<tr>
<td><img src="../../demo_stress_satellite/plots/throughput.png" width="100%"></td>
<td><img src="../../demo_stress_satellite/plots/backlog_queue.png" width="100%"></td>
</tr>
</table>

這組圖最重要的訊息是：

- 平均 throughput 約為 `76.0 Mbps`
- 但新到達流量大約是 `200 Mbps`

所以結果很自然：

- user backlog 持續累積
- relay queue 也持續累積

最終數值為：

- user access backlog = `4.69 Gbits`
- UAV relay queue = `1.50 Gbits`

我會把這個結果解讀成：

這個 stress case 已經超過了目前系統容量，因此 simulator 正常地呈現出「過載下 queue 成長」的現象，而不是程式異常。

---

# 10. 指標二：Service Quality

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

這些指標可以這樣理解：

- `coverage = 1.0`：所有 users 都有被某台 UAV 關聯到
- 但這不代表每位 user 都拿到足夠好的服務品質
- `outage ratio = 0.365`：仍有相當比例的 users 沒有達到最低速率門檻
- `lambda2 = 1.0`：這個案例中的 UAV relay graph 一直保持連通

補充來看，這個 episode 的總能耗約 `49.4 kJ`，能量效率約 `76.9 kbits/J`。

---

# 11. 目前這個 prototype 已經能做什麼？

以目前版本來說，這個系統已經可以：

- 執行 `step-based satellite UAV network simulation`
- 同時觀察：
  - UAV 移動
  - user association
  - PF scheduling
  - multi-hop relay
  - satellite backhaul
  - backlog / queue dynamics
- 自動輸出：
  - trajectory 圖
  - metrics 圖表
  - episode summary

所以它已經足夠作為後續 RL 與 UAV networking 研究的基礎實驗平台。

---

# 12. 目前限制與下一步

目前的限制很清楚：

- 目前 demo 使用的是 `MAPPOStub` 隨機移動，不是訓練完成的策略
- 地圖目前仍是抽象矩形場景，語義還不多
- 模型是 `step-based` queue simulator，不是 packet-level simulator

因此下一步也很直接：

1. 用真正訓練好的 RL policy 取代目前 stub policy
2. 增加更真實的場景設定與地圖語義
3. 進一步把 resource allocation / scheduling 做成可學習模組

本簡報中的程式碼片段，主要用來作為「目前系統邏輯的證據」，幫助說明設計與行為，並不宣稱這是形式化驗證結果。
