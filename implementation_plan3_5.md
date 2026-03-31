# rl_uavnetsim — implementation_plan3.5

## 1. 目標

本文件是 `implementation_plan3.md` 與 `implementation_plan4.md` 的整合版。

目標不是做最完整的網路模擬器，而是做一個：

基於 `proposal 0323v2.md`，實作一個：
- 可直接交給 AI 實作
- 與 MARL 相容
- 設計語義閉合
- 足夠輕量
- 對 congestion / relay / backhaul 仍大致正確

的 research prototype。

本版保留：

- step-based 環境
- single-anchor 架構
- multi-agent trajectory control
- bit-backlog / relay-queue state
- PF scheduler
- per-UAV LinUCB interface

本版不追求：

- packet-level ACK / retransmission
- SimPy event-driven MAC
- collision-level fidelity

---

## 2. 核心原則

### 2.1 Agent 只控制 Trajectory

每台 UAV 是一個 agent。

agent 的 action 僅控制：

- 飛行距離比例 `rho`
- 飛行方向 `psi`

以下元件先當作環境內部規則：

- user association
- PF scheduling
- relay forwarding
- backhaul service

這樣可以把 multi-agent 問題收斂成：

> 多台 UAV 如何移動，才能在 demand、relay、backhaul、energy、connectivity 約束下提升系統表現

### 2.2 使用兩層 backlog，而不是 packet-level

本版明確區分兩種 backlog：

1. `user_access_backlog_bits`
   代表 user 端尚未送上 UAV 的需求

2. `uav_relay_queue_bits_by_user`
   代表已經送上 UAV，但尚未經 relay / backhaul 送到外網的 bits

這比單純 `rate clipping` 更像 congestion control，但仍遠比 packet simulator 簡單。

### 2.3 所有欄位名稱都帶單位語義

命名規則：

- `_bits`：位元量
- `_bps`：速率
- `_j`：能量
- `_norm`：正規化值

AI 實作時必須嚴格遵守，避免 `bits` / `bps` 混用。

### 2.4 一次只啟用一種 backhaul

每次模擬只跑一種：

- `satellite`
- `gbs`

若要比較，請分開做實驗。

### 2.5 使用 coarse resource slots，而不是假 TTI 縮放

一個 slow step 對應：

- `DELTA_T = 1.0 s`

在 step 內切成：

- `NUM_SLOTS_PER_STEP = 10`
- `SLOT_DURATION = 0.1 s`

PF 在每個 slot 真實執行一次，EMA 每個 slot 更新。

這比「跑少量代表性 slot 再乘倍率」更自洽。

---

## 3. 總體模擬流程

每個 step 依序執行：

1. UAV 根據 action 移動
2. Ground users 移動
3. user 產生新 demand，加入 `user_access_backlog_bits`
4. user association
5. 每個 UAV 在 `NUM_SLOTS_PER_STEP` 個 slot 內執行 PF
6. access 傳輸後，bits 從 user backlog 移到 serving UAV 的 relay queue
7. member UAV 根據 relay capacity 將 queue 中 bits 送往 anchor queue
8. anchor 根據 backhaul capacity 將 queue 中 bits 送往外網
9. 更新 backlog / queue / final delivered bits
10. 計算 reward 與 metrics

### 3.1 Step 內部時序說明

本版明確採用以下時間語義：

- `access scheduling` 在每個 `slot` 內執行
- `relay service` 在整個 `step` 結束後統一執行一次
- `backhaul service` 在整個 `step` 結束後統一執行一次

也就是說：

1. 在 step 內先完成 `NUM_SLOTS_PER_STEP` 次 access PF 分配
2. 所有 access 上傳完成後，累積到各 UAV 的 relay queue
3. 再以 step 為單位執行 member → anchor 的 relay 出隊
4. 最後由 anchor 以 step 為單位執行 backhaul 出隊

這是刻意的簡化：

- access 保留較細的時間粒度，讓 PF 的 EMA 更新有意義
- relay / backhaul 保持 step 級別，降低實作複雜度

AI 實作時不要把 relay / backhaul 放進每個 slot 內執行，除非未來明確升級模型。

---

## 4. 關鍵狀態變數

### 4.1 GroundUser

```python
class GroundUser:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    speed: float

    associated_uav_id: int          # -1 if unassociated

    demand_rate_bps: float

    user_access_backlog_bits: float
    arrived_bits_step: float
    access_uploaded_bits_step: float
    delivered_bits_step: float

    final_rate_bps: float           # delivered_bits_step / DELTA_T
    avg_throughput_bps: float       # PF EMA

    mobility_model: BaseMobilityModel
```

### 4.2 UAV

```python
class UAV:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    direction: float

    is_anchor: bool
    residual_energy_j: float

    associated_user_ids: list[int]

    # bits uploaded from users to this UAV during current step
    access_ingress_bits_step: float

    # sparse per-user relay queue living on this UAV
    relay_queue_bits_by_user: dict[int, float]

    # cached aggregates
    relay_queue_total_bits: float
    relay_forwarded_bits_step: float
    backhaul_forwarded_bits_step: float

    energy_model: EnergyModelBase
```

### 4.3 Environment

```python
class EnvState:
    current_step: int
    adjacency_matrix: np.ndarray
    lambda2: float

    backhaul_capacity_bps: float
    total_delivered_bits_step: float
```

---

## 5. 完整且閉合的資料流

這一節是整份計畫最重要的地方。

### 5.1 Demand arrival

對每個 user `j`：

```python
arrived_bits_j = demand_rate_bps_j * DELTA_T
user_access_backlog_bits_j += arrived_bits_j
```

### 5.2 Access scheduling

PF scheduler 的輸出是：

- `slot_rate_bps_ij`
- `slot_bits_ij = slot_rate_bps_ij * SLOT_DURATION`

對 user `j`：

```python
slot_bits_ij <= user_access_backlog_bits_j
```

因此 access stage 只會送出 user 真的還有 backlog 的 bits。

每個 slot 結束後：

```python
user_access_backlog_bits_j -= slot_bits_ij
uav_i.relay_queue_bits_by_user[j] += slot_bits_ij
```

整個 step 統計：

```python
user_j.access_uploaded_bits_step = sum_i sum_slot slot_bits_ij
uav_i.access_ingress_bits_step = sum_j sum_slot slot_bits_ij
```

### 5.3 Relay service on member UAVs

對每個 member UAV `i`：

1. 找到到 anchor 的路徑
2. 計算 bottleneck relay capacity
3. 轉成本 step 可服務 bits

```python
relay_budget_bits_i = relay_capacity_bps_i * DELTA_T
```

隊列總量：

```python
queue_total_i = sum(uav_i.relay_queue_bits_by_user.values())
```

實際本 step 可送：

```python
relay_out_bits_i = min(queue_total_i, relay_budget_bits_i)
```

### 5.4 Proportional dequeue on member UAVs

為了避免 per-packet 排程，本版使用比例出隊：

若 `queue_total_i > 0`，則對 queue 中每個 user `j`：

```python
share_ij = queue_bits_ij / queue_total_i
relay_out_bits_ij = share_ij * relay_out_bits_i
```

更新：

```python
uav_i.relay_queue_bits_by_user[j] -= relay_out_bits_ij
anchor.relay_queue_bits_by_user[j] += relay_out_bits_ij
```

### 5.5 Anchor local ingress

anchor 自己服務到的 user，在 access stage 上傳後，也一律先進入：

```python
anchor.relay_queue_bits_by_user[j]
```

這樣可統一處理：

- anchor 自己的 access traffic
- member 經 relay 送來的 traffic

### 5.6 Backhaul service on anchor

計算 anchor 的 backhaul 可服務 bits：

```python
backhaul_budget_bits = backhaul_capacity_bps * DELTA_T
anchor_queue_total = sum(anchor.relay_queue_bits_by_user.values())
backhaul_out_bits = min(anchor_queue_total, backhaul_budget_bits)
```

### 5.7 Proportional dequeue on anchor

若 `anchor_queue_total > 0`，則對 queue 中每個 user `j`：

```python
anchor_share_j = anchor_queue_bits_j / anchor_queue_total
delivered_bits_j = anchor_share_j * backhaul_out_bits
```

更新：

```python
anchor.relay_queue_bits_by_user[j] -= delivered_bits_j
user_j.delivered_bits_step += delivered_bits_j
```

### 5.8 Final rate

```python
user_j.final_rate_bps = user_j.delivered_bits_step / DELTA_T
```

### 5.9 Queue / backlog invariants

對 user access backlog：

```python
user_access_backlog_next
=
user_access_backlog_prev + arrived_bits - access_uploaded_bits
```

對 member UAV relay queue：

```python
relay_queue_next
=
relay_queue_prev + access_ingress_bits - relay_out_bits
```

對 anchor queue：

```python
anchor_queue_next
=
anchor_queue_prev + anchor_local_access_bits + total_relay_in_bits - backhaul_out_bits
```

這三條必須在測試中逐步驗證。

---

## 6. 為什麼這版比 plan3 / plan4 更穩

### 6.1 比 plan3 多了真正的 relay queue

plan3 只有 user backlog，對 relay 壅塞的表達不夠。

plan3.5 加入：

- `relay_queue_bits_by_user`

因此可以區分：

- access 擁塞
- relay/backhaul 擁塞

### 6.2 比 plan4 更自洽

plan4 的問題是：

- `queue_bits` 有欄位但沒有完整更新式
- `rate` 與 `bits` 單位混用

plan3.5 明確做了：

- queue 的進出隊更新式
- bits/bps 命名規則
- proportional dequeue 規則

---

## 7. System Model

### 7.1 UAV-GU access link

使用 proposal 的 probabilistic LoS/NLoS 模型：

```text
L_los(d)   = alpha_los + 10 * beta_los * log10(d) + X_sigma
L_nlos(d)  = alpha_nlos + 10 * beta_nlos * log10(d) + X_sigma
P_los(phi) = 1 / (1 + a * exp(-b * (phi - a)))
L_avg      = P_los * L_los + (1 - P_los) * L_nlos
g_ij       = 10^(-L_avg / 10)
```

per-subchannel SINR：

```text
SINR_ij^n = P_tx_rf * g_ij / (sigma_access^2 + interference_ij^n)
```

### 7.2 UAV-UAV relay link

```text
g_mn = rho_0 / ||q_m - q_n||^2
SNR_mn = P_tx_uav * g_mn / sigma_a2a^2
```

路徑容量定義為沿路最小 link capacity。

### 7.3 Backhaul

只在 anchor 上計算：

- satellite
- 或 GBS

輸出：

```python
backhaul_capacity_bps
```

### 7.4 Energy

保留兩種模型：

- `SimplifiedEnergyModel`
- `Zeng2019EnergyModel`

v1 預設可先使用 simplified。

---

## 8. Multi-Agent 設計

### 8.1 Agent

每台 UAV 一個 agent。

### 8.2 Action

```python
a_i = {
    "rho": float in [0, 1],
    "psi": float in [-pi, pi],
}
```

### 8.3 Local observation

plan3 的 observation 太弱，plan4 又沒有把 backlog 放進去。

本版使用：

```python
o_i = {
    "self_pos": [3],
    "self_vel": [2],
    "self_energy_norm": [1],
    "self_is_anchor": [1],

    "self_relay_queue_norm": [1],
    "self_associated_user_count_norm": [1],
    "self_est_relay_capacity_norm": [1],

    "other_uav_relative_pos": [(I-1) * 3],
    "other_uav_link_active": [(I-1)],

    "visible_user_relative_pos": [M * 2],
    "visible_user_backlog_norm": [M],
}
```

補充：

- 若 `uav_i` 與 `uav_k` 目前可連通，使用即時位置
- 若不可連通，使用上一個 step 的 last-known position

### 8.4 Global state

```python
s_t = {
    "uav_positions": [I, 3],
    "uav_energies": [I],
    "uav_queue_totals": [I],
    "user_positions": [J, 2],
    "user_access_backlogs": [J],
    "associations": [J],
    "connectivity_matrix": [I, I],
    "backhaul_capacity_norm": [1],
}
```

### 8.5 Reward

本版明確使用：

- `shared team reward`

```python
throughput_norm = total_delivered_bits_step / THROUGHPUT_REF_BITS
energy_norm     = total_energy_step_j / ENERGY_REF_J
outage          = outage_ratio
access_backlog_norm = total_user_access_backlog_bits / ACCESS_BACKLOG_REF_BITS
relay_queue_norm    = total_uav_relay_queue_bits / RELAY_QUEUE_REF_BITS

R_team = (
    + throughput_norm
    - ETA * energy_norm
    - MU * outage
    - BETA_ACCESS * access_backlog_norm
    - BETA_RELAY * relay_queue_norm
    - LAMBDA_CONN * int(lambda2 <= 0)
    - LAMBDA_SAFE * num_safety_violations
)
```

這樣 reward 才真的和 congestion-aware 目標一致。

### 8.5.1 Outage 定義

本版統一使用最終端到端服務速率來判定 outage。

對每個 user `j`：

```python
is_outage_j = int(user_j.final_rate_bps < R_MIN)
```

系統 outage ratio：

```python
outage_ratio = sum_j is_outage_j / NUM_USERS
```

注意：

- 不可使用 access rate 判定 outage
- 不可使用 relay 前的中間 rate 判定 outage
- 必須在 backhaul service 完成、`final_rate_bps` 更新後才判定

這樣才與 end-to-end bottleneck 語義一致。

### 8.6 Heterogeneous role

anchor 與 member 角色不同。

v1 先採用：

- shared actor policy
- `self_is_anchor` 作為 role indicator

若後續訓練不穩，再升級為：

- shared backbone + role embedding
- 或 anchor/member 分離 policy

---

## 9. Resource Allocation

### 9.1 User association

使用：

- strongest feasible UAV

feasibility 用粗略 rate upper bound 判斷：

```python
upper_bound_rate_bps_ji =
    NUM_SUBCHANNELS * SUBCHANNEL_BW * log2(1 + max_sinr_ji)
```

若 `upper_bound_rate_bps_ji < R_MIN`：

- 視為 infeasible

若沒有 feasible UAV：

- `associated_uav_id = -1`

### 9.2 PF scheduler

每個 UAV、每個 slot：

1. 計算 associated users 的 per-subchannel SINR
2. 用 PF score 分配 subchannels
3. 轉成 slot bits
4. 每個 user 的 slot bits 不得超過當前 `user_access_backlog_bits`
5. 更新 `avg_throughput_bps`

### 9.3 LinUCB interface

真實 LinUCB 仍不實作，但介面保留為 per-UAV：

```python
linucb_controllers: dict[int, LinUCBStub]
alpha_i = linucb_controllers[uav_id].select_alpha(context_i)
```

即使 v1 仍固定 `alpha_i = 1.0`，介面也要與 proposal 一致。

---

## 10. Config 建議

```python
DELTA_T = 1.0
NUM_SLOTS_PER_STEP = 10
SLOT_DURATION = DELTA_T / NUM_SLOTS_PER_STEP

NUM_UAVS = 5
NUM_USERS = 60
ANCHOR_UAV_ID = 0

NUM_SUBCHANNELS = 8
SUBCHANNEL_BW = 1e6

USER_DEMAND_RATE_BPS = 0.5e6

BACKHAUL_TYPE = "satellite"   # or "gbs"
```

實作時至少還需要補齊以下常用參數：

```python
# map / motion
MAP_LENGTH = 2000
MAP_WIDTH = 2000
UAV_HEIGHT = 100
V_MAX = 20.0
D_SAFE = 30.0

# access link
P_TX_RF = 0.5
ALPHA_LOS = 1.0
ALPHA_NLOS = 20.0
BETA_LOS = 2.09
BETA_NLOS = 3.75
A_ENV = 4.88
B_ENV = 0.429
SHADOW_STD = 0.0

# common radio constants
CARRIER_FREQ = 2.4e9
LIGHT_SPEED = 3e8
N0 = 4e-21
NF_ACCESS = 7.0
NF_A2A = 5.0
NF_SAT = 3.0
NF_GBS = 5.0

# relay / backhaul
P_TX_UAV = 1.0
RHO_0 = 1e-4
GAMMA_TH_DB = 10.0
A2A_BW = 10e6
B_SAT = 20e6
GBS_BW = 50e6
SAT_POSITION = [1000, 1000, 550e3]
GBS_POSITIONS = [[100, 100, 0]]

# PF
PF_ALPHA_DEFAULT = 1.0
PF_BETA = 0.01
PF_EPSILON = 1e-6
SINR_THRESHOLD_DB = 3.0
R_MIN = 0.5e6

# mobility
MOBILITY_MODEL = "random_walk"
USER_SPEED_MEAN = 2.0
USER_SPEED_MAX = 8.0
USER_DIR_SIGMA = 0.5

# observation
OBS_RADIUS = 500.0
MAX_OBS_USERS_PAD = 30

# energy
ENERGY_MODEL = "simplified"
E_INITIAL = 50000.0
E_MIN = 2000.0
E_HOVER = 200.0
E_FLY = 5.0

# reward normalization
ETA = 0.1
MU = 1.0
BETA_ACCESS = 0.2
BETA_RELAY = 0.2
LAMBDA_CONN = 5.0
LAMBDA_SAFE = 2.0
THROUGHPUT_REF_BITS = NUM_USERS * R_MIN * DELTA_T
ENERGY_REF_J = NUM_UAVS * E_HOVER * DELTA_T
ACCESS_BACKLOG_REF_BITS = NUM_USERS * USER_DEMAND_RATE_BPS * DELTA_T
RELAY_QUEUE_REF_BITS = NUM_USERS * USER_DEMAND_RATE_BPS * DELTA_T
```

上面不代表唯一正確數值，但這些欄位在 v1 幾乎都會被用到，應在 `config.py` 一次定義完整。

---

## 11. 視覺化

本版不是 packet-level，因此不要畫單個 packet 動畫。

應畫：

- UAV 3D trajectory
- user 分布
- association lines
- active relay links
- anchor-backhaul link
- queue / backlog heat indicators

可以用：

- 線寬表示 `bps`
- 顏色表示 backlog / queue 壓力

---

## 12. 指標

### 12.1 必做

- `sum_throughput_bps`
- `coverage_ratio`
- `outage_ratio`
- `jain_fairness`
- `total_energy_j`
- `energy_efficiency`
- `lambda2`
- `total_user_access_backlog_bits`
- `total_uav_relay_queue_bits`

### 12.2 需求滿足率

不要叫 PDR。

請使用：

```python
demand_satisfaction_ratio =
    total_delivered_bits / total_arrived_bits
```

---

## 13. 專案結構

```text
rl_uavnetsim/
├── main.py
├── config.py
├── environment/
│   └── sim_env.py
├── entities/
│   ├── uav.py
│   ├── ground_user.py
│   ├── satellite.py
│   └── ground_base_station.py
├── mobility/
│   ├── base_mobility.py
│   ├── random_walk.py
│   ├── gauss_markov_hotspot.py
│   └── social_force.py
├── channel/
│   ├── a2g_channel.py
│   ├── a2a_channel.py
│   ├── backhaul_channel.py
│   └── sinr_calculator.py
├── energy/
│   └── energy_model.py
├── allocation/
│   ├── user_association.py
│   ├── pf_scheduler.py
│   └── resource_manager.py
├── network/
│   ├── topology.py
│   └── relay.py
├── rl_interface/
│   ├── mdp.py
│   ├── mappo_stub.py
│   └── linucb_stub.py
├── metrics/
│   └── metrics_collector.py
├── visualization/
│   ├── trajectory_visualizer.py
│   └── metrics_plotter.py
└── utils/
    └── helpers.py
```

---

## 14. Milestones

### Milestone 1: Config + Entities + Mobility

完成：

- `config.py`
- `entities/uav.py`
- `entities/ground_user.py`
- `mobility/base_mobility.py`
- `mobility/random_walk.py`

驗證：

- user backlog 初始化正確
- UAV relay queue 初始化正確
- UAV/user 移動正常

### Milestone 2: Channel + Energy

完成：

- `a2g_channel.py`
- `a2a_channel.py`
- `backhaul_channel.py`
- `energy_model.py`
- `entities/satellite.py`
- `entities/ground_base_station.py`

驗證：

- path loss / SINR / capacity 數值合理
- power-speed 曲線合理
- satellite / GBS entity 可正確初始化並被 channel module 使用

### Milestone 3: Association + PF

完成：

- `user_association.py`
- `pf_scheduler.py`
- `resource_manager.py`

驗證：

- OFDMA 約束成立
- slot bits 不超過 user access backlog
- EMA 每 slot 更新合理

### Milestone 4: Relay Queue + Backhaul

完成：

- `relay.py`
- `sim_env.py`

驗證：

- member queue 更新守恆
- anchor queue 更新守恆
- delivered bits 守恆

### Milestone 5: MARL Interface

完成：

- `mdp.py`
- `mappo_stub.py`
- `linucb_stub.py`

驗證：

- `reset()` / `step()` multi-agent 介面正確
- 所有 agent 共用同一個 `R_team`

### Milestone 6: Metrics + Visualization

完成：

- `metrics_collector.py`
- `trajectory_visualizer.py`
- `metrics_plotter.py`

驗證：

- 輸出 queue/backlog 圖表
- 3D GIF 可視化 flow pressure

---

## 15. 必做測試

### 15.1 守恆測試

每個 step 驗證：

```python
user_access_backlog_next
=
user_access_backlog_prev + arrived_bits - access_uploaded_bits
```

```python
member_queue_next
=
member_queue_prev + access_ingress_bits - relay_out_bits
```

```python
anchor_queue_next
=
anchor_queue_prev + anchor_access_ingress_bits + total_relay_in_bits - backhaul_out_bits
```

### 15.2 單位測試

保證：

- `_bits` 欄位不會和 `_bps` 混加
- `final_rate_bps = delivered_bits_step / DELTA_T`

### 15.3 約束測試

- safety distance
- connectivity
- OFDMA orthogonality
- backhaul budget

---

## 16. 結論

`implementation_plan3_5.md` 的定位很明確：

> 以 plan3 的穩定骨架為基底，吸收 plan4 真正有價值的設計，但把 queue、單位、reward、observation 全部補齊。

如果你的目標是：

- 先做出可信的 v1 系統
- 不想被 packet simulator 拖垮
- 又希望「congestion control」不是只停留在口號

那 plan3.5 會比 plan3 與 plan4 都更適合直接開工。
