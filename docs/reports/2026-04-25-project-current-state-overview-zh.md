# rl-uavnetsim 專案現況導覽

日期：2026-04-25  
用途：給未來的自己或新開聊天視窗快速接回專案狀態  
口吻：教學導覽 + 工程交接

---

## 0. 一頁速讀

這個專案是 `rl-uavnetsim`，目標是做一個 **step-based UAV network simulator for MARL research**。它不是 packet-level simulator，也不是 SimPy event-driven simulator；它把系統切成一個一個 step，在每個 step 裡模擬 UAV 移動、ground user 移動、需求到達、user association、access sub-channel allocation、UAV relay、satellite backhaul、metrics/reward。

目前最重要的事實是：

- **MAPPO 目前只控制 UAV movement**。也就是每台 UAV 輸出 `rho` 和 `psi`，決定這個 step 飛多遠、往哪個方向飛。
- **Association、PF scheduling、relay、backhaul 都還是環境規則式邏輯**。MAPPO 沒有直接選 user，也沒有直接選 sub-channel。
- **LinUCB 目前是 alpha controller 的可替換接口，不是完整 LinUCB 研究實作**。目前 `LinUCBStub` 可以記錄 context/reward，也可以回傳固定 alpha，但它不是完整 contextual bandit learning。
- **Noisy-MAPPO 還沒實作**。原 proposal 把 Noisy-MAPPO 當核心方法，但目前程式是 plain MAPPO baseline，並且架構刻意保留未來可替換空間。
- **目前訓練主線是 satellite-only**。demo runner 仍保留 `gbs` backhaul 選項，但 `build_training_env()` 會要求 `backhaul_type == "satellite"`。
- **`compact_v1` 是舊 checkpoint 相容用；`compact_v2` 是現在較推薦的訓練 observation**，多了 local user backlog、local association flag、self associated-user count。
- **evaluation 已改成 schema v2**，可以產生 episode-aware artifacts，也會比較 trained policy 和 static baseline。
- **coverage 現在分成兩種語義**：`coverage_ratio` 只代表 association；`effective_coverage_ratio` 代表 association 且 end-to-end final rate 達標。
- **SAR low-rate coverage 場景已新增**：用低速率需求、有限 access range、分離 hotspots，避免 static UAV 天然覆蓋所有 users。

如果新視窗只想先看 5 個檔案：

1. `README.md`：最短版專案狀態與指令。
2. `rl_uavnetsim/environment/sim_env.py`：整個 simulator 單步流程。
3. `rl_uavnetsim/allocation/resource_manager.py`：access sub-channel / PF scheduling。
4. `rl_uavnetsim/training/mappo_trainer.py`：MAPPO training、evaluation、static baseline。
5. `configs/marl/mappo_satellite_3uav_sar_lowrate_balanced.yaml`：目前較推薦的 SAR low-rate 調參場景。

---

## 1. 目前健康狀態

報告生成前，我重新做了兩個非破壞性檢查。

### Git 狀態

目前 `main` 有未提交的本地修改，這點很重要，因為有些 YAML 可能是你正在跑實驗時手動調過的版本。

```text
## main...origin/main
 M configs/marl/mappo_satellite.yaml
 M configs/marl/mappo_satellite_3uav_coverage.yaml
 M configs/marl/mappo_satellite_3uav_medium.yaml
 M docs/reports/2026-04-13-satellite-demo-report-en.md
 M docs/reports/2026-04-13-satellite-demo-report-zh.md
 M rl_uavnetsim/allocation/user_association.py
 M rl_uavnetsim/environment/sim_env.py
 M rl_uavnetsim/main.py
 M rl_uavnetsim/metrics/metrics_collector.py
 M rl_uavnetsim/rl_interface/mdp.py
 M rl_uavnetsim/training/analyze_run.py
 M rl_uavnetsim/training/configuration.py
 M rl_uavnetsim/training/evaluate.py
 M rl_uavnetsim/training/mappo_trainer.py
 M rl_uavnetsim/visualization/metrics_plotter.py
 M tests/test_allocation.py
 M tests/test_coverage_calibration.py
 M tests/test_metrics_visualization.py
 M tests/test_rl_interface.py
 M tests/test_training_evaluation.py
 M tests/test_training_features.py
?? .codex
?? configs/marl/mappo_satellite_3uav_sar_lowrate_balanced.yaml
?? configs/marl/mappo_satellite_3uav_sar_lowrate_coverage.yaml
?? docs/proposal/
?? docs/relatedwork/
?? docs/reports/2026-04-25-project-current-state-overview-zh.md
?? docs/reports/2026-04-25-relatedwork-disaster-uav-mappo-reading-notes-zh.md
```

意思是：這份報告會描述「目前 working tree 實際看到的狀態」，但要小心其中一部分設定不是乾淨 committed baseline。

### Tests

全測試通過：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m pytest -q tests
```

結果：

```text
74 passed, 15 warnings
```

這代表目前 interface shape、association gate、max access range、effective coverage、reward config、evaluation schema、large-map geometry、channel/allocation/relay 等核心行為至少有 regression tests 保護。不代表訓練出來的 policy 一定好，只代表目前程式接口和已寫下的行為假設是自洽的。

---

## 2. 原始 proposal 想做什麼

原計畫在 `docs/proposal/proposal 0323v2.md`。它的高層故事是：

- 災害或大型疏散時，地面通訊基礎設施可能壞掉或壅塞。
- UAV 可以提供臨時 coverage，但如果沒有穩定 backhaul，UAV network 可能變成資訊孤島。
- 因此 proposal 想做 SAGIN，也就是 **Space-Air-Ground Integrated Network**。
- 架構上想要 UAV access + UAV relay + satellite backhaul。
- 方法上想要 hierarchical approach：
  - Noisy-MAPPO 控制 UAV trajectory。
  - LinUCB 控制 PF scheduling 裡的 fairness weight `alpha`。
  - PF scheduling 在較快時間尺度做 sub-channel assignment。

原 proposal 的核心句子可以概括成：

```text
Noisy-MAPPO handles trajectory.
LinUCB adapts fairness alpha.
PF scheduling handles fast sub-channel assignment.
Satellite backhaul prevents UAV network from becoming an information island.
```

但現在要很誠實地分開：

- 已落地的是 simulator、plain MAPPO baseline、PF scheduling、configurable scenario、evaluation diagnostics。
- 還沒落地的是完整 Noisy-MAPPO、完整 LinUCB learning、Enhanced K-means initialization。
- 現在的程式架構有刻意讓這些東西未來可以替換，但不能把它們寫成已完成。

---

## 3. 目前系統架構

目前最主要的 satellite 模式可以想成：

```text
Ground Users
    ↓ access link
Access / Service UAVs
    ↓ UAV-UAV relay
Gateway-capable UAV
    ↓ satellite backhaul
Satellite
```

重要角色：

- **GroundUser**：地面使用者，有位置、速度、需求產生率、backlog、final rate。
- **UAV**：空中基地台/relay node，有位置、速度、relay queue、是否 gateway-capable。
- **Gateway-capable UAV**：可以把 relay queue 裡的資料送到 satellite。
- **Satellite**：提供 backhaul link。
- **SimEnv**：真正執行 network dynamics 的 step-based environment。
- **PettingZooUavNetEnv**：把 simulator 包成 PettingZoo ParallelEnv，給 TorchRL MAPPO 使用。

目前 training 裡通常是 single gateway satellite case，也就是 UAV 0 是 gateway-capable UAV。這是為了先把 MAPPO baseline 和 evaluation pipeline 跑穩；multi-gateway 或更複雜 backhaul 是後續可擴充方向。

---

## 4. 一個 simulation step 發生什麼事

核心流程在 `rl_uavnetsim/environment/sim_env.py` 的 `SimEnv.step()`。

白話版：

```text
1. reset step counters
2. UAV 根據 action 移動
3. ground users 移動
4. users 產生新 demand，累積到 backlog
5. association：每個 user 選一台 feasible UAV
6. access PF scheduling：分配 sub-channel，上傳 user backlog 到 UAV relay queue
7. 計算 satellite backhaul capacity
8. 建 UAV-UAV relay capacity matrix
9. routing：每台 UAV 找 gateway path
10. relay：每個 step 只 relay 一跳
11. backhaul：gateway UAV 把 queue 送到 satellite
12. 更新 metrics / accounting / EnvState
```

工程 evidence：

- `SimEnv.step()` 先套 UAV action、移動 users，再產生 demand：`rl_uavnetsim/environment/sim_env.py:149`
- association 發生在 access scheduling 前：`rl_uavnetsim/environment/sim_env.py:157`
- access scheduling 呼叫 `run_access_pf_step()`：`rl_uavnetsim/environment/sim_env.py:162`
- routing/relay/backhaul 接在 access 之後：`rl_uavnetsim/environment/sim_env.py:171`

這個順序很重要。因為 user 的 backlog 先增加，然後 association 和 PF scheduling 才看到這些需求。因此 backlog 不是抽象分數，而是真正未服務完、等著被 access/relay/backhaul 消化的 bit 數。

---

## 5. 飛行與資源分配的時間尺度

目前全域時間設定在 `rl_uavnetsim/config.py`：

```python
DELTA_T = 1.0
NUM_SLOTS_PER_STEP = 10
SLOT_DURATION = DELTA_T / NUM_SLOTS_PER_STEP
```

所以目前模型是：

```text
1 simulation step = 1 秒
1 step 裡有 10 個 access scheduling slots
每個 slot = 0.1 秒
```

UAV movement 的時間尺度：

- MAPPO 或 stub policy 每 step 給每台 UAV 一個 action。
- action 包含 `rho` 和 `psi`。
- `rho` 決定這秒要飛多遠，`psi` 決定方向。
- UAV 在 step 開頭一次移動到新位置。

Access allocation 的時間尺度：

- 在同一個 step 裡，`run_access_pf_step()` 會跑 `NUM_SLOTS_PER_STEP` 次。
- 每個 slot 會重新選 sub-channel 上服務哪個 associated user。
- 因此 **allocation 比 UAV movement 快**。

但要注意，目前還不是「邊飛邊在每個 slot 更新幾何位置」。

目前是：

```text
step 開始：UAV 一次移動到新位置
接著在這個新位置下做 10 個 access scheduling slots
step 結束
```

不是：

```text
slot 1: UAV 在中間位置 10%，算 channel
slot 2: UAV 在中間位置 20%，算 channel
...
slot 10: UAV 到達終點
```

這是合理的 step-based abstraction，但如果未來要更真實，可以把 step 內 movement 插值到 slot-level。不過那會讓 channel/interference/rate 計算成本增加，也會讓 simulator 更接近 packet/TTI-level model。

---

## 6. Backlog 是什麼

`backlog` 可以先用一個很生活化的比喻理解：

```text
user 每秒都有新資料想送出去。
送不出去的資料就排隊。
這個隊伍長度就是 backlog。
```

在程式裡，ground user 的 backlog 是：

```text
user.user_access_backlog_bits
```

每個 step：

- user 依照 `demand_rate_bps * DELTA_T` 產生新 bits。
- 新 bits 加到 `user_access_backlog_bits`。
- access PF scheduling 成功服務的 bits 會從 user backlog 扣掉。
- 被 access 上傳成功的 bits 會進入 UAV relay queue。
- relay/backhaul 成功後，才算真正 delivered 到 destination。

所以 backlog 一直上升通常表示：

- demand arrival 大於 access + relay + backhaul 的總服務能力。
- 或 UAV 位置/association/coverage 導致 users 沒有被有效服務。
- 或 reward 沒有給 policy 足夠誘因去改善服務瓶頸。

這不一定是 simulator 壞掉；在 stress/hard scenario 裡，backlog 上升可能正是「系統過載」的正常現象。

---

## 7. Association 怎麼做

Association 的工作是：每個 user 決定要連到哪一台 UAV。

目前不是 max power，也不是直接 max SINR。現在是比較接近：

```text
依 backlog 高低排序 users
對每個 user 估計連到每台 UAV 的 upper-bound rate
如果設定 max_access_range_m，先排除水平距離太遠的 UAV
考慮該 UAV projected load 後得到 proxy rate
proxy rate 必須大於 association_min_rate_bps
選 proxy rate 最大的 UAV
```

工程 evidence：

- `select_strongest_feasible_uav()` 用 `a2g_upper_bound_rate_bps()` 計算 upper-bound rate：`rl_uavnetsim/allocation/user_association.py:34`
- projected load 是目前 load + 1：`rl_uavnetsim/allocation/user_association.py:41`
- proxy rate = upper bound / projected load：`rl_uavnetsim/allocation/user_association.py:43`
- `max_access_range_m` 是 2D horizontal association hard gate，太遠就跳過：`rl_uavnetsim/allocation/user_association.py:48`
- rate feasibility gate 是 `proxy_rate_bps < min_rate_bps` 就跳過：`rl_uavnetsim/allocation/user_association.py:51`
- users 依 backlog descending 排序：`rl_uavnetsim/allocation/user_association.py:72`

這樣設計有幾個含義：

- 它不是「誰訊號最強就給誰」。
- 它有考慮 UAV load，所以會避免所有 users 都擠到同一台 UAV。
- 高 backlog users 會先被處理，較遠且 backlog 低的 users 可能比較晚被拒絕。
- `association_min_rate_bps` 現在是重要 scenario knob，可以讓 coverage 不再永遠等於 1。
- `max_access_range_m` 是更物理直覺的 scenario knob，可以模擬 UAV access coverage radius，不用把 SAR 任務門檻硬拉到不合理的 14 Mbps。

這也是之前解決 `coverage_ratio` 永遠 1.0 的關鍵：單純把 users 放到地圖角落不夠，因為 link budget 太寬；後來先把 association feasibility threshold 做成可設定，再加入 `max_access_range_m`，讓「無人機不可能無限遠服務 user」這件事更合理。

---

## 8. Sub-channel allocation 怎麼做

Access allocation 的主函式是 `run_access_pf_step()`。

它每個 step 做多個 slots，每個 slot 裡：

```text
for each slot:
    for each UAV:
        for each subchannel:
            看這台 UAV 關聯到哪些 users
            排除 backlog 已經沒了的 user
            計算同 subchannel 上其他 UAV 造成的 interference
            計算 SINR
            SINR 不達標就不能用
            計算 instantaneous rate
            用 PF score 選 user
```

PF score 大概是：

```text
score = instantaneous_rate / (avg_throughput + epsilon) ^ alpha
```

意思是：

- `instantaneous_rate` 高：現在 channel 好，值得服務。
- `avg_throughput` 高：過去已經吃很多資源，score 會被壓低。
- `alpha` 越大：越重視公平。
- `alpha` 越小：越偏向當下 throughput。

工程 evidence：

- slot loop 在 `run_access_pf_step()` 裡：`rl_uavnetsim/allocation/resource_manager.py:87`
- 每個 subchannel 會找 best user：`rl_uavnetsim/allocation/resource_manager.py:110`
- interference 來自其他 UAV 在同 subchannel 的 tentative assignment：`rl_uavnetsim/allocation/resource_manager.py:115`
- SINR threshold filter：`rl_uavnetsim/allocation/resource_manager.py:135`
- PF score 計算：`rl_uavnetsim/allocation/resource_manager.py:150`
- tentative assignment 後，final recomputation 會重新算一次 interference/rate：`rl_uavnetsim/allocation/resource_manager.py:188`

這裡有個重要限制：allocation 目前是規則式，不是 MAPPO 控制。MAPPO 只能透過移動 UAV 改變 geometry，間接影響 channel、SINR、association、relay connectivity。

---

## 9. Alpha controller 與 LinUCB 目前的角色

`alpha` 是 PF scheduling 裡控制 throughput/fairness tradeoff 的指數。

目前 `_select_pf_alpha()` 支援三種來源：

- 直接傳 `alpha_by_uav`。
- 傳 `alpha_controllers`，每台 UAV 的 controller 依 context 回傳 alpha。
- 都沒有時使用 `config.PF_ALPHA_DEFAULT`。

這是為了讓未來 LinUCB、rule-based alpha controller、甚至 learned controller 都可以替換進來。

目前 `LinUCBStub` 還不是完整 LinUCB。它做的事情很簡單：

- 有 alpha candidates：`(0.0, 0.5, 1.0, 2.0)`。
- 回傳 fixed alpha 或最接近 fixed alpha 的 candidate。
- 記錄 last context 和 reward history。

所以報告或論文中不能說「已完成 LinUCB adaptive alpha learning」。比較準確的說法是：

```text
目前已建立 alpha controller interface，LinUCB 仍是 stub / placeholder。
```

---

## 10. Relay 與 backhaul 怎麼做

Access scheduling 只是把 user backlog 上傳到 UAV 的 relay queue。資料真正 delivered 需要 relay/backhaul。

目前 relay 流程：

- 建 UAV-UAV capacity matrix。
- 建 adjacency matrix。
- 算 `lambda2` 作為 connectivity 指標。
- routing table 對每台 UAV 找到 gateway path。
- 每個 step 只 forward one hop。
- 使用 staging buffer，避免同一步內資料連跳多 hop。

工程 evidence：

- relay service 從 `execute_relay_service()` 開始：`rl_uavnetsim/network/relay.py:112`
- snapshot queue + next queue + staging buffer：`rl_uavnetsim/network/relay.py:150`
- forward 到 next hop 先放 staging buffer：`rl_uavnetsim/network/relay.py:184`
- step 結尾才 merge staging buffer：`rl_uavnetsim/network/relay.py:192`

Backhaul 流程：

- active gateway UAV 把自己的 relay queue 用 backhaul capacity 送出。
- backhaul capacity 可以來自 satellite channel，也可以在 demo 裡 override。
- delivered bits 會加到對應 user 的 delivered counter。

工程 evidence：

- backhaul service 從 `execute_backhaul_service()` 開始：`rl_uavnetsim/network/relay.py:203`
- gateway budget = capacity * delta_t：`rl_uavnetsim/network/relay.py:235`
- gateway proportional dequeue：`rl_uavnetsim/network/relay.py:243`
- delivered bits 回寫 user：`rl_uavnetsim/network/relay.py:248`

---

## 11. MAPPO training 現況

目前 MAPPO training 走這條路：

```text
YAML RunConfig
    ↓
build_training_env()
    ↓
build_demo_entities() + SimEnv
    ↓
PettingZooUavNetEnv
    ↓
TorchRL-style actor/critic modules
    ↓
rollout, PPO update, checkpoint, eval
```

重要事實：

- `build_training_env()` 目前要求 satellite backhaul：`rl_uavnetsim/training/mappo_trainer.py:77`
- map size、user distribution、association threshold、reward config 都從 YAML 進來。
- `PettingZooUavNetEnv` 使用 agent 名稱 `uav_0`, `uav_1`, ...
- action space 是 2 維：`rho`, `psi`。
- critic 是 centralized critic，吃所有 agent observation concat。
- evaluation 會同時跑 trained policy 和 static baseline。

Static baseline 是什麼？

```text
rho = 0
```

也就是 UAV 不動。這個 baseline 非常重要，因為之前訓練後 UAV 可能學到靜止或同向飛，如果 trained policy 沒贏過 static baseline，就表示目前 reward/scenario 仍不夠有移動誘因。

Evaluation schema v2 會寫：

- `policies.trained`
- `policies.static`
- `comparison.trained_minus_static_reward`
- `comparison.static_baseline_not_beaten`
- movement diagnostics，例如 path length、net displacement、rho stats。

工程 evidence：

- episode-aware evaluation 會記錄 metrics 和 movement：`rl_uavnetsim/training/mappo_trainer.py:740`
- schema v2 summary 包含 trained/static/comparison：`rl_uavnetsim/training/mappo_trainer.py:872`
- staged clean rewrite 避免 stale eval artifacts：`rl_uavnetsim/training/mappo_trainer.py:910`

---

## 12. Observation：compact_v1 與 compact_v2

目前 training observation 不直接用 `rl_interface/mdp.py` 裡的 full observation，而是走 training 專用 compact observation preset registry。

`compact_v1`：

- self position/velocity
- self is gateway capable
- self relay queue
- other UAV positions/velocities
- nearest K users positions

`compact_v2` 在 v1 基礎上多加：

- self associated user count
- local user backlog norm
- local user associated flag

工程 evidence：

- observation preset registry：`rl_uavnetsim/training/observation_presets.py:27`
- `compact_v1` 和 `compact_v2` 都在 registry：`rl_uavnetsim/training/observation_presets.py:80`
- `compact_v2` dim = `8 + 5*(N-1) + 4*K`：`rl_uavnetsim/training/features.py:16`

為什麼要 compact？

因為 MARL 訓練很容易被過大的 observation dimension 拖垮。原本 full observation 會放更多 network state，但 v1/v2 training 先保留幾何、局部 user、基本 urgency signal，讓模型先有機會學到移動和服務品質的關係。

---

## 13. Reward 與 metrics

Reward 目前是 run-aware config，不再只吃全域常數。

主要項目：

- throughput reward
- energy penalty
- outage penalty
- access backlog penalty
- relay queue penalty
- connectivity penalty
- safety penalty
- coverage gap penalty
- effective coverage gap penalty
- fairness gap penalty

工程 evidence：

- `TeamRewardConfig` 定義 reward knobs：`rl_uavnetsim/rl_interface/mdp.py:31`
- `compute_team_reward()` 讀 runtime reward config：`rl_uavnetsim/rl_interface/mdp.py:279`
- coverage/effective coverage/fairness gap penalty 已接進 reward：`rl_uavnetsim/rl_interface/mdp.py:311`

Metrics 的解讀：

- **throughput**：每 step delivered bits / `DELTA_T`。
- **backlog**：還在 user 端等著上傳的 bits。
- **relay queue**：已上傳到 UAV，但還沒 relay/backhaul delivered 的 bits。
- **coverage ratio**：有被某台 UAV associate 的 user 比例。這是 association coverage。
- **effective coverage ratio**：有 association 且 `final_rate_bps >= outage_threshold_bps` 的 user 比例。這是 end-to-end effective coverage，會同時反映 access、relay、backhaul 是否真的把資料送出去。
- **outage ratio**：final rate 低於 threshold 的 user 比例。
- **Jain fairness**：user rate 分布公平性。
- **lambda2**：UAV relay graph 的 algebraic connectivity；大於 0 通常表示連通。
- **demand satisfaction ratio**：delivered bits / arrived bits。
- **energy efficiency**：delivered bits / Joule。

要特別注意：

```text
coverage = 1.0 只表示 users 有 association，不表示服務品質好。
```

如果 coverage 是 1.0，但 outage 很高、backlog 一直上升，代表「都有連上，但服務不夠」。

現在分析 SAR / hard scenario 時，建議同時看：

```text
coverage_ratio            = 有沒有連上 UAV
effective_coverage_ratio  = 有沒有真的達到 end-to-end service threshold
```

後者才比較接近「搜救任務裡這個 user 是否被有效服務」。

---

## 14. Scenario / YAML 怎麼看

目前常見 config：

- `configs/marl/mappo_satellite.yaml`：基本 MAPPO satellite training。
- `configs/marl/mappo_satellite_3uav_medium.yaml`：3 UAV 中等壓力版本。
- `configs/marl/mappo_satellite_3uav_coverage.yaml`：coverage-challenging 場景，你本地目前有改成更長訓練版本。
- `configs/marl/mappo_satellite_3uav_hotspot_hard.yaml`：hotspot + high demand + compact_v2。
- `configs/marl/mappo_satellite_3uav_large_map.yaml`：3000m x 3000m large-map 場景。
- `configs/marl/mappo_satellite_3uav_sar_lowrate_coverage.yaml`：SAR low-rate 場景第一版，低速率需求 + separated hotspots + finite access range。
- `configs/marl/mappo_satellite_3uav_sar_lowrate_balanced.yaml`：目前較推薦的 SAR low-rate 調參版，讓 UAV 看全 users，降低 backlog 主導，強化 effective coverage。

幾個最重要的 knob：

- `num_steps`：每個 episode 有幾個 simulator steps。
- `num_uavs` / `num_users`：agent/user 數量。這會影響 model shape，checkpoint 通常不能跨 shape。
- `user_demand_rate_bps`：每個 user 每秒產生多少需求。
- `association_min_rate_bps`：association feasibility gate。提高它會讓 coverage 更容易掉到 1 以下。
- `max_access_range_m`：UAV-user 2D horizontal association range。這比硬拉高 `association_min_rate_bps` 更像真實覆蓋半徑限制。
- `user_distribution`：目前有 `uniform`、`hotspots`、`separated_hotspots`。其中 `separated_hotspots` 會把 users 放到四個分離 cluster，專門用來避免 static UAV 天然服務全地圖。
- `spawn_margin`：user 初始化離地圖邊界的 margin。
- `map_length_m` / `map_width_m`：large-map geometry。
- `observation.preset`：`compact_v1` 或 `compact_v2`。
- `observation.max_obs_users`：每台 UAV observation 會看到多少 nearby users。SAR balanced 版設成 60，因為場景分散，若只看 20 個 users 可能學不到全局分工。
- `reward.target_effective_coverage` / `reward.effective_coverage_gap_coef`：讓 reward 直接關心 end-to-end effective coverage，而不是只關心 association。
- `reward.*`：訓練誘因。
- `eval.run_static_baseline`：evaluation 是否跑 static baseline。

本地目前 `mappo_satellite_3uav_coverage.yaml` 顯示：

- `num_steps: 200`
- `total_frames: 210000`
- `device: cuda`
- `observation.preset: compact_v2`

這看起來是你正在跑更長訓練的版本。新視窗接手時要記得先看 `git diff configs/marl/mappo_satellite_3uav_coverage.yaml`，確認這是想保留的實驗設定還是臨時改動。

目前 SAR low-rate 方向有兩份 config：

- `mappo_satellite_3uav_sar_lowrate_coverage.yaml`：第一版，`max_obs_users: 20`、`target_effective_coverage: 0.70`、`access_backlog_coef: 0.6`。
- `mappo_satellite_3uav_sar_lowrate_balanced.yaml`：建議下一輪先跑，`max_obs_users: 60`、`target_effective_coverage: 0.50`、`effective_coverage_gap_coef: 12.0`、`access_backlog_coef: 0.25`、`entropy_coef: 0.003`。

balanced 版的設計意圖是：讓 UAV 看得到完整 user 分布，避免 reward 被 backlog 完全主導，並減少 policy 後期亂飄。

---

## 15. 歷史上已解決的問題

### 15.1 Reward normalization mismatch

之前 3 UAV / 30 users / 2 Mbps/user 的 scenario 裡，reward normalization 還用舊的 60 users / 0.5 Mbps/user reference，造成 throughput term 被放大，reward 幾乎變成常數。

現在已改成 run-aware reference scales 和 runtime reward config。

### 15.2 Coverage ratio 永遠 1.0

一開始以為把 user 分散到邊角就會掉 coverage，但 link budget 檢查後發現 2000m x 2000m 地圖上，即使角落 user 也可能還能 association。

後來的方向是：

- 分離 outage threshold 和 association threshold。
- 加 `association_min_rate_bps`。
- 加 `spawn_margin`。
- 加 coverage-challenging scenario。

這樣 coverage 才能在實驗中變成有意義的 signal。

### 15.3 Evaluation 圖表亂線

之前多 episode 的 metrics 會被串成一條線，episode boundary 會亂接，看起來像雜訊。

現在 evaluation schema v2 會輸出 episode-aware artifacts：

- 每個 episode 自己的 summary/history/plots/trajectory。
- aggregate plots 不跨 episode boundary。
- 另有 mean line。

### 15.4 Static baseline 不清楚

之前看 trajectory 時只能肉眼猜 policy 是不是靜止。

現在 evaluation 會記錄：

- per-agent path length
- net displacement
- rho mean/min/max
- trained vs static reward delta

所以如果 UAV 幾乎不動，可以用數字直接指出。

### 15.5 Large-map geometry

之前地圖大小很多地方吃全域 `config.MAP_LENGTH` / `config.MAP_WIDTH`，容易發生 YAML 寫 3000m，但 satellite、normalization、visualization 還在用 2000m 的問題。

現在有 `ScenarioGeometry`，並把 spawn、UAV clamp、user mobility bounds、satellite position、normalization、visualization axes 接到 scenario map size。

### 15.6 `linucb_controllers` 改名

之前命名會讓人以為只能放 LinUCB。現在改成 `alpha_controllers`，語義比較正確：任何能選 PF alpha 的 controller 都可以接。

### 15.7 State / observation redesign

之前 observation 設計沒有充分想清楚。後來改成：

- UAV 位置/速度用 normalized absolute geometry。
- training path 有 compact presets。
- `compact_v1` 保留相容。
- `compact_v2` 加入 urgency/association 相關資訊。

### 15.8 SAR low-rate coverage 語義修正

原本為了讓 coverage 掉到 1 以下，曾經用過 `association_min_rate_bps: 14.0e6`。這在工程上有效，但對搜救任務不太合理，因為真實 SAR 通訊可能只需要幾百 kbps 到 0.5 Mbps。

現在方向改成：

- 保留低速率需求，例如 `0.5 Mbps/user`。
- 用 `max_access_range_m` 表示 UAV access radius。
- 用 `separated_hotspots` 把 users 放到分離區域。
- 保留 `coverage_ratio` 作 association 指標。
- 新增 `effective_coverage_ratio` 作 end-to-end service quality 指標。

這樣比較符合「UAV 必須飛去服務遠端群聚 users」的研究故事，而不是靠不合理的高 Mbps threshold 製造困難。

---

## 16. 當前仍遇到的問題

### 16.1 Policy 可能學成靜止或同向飛

這是目前最重要的研究問題之一。

可能原因：

- scenario 靜態初始部署已經夠好，UAV 不動也能 coverage 高。
- energy penalty 讓移動看起來不划算。
- reward 沒有足夠強調 coverage/fairness/backlog 改善。
- observation 雖有 geometry/backlog，但仍可能不足以形成清楚 action gradient。
- MAPPO baseline 訓練時間或超參數不足。

現在的處理方式：

- 加 static baseline 直接比較。
- 加 coverage/hotspot/large-map hard scenarios。
- 加 `compact_v2`，讓 policy 看到 local backlog 和 association flag。

但這還不保證 policy 一定學會合理飛行，需要繼續實驗。

最近的 SAR low-rate run：

```text
runs/mappo_satellite_3uav_sar_lowrate_coverage/20260425-205041
```

顯示情況已經改善，但還沒完全合理：

- `best.pt` trained reward 明顯打敗 static：`-2363.3` vs `-3795.4`。
- `best.pt` 平均 path length 約 `1306.8 m`，`rho mean = 0.653`，所以不是靜止 policy。
- static effective coverage 約 `0.022`，best trained 約 `0.287`，代表 UAV 移動確實增加有效服務。
- 但 target effective coverage 是 `0.70`，best 只有 `0.287`，差距仍大。
- `latest.pt` 比 `best.pt` 差，代表訓練後期退化。
- best policy 的 `lambda2` 平均約 `0.864`，代表部分 step relay graph 斷線；latest 更差，約 `0.55`。
- trajectory 顯示 UAV 有往某些 hotspot 飛，但比較像「往部分 cluster 衝過去」，還不是穩定的多 UAV 分工覆蓋。

這個 run 的解讀是：新的 SAR 場景已經成功讓 static baseline 變弱，也讓 MAPPO 有移動收益；但場景/observation/reward 還需要調，否則 policy 會追逐部分 hotspot、犧牲 relay connectivity，並且訓練不穩。

### 16.2 Hard scenario 是否足夠有研究價值

目前有 `hotspot_hard`、`large_map`、`sar_lowrate_coverage`、`sar_lowrate_balanced`，但它們是否真的逼出「移動有益且合理」的策略，需要看訓練後：

- trained 是否打敗 static baseline。
- backlog growth 是否變慢。
- coverage/outage/fairness 是否改善。
- UAV path length 是否不是 0，但也不是亂飛。
- `effective_coverage_ratio` 是否提高，而不只是 association coverage 提高。
- `lambda2` 是否維持足夠連通，避免 UAV 飛出去後 relay/backhaul 斷掉。

目前建議下一輪優先跑：

```text
configs/marl/mappo_satellite_3uav_sar_lowrate_balanced.yaml
```

它相對第一版 SAR config 有幾個調整：

- `max_obs_users: 60`，避免 UAV 只看到局部 users。
- `target_effective_coverage: 0.50`，比 0.70 更符合 3 UAV / 4 hotspots 的可達性。
- `effective_coverage_gap_coef: 12.0`，讓有效覆蓋更重要。
- `access_backlog_coef: 0.25`，降低總 backlog 對 reward 的壓倒性主導。
- `entropy_coef: 0.003`，降低後期 policy 亂飄。

### 16.3 Noisy-MAPPO 尚未落地

原 proposal 的 Noisy-MAPPO 是重要研究點，但目前還只是 future direction。

如果要實作，建議順序：

1. 先確認 plain MAPPO baseline 在 hard scenario 能打敗 static。
2. 再加入 noisy critic 或 noisy value augmentation。
3. 比較 MAPPO vs Noisy-MAPPO vs static/random。

### 16.4 LinUCB 還不是完整 learning controller

目前 alpha controller interface 已經存在，但 `LinUCBStub` 沒有完整 UCB update/selection。

未來如果要把 two-timescale resource allocation 做成研究貢獻，應補：

- per-UAV LinUCB state。
- context feature design。
- alpha candidate reward attribution。
- comparison：fixed alpha vs LinUCB alpha。

### 16.5 Step-based abstraction 的精細度

目前 UAV 每 step 移動一次，access scheduling 在 step 內做多個 slots，但 slot 內不更新 UAV/user geometry。

這對目前研究足夠簡潔，但如果要更接近真實 TTI-level system，未來可以加入 slot-level geometry interpolation。

---

## 17. 快速命令

安裝 MARL dependencies：

```bash
pip install -e '.[marl]'
```

跑測試：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m pytest -q tests
```

跑 satellite demo：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.main --steps 12 --backhaul-type satellite --output-dir demo_outputs
```

跑 stress satellite demo：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.main --demo-mode stress --backhaul-type satellite --steps 50 --num-users 100 --output-dir demo_stress_satellite
```

訓練基本 MAPPO：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite.yaml
```

訓練 coverage scenario：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite_3uav_coverage.yaml
```

訓練 hotspot hard：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite_3uav_hotspot_hard.yaml
```

訓練 large-map：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite_3uav_large_map.yaml
```

訓練 SAR low-rate 第一版：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite_3uav_sar_lowrate_coverage.yaml
```

訓練 SAR low-rate balanced 版，這是目前建議下一輪優先跑的版本：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.train --config configs/marl/mappo_satellite_3uav_sar_lowrate_balanced.yaml
```

evaluate checkpoint：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.evaluate --config configs/marl/eval_satellite.yaml --checkpoint runs/<experiment>/<timestamp>/checkpoints/best.pt
```

分析 run：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.analyze_run --run-dir runs/<experiment>/<timestamp>
```

看本地改動：

```bash
git status --short --branch
git diff -- configs/marl/mappo_satellite_3uav_coverage.yaml
```

---

## 18. 新視窗接手建議

如果你開新視窗後只想快速恢復脈絡，建議這樣問：

```text
請先閱讀 docs/reports/2026-04-25-project-current-state-overview-zh.md，
再看 README.md、rl_uavnetsim/environment/sim_env.py、
rl_uavnetsim/allocation/resource_manager.py、
rl_uavnetsim/training/mappo_trainer.py。
請根據目前 git status 判斷哪些是 committed baseline，哪些是本地實驗修改。
```

接著如果要做實驗分析，先跑：

```bash
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m pytest -q tests
MPLCONFIGDIR=/tmp/mpl ~/.venv/bin/python -m rl_uavnetsim.training.analyze_run --run-dir <run-dir>
```

然後重點看：

- trained 是否打敗 static baseline。
- path length / net displacement 是否接近 0。
- backlog 是否一直線性上升。
- association coverage 是否真的會掉到 1 以下。
- effective coverage 是否比 static 明顯提高。
- outage/fairness 是否有改善。
- lambda2 是否常常掉到 0，若掉到 0 代表 UAV 可能飛出去服務 users 但犧牲 relay connectivity。

如果 trained policy 仍然靜止，下一步不要急著改 MAPPO code。應先確認：

- scenario 是否真的讓移動有收益。
- reward 是否對 coverage/outage/backlog/fairness 給了足夠壓力。
- observation 是否包含 policy 需要的 urgency signal。
- static baseline 是否已經太強。

如果 trained policy 有飛，但只衝向部分 hotspot 或後期退化，下一步也不要急著改 MAPPO code。應先確認：

- `max_obs_users` 是否太小，導致 UAV 看不到完整 user 分布。
- `target_effective_coverage` 是否超出當前 UAV 數量/場景幾何可達範圍。
- `access_backlog_coef` 是否太大，讓 reward 只在乎總 backlog，不在乎有效覆蓋與公平。
- `entropy_coef` 是否太高，讓 policy 後期持續探索、破壞已找到的較好策略。
- connectivity penalty 是否足夠防止 UAV 飛太散、relay graph 斷線。

---

## 19. 總結

目前這個專案已經從「proposal 中的概念」走到「可跑、可訓練、可分析」的 simulator baseline：

- step-based network simulation 已完成。
- access / relay / satellite backhaul dynamics 已完成。
- MAPPO training pipeline 已完成。
- compact observation presets 已完成。
- configurable hard scenarios 已完成。
- SAR low-rate finite-range scenario 已完成。
- effective coverage metric/reward 已完成。
- evaluation artifact 和 static baseline diagnostics 已完成。

但研究主張還要小心：

- 不能說 Noisy-MAPPO 已完成。
- 不能說 LinUCB 已完整 learning。
- 不能只看 coverage = 1.0 就說服務很好。
- 不能只看 reward 上升就說 UAV 學會合理移動。

目前最值得繼續做的，是用 `compact_v2 + SAR low-rate balanced scenario + static baseline` 找出一個 trained policy 不只比不動好，而且能穩定分工覆蓋多個 hotspot、維持 relay connectivity 的 case。那會是後續 Noisy-MAPPO / LinUCB / richer scenario 的穩固起點。
