# Disaster UAV / MAPPO Related Work 讀書筆記與專案改進建議

日期：2026-04-25

目的：整理 `docs/relatedwork/` 中與 disaster-area UAV emergency communications、coverage optimization、relay/fairness、O-RAN UAV-BS 相關的論文，回答目前專案最卡的問題：

- 為什麼 static UAV 很容易 coverage ratio 接近 1？
- 是否應該用很高的 `association_min_rate_bps` 製造困難？
- 真實搜救任務中，UAV 覆蓋範圍、user demand、channel、mobility 應該怎麼設計比較合理？
- 下一版 simulator / MAPPO scenario 應該優先改什麼？

這份筆記不是完整文獻回顧，而是「為了讓本專案下一步更合理」的工程導向讀書筆記。

---

## 0. 一頁結論

我不建議繼續用 `association_min_rate_bps: 14.0e6` 當主要困難來源。

`14.0e6` 是 `14 Mbps`，不是 `14 MB/s`，但對搜救任務來說仍然偏高。若任務是座標回報、文字、低碼率語音、簡單影像或感測資料，最低可用需求更可能是數百 kbps 到 1.5 Mbps 左右。把 association 門檻拉到 14 Mbps 雖然能讓 coverage 掉下來，但研究敘事會變得不自然：像是在假設每個受災區 user 都要高速影音，而不是 emergency connectivity。

更合理的做法是保留低速率搜救需求，讓困難來自：

- 災區範圍大或 users 分散。
- UAV 有有限的有效服務範圍或 observation range。
- user 會移動，且分布不是固定在中心。
- urban / disaster channel 有遮蔽、NLoS、shadowing 或額外 loss。
- coverage 用 SINR / outage / effective service 判定，而不只是「能被 association」。
- UAV 數量有限，service load / relay queue / backhaul 會成為瓶頸。
- static baseline 不能同時照顧所有 hotspot，MAPPO 才有理由移動。

下一版我會建議做一個 `sar_lowrate_coverage` 類 scenario：

```yaml
env:
  num_uavs: 3
  num_users: 50 或 60
  map_length_m: 3000.0
  map_width_m: 3000.0
  user_distribution: separated_hotspots 或 poisson_hotspots
  user_speed_model: maxwell_boltzmann
  user_speed_mean_mps: 1.5 到 2.2
  user_demand_rate_bps: 0.5e6
  association_min_rate_bps: 0.5e6
  max_access_range_m: 600.0 到 900.0
  access_extra_loss_db: 4.0 到 8.0

reward:
  outage_threshold_bps: 0.5e6
  target_coverage: 0.80 到 0.90
  coverage_gap_coef: 5.0
  relay_queue_coef: 提高
```

這樣的故事會比 `14 Mbps` 更乾淨：

「每個 user 的需求不高，但災區大、users 分散且移動、UAV 有有效服務半徑與遮蔽限制，所以 static deployment 無法一直覆蓋所有人。MAPPO 必須移動 UAV 來改善 coverage、outage、fairness、backlog 與 relay/backhaul。」

---

## 1. 讀了哪些論文

本輪主要讀與掃描以下 8 篇：

| 檔案 | 對本專案的用途 |
|---|---|
| `Cooperative_UAV_Trajectory_Design_for_Disaster_Area_Emergency_Communications_A_Multiagent_PPO_Method.pdf` | 最貼近目前 MAPPO / disaster trajectory 設計，重點深讀 |
| `Maximizing coverage in UAV-based emergency communication networks.pdf` | coverage reward、rescue priority、低門檻 emergency coverage 很有參考價值 |
| `Multi-UAV_Assisted_Network_Coverage_Optimization_for_Rescue_Operations_using_Reinforcement_Learning.pdf` | rescue members、restricted communication range、energy、connected UAV network |
| `Enhancing_Mobile_Network_Performance_Through_ORAN-Integrated_UAV-Based_Mobility_Management.pdf` | UAV-BS 有明確 communication range，可用來反駁「UAV 覆蓋無限大」 |
| `UAV-Assisted_Relay_Communication_A_Multi-Agent_Deep_Reinforcement_Learning_Approach.pdf` | relay 任務、move/collect/upload action、control-link outage |
| `UAV-assisted fair communications for multi-pair users A multi-agent deep RL method.pdf` | fairness、large sparse area、static deployment 不足、UAV-UAV range |
| `RL-Driven_Security-Aware_Resource_Allocation_for_UAV-Assisted_O-RAN_in_SAR_Operations.pdf` | SAR resource allocation、state/action/reward 可作未來 O-RAN 擴充 |
| `When_RAN_Intelligent_Controller_in_O-RAN_Meets_Multi-UAV_Enable_Wireless_Network.pdf` | UAV trajectory + resource/offloading，偏未來架構參考 |

---

## 2. 深讀：Cooperative UAV Trajectory Design for Disaster Area Emergency Communications

這篇是目前最貼近我們需求的 reference。它處理的是多 UAV 在災區中即時調整 trajectory，幫 mobile users 恢復與 ground base stations 的通信。它的核心不是把 user demand 設得很高，而是把問題建成動態、部分可觀測、多 UAV 協作的 emergency communication problem。

### 2.1 它的系統設定

它的場景概念是：

```text
Mobile Users <-> UAV RF access/relay <-> FSO backhaul <-> Ground Base Stations
```

重點設計：

- Disaster area 是 urban emergency communication scenario。
- Mobile users 的空間分布使用 Poisson pattern。
- Users 會移動，速度模型使用 Maxwell-Boltzmann distribution 描述災區人群的不規則移動。
- UAV 裝 RF module 服務 users，也裝 FSO module 做 backhaul。
- UAV trajectory 是 MAPPO 控制的主要對象。
- 系統是 time-slotted，slot 內 quasi-static。
- 每個 user 最多由一台 UAV 服務。
- 使用 SINR threshold 判定 user 是否 off-state / disconnected。
- 總 bandwidth 與 backhaul capacity 都是限制條件。

這跟我們目前很接近，但有幾個差別：

- 我們目前是 satellite backhaul，不是 FSO-to-GBS。
- 我們目前 user distribution 主要是 uniform / hotspots，還沒有 Poisson / Maxwell-Boltzmann user mobility。
- 我們 association 目前主要用 proxy rate feasibility，不是直接用 SINR/off-state 做 association 失敗判定。
- 我們目前 training 是 single active gateway satellite case；這篇是多 UAV RF/FSO relay。

### 2.2 Channel / coverage 的關鍵

這篇用 probabilistic LoS channel：

- LoS / NLoS path loss。
- elevation angle 影響 LoS probability。
- SINR 包含其他 UAV 的 aggregate interference。
- 如果 user 對所有 UAV 的 SINR 都低於 threshold，就算 disconnected / off-state。

這給我們一個很重要的啟發：

coverage 不應該只看「能不能被選到一台 UAV」，而應該更接近：

```text
effective coverage =
  user 被某台 UAV 服務
  且 SINR / rate / outage 條件達標
```

目前我們的 coverage 容易等於 1，原因之一是 association 先用無干擾 upper-bound rate 做 feasibility。PF scheduling 後即使實際服務品質不好，coverage 仍可能看起來很好。

這裡需要很小心：**下一版不應直接改掉既有 `coverage_ratio` 的語義。**

目前程式中的 `coverage_ratio` 是 association coverage，也就是：

```text
coverage_ratio = associated_user_count / total_user_count
```

它在 `metrics_collector.py` 內由 `user.associated_uav_id >= 0` 計算。這個欄位已經存在於 schema v1/v2 的舊 run，如果我們偷偷把它改成 effective coverage，舊 run 和新 run 會變得不可比較。

因此 metric migration 應該採「新增欄位，不偷換舊欄位」：

```text
coverage_ratio:
  保持原意，代表 association coverage。

effective_coverage_ratio:
  新增欄位，代表 user 被 associate 且實際服務品質達標。

access_range_coverage_ratio:
  可選欄位，代表 user 是否在任一 UAV 的 reliable access range 內。
```

推薦第一版 effective coverage 定義：

```text
effective_coverage_ratio =
  count(user.associated_uav_id >= 0 and user.final_rate_bps >= outage_threshold_bps)
  / total_user_count
```

這樣 `coverage_ratio` 回答「名義上有沒有接上」，`effective_coverage_ratio` 回答「接上後有沒有真的服務到基本 QoS」。這兩個指標應該同時出現在 evaluation summary 裡，並在 schema version 中明確記錄。

### 2.3 Observation / State / Action

這篇的 observation 包含：

- UAV location。
- UAV speed。
- Mobile user location。
- Mobile user speed。
- GBS location。

State 則包含全域 UAV / MU / GBS 的位置與速度。

Action 很像我們：

```text
action = flight radius ratio + flight angle
```

它使用固定 altitude，UAV 在每個 time slot 內選擇飛行半徑與角度。這跟我們目前 `rho / psi` 的設定是同一類設計，因此我們的 action abstraction 本身是合理的。

但 observation 上我們少了幾個東西：

- local user velocity。
- other UAV 是否 gateway-capable。
- other UAV 是否 active gateway。
- GBS / satellite / gateway anchor 的明確位置訊息。
- user cluster / weighted centroid 類 summary。

尤其我們目前 `compact_v2` 的 other UAV block 只有位置與速度，沒有「誰是 gateway」。這可能導致 non-gateway UAV 不知道 relay/backhaul anchor 在哪裡，學出固定漂移或無意義聚集。

### 2.4 Reward

這篇 reward 同時考慮：

- RF throughput。
- FSO/backhaul rate。
- connected users 數量。
- disconnected users 的影響。

這點對我們很重要。上一輪 `hotspot_hard` run 顯示 trained policy 把 access backlog 壓低，但 relay queue 線性上升到約 1.1 Gbits。這表示 policy 可能只是把 traffic 從 user queue 搬進 relay queue，沒有真正改善 end-to-end delivery。

所以我們 reward 不應只看 access 端，也要更明確地懲罰：

- relay queue growth。
- arrived minus delivered 的 end-to-end gap。
- backhaul bottleneck。
- disconnected / ineffective served users。

### 2.5 K-means 的啟發

這篇提出 enhanced K-means，目的不是替代 MAPPO，而是降低 MAPPO 學習困難：

- 用 user locations 找 cluster centroid。
- observation range 之外的 users 不分配給該 UAV。
- 如果某 cluster 空掉，就重新給 UAV 合理位置。
- 最後用 cluster centroids 作為 UAV 初始位置，讓 MAPPO 不要從太差的初始狀態開始學。

這對我們非常有用，因為我們現在有一個問題：如果 static initial deployment 太好，MAPPO 沒理由飛；但如果 initial deployment 太差，MAPPO early training 可能又學不到。K-means baseline 可以變成很好的對照：

```text
static center/orbit baseline
random movement baseline
weighted-kmeans reposition baseline
MAPPO learned trajectory
```

這樣教授會更容易相信實驗：MAPPO 不是跟很弱的 baseline 比，也不是被 hard-coded initial layout 幫忙。

---

## 3. Coverage 類論文給我們的提醒

### 3.1 Maximizing coverage in UAV-based emergency communication networks

這篇很適合回答「搜救 user 到底需要多高 rate」。

它的設定中，coverage threshold 是 `1.5 Mb/s` 等級，而不是十幾 Mbps。模擬參數包含：

- target area 約 `2000 m x 1000 m`。
- 1 台 UAV。
- 25 個 ground nodes。
- 1 個 rescue node。
- UAV height 約 `80 m`。
- UAV speed 約 `10 m/s`。
- transmit power `1 W`。
- bandwidth `5 MHz`。
- carrier frequency `2.4 GHz`。

這篇還有一個值得學的設計：rescue node priority。它不只是追求覆蓋最多普通節點，也特別確保 rescue personnel devices 的通信品質。這比單純最大化總 coverage 更符合救災任務。

對我們的啟發：

- `0.5 Mbps` 到 `1.5 Mbps` 作為 emergency service threshold 是比較自然的。
- 可以加入 user priority，例如 `rescuer` / `victim` / `sensor`。
- reward 可以看 coverage count 的變化，而不是只看絕對 throughput。
- 如果 coverage 增加給更大 reward，coverage 下降給 penalty，可以更直接鼓勵 UAV 去救 currently uncovered users。

### 3.2 Multi-UAV Assisted Network Coverage Optimization for Rescue Operations

這篇把 rescue members 的 unpredictability、UAV energy、restricted communication range 都放進問題。

它的設計重點：

- Rescue Members 會不可預測移動。
- UAV 有有限 energy。
- UAV 有 restricted communication range。
- A2G link 用 SINR threshold 判定是否可連。
- 如果多台 UAV 都可服務同一 RM，選 instantaneous SINR 最高的 UAV。
- UAV 可以同時服務多個 rescue members，但會涉及 TDMA / resource sharing。
- Reward 同時考慮 covered rescue members、data rate、energy、UAV network connectivity、backhaul connectedness。

對我們的啟發：

- 加 `max_access_range_m` 是合理的，不是硬調參。很多 UAV coverage / rescue 類研究都明確承認 communication range 是限制。
- Association 可以從 proxy-rate-first 改成更接近 SINR-first / effective service first。
- Reward 不應只看 access backlog，還應包含 connectivity 與 backhaul connectedness。
- UAV action 可以保留 `rho / psi`，但 observation 應該讓 UAV 知道自己對 gateway/backhaul 的關係。

---

## 4. O-RAN / UAV-BS 類論文對「覆蓋範圍」的啟發

### 4.1 Enhancing Mobile Network Performance Through O-RAN-Integrated UAV-Based Mobility Management

這篇不是 disaster-MAPPO 主線，但它給了我們一個很重要的 reality check：UAV-BS 的有效服務半徑可以很小。

它的 simulation parameters：

- scenario area：`3000 m x 2000 m`。
- users：`300 UEs`。
- UAV-BSs：`20`。
- UAV-BS maximum communication range：`250 m`。
- UAV-BS transmit power：`23 dBm`。
- UAV-BS altitude：`10 m`。
- UAV-BS maximum speed：`30 m/s`。
- 使用 hybrid-building propagation loss model。

這對我們很有幫助。因為我們之前直覺覺得「UAV 在 100 m 高空，覆蓋是不是應該很大？」答案是：不一定。若它是 low-altitude platform、城市小區、O-RAN small cell、受建築遮蔽或 QoS 限制，effective communication range 可以是數百米級。

對我們的啟發：

- `max_access_range_m` 不是不合理；它可以代表 reliable service footprint。
- 如果我們維持 100 m altitude，初始可以不要設到 250 m 那麼激進，但 `600 m ~ 900 m` 是可以先試的。
- 如果未來要做低空 UAV-BS scenario，可加入 `uav_altitude_m: 30~80`，服務範圍就更自然地受限。
- weighted K-means 根據 user QoS 需求定位 UAV-BS，也可以成為我們的 baseline。

---

## 5. Relay / fairness 類論文對目前 relay queue 問題的啟發

### 5.1 UAV-Assisted Relay Communication: A Multi-Agent Deep RL Approach

這篇重點是 UAV 作為 relay，處理 ground sensors 到 ground base stations 的 data packets。

它的設定有幾個跟我們很像：

- 3 UAV。
- fixed altitude `100 m`。
- UAV speed `20 m/s`。
- time slot `1 s`。
- 1 km x 1 km region。
- ground sensors / base stations 分布在區域內。
- 使用 SNR threshold 判定 sensor 是否在 service range。
- action 不只是移動，還包含 collect / upload。

它對我們的最大提醒是：relay 任務不是只有「靠近 user」就好。UAV 還要在適當時機把資料送回 base station / gateway。如果 reward 沒有把 relay queue 或 upload/backhaul 行為設好，policy 可能只會把資料收上來，但不會真正送出去。

這剛好對應我們上一輪看到的問題：

```text
access backlog 下降
relay queue 線性上升
end-to-end demand satisfaction 幾乎固定
```

這表示下一版 reward 應該更重視 end-to-end delivered traffic，而不是只讓 access 端看起來乾淨。

### 5.2 UAV-assisted fair communications for multi-pair users

這篇重點是多 UAV relay、多對 ground users、fairness、throughput、energy、UAV connectivity。

對我們有用的觀點：

- 在 users 稀疏、區域較大時，static deployment 可能不足，UAV mobility 才有價值。
- UAV-UAV link 可以用 communication range `Rc` 簡化表示，這在 relay graph / connectivity 中很常見。
- Fairness 不只是漂亮 metric，而是 reward / objective 的一部分。
- UAV connectivity maintenance 應該跟 throughput 一起考慮。

這支持我們已經有的 `lambda2` / connectivity metric，但也提醒一件事：如果 `lambda2` 永遠是 1，代表 scenario 對 relay graph 太容易，connectivity reward 不會形成學習梯度。

---

## 6. O-RAN / SAR resource allocation 類論文的位置

### 6.1 RL-Driven Security-Aware Resource Allocation for UAV-Assisted O-RAN in SAR Operations

這篇比較偏未來 extension，不是當前最急。它把 state/action/reward 設得更複合：

- state：ground user location、battery、data size、UAV previous location 等。
- action：UAV displacement、user association、key length / security related decision。
- reward：energy、latency、security、constraint violations。

對我們的啟發是：未來若要把 `alpha_controllers`、resource allocation、security 或 latency 加入 MARL，可以把它作為參考。但現在首要問題還是讓 MAPPO movement 在合理 rescue scenario 中有必要、有梯度、有可解釋性。

### 6.2 When RAN Intelligent Controller in O-RAN Meets Multi-UAV Enabled Wireless Network

這篇偏 O-RAN + UAV-BS + offloading 架構，重點是 trajectory 與 task/resource allocation 聯合最佳化。它適合放在未來「可替換 resource allocation / online learning / O-RAN controller」方向，不建議現在就搬進核心 simulator。

---

## 7. 用 related work 回頭診斷我們目前問題

### 7.1 目前 access channel / coverage 偏樂觀

目前 `config.py` 中 access link 主要設定為：

- `P_TX_RF = 0.5 W`
- `NUM_SUBCHANNELS = 8`
- `SUBCHANNEL_BW = 1e6`
- `UAV_HEIGHT = 100 m`
- `SHADOW_STD = 0.0`
- `R_MIN = 0.5e6`

程式位置：

- `rl_uavnetsim/config.py`
- `rl_uavnetsim/channel/a2g_channel.py`
- `rl_uavnetsim/allocation/user_association.py`

我用目前模型粗算過，單台 UAV 到 ground user 的 upper-bound access rate 在 1000 m 左右仍有數十 Mbps 級，在 1500 m 仍可能有十幾 Mbps 級。再加上 association feasibility 用的是 no-interference upper-bound rate / projected load，因此 static UAV 很容易讓 coverage ratio 看起來很高。

這不是 MAPPO 壞掉，而是場景太容易。

### 7.2 `association_min_rate_bps = 14 Mbps` 能製造困難，但不夠像搜救

14 Mbps 的確能讓遠距離或高負載 users association fail，但它改變的是「每個 user 都要高 QoS」這個假設。若專案主軸是 disaster / SAR emergency connectivity，這個假設可能被教授挑戰：

- 為什麼搜救 user 每人需要 14 Mbps？
- 若只傳座標、文字、低碼率語音，0.5 Mbps 不就夠了？
- coverage 下降到底是因為災區困難，還是因為門檻硬拉太高？

因此我建議把 `14e6` 降回比較自然的 emergency threshold，然後從環境、range、mobility、channel、load、backhaul 下手。

### 7.3 Current `compact_v2` 還缺 gateway / anchor 資訊

目前 `compact_v2` 已有：

- self position / velocity。
- self gateway-capable。
- self relay queue。
- self associated user count。
- other UAV positions / velocities。
- local user positions。
- local user backlog。
- local user associated flag。

但 other UAV block 沒有：

- other UAV 是否 gateway-capable。
- other UAV 是否 active gateway。
- other UAV relay queue。
- UAV-to-gateway / UAV-to-UAV link status。

這會讓 service UAV 不知道「該靠近哪台 UAV 才能把資料送出去」。在 satellite single-gateway case，這很重要。

---

## 8. 下一版最推薦的設計方向

### 8.1 不要先用高 Mbps 門檻

保留：

```yaml
association_min_rate_bps: 0.5e6
outage_threshold_bps: 0.5e6
user_demand_rate_bps: 0.5e6
```

若要稍微提高，可以先試：

```yaml
association_min_rate_bps: 1.0e6
outage_threshold_bps: 0.5e6 或 1.0e6
```

但不建議直接從 0.5 Mbps 跳到 14 Mbps。

### 8.2 加 `max_access_range_m`

這是最直接也最有文獻支持的改法。它代表 reliable service footprint，不是電波完全傳不到，而是超過這個範圍不視為可穩定服務。

但它的語義要先定死，否則會變成另一個不透明 knob。我的建議是：

```text
max_access_range_m 是 association hard gate。
```

也就是在 association 階段，若 user 與 UAV 的 2D 或 3D 距離超過 `max_access_range_m`，該 UAV 不可作為此 user 的 serving UAV candidate。這代表 reliable access footprint 的物理假設。

第一版不要把它同時塞進 PF scheduling service mask，也不要只作 evaluation-only metric。原因是：

- 如果只做 evaluation-only，training dynamics 沒變，MAPPO 不會真的面對有限覆蓋限制。
- 如果同時改 association 和 PF，第一版較難判斷 coverage drop 是 association gate 還是 slot-level service mask 造成的。
- Association hard gate 最容易解釋，也和「超過 reliable service footprint 就不建立穩定 access association」一致。

推薦 migration：

```text
EnvConfig.max_access_range_m: float | None = None

None:
  維持舊行為，不加 range gate。

float:
  association candidate 必須同時滿足
  distance(user, uav) <= max_access_range_m
  and proxy_rate_bps >= association_min_rate_bps
```

距離計算第一版建議用 2D horizontal distance，因為 `max_access_range_m` 比較像地面覆蓋半徑；3D path loss 仍由 channel model 另外處理。若之後要改成 3D，應新增 `access_range_distance_mode: "2d" | "3d"`，不要默默改。

建議初始 calibration：

```yaml
max_access_range_m: 800.0
```

可測範圍：

```text
600 m：比較困難
800 m：中等困難
1000 m：較保守
```

如果未來做低空 UAV-BS，例如 altitude 10 m 到 50 m，range 可以更小，例如 250 m 到 500 m。

### 8.3 加 user distribution：Poisson / separated hotspots

目前 `hotspots` 還是可能集中得太規則。建議新增：

```yaml
user_distribution: poisson_hotspots
```

或：

```yaml
user_distribution: separated_hotspots
```

設計方式：

- 3 台 UAV。
- 4 到 5 個 user clusters。
- clusters 分散在地圖不同象限。
- 每個 cluster users 數量不均。
- 有少量 outlier users。

這樣 static 初始 orbit 不會完美覆蓋所有群，MAPPO 才需要決定「該去支援哪一群」。

### 8.4 加 user mobility：Maxwell-Boltzmann 或 group drift

指定 MAPPO disaster paper 明確用 Maxwell-Boltzmann distribution 描述 mobile users。這很適合我們。

第一版可以簡化成：

```yaml
user_speed_model: maxwell_boltzmann
user_speed_rms_mps: 1.5
```

或保留 random walk，但加入 hotspot drift：

```yaml
hotspot_drift_enabled: true
hotspot_drift_speed_mps: 1.0
```

drifting hotspot 很重要，因為它會讓 static baseline 隨時間變差。這比直接獎勵 UAV 移動更自然。

### 8.5 加 channel realism：extra loss / shadowing / blockage

目前 `SHADOW_STD` 名字看起來像標準差，但在現有 access channel 中比較像一個固定加到 path loss 的參數，不是真正每條 link 隨機 shadowing。

建議分兩層：

```yaml
access_extra_loss_db: 6.0
shadowing_std_db: 4.0
```

第一階段先做 deterministic extra loss，方便 calibration。

第二階段再做 per-link stochastic shadowing，並用 seed 控制可重現性。

### 8.6 加 user priority

參考 coverage paper，可以把 users 分成：

```text
rescue_member / normal_user / sensor
```

或：

```text
high_priority / normal_priority
```

Reward 不只看總 coverage，也看 high-priority users 是否被服務。這比所有 user demand 都設高更合理。

### 8.7 compact_v3 observation

建議新增 `compact_v3`，不是改掉 `compact_v2`：

Self block 增加或保留：

- self position / velocity。
- self gateway-capable。
- self active-gateway flag。
- self relay queue。
- self associated user count。

Other UAV block 建議變成：

- other position。
- other velocity。
- other is gateway-capable。
- other is active gateway。
- other relay queue norm。
- optional link-active。

Local user block 建議：

- user position。
- user velocity。
- user backlog。
- associated flag。
- priority flag。
- optional in-range flag。

這樣 MAPPO 才知道：

- user 在哪裡、往哪裡動。
- 哪些 user 重要或急。
- 哪台 UAV 是 gateway / anchor。
- 自己移動後是否可能斷 relay/backhaul。

---

## 9. 建議新增的 scenario YAML

### 9.1 `mappo_satellite_3uav_sar_lowrate_coverage.yaml`

用途：低速率 SAR 場景，測試「不是靠高 Mbps 門檻，而是靠有限 range + 分散 users 讓 static 不滿覆蓋」。

建議設定：

```yaml
seed: 61

env:
  num_steps: 100
  num_uavs: 3
  num_users: 60
  backhaul_type: satellite
  map_length_m: 3000.0
  map_width_m: 3000.0
  user_demand_rate_bps: 0.5e6
  orbit_radius_m: 700.0
  user_distribution: separated_hotspots
  spawn_margin: 0.03
  association_min_rate_bps: 0.5e6
  max_access_range_m: 800.0
  access_extra_loss_db: 6.0
  user_speed_model: maxwell_boltzmann
  user_speed_mean_mps: 1.5

observation:
  preset: compact_v2
  max_obs_users: 20
  obs_radius_m: 900.0

reward:
  outage_threshold_bps: 0.5e6
  target_coverage: 0.85
  coverage_gap_coef: 6.0
  target_fairness: 0.80
  fairness_gap_coef: 3.0
  relay_queue_coef: 0.5
```

### 9.2 `mappo_satellite_3uav_sar_dynamic_hotspots.yaml`

用途：讓 static baseline 不可能長期保持好結果。

新增概念：

```yaml
user_distribution: drifting_hotspots
num_hotspots: 4
hotspot_drift_speed_mps: 1.0
hotspot_spread_m: 180.0
```

這比單純放大地圖更強，因為即使 static 初始覆蓋不錯，user 群也會慢慢漂走。

### 9.3 `mappo_satellite_3uav_kmeans_baseline.yaml`

用途：建立非 RL 的 strong baseline。

不是訓練配置，而是 evaluation / baseline 配置：

```text
static orbit
random movement
weighted-kmeans reposition
MAPPO
```

如果 MAPPO 連 weighted-kmeans 都打不贏，那問題可能不在演算法名稱，而在 observation/reward/scenario 還沒設好。

---

## 10. 實作優先順序

### 第一階段：讓 coverage 合理掉下來

優先做：

1. `max_access_range_m` configurable。
2. `access_extra_loss_db` configurable。
3. `separated_hotspots` / `poisson_hotspots` user distribution。
4. 新增低速率 SAR YAML，不使用 14 Mbps。
5. evaluation 新增 `effective_coverage_ratio`，但保留既有 `coverage_ratio` 語義。
6. 先用 `compact_v2` 做 static / random / MAPPO characterization，不在這一階段改 observation。

這一階段目標：

```text
static coverage 約 0.65 ~ 0.90
trained policy 有機會提升 coverage / outage / backlog
user demand 仍是 0.5 Mbps 等級
若有提升，能確認主要來自 scenario difficulty，而不是 observation 變胖
```

### 第二階段：讓 MAPPO 看得到該看的資訊

優先做 `compact_v3`：

1. other UAV gateway-capable flag。
2. active gateway flag。
3. other UAV relay queue norm。
4. local user velocity。
5. user priority flag。

這一階段目標：

```text
policy 不再只是固定方向漂移
service UAV 能知道 gateway anchor 在哪
UAV movement 對 relay/backhaul 更有意義
```

### 第三階段：加入 dynamic SAR users

優先做：

1. Maxwell-Boltzmann user speed model。
2. drifting hotspot。
3. high-priority rescue users。

這一階段目標：

```text
static baseline 會隨時間變差
MAPPO 必須動態追蹤需求
coverage/fairness/backlog 有更清楚的時間變化
```

這一階段一定要加 fairness / reproducibility rule。若加入 stochastic shadowing、Maxwell-Boltzmann mobility 或 drifting hotspot，trained 與 static baseline 必須使用 common random numbers：

```text
同一個 episode_index：
  trained policy 和 static baseline 使用同一個 scenario seed。
  user initial positions 相同。
  user mobility random stream 相同。
  channel shadowing / blockage random stream 相同。
  hotspot drift trajectory 相同。
```

目前 evaluation 已經用 `run_config.seed + 100_000 + episode_index` 建 episode env，因此 trained/static 在 deterministic env 下是 paired comparison。加入 stochastic channel / drifting hotspot 後，這個規則要升級成明確的 scenario snapshot 或 common-random-number stream，否則 `trained_minus_static_reward` 會因為不同隨機場景而變得很吵。

推薦 migration：

```text
EvalConfig.common_random_numbers: bool = True
EvalConfig.freeze_episode_scenarios: bool = True
```

第一版可以不真的落地 snapshot file，但文件和測試要要求：

```text
trained episode k 和 static episode k 的 users / channel random stream 一致。
```

### 第四階段：補強 baselines

優先做：

1. weighted-kmeans reposition baseline。
2. random waypoint baseline。
3. no-move static baseline 保留。
4. possibly heuristic gateway-aware baseline。

這一階段目標：

```text
不是只證明 MAPPO 比不動好
而是證明 MAPPO 比合理 heuristic 更好或至少可競爭
```

---

## 11. 不建議做的事

### 11.1 不建議直接用移動距離 reward

不要寫：

```text
reward += movement_bonus
```

這會讓 UAV 為了飛而飛。研究上比較難說服人。

比較好的做法是：

```text
UAV 飛了 -> coverage / outage / backlog / fairness 改善 -> reward 變好
```

### 11.2 不建議只加大 `association_min_rate_bps`

這雖然最簡單，但會讓場景假設變形。除非研究目標明確是 high-throughput emergency video / AR / VR，不然不該用 14 Mbps 當主場景。

### 11.3 不建議只放大地圖但不限制 effective range

如果 channel model 仍偏樂觀、association 仍用 upper-bound proxy rate，單純放大地圖可能仍然不夠，或者會變成「遠處完全不可服務」的極端。比較穩的是：

```text
中等大地圖 + finite access range + low-rate demand + user mobility
```

---

## 12. 最後建議

下一版應該把研究主軸從：

```text
高 demand / 高 association threshold 造成壓力
```

改成：

```text
低速率搜救需求下，因為災區大、users 分散與移動、UAV 有有限 effective coverage，
所以需要 MARL 控制 UAV trajectory 來提升 effective coverage 與 end-to-end service。
```

這樣會更貼近 related work，也比較符合你說的搜救任務直覺。

如果要我選一個最小但最有效的下一步，我會做：

1. 新增 `max_access_range_m`。
2. 新增 `separated_hotspots`。
3. 新增 `mappo_satellite_3uav_sar_lowrate_coverage.yaml`。
4. 新增 `effective_coverage_ratio`，但不改舊 `coverage_ratio`。
5. 先用 `compact_v2` 跑 static baseline characterization，確認 static coverage / effective coverage 不再永遠接近 1。

確認 static baseline 真的有缺口後，再投入長時間 MAPPO training。若 `compact_v2` 在新 SAR scenario 裡仍學出固定方向漂移，再進第二階段加 `compact_v3` gateway flags。這樣比較不會把「scenario 變合理」和「observation 變豐富」混在一起，4090 也會比較開心。

等這個 scenario 用 `compact_v2` characterization 過關後，第二階段再建立 `compact_v3` 版本，專門測 gateway-aware observation 是否能改善固定方向漂移與 relay/backhaul 行為。
