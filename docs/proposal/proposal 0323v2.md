---
marp: true
theme: default
title: my proposal
author: 蕭有村
---
<style>
{
  font-size: 26px
}
</style>
**Project Title:** Integrated Congestion Control and Connectivity Stability Optimization for Special Evacuation Missions in Crowded Traffic Scenarios

**Sub-Project Code:** Sub-Project Y3

---

## 1. Introduction

In scenarios of natural disasters (such as earthquakes, tsunamis) or large-scale emergency evacuations, ground communication infrastructure often fails due to physical damage or traffic overload. While existing Unmanned Aerial Vehicle (UAV) assisted networks can provide temporary coverage, they are prone to becoming information islands in the absence of stable Backhaul.

Here, we propose a solution based on the **Space-Air-Ground Integrated Network (SAGIN)**, introducing an **Anchor-Member UAV Collaboration Architecture** and **Low Earth Orbit (LEO) Satellite Backhaul Support**. Addressing the complex coupling problem of resource allocation and trajectory planning, we adopt a **Hierarchical Approach**: using **Noisy-MAPPO (Noisy Multi-Agent PPO)** for trajectory design and **a Two-Timescale Resource Allocation** scheme combining **LinUCB-based Contextual Bandit** for adaptive fairness weight control with **Proportional Fair (PF) Scheduling** for fast sub-channel assignment. This utilizes Noisy Value Functions to solve the issue of insufficient strategy exploration capability in traditional MADRL within dynamic environments, aiming to maximize system connection throughput and stability.

---

## 2. System Model and Problem Formulation

### 2.1 Notations

**Table I: Channel Model Notations**

| Symbol | Definition |
| :--- | :--- |
| $\mathcal{I}, \mathcal{J}, \mathcal{K}$ | UAV set $\{1,\dots,I\}$, User set $\{1,\dots,J\}$, Satellite/GBS set |
| $\mathbf{q}_i(t), \rho_i(t), \psi_i(t)$ | 3D position, flight radius, and flight angle of UAV $i$ at time $t$ |
| $P_{tx}^{RF}, P_{tx}^{UAV}$ | Transmit Power: RF access link, UAV backhaul link ('tx' = Transmitter) |
| $P_{rx}$ | Received Power ('rx' = Receiver) |
| $G_{tx}, G_{rx}$ | Transmit Antenna Gain and Receive Antenna Gain |
---
| Symbol | Definition |
| :--- | :--- |
| $L_{ij}^{LoS}, L_{ij}^{NLoS}$ | LoS and NLoS Path Loss (dB) |
| $P_{ij}^{LoS}$ | Line-of-Sight (LoS) Probability |
| $\Gamma_{ij}(t)$ | Receiver Signal-to-Interference-plus-Noise Ratio (SINR) |
| $\vartheta_{ij}^{RF}(t)$ | Transmission Rate (RF Rate) between UAV $i$ and user $j$ |

---

| Symbol | Definition |
| :--- | :--- |
| $N$ | Number of orthogonal sub-channels |
| $W$ | Bandwidth of each sub-channel |
| $\mathcal{N}_{sub}$ | Set of sub-channels $\{1, \dots, N\}$ |
| $a_{ij}^n(t)$ | Binary Allocation Variable; 1 if user $j$ is assigned to sub-channel $n$ of UAV $i$ |
| $B_i$ | Total bandwidth of UAV $i$ ($B_i = N \times W$) |
| $\mathcal{L}, \lambda_2$ | Laplacian Matrix of the UAV communication network and its second smallest eigenvalue |
| $\sigma^2$ | Noise Power |
| $d_{safe}, V_{max}$ | UAV safety distance and maximum flight speed |
| $E_i^c(t)$ | Energy Consumption of UAV $i$ at time $t$ |

---

**Table II: Problem Formulation Notations**

| Symbol | Definition |
| :--- | :--- |
| $\varpi_{ij}(t)$ | Binary Variable; 1 if UAV $i$ serves user $j$, 0 otherwise |
| $M_{conn}$ | Number of connected users |
| $\delta$ | Minimum SINR threshold for sub-channel usability (filtering) |
| $R_{min}$ | Minimum required aggregate transmission rate for a user to avoid outage |
| $\mathbb{I}_{out, j}(t)$ | Instantaneous outage indicator for user $j$ |
| $\Phi(t)$ | System-wide risk metric (Outage Probability) |
| $\mu$ | Risk sensitivity coefficient (Penalty weight) |

---

**Table III: Two-Timescale Resource Allocation Notations**

| Symbol | Definition |
| :--- | :--- |
| $\alpha$ | Proportional Fair exponent (fairness weight); selected by LinUCB |
| $\mathcal{A}_{\alpha}$ | Discrete action set for fairness weights, e.g., $\{0, 0.5, 1.0, 2.0\}$ |
| $K$ | Number of discrete fairness weight levels ($|\mathcal{A}_{\alpha}|$) |
| $\mathbf{x}_{i,t} \in \mathbb{R}^d$ | Context feature vector observed by UAV $i$ at slow timescale step $t$ |
| $d$ | Dimension of the context feature vector |
---
| Symbol | Definition |
| :--- | :--- |
| $\mathbf{A}_a, \mathbf{b}_a$ | LinUCB design matrix ($d \times d$) and reward vector ($d \times 1$) for action $a$ |
| $\mathbf{M}_a$ | Inverse design matrix $\mathbf{A}_a^{-1}$, maintained via Sherman-Morrison updates |
| $c$ | LinUCB exploration coefficient (UCB width parameter) |
| $S_{j,k}(\tau)$ | PF priority score of user $j$ on sub-channel $k$ at TTI $\tau$ |
| $\bar{R}_j(\tau)$ | Exponential moving average throughput of user $j$ |
| $\beta_{pf}$ | PF forgetting factor for moving average update |
| $\epsilon$ | Numerical guard constant for PF initialization ($\epsilon \ll R_{min}$, e.g., $10^{-6}$ Mbps) |

---

### 2.2 Channel Models

#### 2.2.1 UAV-GU Access Link (Probabilistic LoS/NLoS Model)

The RF channel between UAV $i$ and ground user $j$ adopts a probabilistic Line-of-Sight model. Path Loss is divided into LoS and NLoS scenarios (in dB):

$$
L_{ij}^{LoS} = \alpha^{LoS} + 10 \beta^{LoS} \log_{10} d_{ij} + \mathcal{X}_{\sigma}
$$

$$
L_{ij}^{NLoS} = \alpha^{NLoS} + 10 \beta^{NLoS} \log_{10} d_{ij} + \mathcal{X}_{\sigma}
$$

Where $d_{ij}$ is the Euclidean distance, $\alpha$ is the reference distance loss, $\beta$ is the path loss exponent, and $\mathcal{X}_{\sigma}$ is shadow fading.

---

The probability of LoS occurrence $P_{ij}^{LoS}$ is determined by the elevation angle $\varphi_{ij} = \frac{180}{\pi} \sin^{-1}(\frac{h_i}{d_{ij}})$:

$$
P_{ij}^{LoS}(\varphi_{ij}) = \frac{1}{1 + a \cdot \exp(-b(\varphi_{ij} - a))}
$$

Where $a, b$ are environmental parameters (e.g., Urban, Suburban). The average path loss is:

$$
L_{ij}^{avg} = P_{ij}^{LoS} L_{ij}^{LoS} + (1 - P_{ij}^{LoS}) L_{ij}^{NLoS}
$$

---

The SINR $\Gamma_{ij}^n(t)$ for user $j$ connected to UAV $i$ on sub-channel $n$ is:

$$
\Gamma_{ij}^n(t) = \frac{P_{tx}^{RF} \cdot 10^{-L_{ij}^{avg}/10}}{\sigma^2 + \sum_{k \in \mathcal{I}, k \neq i} \sum_{j' \in \mathcal{J}} a_{kj'}^n(t) \cdot P_{tx}^{RF} \cdot 10^{-L_{kj}^{avg}/10}}
$$

Where the second term in the denominator represents Mean Co-channel Interference from other UAVs using the same sub-channel $n$.

The total transmission rate provided by UAV $i$ to user $j$ is the sum of rates over assigned sub-channels:

$$
\vartheta_{ij}^{RF}(t) = \sum_{n=1}^{N} a_{ij}^n(t) \cdot W \cdot \log_2(1 + \Gamma_{ij}^n(t))
$$

Where $W$ is the bandwidth per sub-channel.

---

#### 2.2.2 UAV-UAV Relay Link (Free Space Model)

To achieve Multi-hop Relay, Member UAVs must maintain connectivity. The channel gain $g_{mn}$ between UAV $m$ and UAV $n$ adopts the free space model:

$$
g_{mn}(t) = \frac{\rho_0}{\|\mathbf{q}_m(t) - \mathbf{q}_n(t)\|^2}
$$

Where $\rho_0$ is the channel gain at a reference distance (1m). The Signal-to-Noise Ratio (SNR) $\gamma_{mn}$ between two UAVs is:

$$
\gamma_{mn}(t) = \frac{P_{tx}^{UAV} \cdot g_{mn}(t)}{\sigma^2}
$$

If $\gamma_{mn}(t) \ge \gamma_{th}$ (communication threshold), an effective link (Edge) is considered to exist between them.

---

#### 2.2.3 UAV-Satellite Backhaul Link (Satellite RF Link)

The Anchor UAV ($u_{anchor}$) is responsible for transmitting aggregated data back to the Low Earth Orbit (LEO) satellite. The transmission model is based on the Friis transmission equation:

$$
P_{rx}^{Sat}(t) = P_{tx}^{UAV} + G_{tx} + G_{rx} - L_{FSPL}(d_{u,s}) - L_{atm}
$$

The backhaul capacity $\vartheta_{anchor}^{Backhaul}$ is:

$$
\vartheta_{anchor}^{Backhaul}(t) = B_{sat} \log_2 \left( 1 + \frac{10^{P_{rx}^{Sat}(t)/10}}{\sigma^2} \right)
$$

(Note: $P_{rx}^{Sat}(t)$ in the above equations is in dB/dBm, converted to linear scale for capacity calculation).

---

#### 2.2.4 Energy Consumption Model

UAV energy consumption mainly consists of **Hovering** and **Flying** components. The energy consumption $E_i^c(t)$ of UAV $i$ at time $t$ is:

$$
E_i^c(t) = \tau_i^{hover}(t) \cdot e^h + l_i^{move}(t) \cdot e^f
$$

Where:
*   $\tau_i^{hover}(t)$: Hovering time (sec).
*   $l_i^{move}(t)$: Flight distance (meter).
*   $e^h$: Hovering energy consumption per unit time (J/s).
*   $e^f$: Flight energy consumption per unit distance (J/m).

---

### 2.3 Problem Formulation

**Risk Metric Definition (Outage Probability)**

To ensure reliability in evacuation missions, we define the outage indicator $\mathbb{I}_{out, j}(t)$ based on the total achieved transmission rate. A user is considered in outage if their aggregated rate falls below the minimum requirement $R_{min}$:

$$
\mathbb{I}_{out, j}(t) = \begin{cases} 
0, & \text{if } \sum_{i \in \mathcal{I}} \vartheta_{ij}^{RF}(t) \ge R_{min} \\
1, & \text{otherwise (Disconnection or insufficient rate)}
\end{cases}
$$

$$
\Phi(t) = \frac{1}{J} \sum_{j \in \mathcal{J}} \mathbb{I}_{out, j}(t)
$$

---

The goal is to maximize the total system throughput while minimizing both energy consumption and outage risk. We formulate the problem as a joint optimization using a weighted sum method:

**Objective Function:**


$$
\max_{\{\mathbf{q}_i(t)\}, \{a_{ij}^n(t)\}} \sum_{t=1}^{T} \left[ \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} \vartheta_{ij}^{RF}(t) - \eta \sum_{i \in \mathcal{I}} E_i^c(t) - \mu \Phi(t) \right]
$$

Where:
* $\eta$: Energy efficiency weight (penalty coefficient for energy consumption).
* $\mu$: Risk sensitivity coefficient (penalty coefficient for outage probability).

---

**Constraints:**

1.  **User Association & Sub-channel Allocation Constraints:** To maintain synchronization and avoid complex joint transmissions, each user associates with at most **one** UAV. We use $\varpi_{ij}(t) \in \{0, 1\}$ to denote association:
    $$
    \sum_{i \in \mathcal{I}} \varpi_{ij}(t) \le 1, \quad \forall j \in \mathcal{J}
    $$
    Sub-channel allocation is governed by the PF scheduler. To maintain OFDMA orthogonality, each sub-channel on a specific UAV serves at most one user per TTI:
    $$
    a_{ij}^n(t) \le \varpi_{ij}(t), \quad \forall i, j, n
    $$
    $$
    \sum_{j \in \mathcal{J}} a_{ij}^n(t) \le 1, \quad \forall i \in \mathcal{I}, \forall n \in \mathcal{N}_{sub}
    $$
---
2.  **Rate Quality Constraint:** An associated user must achieve a minimum aggregate rate. Users that cannot be served at $R_{min}$ by any UAV remain unassociated and are counted as outage in $\Phi(t)$.
    $$
    \vartheta_{ij}^{RF}(t) \ge R_{min}, \quad \text{if } \varpi_{ij}(t) = 1
    $$
    Additionally, only sub-channels satisfying a minimum SINR $\delta$ are considered usable during allocation:
    $$
    \Gamma_{ij}^n(t) \ge \delta, \quad \text{for } a_{ij}^n(t) = 1
    $$

---

3.  **Backhaul Capacity Constraint:** The total system traffic must not exceed the Anchor UAV's backhaul capacity.
    $$
    \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} \vartheta_{ij}^{RF}(t) \le \vartheta_{anchor}^{Backhaul}(t)
    $$
4.  **UAV Mobility Constraint:**
    $$
    \|\mathbf{q}_i(t) - \mathbf{q}_i(t-1)\| \le V_{max} \cdot \Delta t
    $$
    $$
    x_{min} \le x_i(t) \le x_{max}, \quad y_{min} \le y_i(t) \le y_{max}
    $$

---

5.  **Network Connectivity Constraint (Algebraic Connectivity):** To ensure Member UAVs can relay to the Anchor UAV, the entire UAV network must remain connected.
    $$
    \lambda_2(\mathcal{L}(t)) > 0
    $$
6.  **Safety Distance Constraint:**
    $$
    \|\mathbf{q}_m(t) - \mathbf{q}_n(t)\| \ge d_{safe}, \quad \forall m \neq n
    $$
7.  **Energy Constraint:** UAVs must retain sufficient energy to maintain operation or return.
    $$
    E_i(T) \ge E_{min}, \quad \forall i \in \mathcal{I}
    $$

---

## 3. Proposed Solution

### 3.1 State Initialization via Enhanced K-means

Referencing the KMAPPO method in the *Cooperative UAV Trajectory Design* paper, we introduce **Enhanced K-means**.

*   **Enhanced K-means Improvements:**
    *   **Observation Limitation:** Introduce an observation radius $d_{obs}$. If the distance between user $j$ and UAV $i$ is $d_{ij} > d_{obs}$, it is considered invisible ($d_{ij} = \infty$).
    *   **Local Update:** UAVs only use user positions within the "observable range" to update the Cluster Centroid they are responsible for.
*   **Advantage:** Allows the algorithm to start training from a better initial state, significantly improving convergence speed.

---

K-means is placed within the Episode Loop, but outside the Step Loop.


Process:

*   **New Episode Start**

    *   Acquire user location information.
    *   Execute Enhanced K-means $\rightarrow$ Set UAV initial positions $(x_0, y_0)$.

*   **Start Time Steps ($t=1, 2, ..., T$)**

    *   UAVs begin moving according to the PPO policy network.

---

### 3.2 Markov Decision Process (MDP) Definition

**State Space and Observation Space**

*   **Global State ($s_t$):** Used for centralized Critic training.
    $$
    s_t = \{ \underbrace{\mathbf{q}_{anchor}(t), \{\mathbf{q}_m(t)\}_{m \in Members}}_{\text{UAVs States}}, \underbrace{\{\mathbf{q}_j(t)\}_{\forall j \in \mathcal{J}}}_{\text{All MUs States}},\}
    $$

---

*   **Local Observation ($o_t^i$):** Used for decentralized Actor execution. Agent's observation includes its own state and the states of all other UAVs (shared via the network), but only includes locally sensed user states.
    $$
    o_t^i = \{ \underbrace{\mathbf{q}_i(t), \mathbf{v}_i(t), E_i(t)}_{\text{Self Info}}, \underbrace{\{\mathbf{q}_k(t) - \mathbf{q}_i(t)\}_{\forall k \in \mathcal{I}, k \neq i}}_{\text{Other UAVs Relative Pos}}, \underbrace{\text{ID}_{anchor}}_{\text{Anchor ID}}, \underbrace{\{\mathbf{q}_j(t) - \mathbf{q}_i(t)\}_{j \in \mathcal{N}_i(t)}}_{\text{Visible MUs (Relative)}} \}
    $$
    *   $\{\mathbf{q}_k(t) - \mathbf{q}_i(t)\}$: Relative positions of all other UAVs. Since constraints require $\lambda_2 > 0$ (network connectivity), UAVs can exchange position information for formation control and obstacle avoidance.
    *   $\text{ID}_{anchor}$: Indicates which UAV is the Anchor, so Member UAVs can plan relay paths.

---

**Action Space**

$$
a_t^i = \{ \rho_i(t), \psi_i(t) \}
$$

*   $\rho_i(t) \in [0, 1]$: **Flight Radius**.
    *   Represents the flight distance of the UAV within that time step, typically defined as a proportion of the maximum flight distance $R_{max}$ (i.e., actual movement distance is $\rho_i(t) \times R_{max}$).

*   $\psi_i(t) \in [-180^{\circ}, 180^{\circ}]$: **Flight Angle**.
    *   Represents the flight direction of the UAV on the horizontal plane.

---

**Reward Function**

To achieve risk-sensitive and energy-aware control, the reward function is designed as a weighted combination of throughput gain, energy cost, and risk penalty:

$$
R(t) = \underbrace{ \left( \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} \vartheta_{ij}^{RF}(t) \right) }_{\text{Throughput Gain}} - \underbrace{ \eta \sum_{i \in \mathcal{I}} E_i^c(t) }_{\text{Energy Penalty}} - \underbrace{ \mu \Phi(t) }_{\text{Risk Penalty}}
$$

* **Throughput Gain:** Encourages high-rate transmission and extensive user coverage.
* **Energy Penalty:** Penalizes excessive maneuvering and hovering to prolong network lifetime.
* **Risk Penalty:** Penalizes the system when users experience connection outage, ensuring stability.

> **Note:** The penalty coefficients $\eta$ and $\mu$ incorporate unit-conversion factors, normalising $R(t)$ to a dimensionless, $O(1)$ quantity so that throughput (bps), energy (J), and outage probability (dimensionless) contribute at comparable scales during policy gradient updates.

---

### 3.3 Core Algorithm: Noisy-MAPPO

This project adopts **Noisy-Value MAPPO (NV-MAPPO)**, which extends the standard MAPPO framework with a noisy value function.

**Motivation: The Policies Overfitting in Multi-agent Cooperation (POMAC) Problem**

| | |
| :--- | :--- |
| **Cause** | All agents share a centralized advantage $A(s_t,\vec{a}_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$; with limited samples, this estimate is **biased**. |
| **Consequence** | A reward driven by agent $j$'s action incorrectly updates agent $i$'s policy → suboptimal gradient directions → reduced exploration. |

---

**Solution: Noisy Value Function**

To mitigate POMAC, we inject per-agent Gaussian noise into the value function input, producing agent-specific (differentiated) advantage estimates:

1.  **Noise Sampling:** Sample a random noise vector for each agent $i$:
    $$\vec{x}^i \sim \mathcal{N}(0, \sigma^2), \quad \forall i \in \mathcal{I}$$

2.  **Noisy Value Computation:** Concatenate the noise with the global state and feed into the shared centralized critic:
    $$v_b^i(\phi) = V_\phi(\text{concat}(s_b, \vec{x}^i)), \quad \forall i \in \mathcal{I}$$

3.  **Agent-specific Advantages:** Compute GAE advantages $\hat{A}_b^i$ and returns $\hat{R}_b^i$ using $v_b^i$ individually for each agent.

---

**Key Benefits:**

*   **Prevents overfitting:** The noise smooths the shared advantage values, preventing multi-agent policies from overfitting to biased sampled advantages.
*   **Encourages exploration:** Different noise vectors $\vec{x}^i$ per agent drive policies in different directions, encouraging diverse trajectory exploration.
*   **Implicit policy ensemble:** The $N$ policies trained by $N$ noisy value networks are analogous to a policy ensemble, improving robustness.
*   **Noise schedule:** The noise vectors can remain fixed or be periodically shuffled across agents (e.g., every fixed number of episodes), similar to periodic target network updates in DQN.

> The complete training loop integrating Noisy-MAPPO with Two-Timescale Resource Allocation (LinUCB + PF) is presented in **Algorithm 3** (Section 3.5).

---

### 3.4 Two-Timescale Resource Allocation (LinUCB + Proportional Fair)

While the RL agent handles **Trajectory Planning (Outer Loop)**, resource allocation forms a critical **Inner Loop**. Traditional heuristic approaches (e.g., swap matching) suffer from high computational complexity that violates MAC-layer latency requirements. We adopt a **Two-Timescale** architecture aligned with the O-RAN specification:

| Timescale | Component | Role | Cycle |
| :--- | :--- | :--- | :--- |
| **Slow** | LinUCB (Near-RT RIC) | Adaptive Fairness Weight Selection | ~100 ms – 1 s |
| **Fast** | PF Scheduler (O-DU MAC) | Per-TTI Sub-channel Assignment | 1 ms (per TTI) |

---

#### 3.4.1 User Association

Each user $j$ associates with the UAV providing the strongest received power (Max-Power Association):

$$
\varpi_{ij}(t) = 1 \iff i = \arg\max_{i' \in \mathcal{I}} P_{tx}^{RF} \cdot 10^{-L_{i'j}^{avg}/10}
$$

Load balancing is implicitly handled by the PF scheduler: when a UAV becomes congested, the LinUCB agent increases $\alpha$ to protect disadvantaged users, thereby redistributing throughput without explicit swap operations.

---

#### 3.4.2 Slow Timescale: LinUCB Fairness Weight Controller

At each slow timescale decision epoch $t$, each UAV $i$ independently runs a LinUCB agent to select the optimal PF fairness exponent $\alpha$ from a discrete action set.

**Context Vector.** The context $\mathbf{x}_{i,t} \in \mathbb{R}^d$ captures the local radio environment:

$$
\mathbf{x}_{i,t} = \left[ 1, \; \frac{|\mathcal{U}_i(t)|}{J}, \; \frac{|\{j \in \mathcal{U}_i : \hat{\Gamma}_{ij}(t) < \delta \}|}{|\mathcal{U}_i(t)|}, \; \sum_{m \in \mathcal{I}, m \neq i} \frac{1}{\|\mathbf{q}_i(t) - \mathbf{q}_m(t)\|^2}, \; \frac{\|\mathbf{v}_i(t)\|}{V_{max}} \right]^\top
$$
---
where the components are:
*   Bias term (constant 1).
*   Normalized user load $|\mathcal{U}_i(t)| / J$.
*   Edge user ratio: fraction of associated users with reference SINR $\hat{\Gamma}_{ij}(t) = P_{tx}^{RF} \cdot 10^{-L_{ij}^{avg}/10} / \sigma^2$ below threshold $\delta$.
*   Spatial interference proxy: sum of inverse squared distances to neighboring UAVs, reflecting that received interference power decays as $d^{-2}$ under free-space path loss.
*   Normalized speed $\|\mathbf{v}_i(t)\| / V_{max}$.

> **Remark (Feature Scaling):** Before input to LinUCB, each element of $\mathbf{x}_{i,t}$ is normalised to $[0, 1]$. This is essential because LinUCB's regularisation term $\mathbf{I}_d$ applies equal penalisation across all dimensions; without scaling, the interference proxy (which may span several orders of magnitude) would dominate gradient updates and prevent convergence.

---

**Action Space.** The discrete fairness weight set is $\mathcal{A}_{\alpha} = \{\alpha_1, \alpha_2, \dots, \alpha_K\}$, e.g., $K=4$ with $\alpha \in \{0, 0.5, 1.0, 2.0\}$.

*   $\alpha = 0$: Max-Rate scheduling (maximizes throughput).
*   $\alpha = 1$: Classical Proportional Fairness.
*   $\alpha \to \infty$: Max-Min Fairness (strongest protection for disadvantaged users).

---

**LinUCB Decision Rule.** For each action $a \in \mathcal{A}_{\alpha}$, the agent maintains an inverse design matrix $\mathbf{M}_a \in \mathbb{R}^{d \times d}$ and a reward-weighted context vector $\mathbf{b}_a \in \mathbb{R}^d$.

*   **Initialization:** $\mathbf{M}_a \leftarrow \mathbf{I}_d, \quad \mathbf{b}_a \leftarrow \mathbf{0}_d, \quad \forall a \in \mathcal{A}_{\alpha}$

*   **Action Selection (UCB Policy):** Compute the estimated reward parameter and upper confidence bound for each action:
    $$
    \hat{\boldsymbol{\theta}}_a = \mathbf{M}_a \mathbf{b}_a
    $$
    $$
    p_{t,a} = \mathbf{x}_{i,t}^\top \hat{\boldsymbol{\theta}}_a + c \sqrt{\mathbf{x}_{i,t}^\top \mathbf{M}_a \, \mathbf{x}_{i,t}}
    $$
    where $c > 0$ controls the exploration–exploitation trade-off. The selected action is $a_t^* = \arg\max_{a \in \mathcal{A}_{\alpha}} p_{t,a}$, and the corresponding $\alpha_{(a_t^*)}$ is dispatched to the PF scheduler.

---

**Reward Signal.** After one slow-timescale observation period, the agent receives a local reward reflecting throughput efficiency and outage penalty:

$$
r_{i,t} = \frac{1}{|\mathcal{U}_i|} \sum_{j \in \mathcal{U}_i} \frac{\vartheta_{ij}^{RF}(t)}{R_{ref}} - \mu \cdot \Phi_i(t)
$$

where $R_{ref}$ is a reference rate for normalization and $\Phi_i(t) = \frac{1}{|\mathcal{U}_i|} \sum_{j \in \mathcal{U}_i} \mathbb{I}_{out,j}(t)$ is the local outage ratio of UAV $i$.

> **Note:** The weight $\mu$ and the reference rate $R_{ref}$ are chosen such that both terms in $r_{i,t}$ are $O(1)$, bounding the reward in $[-1, 1]$ and keeping the UCB exploration coefficient $c$ interpretable and tunable.

---

**Parameter Update (Sherman-Morrison).** To avoid $O(d^3)$ matrix inversion at each step, we maintain $\mathbf{M}_a = \mathbf{A}_a^{-1}$ directly via the Sherman-Morrison rank-one update, reducing complexity to $O(d^2)$:

$$
\mathbf{b}_{a_t^*} \leftarrow \mathbf{b}_{a_t^*} + r_{i,t} \, \mathbf{x}_{i,t}
$$

$$
\mathbf{M}_{a_t^*} \leftarrow \mathbf{M}_{a_t^*} - \frac{\mathbf{M}_{a_t^*} \, \mathbf{x}_{i,t} \, \mathbf{x}_{i,t}^\top \, \mathbf{M}_{a_t^*}}{1 + \mathbf{x}_{i,t}^\top \, \mathbf{M}_{a_t^*} \, \mathbf{x}_{i,t}}
$$

> **Correctness Note:** This follows from the Sherman-Morrison identity $(\mathbf{A} + \mathbf{u}\mathbf{v}^\top)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1} \mathbf{u} \mathbf{v}^\top \mathbf{A}^{-1}}{1 + \mathbf{v}^\top \mathbf{A}^{-1} \mathbf{u}}$, where $\mathbf{u} = \mathbf{v} = \mathbf{x}_{i,t}$.

---

#### 3.4.3 Fast Timescale: Proportional Fair Scheduler

At each TTI $\tau$ (1 ms granularity), each UAV $i$ independently allocates its $N$ orthogonal sub-channels to associated users $\mathcal{U}_i$ using the PF scheduler with the current $\alpha_{(a_t^*)}$ from LinUCB.

**Algorithm 2: PF Sub-channel Scheduling (per UAV $i$, per TTI $\tau$)**

| Step | Description |
| :--- | :--- |
| **Input** | User set $\mathcal{U}_i$, sub-channels $\mathcal{N}_{sub}$, instantaneous SINR $\Gamma_{ij}^n(\tau)$, threshold $\delta$, current $\alpha_{(a_t^*)}$, history $\bar{R}_j(\tau)$ |
| **Output** | Allocation $a_{ij}^n(\tau)$, achieved rates $\vartheta_{ij}^{RF}(\tau)$ |

---

| Step | Description |
| :--- | :--- |
| **1. SINR Filter** | $\forall j, n$: mark $(j, n)$ as candidate only if $\Gamma_{ij}^n(\tau) \ge \delta$ |
| **2. Priority Score** | $\forall j \in \mathcal{U}_i, \forall n \in \mathcal{N}_{sub}$: compute $S_{j,n}(\tau) = \dfrac{W \log_2(1 + \Gamma_{ij}^n(\tau))}{[\bar{R}_j(\tau) + \epsilon]^{\alpha_{(a_t^*)}}}$ |
| **3. Greedy Alloc** | $\forall n \in \mathcal{N}_{sub}$: assign $a_{ij^*}^n(\tau) = 1$ where $j^* = \arg\max_{j \in \mathcal{U}_i} S_{j,n}(\tau)$ |
| **4. Rate Agg** | $\vartheta_{ij}^{RF}(\tau) = \sum_{n} a_{ij}^n(\tau) \cdot W \log_2(1 + \Gamma_{ij}^n(\tau))$ |
| **5. EMA Update** | $\bar{R}_j(\tau\!+\!1) = (1\!-\!\beta_{pf})\bar{R}_j(\tau) + \beta_{pf} \sum_{n \in \mathcal{K}_j(\tau)} W \log_2(1 + \Gamma_{ij}^n(\tau))$ |

where $\beta_{pf} \in (0, 1)$ is the forgetting factor (typically $\approx 0.01$), and $\mathcal{K}_j(\tau)$ is the set of sub-channels allocated to user $j$ at TTI $\tau$.

> **Complexity:** The PF scheduler runs in $O(N |\mathcal{U}_i|)$ per TTI — a pure arithmetic comparison — fully satisfying the 5G MAC-layer 1 ms scheduling deadline.

---

### 3.5 Overall Integrated Algorithm

The following summarizes the complete method, integrating Enhanced K-means initialization, Noisy-MAPPO trajectory planning, and Two-Timescale Resource Allocation (LinUCB + PF).

**Algorithm 3: Integrated Noisy-MAPPO with Two-Timescale Resource Allocation**

| Step | Description |
| :--- | :--- |
| **Input** | Actor params $\theta$, Critic params $\phi$, noise variance $\sigma^2$, max episodes $E$, horizon $T$, LinUCB exploration $c$, fairness set $\mathcal{A}_{\alpha}$ |
| **Output** | Trained policy $\pi_\theta$, optimized trajectories and allocations |
| **0. Init** | Initialize $\theta, \phi$; sample noise $\vec{x}^i \sim \mathcal{N}(0, \sigma^2)$ for each agent $i$; $\forall i, \forall a$: $\mathbf{M}_a^i \leftarrow \mathbf{I}_d$, $\mathbf{b}_a^i \leftarrow \mathbf{0}_d$ |
---
| Step | Description |
| :--- | :--- |
| **1. Episode Loop** | **for** episode $= 1, \dots, E$: |
| | &emsp; Acquire user locations; run **Enhanced K-means** $\rightarrow$ set UAV initial positions |
| **2. Step Loop** | &emsp; **for** $t = 1, \dots, T$: |
| | &emsp;&emsp; Each agent observes $o_t^i$ and selects action $a_t^i = \pi_\theta^i(o_t^i)$ |
| | &emsp;&emsp; Execute actions $\rightarrow$ obtain new UAV positions $\mathbf{q}_i(t)$ |

---

| Step | Description |
| :--- | :--- |
| **3a. Association** | &emsp;&emsp; Max-Power Association: $\varpi_{ij}(t) = 1$ iff $i = \arg\max_{i'} P_{tx}^{RF} \cdot 10^{-L_{i'j}^{avg}/10}$ |
| **3b. LinUCB** | &emsp;&emsp; **[Slow Timescale]** Each UAV $i$ constructs $\mathbf{x}_{i,t}$; selects $a_t^* = \arg\max_a p_{t,a}$ $\rightarrow$ dispatches $\alpha_{(a_t^*)}$ |
| **3c. PF Sched** | &emsp;&emsp; **[Fast Timescale]** Run **PF Scheduler** (Alg. 2) for each TTI within step $t$ $\rightarrow$ $a_{ij}^n(\tau)$ |
| **3d. LinUCB Update** | &emsp;&emsp; Observe reward $r_{i,t}$; update $\mathbf{b}_{a_t^*}^i$ and $\mathbf{M}_{a_t^*}^i$ via Sherman-Morrison |
| **4. Reward** | &emsp;&emsp; Compute rates $\vartheta_{ij}^{RF}(t)$, outage $\Phi(t)$, energy $E_i^c(t)$ |
| | &emsp;&emsp; $R(t) = \sum_{i,j} \vartheta_{ij}^{RF}(t) - \eta \sum_i E_i^c(t) - \mu \Phi(t)$ |
| | &emsp;&emsp; Store $(s_t, \vec{o}_t, \vec{a}_t, R(t), s_{t+1})$ in buffer |
---
| Step | Description |
| :--- | :--- |
| **5. Policy Update** | &emsp; After episode ends: |
| | &emsp;&emsp; (Optional) Periodically shuffle noise vectors $\vec{x}^i$ across agents |
| | &emsp;&emsp; Sample batch $B$ from buffer |
| | &emsp;&emsp; Noisy value forward: $v_b^i(\phi) = V_\phi(\text{concat}(s_b, \vec{x}^i)), \; \forall i$ |
| | &emsp;&emsp; Compute agent-specific GAE advantages $\hat{A}_b^i$ and returns $\hat{R}_b^i$ |
| | &emsp;&emsp; Update Critic $\phi$: $L(\phi) = \frac{1}{B \cdot I} \sum_{b,i} (v_b^i(\phi) - \hat{R}_b^i)^2$ |
| | &emsp;&emsp; Update Actor $\theta$: $L(\theta) = -\frac{1}{B \cdot I} \sum_{b,i} [\min(r_b^i \hat{A}_b^i, \text{clip}(r_b^i, 1\!-\!\epsilon, 1\!+\!\epsilon)\hat{A}_b^i) - \eta \mathcal{H}(\pi^i)]$ |
| **6. Return** | Trained $\pi_\theta$ for deployment |

---

## 4. Experimental Design

### 4.1 Simulation Environment

*   **Area:** $2 \text{ km} \times 2 \text{ km}$ urban disaster area.
*   **Users:** $J=50 \sim 80$, movement model adopts Maxwell-Boltzmann Distribution (simulating panicked crowds).
*   **UAVs:** $I=3 \sim 12$.
*   **Channel Parameters:** $\alpha^{LoS}=1.0, \alpha^{NLoS}=20, \beta^{LoS}=2.09, \beta^{NLoS}=3.75$ (Urban).

---

### 4.2 Baselines

*   **KMAPPO:** MAPPO using K-means initialization but without the Noisy mechanism.
*   **Greedy-Noisy-KMAPPO:** Proposed Noise MAPPO using K-means initialization with Greedy Allocation.
*   **MADDPG:** Traditional Off-policy method.
*   **PPO (Independent):** Independently trained, no collaboration.
*   **PSO (Particle Swarm Optimization):** Traditional heuristic algorithm as a non-RL Baseline.

### 4.3 Evaluation Metrics

*   **System Throughput:** Total system throughput.
*   **User Coverage Rate:** Proportion of connected users ($M_{conn}/J$).
*   **Convergence Speed:** Number of Episodes required for convergence.
*   **Energy Efficiency:** Transmitted bits per unit energy (Total Throughput / Total Energy).