## 语言版本
- [英文原版](README.md)
- [中文版本](README_zh.md)

# ARG-MAPPO: Spatiotemporal Attention-Driven Intrinsic Motivation
A MARL (Multi-Agent Reinforcement Learning) algorithm framework deeply customized with MAPPO as the backbone, integrated with AREL (Agent-Temporal Attention) core ideas, and a creative intrinsic motivation (curiosity) exploration mechanism based on **Global Surprise Allocation**.

## 🧠 Core Architecture
The information flow of the algorithm follows strict tensor dimensionality reduction and reconstruction logic, and the entire learning process is divided into four progressive stages:

### Stage 1: Curiosity Generation (🔍 Surprise Production)
- **Component**: GlobalRewardPredictor ($B\_Net$)
- **Underlying Logic**: Define "unknown" by predicting the true environmental reward $r$ based on the current state $s$ and joint action $u$. Local Surprise is defined when the network fails to predict environmental dynamics (large prediction error).
- **Calculation Goal**: Collect prediction errors across an entire trajectory and sum to get the total episode surprise $E_{total}$.

### Stage 2: Spatiotemporal Redistribution (🌌 Credit Assignment)
- **Component**: Time_Agent_Transformer ($ARG\_Net$)
- **Underlying Logic**: Allocate total surprise to specific timesteps and agents
- **Feature Extraction**: Project high-dimensional physical observations $E$ to semantic latent space via MLP
- **Key Processes**:
  - **Temporal Attention**: Add causality mask to find "turning points" in time series
  - **Agent Attention**: Calculate interaction intensity between agents at the same timestep to identify "key agents"
  - **Permutation-Invariant Collapse (Deep Sets)**: Collapse $Z$ matrix (fused historical and peer features) to get predicted intrinsic reward $\hat{r}_t$ for each timestep
- **Dual Constraint Optimization**:
  - Regression alignment: $\sum \hat{r}_t \approx E_{total}$ (total score must match)
  - Variance regularization: $\omega \cdot Var(\hat{r}_t)$ (avoid equal distribution, enforce score differentiation)

### Stage 3: Triple Defense System (🛡️ Underlying Protection)
Direct injection of curiosity into PPO is catastrophic. Three defense mechanisms are deployed before reward fusion ($r_{mixed} = r + \beta * e_predict$):

| Defense Mechanism | Implementation | Purpose |
|-------------------|----------------|---------|
| RewardBufferGate (Anti-"Noisy TV") | Maintain sliding queue of last 320 episodes' scores as baseline | Truncate curiosity reward (multiply by 0) if agents act randomly for surprise (environmental score below average). Only "valuable breakthrough" surprises are allowed. |
| Dynamic Batch Normalization (Anti-PPO Value Estimation Degradation) | Local $\mu=0, \sigma=1$ normalization on gated intrinsic rewards with hard extremum truncation | Prevent extremely high variance curiosity rewards from corrupting PPO Critic network and Advantage calculation |
| Nonlinear Gaussian Warm-up (Anti-Cold Start Disaster) | Curiosity weight $\beta$ attached to time-growth coefficient $1 - \exp(-(t/T)^2)$ | Suppress noise in early chaotic stage (random network weights), smoothly release exploration authority when network has basic evaluation capability |

### Stage 4: PPO Optimization (⚔️ Policy Evolution)
The smoothed, dense, and targeted fused reward $r_{mixed}$ is fed into the standard MAPPO backbone, completing policy evolution through GAE advantage calculation and PPO clipped update.

## 📂 Code Structure Mapping
| File/Class Name | Physical Meaning & Responsibility |
|-----------------|------------------------------------|
| `mappo.py` -> MAPPO | Core controller of the algorithm. Coordinates Actor-Critic optimization and ARG mechanism lifecycle. |
| `mappo.py` -> RewardBufferGate | Security inspector. Maintains score baseline and generates physical truncation masks. |
| `arg_net.py` -> Time_Agent_Transformer | Spatiotemporal deconstructor. Executes $Q \cdot K^T$ optimization and outputs high-quality credit assignment blueprint. |
| `b_net.py` -> GlobalRewardPredictor | Source of curiosity. Continuously tracked by Target network to generate prediction errors. |
| `runner.py` -> worker_process | Parallel sampling process. Breaks MARL exploration efficiency bottleneck through lock-free multiprocess sampling. |

## 🛠️ Dimension Flow Memo
When reading/modifying Time_Agent_Transformer, keep track of tensor dimension transformations (physical perspective):

| Dimension | Physical Meaning |
|-----------|------------------|
| `[B, N, T, D]` | God's-eye view of raw physical world observations (Batch, Num-agents, Timesteps, Dimension) |
| `[B*N, T, D]` | Isolated timeline. Each agent independently reviews its own past (Temporal Attention) |
| `[B*T, N, D]` | Cross-section view. All agents interact at the same timestep (Agent Attention) |
| `[B, T, 1]` | Collapsed scalarization. Eliminate individual differences to get team exploration bonus for each timestep |

## 📊 Monitoring & Experimentation (W&B Logging)
The code is deeply integrated with Weights & Biases (wandb). Focus on these key metrics to evaluate system health:

| Metric | Interpretation |
|--------|----------------|
| `ARG/gate_pass_rate` | Gate pass rate (typically high in early exploration, decreases after convergence) |
| `ARG/dynamic_beta` | Gaussian warm-up curve (should expand smoothly as expected) |
| `ARG/arg_loss_var` | Variance loss (if consistently 0, ARG network degenerates to equal distribution) |

### 总结
1. ARG-MAPPO 以 MAPPO 为基础，通过时空注意力机制实现全局惊奇度的智能分配，构建了鲁棒的多智能体内在动机探索框架
2. 三重防御装甲（奖励门控、动态标准化、高斯预热）解决了多智能体好奇心探索中的不稳定性问题
3. 核心监控指标可有效评估算法探索效率和稳定性，重点关注门控放行率、动态beta值和方差损失