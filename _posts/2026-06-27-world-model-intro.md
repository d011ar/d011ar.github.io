---
layout: post
title: "World Model 学习笔记：入门世界模型的3篇工作"
date: 2026-06-27 15:00:00 +0800
categories: Technical Study Notes
tags: [World Model, Reinforcement Learning, Embodied Intelligence, Video Generation, Agent]
toc: true
---

从 World Models 到 PlaNet 再到 DreamerV3 这三篇文章的经典组合，是快速建立对世界模型（World Model）整体认知框架的阅读笔记。

## 摘要

目前世界模型领域已经分裂成很多流派：

- 强化学习 World Model
- 视频生成 World Model
- 具身智能 World Model
- 自动驾驶 World Model
- Foundation World Model

这篇文章通过三篇入门论文快速建立对于世界模型（World Model）的整体认知框架，再决定往视频生成、具身智能、自动驾驶还是Agent方向深入。

---

# 🌍 World Models 学习笔记

## 📚 目录

- [一、World Models (2018)](#一world-models-2018)
- [二、PlaNet (2019)](#二planet-2019)
- [三、DreamerV3 (2023)](#三dreamerv3-2023)
- [四、参考资料](#四参考资料)

---

## 一、🌍 World Models (2018)

> **World Models** 是 David Ha 与 Jürgen Schmidhuber 于 2018 年发表的一篇人工智能研究论文。  
> 这是世界模型领域真正意义上的开山之作。它提出了一种能够 **学习环境内部表示，并在“想象世界”中训练 Agent** 的方法，对后续模型驱动强化学习和现代世界模型研究产生了深远影响。

---

### 1. 💡 核心思想

论文的核心观点是：

> **Agent 不必总是在真实环境中学习，而是可以先学习一个能够预测环境变化的生成式模型，再利用这个模型进行决策训练。**

作者将系统划分为三个主要模块：

- **V（Vision）**：利用变分自编码器 VAE 将高维图像压缩为低维潜在表示。
- **M（Memory）**：利用 MDN-RNN 学习潜在空间中的时间动态，预测未来状态。
- **C（Controller）**：一个非常小的控制器，根据 V 和 M 提供的信息选择动作。

---

### 2. 🧩 技术贡献

这篇论文最具影响力的贡献之一，是证明了：

> **控制策略可以完全在模型自己生成的“梦境环境”（dream environment）中训练。**

作者在 CarRacing 等强化学习环境中展示了完整流程：

1. 收集真实环境中的交互数据；
2. 学习一个生成式环境模型；
3. 在模拟环境中训练控制器；
4. 将训练好的策略迁移回真实环境；
5. 最后获得良好的表现。

这种思想显著降低了真实环境交互成本，也推动了后来大量 **Model-Based Reinforcement Learning** 工作的发展。

---

### 3. ⚙️ 研究方法

这篇论文第一次较明确地提出：

> **Environment Model = World Model**

整体架构可以理解为：

```text
Image Observation
        ↓
      VAE
        ↓
 Latent State z_t
        ↓
   MDN-RNN / LSTM
        ↓
 Future Latent State
        ↓
   Controller
        ↓
      Action
```

---

#### 3.1 VAE：视觉压缩模块

VAE 学习从原始图像到潜在表示的映射：

$$
x_t \rightarrow z_t
$$

其中：

- $x_t$ 表示时刻 $t$ 的原始图像观测；
- $z_t$ 表示压缩后的 latent state。

也就是说，VAE 将高维图像压缩成低维 latent 表示。

---

#### 3.2 MDN-RNN：动态预测模块

MDN-RNN 学习潜在状态随时间变化的规律：

$$
p(z_{t+1} \mid z_t, a_t)
$$

其中：

- $z_t$ 表示当前时刻的 latent state；
- $a_t$ 表示当前时刻采取的动作；
- $z_{t+1}$ 表示下一时刻的 latent state。

也就是说，MDN-RNN 根据当前 latent state 和动作，预测未来 latent state 的概率分布。

---

#### 3.3 Controller：决策模块

Controller 根据 VAE 和 MDN-RNN 提供的信息，在预测出来的世界里进行决策。

可以简单写成：

$$
a_t = C(z_t, h_t)
$$

其中：

- $C$ 表示控制器；
- $z_t$ 表示当前视觉 latent state；
- $h_t$ 表示 RNN 的 hidden state；
- $a_t$ 表示 Agent 选择的动作。

---

### 4. 🚀 影响与后续发展

虽然今天的“世界模型”通常规模远大于 2018 年的实现，但许多现代研究仍然沿用了这篇论文的重要理念：

- 学习潜在状态表示，而不是直接预测像素；
- 在内部模型中进行规划、推理和训练；
- 利用预测未来来提高样本效率；
- 将无监督表示学习、序列建模和强化学习结合起来。

《World Models》不仅提出了一种具体架构，更重新激发了研究社区对 **learning a model of the world** 这一方向的兴趣。

它展示了一种重要思路：

> **先理解世界，再学习行动。**

---

### 📝 Notes

看完这篇论文，你会理解：

- 为什么要学习世界模型；
- latent state 是什么；
- dynamics model 是什么；
- imagination rollout 是什么；
- 为什么“在想象中训练 Agent”是可行的。

---

## 二、🪐 PlaNet (2019)

> **Learning Latent Dynamics for Planning from Pixels** 是 Danijar Hafner 等作者于 2019 年发表的强化学习研究论文。  
> 该论文提出了名为 **PlaNet（Deep Planning Network）** 的模型驱动强化学习方法。它展示了如何 **仅凭像素图像学习环境动态，并在潜在空间中进行高效规划**，是世界模型研究的重要里程碑，也是现代世界模型的重要起点之一。

---

### 1. 💡 核心思想

传统强化学习通常分为两类：

- **Model-Free RL**：直接学习策略或价值函数，往往需要大量交互数据。
- **Model-Based RL**：先学习环境模型，再利用模型进行规划，因此通常具有更高的数据效率。

PlaNet 的关键创新是学习一个：

> **Latent Dynamics Model，即潜在动态模型。**

与其在高维像素空间预测未来图像，PlaNet 选择：

1. 先将观测编码到低维 latent state；
2. 在 latent space 中预测未来状态和奖励；
3. 利用规划算法选择动作。

这样既降低了计算成本，也提升了长期预测能力。

---

### 2. 🧩 技术贡献

PlaNet 提出了几项重要技术。

---

#### 2.1 RSSM：Recurrent State Space Model

RSSM 结合了两类状态：

- deterministic state；
- stochastic state。

PlaNet 中的状态可以写成：

$$
s_t = (h_t, z_t)
$$

其中：

- $h_t$ 表示 deterministic state；
- $z_t$ 表示 stochastic state；
- $s_t$ 表示两者结合后的 belief state。

RSSM 的状态更新可以简化理解为：

$$
h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})
$$

$$
z_t \sim p_\theta(z_t \mid h_t)
$$

其中：

- $h_t$ 负责保留历史信息；
- $z_t$ 负责建模环境中的不确定性；
- $a_{t-1}$ 表示上一时刻的动作；
- $\theta$ 表示模型参数。

这个结构能够更好地建模复杂、部分可观测的环境。

---

#### 2.2 Latent Overshooting

Latent Overshooting 是一种多步变分训练目标，用于提升长期预测的准确性。

它的核心思想是：

> 不只让模型预测下一步，而是鼓励模型在 latent space 中预测多步未来。

可以简化写成：

$$
p(z_{t+k} \mid z_t, a_t, a_{t+1}, \dots, a_{t+k-1})
$$

其中：

- $k$ 表示向未来预测的步数；
- $z_{t+k}$ 表示未来第 $k$ 步的 latent state；
- $a_t, a_{t+1}, \dots, a_{t+k-1}$ 表示从当前时刻到未来的动作序列。

这样可以让模型更关注长期动态，而不是只拟合一步预测。

---

#### 2.3 Online Planning

PlaNet 不直接学习一个固定策略网络，而是在每一步都进行在线规划：

1. 在 latent space 中评估大量候选动作序列；
2. 选择当前最优动作；
3. 执行动作后重新规划。

规划过程可以简化写成：

$$
a_t = \arg\max_{a_t, \dots, a_{t+H}} \sum_{k=0}^{H} \gamma^k r_{t+k}
$$

其中：

- $H$ 表示规划的时间范围；
- $\gamma$ 表示折扣因子；
- $r_{t+k}$ 表示未来第 $k$ 步的预测奖励。

---

### 3. ⚙️ 研究方法

很多人认为：

> **World Models (2018) 是概念验证，而 PlaNet 提出的 RSSM 才让世界模型真正变得实用。**

之前的基本思路可以表示为：

```text
Image
  ↓
Latent
  ↓
 RNN
```

PlaNet 提出的 RSSM 结构是：

$$
s_t = (h_t, z_t)
$$

整体流程可以理解为：

```text
Observation
     ↓
  Encoder
     ↓
 Latent State
     ↓
   RSSM
     ↓
 Future State / Reward Prediction
     ↓
 Planning in Latent Space
     ↓
   Action
```

这个结构后来成为 Dreamer、DreamerV2 和 DreamerV3 的基础。

---

### 4. 🚀 影响与后续发展

PlaNet 在多个视觉连续控制任务上取得了重要结果：

- 仅依赖像素输入，就取得了与当时强 model-free 方法相当甚至更好的表现；
- 所需环境交互次数显著减少；
- 证明了世界模型与潜在空间规划的可行性。

不过，PlaNet 也存在一些限制：

- 每一步都需要在线规划，推理开销相对较高；
- 在极长时间跨度或高度随机环境中，模型误差可能逐渐累积；
- 方法主要针对连续控制任务设计，对离散动作和更复杂现实场景仍需进一步扩展。

PlaNet 后来成为 Dreamer 系列算法的重要基础。  
如果继续阅读 Dreamer 系列文章，会发现其中大量公式和思想都可以追溯到 PlaNet。

---

### 📝 Notes

看完这篇论文，你会理解：

- 世界模型的核心概念；
- latent dynamics model 是什么；
- belief state 是什么；
- RSSM 是什么；
- planning in latent space 是什么。

---

## 三、💭 DreamerV3 (2023)

> **Mastering Diverse Domains through World Models** 是 Danijar Hafner、Jurgis Pasukonis、Jimmy Ba 和 Timothy Lillicrap 于 2023 年发表的强化学习论文，也是 **DreamerV3** 算法的代表性论文。  
> 它提出了一种能够 **在众多不同任务上使用同一组超参数完成训练** 的世界模型方法，被认为是模型驱动强化学习的重要进展之一。

---

### 1. 💡 核心思想

DreamerV3 的核心观点是：

> **先学习环境模型，再利用模型“想象”未来，并在想象轨迹中训练策略。**

DreamerV3 不是完全依靠真实环境反复试错，而是先训练一个能够预测未来状态和奖励的 world model。

随后，Agent 在这个内部模型中生成大量想象轨迹：

$$
\tau = (s_t, a_t, r_t, s_{t+1}, a_{t+1}, r_{t+1}, \dots)
$$

然后利用这些 imagined trajectories 训练：

- Actor；
- Critic。

相比传统 model-free reinforcement learning，这种方式通常具有更高的数据效率。

---

### 2. 🧩 技术贡献

DreamerV3 最重要的贡献并不是提出全新的 Dreamer 框架，而是在 DreamerV2 的基础上，通过一系列增强稳定性的技术，使算法能够跨多个领域稳定工作。

主要贡献包括：

---

#### 2.1 统一超参数

DreamerV3 在超过 150 个任务上使用同一组超参数，而不需要针对不同环境反复调参。

---

#### 2.2 训练稳定性

通过归一化、KL balancing、目标变换等技术，提升跨任务训练的鲁棒性。

---

#### 2.3 Scaling 能力

模型规模越大，通常不仅最终性能更高，数据利用效率也更好。

---

#### 2.4 跨领域泛化

DreamerV3 覆盖了多种任务类型：

- 连续控制；
- 离散动作；
- 视觉输入；
- 低维状态输入；
- 2D 环境；
- 3D 环境；
- 开放世界环境。

---

### 3. ⚙️ 研究方法

DreamerV3 的整体训练框架可以表示为：

```text
Observation
     ↓
  Encoder
     ↓
   RSSM
     ↓
World Model
     ↓
Imagined Rollout
     ↓
Actor-Critic Learning
     ↓
   Action
```

其中核心流程是：

1. **Representation Learning**  
   从观测中学习 latent state。

2. **Dynamics Learning**  
   学习 latent state 的时间演化规律。

3. **Reward Prediction**  
   在 latent space 中预测奖励。

4. **Imagination Rollout**  
   在 world model 中生成未来轨迹。

5. **Policy Learning**  
   用想象轨迹训练 Actor 和 Critic。

---

#### 3.1 World Model 学习目标

DreamerV3 的 world model 需要同时学习：

- 表征模型；
- 动态模型；
- 奖励模型；
- continuation model。

可以简化理解为：

$$
p_\theta(s_{t+1} \mid s_t, a_t)
$$

$$
p_\theta(r_t \mid s_t)
$$

$$
p_\theta(c_t \mid s_t)
$$

其中：

- $s_t$ 表示 latent state；
- $a_t$ 表示动作；
- $r_t$ 表示奖励；
- $c_t$ 表示 episode 是否继续；
- $\theta$ 表示 world model 的参数。

---

#### 3.2 Imagination Rollout

DreamerV3 利用学到的 world model 在 latent space 中生成想象轨迹：

$$
s_{t+1} \sim p_\theta(s_{t+1} \mid s_t, a_t)
$$

$$
a_t \sim \pi_\phi(a_t \mid s_t)
$$

其中：

- $\pi_\phi$ 表示 Actor；
- $\phi$ 表示 Actor 的参数；
- $s_{t+1}$ 是 world model 预测出的下一步 latent state。

也就是说，DreamerV3 不一定每次都在真实环境中试错，而是可以在模型内部进行大量想象 rollout。

---

#### 3.3 Actor-Critic 学习

Actor 的目标是最大化想象轨迹中的累计回报：

$$
J(\phi) = \mathbb{E}_{\pi_\phi, p_\theta}
\left[
\sum_{k=0}^{H} \gamma^k r_{t+k}
\right]
$$

其中：

- $J(\phi)$ 表示 Actor 的优化目标；
- $H$ 表示 imagination rollout 的长度；
- $\gamma$ 表示折扣因子；
- $r_{t+k}$ 表示未来第 $k$ 步的预测奖励。

Critic 则学习估计状态价值：

$$
V_\psi(s_t) \approx \mathbb{E}
\left[
\sum_{k=0}^{H} \gamma^k r_{t+k}
\right]
$$

其中：

- $V_\psi$ 表示 Critic；
- $\psi$ 表示 Critic 的参数。

---

### 4. 🧪 实验与结果

论文在多个经典强化学习基准上进行了统一评测，包括：

- DeepMind Control Suite；
- Atari；
- ProcGen；
- DMLab；
- Minecraft；
- BSuite 等。

其中最受关注的成果之一是：

> DreamerV3 在 **没有人类演示数据、没有课程学习 curriculum** 的情况下，仅依靠稀疏奖励，学会在 Minecraft 中从零开始获取钻石。

这一任务长期以来被视为强化学习中的高难度挑战，因为它需要：

- 长时间规划；
- 多阶段工具制作；
- 复杂探索能力；
- 稀疏奖励下的稳定学习。

---

### 5. 🚀 影响与后续发展

DreamerV3 推动了 **Model-Based Reinforcement Learning** 的发展。

它证明：

> 一个设计足够稳健的世界模型，可以跨越机器人控制、电子游戏和开放世界环境，而不需要针对每个领域进行大量人工调优。

这种“一套算法、多领域适用”的理念，对构建更通用的强化学习系统具有重要影响，也成为后续许多世界模型研究和改进工作的基础。

---

### 📝 Notes

看完这篇论文，你会理解完整的世界模型训练框架：

- Reconstruction；
- Dynamics Learning；
- Imagination；
- Policy Learning；
- Actor-Critic；
- 为什么 Dreamer 系列从 planning 转向了 learning policy in imagination。

---

## 四、📌 参考资料

- [World Models](https://arxiv.org/abs/1803.10122)
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
