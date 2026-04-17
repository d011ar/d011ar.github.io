---
layout: post
title: "从零开始系统学习大模型与 Agent：一份可以真正执行的完整学习笔记"
date: 2026-04-17 15:00:00 +0800
categories: Technical Study Notes
tags: [LLM, Agent, Transformer, PyTorch, HuggingFace]
toc: true
---

一份面向初学者同时也是写给自己的系统化学习笔记，从数学基础、Python/PyTorch、深度学习，到 Transformer、微调、AI infra、RAG 与 Agent，整理出完整且可执行的大模型学习路线。

## 摘要

这篇文章尝试回答一个很实际的问题：**如果现在从零开始学习大模型与 Agent，应该按什么顺序学，才能不只停留在知道很多名词的阶段？**

大模型与 Agent 的学习路线不应该被理解成单线程任务，而应该拆成几条并行推进的主线。首先建立 **Python、PyTorch、Git、Linux** 工程地基；然后通过损失函数、反向传播、优化器和 CNN 建立深度学习直觉；接着进入 **Transformer、Decoder-only、大模型微调与推理工程**；再系统补上 AI infra 与大模型工程能力，包括 **混合精度、compile、Attention 算子优化、KV Cache、量化、分布式训练与 MoE**；最后再把能力落到 **RAG 和 Agent** 这些真正有实际应用价值的系统上。

这篇笔记不是速成清单，而是我整理出来的一份系统化学习地图：每一块知识为什么要学、学到什么程度算够、它和前后知识怎么衔接、有哪些值得长期反复阅读的资料。希望它能成为一份真正可执行、可持续补充、可长期维护的学习笔记。

---

## 目录

- [一、大模型与 Agent 起步的学习路线](#一大模型与-agent-起步的学习路线)
- [二、为什么开始学习大模型与 Agent 时有种深深的虚无感](#二为什么开始学习大模型与-agent-时有种深深的虚无感)
- [三、数学与算法基础：不用学成专家，但一定要学到能理解模型为什么这样工作](#三数学与算法基础不用学成专家但一定要学到能理解模型为什么这样工作)
  - [1. 微积分：重点不是做题，而是理解优化](#1-微积分重点不是做题而是理解优化)
  - [2. 线性代数：真正的大模型语言](#2-线性代数真正的大模型语言)
  - [3. 概率论与统计：理解语言模型为什么在预测概率分布](#3-概率论与统计理解语言模型为什么在预测概率分布)
  - [4. 数据结构与算法：不是核心，但很影响工程能力](#4-数据结构与算法不是核心但很影响工程能力)
- [四、真正的起点：先把工程基础补到能用](#四真正的起点先把工程基础补到能用)
  - [1. Python](#1-python)
  - [2. PyTorch](#2-pytorch)
  - [3. Git](#3-git)
  - [4. Linux](#4-linux)
- [五、深度学习基础：先真正理解“训练”到底是什么](#五深度学习基础先真正理解训练到底是什么)
- [六、不要急着上大模型：先通过小项目建立训练直觉](#六不要急着上大模型先通过小项目建立训练直觉)
- [七、进入大模型之前，先认识 Hugging Face 生态](#七进入大模型之前先认识-hugging-face-生态)
- [八、Transformer：这是理解大模型的真正核心](#八transformer这是理解大模型的真正核心)
- [九、大模型的训练流程：预训练、微调与偏好优化](#九大模型训练流程预训练微调与偏好优化)
- [十、大模型工程与 AI Infra：训练、推理与系统优化为什么越来越重要](#十大模型工程与-ai-infra训练推理与系统优化为什么越来越重要)
  - [1. 先建立 AI Infra 视角：为什么“会训练”不等于“会落地”](#1-先建立-ai-infra-视角为什么会训练不等于会落地)
  - [2. 显存与吞吐：先学会估算资源，而不是先盲目开跑](#2-显存与吞吐先学会估算资源而不是先盲目开跑)
  - [3. 混合精度训练：AMP、FP16、BF16 是第一层加速常识](#3-混合精度训练ampf16bf16-是第一层加速常识)
  - [4. `torch.compile`：为什么现在 PyTorch 会把“编译优化”放到主线里](#4-torchcompile为什么现在-pytorch-会把编译优化放到主线里)
  - [5. Attention 内核优化：SDPA、FlashAttention 与更快的注意力实现](#5-attention-内核优化sdpaflashattention-与更快的注意力实现)
  - [6. KV Cache：推理优化的关键，不只是“知道这个词”](#6-kv-cache推理优化的关键不只是知道这个词)
  - [7. 分布式训练：DDP、FSDP、DeepSpeed 应该怎么理解](#7-分布式训练ddpfsdpdeepspeed-应该怎么理解)
  - [8. 量化：让模型跑得动、放得下、成本更低](#8-量化让模型跑得动放得下成本更低)
  - [9. MoE：为什么它既是模型设计问题，也是系统问题](#9-moe为什么它既是模型设计问题也是系统问题)
  - [10. 大模型工程与 AI Infra 应该掌握的基本面](#10-大模型工程与-ai-infra-应该掌握的基本面)
- [十一、评估：不只是跑出结果，而是知道怎么判断结果](#十一评估不只是跑出结果而是知道怎么判断结果)
- [十二、RAG：这是大模型实用化最重要的能力之一](#十二rag这是大模型实用化最重要的能力之一)
- [十三、Agent：从会回答问题到能主动完成任务](#十三agent从会回答问题到能主动完成任务)
- [十四、学到什么程度，才算真的学会了](#十四学到什么程度才算真的学会了)
- [十五、学习过程中容易踩的坑](#十五学习过程中容易踩的坑)
- [十六、给初学者也是我自己的执行建议](#十六给初学者也是我自己的执行建议)
- [十七、如果让我从零开始，我会怎么学](#十七如果让我从零开始我会怎么学)
- [十八、参考资料](#十八参考资料)
- [结语](#结语)

---

## 一、大模型与 Agent 起步的学习路线

如果把整条学习路线压缩成一句话，我会这样概括：

> **先用 Python、PyTorch、Git、Linux 建立工程地基；再用损失函数、反向传播、优化器、CNN 建立深度学习直觉；然后攻克 Transformer、Decoder-only、预训练与微调；再系统补上 AI infra 与大模型工程能力，包括混合精度、compile、Attention 算子优化、KV Cache、量化、分布式训练与 MoE；最后把这些能力落到 RAG 和 Agent 系统上。**

这条路线并不意味着你必须前一章完全学完，才能开始下一章。更合理的方式是：

- 基础线并行补
- 工程线尽早动手
- 大模型主线逐步深入
- 应用线从最小项目开始落地

我更愿意把整个过程看成 8 个阶段：

1. 数学与算法基础  
2. 计算机四件套：Python、PyTorch、Git、Linux  
3. 深度学习基础  
4. 经典神经网络实践  
5. Transformer 与大模型核心  
6. 大模型训练与微调工程  
7. RAG 与检索增强  
8. Agent 与工具调用系统  

> **Note**
> 真正高效的学习方式，不是等全懂了再动手，而是先建立最小可行理解，再通过实践反过来加深理论。

---

## 二、为什么开始学习大模型与 Agent 时有种深深的虚无感

我认为主要有三个原因。

### 1. 只看概念，不写代码

知道 Attention 很重要，知道 RAG 很火，知道 Agent 能调工具，但没有真正自己跑过：

- 一个 tokenizer
- 一次 LoRA 微调
- 一个最小检索流程
- 一个可调用工具的 Agent

这样学到最后，概念越来越多，但没有抓手。

### 2. 只会跑 demo，不理解结构

复制一段代码能跑，并不等于真正理解了它。  
真正的理解至少包括：

- self-attention 到底在算什么
- LoRA 为什么能省显存
- 为什么 chunk 策略会影响 RAG 表现
- 为什么 decoder-only 成为 LLM 主流

### 3. 学习顺序混乱

典型表现包括：

- Python 和 PyTorch 还不熟，就去看多 Agent 架构
- 训练循环都没真正理解，就去研究 DPO/PPO
- embedding 和向量检索没学清楚，就直接搭企业知识库问答

> **Tip**
> 很多“学不明白”的问题，本质不是智力问题，而是顺序问题。

---

## 三、数学与算法基础：不用学成专家，但一定要学到“能理解模型为什么这样工作”

很多人一提到深度学习，就觉得必须先补一整套高等代数、最优化还有概率论来搞清楚矩阵运算、梯度下降和最小化损失函数等这些概念。

但实际上：

> **大多数做大模型应用、微调、RAG、Agent 的人，不需要靠数学拉开差距，但必须有足够的数学直觉，保证自己看得懂训练、注意力、概率和优化。**

---

### 1. 微积分：重点不是做题，而是理解优化

深度学习训练本质上是在做参数优化。  
模型前向传播得到输出，损失函数衡量误差，反向传播计算梯度，再通过梯度下降更新参数。

这一条线里，最值得掌握的是：

- 导数和偏导数
- 链式法则
- 梯度的几何意义
- 多元函数优化的直觉
- 学习率、局部最优、梯度消失/爆炸这些现象背后的原因

如果你问我“微积分学到什么程度算够”，我的标准是：

- 能解释为什么 loss 可以指导参数更新
- 能解释为什么反向传播从后往前算
- 能理解学习率过大为什么会震荡
- 能理解深层网络为什么容易有梯度问题

推荐资料：

- [PyTorch Autograd 教程](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

---

### 2. 线性代数：真正的大模型语言

如果说微积分让你理解模型怎么学，那么线性代数让你理解模型在算什么。

你会在几乎所有大模型核心结构中遇到线性代数：

- embedding 是向量
- 线性层是矩阵变换
- attention 的 Q / K / V 是线性映射后的表示
- 相似度和注意力权重常常建立在点积之上

最值得掌握的内容：

- 向量与矩阵
- 矩阵乘法
- 点积、范数、相似度
- 线性变换
- 特征值与特征向量（理解层面）
- 张量的基本概念

如果你能真正理解点积为什么能表示相关性以及线性层为什么本质上是在做特征空间变换，后面学 Transformer 会轻松很多。

---

### 3. 概率论与统计：理解语言模型为什么在预测概率分布

语言模型底层最重要的任务之一是：

> 给定前文，预测下一个 token 的概率分布。

所以你必须具备最基本的概率论直觉，至少要理解：

- 条件概率
- 随机变量
- 期望和方差
- 熵、交叉熵
- KL 散度
- 抽样和估计的基本思想

这些概念会直接关联到：

- softmax 输出是什么
- 交叉熵为什么常用于分类与语言建模
- 困惑度（PPL）为什么常作为语言模型指标
- KL 散度为什么会出现在蒸馏和对齐任务里

---

### 4. 数据结构与算法：不是核心，但很影响工程能力

这部分虽然不是大模型最核心的理论，但非常影响你的代码质量和工程思维。

建议至少掌握：

- 数组、链表、栈、队列、哈希表
- 树、图
- BFS / DFS
- 双指针、滑动窗口、二分
- 基本动态规划
- 时间复杂度与空间复杂度分析

这会直接帮助你：

- 写更清晰的数据处理逻辑
- 理解检索与索引结构
- 设计更合理的 Agent 执行流程

---

## 四、真正的起点：先把工程基础补到能用

**Python、PyTorch、Git、Linux** 可以被看作是学习深度学习和大模型与 Agent 的工程基础“四件套”。

---

### 1. Python

Python 是整个 AI 工程生态的主语言。  
你不需要一开始就把所有语法学得很深，但至少要达到能独立写脚本、能看懂项目代码、能改训练逻辑的程度。

建议重点掌握：

- 变量、条件、循环、函数
- 列表、字典、集合
- 模块与包
- 文件读写
- 类和对象
- 异常处理
- 虚拟环境
- 常用科学计算库：`numpy`、`pandas`、`matplotlib`

推荐资料：

- [Python 官方教程](https://docs.python.org/3/tutorial/index.html)
- [Python 官方文档首页](https://docs.python.org/3/)

---

### 2. PyTorch

PyTorch 是深度学习最重要的框架之一。它非常适合作为真正理解训练流程的入口。

你至少要掌握这些内容：

- Tensor
- Dataset / DataLoader
- `nn.Module`
- forward
- loss
- optimizer
- backward
- `train()` / `eval()`
- checkpoint 保存与加载

推荐资料：

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Autograd 教程](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [PyTorch Tutorials 首页](https://docs.pytorch.org/tutorials/)

> **Note**
> 学 PyTorch 的最低标准不是“看懂教程”，而是“能自己写一个完整训练循环”。

---

### 3. Git

Git 不只是协作工具，它也是你自己的实验时间机器。

至少要学会：

- `git init`
- `git add`
- `git commit`
- `git status`
- `git log`
- `git branch`
- `git checkout`
- `git merge`
- `.gitignore`
- GitHub 基础工作流

它会帮助你：

- 管理实验版本
- 回滚错误修改
- 保留 baseline 分支
- 让项目结构长期保持整洁

---

### 4. Linux

你不需要变成 Linux 专家，但要具备在服务器上跑实验的工程能力。

至少掌握：

- 文件与目录操作
- 权限
- `grep / find / tail / head`
- 进程查看
- 后台运行
- `tmux`
- 环境变量
- Python / CUDA 环境检查

很多训练问题，本质上不是模型问题，而是环境问题。

---

## 五、深度学习基础：先真正理解“训练”到底是什么

深度学习最核心的不是“模型结构长什么样”，而是下面这条主线：

1. 数据进入模型  
2. 前向传播得到输出  
3. 用损失函数衡量误差  
4. 反向传播计算梯度  
5. 优化器更新参数  

PyTorch 官方基础教程正是围绕这个过程组织的。

推荐资料：

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

### 这一阶段至少要搞懂的内容

#### 损失函数
- MSE / MAE
- 交叉熵
- BCE
- KL 散度（了解）

#### 激活函数
- Sigmoid
- Tanh
- ReLU
- GELU
- Softmax

#### 优化器
- SGD
- Momentum
- Adam
- AdamW

#### 训练技巧
- Dropout
- BatchNorm / LayerNorm
- Residual Connection
- Learning Rate Scheduler
- Gradient Clipping
- Early Stopping

> **Tip**
> 对初学者来说，真正重要的不是背定义，而是把这些概念和训练现象对应起来。

比如：

- 为什么 loss 不降
- 为什么训练集效果好、验证集差
- 为什么换优化器后收敛曲线不一样
- 为什么学习率调度会影响训练稳定性

---

## 六、不要急着上大模型：先通过小项目建立训练直觉

这是我特别想强调的一点。

在正式进入 Transformer 和大模型之前，最好先通过几个经典小任务，建立你对训练过程的直觉。

比如：

- MNIST / FashionMNIST
- CIFAR-10
- 猫狗分类
- 一个最简单的文本分类任务

这些项目的意义不在于“先进”，而在于你会第一次真正面对这些问题：

- 数据怎么读
- 模型怎么定义
- loss 怎么下降
- 什么是过拟合
- 训练曲线和验证曲线为什么分叉
- 学习率、batch size、正则化分别会带来什么影响

推荐资料：

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

---

## 七、进入大模型之前，先认识 Hugging Face 生态

如果说 PyTorch 是深度学习学习期的主战场，那 Hugging Face 基本就是大模型学习期最重要的生态入口之一。

### 1. Hugging Face Hub

Hub 不只是“模型下载站”，它更像一个围绕模型、数据集和 demo 的协作平台。

推荐资料：

- [Hugging Face Hub 文档](https://huggingface.co/docs/hub/index)
- [huggingface_hub Quickstart](https://huggingface.co/docs/huggingface_hub/en/quick-start)

### 2. Transformers

Transformers 是最重要的主干库之一。  
它负责模型加载、推理、微调和很多工程接口。

推荐资料：

- [Transformers Quickstart](https://huggingface.co/docs/transformers/en/quicktour)

### 3. Datasets

Datasets 提供了非常方便的数据读取、处理和组织方式。

推荐资料：

- [Datasets Quickstart](https://huggingface.co/docs/datasets/quickstart)
- [Datasets 文档首页](https://huggingface.co/docs/datasets/index)

### 4. LLM Course

如果只推荐一个系统入口，我会非常推荐 Hugging Face 的 LLM Course。

推荐资料：

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1)

> **Note**
> 学 Hugging Face 生态时，不要只停留在“会复制 quickstart”。更重要的是搞懂：
> - tokenizer 和模型为什么必须配套
> - dataset 到底怎么进入训练流程
> - Hub、Transformers、Datasets、PEFT、TRL 之间是什么关系

---

## 八、Transformer：这是理解大模型的真正核心

如果说只能选一个最重要的大模型基础主题，那一定是 **Transformer**。

经典论文：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### 为什么 Transformer 重要

因为今天绝大多数主流大语言模型，虽然各有变体，但核心坐标系仍然来自 Transformer。

如果不理解 Transformer，你很难真正理解：

- token 表示是怎么被上下文更新的
- 为什么上下文长度重要
- 为什么 KV Cache 可以优化生成
- 为什么 LoRA 常作用于 attention 相关层
- 为什么 decoder-only 成为 LLM 主流路线

### 学 Transformer 时必须真正看懂的部分

- Embedding
- Positional Encoding
- Self-Attention
- Multi-Head Attention
- Feed Forward Network
- Residual Connection
- LayerNorm

### Q / K / V 的直觉

一种很有用的理解方式是：

- Query：我在找什么
- Key：我有什么特征
- Value：如果你关注我，我提供什么信息

self-attention 的过程，本质上就是：

1. 用 Query 和所有 Key 计算相关性  
2. 通过 softmax 得到权重  
3. 用这些权重对 Value 做加权汇总  

### Encoder、Decoder、Decoder-only 一定要分清

Hugging Face 的课程把三种范式区分得很清楚：

- Encoder-only：更偏理解任务
- Decoder-only：更偏自回归生成
- Encoder-Decoder：更适合 seq2seq 任务

如果你的目标是学 LLM，那更核心的学习内容是 **Decoder-only**。

推荐资料：

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## 九、大模型的训练流程：预训练、微调与偏好优化

理解大语言模型，不仅要知道它的结构，还要知道它是怎样一步步从“能够预测下一个词”发展到“能够理解指令、完成任务，并尽量符合人类偏好”的。

通常可以把这一过程分成几个阶段：**预训练、微调，以及偏好优化**。

### 1. 预训练

预训练是大语言模型最基础、也最关键的阶段。在这一阶段，模型会在海量文本数据上学习语言中的统计规律、知识模式和表达方式。预训练的意义在于，它让模型先获得通用的语言能力，例如：

- 理解上下文
- 学习语法和常识
- 积累跨领域知识
- 形成较强的迁移能力

即使你不亲自从零开始训练一个大模型，也仍然需要理解预训练，因为它决定了模型能力的上限和基础。尤其需要明白以下几点：

- **预训练为什么成本极高**：因为它需要极大规模的数据、算力和训练时间。
- **预训练为什么能带来广泛的泛化能力**：因为模型在海量多样化语料中学到的是较通用的语言规律，而不是某一个具体任务。
- **为什么说预训练是下游能力的底座**：后续的指令跟随、问答、代码生成等能力，通常都建立在预训练得到的基础能力之上。

### 2. 监督微调（Supervised Fine-Tuning, SFT）

经过预训练后，模型已经具备较强的续写能力，但这并不意味着它已经会听指令。

例如，一个只经过预训练的模型看到“请总结下面这段文字”，它未必能稳定地按照要求输出总结，它更可能只是继续生成与上下文统计上更接近的文本。

这时就需要进行**SFT**。  
这一阶段会使用“指令—回答”形式的数据来训练模型，让模型学会：

- 理解任务要求
- 按照指定格式作答
- 更稳定地完成问答、翻译、摘要、分类等任务

因此，监督微调的作用，可以理解为：

> 让模型从会生成语言，进一步变成会按照要求完成任务。

也就是说，预训练主要解决模型懂不懂语言的问题，而监督微调主要解决模型能不能按人的要求做事的问题。

推荐资料：

- [Transformers Quickstart](https://huggingface.co/docs/transformers/en/quicktour)
- [TRL 文档](https://huggingface.co/docs/trl/index)

### 3. 参数高效微调（PEFT）与 LoRA

当我们想让模型适应某个具体任务或领域时，传统做法是对模型全部参数进行微调。  
但对于大语言模型来说，这种方式往往代价很高，因为模型参数量巨大，显存和计算资源消耗都非常高。

因此，实践中常常采用**参数高效微调**（Parameter-Efficient Fine-Tuning, PEFT）。

它的核心思想是：

> 不去更新模型中的全部参数，而是只训练少量新增参数，或只调整很小一部分参数，从而以更低成本完成微调。

在参数高效微调方法中，**LoRA**（Low-Rank Adaptation，低秩适配）是最常见、也最适合初学者入门的一种方法。

LoRA 的基本思路不是直接修改原有大矩阵，而是在某些线性层旁边增加一组低秩参数，用较少的可训练参数去近似完成模型的调整。这样做的好处是：

- 显存占用更低
- 训练成本更小
- 部署和保存更方便
- 对初学者更容易上手

因此，对刚开始学习微调的人来说，通常更建议先掌握 LoRA，再去接触全参数微调。

推荐资料：

- [PEFT 文档](https://huggingface.co/docs/peft/en/index)
- [LoRA 文档](https://huggingface.co/docs/peft/package_reference/lora)
- [Transformers 中的 PEFT](https://huggingface.co/docs/transformers/en/peft)

> **提示**  
> 对初学者来说，第一次学习大模型微调时，优先掌握 LoRA 往往比直接学习全参数微调更实用。

### 4. 偏好优化：DPO、PPO 与 RLHF

当模型已经具备基本的指令跟随能力后，人们往往还希望它的回答能够**更符合人类偏好**，例如：

- 更有帮助
- 更自然
- 更安全
- 更符合价值判断或输出风格要求

这就进入了**偏好优化**（preference optimization）的阶段。这里有几个常见概念：

- **RLHF**：Reinforcement Learning from Human Feedback，基于人类反馈的强化学习，利用人类偏好反馈来优化模型输出的整体框架。
- **PPO**：Proximal Policy Optimization，近端策略优化，是一种强化学习算法，早期在 RLHF 中被广泛使用。
- **DPO**：Direct Preference Optimization，直接偏好优化，是一种近年来非常受关注的方法。它不需要像传统强化学习那样显式训练奖励模型并进行复杂采样，理解和实现通常更直接一些。

对初学者来说，可以先建立一个清晰的分工认识：

- **监督微调（SFT）**主要解决“模型能不能按要求完成任务”；
- **偏好优化（例如 DPO、PPO）**主要解决“模型在多个可行答案中，是否更倾向于输出人类更喜欢的那个”。

简单说：

- SFT 让模型“会做事”
- DPO / PPO 让模型“做得更符合偏好”

在入门阶段，**DPO 通常比 PPO 更容易理解**，因为它的训练流程相对直接，工程复杂度也通常更低，所以常常更适合作为学习偏好优化的第一站。

推荐资料：

- [TRL 文档首页](https://huggingface.co/docs/trl/index)
- [DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [PPO Trainer](https://huggingface.co/docs/trl/ppo_trainer)

### 5. 对训练流程的整体认识

学习这一部分时，不必一开始就纠结所有训练细节，更重要的是先建立整体框架：

1. **预训练**：让模型学会语言规律和通用知识。  
2. **监督微调**：让模型学会理解指令并完成具体任务。  
3. **参数高效微调（如 LoRA）**：让微调在资源有限的情况下也能实际进行。  
4. **偏好优化（如 DPO、PPO）**：让模型输出更符合人类偏好，而不仅仅是“能答出来”。

## 十、大模型工程与 AI Infra：训练、推理与系统优化为什么越来越重要

当模型变大以后，问题就不只是会不会训练，而是：

- 能不能装得下
- 跑得快不快
- 显存够不够
- 推理延迟高不高
- 吞吐能不能上去
- 集群成本能不能接受
- 训练和部署是否稳定

如果说前面的章节更多在回答模型是什么、怎么学、怎么微调，那这一章更关心的是：

> **模型如何真正跑在现实世界里。**

在今天的大模型业界落地里，AI infra 不是附属知识，而是核心竞争力的一部分。很多时候，模型效果差距没有你想象的大，真正决定一个系统能不能上线、能不能承受用户量、能不能把训练成本压下来、能不能支持长上下文和低延迟推理的，往往正是这一层系统优化能力。

> **Note**
> 学大模型工程，不是为了把自己立刻变成 CUDA 内核工程师，而是为了建立一种系统视角：
> 你要逐渐明白一个模型为什么慢、慢在哪里、能从哪一层优化、哪些优化是在算子层、哪些优化是在图编译层、哪些优化是在分布式和内存管理层。

### 1. 先建立 AI Infra 视角：为什么“会训练”不等于“会落地”

很多初学者会默认认为：只要模型结构、数据和训练代码没有问题，剩下的就是把它跑起来。

但在真实工程环境中，这远远不够。你很快就会遇到这些问题：

- 同样的模型，为什么别人训练得更快
- 为什么同样的 batch size，你会爆显存
- 为什么推理时首 token 延迟很高
- 为什么长上下文一上去就速度骤降
- 为什么多卡训练并没有线性提速
- 为什么推理吞吐和单条响应速度之间常常需要权衡

这些问题背后，涉及的是完整的 AI infra 视角：**算子、内存、并行、通信、缓存、图编译和部署策略**。PyTorch 近几年的主线更新也明显在往这边推进，比如 [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)、[AMP](https://docs.pytorch.org/docs/stable/amp.html)、[SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 和 [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) 都已经进入官方主文档和教程主线。

### 2. 显存与吞吐：先学会估算资源，而不是先盲目开跑

大模型工程最现实的问题，通常不是能不能写代码，而是能不能放得下。  
Hugging Face Accelerate 提供了模型内存估算工具，就是为了帮助你在训练前先判断资源需求。

推荐资料：

- [Accelerate Model Memory Estimator](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator)

你至少要逐渐建立一种“资源拆解意识”：

- **参数**会占显存
- **梯度**会占显存
- **优化器状态**会占显存
- **激活值**会占显存
- **KV cache** 在推理阶段也会持续占显存

同样地，除了显存放不放得下，你还要开始关心 **吞吐（throughput）** 和 **延迟（latency）**。吞吐更关注单位时间能处理多少请求，延迟更关注单个请求多久返回；这两个目标在系统里并不总是天然一致。KV cache、量化、批处理、静态 cache、编译优化，很多时候都是在这两者之间做平衡。

### 3. 混合精度训练：AMP、FP16、BF16 是第一层加速常识

Automatic Mixed Precision，也就是 AMP，是今天训练和推理中几乎默认要理解的优化手段。PyTorch 官方文档明确说明，AMP 会让一部分算子使用 `float16` 或 `bfloat16` 这类更低精度类型，而把更需要动态范围的算子保留在 `float32`，从而降低内存占用并提升速度。

推荐资料：

- [Automatic Mixed Precision (torch.amp)](https://docs.pytorch.org/docs/stable/amp.html)

初学者至少应该区分这些概念：

- **FP32**：传统单精度，稳定但更慢、更占显存  
- **FP16**：速度快、显存省，但更容易数值不稳定  
- **BF16**：很多现代训练里非常常见，通常比 FP16 更稳一些，同时仍保留低精度优势  
- **autocast**：自动选择更合适的精度执行不同算子  
- **gradient scaling**：防止低精度训练时梯度下溢  

这一层你不一定需要一开始就研究非常细的数值分析，但必须知道：  
**混合精度不是“可选的小技巧”，而是现代深度学习训练与推理的基础常识。**

### 4. `torch.compile`：为什么现在 PyTorch 会把“编译优化”放到主线里

如果前几年学 PyTorch，可能会觉得它强调的是动态图、灵活和易于研究；但现在，`torch.compile` 已经成为 PyTorch 官方主推的性能优化入口之一。官方教程的表述很直接：`torch.compile` 可以通过 JIT 编译和后端优化让 PyTorch 代码跑得更快，而且通常只需要很少代码改动。

推荐资料：

- [torch.compile 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

对学习者来说，最重要的不是立刻深究编译器内部细节，而是先理解：

- 为什么 Python 层解释执行会带来开销  
- 为什么把图捕获并编译后，能融合更多算子  
- 为什么一些模型在 `torch.compile` 下收益明显，而另一些收益有限  
- 为什么动态图、cache 策略和编译之间会有兼容性问题  

这一点和 Hugging Face 的 KV cache 文档也呼应得很明显：不同 cache 类型对 `torch.compile()` 的支持并不相同，例如 Static Cache 支持 `torch.compile()`，而 Dynamic Cache 通常不支持。

### 5. Attention 内核优化：SDPA、FlashAttention 与更快的注意力实现

Attention 是 Transformer 的核心，同时也是性能热点之一。PyTorch 已经把 `scaled_dot_product_attention` 做成了官方接口，并且专门写了高性能 tutorial，强调 fused implementation 相比朴素实现会有明显性能收益。

推荐资料：

- [scaled_dot_product_attention 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)

而 FlashAttention 则是这条路线中最有代表性的工作之一。它通过优化 attention 的内存访问方式，显著改善速度和显存表现。

推荐资料：

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention 官方仓库](https://github.com/dao-ailab/flash-attention)

这部分不一定要一上来就去啃 CUDA 内核，但一定要建立这个意识：  
**Attention 的优化不只是数学问题，也是内存访问模式和硬件友好性问题。**  
这正是为什么现在“更快的 attention”往往首先是一个系统优化问题，而不是一个纯架构命名问题。

### 6. KV Cache：推理优化的关键，不只是“知道这个词”

KV cache 在今天的 LLM 推理中已经是基本概念。Hugging Face 文档对它的定义很明确：KV cache 会保存之前 token 的 key-value 对，在后续生成中复用，避免重复计算；同时官方也明确提醒，cache 应该用于推理，不建议在训练时开启。

推荐资料：

- [Transformers Cache Explanation](https://huggingface.co/docs/transformers/cache_explanation)
- [Transformers KV Cache](https://huggingface.co/docs/transformers/kv_cache)

更值得学的是它背后的工程取舍。Hugging Face 现在已经把多种 cache 策略写进文档里，比如：

- **Dynamic Cache**：更灵活，但通常不支持 `torch.compile()`  
- **Static Cache**：更适合与 `torch.compile()` 结合，但内存占用更高  
- **Quantized Cache**：更省内存，但功能支持更有限  
- 还有 **offloading** 一类思路，用 CPU/GPU 间迁移来换显存空间  

也就是说，KV cache 不再只是“知道原理”就够了，而是要逐渐理解它和 **内存、吞吐、延迟、编译兼容性** 的关系。

### 7. 分布式训练：DDP、FSDP、DeepSpeed 应该怎么理解

很多人第一次接触多卡训练时，容易把所有东西都混成一个概念。更好的理解方式是分层看。

- **DDP** 更像经典的数据并行  
- **FSDP** 更进一步，把参数、梯度和优化器状态做分片，降低单卡内存占用  
- **DeepSpeed** 提供更完整的大模型训练优化和系统支持  

PyTorch FSDP 文档和教程都明确指出，FSDP 的目标之一就是通过 sharding 参数、梯度和优化器状态，降低 GPU 内存占用，让单卡放不下的模型变得可训练。

推荐资料：

- [FSDP 官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)
- [FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)

这一层对初学者最重要的不是先去配置复杂环境，而是先想明白：  
**为什么复制整份模型到每张卡上会浪费内存，为什么分片、通信和重组是必要的。**

### 8. 量化：让模型跑得动、放得下、成本更低

Transformers 文档把量化定义为：用更低精度的数据类型表示权重和激活，以降低内存和计算成本；bitsandbytes 则是 Hugging Face 生态中常见的 8-bit / 4-bit 量化方案。

推荐资料：

- [Transformers Quantization](https://huggingface.co/docs/transformers/en/main_classes/quantization)
- [bitsandbytes 文档](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)

这部分建议你强调三件事：

- 量化并不只是“能不能省显存”，还关系到部署成本  
- 量化往往和推理速度、带宽、内存占用一起讨论  
- 量化和 KV cache、batching、服务框架往往是一起设计的  

### 9. MoE：为什么它既是模型设计问题，也是系统问题

MoE，也就是 Mixture of Experts，这几年已经从论文热点越来越变成实际工程讨论中的重要对象。它不只是模型结构话题，还涉及训练方式、路由机制，以及推理服务时需要考虑的 trade-off。

推荐资料：

- [Mixture of Experts Explained](https://huggingface.co/blog/moe)

MoE 非常适合作为理解现在大模型工程已经不只是单卡单模型的一个窗口，它恰好体现了大模型架构和系统工程之间的交叉：

- 从模型角度看，它是稀疏激活和专家路由  
- 从系统角度看，它涉及专家并行、负载均衡、通信开销、路由热点和推理调度  

### 10. 大模型工程与 AI Infra 应该掌握的基本面

如果按是否真正掌握来衡量，这一部分建议的标准是：

- 你知道训练和推理为什么会慢，而且能大致说出慢在哪里  
- 你知道 AMP、`torch.compile`、SDPA / FlashAttention、KV cache、量化分别在解决什么问题  
- 你知道 DDP、FSDP、DeepSpeed 大致分别在什么层面起作用  
- 你知道 MoE 不只是模型结构问题，还涉及路由和系统成本  
- 你不一定能自己写 CUDA kernel，但你已经有能力读懂主流工程文章和官方文档，知道该往哪一层排查性能问题  

> **Tip**
> 这一章真正想训练的能力，不是“背下所有缩写”，而是形成一种性能分析视角：
> 一个模型跑不快，到底应该先查精度、算子、缓存、编译、并行，还是内存和通信？

---

## 十一、评估：不只是跑出结果，而是知道怎么判断结果

我们可能常常高估了模型能输出点什么的意义，低估了模型评估的价值。

实际上，真正进入研究和工程阶段以后，越来越发现：

> 模型跑起来只是开始；  
> 知道它到底好不好、为什么好、哪里不好，才是关键。

### 至少应该熟悉的指标

- Accuracy
- Precision
- Recall
- F1
- BLEU
- ROUGE
- Perplexity（PPL）

### 为什么 PPL 常见，但不能代表一切

PPL 衡量的是语言模型对 token 序列的预测能力。  
它很重要，但并不等于：

- 对话质量好
- 指令遵循能力强
- RAG 表现稳定

所以自动指标、任务指标和人工评估往往需要结合使用。

---

## 十二、RAG：这是大模型实用化最重要的能力之一

如果只依赖模型参数内的知识来回答问题，很快就会遇到：

- 知识过时
- 私有文档无法使用
- 容易幻觉
- 回答不可追溯

RAG 的意义就在这里。

推荐资料：

- [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)

### RAG 的基本范式

一个典型 RAG 流程通常是：

1. 用户提问  
2. 把问题转成 embedding  
3. 在向量库中检索相关文本块  
4. 把检索结果拼接进 prompt  
5. 让 LLM 基于上下文生成答案  

### 建 RAG 时要学会的核心组件

#### Embedding
- [Sentence Transformers Quickstart](https://sbert.net/docs/quickstart.html)

#### 向量检索
- [FAISS 官方文档](https://faiss.ai/index.html)

#### 评估
- [Ragas Get Started](https://docs.ragas.io/en/latest/getstarted/)
- [Ragas Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)

> **Tip**
> 很多人以为 RAG 的难点只是“接个向量库”。  
> 实际上，chunk 策略、召回质量、上下文拼接方式和评估方法，都会显著影响效果。

---

## 十三、Agent：从“会回答问题”到“能主动完成任务”

> **Agent 是一种围绕任务目标动态决定下一步、调用工具、读取外部状态并推进任务完成的系统。**

推荐资料：

- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)

### Agent 和普通 LLM 应用的区别

普通应用通常是：

- 输入 prompt
- 输出回答

Agent 更像：

- 理解任务
- 决定下一步
- 调用工具
- 观察工具结果
- 再决定下一步
- 直到完成目标

### Agent 的核心组成

一个最小 Agent 系统通常包括：

- 模型
- 工具
- 状态 / 记忆
- 控制流
- 停止条件

### Workflow 和 Agent 不一样

这点非常重要。

- Workflow：步骤大多是预定义的
- Agent：步骤可以动态决定

推荐资料：

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)

### 多 Agent 很火，但不一定适合作为起点

推荐资料：

- [LangChain Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)

> **Note**
> 初学 Agent 时，更合理的顺序通常是：
> 1. 单工具调用
> 2. 单 Agent 循环
> 3. RAG + 工具 Agent
> 4. 再考虑多 Agent

---

## 十四、学到什么程度，才算真的学会了

用“可操作标准”来判断是否掌握，而不是“读过看过”。

### 数学基础
- 能看懂大部分常见深度学习公式
- 不会一看到 attention 公式就完全断线
- 能理解梯度、概率分布、交叉熵的基本含义

### Python / PyTorch
- 能自己写训练循环
- 能自己定义 Dataset 和 Model
- 能 debug 常见 shape / device 错误
- 能保存和恢复模型

### 深度学习基础
- 能解释 loss、backprop、optimizer、overfitting
- 能独立完成一个小任务训练

### Transformer
- 能讲清 self-attention、位置编码、encoder/decoder
- 能理解 decoder-only 为什么适合 LLM

### 微调与工程
- 跑通过一次 LoRA 或其他 PEFT 微调
- 理解量化、KV Cache、混合精度的意义
- 理解 `torch.compile`、SDPA / FlashAttention 这类优化是在解决什么问题
- 知道显存大致会从哪里爆
- 对 DDP / FSDP / DeepSpeed 的作用层次有基本判断

### RAG
- 做过 embedding + 检索 + 生成
- 理解 chunk、top-k、rerank 的基本影响
- 做过最小评估

### Agent
- 做过一个带工具的 Agent
- 理解 workflow 和 agent loop 的区别
- 知道什么时候根本没必要用 Agent

---

## 十五、学习过程中容易踩的坑

### 坑一：觉得必须先把理论知识全学完
过分在意理论学习可能会阻碍我们的实操，了解基本的数学理论后我们可以在实验中边做边补齐更深入的数学理论知识。

### 坑二：只看视频，不写代码
这个领域最怕“知识幻觉”。我们以为自己懂了，很多时候只是看懂了讲解，但不知道如何落地到一个具体项目。

### 坑三：一上来就追最热名词
MCP、Multi-Agent、Long Context、Reasoning 都可以学，但前提是基础不能空心化。

### 坑四：把一切都叫 Agent
不是所有带 prompt 的流程都算 Agent。  
真正的 Agent 通常包含：

- 动态决策
- 工具调用
- 状态管理
- 循环执行

### 坑五：只写代码不评估
只会说“看起来还可以”，很难真正走远。评估 Agent 功能的有效性和各个层面的性能也是重要环节。

---

## 十六、给初学者也是我自己的执行建议

第一，**先把工程基础补齐，再进深度学习。**  
第二，**先学会训练一个小模型，再谈大模型。**  
第三，**Transformer 一定要理解掌握，不要只会背名词。**  
第四，**LoRA、混合精度、compile、Attention 优化、KV Cache、量化，是最值得尽早建立概念的大模型工程能力。**  
第五，**RAG 比很多人想象得更重要，因为它最接近真实应用。**  
第六，**Agent 的正确打开方式是从单 Agent + 工具开始，而不是一上来做多 Agent 系统。**

---

## 十七、如果让我从零开始，我会怎么学

### 第一阶段：打地基
- Python 官方教程
- PyTorch Learn the Basics
- Autograd 教程
- Git 与 Linux 基础

### 第二阶段：建立训练直觉
- 图像分类或文本分类
- 自己写完整训练循环
- 学会看 loss curve 和验证集表现

### 第三阶段：进入 Transformer 与 Hugging Face
- Hugging Face LLM Course
- Transformers Quickstart
- Datasets / Tokenizers / Hub
- 真正看懂 Transformer 原论文

### 第四阶段：进入大模型微调与工程
- LoRA / PEFT
- 混合精度（AMP、FP16、BF16）
- `torch.compile`
- SDPA / FlashAttention
- KV Cache
- 量化
- 显存估算
- DDP / FSDP / DeepSpeed 的基本认识
- 初步理解 MoE 的系统代价

### 第五阶段：做一个最小 RAG
- Sentence Transformers
- FAISS
- 最简单的问答系统
- 用 Ragas 做一次基本评估

### 第六阶段：做一个最小 Agent
- 一个单 Agent
- 两到三个工具
- 明确任务目标
- 简单状态管理
- 再去理解 LangGraph 的优势

---

## 十八、参考资料

### 1. Python、工程基础与环境

- [Python 官方教程](https://docs.python.org/3/tutorial/index.html)
- [Python 官方文档首页](https://docs.python.org/3/)
- [PyTorch Tutorials 首页](https://docs.pytorch.org/tutorials/)

### 2. PyTorch 基础与训练入门

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Autograd 教程](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

### 3. Hugging Face 生态入门

- [Hugging Face Hub 文档](https://huggingface.co/docs/hub/index)
- [huggingface_hub Quickstart](https://huggingface.co/docs/huggingface_hub/en/quick-start)
- [Transformers Quickstart](https://huggingface.co/docs/transformers/en/quicktour)
- [Datasets Quickstart](https://huggingface.co/docs/datasets/quickstart)
- [Datasets 文档首页](https://huggingface.co/docs/datasets/index)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1)

### 4. Transformer 与经典基础论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### 5. 微调、对齐与训练流程

- [PEFT 文档](https://huggingface.co/docs/peft/en/index)
- [LoRA 文档](https://huggingface.co/docs/peft/package_reference/lora)
- [Transformers 中的 PEFT](https://huggingface.co/docs/transformers/en/peft)
- [TRL 文档](https://huggingface.co/docs/trl/index)
- [DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [PPO Trainer](https://huggingface.co/docs/trl/ppo_trainer)

### 6. 大模型工程与 AI Infra：精度、编译与算子优化

- [Automatic Mixed Precision (torch.amp)](https://docs.pytorch.org/docs/stable/amp.html)
- [torch.compile 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [scaled_dot_product_attention 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch SDPA Tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)

### 7. 推理优化：KV Cache、量化与内存

- [Transformers Cache Explanation](https://huggingface.co/docs/transformers/cache_explanation)
- [Transformers KV Cache](https://huggingface.co/docs/transformers/kv_cache)
- [Transformers Quantization](https://huggingface.co/docs/transformers/en/main_classes/quantization)
- [bitsandbytes 文档](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [Accelerate Model Memory Estimator](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator)

### 8. 分布式训练与系统扩展

- [FSDP 官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)
- [FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)

### 9. 高性能 Attention 与前沿系统优化

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention 官方仓库](https://github.com/dao-ailab/flash-attention)
- [Mixture of Experts Explained](https://huggingface.co/blog/moe)

### 10. RAG、检索与评估

- [Sentence Transformers Quickstart](https://sbert.net/docs/quickstart.html)
- [FAISS 官方文档](https://faiss.ai/index.html)
- [Ragas Get Started](https://docs.ragas.io/en/latest/getstarted/)
- [Ragas Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)

### 11. Agent 与工作流编排

- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)

---

## 结语

大模型与 Agent 的学习实际上并不是“先决定要不要做研究”，也不是“先决定以后做哪条技术栈”，而是一个更基础的问题：

> 你愿不愿意用一种长期、系统、反复动手的方式，把这门复杂技术真正搭成自己的能力结构。

如果只是想知道点新名词，那几天就够。  
但如果目标是让自己真正具备：

- 看懂模型结构
- 跑通训练与微调
- 搭建 RAG
- 构建 Agent
- 读懂官方文档
- 能继续向更深方向走

那就必须接受这是一条需要分阶段推进、但每一步都能看到成果的学习路径。

这篇笔记总结出一条学习路径主线：

**基础工程 → 深度学习 → Transformer → 微调与工程 → RAG → Agent。**

循序渐进的学习和动手，这条路其实没有看上去那么不可攀登。