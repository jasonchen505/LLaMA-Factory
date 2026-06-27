# LLM/Agent 后训练算法实习面试 -- 五类问题应对指南

---

## 目录

- [第一类：底层原理深挖](#第一类底层原理深挖)
- [第二类：实验与方案验证能力](#第二类实验与方案验证能力)
- [第三类：问题定位与排查能力](#第三类问题定位与排查能力)
- [第四类：工程落地能力](#第四类工程落地能力)
- [第五类：业务与场景理解](#第五类业务与场景理解)

---

## 第一类：底层原理深挖

> **面试官意图**：不是考你能不能背概念，而是看你是否理解"为什么这么设计"、"解决了什么问题"、"有什么局限"、"怎么改进"。

---

### Q1: LoRA 为什么有效？它的核心假设是什么？有什么局限？

**回答框架：问题 -> 假设 -> 设计 -> 局限 -> 改进**

**解决什么问题**：
全参数微调一个 7B 模型需要 ~120GB 显存（bf16），存储一份完整模型副本成本高昂。核心问题是：下游任务是否真的需要更新所有参数？

**核心假设**：
预训练模型已经学到了丰富的特征表示，下游任务的适配发生在低维子空间中。即权重更新矩阵 ΔW 具有低秩特性（low intrinsic rank）。

**设计原理**：
- 冻结预训练权重 W（d×d），仅训练 ΔW = BA（B: d×r, A: r×d, r << d）
- 参数量从 d² 降到 2dr，当 r=16, d=4096 时，参数量减少 128 倍
- 推理时可合并 W' = W + BA，无额外推理延迟

**LLaMA-Factory 实现细节**（源码 `model/adapter.py`）：
```python
# 当 lora_target="all" 时，自动检测所有 Linear 模块
def find_all_linear_modules(model, freeze_vision_tower):
    # 遍历 model.named_modules()，找到所有 nn.Linear
    # 排除视觉塔和多模态投影器（get_forbidden_modules()）
```
- `cast_trainable_params_to_fp32`：QLoRA 中 4-bit 基础模型 + fp32 适配器，确保梯度计算精度

**局限性**：
1. **秩的选择是全局的**：不同层的最优 rank 可能不同，但标准 LoRA 对所有层使用相同 rank
2. **线性约束**：只能捕捉线性子空间中的变化，非线性任务适配能力有限
3. **任务干扰**：多任务场景下，不同任务的 LoRA 可能冲突
4. **初始化敏感**：标准零初始化可能不是最优的（PiSSA 用 SVD 初始化解决此问题）

**改进方向**（在 LLaMA-Factory 中已有实现）：
| 改进 | 解决的问题 | 实现 |
|------|-----------|------|
| **DoRA** | 幅度-方向解耦 | `use_dora=True`，LoRA 仅更新方向 |
| **rsLoRA** | 大 rank 训练不稳定 | 缩放因子改为 alpha/sqrt(rank) |
| **LoRA+** | A/B 矩阵学习率不匹配 | `loraplus_lr_ratio`，B 矩阵用更大学习率 |
| **PiSSA** | 零初始化信息损失 | SVD 初始化保留主成分 |
| **OFT** | 低秩加法破坏超球面结构 | 正交变换保持几何结构 |

---

### Q2: DPO 的数学推导是什么？相比 PPO 有什么本质优势和劣势？

**回答框架：从 RLHF 目标出发 -> 闭式解 -> 反解 reward -> 代入 BT 模型**

**推导过程**：

RLHF 的优化目标：
```
max_π E_{x~D, y~π(·|x)} [r(x,y)] - β * KL(π || π_ref)
```

这个带 KL 约束的优化有闭式解：
```
π*(y|x) = π_ref(y|x) * exp(r(x,y)/β) / Z(x)
```
其中 Z(x) = Σ_y π_ref(y|x) * exp(r(x,y)/β) 是配分函数。

反解 reward：
```
r(x,y) = β * log(π*(y|x)/π_ref(y|x)) + β * log Z(x)
```

代入 Bradley-Terry 偏好模型（Z(x) 消去，因为只关心 reward 差值）：
```
P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))
                = σ(β * log(π*(y_w|x)/π_ref(y_w|x)) - β * log(π*(y_l|x)/π_ref(y_l|x)))
```

取负对数似然得到 DPO 损失：
```
L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

**LLaMA-Factory 实现**（`train/dpo/trainer.py`）：
```python
def concatenated_forward(self, model, batch):
    # chosen 和 rejected 拼接在同一个 batch 中处理
    labels = batch.pop("labels")
    all_logits = model(**batch, return_dict=True, use_cache=False).logits
    all_logps, valid_length = get_batch_logps(logits=all_logits, labels=labels)
    # 拆分 chosen 和 rejected
    batch_size = batch["input_ids"].size(0) // 2
    chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
```

**相比 PPO 的优势**：
1. **无需训练奖励模型**：直接从偏好数据学习，减少训练管线复杂度
2. **无需在线采样**：PPO 需要在训练过程中不断生成样本，DPO 使用离线数据
3. **训练更稳定**：没有 PPO 的 clipping、GAE、value function 等复杂组件
4. **超参更少**：主要就是 beta，PPO 有 clip_range, vf_coef, ent_coef 等

**DPO 的本质劣势**：
1. **离线局限**：数据分布固定，无法探索新的回复空间
2. **分布偏移**：训练数据的 π_data 与当前策略 π 不一致，导致估计偏差
3. **表达能力受限**：隐式奖励模型受限于策略模型的表达能力
4. **对数据质量敏感**：chosen/rejected 质量直接影响效果

**LLaMA-Factory 支持的 DPO 变体**（解决不同局限）：

| 变体 | 解决的问题 | 代码实现 |
|------|-----------|---------|
| **IPO** | DPO 对数据噪声敏感 | 平方损失替代 log sigmoid |
| **SimPO** | chosen/rejected 概率同时降低 | 添加 gamma margin |
| **ORPO** | 需要参考模型 | 结合 SFT loss + odds ratio |
| **BCO** | 奖励基线不稳定 | 运行均值自适应基线 |
| **KTO** | 需要配对数据 | 二元反馈即可 |

---

### Q3: PPO 中的 Value Head 是什么？为什么需要它？KL 惩罚的作用是什么？

**Value Head 的作用**：

PPO 是 Actor-Critic 架构：
- **Actor**：策略网络（语言模型本身），输出 action 分布
- **Critic**：价值网络，估计状态价值 V(s)

Value Head 是在 CausalLM 最后一层隐藏状态上添加的线性层：
```python
# train/ppo/trainer.py 中加载
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
# 输出：logits（策略输出） + value（状态价值标量）
```

**为什么需要 Value Head**：
- 优势函数 A(s,a) = R(s,a) - V(s) 需要 V(s) 作为基线
- 没有 V(s) 作为基线，直接用 R(s) 会导致高方差
- V(s) 提供了"这个状态本身有多好"的估计，A(s,a) 表示"这个 action 比平均水平好多少"

**KL 惩罚的作用**：
```
reward_total = reward_model(x,y) - β * KL(π || π_ref)
```
1. **防止奖励黑客**：没有 KL 约束，模型会找到 reward model 的漏洞，生成高 reward 但无意义的文本
2. **保持生成质量**：防止策略偏离参考模型太远，保持语言流畅性
3. **稳定训练**：限制每步更新幅度

**LLaMA-Factory PPO 实现的奖励模型切换**（`ppo_utils.py:replace_model()`）：
```python
def replace_model(model, target):
    # 在 "default" 和 "reward" 适配器之间切换
    model.pretrained_model.set_adapter(target)
    # 同时切换 value head 权重
    v_head_layer.weight.data = model.get_buffer(f"{target}_head_weight")
    v_head_layer.bias.data = model.get_buffer(f"{target}_head_bias")
```
这允许策略模型和奖励模型共享骨干网络，通过 LoRA 适配器切换实现双模型功能。

---

### Q4: Neat Packing 的 4D 注意力 mask 是怎么实现的？为什么标准 packing 有问题？

**标准 packing 的问题**：

将多个序列拼接成一个固定长度序列时：
```
[A1, A2, A3, B1, B2, B3, C1, C2, C3]
```
标准因果注意力会让 A3 能看到 B1、C1 等，造成**序列间交叉注意力污染**。

**Neat Packing 的解决方案**：

1. 每个子序列分配唯一的 attention mask ID：
```
attention_mask = [1, 1, 1, 2, 2, 2, 3, 3, 3]
```

2. `prepare_4d_attention_mask()` 将 2D mask 转为 4D 块对角因果 mask：
```
[[1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 1, 1, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 1]]
```

3. 贪心背包算法 `greedy_knapsack()` 优化打包效率，`PackingParams` 跟踪子序列边界

**效果**：训练效率提升（减少 padding 浪费），同时保证序列间无信息泄露。

---

### Q5: Flash Attention 的原理是什么？为什么能同时加速和省内存？

**传统 Attention 的问题**：

标准注意力：S = QK^T（N×N 矩阵），P = softmax(S)，O = PV

- **内存瓶颈**：需要存储 N×N 的注意力矩阵 S 和 P，内存 O(N²)
- **计算瓶颈**：S 和 P 需要从 GPU HBM（高带宽内存）读写多次

**Flash Attention 的核心思想**：IO-aware tiling

1. **分块计算**：将 Q、K、V 分成小块，每块大小由 SRAM（片上内存，~20MB）决定
2. **在 SRAM 中完成计算**：每个 Q 块与所有 K 块计算局部 softmax，不需要存储完整 N×N 矩阵
3. **Online Softmax**：通过维护 running max 和 running sum，增量更新 softmax，避免全局归一化
4. **重计算替代存储**：反向传播时重新计算注意力矩阵，而非存储

**结果**：
- 内存：O(N²) → O(N)（不需要存完整注意力矩阵）
- 速度：减少 HBM 访问次数，计算受限而非内存受限

---

### Q6: 解释 DFT Loss、ASFT Loss、EAFT Loss 的设计动机和实现

**标准 Cross-Entropy Loss 的问题**：

标准 CE loss 对所有 token 等权平均，但实践中：
- 简单 token（如常见词）的 loss 很低，贡献小
- 困难 token（如专业术语）的 loss 很高，主导梯度
- 这可能导致模型过度关注困难样本而忽略整体质量

**DFT Loss（Dynamic Fine-Tuning Loss）**（`trainer_utils.py:639`）：

```python
def _dft_cross_entropy(source, target, num_items_in_batch, ignore_index=-100):
    per_token_loss = F.cross_entropy(source, target, ignore_index=ignore_index, reduction="none")
    valid_losses = per_token_loss[valid_mask]
    with torch.no_grad():
        target_probs = torch.exp(-valid_losses)  # 模型对正确答案的置信度
    weighted_losses = valid_losses * target_probs  # 置信度高的 token 权重低
```

**设计动机**：模型已经学好的 token（置信度高）降低权重，让模型关注需要学习的 token。

**ASFT Loss（Adaptive SFT Loss）**（`trainer_utils.py:686`）：

```python
def asft_loss_func(outputs, labels, ref_logits, asft_alpha=0.1):
    dft_loss = _dft_cross_entropy(policy_logits, policy_labels)
    kl_loss = _kl_divergence(policy_logits, ref_logits, policy_labels)  # 与参考模型的 KL
    return dft_loss + asft_alpha * kl_loss
```

**设计动机**：SFT 容易过拟合和灾难性遗忘。添加与参考模型的 KL 散度约束，防止偏离太远。

**EAFT Loss（Entropy-Adaptive Fine-Tuning Loss）**（`trainer_utils.py:768`）：

```python
def _eaft_cross_entropy(source, target, num_items_in_batch, alpha=1.0):
    per_token_loss = F.cross_entropy(source, target, reduction="none")
    # 近似计算输出分布的熵
    topk_val, _ = torch.topk(source_detached, k=20, dim=-1)
    entropy_approx = -(probs_topk * log_probs_topk).sum(dim=-1)
    # 熵高（不确定）的 token 权重高
    adaptive_weight = torch.pow(entropy_approx / 3.0, alpha)
    weighted_losses = valid_losses * adaptive_weight
```

**设计动机**：模型在某些 token 上输出分布熵高（不确定），这些 token 更需要学习。自适应加权。

---

### Q7: 解释 GaLore 和 APOLLO 的原理，为什么能替代 Adam？

**Adam 的内存瓶颈**：

Adam 为每个参数维护两个状态：一阶动量 m 和二阶动量 v，内存占用 = 2 × 参数量 × 4 bytes。对于 7B 模型 bf16 训练，Adam 状态占 ~56GB。

**GaLore（Gradient Low-Rank Projection）**：

核心思想：梯度矩阵是低秩的，投影到低维子空间后再计算 Adam 更新。

```
g_proj = P^T * g * Q    (P, Q 是投影矩阵，rank=r)
m, v 在投影空间中维护（大小从 d×d 变为 d×r + r×d）
更新完再投影回来
```

**LLaMA-Factory 实现**（`trainer_utils.py:200`）：
```python
# 层级模式：每个参数独立优化器，通过 hook 触发
if finetuning_args.galore_layerwise:
    for param in trainable_params:
        param.register_post_accumulate_grad_hook(optimizer_hook)
    optimizer = DummyOptimizer(lr=training_args.learning_rate, optimizer_dict=optimizer_dict)
```

**局限**：层级 GaLore 不支持梯度累积（`gradient_accumulation_steps` 必须为 1）。

**APOLLO**：类似 GaLore 但使用 SVD/随机投影进行梯度压缩，支持更灵活的投影方式。

---

## 第二类：实验与方案验证能力

> **面试官意图**：看你怎么做实验、怎么评估、怎么证明方案有效。能讲清楚"为什么选这个方法"、"怎么对比"、"结果怎么分析"。

---

### Q8: 你怎么设计实验来验证 LoRA 的有效性？怎么选择 rank？

**实验设计方案**：

**第一步：基线对比**
```
实验组1：Full Fine-tuning（全参数微调）—— 上界
实验组2：LoRA rank=8
实验组3：LoRA rank=16
实验组4：LoRA rank=32
实验组5：LoRA rank=64
控制变量：相同数据、相同 epoch、相同学习率、相同数据划分
```

**评估指标**：
- **任务指标**：准确率/BLEU/ROUGE/人工评估
- **训练效率**：显存占用、训练速度（tokens/sec）、收敛步数
- **泛化能力**：验证集 vs 测试集表现差距（过拟合程度）

**关键细节（面试追问）**：
- 学习率需要分别调优：LoRA 通常用 1e-4，Full FT 用 2e-5
- 数据量小时 LoRA 更不容易过拟合
- 需要多次实验取均值，关注方差

**Rank 选择策略**：
1. 从 rank=16 开始，观察验证集 loss 曲线
2. 如果欠拟合（loss 下降慢），增大 rank
3. 如果过拟合（验证集 loss 上升），减小 rank 或增加 dropout
4. 通常 rank=16-32 是最优平衡点

---

### Q9: DPO 训练中怎么判断模型是否学到了偏好？监控哪些指标？

**关键监控指标**（LLaMA-Factory `dpo/trainer.py` 自动记录）：

```python
metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean()
metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean()
metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean()
```

**判断标准**：
1. **reward accuracy 应该上升**：chosen reward > rejected reward 的比例增加
2. **reward margin 应该增大**：chosen 和 rejected 的 reward 差距增大
3. **chosen logps 不应大幅下降**：如果 chosen logps 也下降，说明模型在"摆烂"
4. **rejected logps 应该下降**：模型学会了避免不好的回复

**常见问题排查**：
| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| 两个 reward 都上升 | 模型没有学到区分能力 | 检查数据质量，增大 beta |
| chosen reward 下降 | 过拟合或灾难性遗忘 | 减小学习率，增加 ftx_gamma |
| reward accuracy 不变 | 数据对太相似或 beta 太小 | 增大 beta，改进数据 |

---

### Q10: 怎么评估 SFT 后模型的质量？自动化指标够吗？

**自动化指标的局限**：

1. **BLEU/ROUGE**：只衡量 n-gram 重叠，不衡量语义正确性
2. **Perplexity**：低 PPL 不等于高质量回复
3. **Accuracy**：只适用于有标准答案的任务

**推荐的评估方案**：

1. **自动评估 + 人工评估结合**
   - 自动：BLEU、ROUGE、准确率（用于快速迭代）
   - 人工：流畅性、有用性、安全性（用于最终评估）

2. **LLM-as-Judge**
   - 用 GPT-4 等强模型对回复打分
   - 成本低于人工，质量高于纯自动指标

3. **对比评估**
   - 与基线模型做 A/B 测试
   - Win/Tie/Lose 比例

**LLaMA-Factory 的评估支持**（`sft/workflow.py`）：
```python
if training_args.predict_with_generate:
    metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
elif finetuning_args.compute_accuracy:
    metric_module["compute_metrics"] = ComputeAccuracy()
```

---

## 第三类：问题定位与排查能力

> **面试官意图**：看你遇到实际问题时怎么排查、怎么定位根因、怎么解决。这比"做了什么"更能体现真实能力。

---

### Q11: 模型微调后效果反而变差了，怎么排查？

**系统性排查流程**：

**第一步：检查数据**
```bash
# 查看数据格式是否正确
llamafactory-cli train --dataset your_data --dry_run
```
- 数据中是否存在噪声/错误标签？
- prompt/response 格式是否与 template 匹配？
- 数据量是否足够？（< 100 条可能不够）

**第二步：检查训练配置**
- 学习率是否过大？（LoRA 推荐 1e-4 ~ 5e-5）
- 是否过拟合？（观察 train loss vs eval loss）
- Template 是否正确？（训练和推理必须一致）

**第三步：检查模型加载**
- LoRA 是否正确加载？（检查 trainable params 数量）
- 量化是否正确？（QLoRA 的 cast_trainable_params_to_fp32）
- 特殊 token 是否正确添加？

**第四步：检查推理配置**
- 生成参数：temperature、top_p、top_k 是否合理
- 是否使用了正确的 template 进行推理
- 停止词是否正确设置

**LLaMA-Factory 排查工具**：
```python
# 打印所有参数的名称、dtype、device、trainable 状态
model_args.print_param_status = True
```

---

### Q12: 训练过程中 loss 突然飙升或变成 NaN，怎么排查？

**Loss 飙升的常见原因**：

| 原因 | 排查方法 | 解决方案 |
|------|----------|----------|
| 学习率过大 | 检查 lr 曲线 | 降低学习率或加 warmup |
| 梯度爆炸 | 检查 grad_norm | 设置 max_grad_norm（默认1.0） |
| 数据问题 | 检查 loss 飙升时的 batch | 清洗数据，检查异常样本 |
| 数值溢出 | 检查是否使用 fp16 无 loss scaling | 使用 bf16 或开启 loss scaling |

**NaN 的排查**：
1. **检查数据**：是否有空序列、超长序列、特殊字符
2. **检查量化**：QLoRA 中适配器必须为 fp32
3. **检查 DeepSpeed**：ZeRO-3 下检查参数分片是否正确
4. **梯度裁剪**：`max_grad_norm: 1.0`
5. **混合精度**：bf16 比 fp16 数值更稳定

**LLaMA-Factory 的防护措施**：
- `cast_trainable_params_to_fp32`：确保 QLoRA 适配器精度
- `patch_model()` 中的 `prepare_model_for_training()`：启用梯度检查点
- `neat_packing`：避免交叉注意力导致的数值问题

---

### Q13: 多 GPU 训练时 loss 与单 GPU 不一致，怎么排查？

**常见原因**：

1. **Batch size 不同**：多 GPU 总 batch size = per_device × GPU 数 × gradient_accumulation
   - 学习率需要按比例调整（线性缩放规则）

2. **数据划分问题**：
   - 不同 GPU 的数据分布不一致
   - `DistributedSampler` 的 shuffle seed 不同

3. **梯度同步问题**：
   - DDP 中梯度未正确 all-reduce
   - `find_unused_parameters` 设置不当

4. **随机性**：
   - 不同 GPU 的 dropout mask 不同
   - 数据增强的随机性

**排查步骤**：
```python
# 1. 固定所有随机种子
training_args.seed = 42
training_args.data_seed = 42

# 2. 关闭 shuffle
finetuning_args.disable_shuffling = True

# 3. 检查梯度是否一致
# 在单 GPU 和多 GPU 上分别运行 1 step，比较梯度
```

---

### Q14: 模型上线后推理速度突然变慢，怎么排查？

**排查清单**：

1. **硬件层面**
   - GPU 使用率是否正常？（`nvidia-smi`）
   - 是否有其他进程占用 GPU？
   - 内存是否 swap？（OOM 后的部分计算卸载到 CPU）

2. **模型层面**
   - 是否正确加载了量化模型？
   - 是否使用了正确的推理后端（HF/vLLM/SGLang）？
   - batch size 是否过大导致 OOM swap？

3. **数据层面**
   - 输入序列是否异常长？
   - 是否有特殊的 token 导致生成停不下来？（缺少停止词）

4. **系统层面**
   - 网络延迟（API 调用场景）
   - 日志写入 IO 瓶颈
   - 内存泄漏导致 GC 频繁

**LLaMA-Factory 推理优化**：
```yaml
infer_backend: vllm  # 使用 vLLM 加速
vllm_enforce_eager: true  # 如果遇到 CUDA graph 问题
```

---

## 第四类：工程落地能力

> **面试官意图**：看你是否有实际工程经验，理论可行的方案在工程上怎么落地，怎么保证系统稳定。

---

### Q15: 怎么把微调后的模型部署成生产级 API？

**LLaMA-Factory 部署方案**：

```bash
# 方案1：LLaMA-Factory 内置 API 服务器
API_PORT=8000 llamafactory-cli api examples/inference/qwen3.yaml infer_backend=vllm

# 方案2：vLLM 独立部署（推荐生产环境）
vllm serve your_merged_model --port 8000 --tensor-parallel-size 4
```

**生产级部署考虑**：

1. **模型导出与合并**
   ```bash
   # LoRA 合并到基础模型
   llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
   # 导出后可以独立部署，不依赖 PEFT
   ```

2. **推理引擎选择**
   | 引擎 | 优势 | 适用场景 |
   |------|------|----------|
   | HF Transformers | 简单、兼容性好 | 开发测试 |
   | vLLM | PagedAttention、高吞吐 | 生产环境 |
   | SGLang | RadixAttention、低延迟 | Agent 场景 |

3. **服务稳定性**
   - 请求超时设置
   - 并发限制（防止 GPU OOM）
   - 健康检查端点
   - 自动重启策略

4. **监控与告警**
   - 请求延迟 P50/P95/P99
   - GPU 利用率和显存
   - 生成质量指标（回复长度分布、停止原因）
   - 错误率

5. **数据回滚**
   - 保存模型版本和配置快照
   - A/B 测试或灰度发布
   - 快速回滚到上一版本的机制

---

### Q16: 训练 70B 模型，只有 4 张 A100-80G，怎么配置？

**内存分析**：
- 70B 模型 bf16 参数：~140GB
- 4 张 A100-80G 总显存：320GB
- 全参数微调需要 ~1200GB —— 不可行

**可行方案**：

**方案1：DeepSpeed ZeRO-3 + QLoRA**
```yaml
# qwen3_70b_qlora_sft.yaml
model_name_or_path: Qwen/Qwen3-70B
finetuning_type: lora
quantization_bit: 4
lora_rank: 16
deepspeed: examples/deepspeed/ds_z3_config.json
```
- 4-bit 量化基础模型：~35GB
- LoRA 参数：~200MB
- ZeRO-3 分片优化器/梯度：~40GB/GPU
- 总计：~80GB/GPU，刚好可行

**方案2：FSDP + QLoRA**
```yaml
# 使用 PyTorch 原生 FSDP
finetuning_type: lora
quantization_bit: 4
```

**关键配置**：
```yaml
# DeepSpeed ZeRO-3 配置
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},  # 优化器卸载到 CPU
        "offload_param": {"device": "cpu"}       # 参数卸载到 CPU
    }
}
```

**性能优化**：
- 梯度检查点：`gradient_checkpointing: true`
- Flash Attention：`flash_attn: fa2`
- 序列打包：`packing: true`（减少 padding 浪费）

---

### Q17: 数据量很大（百万级），怎么高效处理？

**LLaMA-Factory 的流式处理方案**：

```yaml
streaming: true
max_steps: 10000  # 流式模式必须指定 max_steps
```

**大数据处理策略**：

1. **预处理缓存**
   ```yaml
   tokenized_path: ./cached_tokenized_data
   # 第一次处理后缓存，后续直接加载
   ```

2. **数据混合策略**
   ```yaml
   dataset: dataset1,dataset2,dataset3
   mix_strategy: interleave_over  # 交叉采样，平衡数据源
   ```

3. **多进程数据加载**
   ```yaml
   dataloader_num_workers: 4
   prefetch_factor: 2
   ```

4. **分布式数据处理**
   - 使用 `datasets` 库的流式加载
   - 避免在主进程中预处理所有数据

---

### Q18: 怎么保证训练过程的可复现性？

**可复现性清单**：

```yaml
# 1. 固定随机种子
seed: 42
data_seed: 42

# 2. 关闭 shuffle（调试时）
disable_shuffling: true

# 3. 固定 CUDA 算法
# 在代码中设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 4. 记录完整配置
# LLaMA-Factory 自动保存 args 到 output_dir
```

**实验管理**：
- 使用 WandB/SwanLab 记录所有超参数和指标
- 保存 checkpoint 和配置快照
- 使用版本控制管理数据和代码

---

## 第五类：业务与场景理解

> **面试官意图**：看你是否理解技术方案的业务价值，能否在资源有限的情况下做出合理决策。

---

### Q19: LoRA vs Full FT vs 冻结微调，什么场景选什么？

**决策矩阵**：

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 数据量小 (< 1K) + 资源有限 | LoRA | 参数少，不容易过拟合 |
| 数据量中等 (1K-100K) | LoRA rank=32 | 平衡效率和效果 |
| 数据量大 (> 100K) + 充足资源 | Full FT | 最大化模型能力 |
| 只需适配输出格式 | 冻结微调 + 少量层 | 最快、最省 |
| 多任务/多租户 | LoRA（每个任务一个适配器） | 适配器热切换 |
| 边缘设备部署 | QLoRA 导出量化模型 | 最小化模型体积 |

**成本分析**：
| 方法 | 7B 模型显存 | 训练时间 (相对) | 存储 (相对) |
|------|------------|----------------|------------|
| Full FT bf16 | ~60GB | 1x | 1x |
| LoRA rank=16 | ~16GB | 0.3x | 0.01x |
| QLoRA 4-bit | ~6GB | 0.4x | 0.01x |
| Freeze (last 2层) | ~14GB | 0.2x | 0.005x |

---

### Q20: SFT 数据质量 vs 数量，怎么权衡？

**关键洞察**：数据质量远比数量重要。

**实验证据**：
- LIMA 论文用 1000 条高质量数据就能训出不错的模型
- Alpaca 用 52K 条 GPT-4 生成的数据效果好于 520K 条低质量数据

**数据质量维度**：

1. **多样性**：覆盖不同任务类型、难度级别、领域
2. **正确性**：答案准确、无矛盾
3. **格式一致性**：统一的对话格式
4. **指令复杂度**：适当的难度梯度

**LLaMA-Factory 数据配置**：
```yaml
# 使用多个数据集，控制采样
dataset: alpaca_en,code_alpaca_en,math_en
mix_strategy: interleave_over  # 平衡采样
num_samples: 10000  # 每个数据集最多采样 10K
```

**资源有限时的优先级**：
1. 首先确保数据质量（清洗、去重、格式化）
2. 然后增加多样性（覆盖更多任务类型）
3. 最后才是增加数量

---

### Q21: Agent/Tool-Calling 微调，关键挑战是什么？

**核心挑战**：

1. **工具选择准确性**：模型需要理解每个工具的功能和适用场景
2. **参数提取准确性**：从用户指令中正确提取工具调用参数
3. **多轮工具调用**：处理工具返回结果，决定是否继续调用
4. **错误处理**：工具调用失败时的恢复策略

**LLaMA-Factory 的 Agent 微调支持**：

```python
# 模板中的工具格式化
format_function=FunctionFormatter(slots=["<tool_call>\n{{content}}\n</tool_call>"])
format_observation=StringFormatter(slots=["<tool_response>\n{{content}}\n</tool_response>"])
format_tools=ToolFormatter(slots=["# Tools\n\n{{content}}\n"])
```

**训练数据格式**：
```json
{
    "conversations": [
        {"role": "user", "content": "今天北京天气怎么样？"},
        {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}\n</tool_call>"},
        {"role": "tool", "content": "{\"temperature\": 25, \"condition\": \"晴\"}"},
        {"role": "assistant", "content": "今天北京天气晴朗，温度25度。"}
    ]
}
```

**评估指标**：
- 工具选择准确率
- 参数提取准确率
- 端到端任务完成率
- 多轮调用成功率

---

### Q22: 如果资源只够优化一个环节，应该优化什么？

**优先级排序**（投入产出比从高到低）：

1. **数据质量**（最高 ROI）
   - 清洗噪声数据
   - 增加高质量样本
   - 改善数据多样性
   - 成本：人力时间，无额外计算资源

2. **Prompt/Template 优化**
   - 确保训练和推理 template 一致
   - 优化系统提示
   - 成本：几乎为零

3. **超参数调优**
   - 学习率搜索（1e-5 到 1e-4）
   - LoRA rank 调整
   - 成本：少量 GPU 时间

4. **训练策略**
   - 增加 epoch（如果欠拟合）
   - 添加正则化（如果过拟合）
   - 成本：中等 GPU 时间

5. **模型选择**（最低优先级）
   - 换更大的基础模型
   - 成本：大量 GPU 时间和资金

**面试回答模板**：
> "如果资源有限，我首先会优化数据质量，因为这是投入产出比最高的。具体来说：检查数据中的噪声和错误标签，增加高质量样本，改善数据多样性。这不需要额外计算资源，但往往能带来最大的效果提升。其次我会优化 prompt template 和超参数，这些成本很低但影响显著。最后才考虑换更大的模型。"

---

### Q23: 微调后的模型怎么评估其在真实场景中的表现？

**离线评估**（训练时）：
- 标准 benchmark（MMLU、C-Eval 等）
- 领域特定测试集
- 生成质量指标

**在线评估**（上线后）：
1. **A/B 测试**
   - 新旧模型同时服务
   - 对比用户满意度、任务完成率

2. **用户反馈收集**
   - 点赞/点踩机制
   - 用户主动标注

3. **自动监控**
   - 回复长度分布（异常短/长可能有问题）
   - 停止原因（正常 EOS vs 截断 vs 重复）
   - 延迟分布

4. **定期人工评估**
   - 抽样人工打分
   - 与标注团队合作

**关键指标**：
| 维度 | 指标 |
|------|------|
| 质量 | 流畅性、准确性、有用性 |
| 安全 | 有害内容比例、越狱成功率 |
| 效率 | 延迟、吞吐量、成本 |
| 业务 | 用户留存、任务转化率 |

---

## 附录：面试自我介绍与项目讲述框架

### STAR 法则讲述项目

**Situation（背景）**：
> 我在研究/实习中负责 XX 领域的 LLM 微调工作。

**Task（任务）**：
> 目标是将通用 LLM 适配为 XX 领域的专业模型，需要在 XX 数据上训练，部署到 XX 场景。

**Action（行动）**：
> 我使用 LLaMA-Factory 框架，具体做了：
> 1. 数据方面：XX（数据清洗/格式化/增强）
> 2. 训练方面：XX（选择了 XX 方法，因为 XX）
> 3. 优化方面：XX（解决了 XX 问题）

**Result（结果）**：
> 最终模型在 XX 指标上提升了 XX%，部署后 XX 业务指标改善了 XX%。

### 讲述问题解决的模板

> **遇到的问题**：训练后模型在 XX 场景下表现不佳。
>
> **排查过程**：我首先检查了数据质量，发现 XX；然后检查了训练配置，发现 XX；最后通过 XX 实验定位到根因。
>
> **解决方案**：我采取了 XX 方法，具体是 XX。
>
> **效果验证**：通过 XX 实验证明方案有效，XX 指标提升了 XX%。

---

*本文档基于 LLaMA-Factory 源码深度分析编写，覆盖 LLM 算法实习面试的五类核心问题。*
*最后更新：2026-06-27*
