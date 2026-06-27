# 增量学习笔记
## 对比前两轮文档的新发现与深层理解

---

## 一、前两轮文档覆盖范围回顾

| 轮次 | 文档 | 覆盖内容 |
|------|------|----------|
| 第一轮 | `LLM_Interview_Preparation.md` | 架构概览、核心模块、关键算法原理、面试问答 |
| 第二轮 | `Interview_Question_Response_Guide.md` | 五类面试问题应对：原理深挖、实验验证、问题排查、工程落地、业务理解 |

**本轮增量**：以下内容是前两轮未涉及或仅浅层提及，通过实际阅读源码和制定复现计划后新发现的深层细节。

---

## 二、数据处理系统的深层细节

### 2.1 数据验证与容错机制

**发现位置**：`data/processor/supervised.py:113-117`

```python
for i in range(len(examples["_prompt"])):
    if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
        logger.warning_rank0(
            "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
        )
        continue
```

**关键发现**：
- SFT 数据处理器会**静默丢弃**格式不正确的样本
- prompt 必须是奇数条消息（user/assistant 交替，以 user 结尾）
- response 必须恰好 1 条
- 这意味着如果你的数据格式不对，不会报错，只是数据量变少

**面试深挖点**：
> "数据训练后效果不好，除了检查数据质量，还要检查数据是否被静默丢弃了。可以通过日志中的 warning 信息确认。"

### 2.2 mask_history 的实现细节

**发现位置**：`data/processor/supervised.py:69-100`

```python
if self.data_args.mask_history:
    encoded_pairs = encoded_pairs[::-1]  # 反转！优先保留最后一轮
```

**关键发现**：
- `mask_history=True` 时，对话轮次被**反转**处理
- 从最后一轮开始向前填充，直到达到 `cutoff_len`
- 只有最后一轮的 response 计算 loss，历史轮次的 response 被 mask 为 `IGNORE_INDEX`
- 这是一种"优先保留最新对话上下文"的策略

**与前两轮的区别**：前两轮只说了"只训练最后一轮"，没有说清楚反转填充的策略。

### 2.3 efficient_eos 的特殊处理

**发现位置**：`data/processor/supervised.py:85-86, 102-104`

```python
elif self.template.efficient_eos and turn_idx != 0:
    source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)

if self.template.efficient_eos:
    input_ids += [self.tokenizer.eos_token_id]
    labels += [self.tokenizer.eos_token_id]
```

**关键发现**：
- `efficient_eos` 模式下，每轮对话的开头会添加 EOS token 作为分隔符
- 这个 EOS token 在 label 中也被标记为需要预测
- 最终序列末尾也会添加 EOS token
- 这让模型学习到"轮次边界"的概念

### 2.4 infer_seqlen 的长度推断

**发现位置**：`data/processor/processor_utils.py`

```python
def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
```

**关键发现**：
- 当总长度超过 cutoff_len 时，会智能截断
- 优先保留 target（response），截断 source（prompt）
- 这确保了模型总能学习到完整的回复

---

## 三、模型加载与补丁系统的深层细节

### 3.1 模型类型自动检测逻辑

**发现位置**：`model/loader.py:159-166`

```python
if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
    load_class = AutoModelForImageTextToText
elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
    load_class = AutoModelForSeq2SeqLM
elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio-text for qwen omni
    load_class = AutoModelForTextToWaveform
else:
    load_class = AutoModelForCausalLM  # text-only
```

**关键发现**：
- 模型加载类不是根据模型名称判断，而是根据 `config` 类型在 `_model_mapping` 中的注册
- Qwen3-Omni 等特殊模型需要额外处理：`model = getattr(model, "thinker")`
- 这意味着添加新模型支持时，需要确保 transformers 库中有正确的映射

### 3.2 Qwen3.5 的 Forward Patching

**发现位置**：`model/patcher.py:141-150`

```python
def patch_qwen3_5_forward_gpu(model: "PreTrainedModel") -> None:
    """Patch the forward method of Qwen3_5ForConditionalGeneration to support cu_seqlens input only patch when do training.
```

**关键发现**：
- Qwen3.5 使用了线性注意力（Gated Delta Net），不是标准 Transformer
- 需要特殊的 forward patch 来支持变长序列（varlen attention）
- 依赖 `flash-linear-attention` 库的 `causal_conv1d` 和 `chunk_gated_delta_rule`
- NPU 环境还需要替换为 `triton_ascend` 兼容的实现

**面试深挖点**：
> "LLaMA-Factory 不仅支持标准 Transformer，还支持线性注意力等新架构。这些架构需要特殊的 patch 来保证训练正确性。"

### 3.3 Conv3D 的 PyTorch 版本兼容性检查

**发现位置**：`model/loader.py:197-205`

```python
if is_torch_version_greater_than("2.9.0") and not is_torch_version_greater_than("2.10.0"):
    if any(isinstance(m, torch.nn.Conv3d) for m in model.modules()):
        raise ValueError(
            "Unsupported torch version detected: torch 2.9.x with Conv3D. "
            "This combination is known to cause severe performance regression."
        )
```

**关键发现**：
- PyTorch 2.9.x 与 Conv3D 存在已知的严重性能回退
- LLaMA-Factory 在加载模型时会主动检测并拒绝
- 这种版本兼容性检查在实际工程中非常重要

---

## 四、训练系统的深层细节

### 4.1 DummyOptimizer 的设计巧妙之处

**发现位置**：`train/trainer_utils.py:68-84`

```python
class DummyOptimizer(torch.optim.Optimizer):
    r"""A dummy optimizer used for the GaLore or APOLLO algorithm."""
    def __init__(self, lr=1e-3, optimizer_dict=None):
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": lr})

    def zero_grad(self, set_to_none=True): pass
    def step(self, closure=None): pass
```

**关键发现**：
- GaLore/APOLLO 的层级模式下，每个参数有自己的优化器
- 但 HuggingFace Trainer 期望一个统一的优化器
- DummyOptimizer 是一个"空壳"，实际优化通过 `register_post_accumulate_grad_hook` 钩子完成
- 这是一种巧妙的"欺骗" Trainer 的设计模式

**面试深挖点**：
> "GaLore 的层级模式不能与梯度累积一起使用，因为 `post_accumulate_grad_hook` 在每个 micro-batch 后触发，而不是在累积完成后。"

### 4.2 ASFT Loss 的参考模型前向传播

**发现位置**：`train/sft/trainer.py:151-162`

```python
def compute_loss(self, model, inputs, *args, **kwargs):
    if self.finetuning_args.use_asft_loss:
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )
            ref_logits = ref_outputs.logits
        outputs = model(**inputs)
        return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
```

**关键发现**：
- ASFT 训练时，每个 batch 都需要参考模型做一次前向传播
- 这意味着显存需要同时容纳策略模型和参考模型
- 参考模型使用 `torch.no_grad()` 避免梯度计算
- 如果使用 LoRA，参考模型可以通过 `disable_adapter()` 实现，不需要额外加载

### 4.3 FP8 训练的配置流程

**发现位置**：`train/sft/trainer.py:62-65`

```python
if training_args.fp8:
    configure_fp8_environment(training_args)
    if getattr(training_args, "fp8_backend", "auto") == "te":
        patch_accelerator_for_fp8()
```

**关键发现**：
- FP8 训练需要在 Trainer 初始化前配置环境
- 支持多种 FP8 后端：torchao、Transformer Engine (TE)、msamp
- RTX 4090 支持 FP8（通过 Transformer Engine），可以进一步减少显存

### 4.4 BAdam 的两种模式

**发现位置**：`train/trainer_utils.py:412-470`

**关键发现**：
- `badam_mode: "layer"`：层级模式，每次只更新一层参数，按 `switch_interval` 轮换
- `badam_mode: "ratio"`：比例模式，每次更新固定比例的参数
- 层级模式的 `switch_mode` 支持 `ascending`（从底层到顶层）和 `descending`（从顶层到底层）
- 层级模式下需要 `find_unused_parameters=True`（DDP）

---

## 五、DPO 训练的深层细节

### 5.1 训练时禁用 Dropout

**发现位置**：`train/dpo/trainer.py:57-60`

```python
if disable_dropout:
    disable_dropout_in_model(model)
    if ref_model is not None:
        disable_dropout_in_model(ref_model)
```

**关键发现**：
- DPO 训练默认禁用所有 Dropout
- 原因：Dropout 引入的随机性会导致 chosen/rejected 的 log prob 估计有噪声
- 这在标准 SFT 中不需要，但在偏好学习中很重要

### 5.2 参考模型的多种准备方式

**发现位置**：`train/dpo/trainer.py:94-109`

```python
if ref_model is not None:
    if self.is_deepspeed_enabled:
        self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
    elif self.is_fsdp_enabled:
        if self.accelerator.is_fsdp2:
            self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
        else:
            self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
    else:
        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
```

**关键发现**：
- 参考模型的准备方式取决于分布式策略
- DeepSpeed、FSDP、FSDP2、普通 DDP 各有不同的准备方法
- LoRA 训练时 ref_model=None，通过 `disable_adapter()` 实现

### 5.3 BCO 的运行均值基线

**发现位置**：`train/dpo/trainer.py:168-185`

```python
def bco_loss(self, chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps):
    chosen_rewards = self.beta * chosen_logratios
    rejected_rewards = self.beta * rejected_logratios
    rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
    self.running.update(rewards)  # 动态更新基线
    delta = self.running.mean
```

**关键发现**：
- BCO 使用 `RunningMoments` 动态计算奖励的运行均值作为基线
- 基线会随训练进程自适应调整
- 这比固定基线更稳定，但引入了训练过程中的非平稳性

---

## 六、PPO 训练的深层细节

### 6.1 奖励模型的三种类型

**发现位置**：`train/trainer_utils.py:151-190`

| 类型 | `reward_model_type` | 实现方式 | 显存需求 |
|------|---------------------|----------|----------|
| API 服务器 | `"api"` | HTTP 请求外部服务 | 无额外显存 |
| LoRA 共享骨干 | `"lora"` | 策略模型加载 RM 适配器 | 最小 |
| 独立完整模型 | `"full"` | 单独加载 RM 模型 | 最大 |

**LoRA 共享骨干的实现细节**：
```python
model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
model.register_buffer("reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False)
model.register_buffer("reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False)
```
- 策略模型和奖励模型共享同一个基础模型
- 通过 LoRA 适配器切换（`set_adapter("default")` vs `set_adapter("reward")`）
- Value Head 权重存储为 buffer，切换时复制

### 6.2 dump_layernorm / restore_layernorm

**发现位置**：`train/ppo/ppo_utils.py:65-80`

```python
def dump_layernorm(model):
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:
            layer_norm_params[name] = param.data.detach().clone()
            param.data = param.data.to(model.config.torch_dtype)
    return layer_norm_params
```

**关键发现**：
- PPO 训练中，LayerNorm 参数保持 fp32 精度
- 在某些操作前需要将它们临时转为模型 dtype（如 bf16）
- 操作后需要恢复 fp32 精度
- 这是为了避免 DeepSpeed ZeRO-3 下的精度问题

---

## 七、Collator 的深层细节

### 7.1 Dummy Image 机制

**发现位置**：`data/collator.py:341-349`

```python
if (
    self.template.mm_plugin.image_token is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
):
    fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
    fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
```

**关键发现**：
- 当 batch 中没有任何图像输入时，会注入一个假图像（64×64 白色）
- 目的：避免 DeepSpeed ZeRO-3 / FSDP 下多进程不同步导致的 hang
- 假图像只在 batch[0] 中注入，其他样本不需要

### 7.2 MRoPE 的 Packing 支持

**发现位置**：`data/collator.py:204-322`

**关键发现**：
- Qwen2-VL 使用 MRoPE（多分辨率位置编码），position_ids 是 3 维的 `(3, batch, seq_len)`
- Packing 场景下，每个子序列需要独立计算 MRoPE position_ids
- 计算完成后需要验证 `position_ids.shape` 是否正确
- 这是 VLM + Packing 组合的复杂工程实现

---

## 八、优化器系统的深层细节

### 8.1 Muon 优化器的参数分组策略

**发现位置**：`train/trainer_utils.py:498-524`

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
```

**关键发现**：
- Muon 优化器只用于 2D 参数（矩阵），不用于 embedding 和 lm_head
- 1D 参数（bias, LayerNorm）使用标准 AdamW
- 这是因为 Muon 的正交化操作只对矩阵有意义

### 8.2 LoRA+ 的参数分组

**发现位置**：`train/trainer_utils.py:370-409`

```python
param_groups = [
    dict(params=param_dict["lora_a"], lr=default_lr, ...),
    dict(params=param_dict["lora_b"], lr=loraplus_lr, ...),
    dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
    dict(params=param_dict["embedding"], lr=embedding_lr, ...),
]
```

**关键发现**：
- LoRA 的 A 矩阵和 B 矩阵使用不同学习率
- B 矩阵的学习率 = A 矩阵的学习率 × `loraplus_lr_ratio`
- B 矩阵中的 bias 参数不使用权重衰减
- Embedding 参数有独立的学习率

### 8.3 warmup_stable_decay 调度器

**发现位置**：`train/trainer_utils.py:556-570`

```python
if training_args.lr_scheduler_type == "warmup_stable_decay":
    num_warmup_steps = training_args.get_warmup_steps(num_training_steps)
    remaining_steps = num_training_steps - num_warmup_steps
    num_stable_steps = remaining_steps // 3  # 默认 1/3 用于稳定期
    num_decay_steps = remaining_steps - num_stable_steps
```

**关键发现**：
- 除了标准的 cosine/linear 调度器，还支持 warmup-stable-decay 三阶段调度
- 稳定期占总步数的 1/3
- 这种调度器在长训练中更稳定

---

## 九、配置系统的深层细节

### 9.1 YAML 配置的覆盖机制

**发现位置**：`hparams/parser.py` 的 `read_args()`

**关键发现**：
- YAML 配置文件中的值可以被命令行参数覆盖
- 使用 OmegaConf 进行合并
- 命令行参数格式：`key=value`（不是 `--key value`）
- 例如：`llamafactory-cli train config.yaml learning_rate=5e-5`

### 9.2 参数验证的交叉约束

**发现位置**：`hparams/parser.py` 的 `get_train_args()`

**关键发现的验证逻辑**：
- `quantization_bit` 不为 None 时，`finetuning_type` 必须是 `lora` 或 `oft`
- `use_galore` 和 `use_apollo` 不能同时启用
- `packing` 和 `neat_packing` 需要 Flash Attention 支持
- `use_badam` 时 `gradient_accumulation_steps` 必须为 1（层级模式）
- PPO 训练时 `reward_model` 必须指定

---

## 十、工程实践中的关键发现

### 10.1 梯度累积与 loss 计算的陷阱

**发现位置**：`train/sft/trainer.py:69-71`

```python
if processor is not None:
    self.model_accepts_loss_kwargs = False  # avoid wrong loss under gradient accumulation
```

**关键发现**：
- 当使用 processor（多模态模型）时，需要禁用 `model_accepts_loss_kwargs`
- 否则在梯度累积时 loss 计算会出错
- 这是 transformers 库的一个已知 bug（PR #36044）

### 10.2 推理时 prompt 的清除

**发现位置**：`train/sft/trainer.py:185-188`

```python
if generated_tokens is not None and self.args.predict_with_generate:
    generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
```

**关键发现**：
- `predict_with_generate` 时，生成的 token 包含 prompt 部分
- 需要将 prompt 部分替换为 pad_token_id，只保留生成部分
- 这确保了评估指标只计算生成内容

### 10.3 export_model 的 dtype 转换策略

**发现位置**：`train/tuner.py:180-191`

```python
if getattr(model, "quantization_method", None) is not None:
    setattr(model.config, "torch_dtype", torch.float16)
else:
    if model_args.infer_dtype == "auto":
        output_dtype = getattr(model.config, "torch_dtype", torch.float32)
        if output_dtype == torch.float32:
            output_dtype = infer_optim_dtype(torch.bfloat16)
```

**关键发现**：
- 量化模型导出时使用 float16（不是 bf16）
- 非量化模型自动推断最优 dtype：优先 bf16，回退到 float32
- `infer_optim_dtype()` 检查当前 GPU 是否支持 bf16

---

## 十一、与前两轮的关键差异总结

| 维度 | 前两轮覆盖 | 本轮新增 |
|------|-----------|---------|
| **数据处理** | 格式、模板、label masking | 静默丢弃验证、mask_history 反转策略、efficient_eos、infer_seqlen |
| **模型加载** | 加载流程、适配器初始化 | 类型自动检测、Qwen3.5 patch、Conv3D 兼容性检查 |
| **DPO 训练** | 损失函数数学推导 | 禁用 Dropout 原因、参考模型多后端准备、BCO 运行均值 |
| **PPO 训练** | Actor-Critic 架构、KL 惩罚 | 三种 RM 类型、dump_layernorm、LoRA 共享骨干实现 |
| **优化器** | GaLore/APOLLO 原理 | DummyOptimizer hook 机制、Muon 参数分组、LoRA+ 分组策略 |
| **Collator** | Neat Packing 4D mask | Dummy Image 机制、MRoPE Packing 支持 |
| **配置系统** | 5 个数据类、解析流程 | YAML 覆盖机制、交叉约束验证 |
| **工程细节** | 内存优化、速度优化 | 梯度累积 loss 陷阱、推理 prompt 清除、dtype 转换策略 |

---

*本笔记记录了通过制定复现计划和深入源码后，对比前两轮文档的增量学习内容。*
*最后更新：2026-06-27*
