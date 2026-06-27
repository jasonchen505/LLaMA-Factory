# LLaMA-Factory 全流程复现计划
## 硬件环境：8 × RTX 4090 (24GB) = 192GB 总显存

---

## 硬件资源评估

### RTX 4090 关键参数

| 参数 | 值 |
|------|-----|
| 显存 | 24GB GDDR6X |
| FP32 算力 | 82.6 TFLOPS |
| BF16 算力 | 165.2 TFLOPS（RTX 4090 不原生支持 BF16，需确认 CUDA 版本） |
| 互联 | PCIe 4.0 ×16（无 NVLink） |
| 卡数 | 8 张 |
| 总显存 | 192GB |

### 关键限制

1. **无 NVLink**：GPU 间通信走 PCIe，带宽 ~32GB/s（vs NVLink 600GB/s），DDP 通信开销较大
2. **24GB 显存**：单卡无法加载 7B bf16 模型（需 ~14GB 参数 + 优化器状态）
3. **BF16 支持**：RTX 4090 支持 BF16 但需确认 PyTorch 版本

### 各模型规模的可行性分析

| 模型规模 | Full FT | LoRA (bf16) | QLoRA (4-bit) | 推荐方案 |
|----------|---------|-------------|---------------|----------|
| **1.5B** | 8×4090 可行 | 单卡可行 | 单卡可行 | LoRA 单卡 |
| **4B** | 8×4090 + ZeRO-2 | 单卡可行 | 单卡可行 | LoRA 单卡或 DDP |
| **7B** | 8×4090 + ZeRO-3 | 2 卡 DDP 可行 | 单卡可行 | Full FT + ZeRO-3 |
| **14B** | 8×4090 + ZeRO-3 + Offload | 4 卡 DDP 可行 | 2 卡可行 | LoRA 4卡 或 QLoRA 2卡 |
| **32B** | 不可行 | 8 卡 + ZeRO-2 可行 | 4 卡可行 | QLoRA + ZeRO-3 |
| **70B** | 不可行 | 需 ZeRO-3 + Offload | 8 卡 + ZeRO-3 可行 | QLoRA + ZeRO-3 |

---

## 全流程复现计划

### Phase 0: 环境搭建与验证（Day 1）

**目标**：确保框架正常运行，理解 CLI 和配置系统。

```bash
# 1. 安装 LLaMA-Factory
cd /data/home/yizhou/LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
pip install -r requirements/deepspeed.txt

# 2. 验证安装
llamafactory-cli version
llamafactory-cli env

# 3. 验证 GPU 识别
python -c "import torch; print(torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_mem/1024**3:.1f}GB') for i in range(torch.cuda.device_count())]"

# 4. 测试 Web UI（可选）
llamafactory-cli webui
```

**学习目标**：
- 理解 `cli.py` → `launcher.py` 的命令分发机制
- 理解 `hparams/parser.py` 的配置解析流程
- 熟悉 YAML 配置文件的结构

---

### Phase 1: 数据准备与理解（Day 1-2）

**目标**：理解数据格式、模板系统、数据处理流程。

#### 1.1 内置数据集探索

```bash
# 查看内置数据集
cat data/dataset_info.json | python -m json.tool | head -100

# 查看不同格式的数据
cat data/alpaca_en_demo.json | head -20       # Alpaca 格式 SFT
cat data/dpo_en_demo.json | head -20          # DPO 偏好数据
cat data/kto_en_demo.json | head -20          # KTO 二元反馈
cat data/glaive_toolcall_en_demo.json | head -20  # Agent 工具调用
```

#### 1.2 自定义数据集制作

创建一个小型自定义数据集用于后续实验：

```json
// data/custom_sft_demo.json
[
  {
    "instruction": "请用一句话解释什么是LoRA",
    "input": "",
    "output": "LoRA是一种参数高效微调方法，通过在冻结的预训练权重旁边添加低秩分解矩阵来实现高效的下游任务适配。"
  }
]
```

在 `data/dataset_info.json` 中注册：
```json
"custom_sft_demo": {
  "file_name": "custom_sft_demo.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

#### 1.3 模板系统验证

```bash
# 验证不同模板的编码效果
python -c "
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.model import load_tokenizer
from llamafactory.hparams import ModelArguments

model_args = ModelArguments(model_name_or_path='Qwen/Qwen3-4B-Instruct-2507')
tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module['tokenizer']

from llamafactory.hparams import DataArguments
data_args = DataArguments(template='qwen3_nothink')
template = get_template_and_fix_tokenizer(tokenizer, data_args)

messages = [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi there!'}
]
prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages)
print(f'Prompt tokens: {tokenizer.decode(prompt_ids)}')
print(f'Response tokens: {tokenizer.decode(response_ids)}')
"
```

**学习目标**：
- 理解 Alpaca / ShareGPT 两种数据格式
- 理解 `template.py` 中 `TEMPLATES` 注册机制
- 理解 `processor/supervised.py` 中的 label masking 逻辑
- 理解 `collator.py` 中的数据批处理流程

---

### Phase 2: SFT 基础实验（Day 2-4）

**目标**：掌握 SFT 全流程，对比不同配置的效果。

#### 2.1 单卡 LoRA SFT（最基础）

```bash
# 使用内置示例配置
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

**配置详解**（`qwen3_lora_sft.yaml`）：
```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507  # 基础模型
stage: sft                                         # 训练阶段
finetuning_type: lora                              # 微调方式
lora_rank: 8                                       # LoRA 秩
lora_target: all                                   # 目标所有线性层
template: qwen3_nothink                            # 对话模板
cutoff_len: 2048                                   # 最大序列长度
per_device_train_batch_size: 1                     # 单卡 batch size
gradient_accumulation_steps: 8                     # 梯度累积
learning_rate: 1.0e-4                              # 学习率
bf16: true                                         # 混合精度
```

**显存占用估算**（4B 模型 LoRA rank=8）：
- 模型参数：~8GB (bf16)
- LoRA 参数：~50MB
- 优化器状态：~100MB
- 激活值：~2-4GB
- 总计：~12GB，单卡 4090 可行

#### 2.2 多卡 DDP SFT（提升速度）

```bash
# 自动使用 torchrun 启动 8 卡 DDP
NPROC_PER_NODE=8 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
# 或指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

**配置调整**：
```yaml
per_device_train_batch_size: 2   # 8卡可增大单卡 batch
gradient_accumulation_steps: 4    # 有效 batch = 2 * 4 * 8 = 64
```

#### 2.3 7B 全参数微调（DeepSpeed ZeRO-3）

```bash
# 7B 全参数微调需要 DeepSpeed ZeRO-3
cat > examples/train_full/qwen3_7b_full_sft_8x4090.yaml << 'EOF'
model_name_or_path: Qwen/Qwen3-8B
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
max_samples: 1000
preprocessing_num_workers: 16
dataloader_num_workers: 4

output_dir: saves/qwen3-8b/full/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
gradient_checkpointing: true
EOF

llamafactory-cli train examples/train_full/qwen3_7b_full_sft_8x4090.yaml
```

**显存分析**（8B 模型 Full FT + ZeRO-3）：
- 模型参数分片：~2GB/GPU (bf16, 8卡分片)
- 优化器状态分片：~6GB/GPU (fp32 m+v, 8卡分片)
- 梯度分片：~2GB/GPU
- 激活值（梯度检查点）：~4-6GB/GPU
- 总计：~16GB/GPU，8×4090 可行

#### 2.4 14B 模型 QLoRA

```bash
cat > examples/train_qlora/qwen3_14b_qlora_sft.yaml << 'EOF'
model_name_or_path: Qwen/Qwen3-14B
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
lora_rank: 16
lora_target: all

dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
preprocessing_num_workers: 16

output_dir: saves/qwen3-14b/qlora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
gradient_checkpointing: true
EOF

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_qlora/qwen3_14b_qlora_sft.yaml
```

#### 2.5 推理与评估

```bash
# HF 推理
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml

# vLLM 推理（更快）
cat > examples/inference/qwen3_lora_sft_vllm.yaml << 'EOF'
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-4b/lora/sft
template: qwen3_nothink
infer_backend: vllm
trust_remote_code: true
EOF

llamafactory-cli chat examples/inference/qwen3_lora_sft_vllm.yaml

# LoRA 合并导出
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

**学习目标**：
- 理解 LoRA vs Full FT vs QLoRA 的显存/速度/效果权衡
- 理解 DeepSpeed ZeRO 各阶段的工作原理
- 理解梯度检查点的原理和效果
- 理解 `model/loader.py` → `adapter.py` 的模型加载和适配器初始化流程

---

### Phase 3: 偏好对齐实验（Day 4-6）

**目标**：掌握 DPO/KTO 等偏好学习方法。

#### 3.1 DPO 训练

```bash
# DPO 训练
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml
```

**配置详解**：
```yaml
stage: dpo
pref_beta: 0.1          # 温度参数
pref_loss: sigmoid       # DPO 损失类型
# 其他可选: orpo, simpo, ipo, hinge, kto_pair
```

**对比实验**：
```bash
# DPO vs ORPO vs SimPO
for loss in sigmoid orpo simpo; do
    sed "s/pref_loss: sigmoid/pref_loss: ${loss}/" examples/train_lora/qwen3_lora_dpo.yaml > /tmp/dpo_${loss}.yaml
    llamafactory-cli train /tmp/dpo_${loss}.yaml
done
```

#### 3.2 KTO 训练（无需配对数据）

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_kto.yaml
```

**KTO vs DPO 的数据格式差异**：
- DPO：需要 chosen/rejected 配对
- KTO：只需 binary label（true/false）

#### 3.3 奖励模型训练

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_reward.yaml
```

**奖励模型的作用**：
- 训练一个能对回复质量打分的模型
- 为 PPO 训练提供奖励信号
- 可用于推理时的 Best-of-N 采样

**学习目标**：
- 理解 DPO 的数学推导和实现细节（`train/dpo/trainer.py`）
- 理解不同偏好损失的适用场景
- 理解奖励模型的训练和使用方式

---

### Phase 4: PPO/RLHF 全流程（Day 6-8）

**目标**：完整走通 RLHF 流程（SFT → RM → PPO）。

#### 4.1 PPO 训练配置

```bash
cat > examples/train_lora/qwen3_lora_ppo.yaml << 'EOF'
### model
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: ppo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### reward model
reward_model: saves/qwen3-4b/lora/reward  # 使用 Phase 3 训练的 RM
reward_model_type: lora                     # LoRA 类型的 RM

### dataset
dataset: kto_en_demo  # PPO 使用 prompt 数据
template: qwen3_nothink
cutoff_len: 2048

### output
output_dir: saves/qwen3-4b/lora/ppo
logging_steps: 10
save_steps: 500
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 1.0
bf16: true
EOF

llamafactory-cli train examples/train_lora/qwen3_lora_ppo.yaml
```

**PPO 训练的资源需求**：
- 需要同时加载：策略模型 + 参考模型 + 奖励模型
- 使用 LoRA 共享骨干网络可以减少显存
- 8×4090 训练 4B 模型 PPO 可行

#### 4.2 完整 RLHF 流程图

```
Step 1: SFT                Step 2: RM                Step 3: PPO
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│  Base Model  │          │  SFT Model   │          │  Policy Model│
│  + SFT Data  │───────>  │  + Pref Data │───────>  │  + RM        │
│  = SFT Model │          │  = RM Model  │          │  + Ref Model │
└──────────────┘          └──────────────┘          │  + Prompt Data│
                                                    │  = Aligned   │
                                                    └──────────────┘
```

**学习目标**：
- 理解 PPO 训练中的 Actor-Critic 架构
- 理解 `train/ppo/trainer.py` 中的 `ppo_train()` 循环
- 理解 `ppo_utils.py` 中的 `replace_model()` 奖励模型切换
- 理解 KL 惩罚和 Value Head 的作用

---

### Phase 5: Agent/Tool-Calling 微调（Day 8-9）

**目标**：掌握工具调用能力的微调。

#### 5.1 Agent 数据格式

```bash
# 查看工具调用数据格式
cat data/glaive_toolcall_en_demo.json | python -m json.tool | head -50
```

**数据格式关键点**：
- `function_call` 角色：模型生成的工具调用
- `observation` 角色：工具返回的结果
- `tools` 字段：可用工具的定义

#### 5.2 Agent SFT 训练

```bash
cat > examples/train_lora/qwen3_lora_agent.yaml << 'EOF'
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all

dataset: glaive_toolcall_en_demo
template: qwen3_nothink
cutoff_len: 4096   # Agent 数据通常更长
max_samples: 5000

output_dir: saves/qwen3-4b/lora/agent
logging_steps: 10
save_steps: 500
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 3.0
bf16: true
EOF

llamafactory-cli train examples/train_lora/qwen3_lora_agent.yaml
```

**学习目标**：
- 理解 `data/tool_utils.py` 中的工具调用解析
- 理解模板中 `format_function` / `format_tools` / `format_observation` 的作用
- 理解 Agent 微调与普通 SFT 的区别

---

### Phase 6: 多模态微调（Day 9-10）

**目标**：掌握视觉语言模型的微调。

#### 6.1 VLM LoRA SFT

```bash
# 使用 Qwen2-VL 或 Qwen3-VL
llamafactory-cli train examples/train_lora/qwen3vl_lora_sft.yaml
```

**多模态训练的关键差异**：
- 数据包含图像/视频
- 使用 `AutoModelForImageTextToText` 加载模型
- `mm_plugin` 处理多模态输入
- 需要更长的序列长度（图像 token 很多）

**显存考虑**：
- 图像 token 数量大（一张图可能 500+ tokens）
- 需要减小 batch size 或增大梯度累积
- 4B VLM 模型 LoRA：~16-20GB/GPU

**学习目标**：
- 理解 `data/mm_plugin.py` 的多模态数据处理
- 理解视觉编码器和语言模型的连接方式
- 理解 MRoPE（Qwen2-VL 的多分辨率位置编码）

---

### Phase 7: 高级优化技术实验（Day 10-12）

**目标**：掌握各种高效训练和推理优化技术。

#### 7.1 序列打包 (Packing)

```bash
cat > examples/train_lora/qwen3_lora_sft_packed.yaml << 'EOF'
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: sft
finetuning_type: lora
lora_rank: 8
lora_target: all

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
packing: true           # 启用序列打包
neat_packing: true      # 无交叉注意力打包

output_dir: saves/qwen3-4b/lora/sft_packed
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
report_to: none
EOF

llamafactory-cli train examples/train_lora/qwen3_lora_sft_packed.yaml
```

#### 7.2 不同优化器对比

```bash
# GaLore 优化器
llamafactory-cli train examples/extras/galore/llama3_full_sft.yaml

# Adam-mini
llamafactory-cli train examples/extras/adam_mini/qwen2_full_sft.yaml

# LoRA+（不同学习率）
llamafactory-cli train examples/extras/loraplus/llama3_lora_sft.yaml

# OFT
llamafactory-cli train examples/extras/oft/llama3_oft_sft.yaml
```

#### 7.3 不同损失函数对比

```bash
# DFT Loss
llamafactory-cli train examples/extras/dft/qwen2_full_sft.yaml

# ASFT Loss
llamafactory-cli train examples/extras/asft/qwen2_full_asft.yaml

# EAFT Loss
llamafactory-cli train examples/extras/eaft/qwen25_05b_eaft_full.yaml
```

#### 7.4 vLLM 推理部署

```bash
# 方式1：LLaMA-Factory 内置 API
API_PORT=8000 llamafactory-cli api examples/inference/qwen3.yaml infer_backend=vllm

# 方式2：直接用 vLLM（合并后的模型）
vllm serve saves/qwen3_sft_merged --port 8000 --tensor-parallel-size 2

# 测试 API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "saves/qwen3_sft_merged",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**学习目标**：
- 理解 packing 对训练效率的提升
- 理解不同优化器的内存/速度权衡
- 理解 vLLM 的 PagedAttention 原理
- 理解模型导出和部署流程

---

### Phase 8: 32B/70B 大模型训练（Day 12-14）

**目标**：在 8×4090 上训练大模型，掌握工程优化技巧。

#### 8.1 32B QLoRA + DeepSpeed ZeRO-3

```bash
cat > examples/train_qlora/qwen3_32b_qlora_sft.yaml << 'EOF'
model_name_or_path: Qwen/Qwen3-32B
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
lora_rank: 16
lora_target: all
deepspeed: examples/deepspeed/ds_z3_config.json

dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
preprocessing_num_workers: 16

output_dir: saves/qwen3-32b/qlora/sft
logging_steps: 10
save_steps: 200
plot_loss: true
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
EOF

llamafactory-cli train examples/train_qlora/qwen3_32b_qlora_sft.yaml
```

**显存分析**（32B QLoRA + ZeRO-3）：
- 4-bit 量化模型：~16GB
- LoRA 参数分片：~10MB/GPU
- 优化器状态分片：~4GB/GPU
- 激活值（梯度检查点）：~4GB/GPU
- 总计：~24GB/GPU，刚好在 4090 极限

#### 8.2 70B QLoRA + DeepSpeed ZeRO-3 + CPU Offload

```bash
cat > examples/deepspeed/ds_z3_offload_config.json << 'EOF'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": {"enabled": "auto"},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
EOF

cat > examples/train_qlora/qwen3_70b_qlora_sft.yaml << 'EOF'
model_name_or_path: Qwen/Qwen3-70B
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
quantization_bit: 4
lora_rank: 16
lora_target: all
deepspeed: examples/deepspeed/ds_z3_offload_config.json

dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 1024   # 大模型减小序列长度
preprocessing_num_workers: 16

output_dir: saves/qwen3-70b/qlora/sft
logging_steps: 10
save_steps: 100
report_to: none

per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 1.0
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
EOF

llamafactory-cli train examples/train_qlora/qwen3_70b_qlora_sft.yaml
```

**学习目标**：
- 理解大模型训练的显存瓶颈和优化策略
- 理解 CPU Offload 的性能代价（PCIe 通信）
- 理解梯度累积和 batch size 的关系

---

## 每日学习检查清单

### Day 1 检查
- [ ] 环境安装成功，`llamafactory-cli env` 输出正常
- [ ] 理解 CLI 入口 → launcher → tuner 的调用链
- [ ] 理解 YAML 配置与 5 个参数数据类的对应关系

### Day 2 检查
- [ ] 理解 Alpaca / ShareGPT 数据格式
- [ ] 理解模板系统的 encode_oneturn / encode_multiturn
- [ ] 理解 label masking 的实现逻辑
- [ ] 完成自定义数据集制作

### Day 3 检查
- [ ] 完成单卡 LoRA SFT 训练
- [ ] 完成多卡 DDP SFT 训练
- [ ] 完成 7B 全参数微调
- [ ] 理解 LoRA vs Full FT 的显存/效果差异

### Day 4 检查
- [ ] 完成推理测试（HF 和 vLLM）
- [ ] 完成 LoRA 合并导出
- [ ] 理解推理后端的选择策略

### Day 5 检查
- [ ] 完成 DPO 训练
- [ ] 完成 KTO 训练
- [ ] 理解 DPO vs KTO 的数据需求差异
- [ ] 理解不同偏好损失的数学原理

### Day 6 检查
- [ ] 完成奖励模型训练
- [ ] 理解 PPO 的 Actor-Critic 架构
- [ ] 理解 KL 惩罚的作用

### Day 7 检查
- [ ] 完成 PPO 训练
- [ ] 理解 replace_model 的适配器切换机制
- [ ] 理解 Value Head 的作用

### Day 8 检查
- [ ] 完成 Agent 微调
- [ ] 理解工具调用的数据格式
- [ ] 理解 Agent 微调与普通 SFT 的区别

### Day 9 检查
- [ ] 完成多模态微调（如有 VLM 模型）
- [ ] 理解 mm_plugin 的工作原理

### Day 10 检查
- [ ] 完成序列打包实验
- [ ] 完成不同优化器对比
- [ ] 完成不同损失函数对比

### Day 11 检查
- [ ] 完成 vLLM 部署
- [ ] 完成 API 测试
- [ ] 理解 PagedAttention 原理

### Day 12 检查
- [ ] 完成 32B QLoRA 训练
- [ ] 理解大模型训练的工程挑战

### Day 13-14 检查
- [ ] 完成 70B QLoRA 训练（如时间允许）
- [ ] 整理所有实验结果
- [ ] 总结关键学习点

---

## 关键实验对比表

| 实验 | 模型 | 方法 | 目标 | 预期显存 | 预期时间 |
|------|------|------|------|----------|----------|
| 1 | 4B | LoRA SFT | 基线 | ~12GB | 30min |
| 2 | 4B | DDP 8卡 LoRA SFT | 多卡效率 | ~12GB/GPU | 10min |
| 3 | 8B | Full FT ZeRO-3 | 全参数微调 | ~16GB/GPU | 2h |
| 4 | 4B | DPO | 偏好对齐 | ~14GB | 30min |
| 5 | 4B | KTO | 二元反馈 | ~14GB | 30min |
| 6 | 4B | RM | 奖励模型 | ~14GB | 30min |
| 7 | 4B | PPO | 完整 RLHF | ~20GB/GPU | 2h |
| 8 | 4B | Agent SFT | 工具调用 | ~14GB | 30min |
| 9 | 4B | Packing SFT | 效率优化 | ~14GB | 20min |
| 10 | 14B | QLoRA | 大模型 | ~12GB/GPU | 1h |
| 11 | 32B | QLoRA ZeRO-3 | 大模型极限 | ~24GB/GPU | 3h |

---

*本计划基于 8×RTX 4090 (24GB) 硬件环境制定，可根据实际训练速度和显存占用动态调整。*
