# 训练指南（中文）

## 1. 数据准备

### 必需文件格式

每个音频样本在 `data` 目录下需要且只需要以下 3 个文件：

1. **`filename.mp3`** —— 音频文件
2. **`filename_prompt.txt`** —— 音频特征（用英文逗号分隔的标签）
3. **`filename_lyrics.txt`** —— 歌词（可选，但推荐提供）

### 示例目录结构

```
data/
├── test_track_001.mp3
├── test_track_001_prompt.txt
└── test_track_001_lyrics.txt
```

### 文件内容格式

#### `*_prompt.txt` —— 音频标签

使用简单的英文逗号分隔标签，描述声音特征、乐器、风格、情绪等。

**示例：**

```
melodic techno, male vocal, electronic, emotional, minor key, 124 bpm, synthesizer, driving, atmospheric
```

**编写标签的建议：**

- 包含**风格**（如 "rap", "pop", "rock", "electronic"）
- 包含**人声类型**（如 "male vocal", "female vocal", "spoken word"）
- 包含**实际可听到的乐器**（如 "guitar", "piano", "synthesizer", "drums"）
- 包含**情绪/能量**（如 "energetic", "calm", "aggressive", "melancholic"）
- 如已知，加入**速度/节拍**（如 "120 bpm", "fast tempo", "slow tempo"）
- 如已知，加入**调式/调性**（如 "major key", "minor key", "C major"）

#### `*_lyrics.txt` —— 歌词

按常见的主歌/副歌结构编写标准歌词文本。

**示例：**

```
[Verse]
Lately I've been wondering
Why do I do this to myself
I should be over it

[Chorus]
It makes me want to cry
If you knew what you meant to me
I wonder if you'd come back
```

### ⚠️ 注意事项

- **文件命名必须严格遵守约定**：`filename.mp3`、`filename_prompt.txt`、`filename_lyrics.txt`
- **不支持 JSON**：转换脚本仅读取上述简单文本文件
- **不使用复杂多变体描述**：仅支持逗号分隔的简单标签格式

## 2. 转换为 Hugging Face 数据集格式

运行以下命令将原始数据转换为训练数据集：

```bash
python convert2hf_dataset.py --data_dir "./data" --repeat_count 2000 --output_name "zh_lora_dataset"
```

**参数说明：**

- `--data_dir`：包含 MP3、prompt 与 lyrics 文件的数据目录路径
- `--repeat_count`：对数据重复的次数（小数据集建议更高的次数）
- `--output_name`：输出数据集目录名称

### 转换器会生成什么

转换器会处理文件，并生成包含以下特征的 Hugging Face 数据集：

```python
Dataset Features:
{
        'keys': string,              # 文件标识（如 "test_track_001"）
        'filename': string,          # MP3 文件路径
        'tags': list[string],        # 解析后的标签数组
        'speaker_emb_path': string,  # 为空，不使用
        'norm_lyrics': string,       # 完整歌词文本
        'recaption': dict            # 为空，不使用
}
```

**示例（处理后的样本）：**

```python
{
        'keys': 'test_track_001',
        'filename': 'data/test_track_001.mp3',
        'tags': ['melodic techno', 'male vocal', 'electronic', 'emotional', 'minor key', '124 bpm', 'synthesizer', 'driving', 'atmospheric'],
        'speaker_emb_path': '',
        'norm_lyrics': '[Verse]\nLately I\'ve been wondering\nWhy do I do this to myself...',
        'recaption': {}
}
```

## 3. 配置 LoRA 参数

请参考 `config/zh_rap_lora_config.json` 进行 LoRA 相关参数配置。

如果显存不足，可以在配置文件中适当降低 `r` 与 `lora_alpha`，例如：

```json
{
  "r": 16,
  "lora_alpha": 32,
  "target_modules": [
    "linear_q",
    "linear_k",
    "linear_v",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0"
  ]
}
```

## 4. 运行训练

使用下述参数说明运行 `python trainer.py`：

# 训练器参数说明

## 1. 通用设置（General Settings）

1. **`--num_nodes`**：训练所使用的节点数量，整数，默认 1。用于多机分布式训练时设置节点数；单机训练保持默认即可。
2. **`--shift`**：浮点数，默认 3.0。具体作用依实现而定，通常用于模型内部的偏移/校正等计算。

## 2. 训练超参数（Training Hyperparameters）

1. **`--learning_rate`**：学习率，浮点数，默认 `1e-4`。学习率越小越稳定但收敛更慢；过大可能导致发散或震荡。
2. **`--num_workers`**：数据加载的并发进程数，整数，默认 8。更高的并发可提升加载速度，但会占用更多 CPU/内存资源。
3. **`--epochs`**：训练遍历数据集的轮数，整数，默认 -1。为 -1 时按其他停止条件（如最大步数）终止；设为正整数则按指定轮数停止。
4. **`--max_steps`**：训练的最大学习步数，整数，默认 2,000,000。达到该步数即停止训练。
5. **`--every_n_train_steps`**：整数，默认 2000。控制如保存检查点、记录日志等操作的执行间隔（每多少步执行一次）。

## 3. 数据集与实验设置（Dataset and Experiment Settings）

1. **`--dataset_path`**：Hugging Face 数据集路径，字符串，默认 `./zh_lora_dataset`。需确保该路径数据格式正确。
2. **`--exp_name`**：实验名称，字符串，默认 `chinese_rap_lora`。用于区分不同实验、管理日志与检查点。

## 4. 训练精度与梯度设置（Training Precision and Gradient Settings）

1. **`--precision`**：训练精度，字符串，默认 `32`（通常指 FP32）。更高精度更耗显存与算力；可按硬件能力调整。
2. **`--accumulate_grad_batches`**：梯度累积步数，整数，默认 1。设为 4 表示累计 4 个 batch 再进行一次优化更新，可在显存受限时模拟大 batch。
3. **`--gradient_clip_val`**：梯度裁剪阈值，浮点数，默认 0.5。用于防止梯度爆炸。
4. **`--gradient_clip_algorithm`**：梯度裁剪算法，字符串，默认 `norm`。

## 5. 检查点与日志设置（Checkpoint and Logging Settings）

1. **`--devices`**：使用的设备数量（如 GPU 数），整数，默认 1。多 GPU 并行训练可增大该值。
2. **`--logger_dir`**：日志目录，字符串，默认 `./exps/logs/`。
3. **`--ckpt_path`**：恢复训练的检查点路径，字符串，默认 None。为 None 则从头开始训练。
4. **`--checkpoint_dir`**：训练时保存检查点的目录，字符串，默认 None（可能使用框架默认位置）。

## 6. 验证与数据加载重载（Validation and Reloading Settings）

1. **`--reload_dataloaders_every_n_epochs`**：每隔多少个 epoch 重新构建/重载数据加载器，整数，默认 1。用于确保每个 epoch 的数据顺序/处理策略刷新。
2. **`--every_plot_step`**：生成可视化（如曲线图）间隔步数，整数，默认 2000。
3. **`--val_check_interval`**：验证间隔，整数或 None，默认 None。为正整数时表示每隔多少步做验证；None 则不定期验证。
4. **`--lora_config_path`**：LoRA 配置文件路径，字符串，默认 `config/zh_rap_lora_config.json`，包含如低秩矩阵 rank、LoRA 参数学习率等设置。
