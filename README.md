# GRPO Training Example

This repository provides a simple example of using Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO). GRPO is designed to enhance model capabilities while optimizing memory usage during training.

## System Requirements

### Hardware Requirements
- **Recommended Configuration:**
  - 4x GPUs with 48GB+ VRAM each
  - 256GB System RAM
- **Minimum Configuration:**(need to modify related config ↓)
  - 2x GPUs with 24GB+ VRAM each
  - One dedicated GPU for vLLM operation

### Configuration Files
For custom hardware setups(2xGPU configuration), modify the following configuration files:
- `zero3.yaml`: DeepSpeed configuration
- `launch_train.sh` or `launch_train.ipynb`: Training launch scripts (choose based on preference)

## Training Pipeline

### 1. Dataset Preparation
```sh
huggingface-cli download trl-lib/tldr --local-dir tldr_dataset_ori --repo-type dataset
```

### 2. Data Processing
Execute the processing script to adapt data for dialogue model training:
```sh
python process_dataset.py
```
Note: Dialogue format ensures stable model performance and prevents overfitting/repetitive outputs.

### 3. Training
Choose your preferred training method:
- Shell script: `launch_train.sh`
- Jupyter notebook: `launch_train.ipynb`

### 4. Training Outputs
- Intermediate checkpoints: Available at `trainOutput/checkpoint-{step}`
- Final model: Saved at `trainOutput/final-checkpoint`

## Model Evaluation

To compare model performance at different training stages:
1. Modify checkpoint path in `demo_in_chat.py`
2. Run the script to observe model behavior

Note: As this is a simple RL experiment, formal evaluation metrics are not implemented.


