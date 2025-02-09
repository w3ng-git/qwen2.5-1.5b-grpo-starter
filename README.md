# GRPO Training Example

This repository provides a simple example of using GRPO(Group Relative Policy Optimization) Trainer, GRPO is a variant of Proximal Policy Optimization (PPO). GRPO is designed to enhance model capabilities in specific domains.

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

## Reward Functions

### Simple Length-based Reward

The basic reward function used in this example aims for a target length of 200 characters, with maximum reward (0) at the target:

```python
def reward_len(completions, **kwargs):
    return [-abs(200 - len(completion)) for completion in completions]
```

### Alternative Smooth Reward (Optional)

A more sophisticated version with smooth transitions and tolerance range is also available, though not necessary for this simple experiment:

```python
def reward_len(completions, **kwargs):
    """
    Reward function: Aligns text length with a tolerance range for maximum reward
    
    Features:
    1. Zero penalty within target length ±tolerance range
    2. Uses square root function to smooth penalty growth outside the range
    3. Scale factor controls penalty intensity
    """
    target_len = 300  # Target character length
    tolerance = 30    # Allowed error range
    scale = 0.05     # Reward scaling factor
    
    rewards = []
    for completion in completions:
        # Calculate difference from target length
        diff = abs(target_len - len(completion)) - tolerance
        # Zero within tolerance range, calculate penalty if exceeded
        reward = -scale * (max(diff, 0) ** 0.5)
        rewards.append(reward)
    
    return rewards
```

Note: For this experimental setup, the simple reward function proved sufficient, and the smooth version is not necessary.

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

## Training Results(~100 step, total 300)

Using a simple length-based reward function (maximum reward of 0 at 200 tokens), the training demonstrates effective control of model output:

- Initial completion length: ~115 tokens
- Final stabilized length: ~46 tokens
- Reward improvement: from -368.4 to -41.3

The results show that even with this simple reward mechanism, the model successfully learned to generate more concise outputs while maintaining stability. Real-world testing confirms significant reduction in output length compared to the base model.
```raw
{'loss': 0.0001, 'grad_norm': 1.6272573444330736, 'learning_rate': 1.6393442622950818e-07, 'completion_length': 115.18112258911133, 'rewards/reward_len': -368.4007911682129, 'reward': -368.4007911682129, 'reward_std': 284.2543468475342, 'kl': 0.0018751144409179688, 'epoch': 0.03}
{'loss': 0.0001, 'grad_norm': 0.7721365771189117, 'learning_rate': 3.2786885245901637e-07, 'completion_length': 114.90390939712525, 'rewards/reward_len': -369.5871212005615, 'reward': -369.5871212005615, 'reward_std': 292.81192512512206, 'kl': 0.003034496307373047, 'epoch': 0.07}
{'loss': 0.0005, 'grad_norm': 1.4574345784614582, 'learning_rate': 4.918032786885245e-07, 'completion_length': 103.0485704421997, 'rewards/reward_len': -312.80860328674316, 'reward': -312.80860328674316, 'reward_std': 250.52502326965333, 'kl': 0.01207427978515625, 'epoch': 0.1}
{'loss': 0.0028, 'grad_norm': 1.8255208323482197, 'learning_rate': 6.557377049180327e-07, 'completion_length': 77.25885653495789, 'rewards/reward_len': -206.15013647079468, 'reward': -206.15013647079468, 'reward_std': 180.6353874206543, 'kl': 0.06939697265625, 'epoch': 0.13}
{'loss': 0.0061, 'grad_norm': 3.0441187436347694, 'learning_rate': 8.196721311475409e-07, 'completion_length': 54.783074378967285, 'rewards/reward_len': -110.58463907241821, 'reward': -110.58463907241821, 'reward_std': 90.85162644386291, 'kl': 0.15252685546875, 'epoch': 0.16}
{'loss': 0.0095, 'grad_norm': 3.475613910689978, 'learning_rate': 9.83606557377049e-07, 'completion_length': 47.682292985916135, 'rewards/reward_len': -70.22526235580445, 'reward': -70.22526235580445, 'reward_std': 52.54481310844422, 'kl': 0.23743896484375, 'epoch': 0.2}
{'loss': 0.0365, 'grad_norm': 3.734326871973368, 'learning_rate': 9.966191788709714e-07, 'completion_length': 45.15937623977661, 'rewards/reward_len': -52.44257960319519, 'reward': -52.44257960319519, 'reward_std': 37.93485209941864, 'kl': 0.9111328125, 'epoch': 0.23}
{'loss': 0.0101, 'grad_norm': 9.539953564006392, 'learning_rate': 9.849910750108717e-07, 'completion_length': 45.174740839004514, 'rewards/reward_len': -47.45143332481384, 'reward': -47.45143332481384, 'reward_std': 33.90584886074066, 'kl': 0.252734375, 'epoch': 0.26}
{'loss': 0.0114, 'grad_norm': 6.496857889148631, 'learning_rate': 9.652679879607843e-07, 'completion_length': 46.4753918170929, 'rewards/reward_len': -44.792188835144046, 'reward': -44.792188835144046, 'reward_std': 33.149115562438965, 'kl': 0.28583984375, 'epoch': 0.3}
{'loss': 0.1079, 'grad_norm': 1.0094380637774818, 'learning_rate': 9.377791156510454e-07, 'completion_length': 46.20989732742309, 'rewards/reward_len': -41.621094942092896, 'reward': -41.621094942092896, 'reward_std': 30.73620209693909, 'kl': 2.69691162109375, 'epoch': 0.33}
{'loss': 0.1309, 'grad_norm': 82.97760249492144, 'learning_rate': 9.029832746882371e-07, 'completion_length': 46.09713678359985, 'rewards/reward_len': -41.313152360916135, 'reward': -41.313152360916135, 'reward_std': 31.263010478019716, 'kl': 3.26756591796875, 'epoch': 0.36}
```
