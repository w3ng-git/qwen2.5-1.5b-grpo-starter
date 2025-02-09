from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

dataset = load_from_disk("tldr_dataset_processed")

# Simple Reward function to align text to target length. ["] https://huggingface.co/docs/trl/main/en/grpo_trainer
def reward_len(completions, **kwargs):
    return [-abs(200 - len(completion)) for completion in completions]

# Improved smooth reward function, unnecessary！
# def reward_len(completions, **kwargs):
#     """
#     Reward function: Aligns text length with a tolerance range for maximum reward
    
#     Features:
#     1. Zero penalty within target length ±tolerance range
#     2. Uses square root function to smooth penalty growth outside the range
#     3. Scale factor controls penalty intensity
#     """
#     target_len = 300  # Target character length
#     tolerance = 30    # Allowed error range
#     scale = 0.05     # Reward scaling factor
    
#     rewards = []
#     for completion in completions:
#         # Calculate difference from target length
#         diff = abs(target_len - len(completion)) - tolerance
#         # Zero within tolerance range, calculate penalty if exceeded
#         reward = -scale * (max(diff, 0) ** 0.5)
#         rewards.append(reward)
    
#     return rewards

training_args = GRPOConfig(
    # beta = 0.04, change if you need to
    
    # BF16 configuration
    bf16=True,  # Enable bf16 training

    learning_rate = 1e-06,
    
    # Output and logging settings
    output_dir="trainOutput",  # Output directory
    logging_dir="logOutput",
    logging_steps=10,          # Log every 10 steps
    
    # Batch and generation settings
    per_device_train_batch_size=8,  # Batch size
    num_generations=4,  # Number of generations per prompt
    
    # Memory optimization settings
    gradient_checkpointing=True,  # Enable gradient checkpointing
    use_vllm=True,  # Use vLLM optimization
    vllm_device='auto',
    vllm_gpu_memory_utilization=0.7,  # Control GPU memory usage
    
    # Sequence length settings
    max_prompt_length=1024,  # Limit input length
    max_completion_length=640,  # Limit generation length, ATTENTION: THIS SHOULD BE MODIFIED IF THE MODEL SHOULD GENERATE MORE THAN THAT
    
    # Learning rate settings
    lr_scheduler_type='cosine',  # Cosine learning rate schedule
    warmup_ratio=0.2,  # Warmup ratio
    
    # Training epochs
    num_train_epochs=1,
    
    # Checkpoint saving settings
    save_strategy="steps",  # Save by steps
    save_steps=30,        # Save every 30 steps
    save_total_limit=8,    # Keep maximum 8 checkpoints ATTENTION: YOU NEED TO MAKE SURE MODEL 'output_dir' empty!

    report_to = ["tensorboard"],
)

trainer = GRPOTrainer(
    model="qwen2.5-1.5b-instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)


trainer.train()

# Save the final model after training
trainer.model.save_pretrained(training_args.output_dir + "/final_checkpoint") # grpo seems won't save the model by default, so, manually save it
