# train_grpo.py
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer

dataset = load_from_disk("tldr_dataset_processed")

# 奖励函数 对齐20个字符
def reward_len(completions, **kwargs):
    return [-abs(200 - len(completion)) for completion in completions]

# 平滑更优的奖励函数
# def reward_len(completions, **kwargs):
#     """
#     奖励函数：对齐字符长度，允许一定范围内获得最大奖励
    
#     特点：
#     1. 在目标长度±tolerance范围内，奖励为0
#     2. 超出范围时，使用平方根函数缓和惩罚增长
#     3. scale因子控制惩罚强度
#     """
#     target_len = 300  # 目标字符长度
#     tolerance = 30    # 允许的误差范围
#     scale = 0.05     # 奖励缩放因子
    
#     rewards = []
#     for completion in completions:
#         # 计算与目标长度的差距
#         diff = abs(target_len - len(completion)) - tolerance
#         # 在容忍范围内为0，超出范围则计算惩罚
#         reward = -scale * (max(diff, 0) ** 0.5)
#         rewards.append(reward)
    
#     return rewards

training_args = GRPOConfig(
    # beta = 0.4, 
    
    # 添加 bf16 配置
    bf16=True,  # 启用 bf16 训练

    learning_rate = 1e-06,
    
    # 输出和日志设置
    output_dir="trainOutput",  # 输出目录
    logging_dir="logOutput",
    logging_steps=10,          # 每10步记录一次日志
    
    # 批处理和生成设置
    per_device_train_batch_size=8,  # 批大小
    num_generations=4,  # 每个prompt的生成数量
    
    # 显存优化设置
    gradient_checkpointing=True,  # 启用梯度检查点
    use_vllm=True,  # 使用vLLM优化
    vllm_device='auto',
    vllm_gpu_memory_utilization=0.7,  # 控制GPU显存使用率
    
    # 序列长度设置
    max_prompt_length=1024,  # 限制输入长度
    max_completion_length=640,  # 限制生成长度, ATTENTION: THIS SHOULD BE MODIFIED IF THE MODEL SHOULD GENERATE MORE THAN THAT
    
    # 学习率设置
    lr_scheduler_type='cosine',  # 余弦学习率调度
    warmup_ratio=0.2,  # 预热比例
    
    # 训练轮数
    num_train_epochs=1,
    
    # 检查点保存设置
    save_strategy="steps",  # 按步数保存
    save_steps=30,        # 每100步保存一次
    save_total_limit=8,    # 最多保存8个检查点 ATTENTION: YOU NEED TO MAKE SURE MODEL 'output_dir' empty!

    report_to = ["tensorboard"],
)

trainer = GRPOTrainer(
    model="qwen2.5-1.5b",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)


trainer.train()

# 在训练结束后保存最终模型
trainer.model.save_pretrained(training_args.output_dir + "/final_checkpoint") # grpo seems won't save the model by default, so, manually save it