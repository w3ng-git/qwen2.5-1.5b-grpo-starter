import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def process_dataset(tokenizer):
    # 加载数据集
    dataset = load_dataset("tldr_dataset_ori")
    train_data = dataset["train"]

    # 打乱后选择前一半数据
    half_train_data = train_data.shuffle(seed=42).select(range(len(train_data) // 2))
    # 对每个样本应用对话模板，包括system message
    def apply_chat_template(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant to summarizes text."},
            {"role": "user", "content": example["prompt"]}
        ]
        example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return example

    processed_data = half_train_data.map(apply_chat_template)
    print(processed_data[0]['prompt'])
    # 保存处理后的数据集到磁盘
    processed_data.save_to_disk("tldr_dataset_processed")
    print("处理后的训练数据已保存到 'tldr_dataset_processed' 文件夹中")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理TLDR数据集并应用对话模板")
    parser.add_argument("--tokenizer", type=str, required=True, help="Hugging Face的模型tokenizer名称或路径")
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        print(f"加载tokenizer时出错: {e}")
        print("请确保提供了有效的Hugging Face模型tokenizer名称或路径")
        exit(1)

    process_dataset(tokenizer)