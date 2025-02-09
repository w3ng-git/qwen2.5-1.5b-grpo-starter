"""
Dataset Processing Script

This script processes the TLDR dataset to ensure it's presented in a conversational format.
We adapt the data to use chat templates because dialogue models have shown more stable performance,
and we need to conform to the chat template structure expected by these models.
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def process_dataset(tokenizer):
    # Load dataset
    dataset = load_dataset("tldr_dataset_ori")
    train_data = dataset["train"]

    # Shuffle and select first half of the data
    half_train_data = train_data.shuffle(seed=42).select(range(len(train_data) // 2))
    
    # Apply chat template to each sample, including system message
    def apply_chat_template(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant to summarizes text."},
            {"role": "user", "content": example["prompt"]}
        ]
        example["prompt"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return example

    processed_data = half_train_data.map(apply_chat_template)
    print(processed_data[0]['prompt'])
    
    # Save processed dataset to disk
    processed_data.save_to_disk("tldr_dataset_processed")
    print("Processed training data has been saved to 'tldr_dataset_processed' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TLDR dataset and apply chat template")
    parser.add_argument("--tokenizer", type=str, required=True, help="Hugging Face model tokenizer name or path")
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure you provided a valid Hugging Face model tokenizer name or path")
        exit(1)

    process_dataset(tokenizer)
