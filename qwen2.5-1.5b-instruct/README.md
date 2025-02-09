# Required: QwenN-N-instruct model checkpoint must be located in this directory.
##  Model Directory Structure

Please place your Qwen instruction model (QwenN-N-instruct) in this directory with the following structure:
(qwen2.5-1.5b-instruct or any other model you like)
.
├── README.md                 # This file (4.92 KB)
├── config.json              # Model configuration (660 B)
├── generation_config.json   # Generation settings (242 B)
├── merges.txt              # BPE merges file (1.67 MB)
├── model.safetensors       # Model weights (3.09 GB)
├── tokenizer.json          # Tokenizer configuration (7.03 MB)
├── tokenizer_config.json   # Tokenizer settings (7.31 KB)
└── vocab.json              # Vocabulary file (2.78 MB)

All files are essential for the model to function properly. Ensure all components are present before running the training script.

Note: `model.safetensors` is stored using Git LFS due to its large size.
