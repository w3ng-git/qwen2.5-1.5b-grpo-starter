# Data Processing
To process the dataset, execute the following command **(from the parent directory)**:
 
```sh
python process_dataset.py
```

After successful processing, come back here, and the directory will contain these files:
```cmd
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----          2/9/2025   6:32 PM       98100872 data-00000-of-00001.arrow
-a----          2/9/2025   6:32 PM           1480 dataset_info.json
-a----          2/9/2025   6:32 PM            250 state.json
```
To verify the processed dataset, you can load and inspect it using the Hugging Face Datasets library:

python
from datasets import load_from_disk
dataset = load_from_disk("tldr_dataset_processed")
