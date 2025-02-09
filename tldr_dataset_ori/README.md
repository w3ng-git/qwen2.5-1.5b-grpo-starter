# Raw, Original Dataset
To download the required dataset, execute the following command from the project **root directory**:
```sh
huggingface-cli download trl-lib/tldr --local-dir tldr_dataset_ori --repo-type dataset
```

```
Upon completion, this directory structure should appear as follows:

(.)
 ├─.cache
 │  └─huggingface
 │      └─download
 │          └─data
 └─data
```
The dataset will be downloaded and stored in the 'tldr_dataset_ori' directory, which is required for subsequent processing steps.
