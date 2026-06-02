from datasets import load_from_disk
import os

small_data_path = "data/my_saved_dataset_small"
large_data_path = "data/my_saved_dataset_large"

if os.path.exists(small_data_path) and os.path.exists(os.path.join(small_data_path, "dataset_info.json")):
    print("Loading small dataset from local directory...")
    small_dataset = load_from_disk(small_data_path)
    print(small_dataset.column_names, small_dataset.num_rows)
else:
    print("Small dataset not found locally. Please run datasets_small.py to create it.")

if os.path.exists(large_data_path) and os.path.exists(os.path.join(large_data_path, "dataset_info.json")):
    print("Loading large dataset from local directory...")
    large_dataset = load_from_disk(large_data_path)
    print(large_dataset.column_names, large_dataset.num_rows)
else:
    print("Large dataset not found locally. Please run datasets_large.py to create it.")