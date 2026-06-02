from datasets import load_dataset, Dataset, load_from_disk
import os
from dotenv import load_dotenv

load_dotenv()

dataset_string = "BAAI/IndustryCorpus_politics"
saved_dataset_path = "data/my_saved_dataset_large"
num_samples = 100_000

token = os.getenv("HF_TOKEN")
if token:
    print(f"Using Hugging Face token: {token[:4]}...")
else:
    print("HF_TOKEN is not set.")

local_path = saved_dataset_path
if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "dataset_info.json")):
    print("Loading dataset from local directory...")
    dataset = load_from_disk(local_path)
else:
    print("Streaming and sampling from Hugging Face...")
    stream = load_dataset(dataset_string, split="train", streaming=True, token=token)
    print("Loaded dataset stream. Sampling...")
    column_to_save = "text"
    samples = []
    i = 0
    for example in stream:
        if column_to_save in example:
            samples.append({column_to_save: example[column_to_save]})
        else:
            continue
        i += 1
        if i % 1000 == 0:
            print(f"Sampled {i} examples so far...")
        if i >= num_samples:
            break
    print(f"Finished sampling {i} examples. Saving to disk...")
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(local_path)
print(dataset)