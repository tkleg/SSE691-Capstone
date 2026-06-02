from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch

local_path = "data/my_saved_model"

if not os.path.exists(local_path):
	raise FileNotFoundError(f"Model not found: {local_path}")

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSequenceClassification.from_pretrained(local_path)

#Sample from the https://huggingface.co/bucketresearch/politicalBiasBERT
text = "I cannot stand democrats. They ruin this country. Praise the Republican party!"

inputs = tokenizer(text, return_tensors="pt")
labels = torch.tensor([0])
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]

# [0] -> left
# [1] -> center
# [2] -> right
print(logits.softmax(dim=-1)[0].tolist())