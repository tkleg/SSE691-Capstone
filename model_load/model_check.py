from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch

local_path = "data/my_saved_model"

if not os.path.exists(local_path):
	raise FileNotFoundError(f"Model not found: {local_path}")

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSequenceClassification.from_pretrained(local_path)

#Sample from the https://huggingface.co/bucketresearch/politicalBiasBERT
texts = ["I cannot stand democrats. They ruin this country. Praise the Republican party!", "I cannot stand republicans. They ruin this country. Praise the Democratic party!", "Both sides are bad. I don't like either of them."]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
labels = torch.tensor([0, 1, 2])
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]

# [0] -> left
# [1] -> center
# [2] -> right
results = logits.softmax(dim=-1).tolist()
for text, result in zip(texts, results):
	print(f"Text: {text}")
	print(f"Left: {result[0]:.4f}, Center: {result[1]:.4f}, Right: {result[2]:.4f}\n")