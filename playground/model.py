#demo code from https://huggingface.co/bucketresearch/politicalBiasBERT modified to load model if saved
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from os import path

text = "your text here"

local_dir = "playground/my_saved_model"

if path.exists(local_dir) and path.exists(path.join(local_dir, "config.json")):
	print("Loading model and tokenizer from local directory...")
	tokenizer = AutoTokenizer.from_pretrained(local_dir)
	model = AutoModelForSequenceClassification.from_pretrained(local_dir)
else:
	print("Loading model and tokenizer from Hugging Face...")
	tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
	model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
	tokenizer.save_pretrained(local_dir)
	model.save_pretrained(local_dir)

inputs = tokenizer(text, return_tensors="pt")
labels = torch.tensor([0])
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]

# [0] -> left 
# [1] -> center
# [2] -> right
print(logits.softmax(dim=-1)[0].tolist())