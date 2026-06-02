#demo code from https://huggingface.co/bucketresearch/politicalBiasBERT modified to load model if saved
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")

path = "data/my_saved_model"

if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
	print("Loading model and tokenizer from local directory...")
	tokenizer = AutoTokenizer.from_pretrained(path)
	model = AutoModelForSequenceClassification.from_pretrained(path)
else:
	print("Loading model and tokenizer from Hugging Face...")
	tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
	model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT", token=token)
	tokenizer.save_pretrained(path)
	model.save_pretrained(path)