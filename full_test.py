
from datasets import load_from_disk
from transformers import (
	AutoConfig,
	AutoModel,
	AutoModelForMaskedLM,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	Trainer,
	TrainingArguments,
)


large_dataset_path = "data/my_saved_dataset_large"
base_model_path = "data/my_saved_model"
output_model_path = "data/my_pretrained_model"
text_column = "text"

print("Loading large dataset...")
large_dataset = load_from_disk(large_dataset_path)

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# `my_saved_model` is a sequence-classification checkpoint.
# Load its encoder, then initialize a fresh MLM head on top.
config = AutoConfig.from_pretrained(base_model_path)
encoder_model = AutoModel.from_pretrained(base_model_path)
model = AutoModelForMaskedLM.from_config(config)

encoder_state = {
	k: v
	for k, v in encoder_model.state_dict().items()
	if not k.startswith("pooler.")
}
model.base_model.load_state_dict(encoder_state, strict=True)
model.tie_weights()
print("Initialized MLM model from base encoder weights and fresh MLM head.")

def tokenize_batch(batch):
	return tokenizer(
		batch[text_column]	
		, truncation=True,
		max_length=64,
	)

print("Tokenizing text column...")
tokenized_dataset = large_dataset.map(tokenize_batch, batched=True, remove_columns=large_dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer,
	mlm=True,
	mlm_probability=0.15,
)

training_args = TrainingArguments(
	output_dir=output_model_path,
	per_device_train_batch_size=4,
	learning_rate=5e-5,
	num_train_epochs=1,
	logging_steps=50,
	save_steps=500,
	weight_decay=0.01,
	remove_unused_columns=False,
	report_to="none",
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_dataset,
	data_collator=data_collator
)

print("Starting training on unlabeled text...")
trainer.train()

print("Saving trained model and tokenizer...")
trainer.save_model(output_model_path)
tokenizer.save_pretrained(output_model_path)

print(f"Finished. Saved to {output_model_path}")

