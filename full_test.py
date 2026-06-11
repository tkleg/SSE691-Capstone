
seed = 42

large_dataset_path = "data/my_saved_dataset_large"
base_model_path = "data/my_saved_model"
output_model_path = "data/my_pretrained_model"
text_column = "text"

print("Loading large dataset...")
large_dataset = load_from_disk(large_dataset_path)
num_samples = 1000
large_dataset = large_dataset.shuffle(seed=seed).select(range(num_samples))

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# `my_saved_model` is a sequence-classification checkpoint.
# Load its encoder, then initialize a fresh MLM head on top.
#config = AutoConfig.from_pretrained(base_model_path)
#encoder_model = AutoModel.from_pretrained(base_model_path)
model = AutoModelForMaskedLM.from_pretrained(base_model_path).to("xpu")

# encoder_state = {
#	k: v
#	for k, v in encoder_model.state_dict().items()
#	if not k.startswith("pooler.")
#}
#model.base_model.load_state_dict(encoder_state, strict=True)
#model.tie_weights()
#print("Initialized MLM model from base encoder weights and fresh MLM head.")

def tokenize_batch_large_dataset(batch):
	return tokenizer(
		batch[text_column]	
		, truncation=True,
		max_length=64,
	)

print("Tokenizing text column...")
tokenized_dataset = large_dataset.map(tokenize_batch_large_dataset, batched=True, remove_columns=large_dataset.column_names)

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
model.save_pretrained(output_model_path)
unlabaled_trained_model = model.to("xpu")
print(f"Finished unsupervised training. Saved to {output_model_path}")

small_dataset_path = "data/my_saved_dataset_small"
print("Loading small dataset...")
small_dataset = load_from_disk(small_dataset_path)

#def clean_label(value):
#	return str(value).strip().lower()

#small_dataset = small_dataset.map(
#	lambda example: {
#		"politics": clean_label(example["politics"]),
#		"sentiment": clean_label(example["sentiment"]),
#	}
#)

#small_dataset = small_dataset.filter(
#	lambda example: example["politics"] in {"left", "right"}
#	and example["sentiment"] in {"positive", "negative"}
#)

small_dataset = small_dataset.class_encode_column("sentiment")

# Stratify on the joint label so train/test keep equal portions by side and sentiment.
#small_dataset = small_dataset.map(
#	lambda example: {
#		"split_stratify": f"{example['politics']}|{example['sentiment']}"
#	}
#)

#small_dataset = small_dataset.class_encode_column("split_stratify")

split_dataset = small_dataset.train_test_split(
	test_size=0.25,
	seed=seed,
	stratify_by_column="sentiment"
)

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

def tokenize_small_dataset_batch(batch):
	return tokenizer(
		batch[text_column],
		truncation=True,
		max_length=64,
		padding="max_length",
	)

print("Tokenizing small dataset...")
tokenized_train_dataset = train_dataset.map(tokenize_small_dataset_batch, batched=True, remove_columns=train_dataset.column_names)
tokenized_test_dataset = test_dataset.map(tokenize_small_dataset_batch, batched=True, remove_columns=test_dataset.column_names)

training_args.output_dir = "data/my_finetuned_model"
small_trainer = Trainer(
	model=unlabaled_trained_model,
	args=training_args,
	train_dataset=tokenized_train_dataset,
	eval_dataset=tokenized_test_dataset
	#data_collator=data_collator,
)
print("Starting supervised fine-tuning on small dataset...")
small_trainer.train()
print("Saving fine-tuned model and tokenizer...")
small_trainer.save_model("data/my_finetuned_model")
print("Finished supervised fine-tuning. Saved to data/my_finetuned_model")
eval_results = small_trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 1) Build numeric labels from string labels
label2id = {"positive": 0, "negative": 1}
id2label = {0: "positive", 1: "negative"}

def tokenize_with_labels(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=64,
        padding="max_length",
    )
    tokens["labels"] = [label2id[v] for v in batch["sentiment"]]
    return tokens

# 2) Tokenize train/test sets and keep labels
tokenized_train_for_cls = train_dataset.map(
	tokenize_with_labels,
	batched=True,
	remove_columns=train_dataset.column_names
)
tokenized_test_for_cls = test_dataset.map(
    tokenize_with_labels,
    batched=True,
    remove_columns=test_dataset.column_names
)

# 3) Classification model (not MLM head)
clf_model = AutoModelForSequenceClassification.from_pretrained(
    output_model_path,
    num_labels=2,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# 4) Train on split train, then predict on split test
cls_args = TrainingArguments(
	output_dir="tmp_eval",
	report_to="none",
	per_device_train_batch_size=8,
	per_device_eval_batch_size=8,
	learning_rate=5e-5,
	num_train_epochs=1,
)
pred_trainer = Trainer(
	model=clf_model,
	args=cls_args,
	train_dataset=tokenized_train_for_cls,
	eval_dataset=tokenized_test_for_cls,
)
pred_trainer.train()
pred_output = pred_trainer.predict(tokenized_test_for_cls)

pred_ids = np.argmax(pred_output.predictions, axis=-1)
true_ids = np.array(tokenized_test_for_cls["labels"])
accuracy = (pred_ids == true_ids).mean()

print("Accuracy:", float(accuracy))