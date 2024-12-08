from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load tokenizer and model for sequence classification
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large", num_labels=3)

# Load SNLI dataset
mnli_train = load_dataset("nyu-mll/multi_nli", split="train")

mnli_train = mnli_train.filter(lambda x: x["label"] != -1)

# Preprocessing function
def preprocess_data(examples):
    inputs = [
        f"{genre}: {premise}" 
        for premise, genre in zip(examples["premise"], examples["genre"])
    ]
    targets = examples["hypothesis"]
    labels = examples["label"]
    tokenized = tokenizer(inputs, targets, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = labels
    return tokenized

# Tokenize datasets
tokenized_train = mnli_train.map(preprocess_data, batched=True)

# Convert datasets to PyTorch format
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5000,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./roberta_mnli")
tokenizer.save_pretrained("./roberta_mnli")
