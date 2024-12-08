import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load tokenizer and model
model_checkpoint = "./roberta_mnli"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

# Load JSON data
def load_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

fever = load_jsonl("nli_fever/train_fitems.jsonl")

# Preprocess data
label_mapping = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

def preprocess_custom_data(examples):
    inputs = examples["context"]
    targets = examples["query"]
    labels = [label_mapping[label] for label in examples["label"]]
    tokenized = tokenizer(inputs, targets, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = labels
    return tokenized


# Convert to Hugging Face Dataset
fever = Dataset.from_list(fever)
# fever_test = Dataset.from_list(fever_test)
tokenized = fever.map(preprocess_custom_data, batched=True)
# tokenized_test = fever_test.map(preprocess_custom_data, batched=True)

# Convert to PyTorch tensors
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_fever",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./roberta_fever_mnli")
tokenizer.save_pretrained("./roberta_fever_mnli")
