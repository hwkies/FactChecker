import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch

# Load tokenizer and model
model_checkpoint = "./roberta_multinli"  # Path to the previously fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

# Load JSON data
def load_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

fever = load_jsonl("nli_fever/train_fitems.jsonl")
fever += load_jsonl("nli_fever/train_fitems2.jsonl")
# fever_test = load_jsonl("nli_fever/test_fitems.jsonl")

# Preprocess data
label_mapping = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

def preprocess_custom_data(examples):
    inputs = [
        f"query: {example['query']} context: {example['context']}" 
        for example in examples
    ]
    labels = [label_mapping[example["label"]] for example in examples]
    tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = labels
    return tokenized

# Convert to Hugging Face Dataset
fever = Dataset.from_list(fever)
# fever_test = Dataset.from_list(fever_test)
tokenized = fever.map(preprocess_custom_data, batched=True)
# tokenized_test = fever_test.map(preprocess_custom_data, batched=True)

# Convert to PyTorch tensors
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized = tokenized.train_test_split(test_size=0.2)
tokenized_train = tokenized["train"]
tokenized_test = tokenized["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_custom",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Compute metrics function
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./roberta_fever_mnli")
tokenizer.save_pretrained("./roberta_fever_mnli")
