from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model_mnli = AutoModelForSequenceClassification.from_pretrained("./roberta_mnli")
tokenizer_mnli = AutoTokenizer.from_pretrained("./roberta_mnli")
model_fever = AutoModelForSequenceClassification.from_pretrained("./roberta_fever_mnli")
tokenizer_fever = AutoTokenizer.from_pretrained("./roberta_fever_mnli")

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

label_mapping = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

# Load or prepare the evaluation dataset
mnli_test = load_dataset("nyu-mll/multi_nli", split="validation_matched")
fever_test = load_jsonl("nli_fever/eval_fitems.jsonl")
fever_test = Dataset.from_list(fever_test)

def preprocess_function(examples):
    inputs = [
        f"{genre}: {premise}" 
        for premise, genre in zip(examples["premise"], examples["genre"])
    ]
    targets = examples["hypothesis"]
    labels = examples["label"]
    tokenized = tokenizer_mnli(inputs, targets, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = labels
    return tokenized

def preprocess_custom_data(examples):
    inputs = examples["context"]
    targets = examples["query"]
    labels = [label_mapping[label] for label in examples["label"]]
    tokenized = tokenizer_fever(inputs, targets, padding="max_length", truncation=True, max_length=128)
    tokenized["labels"] = labels
    return tokenized

tokenized_mnli = mnli_test.map(preprocess_function, batched=True)
tokenized_fever = fever_test.map(preprocess_custom_data, batched=True)

tokenized_mnli.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_fever.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)  # Convert labels to numpy
    predictions_np = predictions.numpy()  # Convert predictions to numpy
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions_np, average="weighted"),
        "precision": precision_score(labels, predictions_np, average="weighted"),
        "recall": recall_score(labels, predictions_np, average="weighted"),
    }

# Define evaluation arguments
training_args_mnli = TrainingArguments(
    output_dir="./results_eval_mnli",
    do_train=False,
    do_eval=True
)

# Define evaluation arguments
training_args_fever = TrainingArguments(
    output_dir="./results_eval_fever",
    do_train=False,
    do_eval=True
)

# Initialize the Trainer
trainer_mnli = Trainer(
    model=model_mnli,
    args=training_args_mnli,
    eval_dataset=tokenized_mnli,
    compute_metrics=compute_metrics,
)

# Initialize the Trainer
trainer_fever = Trainer(
    model=model_fever,
    args=training_args_fever,
    eval_dataset=tokenized_fever,
    compute_metrics=compute_metrics,
)

# Re-evaluate the model
metrics_mnli = trainer_mnli.evaluate()
metrics_fever = trainer_fever.evaluate()

with open(f"eval_metrics.json", "w") as f:
    json.dump({"metrics_mnli": metrics_mnli, "metrics_fever": metrics_fever}, f)