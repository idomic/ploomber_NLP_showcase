# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
# ---

# %% tags=["soorgeon-imports"]
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_metric
import numpy as np
from exported import preprocess_function, compute_metrics, softmax, test_interference

# %% tags=["parameters"]
upstream = ['get-clean-data']
product = None

# %% [markdown] tags=[]
# ## Training the model

# %%
metric = load_metric("accuracy")

# %%
train_dataset = Dataset.from_file(f"{upstream['get-clean-data']['train_dataset']}/dataset.arrow")
test_dataset  = Dataset.from_file(f"{upstream['get-clean-data']['test_dataset']}/dataset.arrow")
num_labels = 41

# %%
model_checkpoint = "microsoft/xtremedistil-l6-h256-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


# %%
def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"],
                   padding="max_length",max_length=201 ,
                   truncation=True)

def compute_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# %%
encoded_train_dataset = train_dataset.map(preprocess_function, tokenizer,batched=True)
encoded_test_dataset = test_dataset.map(preprocess_function, tokenizer,batched=True)

# %%
metric_name = "accuracy"
batch_size= 16
args = TrainingArguments(
    f"finetuned",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

# %%
validation_key = "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_test_dataset,
    tokenizer=tokenizer,
    
    compute_metrics=compute_metrics(metric)
)

# %%
trainer.train()

# %%
trainer.save()
