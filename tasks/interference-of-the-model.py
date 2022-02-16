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
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import numpy as np
from exported import preprocess_function, compute_metrics, softmax, test_interference

# %% tags=["parameters"]
upstream = ['train']
product = None

# %% [markdown] tags=[]
# ## Interference of the model

# %%
num_labels = 41

# %%
best_model = f"./finetuned/best-model/"

tokenizer = AutoTokenizer.from_pretrained(best_model, use_fast=True)

model_test = AutoModelForSequenceClassification.from_pretrained(
    best_model,
    num_labels=num_labels)


# %%
#source: https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines/text_classification.py
def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

def test_interference(trained_model, text, tokenizer):
    text_pt = tokenizer([text],
                    padding="max_length",max_length=201,
                    truncation=True,return_tensors="pt")
    return softmax(trained_model(**text_pt)[0][0].detach().numpy())


# %%

np.argmax(test_interference(model_test, "A bird was flying today", tokenizer))


