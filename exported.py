import numpy as np


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"],
                   padding="max_length",max_length=201 ,
                   truncation=True)


def compute_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

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