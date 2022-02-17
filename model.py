from multiprocessing import Lock
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
num_labels = 14
model_checkpoint = f"./output/finetuned/best-model/"


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


reduced_cats = {
    "CULTURE & ARTS": ["ARTS", "ARTS & CULTURE", "CULTURE & ARTS"],
    "BUSINESS": ["BUSINESS", "MONEY"],
    "EDUCATION": ["EDUCATION", "COLLEGE"],
    "ENTERTAINMENT": ["MEDIA", "ENTERTAINMENT", "COMEDY"],
    "HEALTH & LIVING": ["HOME & LIVING", "WELLNESS", "HEALTHY LIVING",
                        "STYLE & BEAUTY",
                        "PARENTS", "STYLE", "FOOD & DRINK", "TASTE", "PARENTING", "DIVORCE", "WEDDINGS"],
    "RELIGION": ["RELIGION"],
    "POLITICS": ["POLITICS", "BLACK VOICES", "LATINO VOICES", "QUEER VOICES", "WOMEN"],
    "SPORTS": ["SPORTS"],
    "TRAVEL": ["TRAVEL"],
    "NEWS": ["GOOD NEWS", "THE WORLDPOST", "WORLDPOST", "WORLD NEWS", "WEIRD NEWS", "CRIME"],
    "ENVIRONMENT": ["GREEN", "ENVIRONMENT"],
    "SCIENCE": ["SCIENCE"],
    "TECH": ["TECH"],
    "OTHER": ["IMPACT", "FIFTY"]
}

label_map = {}
for i, cats in enumerate(reduced_cats.values()):
    for cat in cats:
        label_map[cat] = i

num_labels = len(reduced_cats.keys())
MUTEX = Lock()


class Model:
    lookup_label = {i: v for i, v in enumerate(reduced_cats.values())}

    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                        num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, use_fast=True)

    def predict(self, text):

        MUTEX.acquire()
        try:
            text_pt = self.tokenizer([text],
                                     padding="max_length", max_length=201,
                                     truncation=True, return_tensors="pt")
            # return np.argmax(model_test(**text_pt)[0][0].detach().numpy())
            result = softmax(self.model(**text_pt)[0][0].detach().numpy())
            return self.lookup_label[np.argmax(result)]
        except Exception as e:
            print(e)
        finally:
            # pass
            MUTEX.release()
        return ["error"]


model = Model()


def get_model():
    return model
