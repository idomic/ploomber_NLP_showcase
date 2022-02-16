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
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
from exported import preprocess_function, compute_metrics, softmax, test_interference

# %% tags=["parameters"]
upstream = None
product = None

# %% [markdown]
# # Topic classification
# ## Downloading and cleaning data

# %%



SEED = 42

# %%
dataset = load_dataset("Fraser/news-category-dataset")
num_labels = len(set(dataset["train"]["category_num"]))


# %%
reduced_categories = {
  "CULTURE & ARTS":["ARTS","ARTS & CULTURE","CULTURE & ARTS"],
 "BUSINESS": ["BUSINESS","MONEY"],
 "EDUCATION": ["EDUCATION","COLLEGE"],
 "COMEDY & ENTERTAINMENT" : ["COMEDY","ENTERTAINMENT","MEDIA"],
 "HEALTH & LIVING": ["WELLNESS","HEALTHY LIVING",
                     "STYLE & BEAUTY","HOME & LIVING",
                     "PARENTS","STYLE","FOOD & DRINK","TASTE","PARENTING","DIVORCE","WEDDINGS"],
 "RELIGION" : ["RELIGION"],
 "POLITICS" : ["POLITICS","BLACK VOICES","LATINO VOICES","QUEER VOICES","WOMEN"],
 "SPORTS" : ["SPORTS"],
 "TRAVEL" : ["TRAVEL"],
 "NEWS" :["GOOD NEWS","THE WORLDPOST","WORLDPOST","WORLD NEWS","WEIRD NEWS","CRIME"],
 "ENVIRONMENT" : ["GREEN","ENVIRONMENT"],
 "SCIENCE": ["SCIENCE"],
 "TECH": ["TECH"],
 "OTHER" : ["IMPACT","FIFTY"]
 }



# %%
label_map = {}
for i,cats in enumerate(reduced_categories.values()):
  for cat in cats:
    label_map[cat] = i

# %%
train_dataset = Dataset.from_dict({
    "text" : dataset["train"]["headline"],
    "label": pd.Series(dataset["train"]["category"]).replace(label_map).tolist()
})

# %%

test_dataset = Dataset.from_dict({
    "text" : dataset["test"]["headline"],
    "label": pd.Series(dataset["test"]["category"]).replace(label_map).tolist()
})

# %%
train_dataset.save_to_disk(product['train_dataset']) #"./output/train_dataset"
test_dataset.save_to_disk(product['test_dataset']) # "./output/test_dataset"
