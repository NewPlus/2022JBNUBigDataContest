#!/usr/bin/env python
# coding: utf-8

# In[54]:


get_ipython().run_line_magic('pip', 'install tqdm')
get_ipython().run_line_magic('pip', 'install torch')
get_ipython().run_line_magic('pip', 'install torchsummary')
get_ipython().run_line_magic('pip', 'install transformers')
get_ipython().run_line_magic('pip', 'install datasets')
get_ipython().run_line_magic('pip', 'install numpy')
get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install -q albumentations')
get_ipython().run_line_magic('pip', 'install opencv-python')


# In[55]:


# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from sklearn.model_selection import train_test_split
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
from torch import nn, Tensor

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import pandas as pd
import numpy as np
import time
import copy
from PIL import Image
import tqdm


# In[56]:


def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# # **Data Review**

# # Art Style Image Classification-10 
# 
# ####  1.인상주의, 2.르네상스 초현실주의, 3.아르누보, 4.바로크, 5.표현주의, 6.낭만주의, 7.우키요에, 8.포스트, 9.인상주의, 10.실재론

# * Load Image Data

# In[57]:


train_path = "train/train/"
test_path = "test/test/"
submission_path = ""


# In[58]:


from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir=train_path)
print(dataset)
dataset_test = load_dataset("imagefolder", data_dir=test_path)
print(dataset_test)


# In[60]:


from datasets import load_metric

metric = load_metric("accuracy")


# In[61]:


example = dataset["train"][10]
example


# In[62]:


dataset["train"].features


# In[63]:


labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]


# In[64]:


labels


# In[65]:


from transformers import AutoFeatureExtractor

MODEL_NAME = "microsoft/beit-base-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
feature_extractor


# In[66]:


from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomGrayscale,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ColorJitter,
    ToTensor,
)

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(hue=0.5),
            ColorJitter(saturation=0.5),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


# In[67]:


splits = dataset["train"].train_test_split(test_size=0.1)

train_ds = splits['train']
val_ds = splits['test']


# In[68]:


train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


# In[69]:


train_ds[0]


# In[70]:


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True,
)


# In[71]:


batch_size = 32

args = TrainingArguments(
    MODEL_NAME,
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=50,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True, # 좀 더 빠른 train이 된다??
)


# In[73]:


import numpy as np

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


# # BEiT - BERT Pre-Training of Image Transformers 
# - Inspired by BERT, BEiT is the first paper that makes self-supervised pre-training of Vision Transformers (ViTs) outperform supervised pre-training.
# - Paper : [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
# 
# # 참고자료 출처
# - huggingface trainer 사용법 : https://huggingface.co/docs/transformers/training#train
# - huggingface BEiT 사용법(docs) : https://huggingface.co/docs/transformers/model_doc/beit
# - huggingface Image Classification Tutorial : https://huggingface.co/docs/transformers/tasks/image_classification

# In[74]:


import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# In[75]:


len(labels)


# In[77]:


from torch import nn
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss


# In[78]:


trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()
model.save_pretrained('./result/best_model')


# In[79]:


from transformers import AutoModelForImageClassification, AutoFeatureExtractor

repo_name = MODEL_NAME+"/checkpoint-15200"

feature_extractor = AutoFeatureExtractor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)


# In[82]:


from transformers import pipeline

pipe = pipeline("image-classification", 
                model=model,
                feature_extractor=feature_extractor)
answers = [pipe(img)[0]['label'] for img in dataset_test['train']['image']]
answers


# In[83]:


csv_list = pd.DataFrame({'id': range(len(answers)), 'label': answers})
csv_list.to_csv("TestResult2_vit_40.csv", index = False)


# 1.인상주의, 2.르네상스 초현실주의, 3.아르누보, 4.바로크, 5.표현주의, 6.낭만주의, 7.우키요에, 8.포스트, 9.인상주의, 10.실재론
