#!/usr/bin/env python
# coding: utf-8

# # Task: 문서를 보고 category2를 예측

# **학습 데이터에 대한 통계정보 시각화**

# # **전체를 한번에 보고 바로 소분류 예측**

# In[2]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install scikit-learn')


# In[3]:


import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers import BertTokenizer

from sklearn.metrics import precision_recall_fscore_support


# In[4]:


def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[5]:


train_path = "train.csv"
test_path = "test.csv"
submission_path = ""


# In[6]:


df = pd.read_csv(train_path, encoding='utf-8')

for cate1 in df['category1'].unique():
    condition = df['category1'] == cate1
    words = set()
    tokenized_words = set()
#     print(condition)
#     for sent in df[condition]['text'].values:
#         tokenized_words.update(word_tokenize(sent))  
#         words.update(sent.split(" "))
#     print(f"{cate1}의 토크나이즈 전: {len(words)} -> 토크나이즈 후: {len(tokenized_words)}")

labels_dictionary = {k: v for k,v in zip(df['category2'].unique(), range(0,len(df['category2'].unique())))}
labels_dictionary_reverse = {v: k for k,v in zip(df['category2'].unique(), range(0,len(df['category2'].unique())))}


# # DistilBERT
# - DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher.
# - BERT 시도 후, DistilBERT 모델로 성능 향상 후 최종 제출하였음.
# 
# # 참고자료 출처
# - huggingface trainer 사용법 : https://huggingface.co/docs/transformers/training#train
# - koBERT를 활용한 한국어 데이터 문장 관계 Baseline : https://dacon.io/competitions/official/235875/codeshare/4520

# In[7]:


MODEL_NAME = 'distilbert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

print(len(df['category2'].unique()))

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
    num_labels=len(df['category2'].unique()),
    label2id=labels_dictionary,
    id2label=labels_dictionary_reverse)


# In[1]:


X = df['text']
y = df['category2']
class_weights = (1 - (y.value_counts().sort_index() / len(df))).values
class_weights = torch.from_numpy(class_weights).float().to(device)
class_weights # Imbalanced classes 조정


# In[8]:


X_train, X_valid = train_test_split(df, test_size=0.2, shuffle=True)

tokenized_train = tokenizer(
    list(X_train.text),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

tokenized_eval = tokenizer(
    list(X_valid.text),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

print(tokenized_train['input_ids'][0])
print(tokenizer.decode(tokenized_train['input_ids'][0]))


# In[9]:


class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, label):
        self.pair_dataset = pair_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['label'] = torch.tensor(self.label[idx])
        
        return item

    def __len__(self):
        return len(self.label)
    
def label_to_num(label):
    label_dict = labels_dictionary
    num_label = []

    for v in label:
        num_label.append(label_dict[v])
    
    return num_label

train_label = label_to_num(X_train['category2'].values)
eval_label = label_to_num(X_valid['category2'].values)


# In[10]:


train_dataset = BERTDataset(tokenized_train, train_label)
eval_dataset = BERTDataset(tokenized_eval, eval_label)

print(train_dataset.__len__())
print(train_dataset.__getitem__(0))
print(tokenizer.decode(train_dataset.__getitem__(0)['input_ids']))
print(train_dataset.__getitem__(0)['label'])
print(tokenizer.decode(train_dataset.__getitem__(0)['label']))


# In[11]:


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


# In[12]:


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'f1': f1
    }


# In[13]:


training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    save_total_limit=5,
    optim="adamw_torch",
    save_steps=100,
    evaluation_strategy='steps',
    eval_steps=100,
    logging_steps=64,
    load_best_model_at_end = True,
    fp16=True # 좀 더 빠른 train이 된다??
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained('./result/best_model')


# In[14]:


df_test = pd.read_csv(test_path, encoding='utf-8')
df_test


# In[16]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Tokenizer_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MODEL_NAME = './result/checkpoint-4700'
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(df['category2'].unique()),
    label2id=labels_dictionary,
    id2label=labels_dictionary_reverse,
)


# In[17]:


trainer.evaluate()


# In[18]:


df_test = pd.read_csv(test_path, encoding='utf-8')
df_test


# In[19]:


import torch, gc
gc.collect()
torch.cuda.empty_cache()

output_pred = []
output_prob = []
model.eval()
txt_test = [tokenizer(_, return_tensors="pt", max_length=256, truncation=True, padding=True) for _ in list(df_test['text'])]
logits_list = []
for i in tqdm(txt_test):
    with torch.no_grad():
        outputs = model(**i).logits
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)


# In[ ]:


answers = []
for i in output_pred:
    answers.append(labels_dictionary_reverse[i])
answers


# In[ ]:


df_test['category2'] = answers
df_test


# In[ ]:


df_test.to_csv(submission_path+'submission_11.csv', index=False, columns=['id', 'category2'])

