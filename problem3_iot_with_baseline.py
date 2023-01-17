#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data load
# 
# - 약 350일 에너지 사용량에 대한 데이터
# - 12월 마지막 15일 가량의 데이터 누락
# - 1월 ~ 11월 까지 데이터만 사용해서 실험 진행
# - 매 달 마지막 주의 에너지 사용량을 예측 

# In[2]:


path = "../input/jbnu-bigdata2022-iot/IoT_train.csv"
train = pd.read_csv(path)

train.head(), train.columns, len(train)


# # Data Visualization

# In[3]:


train['temperature'].plot(figsize=(25, 5))


# In[4]:


train['humidity'].plot(figsize=(25, 5))


# In[5]:


train['visibility'].plot(figsize=(25, 5))


# In[6]:


train['pressure'].plot(figsize=(25, 5))


# In[7]:


train['use [kW]'].plot(figsize=(25, 5)) #D calendar day frequency


# # Data proprocessing

# In[8]:


months = {'jan', 'feb', 'mar', 'apr', 'may', 'june', 'aug', 'july', 'sep', 'oct', 'nov', 'dec'}

total = {} # 350일 데이터를 월 단위로 구분하여 저장
total_df = {} # 월 단위 데이터를 pandas 데이터 프레임 형태로 저장


for i in range(len(months)):
    total[i] = []
date = train["time"].tolist()
use = train['use [kW]'].tolist()

for d, u in zip(date, use):
    total[int(d.split("/")[1])-1].append([d, u])

for i in range(len(total)):
    tmp_date = []
    tmp_use = []
    for j in total[i]:
        tmp_date.append(j[0])
        tmp_use.append(j[1])
    total_df[i] = pd.DataFrame({"use": tmp_use}, index=tmp_date)

type(total_df[0]), len(total_df), total_df


# In[9]:


# baseline에서 설정한 데이터 구분 단위 (성능 향상을 위해 수정 필요) !!!!!!!!!
# 일주일 * 24 시간 * 60분, 즉 일주일 단위로 데이터를 구분
# 일주일의 에너지 사용량을 입력받아 다음 주 에너지 사용량을 예측

ran = 7 * 24 * 60

def get_x_y_data(df, infer=False):
    x_data = []
    cnt = 0
    y_data = []
    for i in range(len(df)-ran, 0, -ran):
        x_data.append(np.array(df.iloc[i-ran:i]['use']).astype(float))
        y_data.append(np.array(df.iloc[i:i+ran]['use']).astype(float))
    
    return x_data, y_data


# In[10]:


# x_train(이전 주 예측량), y_train(다음 주 예측량) => 학습 데이터 입력, 라벨
x_train = []
y_train = []

# x_test(매 달 마지막 주의 이전 주 사용량), y_test(예측해야하는 값 99.9999로 설정되어 있음) => 
x_test = []
y_test = []

for i in range(11):
    usedf = total_df[i]
    x, y = get_x_y_data(usedf)
    x_test.append(x[0])
    y_test.append(y[0])
    if i == 1:
        x_train.extend(x[1:])
        y_train.extend(y[1:])
    else:
        x_train.extend(x[1:-1])
        y_train.extend(y[1:-1])    
        
len(x_train), len(x_test)


# In[11]:


x[0], len(x[0]), len(x)


# In[12]:


y[0], y[-1], len(y[-1]), len(y[0])


# In[13]:


from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = None
        if Y is not None:
            self.Y = Y

    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        else:
            return torch.Tensor(self.X[index])

    def __len__(self):
        return len(self.X)


# In[14]:


train_dataset = CustomDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=True, num_workers=0)

test_dataset = CustomDataset(x_test)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)


# In[15]:


del train
del x_train
del y_train
del x
del y
del months
del total
del total_df


# ## Model

# In[16]:


# 모델 정의 (성능 향상을 위해 hyperparameter, architecture 수정 필요)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.lstm = nn.LSTM(input_size=10080, hidden_size=256, num_layers=1, batch_first=True)
        
        self.multioutput_reg = nn.Sequential(
            nn.Linear(in_features=256, out_features=5040), 
            nn.ReLU(),
            nn.Linear(in_features=5040, out_features=10080),
        )
        
    def forward(self, x):
        hidden, _ = self.lstm(x)
        output = self.multioutput_reg(hidden)
        return output


# ## Train

# In[17]:


def train(model, optimizer, train_loader, scheduler, device):
    model.to(device)
    criterion = nn.L1Loss().to(device)
    
    best_loss = 9999999
    best_model = None
    
    for epoch in range(1, 5):
        model.train()
        train_loss = []
        for X, Y in iter(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            
            output = model(X)
            loss = criterion(output, Y)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        _train_loss = np.mean(train_loss)
        print(_train_loss)
        
#         val_mae = validation(model, val_loader, criterion, device)
#         print(f'Epoch : [{epoch}] Train Loss : [{_train_loss:.5f}] Val MAE : [{val_mae:.5f}]')
        
#         if scheduler is not None:
#             scheduler.step(val_mae)
            
#         if best_loss > val_mae:
#             best_loss = val_mae
#             best_model = model 
    return model


# In[18]:


def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for X, Y in iter(val_loader):
            X = X.to(device)
            Y = Y.to(device)
            
            output = model(X)
            
            loss = criterion(output, Y)
            
            val_loss.append(loss.item())
    
    _val_loss = np.mean(val_loss)
    return _val_loss    


# In[19]:


model = BaseModel()

## Experiments Setup
# hyperparameter 정의 (성능 향상을 위해 hyperparameter, loss function, optimizer 수정 필요)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-5) # lr 수정
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4,threshold_mode='abs',min_lr=1e-8, verbose=True)
best_model = train(model, optimizer, train_loader, scheduler, device)


# In[20]:


def inference(model, test_loader):
    model.to(device)
    model.eval()
    pred = []
    with torch.no_grad():
        for X in iter(test_loader):
            X = X.to(device)
            output = model(X)
            pred.extend(output.cpu().tolist())
    return pred        


# In[21]:


pred = inference(model, test_loader)

# 11 * 10080 -> 11개월 마지막 주 예측량
len(pred), len(pred[0])


# In[22]:


path = "../input/jbnu-bigdata2022-iot/IoT_sample_submission.csv"
test = pd.read_csv(path)

test.head(), test.columns, len(test)


# In[23]:


# sample_submission 파일에서 예측해야 하는 달의 마지막 주 예측량이 16.0으로 설정되어 있음
# 16.0 값을 모델에 예측한 prediction으로 수정

use = test["use [kW]"].tolist()

for i in range(11):
    start = use.index(16.0)
    use[start:start+ran] = pred[i]


# In[24]:


test["use [kW]"] = use

test.head()


# In[25]:


test.to_csv('./submit.csv', index=False)

