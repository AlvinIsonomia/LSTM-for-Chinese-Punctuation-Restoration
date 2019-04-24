#!/home/data/liuchang/anaconda3/envs/py3/bin/python
# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os 
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import sys
import multiprocessing
import gensim    
import torch
from torch.utils.data import *
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
# xlist = np.load('/home/data/liuchang/data/wikiw2v.npy') 
# ylist = np.load('/home/data/liuchang/data/wikipunc.npy') 
# del xlist
# del ylist
# gc.collect()

# dataloader
batch_size = 1
class wikiset(Dataset):
    def __init__(self):
        print('dataset loading...')
        self.xlist = np.load('/home/data/liuchang/data/wikiw2v.npy') 
        self.ylist = np.load('/home/data/liuchang/data/wikipunc.npy') 
        
    def __len__(self):
        return len(self.ylist)
    def __getitem__(self,idx):
        return torch.FloatTensor(self.xlist[idx]),torch.FloatTensor(self.ylist[idx])

teset = wikiset()        
seqloader = DataLoader(teset,batch_size=1,shuffle = False ,num_workers = 4)            

class TLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(TLSTM,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim,hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return torch.zeros(1,1,self.hidden_dim),torch.zeros(1,1,self.hidden_dim)

    def forward(self,inputs):
        lstm_out, self.hidden = self.lstm(inputs.view(len(inputs),1,-1),self.hidden)
        tag_space = self.fc(lstm_out.view(len(inputs),-1))
        tag_scores = F.softmax(tag_space,dim = 1)
        return tag_scores
lstm = TLSTM(200,100,6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_function = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters())
# lstm = nn.DataParallel(lstm)
lstm.to(device)
# idata = iter(seqloader)
totalloss = []
for epoch in range(0,50):
    for batch in range(0,50):
        sumloss = 0
    #     for i in tqdm(range(0,100)):
        for i in range(200*batch,200*(batch+1)):        
            x, y = teset[i]
            lstm.zero_grad() # 清空梯度
            lstm.hidden = lstm.init_hidden() # 清空隐层状态
            tag_scores = lstm(x) # 前向计算
            loss = loss_function(tag_scores,y.float()) # 计算损失函数
            loss.backward()
            optimizer.step()
        sumloss += loss.data
        totalloss.append(sumloss)
        torch.save(lstm.state_dict(), 'tlstm.pkl')
        print('batch',batch,'epoch',epoch,'total loss in this epoch:',sumloss)   
