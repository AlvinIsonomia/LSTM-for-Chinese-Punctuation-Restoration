#!/home/data/liuchang/anaconda3/envs/py3/bin/python
# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import os 
import os.path
import gensim    
import torch
from torch.utils.data import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import jieba

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
model = gensim.models.Word2Vec.load('/home/data/liuchang/data/Tinywikizh.model')
lstm.load_state_dict(torch.load('tlstm.pkl'))


def print_punc(sentence):
    senseg = ' '.join(jieba.cut(sentence,cut_all=False))
#     print(senseg)
    zerovec = np.zeros(200)
    xpara = []
    for word in senseg.split():
        try:
            xpara.append(model[word])
        except:
            xpara.append(zerovec)
            # torch.load(lstm.state_dict(), 'tlstm.pkl')
    x = torch.FloatTensor(xpara)
    with torch.no_grad():
        lstm.hidden = lstm.init_hidden()
        tag_scores = lstm(x)
    # y = lstm(xpara[0])
#     print('raw sentence:',sentence)
    a,index = torch.max(tag_scores,dim = 1) 
#     print('punc index:',index)
    punc_dict = {0:'',1:'，',2:'。',3:'！',4:'？',5:'、'}
    word = senseg.split()
    puncsen = ''
    for i in range(0,len(tag_scores)):
        puncsen += word[i]+punc_dict[int(index[i])]
    print(puncsen)

while(True):
    sentence = input("输入待断句的部分：\n")
    print_punc(sentence)
