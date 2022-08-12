# wxy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Set, Union
import random
from torch import nn, optim
import h5py
import sys
from docopt import docopt
import time
import math
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
# python train_new.py --cuda=0 decode --model_path="model_trans" --en="result_en_test.hdf5" --target_dict="dict_target.txt"

def read(filename):
    with open(filename, "rb") as file:  # 读数据
        words = eval(file.readline().strip().decode("utf-8"))
    return words

def decode():
    model_path = "model_trans"
    words_en = read("dict_en.txt")
    print(1)
    words_re_en = {i: w for w, i in words_en.items()}
    words_target = read("dict_target.txt")
    print(2)
    words_re_target = {i: w for w, i in words_target.items()}
    print(3)
    model = Transformer.load(model_path)
    print(4)
    model.eval()
    print(5)
    de_predict = []
    device = torch.device("cuda:" + args['--cuda'] if args['--cuda'] else "cpu")  # 分配设备
    print('use device: %s' % device, file=sys.stderr)
    model = model.to(device)
    with h5py.File("result_en_test.hdf5", 'r') as f:
        nword_en = f['nword'][()]
        ndata = f['ndata'][()]
        data_en = f['group']
        torch.manual_seed(0)  # 固定随机种子
        for en_src in make_test(data_en, ndata):
            if args['--cuda']:
                en_src = en_src.to(device)
            en_outputs = model.encoder(en_src)
            de_input = torch.zeros(en_src.size(0), 0).type_as(en_src.data)  # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
            flag = False
            next_symbol = torch.tensor([[words_en["<sos>"]] for i in range(en_src.size(0))])
            flag_test = [0 for i in range(en_src.size(0))]
            flag_true = [1 for i in range(en_src.size(0))]
            while not terminal:
                # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
                de_input = torch.cat([de_input.to(device), torch.tensor([[next_symbol]], dtype=en_src.dtype).to(device)],-1)
                de_outputs= model.decoder(de_input, en_src, en_outputs)
                de_outputs = model.classifier(de_outputs)
                result= torch.argmax(dec_outputs,dim=-1, keepdim=False)
                next_word = result[:, -1].reshape(-1, 1)  # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
                next_symbol = next_word
                for i in range(en_src.size(0)):
                    if next_word[i].item() == words_target["<eos>"]:
                        flag_test[i] = 1
                if flag_test == flag_true:
                    flag = True
                print(next_word)
            de_predict.append(de_input)  # 把结果存入
    with open("result_test_de.txt", 'w') as file:
        l = len(de_predict)
        temp = ""
        for i in range(l):
            for j in range(de_predict[i].size(0)):
                for k in range(de_predict[i].size(1)):
                    temp = temp + str(words_re_target[de_predict[i][j][k].item()]) + ' '
            file.write(temp + '\n')