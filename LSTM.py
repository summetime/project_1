# wxy

'''Usage:
    LSTM.py --cuda=<int> train --embedding_dim=<int> --hidden_dim=<int>  [options]
    LSTM.py --cuda=<int> decode --model="" --n=<int>
'''
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn.functional as F
from torch import nn, optim
import h5py
import sys
from docopt import docopt
import time


class LSTM(nn.Module):
    def __init__(self, vsize, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.embed_dim = embedding_dim
        self.vsize = vsize
        self.hidden_dim = hidden_dim
        self.lstm = nn.Sequential(
            nn.Linear(self.embed_dim + self.hidden_dim, self.hidden_dim * 4),
            nn.LayerNorm((4, self.hidden_dim)),  # 归一化
        )
        self.net = nn.Sequential(
            nn.Embedding(self.vsize, self.embed_dim),
            nn.Dropout(p=0.1),  # dropout训练
            nn.Linear(self.hidden_dim, self.vsize),
        )
        self.sig = nn.Sigmoid()
        self.tanh = nn.GELU()
        self.net[-1].weight = self.net[0].weight  # 将Linear的weight绑定Embedding的weight

    def forward(self, input, hidden, cell):
        seql = input.size(1)
        input = self.net[0](input)  # (batch_size, seq_length, embed_dim)
        output = []
        for i in range(seql):
            data = torch.cat([input.narrow(1, i, 1), hidden], dim=-1)  # (bsize,1,embed_dim+embed_dim)
            # input gate:(batch_size, 1, embed_dim)
            out = self.lstm[0](data)
            size = list(out.size())
            size[-1] = 4
            size.append(32)  # size:(bsize,1,4,32)
            i, f, h, o = self.lstm[1](out.view(size)).unbind(-2)
            i = self.net[1](self.sig(i))  # 决定保留多少
            f = self.net[1](self.sig(f))  # 遗忘门 决定丢弃多少
            h = self.net[1](self.tanh(h))
            o = self.net[1](self.sig(o))
            cell = f * cell + i * h  # 输入门
            hidden = o * cell  # 下一层隐状态
            output.append(hidden)
        output = self.net[-1](torch.cat(output, dim=1))  # (bsize,seql,vsize)
        return output

    def init_hidden(self, input_x, input_y):
        return torch.nn.Parameter(torch.zeros(input_x, input_y, 32))

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']

        model = LSTM(**args)
        model.load_state_dict(params['state_dict'])  # 加载之前存储的数据

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {  # 相应参数的设定
            'args': dict(vsize=self.vsize, embedding_dim=self.embed_dim, hidden_dim=self.hidden_dim),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


    def decode(self, input, hidden, cell,num):
        seql = input.size(0)
        input = self.net[0](input)  # (seql, embed_dim)
        output = []
        for i in range(seql):
            data = torch.cat([input[i], hidden], dim=-1)  # (embed_dim+embed_dim)
            out = self.lstm[0](data)
            size = list(out.size())
            size[-1] = 4
            size.append(32)  # size:(bsize,1,4,32)
            i, f, h, o = self.lstm[1](out.view(size)).unbind(-2)
            i = self.net[1](self.sig(i))  # 决定保留多少
            f = self.net[1](self.sig(f))  # 遗忘门 决定丢弃多少
            h = self.net[1](self.tanh(h))
            o = self.net[1](self.sig(o))
            cell = f * cell + i * h  # 输入门
            hidden = o * cell  # 下一层隐状态
        for i in range(seql-1, num):
            data = torch.cat([input[i], hidden], dim=-1)  # (embed_dim+embed_dim)
            out = self.lstm[0](data)
            size = list(out.size())
            size[-1] = 4
            size.append(32)  # size:(bsize,1,4,32)
            i, f, h, o = self.lstm[1](out.view(size)).unbind(-2)
            i = self.net[1](self.sig(i))  # 决定保留多少
            f = self.net[1](self.sig(f))  # 遗忘门 决定丢弃多少
            h = self.net[1](self.tanh(h))
            o = self.net[1](self.sig(o))
            cell = f * cell + i * h  # 输入门
            hidden = o * cell  # 下一层隐状态
            input = torch.cat((input, hidden.unsqueeze(0)), dim=0)
        output = self.net[-1](input)  # (bsize,seql,vsize)
        return output


# 返回一个batch和对应的target
def make_target(data, ndata):
    for i in range(ndata):
        batch = data[str(i)][:]
        batch = torch.LongTensor(batch)
        input = batch.narrow(1, 0, batch.shape[1] - 1)
        target = batch.narrow(1, 1, batch.shape[1] - 1)
        yield input, target  # 测试数据 正确结果


def train(args: Dict):
    model_save_path = "model.LSTM"
    with h5py.File("result.hdf5", 'r') as f:
        nword = f['nword'][()] + 2
        ndata = f['ndata'][()]
        data = f['group']
        torch.manual_seed(0)  # 固定随机种子

        model = LSTM(vsize=nword, embedding_dim=int(args['--embedding_dim']),
                     hidden_dim=int(args['--hidden_dim']))  # 模型初始化

        model.train()

        for param in model.parameters():  # model.parameters()保存的是Weights和Bais参数的值
            torch.nn.init.uniform_(param, a=-0.001, b=0.001)  # 给weight初始化

        device = torch.device("cuda:0" if args['--cuda'] else "cpu")  # 分配设备

        print('use device: %s' % device, file=sys.stderr)
        if args['--cuda']:
            model = model.to(device)  # 放置数据
        Loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')  # 损失函数初始化
        if args['--cuda']:
            Loss = Loss.to(device)

        optimizer = torch.optim.Adam(model.parameters())  # 优化函数初始化 学习率
        # 学习率更新
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10,
                                                               min_lr=1e-7)  # 在发现loss不再降低或者acc不再提高之后，降低学习率 触发条件后lr*=factor；

        cuda = 0
        print('start training')
        for epoch in range(20):
            cuda = 0
            for batch, target in make_target(data, ndata):
                if args['--cuda']:
                    batch = batch.to(device)
                    target = target.to(device)
                # forward
                cuda += 1
                hidden = model.init_hidden(batch.size(0), 1).to(device)
                cell = hidden
                out = model(batch, hidden, cell)
                loss = Loss(out.transpose(1, 2), target)
                # backward
                loss.backward()
                if cuda % 10 == 0:
                    optimizer.step()  # 更新所有参数
                    optimizer.zero_grad()  # 将模型的参数梯度初始化为0
                if cuda % 100 == 0:  # 打印loss
                    print("This is {0} epoch,This is {1} batch".format(epoch, cuda), 'loss = ', '{:.6f}'.format(loss/nword))
                if cuda % 1000 == 0:  # 更新学习率
                    scheduler.step(loss)
                if cuda % 1000 == 0:  # 保存模型
                    print('save currently model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)


def decode(args: Dict):
    model_save_path = args['--model']
    num = int(args['--n'])
    input = ['I', 'just', 'signed']
    with open("dict.txt", "rb") as file:  # 读数据
        words = eval(file.readline().strip().decode("utf-8"))
    words_re = {i: w for w, i in words.items()}
    input = [words[i] for i in input]

    model = LSTM.load(model_save_path)
    model.eval()

    # device = torch.device("cuda:0")  # 在cuda上运行
    device = torch.device("cuda:" + args['--cuda'] if args['--cuda'] else "cpu")  # 分配设备
    print('use device: %s' % device, file=sys.stderr)
    model = model.to(device)
    hidden = torch.zeros(32,device=device)
    cell = torch.zeros(32,device=device)
    result = []
    with torch.no_grad():
        d = torch.LongTensor(input).to(device)
        out = model.decode(d, hidden,cell, num)
        out = torch.argmax(out, dim=-1)
    out = out.tolist()
    output = [words_re[k] for k in out]  # sentence
    print(output)
    output = " ".join(output).replace("@@ ", "")
    print(output)


if __name__ == '__main__':

    args = docopt(__doc__)
    if args['--cuda']:
        torch.cuda.manual_seed(1)  # 为特定GPU设置种子，生成随机数
    if args['train']:  # 跑模型：
        train(args)
    elif args['decode']:  # 解码：
        decode(args)
    else:
        raise RuntimeError('invalid run mode')
