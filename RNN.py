# wxy
# RNNs每次只输入一个单词，按照句子的顺序逐个输入，同时维护一个或多个hidden（RNN：1个，hidden，LSTM：2个，hidden & cell）来存储前面输入单词的信息：
# ht = model(h_{t-1},i_t}
# 3 * 32 -> 128 ->32
# 32+32->128->32
# h_0 nn.Parameter
# 输入隐状态和当前词向量 形成新的隐状态

'''Usage:
    RNN.py --cuda=<int> train --embedding_dim=<int> --hidden_dim=<int>  [options]
    RNN.py decode --model="" --n=<int>
'''
import random
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn.functional as F
from torch import nn, optim
import h5py
import sys
from docopt import docopt
import time


class RNN(nn.Module):
    def __init__(self, vsize, embedding_dim,hidden_dim):
        super(RNN, self).__init__()

        self.embed_dim = embedding_dim
        self.vsize = vsize
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(self.vsize, self.embed_dim)  # 构建Embedding
        self.net = nn.Sequential(  # 封装容器(模型执行顺序) 隐藏层
            nn.Linear(self.embed_dim + self.hidden_dim, hidden_dim),
            nn.GELU(),  # 激活函数
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, self.embed_dim, bias=False),
            nn.Linear(self.embed_dim, self.vsize),
        )
        self.classifier[-1].weight = self.embed.weight  # 将Linear的weight绑定Embedding的weight

    def forward(self, input,hidden):
        seql = input.size(1)
        input = self.embed(input)  # (bsize, seq_length, embed_dim)
        output = []
        for i in input.unbind(1):
            data = torch.cat([i, hidden], dim=-1)  # (bsize,embed_dim+embed_dim)
            hidden = self.net(data)  # (bsize,32)
            output.append(hidden)
        output = self.classifier(torch.stack(output,dim=1)) #(bsize,seql,vsize)
        return output

    def init_hidden(self, input_x, input_y):
        return torch.nn.Parameter(torch.zeros(input_x, input_y))


    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']

        model = RNN(**args)
        model.load_state_dict(params['state_dict'])  # 加载之前存储的数据

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {  # 相应参数的设定
            'args': dict(vsize=self.vsize, embedding_dim=self.embed_dim,hidden_dim=self.hidden_dim),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

    def decode(self, input, hidden,n):
        seql = input.size(0) #(seql)
        input = self.embed(input)  # (seql, embed_dim)
        output = []
        for i in range(seql):
            data = torch.cat([input[i], hidden], dim=-1)  # (embed_dim+embed_dim)
            hidden = self.net(data)  # (32)
        for i in range(seql-1,n):
            data = torch.cat([input[i], hidden], dim=-1)  # (embed_dim+embed_dim)
            hidden = self.net(data)  # (32)
            # print('input:',input.size(),'hidden:',hidden.size())
            input = torch.cat((input, hidden.unsqueeze(0)),dim=0)
        output = self.classifier(input)  # (1,seql,vsize)
        return output


# 返回一个batch和对应的target
def make_target(data, ndata):
    l = list(range(ndata))
    random.shuffle(l)
    for i in l:
        batch = data[str(i)][:]
        batch = torch.LongTensor(batch)
        input = batch.narrow(1, 0, batch.size(1)-1)
        target = batch.narrow(1, 1, batch.size(1)-1)
        yield input, target  # 测试数据 正确结果


def train(args: Dict):
    model_save_path = "model.RNN"
    with h5py.File("result.hdf5", 'r') as f:
        nword = f['nword'][()] + 1
        ndata = f['ndata'][()]
        data = f['group']
        torch.manual_seed(0)  # 固定随机种子

        model = RNN(vsize=nword, embedding_dim=int(args['--embedding_dim']),hidden_dim=int(args['--hidden_dim']))  # 模型初始化

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

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000, gamma=0.1)

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
                hidden = model.init_hidden(batch.size(0), 32).to(device)
                out = model(batch,hidden)
                loss = Loss(out.transpose(1,2), target)
                # backward
                loss.backward()
                if cuda % 10 == 0:
                    optimizer.step()  # 更新所有参数
                    optimizer.zero_grad()  # 将模型的参数梯度初始化为0
                if cuda % 100 == 0:  # 打印loss
                    print("This is {0} epoch,This is {1} batch".format(epoch,cuda), 'loss = ','{:.6f}'.format(loss/nword))
                if cuda % 1000 == 0:  # 更新学习率
                    scheduler.step(loss)
                if cuda % 100 == 0:  # 保存模型
                    print('save currently model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)


def decode(args: Dict):
    model_save_path = args['--model']
    num =int(args['--n'])
    input = ['how', 'are', 'you']
    with open("dict.txt", "rb") as file:  # 读数据
        words = eval(file.readline().strip().decode("utf-8"))
    words_re = {i: w for w, i in words.items()}
    input = [words[i] for i in input]

    model = RNN.load(model_save_path)
    model.eval()

    device = torch.device("cuda:0")  # 在cuda上运行
    print('use device: %s' % device, file=sys.stderr)
    model = model.to(device)
    hidden = torch.zeros(32).to(device)
    result = []
    with torch.no_grad():
        # d = torch.LongTensor(input).to(device)
        d = torch.LongTensor(input).to(device)
        out = model.decode(d,hidden,num)
        out = torch.argmax(out,dim=-1)
    out = out.tolist()
    output = [words_re[k] for k in out]  # sentence
    print(output)
    output = " ".join(output).replace("@@ "," ")
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
