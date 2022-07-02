# wxy
'''Usage:
    3-gram.py --cuda=<int> train --hidden_size=<int> --embedding_dim=<int> --context_size=<int>  [options]
    3-gram.py decode --model=""
'''
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn.functional as F
from torch import nn, optim
import h5py
import sys
from docopt import docopt
import time



class Predict(nn.Module):

    def __init__(self, vsize, hidden_size, embed_size, context_size):
        super(Predict, self).__init__()

        self.hidden_size = hidden_size
        self.embed_dim = embed_size
        self.vsize = vsize
        self.ngram = context_size
        self.embedding = nn.Embedding(self.vsize, self.embed_dim)  # 构建Embedding
        self.net = nn.Sequential(  # 封装容器(模型执行顺序)
            nn.Linear(self.embed_dim * 3, self.hidden_size),
            nn.GELU(),  # 激活函数
            nn.Linear(self.hidden_size, self.embed_dim, bias=False)
        )
        self.classifier = nn.Linear(self.embed_dim, self.vsize)

        self.classifier.weight = self.embedding.weight  # 将Linear的weight绑定Embedding的weight

    def forward(self, input):  # input是输入的语料 几行几个词
        output = self.embedding(input)  # 将输入数据转换成 embedding类型 # 得到词嵌入 torch.Size([9, 256, 32])
        _l = []
        ndata = input.size(-1) - self.ngram  # input.size是一句话有多少个词 一已知3个 预测下一个
        for i in range(self.ngram):
            temp = output.narrow(1, i, ndata)  # torch.Size([9, 253, 32])
            _l.append(temp)
        _input_net = torch.cat(_l, dim=-1)  # torch.Size([9, 253, 96]) 9行 253组 每组三个词预测下一个值
        output = self.classifier(self.net(_input_net))  # 再一次linear

        # output = F.log_softmax(output, dim=2)  # 把多维向量压缩到（0,1）得到概率分布，最大化条件概率

        # print(output.shape)  out: torch.Size([9, 253, 51802]) 返回所以的值是看哪个值最大 最大的就是最有可能的下一个词

        return output

    @staticmethod
    def load(model_path: str):

        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']

        model = Predict(**args)
        model.load_state_dict(params['state_dict'])  # 加载之前存储的数据

        return model

    def save(self, path: str):

        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {  #相应参数的设定
            'args': dict(vsize=self.vsize, hidden_size=self.hidden_size, embed_size=self.embed_dim,
                         context_size=self.ngram),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

    def decode(self, input):

        emb = self.embedding(input)  # 将输入数据转换成 embedding类型 # 得到词嵌入 #torch.Size([3, 32])

        emb = emb.view(1, -1)  # 将词向量拼在一起 # view()函数作用是将一个多行的Tensor,拼接成一行。 #torch.Size([1, 96])

        output = self.net(emb)  # 两次linear torch.Size([1, 32])

        output = self.classifier(output)  # 再一次linear torch.Size([1, 80]) 出来的结果是分数

        # output = F.log_softmax(output, dim=1)  # 把多维向量压缩到（0,1）得到概率分布，最大化条件概率 torch.Size([1, 80])
        return output


# 返回一个batch和对应的target
def make_target(data, ndata):

    for i in range(ndata):
        batch = data[str(i)][:]
        batch = torch.LongTensor(batch)
        target = batch.narrow(1, 3, batch.shape[1] - 3)  # [bsize,seql-3]
        yield batch, target  # 测试数据 正确结果


def train(args: Dict):

    model_save_path = "model.bin"
    with h5py.File("result.hdf5", 'r') as f:
        nword = f['nword'][()] + 2
        ndata = f['ndata'][()]
        data = f['group']
        torch.manual_seed(0)  # 固定随机种子

        model = Predict(vsize=nword, hidden_size=int(args['--hidden_size']), embed_size=int(args['--embedding_dim']),
                        context_size=int(args['--context_size']))  # 模型初始化
        # # model = Predict(vsize = nword , hidden_size = 128, embed_size = 32,context_size = 3)  # 模型初始化
        # model = Predict.load(model_save_path)
        # model.eval()

        model.train()

        for param in model.parameters():  # model.parameters()保存的是Weights和Bais参数的值
            torch.nn.init.uniform_(param, a=-0.001, b=0.001)  # 给weight初始化

        device = torch.device("cuda:0" if args['--cuda'] else "cpu")  # 分配设备

        print('use device: %s' % device, file=sys.stderr)
        if args['--cuda']:
            model = model.to(device)  # 放置数据
        Loss = nn.CrossEntropyLoss(ignore_index = 0,reduction = 'sum' )  # 损失函数初始化
        if args['--cuda']:
            Loss = Loss.to(device)

        optimizer = torch.optim.Adam(model.parameters()) # 优化函数初始化 学习率
        # 学习率更新
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10,
                                                               min_lr=1e-7)  #在发现loss不再降低或者acc不再提高之后，降低学习率 触发条件后lr*=factor；
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
                out = model(batch)
                out = out.transpose(1,2)
                loss = Loss(out, target)
                # backward
                loss.backward()
                if cuda % 10 == 0:
                    optimizer.step()  # 更新所有参数
                    optimizer.zero_grad()  # 将模型的参数梯度初始化为0
                if cuda % 100 == 0:  # 打印loss
                    print("This is {0} epoch,This is {1} batch".format(epoch,cuda), 'loss = ','{:.6f}'.format(loss))
                if cuda % 1000 == 0: #更新学习率
                    scheduler.step(loss)
                if cuda % 2000 == 0:  # 保存模型
                    print('save currently model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)




def decode(args: Dict):
    model_save_path = args['--model']
    input = ['I', 'should', 'like']
    with open("dict.txt", "rb") as file:  # 读数据
        words = eval(file.readline().strip().decode("utf-8"))

    words_re = {i: w for w, i in words.items()}
    input = [words[i] for i in input]

    model = Predict.load(model_save_path)
    model.eval()

    device = torch.device("cuda:0")  # 在cuda上运行
    print('use device: %s' % device, file=sys.stderr)
    # model = model.to(device)
    # input = input.to(device)
    with torch.no_grad():
        for i in range(50):  # 解码50个词
            d = input[i:i + 3]
            # d = torch.LongTensor(d).to(device)
            d = torch.LongTensor(d)
            out = torch.argmax(model.decode(d))
            input.append(out.item())

        out = [words_re[k] for k in input]  # sentence
    output = "".join(out).replace("@@ ", "")
    print(output)


if __name__ == '__main__':

    args = docopt(__doc__)
    if args['--cuda']:
        torch.cuda.manual_seed(1)  #为特定GPU设置种子，生成随机数
    if args['train']:  # 跑模型：
        train(args)
    elif args['decode']:  # 解码：
        decode(args)
    else:
        raise RuntimeError('invalid run mode')

