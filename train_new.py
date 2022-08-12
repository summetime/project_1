# wxy
'''Usage:
    train.py --cuda=<int> train --en="" --de="" --target="" --model_save_path="" --embedding_dim=<int> --N=<int> --heads=<int> --dropout=<float>  [options]
    train.py --cuda=<int> decode --model_path="" --en="" --de_dict="" --target_dict=""
'''
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


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, d_k, mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        s = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充s（用-1e9填充s中与mask中值为1位置相对应的元素）
        # mask掉那些为了padding长度增加的token，让其通过softmax计算后为0
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e9)

        p = F.softmax(s, dim=-1)  # 对最后一个维度(v)做softmax
        # s : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        output = torch.matmul(p, V)  # output: [batch_size, n_heads, len_q, d_v]

        return output


class FeedForward(nn.Module):

    def __init__(self, embedding_dim, device, d_ff=2048):
        super().__init__()

        self.embed = embedding_dim
        self.device = device

        # We set d_ff as a default to 2048
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, embedding_dim),
        )
        self.drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        residual = input
        output = self.net(input)
        output = self.norm(self.drop(output) + residual)
        return output.to(self.device) # [bsize, seql, embedding_dim]
        # return nn.LayerNorm(self.embed).to(self.device)(output + residual)  # [bsize, seql, embedding_dim]


class Mask():
    def __init__(self):
        super(Mask, self).__init__()

    def en_mask(self, seq_q, seq_k):
        # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
        """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
        encoder和decoder都可能调用这个函数，所以seq_len视情况而定
        seq_q: [bsize, sqel]
        seq_k: [bsize, sqel]
        """
        bsize, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
        bsize, len_k = seq_k.size()
        mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], 原来为0的设为true其他false
        return mask.expand(bsize, len_q, len_k)  # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)

    def de_mask(self, seq):
        """
        seq: [batch_size, tgt_len]
        """
        subsequence_mask = np.triu(np.ones((seq.size(0), seq.size(1), seq.size(1))),k=1)  # 生成一个上三角矩阵 [batch_size, tgt_len, tgt_len]
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 数组换成tensor
        return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, embedding_dim, device):
        super().__init__()

        self.embed = embedding_dim  # 512
        self.d_k = embedding_dim // heads  # 512 / 8
        self.h = heads  # 8
        self.device = device

        self.q_linear = nn.Linear(embedding_dim, heads * self.d_k)
        self.v_linear = nn.Linear(embedding_dim, heads * self.d_k)
        self.k_linear = nn.Linear(embedding_dim, heads * self.d_k)
        self.attention = ScaledDotProductAttention()

        self.out = nn.Linear(heads * self.d_k, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        residual, bsize = q, q.size(0)
        # q:[bsize,nquery,embedding_dim] k:[bsize,sqel,embedding_dim] v:[bsize,sqel,embedding_dim]
        # 将Q K V分为8个部分 分别计算
        k = self.k_linear(k).view(bsize, -1, self.h, self.d_k).transpose(1,2)  # [bsize,nquery,heads,embedding_dim/8] → [bsize,heads,nquery,embedding_dim/8]
        q = self.q_linear(q).view(bsize, -1, self.h, self.d_k).transpose(1,2)  # [bsize,sqel,heads,embedding_dim/8] → [bsize,heads,sqel,embedding_dim/8]
        v = self.v_linear(v).view(bsize, -1, self.h, self.d_k).transpose(1,2)  # [bsize,sqel,heads,embedding_dim/8] → [bsize,heads,sqel,embedding_dim/8]
        # mask矩阵扩充成4维
        # mask: [bsize, seql, seql] -> [bsize, 1, seql, seql]->[bsize,n_heads, seql, seql]
        mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1) #扩充之后在n_heads出重复填充

        ss = self.attention(q, k, v, self.d_k, mask) # [batch_size, n_heads, len_q, d_v]
        # 拼接
        concat = ss.transpose(1, 2).contiguous().view(bsize, -1, self.h *self.d_k) #维度转换 → contiguous()拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列
        output = self.out(concat) #过linear
        output = self.norm(self.drop(output) + residual)
        return output.to(self.device)
        # return nn.LayerNorm(self.embed).to(self.device)(output + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=1100):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)  # 位置编码矩阵，维度[max_len, embedding_dim]
        position = torch.arange(0.0, max_len).unsqueeze(1)  # 单词位置  [max_len,1]
        div_term = torch.exp(torch.arange(0.0, embedding_dim, 2) * (- math.log(1e4) / embedding_dim))  # 使用exp和log实现幂运算 [1,embedding_dim/2]
        pe[:, 0:: 2] = torch.sin(position * div_term)  # 隔行处理
        pe[:, 1:: 2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 增加批次维度，[1, max_len, embedding_dim]
        self.register_buffer('pe', pe)  # 将位置编码矩阵注册为buffer(不参加训练)

    def forward(self, x):  # 将一个批次中语句所有词向量与位置编码相加
        """
            x: [batch_size,seq_len,embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):

    def __init__(self, embedding_dim, heads,device):
        super().__init__()

        self.attn = MultiHeadAttention(heads, embedding_dim, device)
        self.ff = FeedForward(embedding_dim, device)

    def forward(self, input, mask):
        """E
            inputs: [bsize, sqel, embedding_dim]
            mask: [bsize, sqel, seql]  mask矩阵(pad mask or sequence mask)
        """
        # outputs: [bsize, sqel, embedding_dim], attn: [bsize, n_heads, sqel, sqel]
        output = self.attn(input, input, input, mask)  # enc_inputs to same Q,K,V（未线性变换前）
        output = self.ff(output)  # output: [bsize, sqel, embedding_dim]
        return output


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, dropout,device):
        super().__init__()
        self.N = N
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, heads,device) for _ in range(N)])
        self.mask = Mask()
        self.device = device

    def forward(self, input):
        """
            enc_inputs: [bsize, sqel]
        """
        x = self.embed(input)  # [bsize, sqel, embedding_dim]
        x = self.pe(x.transpose(0, 1)).transpose(0, 1)  # 位置加输入
        mask = self.mask.en_mask(input, input)  # [bsize, sqel, sqel]
        for layer in self.layers:
            x = layer(x, mask)  # [bsize, sqel, embedding_dim], mask: [bsize, n_heads, sqel, sqel]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads, device):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(heads, embedding_dim,device)
        self.en_attn = MultiHeadAttention(heads, embedding_dim, device)
        self.pos_ffn = FeedForward(embedding_dim, device)

    def forward(self, tgt_inputs, src_outputs, self_attn_mask, attn_mask):
        """
        tgt_inputs: [bsize, tgt_len, embedding] decoder输入
        src_outputs: [bsize, src_len, embedding] encoder输出
        self_attn_mask: [bsize, tgt_len, tgt_len]
        attn_mask: [bsize, tgt_len, src_len]
        """
        outputs= self.self_attn(tgt_inputs, tgt_inputs, tgt_inputs,self_attn_mask)
        outputs = self.en_attn(outputs, src_outputs, src_outputs,attn_mask)  # Q:decoder  K,V:encoder
        outputs = self.pos_ffn(outputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embedding_dim, N, heads, dropout,device):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, embedding_dim)  # Decoder词表
        self.pos_emb = PositionalEncoding(embedding_dim, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, heads,device) for _ in range(N)])
        self.mask = Mask()
        self.device = device

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, embedding]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, embedding]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(self.device)  # [batch_size, tgt_len, embedding]
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        enc_attn_mask = self.mask.en_mask(dec_inputs, dec_inputs).to(self.device)  # [batch_size, tgt_len, tgt_len]
        # print(enc_attn_mask)
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_mask = self.mask.de_mask(dec_inputs).to(self.device)  # [batch_size, tgt_len, tgt_len]
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_mask + enc_attn_mask),0).to(self.device)  # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回，否则返回0
        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        dec_enc_attn_mask = self.mask.en_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask,dec_enc_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs


class Transformer(nn.Module):

    def __init__(self, src_vocab, target_vocab, embedding_dim, N, heads, dropout, device):
        super().__init__()
        self.s_vocab = src_vocab
        self.t_vocab = target_vocab
        self.embed = embedding_dim
        self.N = N
        self.heads = heads
        self.dropout = dropout
        self.encoder = Encoder(src_vocab, embedding_dim, N, heads, dropout,device).to(device)
        self.decoder = Decoder(target_vocab, embedding_dim, N, heads, dropout,device).to(device)
        self.classifier = nn.Linear(embedding_dim, target_vocab,bias=False).to(device)
        self.device = device

    def forward(self, src, target):
        """Transformers的输入：两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # enc_outputs: [batch_size, src_len, embedding_dim],
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, embedding_dim]
        en_outputs= self.encoder(src)
        # dec_outputs: [batch_size, tgt_len, d_model]
        de_outputs= self.decoder(target, src, en_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> [batch_size, tgt_len, tgt_vocab_size]
        de_outputs = F.softmax(self.classifier(de_outputs))
        return de_outputs

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']

        model = Transformer(**args)
        model.load_state_dict(params['state_dict'],False)  # 加载之前存储的数据

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {  # 相应参数的设定
            'args': dict(src_vocab=self.s_vocab, target_vocab=self.t_vocab, embedding_dim=self.embed, N=self.N,
                         heads=self.heads, dropout=self.dropout, device=self.device),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


# 返回一个batch和对应的target
def make_target(data_en, data_de,data_target, ndata):
    l = list(range(ndata))
    random.shuffle(l)
    for i in l:
        en_src = data_en[str(i)][:]
        de_src = data_de[str(i)][:]
        target = data_target[str(i)][:]
        en_src = torch.LongTensor(en_src)
        de_src = torch.LongTensor(de_src)
        target = torch.LongTensor(target)
        yield en_src,de_src, target  # 输入 输出  正确结果

def make_test(data_en,ndata):
    l = list(range(ndata))
    random.shuffle(l)
    for i in l:
        en_src = data_en[str(i)][:]
        en_src = torch.LongTensor(en_src)
        yield en_src  # 输入 输出  正确结果


def train(args: Dict):
    model_save_path = args['--model_save_path']
    with h5py.File(args['--en'], 'r') as f1, h5py.File(args['--de'], 'r') as f2,h5py.File(args['--target'], 'r') as f3:
        nword_en = f1['nword'][()]
        nword_de = f2['nword'][()]
        ndata = f1['ndata'][()]
        data_en = f1['group']
        data_de = f2['group']
        data_target = f3['group']
        torch.manual_seed(0)  # 固定随机种子

        device = torch.device("cuda:" + args['--cuda'] if args['--cuda'] else "cpu")  # 分配设备

        model = Transformer(src_vocab=nword_en, target_vocab=nword_de, embedding_dim=int(args['--embedding_dim']),
                            N=int(args['--N']), heads=int(args['--heads']), dropout=float(args['--dropout']),
                            device=device)  # 模型初始化

        model.train()

        for param in model.parameters():  # model.parameters()保存的是Weights和Bais参数的值
            torch.nn.init.uniform_(param, a=-0.001, b=0.001)  # 给weight初始化

        print('use device: %s' % device, file=sys.stderr)
        if args['--cuda']:
            model = model.to(device)  # 放置数据
        Loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')  # 损失函数初始化

        if args['--cuda']:
            Loss = Loss.to(device)

        optimizer = torch.optim.Adam(model.parameters(),betas = (0.9, 0.98),eps = 1e-09)  # 优化函数初始化 学习率
        # 学习率更新
        scheduler = get_schedule(optimizer=optimizer, warmup_steps=8000,embedding_dim=int(args['--embedding_dim']))  # 在发现loss不再降低或者acc不再提高之后，降低学习率 触发条件后lr*=factor；

        cuda = 0
        print('start training')
        for epoch in range(20):
            cuda = 0
            bleu_score = 0
            for en_src, de_src,target in make_target(data_en, data_de, data_target,ndata):
                if args['--cuda']:
                    en_src = en_src.to(device)
                    de_src = de_src.to(device)
                    target = target.to(device)
                # forward
                cuda += 1
                out = model(en_src, de_src)
                # log_pre = torch.log(out)
                # labels = F.softmax(target)
                # kl = F.kl_div(log_pre, labels, reduction='batchmean')
                # print('kl:',kl)
                loss = Loss(out.transpose(1,2), target)
                output = torch.argmax(out, dim=-1)
                bleu_score += bleu(target, output, device)
                loss.backward()
                if cuda % 10 == 0:
                    optimizer.step()  # 更新所有参数
                    optimizer.zero_grad()  # 将模型的参数梯度初始化为0
                    scheduler.step()
                    print('This is bleu_score:', bleu_score) #计算bleu值
                    bleu_score = 0
                if cuda % 100 == 0:  # 打印loss
                    print("This is {0} epoch,This is {1} batch".format(epoch, cuda), 'loss = ',
                          '{:.6f}'.format(loss/nword_en),'bleu_score=',bleu_score)
                if cuda % 2000 == 0:  # 保存模型
                    print('save currently model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

def read(filename):
    with open(filename, "rb") as file:  # 读数据
        words = eval(file.readline().strip().decode("utf-8"))
    return words

def decode(args: Dict):
    model_path = args['--model_path']
    words_de = read(args['--de_dict'])
    words_re_de = {i: w for w, i in words_de.items()}
    words_target = read(args['--target_dict'])
    words_re_target = {i: w for w, i in words_target.items()}
    device = torch.device("cuda:" + args['--cuda'] if args['--cuda'] else "cpu")
    model = Transformer.load(model_path)
    print("模型读取成功")
    model.eval()
    de_predict = []

    print('use device: %s' % device, file=sys.stderr)
    model = model.to(device)
    with h5py.File(args['--en'], 'r') as f:
        nword_en = f['nword'][()]
        ndata = f['ndata'][()]
        data_en = f['group']
        torch.manual_seed(0)  # 固定随机种子
        index = 0
        for en_src in make_test(data_en, ndata):
            if args['--cuda']:
                en_src = en_src.to(device)
            ff = 0
            en_outputs = model.encoder(en_src)
            de_input = torch.zeros(en_src.size(0), 0).type_as(en_src.data)  # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
            flag = False
            next_symbol = torch.tensor([[words_de["<sos>"]] for i in range(en_src.size(0))])
            print('index:',index)
            flag_test = [0 for i in range(en_src.size(0))]
            flag_true = [1 for i in range(en_src.size(0))]
            while not flag or ff <= 100:
                # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
                de_input = torch.cat([de_input.to(device), next_symbol.to(device)],-1).to(device)
                de_outputs= model.decoder(de_input, en_src, en_outputs)
                de_outputs = model.classifier(de_outputs)
                result= torch.argmax(de_outputs,dim=-1, keepdim=False)
                next_word = result[:, -1].reshape(-1, 1)  # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
                next_symbol = next_word
                ff += 1
                for i in range(en_src.size(0)):
                    if next_word[i].item() == words_target["<eos>"]:
                        flag_test[i] = 1
                if flag_test == flag_true:
                    flag = True
            de_predict.append(de_input)  # 把结果存入
            index += 1
    with open("result_test_de.txt", 'w') as file:
        l = len(de_predict)
        temp = ""
        for i in range(l):
            for j in range(de_predict[i].size(0)):
                for k in range(de_predict[i].size(1)):
                    temp = temp + str(words_re_target[de_predict[i][j][k].item()]) + ' '
            file.write(temp + '\n')


def get_schedule(optimizer, warmup_steps, embedding_dim, last_epoch=-1):

    def lr_lambda(current_step):
        current_step += 1
        return (embedding_dim ** -0.5) * min(current_step ** -0.5, current_step * (warmup_steps ** -1.5))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def bleu(reference,candidate,device):
    score = 0
    reference = reference.cpu().detach().numpy().tolist()
    candidate = candidate.cpu().detach().numpy().tolist()
    l = len(candidate)
    for i in range(l):
        score += sentence_bleu([reference[i]], candidate[i],weights=[0.25,0.25,0.25,0.25])
    return score

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['--cuda']:
        torch.cuda.manual_seed(1)  # 为特定GPU设置种子，生成随机数
    if args['train']:  # 跑模型：
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')

# python train_new.py --cuda=0 train --en="result_en.hdf5" --de="result_de.hdf5" --target="result_target.hdf5" --model_save_path="model.trans" --embedding_dim=512 --N=6 --heads=8 --dropout=0.1
# python train.py --cuda=0 decode --model_path="model_trans" --en="result_en_test.hdf5" --en_dict="dict_en.txt" --target_dict="dict_target.txt"
# python train.py --cuda=0 train --en="result_en1.hdf5" --de="result_de1.hdf5" --target="result_target1.hdf5" --model_save_path="model1.trans" --embedding_dim=512 --N=6 --heads=8 --dropout=0.1
