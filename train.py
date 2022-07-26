# wxy
# -*- coding:utf-8 -*-
'''Usage:
    train.py --cuda=<int> train --src="" --target="" --model_save_path="" --embedding_dim=<int> --N=<int> --heads=<int> --dropout=<float>  [options]
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

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, d_k, mask,dropout=None):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        s = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充s中与attn_mask中值为1位置相对应的元素）
        # mask掉那些为了padding长度增加的token，让其通过softmax计算后为0
        if mask is not None:
            mask = mask.unsqueeze(1)
            s = s.masked_fill(mask == 0, -1e9)

        p = F.softmax(s, dim=-1) # 对最后一个维度(v)做softmax
        # s : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        output = torch.matmul(p, V)  # output: [batch_size, n_heads, len_q, d_v]
        # output：[[z1,z2,...],[...]]向量, p注意力稀疏矩阵（用于可视化的）
        if dropout is not None:
            s = dropout(s)

        return output,mask

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, d_ff=2048, dropout=0.1):
        super().__init__()

        self.embed = embedding_dim

        # We set d_ff as a default to 2048
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embedding_dim),
        )

    def forward(self, input):
        residual = input
        print('r:',residual.size())
        output = self.net(input)
        print('o:',output.size())
        # return nn.LayerNorm(d_model).to(device)(output + residual)
        n = nn.LayerNorm(self.embed)
        output = n(output + residual)
        print('o:',output.size())
        return  output# [bsize, seql, embedding_dim]



class Mask():
    def __init__(self):
        super(Mask, self).__init__()

    def en_mask(self,seq_q, seq_k):
        # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
        """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
        encoder和decoder都可能调用这个函数，所以seq_len视情况而定
        seq_q: [bsize, sqel]
        seq_k: [bsize, sqel]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        """
        bsize, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
        bsize, len_k = seq_k.size()
        # eq(zero) is PAD token
        # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
        mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return mask.expand(bsize, len_q, len_k)  # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)

    def de_mask(self,seq):
        """建议打印出来看看是什么的输出（一目了然）
        seq: [batch_size, tgt_len]
        """
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        # attn_shape: [batch_size, tgt_len, tgt_len]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        return subsequence_mask  # [batch_size, tgt_len, tgt_len]

class MultiHeadAttention(nn.Module):

    def __init__(self, heads, embedding_dim, dropout=0.1):
        super().__init__()

        self.embed = embedding_dim   # 512
        self.d_k = embedding_dim // heads  # 512 / 8
        self.h = heads  # 8

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v, mask=None):
        residual, bsize = q, q.size(0)
        # q:[bsize,nquery,embedding_dim] k:[bsize,sqel,embedding_dim] q:[bsize,sqel,embedding_dim]
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bsize, -1, self.h, self.d_k).transpose(1, 2) # [bsize,nquery,heads,embedding_dim/8] → [bsize,heads,nquery,embedding_dim/8]
        q = self.q_linear(q).view(bsize, -1, self.h, self.d_k).transpose(1, 2) # [bsize,sqel,heads,embedding_dim/8] → [bsize,heads,sqel,embedding_dim/8]
        v = self.v_linear(v).view(bsize, -1, self.h, self.d_k).transpose(1, 2) # [bsize,sqel,heads,embedding_dim/8] → [bsize,heads,sqel,embedding_dim/8]
        # 因为是多头，所以mask矩阵要扩充成4维的
        # mask: [bsize, seql, seql] -> [bsize,n_heads, seql, seql]
        mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)

        # calculate attention using function we will define next
        ss ,mask= self.attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = ss.transpose(1, 2).contiguous().view(bsize , -1, self.embed)
        output = self.out(concat)
        n = nn.LayerNorm(self.embed)
        output = n(output+residual)

        return output,mask

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim) # 位置编码矩阵，维度[max_len, embedding_dim]
        position = torch.arange(0.0, max_len).unsqueeze_(1) # 单词位置

        div_term = torch.exp(torch.arange(0.0, embedding_dim, 2) * (- math.log(1e4) / embedding_dim))  # 使用exp和log实现幂运算
        div_term.unsqueeze_(0)
        pe[:, 0 : : 2] = torch.sin(torch.mm(position, div_term)) # 计算单词位置沿词向量维度的纹理值
        pe[:, 1 : : 2] = torch.cos(torch.mm(position, div_term))
        pe.unsqueeze_(0) # 增加批次维度，[1, max_len, embedding_dim]
        self.register_buffer('pe', pe) # 将位置编码矩阵注册为buffer(不参加训练)

    def forward(self, x): # 将一个批次中语句所有词向量与位置编码相加
        """
            x: [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, x.size(1), :]  # 注意，位置编码不参与训练，因此设置requires_grad=False
        return self.dropout(x)

class EncoderLayer(nn.Module):

    def __init__(self, embedding_dim, heads, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(heads, embedding_dim, dropout=dropout)
        self.ff = FeedForward(embedding_dim, dropout=dropout)


    def forward(self, input, mask):
        """E
            inputs: [bsize, sqel, embedding_dim]
            mask: [bsize, sqel, seql]  mask矩阵(pad mask or sequence mask)
        """
        # outputs: [bsize, sqel, embedding_dim], attn: [bsize, n_heads, sqel, sqel]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        output, mask = self.attn(input, input, input,mask)  # enc_inputs to same Q,K,V（未线性变换前）
        output = self.pos_ffn(output) # output: [bsize, sqel, embedding_dim]
        return output, mask

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim,heads) for _ in range(N)])
        self.mask = Mask()

    def forward(self, input):
        """
            enc_inputs: [bsize, sqel]
        """
        x = self.embed(input) # [bsize, sqel, embedding_dim]
        x = self.pe(x)
        attn = []
        mask = self.mask.en_mask(input,input) # [bsize, sqel, sqel]
        for layer in self.layers:
            x, att = layer(x, mask) #[bsize, sqel, embedding_dim], mask: [bsize, n_heads, sqel, sqel]
            attn.append(att)
        return x,attn


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(heads, embedding_dim, dropout=dropout)
        self.dec_enc_attn = MultiHeadAttention(heads, embedding_dim, dropout=dropout)
        self.pos_ffn = FeedForward(embedding_dim, dropout=dropout)

    def forward(self, tgt_inputs, src_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        tgt_inputs: [batch_size, tgt_len, d_model]
        src_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        tgt_inputs, dec_self_attn = self.dec_self_attn(tgt_inputs, tgt_inputs, tgt_inputs,
                                                       dec_self_attn_mask)  # 这里的Q,K,V全是Decoder自己的输入
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        tgt_inputs, dec_enc_attn = self.dec_enc_attn(tgt_inputs, src_outputs, src_outputs,
                                                     dec_enc_attn_mask)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        tgt_inputs = self.pos_ffn(tgt_inputs)  # [batch_size, tgt_len, d_model]
        return tgt_inputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embedding_dim, N, heads, dropout):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, embedding_dim)  # Decoder输入的embed词表
        self.pos_emb = PositionalEncoding(embedding_dim)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, heads) for _ in range(N)])  # Decoder的blocks
        self.mask = Mask()

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = self.mask.en_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = self.mask.de_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        dec_enc_attn_mask = self.mask.en_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):

    def __init__(self, src_vocab, target_vocab, embedding_dim, N, heads, dropout):
        super().__init__()
        self.s_vocab = src_vocab
        self.t_vocab = target_vocab
        self.embed = embedding_dim
        self.N = N
        self.heads = heads
        self.dropout = dropout
        self.encoder = Encoder(src_vocab, embedding_dim, N, heads, dropout)
        self.decoder = Decoder(target_vocab, embedding_dim, N, heads, dropout)
        self.classifier = nn.Linear(embedding_dim, target_vocab)

    def forward(self, src, target):
        """Transformers的输入：两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        en_outputs, enc_self_attns = self.encoder(src)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        de_outputs, dec_self_attns, dec_enc_attns = self.decoder(target, src, en_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.classifier(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']

        model = Transformer(**args)
        model.load_state_dict(params['state_dict'])  # 加载之前存储的数据

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {  # 相应参数的设定
            'args': dict(src_vocab=self.s_vocab, target_vocab=self.t_vocab, embedding_dim=self.embed, N=self.N, heads=self.heads, dropout=self.dropout),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


# 返回一个batch和对应的target
def make_target(data_en,data_de, ndata):
    l = list(range(ndata))
    random.shuffle(l)
    for i in l:
        src = data_en[str(i)][:]
        target = data_de[str(i)][:]
        src = torch.LongTensor(src)
        target = torch.LongTensor(target)
        yield src, target  # 输入 输出  正确结果


def train(args: Dict):
    model_save_path = args['--model_save_path']
    with h5py.File(args['--src'], 'r') as f1,h5py.File(args['--target'], 'r') as f2:
        nword_en = f1['nword'][()]
        nword_de = f2['nword'][()]
        ndata = f1['ndata'][()]
        data_en = f1['group']
        data_de = f2['group']
        torch.manual_seed(0)  # 固定随机种子

        model = Transformer(src_vocab=nword_en, target_vocab=nword_de, embedding_dim=int(args['--embedding_dim']), N=int(args['--N']), heads=int(args['--heads']), dropout=float(args['--dropout']))  # 模型初始化

        model.train()

        for param in model.parameters():  # model.parameters()保存的是Weights和Bais参数的值
            torch.nn.init.uniform_(param, a=-0.001, b=0.001)  # 给weight初始化

        device = torch.device("cuda:" + args['--cuda'] if args['--cuda'] else "cpu")  # 分配设备

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
            for src, target in make_target(data_en,data_de, ndata):
                if args['--cuda']:
                    src = src.to(device)
                    target = target.to(device)
                # forward
                cuda += 1
                out = model(src, target)
                loss = Loss(out.transpose(1, 2), target)
                # backward
                loss.backward()
                if cuda % 10 == 0:
                    optimizer.step()  # 更新所有参数
                    optimizer.zero_grad()  # 将模型的参数梯度初始化为0
                if cuda % 100 == 0:  # 打印loss
                    print("This is {0} epoch,This is {1} batch".format(epoch, cuda), 'loss = ',
                          '{:.6f}'.format(loss / nword))
                if cuda % 1000 == 0:  # 更新学习率
                    scheduler.step(loss)
                if cuda % 100 == 0:  # 保存模型
                    print('save currently model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)


if __name__ == "__main__":
    args = docopt(__doc__)
    if args['--cuda']:
        torch.cuda.manual_seed(1)  # 为特定GPU设置种子，生成随机数
    if args['train']:  # 跑模型：
        train(args)
    else:
        raise RuntimeError('invalid run mode')