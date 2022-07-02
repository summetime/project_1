# wxy
import random
import h5py
import numpy as np
import torch


# 1、BPE之前先把长度超过128个token的句子丢掉
def getText():
    with open("train.txt", "w", encoding='utf-8') as writ:
        with open("corpus.tc.en", "r", encoding='utf-8') as file:
            for line in file:
                tmp = line.strip()
                if len(tmp.split()) <= 128:
                    writ.write(tmp)
                    writ.write('\n')


# 2、在miniconda下使用命令行做BPE ：
# （1）在训练集上学BPE——得到分词结果 统计每一个连续字节对，并保存为code_file
# subword-nmt learn-bpe -s 5000 --input D:\pythonProject\week_6\train.txt --output D:\pythonProject\week_6\codes_file.txt
# （2）对训练集初步BPE的结果统计vocabulary 统计每一个连续字节对出现的频率——词频统计
# subword-nmt apply-bpe -c D:\pythonProject\week_6\codes_file.txt --input D:\pythonProject\week_6\corpus.tc.en | subword-nmt get-vocab --output D:\pythonProject\week_6\vocab_file.txt
# （3）同时使用学到的BPE和统计的词频对训练集应用BPE
# subword-nmt apply-bpe -c D:\pythonProject\week_6\codes_file.txt --vocabulary  D:\pythonProject\week_6\vocab_file.txt --vocabulary-threshold 10 --input D:\pythonProject\week_6\train.txt --output D:\pythonProject\week_6\train_file_BPE.txt
# （4）用训练集上学到的BPE和统计的词频对验证集应用BPE
# subword-nmt apply-bpe -c D:\pythonProject\week_6\codes_file.txt --vocabulary  D:\pythonProject\week_6\vocab_file.txt --vocabulary-threshold 10 --input D:\pythonProject\week_6\test.txt --output D:\pythonProject\week_6\train_file_BPE.txt

# BPE处理完之后对数据集进行排序，使相似长度的句子聚合在一起（长度低于3或长于256的丢弃，因为下周的具体任务和内存限制），同一长度的句子通过random包的shuffle方法做一次顺序打乱
def Sort():
    lines = {}  # 按照句子长度存储
    with open("train_file_BPE.txt", "r", encoding='utf-8') as file:
        for line in file:
            temp = line.strip()  # 统计词长
            l = len(temp.split())
            if 3 <= l <= 256:
                if l in lines: #句长相同添加到lines字典中
                    lines[l].append(temp)
                else:
                    lines[l] = [temp]
    # 把句子数相同的打乱顺序
    for line in lines.values():
        if len(line) > 1:
            random.shuffle(line) #打乱顺序

    # 存储排序好的字典
    with open("train_file_BPE_sort.txt", "w", encoding="utf-8") as file:
        for i in sorted(lines, reverse=True):  # 从高到低存储
            file.write('\n'.join(lines[i]))
            file.write('\n')
            # for j in lines[i]:
            #     file.write(str(j))



# 4、词频统计：收集数据集上的词典，建立单词到索引的一对一映射并保存结果，（0索引对应padding，其它单词的索引从1开始累加，单词顺序从高频到低频）；
def handle_1():
    temp_words = {}

    with open("train_file_BPE_sort.txt", "r", encoding="utf-8") as file:
        for line in file:
            temp = line.strip()
            if temp:
                for word in temp.split():
                    temp_words[word] = temp_words.get(word, 0) + 1  # 统计词频
    # words["<pad>"] = 0
    # i = 1
    words = {word:i for i,word in enumerate(sorted(temp_words, reverse=True),0)}
    # with open("dict.txt",'wb') as file:
    #     file.write(repr(words).encode("utf-8"))
    # for i,word in enumerate(sorted(temp_words, reverse=True),1):  # 按照高频到低频存储
    #     words[word] = i
        # i += 1
    return words


# 5、根据词典索引，将排序后的数据切分为batch（每个batch包含的token数量尽可能多，但不要超过2560个token）并转换为tensor（为matrix，batch size * seql,
# batch size: 这个batch中包含几个句子，seql: 这些句子中最长句子的长度（短句使用padding，映射索引为0，填充到相同长度）），

def handle_2(datas, file):
    index = 0
    batch = []
    matrix_line = 1  # 目前矩阵有几行
    batch_flag = True  # 是否开始下一个batch
    for line in file:
        tmp = line.strip()
        if tmp:
            tmp = tmp.split()
            if batch_flag:  # 找出当前batch的大小
                seql = len(tmp) # 词长
                batch_size = int(2560 / len(tmp)) # 行数
                batch = []  # 创建一个batch
                batch_flag = False
            if matrix_line < batch_size:
                matrix_line += 1
                batch.append([datas.get(w, 1) for w in tmp] + [0 for _ in range(seql - len(tmp))])  # 把当前行的数据存入batch
            else:  # 当前batch已经存储好 开始寻找下一个batch
                batch_flag = True
                matrix_line = 1
                yield batch,index
                index += 1
    if batch:
        yield batch,index


# 按hdf5格式存入文件。需要学会yield语法（生成器）的使用。HDF5存储格式：src(一个group)/<k, v>：存转换后的数据，k为数据的索引，从0自增，v为具体数据张量；
# ndata:一个只有一个元素的向量，存src中数据的数量；nword:一个只有一个元素的向量，存第3步收集的词典大小。
def save(datas, file, f5):
    group = f5.create_group("group")
    index = 0
    for batch,index in handle_2(datas, file):
        matrix_array = np.array(batch, dtype=np.int32) #将batch转成numpy类型存储
        group.create_dataset(str(index), data=matrix_array)
    f5["ndata"] = np.array(index, dtype=np.int32)
    f5["nword"] = np.array(len(datas), dtype=np.int32)


if __name__ == "__main__":
    # getText()
    Sort()
    words = handle_1()
    with open("train_file_BPE_sort.txt", 'r', encoding="utf-8") as file, h5py.File("result.hdf5", 'w') as f5:
        save(words, file, f5)
