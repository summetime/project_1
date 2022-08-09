# wxy
import random
import h5py
import numpy as np
import torch
import sys


def Sort_en(filename1, filename2):
    with open(filename2, "w", encoding='utf-8') as file1:
        with open(filename1, "r", encoding='utf-8') as file:
            for line in file:
                temp = line.strip()
                n = '<sos> '
                m = ' <eos>'
                file1.write(''.join([n, temp, m]))
                file1.write('\n')
def Sort_target(filename1, filename2):
    with open(filename2, "w", encoding='utf-8') as file1:
        with open(filename1, "r", encoding='utf-8') as file:
            for line in file:
                temp = line.strip()
                m = ' <eos>'
                file1.write(''.join([temp, m]))
                file1.write('\n')
# 4、词频统计：收集数据集上的词典，建立单词到索引的一对一映射并保存结果，（0索引对应padding，其它单词的索引从1开始累加，单词顺序从高频到低频）；

def load(fileName):
    with open(fileName, "rb") as frd:
        tmp = frd.read().strip()  # 读文件并且去除
        rs = eval(tmp.decode("utf-8"))  # eval() 函数计算或执行参数 读文件使用decode
    return rs


# 5、根据词典索引，将排序后的数据切分为batch（每个batch包含的token数量尽可能多，但不要超过2560个token）并转换为tensor（为matrix，batch size * seql,
# batch size: 这个batch中包含几个句子，seql: 这些句子中最长句子的长度（短句使用padding，映射索引为0，填充到相同长度）），
# 查找每个batch的大小
def handle_2(file):
    batch_size = []
    tt = []
    lines = 0  # 目前矩阵有几行
    seql = 3
    bsize = int(2560 / 3)
    for line in file:
        temp = line.strip()
        if temp :
            temp = temp.split()
            l = len(temp) # 词长
            if lines < bsize:  # 找出当前batch的大小
                lines += 1  # 读取行数加一
                if l > seql:
                    if lines * l > 2560:
                        batch_size.append((lines-1, seql))  # 再加会超 则直接存为一个batch
                        lines = 1  # 将本行加入下一个batch
                    seql = l
                    bsize = int(2560 / seql)
            else:
                batch_size.append((bsize, seql))  # 当前batch读取完毕 存储
                lines = 1  # 重新读取下一个batch 并把当前行加入
                seql = l
                bsize = int(2560 / seql)
    if lines > 0:
        batch_size.append((lines, seql))
        tt.append([len(temp), l, lines, bsize, seql])
    with open('batch.txt', "w", encoding="utf-8") as file:
        for line in batch_size:
            file.write(str(line) + ' ')
    return batch_size


# 根据上次存储每个batch大小 存储下一个文件
def handle_3(words, file, batch_size):
    index = 0
    batch = []
    matrix_line = 0
    bsize = batch_size[0][0]  # 行数
    seql = batch_size[0][1]  # 词长
    for line in file:
        tmp = line.strip()
        if tmp:
            tmp = tmp.split()
            if matrix_line < bsize:
                batch.append([words.get(w, 1) for w in tmp] + [0 for _ in range(seql - len(tmp))])  # 把当前行的数据存入batch
                matrix_line += 1
            else:  # 当前batch已经存储好 开始寻找下一个batch
                yield batch,index
                index += 1
                bsize = batch_size[index][0]  # 行数
                seql = batch_size[index][1]  # 词长
                batch= []  # 创建一个batch
                batch.append([words.get(w, 1) for w in tmp] + [0 for _ in range(seql - len(tmp))])  # 把当前行的数据存入batch
                matrix_line = 1
    if batch:
        yield batch, index


# 按hdf5格式存入文件。需要学会yield语法（生成器）的使用。HDF5存储格式：src(一个group)/<k, v>：存转换后的数据，k为数据的索引，从0自增，v为具体数据张量；
# ndata:一个只有一个元素的向量，存src中数据的数量；nword:一个只有一个元素的向量，存第3步收集的词典大小。
def save(words_en, file_en, f5_en, batch_size):
    group_en = f5_en.create_group("group")
    index = 0
    for batch_en, index in handle_3(words_en, file_en, batch_size):
        matrix_array_en = np.array(batch_en, dtype=np.int32)  # 将batch转成numpy类型存储
        group_en.create_dataset(str(index), data=matrix_array_en)

    f5_en["ndata"] = np.array(index, dtype=np.int32)
    f5_en["nword"] = np.array(len(words_en), dtype=np.int32)


if __name__ == "__main__":
    Sort_en(sys.argv[1], sys.argv[2])  # BPE BPE_sort en
    Sort_target(sys.argv[5], sys.argv[6])
    print(1)
    words_en = load(sys.argv[3])  # BPE_sort dict  en
    print(2)
    with open(sys.argv[2], 'r', encoding="utf-8") as file_en:  # result
        batch_size = handle_2(file_en)
    print(3)
    with open(sys.argv[2], 'r', encoding="utf-8") as file_en,h5py.File(sys.argv[4], 'w') as f5_en:  # result
        save(words_en, file_en, f5_en, batch_size)
    print(4)

# python ..\GIT\pre_test_data.py test.tc.en test_sort_en.txt dict_en.txt result_en_test.hdf5
# python pre_test_data.py test.BPE.en test_sort_en.txt dict_en.txt result_en_test.hdf5 test.BPE.de test_sort_de.txt