# wxy
import random
import h5py
import numpy as np
import torch
import sys

def Sort_de(filename1, filename2):
    with open(filename2, "w", encoding='utf-8') as file1:
        with open(filename1, "r", encoding='utf-8') as file:
            for line in file:
                temp = line.strip()
                n = '<sos> '
                m = ' <eos>'
                file1.write(''.join([n,temp, m]))
                file1.write('\n')


def Sort_en(filename1, filename2):
    with open(filename2, "w", encoding='utf-8') as file1:
        with open(filename1, "r", encoding='utf-8') as file:
            for line in file:
                temp = line.strip()
                n = '<sos> '
                file1.write(''.join([n,temp]))
                file1.write('\n')


def Sort_target(filename1, filename2):
    with open(filename2, "w", encoding='utf-8') as file1:
        with open(filename1, "r", encoding='utf-8') as file:
            for line in file:
                temp = line.strip()
                m = ' <eos>'
                file1.write(''.join([temp,m]))
                file1.write('\n')


# 4、词频统计：收集数据集上的词典，建立单词到索引的一对一映射并保存结果，（0索引对应padding，其它单词的索引从1开始累加，单词顺序从高频到低频）；
def handle_1(filename1, filename2):
    temp_words = {}

    with open(filename1, "r", encoding="utf-8") as file:
        for line in file:
            temp = line.strip()
            if temp:
                for word in temp.split():
                    temp_words[word] = temp_words.get(word, 0) + 1  # 统计词频
    words["<pad>"] = 0
    words["<unk>"] = 1
    words = {word: i for i, word in enumerate(sorted(temp_words, reverse=True), 2)}
    with open(filename2, 'wb') as file:
        file.write(repr(words).encode("utf-8"))
    return words


# 5、根据词典索引，将排序后的数据切分为batch（每个batch包含的token数量尽可能多，但不要超过2560个token）并转换为tensor（为matrix，batch size * seql,
# batch size: 这个batch中包含几个句子，seql: 这些句子中最长句子的长度（短句使用padding，映射索引为0，填充到相同长度）），

def handle_2(words_de, file_de, words_en, file_en, words_target, file_target):
    index = 0
    batch_de = []
    batch_en = []
    batch_target = []
    matrix_line = 1  # 目前矩阵有几行
    batch_flag = True  # 是否开始下一个batch
    for line_de, line_en, line_target in zip(file_de, file_en, file_target):
        tmp_en = line_en.strip()
        tmp_de = line_de.strip()
        tmp_target = line_target.strip()
        if tmp_en and tmp_de and tmp_target:
            if batch_flag:  # 找出当前batch的大小
                max = 256  # 词长
                batch_size = int(2560 / seql)  # 行数
                batch_en = []  # 创建一个batch
                batch_de = []
                batch_target = []
                batch_flag = False
            if matrix_line < batch_size:
                matrix_line += 1
                batch_en.append(
                    [words_en.get(w, 1) for w in tmp_en] + [0 for _ in range(seql - len(tmp_en))])  # 把当前行的数据存入batch
                batch_de.append(
                    [words_de.get(w, 1) for w in tmp_de] + [0 for _ in range(seql - len(tmp_de))])  # 把当前行的数据存入batch
                batch_target.append([words_target.get(w, 1) for w in tmp_target] + [0 for _ in range(
                    seql - len(tmp_target))])  # 把当前行的数据存入batch
            else:  # 当前batch已经存储好 开始寻找下一个batch
                batch_flag = True
                matrix_line = 1
                yield batch_en, batch_de, batch_target, index
                index += 1
    if batch_de and batch_en:
        yield batch_en, batch_de, batch_target, index


# 按hdf5格式存入文件。需要学会yield语法（生成器）的使用。HDF5存储格式：src(一个group)/<k, v>：存转换后的数据，k为数据的索引，从0自增，v为具体数据张量；
# ndata:一个只有一个元素的向量，存src中数据的数量；nword:一个只有一个元素的向量，存第3步收集的词典大小。
def save(words_en, file_en, f5_en, words_de, file_de, f5_de, words_target, file_target, f5_target):
    group_en = f5_en.create_group("group")
    group_de = f5_de.create_group("group")
    group_target = f5_target.create_group("group")
    index = 0
    for batch_en, batch_de, batch_target, index in handle_2(words_en, file_en, words_de, file_de, words_target,
                                                            file_target):
        matrix_array_en = np.array(batch_en, dtype=np.int32)  # 将batch转成numpy类型存储
        group_en.create_dataset(str(index), data=matrix_array_en)
        matrix_array_de = np.array(batch_de, dtype=np.int32)  # 将batch转成numpy类型存储
        group_de.create_dataset(str(index), data=matrix_array_de)
        matrix_array_target = np.array(batch_target, dtype=np.int32)  # 将batch转成numpy类型存储
        group_target.create_dataset(str(index), data=matrix_array_target)
    f5_en["ndata"] = np.array(index, dtype=np.int32)
    f5_en["nword"] = np.array(len(words_en), dtype=np.int32)
    f5_de["ndata"] = np.array(index, dtype=np.int32)
    f5_de["nword"] = np.array(len(words_de), dtype=np.int32)
    f5_target["ndata"] = np.array(index, dtype=np.int32)
    f5_target["nword"] = np.array(len(words_target), dtype=np.int32)


if __name__ == "__main__":
    Sort_de(sys.argv[1], sys.argv[2])  # BPE BPE_sort de
    Sort_en(sys.argv[3], sys.argv[4])  # BPE BPE_sort en
    Sort_target(sys.argv[3], sys.argv[5])  # BPE BPE_sort target
    words_de = handle_1(sys.argv[2], sys.argv[6])  # BPE_sort dict  de
    words_en = handle_1(sys.argv[4], sys.argv[7])  # BPE_sort dict  en
    words_target = handle_1(sys.argv[5], sys.argv[8])  # BPE_sort dict  target
    with open(sys.argv[2], 'r', encoding="utf-8") as file_de, open(sys.argv[4], 'r', encoding="utf-8") as file_en, open(
            sys.argv[5], 'r', encoding="utf-8") as file_target, h5py.File(sys.argv[9], 'w') as f5_de, h5py.File(
            sys.argv[10], 'w') as f5_en, h5py.File(sys.argv[11], 'w') as f5_target:  # result
        save(words_de, file_de, f5_de, words_en, file_en, f5_en, words_target, file_target, f5_target)

# python Pretreatment.py commoncrawl_BPE_de.txt commoncrawl_BPE_sort_de.txt commoncrawl_BPE_en.txt commoncrawl_BPE_sort_en.txt commoncrawl_BPE_sort_target.txt commoncrawl_dict_de.txt commoncrawl_dict_en.txt commoncrawl_dict_target.txt commoncrawl_result_de.hdf5 commoncrawl_result_en.hdf5 commoncrawl_result_target.hdf5
