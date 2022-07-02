# wxy
import math
import re
from tqdm import tqdm

<<<<<<< HEAD

# 读取文件 返回字典
=======
#读取文件 返回字典
>>>>>>> dc518363c17b16eb076ac8f49999755fae573a42
def readFile(fileName):
    words = {}
    count = 0
    sum = 0
    # 读取文件有多少行
    with open(fileName, "rb") as file:
        for line in file:
            tmp = line.strip()
            if tmp:
                count = count + 1
    pbar = tqdm(total=count)  # 根据行数显示程序进度
    pbar.set_description("当前进度")

    with open(fileName, "rb") as file:
        for line in file:
            tmp = line.strip()
            if tmp:
                tmp = tmp.decode("utf-8")
<<<<<<< HEAD
                for word in set(tmp.split()):
                    if word:
                        words[word] = words.get(word, 0) + 1  # 更新当前词出现的频率
            pbar.update(1)  # 进度条更新显示
    pbar.close()
    return words, count


def getAll(filename1, filename2):
    v = set()
    with open(filename1, "rb") as file:
        for line in file:
            tmp = line.strip()
            if tmp:
                tmp = tmp.decode("utf-8")
                for word in tmp.split():
                    if word not in v:
                        v.add(word)
    with open(filename2, "rb") as file:
        for line in file:
            tmp = line.strip()
            if tmp:
                tmp = tmp.decode("utf-8")
                for word in tmp.split():
                    if word not in v:
                        v.add(word)
    return v


# 计算TFIDF: TF * IDF
# TF= 每一个词出现的概率
# IDF= -log P 区别度
# TF=当前词出现的句子数/总句子数
# IDF=-log（（当前词在en文档出现的句子数+当前词在de文档出现的句子数）/en文档和de文档的总句子数）
def TFIDF(filename1, filename2):
    tfidf_de = {}
    tfidf_en = {}
    unk_list = []
    v = getAll(filename1, filename2)
    words_en, sum_en = readFile(filename1)
    words_de, sum_de = readFile(filename2)
    sum = sum_de + sum_en
    for key in words_en:
        if key not in words_de:
            t = ("en", key)
            unk_list.append(t)
    for key in words_de:
        if key not in words_en:
            t = ("de", key)
            unk_list.append(t)
    for key in words_en:
        words_en[key] += 1
    for key in words_de:
        words_de[key] += 1
    for key in v:
        for key, values in unk_list:
            if values in values:
                idf = - math.log((words_en[key] + words_de[key]) / (sum + len(words_en))+len(words_de))
                tfidf_en[key] = (words_en[key]) / (sum_en + len(words_en)) * idf  # 每个词在en文件中的TFIDF TF计算不用做平滑处理
                tfidf_de[key] = (words_de[key]) / (sum_de + len(words_de)) * idf  # 每个词在de文件中的TFIDF TF计算不用做平滑处理
            elif key == "en":
                idf = - math.log((words_en[key] + 1) / (sum + len(words_en))+len(words_de))
                tfidf_en[key] = (words_en[key]) / (sum_en + len(words_en) ) * idf  # 每个词在en文件中的TFIDF TF计算不用做平滑处理
                tfidf_de[key] = 1 / (sum_de + len(words_de) ) * idf  # 每个词在de文件中的TFIDF TF计算不用做平滑处理
            else:
                idf = - math.log((1 + words_de[key]) / (sum + len(words_en))+len(words_de))
                tfidf_en[key] = 1 / (sum_en + len(words_en)) * idf  # 每个词在en文件中的TFIDF TF计算不用做平滑处理
                tfidf_de[key] = (words_de[key]) / (sum_de + len(words_de)) * idf  # 每个词在de文件中的TFIDF TF计算不用做平滑处理

=======
                list = re.split(r'[.!?:]', tmp)
                sum = sum + len(list)
                for item in list:
                    for word in set(item.split()):
                        if word:
                            words[word] = words.get(word, 0) + 1  # 更新当前词出现的频率
            pbar.update(1)  # 进度条更新显示
    pbar.close()
    return words,sum

#计算TFIDF: TF * IDF
#TF= 每一个类别出现的概率
#IDF= -log P  P：当前词出现的概率 假如当前词在dn中出现的次数为100 在en 出现的概率为300 则P=（100+300）/20k
def TFIDF(filename1, filename2):
    tfidf_de = {}
    tfidf_en = {}
    words_en ,sum_en= readFile(filename1)
    words_de ,sum_de= readFile(filename2)
    sum = sum_de + sum_en
    for key , value in words_en.items():
        if key in words_de:
            count_de = words_de[key] + 1
        else:
            count_de = 1
        idf = - math.log((value + count_de) / (sum + 1))
        tfidf_en[key] = (value / sum_en) * idf  #en文件中每个词在en文件中的TFIDF
        tfidf_de[key] = (count_de - 1) / sum_de * idf  #en文件中每个词在de文件中的TFIDF
>>>>>>> dc518363c17b16eb076ac8f49999755fae573a42
    return tfidf_en, tfidf_de


def save(filename, obj):
    with open(filename, "wb") as file:
        file.write(repr(obj).encode("utf-8"))

<<<<<<< HEAD

def load(fileName):
    with open(fileName, "rb") as frd:
        tmp = frd.read().strip()  # 读文件并且去除
        rs = eval(tmp.decode("utf-8"))  # eval() 函数计算或执行参数 读文件使用decode
    return rs
=======
def load(fileName):
	with open(fileName, "rb") as frd:
		tmp = frd.read().strip() #读文件并且去除
		rs = eval(tmp.decode("utf-8"))  #eval() 函数计算或执行参数 读文件使用decode
	return rs
>>>>>>> dc518363c17b16eb076ac8f49999755fae573a42


if __name__ == "__main__":
    tfidf_en, tfidf_de = TFIDF("corpus.tc.en", "corpus.tc.de")
    save("result_en.txt", tfidf_en)
    save("result_de.txt", tfidf_de)
<<<<<<< HEAD
=======


>>>>>>> dc518363c17b16eb076ac8f49999755fae573a42
