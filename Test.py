# wxy
import sys
import math

<<<<<<< HEAD
def TFIDF():
    words_en = {}
    words_de = {}
    tfidf_en = {}
    tfidf_de = {}
    with open("result_de.txt", "rb") as file:
        for line in file:
            tmp = eval(line.strip().decode("utf-8"))
            words_en.update(tmp)
    with open("result_en.txt", "rb") as file:
        for line in file:
            tmp = eval(line.strip().decode("utf-8"))
            words_de.update(tmp)
    str = input("请输入要测试的句子")
    for item in str.split():
        if item in words_en:  #查看当前词在en文件中存在的句子数
            tfidf_en[item] = words_en[item]
        else:
            tfidf_en[item] = 0
        if item in words_de:
            tfidf_de[item] = words_de[item]
        else:
            tfidf_de[item] = 0
=======

def TFIDF():
    words_en = {}
    words_de = {}
    with open("de.txt", "rb") as file:
        for line in file:
            tmp = eval(line.strip().decode("utf-8"))
            words_de.update(tmp)
    with open("en.txt", "rb") as file:
        for line in file:
            tmp = eval(line.strip().decode("utf-8"))
            words_en.update(tmp)
    sum_en = words_en['@@@']
    sum_de = words_de['@@@']
    sum = sum_de + sum_en
    print(sum_en,sum_de)
    str = input("请输入要测试的句子")
    tfidf_en = {}
    tfidf_de = {}
    for item in str.split():
        count_en = 0
        count_de = 0
        if item in words_en:  #查看当前词在en文件中存在的行数
            count_en = words_en[item] + 1
        else:
            count_en = 1
        if item in words_de:  #查看当前词在de文件中存在的行数
            count_de = words_de[item] + 1
        else:
            count_de = 1
        idf = - math.log((count_en + count_de) / (sum + 1))
        tfidf_en[item] = (count_en -1 )/sum_en * idf #en文件中每个词在en文件中的TFIDF
        tfidf_de[item] = (count_de -1 )/sum_de * idf #en文件中每个词在de文件中的TFIDF
>>>>>>> dc518363c17b16eb076ac8f49999755fae573a42
    return tfidf_en, tfidf_de


if __name__ == "__main__":
    tfidf_en, tfidf_de=TFIDF()
<<<<<<< HEAD
    sum_en = len(tfidf_en.keys())
    sum_de = len(tfidf_de.keys())
    if sum_en >= sum_de:
        print("当前句子大概率属于en文件")
    else:
        print("当前句子大概率属于de文件")
=======
    for key, value in tfidf_de.items():
        print(key,"在de文件中:",value,"在en文件中:",tfidf_en[key])
        if value >= tfidf_en[key]:
            print("当前{0}大概率是de文件中的词".format(key))
        else:
            print("当前{0}大概率是en文件中的词".format(key))
>>>>>>> dc518363c17b16eb076ac8f49999755fae573a42


