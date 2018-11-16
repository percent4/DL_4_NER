# -*- coding: utf-8 -*-

import pickle
import numpy as np
from collections import Counter
from itertools import accumulate
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import BASE_DIR, CONSTANTS, load_data

# 设置matplotlib绘图时的字体
mpl.rcParams['font.sans-serif']=['SimHei']

# 数据查看
def data_review():

    # 数据导入
    input_data = load_data()

    # 基本的数据review
    sent_num = input_data['sent_no'].astype(np.int).max()
    print("一共有%s个句子。\n"%sent_num)

    vocabulary = input_data['word'].unique()
    print("一共有%d个单词。"%len(vocabulary))
    print("前10个单词为：%s.\n"%vocabulary[:11])

    pos_arr = input_data['pos'].unique()
    print("单词的词性列表：%s.\n"%pos_arr)

    ner_tag_arr = input_data['tag'].unique()
    print("NER的标注列表：%s.\n" % ner_tag_arr)

    df = input_data[['word', 'sent_no']].groupby('sent_no').count()
    sent_len_list = df['word'].tolist()
    print("句子长度及出现频数字典：\n%s." % dict(Counter(sent_len_list)))

    # 绘制句子长度及出现频数统计图
    sort_sent_len_dist = sorted(dict(Counter(sent_len_list)).items(), key=itemgetter(0))
    sent_no_data = [item[0] for item in sort_sent_len_dist]
    sent_count_data = [item[1] for item in sort_sent_len_dist]
    plt.bar(sent_no_data, sent_count_data)
    plt.title("句子长度及出现频数统计图")
    plt.xlabel("句子长度")
    plt.ylabel("句子长度出现的频数")
    plt.savefig("%s/句子长度及出现频数统计图.png" % BASE_DIR)
    plt.close()

    # 绘制句子长度累积分布函数(CDF)
    sent_pentage_list = [(count/sent_num) for count in accumulate(sent_count_data)]

    # 寻找分位点为quantile的句子长度
    quantile = 0.9992
    #print(list(sent_pentage_list))
    for length, per in zip(sent_no_data, sent_pentage_list):
        if round(per, 4) == quantile:
            index = length
            break
    print("\n分位点为%s的句子长度:%d." % (quantile, index))

    # 绘制CDF
    plt.plot(sent_no_data, sent_pentage_list)
    plt.hlines(quantile, 0, index, colors="c", linestyles="dashed")
    plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
    plt.text(0, quantile, str(quantile))
    plt.text(index, 0, str(index))
    plt.title("句子长度累积分布函数图")
    plt.xlabel("句子长度")
    plt.ylabel("句子长度累积频率")
    plt.savefig("%s/句子长度累积分布函数图.png" % BASE_DIR)
    plt.close()

# 数据处理
def data_processing():
    # 数据导入
    input_data = load_data()

    # 标签及词汇表
    labels, vocabulary = list(input_data['tag'].unique()), list(input_data['word'].unique())

    # 字典列表
    word_dictionary = {word: i+1 for i, word in enumerate(vocabulary)}
    inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i+1 for i, label in enumerate(labels)}
    output_dictionary = {i+1: labels for i, labels in enumerate(labels)}

    dict_list = [word_dictionary, inverse_word_dictionary,label_dictionary, output_dictionary]

    # 保存为pickle形式
    for dict_item, path in zip(dict_list, CONSTANTS[1:]):
        with open(path, 'wb') as f:
            pickle.dump(dict_item, f)

#data_review()