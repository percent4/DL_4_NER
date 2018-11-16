# -*- coding: utf-8 -*-
# Name entity recognition for new data

# Import the necessary modules
import pickle
import numpy as np
from utils import CONSTANTS
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk import word_tokenize

# 导入字典
with open(CONSTANTS[1], 'rb') as f:
    word_dictionary = pickle.load(f)
with open(CONSTANTS[4], 'rb') as f:
    output_dictionary = pickle.load(f)

try:
    # 数据预处理
    input_shape = 60
    sent = "Home to the headquarters of the United Nations, New York is an important center for international diplomacy."
    new_sent = word_tokenize(sent)
    new_x = [[word_dictionary[word] for word in new_sent]]
    x = pad_sequences(maxlen=input_shape, sequences=new_x, padding='post', value=0)

    # 载入模型
    model_save_path = CONSTANTS[0]
    lstm_model = load_model(model_save_path)

    # 模型预测
    y_predict = lstm_model.predict(x)

    ner_tag = []
    for i in range(0, len(new_sent)):
        ner_tag.append(np.argmax(y_predict[0][i]))

    ner = [output_dictionary[i] for i in ner_tag]
    print(new_sent)
    print(ner)

    # 去掉NER标注为O的元素
    ner_reg_list = []
    for word, tag in zip(new_sent, ner):
        if tag != 'O':
            ner_reg_list.append((word, tag))

    # 输出模型的NER识别结果
    print("NER识别结果：")
    if ner_reg_list:
        for i, item in enumerate(ner_reg_list):
            if item[1].startswith('B'):
                end = i+1
                while end <= len(ner_reg_list)-1 and ner_reg_list[end][1].startswith('I'):
                    end += 1

                ner_type = item[1].split('-')[1]
                ner_type_dict = {'PER': 'PERSON: ',
                                'LOC': 'LOCATION: ',
                                'ORG': 'ORGANIZATION: ',
                                'MISC': 'MISC: '
                                }
                print(ner_type_dict[ner_type],\
                    ' '.join([item[0] for item in ner_reg_list[i:end]]))
    else:
        print("模型并未识别任何有效命名实体。")

except KeyError as err:
    print("您输入的句子有单词不在词汇表中，请重新输入！")
    print("不在词汇表中的单词为：%s." % err)

