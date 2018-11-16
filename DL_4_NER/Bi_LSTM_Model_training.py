# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from utils import BASE_DIR, CONSTANTS, load_data
from data_processing import data_processing
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, LSTM, Dense, Embedding, TimeDistributed


# 模型输入数据
def input_data_for_model(input_shape):

    # 数据导入
    input_data = load_data()
    # 数据处理
    data_processing()
    # 导入字典
    with open(CONSTANTS[1], 'rb') as f:
        word_dictionary = pickle.load(f)
    with open(CONSTANTS[2], 'rb') as f:
        inverse_word_dictionary = pickle.load(f)
    with open(CONSTANTS[3], 'rb') as f:
        label_dictionary = pickle.load(f)
    with open(CONSTANTS[4], 'rb') as f:
        output_dictionary = pickle.load(f)
    vocab_size = len(word_dictionary.keys())
    label_size = len(label_dictionary.keys())

    # 处理输入数据
    aggregate_function = lambda input: [(word, pos, label) for word, pos, label in
                                            zip(input['word'].values.tolist(),
                                                input['pos'].values.tolist(),
                                                input['tag'].values.tolist())]

    grouped_input_data = input_data.groupby('sent_no').apply(aggregate_function)
    sentences = [sentence for sentence in grouped_input_data]

    x = [[word_dictionary[word[0]] for word in sent] for sent in sentences]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[word[2]] for word in sent] for sent in sentences]
    y = pad_sequences(maxlen=input_shape, sequences=y, padding='post', value=0)
    y = [np_utils.to_categorical(label, num_classes=label_size + 1) for label in y]

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


# 定义深度学习模型：Bi-LSTM
def create_Bi_LSTM(vocab_size, label_size, input_shape, output_dim, n_units, out_act, activation):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim,
                        input_length=input_shape, mask_zero=True))
    model.add(Bidirectional(LSTM(units=n_units, activation=activation,
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(label_size + 1, activation=out_act)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 模型训练
def model_train():

    # 将数据集分为训练集和测试集，占比为9:1
    input_shape = 60
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = input_data_for_model(input_shape)
    train_end = int(len(x)*0.9)
    train_x, train_y = x[0:train_end], np.array(y[0:train_end])
    test_x, test_y = x[train_end:], np.array(y[train_end:])

    # 模型输入参数
    activation = 'selu'
    out_act = 'softmax'
    n_units = 100
    batch_size = 32
    epochs = 10
    output_dim = 20

    # 模型训练
    lstm_model = create_Bi_LSTM(vocab_size, label_size, input_shape, output_dim, n_units, out_act, activation)
    lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    # 模型保存
    model_save_path = CONSTANTS[0]
    lstm_model.save(model_save_path)
    plot_model(lstm_model, to_file='%s/LSTM_model.png' % BASE_DIR)

    # 在测试集上的效果
    N = test_x.shape[0]  # 测试的条数
    avg_accuracy = 0  # 预测的平均准确率
    for start, end in zip(range(0, N, 1), range(1, N+1, 1)):
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end])
        input_sequences, output_sequences = [], []
        for i in range(0, len(y_predict[0])):
            output_sequences.append(np.argmax(y_predict[0][i]))
            input_sequences.append(np.argmax(test_y[start][i]))

        eval = lstm_model.evaluate(test_x[start:end], test_y[start:end])
        print('Test Accuracy: loss = %0.6f accuracy = %0.2f%%' % (eval[0], eval[1] * 100))
        avg_accuracy += eval[1]
        output_sequences = ' '.join([output_dictionary[key] for key in output_sequences if key != 0]).split()
        input_sequences = ' '.join([output_dictionary[key] for key in input_sequences if key != 0]).split()
        output_input_comparison = pd.DataFrame([sentence, output_sequences, input_sequences]).T
        print(output_input_comparison.dropna())
        print('#' * 80)

    avg_accuracy /= N
    print("测试样本的平均预测准确率：%.2f%%." % (avg_accuracy * 100))

model_train()
