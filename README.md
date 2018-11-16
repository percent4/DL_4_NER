# DL_4_NER
Using Bi-LSTM model for NER in English.

语料库train.txt的前15行：

```
played	on	Monday	(	home	team	in	CAPS	)	:
VBD	IN	NNP	(	NN	NN	IN	NNP	)	:
O	O	O	O	O	O	O	O	O	O
American	League
NNP	NNP
B-MISC	I-MISC
Cleveland	2	DETROIT	1
NNP	CD	NNP	CD
B-ORG	O	B-ORG	O
BALTIMORE	12	Oakland	11	(	10	innings	)
VB	CD	NNP	CD	(	CD	NN	)
B-ORG	O	B-ORG	O	O	O	O	O
TORONTO	5	Minnesota	3
TO	CD	NNP	CD
B-ORG	O	B-ORG	O
......
```



使用Keras创建Bi-LSTM模型的示意图如下：

![Bi-LSTM模型的示意图](https://github.com/percent4/DL_4_NER/blob/master/NERSystem/LSTM_model.png)


该模型在原始数据的训练集上的准确率在99%以上，在测试集上的准确率在95%以上。

对新的数据集进行测试：

自己想的三个句子：

输入为：
```
sent = 'James is a world famous actor, whose home is in London.'
```
输出结果为：

```
['James', 'is', 'a', 'world', 'famous', 'actor', ',', 'whose', 'home', 'is', 'in', 'London', '.']
['B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O']
NER识别结果：
PERSON:  James
LOCATION:  London
```
输入为：
```
sent = 'Oxford is in England, Jack is from here.'
```
输出为：
```
['Oxford', 'is', 'in', 'England', ',', 'Jack', 'is', 'from', 'here', '.']
['B-PER', 'O', 'O', 'B-LOC', 'O', 'B-PER', 'O', 'O', 'O', 'O']
NER识别结果：
PERSON:  Oxford
LOCATION:  England
PERSON:  Jack
```

输入为：
```
sent = 'I love Shanghai.'
```
输出为：
```
['I', 'love', 'Shanghai', '.']
['O', 'O', 'B-LOC', 'O']
NER识别结果：
LOCATION:  Shanghai
```

在上面的例子中，只有Oxford的识别效果不理想，模型将它识别为PERSON，其实应该是ORGANIZATION。

&emsp;&emsp;接下来是三个来自CNN和wikipedia的句子：

输入为：
```
sent = "the US runs the risk of a military defeat by China or Russia"
```
输出为：
```
['the', 'US', 'runs', 'the', 'risk', 'of', 'a', 'military', 'defeat', 'by', 'China', 'or', 'Russia']
['O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC']
NER识别结果：
LOCATION:  US
LOCATION:  China
LOCATION:  Russia
```
输入为：
```
sent = "Home to the headquarters of the United Nations, New York is an important center for international diplomacy."
```
输出为：
```
['Home', 'to', 'the', 'headquarters', 'of', 'the', 'United', 'Nations', ',', 'New', 'York', 'is', 'an', 'important', 'center', 'for', 'international', 'diplomacy', '.']
['O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
NER识别结果：
ORGANIZATION:  United Nations
LOCATION:  New York
```

输入为：
```
sent = "The United States is a founding member of the United Nations, World Bank, International Monetary Fund."
```
输出为:
```
['The', 'United', 'States', 'is', 'a', 'founding', 'member', 'of', 'the', 'United', 'Nations', ',', 'World', 'Bank', ',', 'International', 'Monetary', 'Fund', '.']
['O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'B-ORG', 'I-ORG', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']
NER识别结果：
LOCATION:  United States
ORGANIZATION:  United Nations
ORGANIZATION:  World Bank
ORGANIZATION:  International Monetary Fund
```

&emsp;&emsp;这三个例子识别全部正确。
