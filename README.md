# Easy_TextCnn_Rnn
tensorflow TxetCnn RNN

## 本文博客地址：
https://www.jianshu.com/p/f95d472b13ea

# 数据集：
本实验是使用THUCNews的一个子集进行训练与测试，数据集请自行到THUCTC：一个高效的中文文本分类工具包下载，请遵循数据提供方的开源协议;

文本类别涉及10个类别：categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']，每个分类6500条数据；

cnews.train.txt: 训练集(5000*10)

cnews.val.txt: 验证集(500*10)

cnews.test.txt: 测试集(1000*10)

数据集下载链接来源：https://github.com/cjymz886/text-cnn

训练所用的数据，以及训练好的词向量可以下载：链接: https://pan.baidu.com/s/1gka7SgYIRijSaXgRfYZzwA ，密码: mmbk


# 1.利用TextCnn 进行文本分类
## 模型参数
parameters.py

## 预处理
预训练词向量进行embedding

对句子分词，去标点符号

去停用词

文字转数字

padding等

程序在data_processing.py

## 运行步骤
Training.py 
![训练模型与模型在测试集结果](Easy_TextCnn_Rnn/TextCnn/image/train.jpeg)

predict.py 模型用来对文本分类预测

网络结构与图片基本一致





# 2.利用RNN进行文本分类


## 参考
1.Convolutional Neural Networks for Sentence Classification 

2.https://github.com/cjymz886/text-cnn

3.http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow
