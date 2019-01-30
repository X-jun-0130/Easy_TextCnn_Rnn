#encoding:utf-8
class parameters(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 100      # 词向量维度
    num_classes = 10        # 类别数
    vocab_size = 10000       # 词汇表达小
    pre_trianing = None      #use vector_char trained by word2vec
    seq_length = 250
    num_layers= 2          # 隐藏层层数
    hidden_dim = 100        # 隐藏层神经元

    keep_prob = 0.5        # dropout保留比例
    learning_rate = 0.001    # 学习率
    clip = 5.0
    lr_decay = 0.9           #learning rate decay
    batch_size = 64          # 每批训练大小
    num_epochs = 3           # 总迭代轮次


    train_filename='./data/cnews.train.txt'  #train data
    test_filename='./data/cnews.test.txt'    #test data
    val_filename='./data/cnews.val.txt'      #validation data
    vocab_filename='./data/vocab_word.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file
