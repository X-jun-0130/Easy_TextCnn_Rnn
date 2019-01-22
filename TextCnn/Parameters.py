# -*- coding: utf-8 -*-
class Parameters(object):

    embedding_size = 100     #dimension of word embedding
    vocab_size = 10000       #number of vocabulary
    pre_trianing = None      #use vector_char trained by word2vec

    seq_length = 600         #max length of sentence
    num_classes = 10         #number of labels

    num_filters = 128        #number of convolution kernel
    kernel_size = [2, 3, 4]  #size of convolution kernel

    keep_prob = 0.5          #droppout
    lr= 0.001                #learning rate
    lr_decay= 0.9            #learning rate decay
    clip= 5.0                #gradient clipping threshold

    num_epochs = 5           #epochs
    batch_size = 64          #batch_size


    train_filename='./data/cnews.train.txt'  #train data
    test_filename='./data/cnews.test.txt'    #test data
    val_filename='./data/cnews.val.txt'      #validation data
    vocab_filename='./data/vocab_word.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file
