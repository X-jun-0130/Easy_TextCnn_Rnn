#encoding:utf-8
import tensorflow as tf

class TextConfig():

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

    num_epochs = 10          #epochs
    batch_size = 64          #batch_size
    print_per_batch = 100    #print result

    train_filename='./data/cnews.train.txt'  #train data
    test_filename='./data/cnews.test.txt'    #test data
    val_filename='./data/cnews.val.txt'      #validation data
    vocab_filename='./data/vocab_word.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file

class TextCnn(object):
    def __init__(self, config):

        self.config = config
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.cnn()
    def cnn(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_trianing))
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_expand = tf.expand_dims(embedding_input, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.config.kernel_size):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name= 'w')
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedding_expand, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(h, ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name='pool')#
                pooled_outputs.append(pooled)

        num_filter_total = self.config.num_filters * len(self.config.kernel_size)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_pro)

        with tf.name_scope('output'):
            w = tf.get_variable("w", shape=[num_filter_total, self.config.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='b')
            #l2_loss += tf.nn.l2_loss(w)
            #l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name='scores')
            self.pro = tf.nn.softmax(self.scores)
            self.predicitions = tf.argmax(self.pro, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels= self.input_y)
            self.loss = tf.reduce_mean(losses)  #对交叉商取均值非常有必要

        with tf.name_scope('optimizer'):
            #退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            #learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
                                                       #decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            #optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            #self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            #no.2
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            #对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            #global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predicitions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')







