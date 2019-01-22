#encoding:utf-8
import tensorflow as tf
from Parameters import Parameters as pm
from data_processing import *

class TextCnn(object):

    def __init__(self):

        self.input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y')
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable("embeddings", shape=[pm.vocab_size, pm.embedding_size],
                                             initializer=tf.constant_initializer(pm.pre_trianing))
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_expand = tf.expand_dims(embedding_input, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(pm.kernel_size):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, pm.embedding_size, 1, pm.num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name= 'w')
                b = tf.Variable(tf.constant(0.1, shape=[pm.num_filters]), name='b')
                conv = tf.nn.conv2d(self.embedding_expand, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(h, ksize=[1, pm.seq_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name='pool')
                pooled_outputs.append(pooled)

        num_filter_total = pm.num_filters * len(pm.kernel_size)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_pro)

        with tf.name_scope('output'):
            w = tf.get_variable("w", shape=[num_filter_total, pm.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[pm.num_classes]), name='b')

            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name='scores')
            self.pro = tf.nn.softmax(self.scores)
            self.predicitions = tf.argmax(self.pro, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)  #对交叉熵取均值非常有必要

        with tf.name_scope('optimizer'):
            #退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            #learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
                                                       #decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            #optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            #self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            #no.2
            optimizer = tf.train.AdamOptimizer(pm.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            #对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            #global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predicitions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')

    def feed_data(self, x_batch, y_batch, keep_prob):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.keep_pro: keep_prob
        }
        return feed_dict

    def evaluate(self, sess, x, y):
        batch_eva = batch_iter(x, y, pm.batch_size)
        for x_batch, y_batch in batch_eva:
            feed_dict = self.feed_data(x_batch, y_batch, 1.0)
            loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return loss, accuracy




