import tensorflow as tf
from Parameters_rnn import parameters as pm
from data_processing_rnn import batch_iter, sequence


class RnnModel(object):

    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='sequen_length')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.rnn()

    def rnn(self):

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=[pm.vocab_size, pm.embedding_dim],
                                        initializer=tf.constant_initializer(pm.pre_trianing))
            self.embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('cell'):
            cell = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            cells = [cell for _ in range(pm.num_layers)]
            Cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


        with tf.name_scope('rnn'):
            #hidden一层 输入是[batch_size, seq_length, hidden_dim]
            #hidden二层 输入是[batch_size, seq_length, 2*hidden_dim]
            #2*hidden_dim = embendding_dim + hidden_dim
            output, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=self.embedding_input, sequence_length=self.seq_length, dtype=tf.float32)
            output = tf.reduce_sum(output, axis=1)
            #output:[batch_size, seq_length, hidden_dim]

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(output, keep_prob=self.keep_prob)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([pm.hidden_dim, pm.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[pm.num_classes]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))#计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            #对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            #global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    def feed_data(self, x_batch, y_batch, seq_len, keep_prob):

        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.seq_length: seq_len,
                     self.keep_prob: keep_prob}

        return feed_dict

    def evaluate(self, sess, x, y):

        batch_test = batch_iter(x, y, pm.batch_size)
        for x_batch, y_batch in batch_test:
            seq_len = sequence(x_batch)
            feet_dict = self.feed_data(x_batch, y_batch, seq_len, 1.0)
            loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feet_dict)

        return loss, accuracy



