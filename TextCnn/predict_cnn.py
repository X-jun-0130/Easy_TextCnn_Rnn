#encoding:utf-8
import os
import jieba
import random
import numpy as np
from Text_Cnn import *
from data_processing import read_category, get_wordid, get_word2vec, process
from Parameters import Parameters as pm


def read_file(filename):

    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    contents = []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                _, content = line.split('\t')
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        word.extend(jieba.lcut(blk))
                contents.append(word)
            except:
                pass
    return contents


def Process(sentences, word_to_id, max_length=600):


    data_id = []
    for i in range(len(sentences)):
        data_id.append([word_to_id[x] for x in sentences[i] if x in word_to_id])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')

    return x_pad


def val():
    pre_label = []
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/Text_cnn')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    val_x, val_y = process(pm.val_filename, wordid, cat_to_id, max_length=600)
    batch_val = batch_iter(val_x, val_y, batch_size=64)
    for x_batch, y_batch in batch_val:
        pre_lab = session.run(model.predicitions, feed_dict={model.input_x: x_batch,
                                                             model.keep_pro: 1.0})
        pre_label.extend(pre_lab)
        label.extend(y_batch)
    return pre_label, label




if  __name__ == '__main__':

    pm = pm
    sentences = []
    label2 = []
    categories, cat_to_id = read_category()
    wordid = get_wordid(pm.vocab_filename)
    pm.vocab_size = len(wordid)
    pm.pre_trianing = get_word2vec(pm.vector_word_npz)
    model = TextCnn()

    pre_label, label = val()

    correct = np.equal(pre_label, np.argmax(label, 1))
    accuracy = np.mean(np.cast['float32'](correct))
    print('accuracy:', accuracy)
    print(pre_label[:10])
    print(np.argmax(label, 1)[:10])
