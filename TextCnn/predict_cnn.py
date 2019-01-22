#encoding:utf-8
import os
import jieba
import random
from Text_Cnn import *
from data_processing import read_category, get_wordid, get_word2vec
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


def process(sentences, word_to_id, max_length=600):


    data_id = []
    for i in range(len(sentences)):
        data_id.append([word_to_id[x] for x in sentences[i] if x in word_to_id])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')

    return x_pad



def predict(sentences2id):
    save_path = tf.train.latest_checkpoint('./checkpoints/bText_cnn')

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    predict = session.run(model.predicitions, feed_dict={model.input_x: sentences2id,
                                                           model.keep_pro: 1.0})
    return predict







if  __name__ == '__main__':

    pm = pm
    sentences = []
    label = []
    categories, cat_to_id = read_category()
    wordid = get_wordid(pm.vocab_filename)
    pm.vocab_size = len(wordid)
    pm.pre_trianing = get_word2vec(pm.vector_word_npz)
    model = TextCnn()

    with codecs.open(pm.val_filename, 'r', encoding='utf-8') as f:
        sample = random.sample(f.readlines(), 10)

    for sentence in sample:
        sentence = sentence.strip().split('\t')
        sentences.append(sentence[1])
        label.append(sentence[0])

    sentences2id = process(sentences, wordid, max_length=600)


    modellabel=predict(sentences2id)
    label2word = [categories[i] for i in modellabel]

    for k in range(len(sentences)):
        print(sentences[k][:50]+'...')
        print('正确标签: '+str(label[k]))
        print('预测标签: '+str(label2word[k]))




