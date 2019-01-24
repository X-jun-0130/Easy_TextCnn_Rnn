#encoding:utf-8
import numpy as np
from Text_Cnn import *
from data_processing import read_category, get_wordid, get_word2vec, process，batch_iter, sequence
from Parameters import Parameters as pm

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
