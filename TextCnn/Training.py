#encoding:utf-8
from Text_Cnn import *
from data_processing import *
from Parameters import Parameters as pm
import os


def train():

    tensorboard_dir = './tensorboard/Text_cnn'
    save_dir = './checkpoints/Text_cnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Loading Training data...")
    x_train, y_train = process(pm.train_filename, wordid, cat_to_id, pm.seq_length)
    x_val, y_val = process(pm.val_filename, wordid, cat_to_id, pm.seq_length)
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch + 1)
        num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_train, y_train, pm.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = model.feed_data(x_batch, y_batch, pm.keep_prob)
            _, global_step, train_summary, train_loss, train_accuracy = session.run([model.optimizer, model.global_step,
                                                                   merged_summary, model.loss, model.accuracy], feed_dict=feed_dict )
            if global_step % 100 == 0:
                val_loss, val_accuracy = model.evaluate(session, x_val, y_val)
                print(global_step, train_loss, train_accuracy, val_loss, val_accuracy)

            if (global_step + 1) % num_batchs == 0:
                print("Saving model...")
                saver.save(session, save_path, global_step=global_step)

        pm.lr *= pm.lr_decay

if __name__ == '__main__':

    pm = pm
    filenames = [pm.train_filename, pm.test_filename, pm.val_filename]

    categories, cat_to_id = read_category()
    wordid = get_wordid(pm.vocab_filename)
    pm.vocab_size = len(wordid)

    pm.pre_trianing = get_word2vec(pm.vector_word_npz)

    model = TextCnn()

    train()
