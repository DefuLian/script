import tensorflow as tf
import numpy as np
from seq4model import classifier_seq
from test import processing,get_batch,get_test
from evaluation import computing_metric
def sequence_mask(seq_len, max_seq_len):
    mask = np.zeros([len(seq_len), max_seq_len])
    for (b,v) in enumerate(seq_len):
        for i in range(v):
            mask[b, i] = 1
    return mask



def main():
    import os.path
    import json
    filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
    loc_seq_index = processing(filename)
    loc_seq_index = loc_seq_index[:2000]
    num_locations = max(l for (u, time_loc) in loc_seq_index for t,l in time_loc) + 1
    batch_size = 30
    max_seq_len = 10
    epocs = 100
    embedding_size = 100

    test = get_test(loc_seq_index, batch_size, max_seq_len)

    seq_input = tf.placeholder(tf.int32, shape=[None, None], name='input_seq')
    class_output = tf.placeholder(tf.int32, shape=[None], name='output_class')
    seq_len = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
    weight_mask = tf.placeholder(tf.float32, shape=[None, None], name='weight_mask')
    loss, top_k = classifier_seq(seq=seq_input, labels=class_output, weight_mask=weight_mask, num_loc=num_locations,
                             embed_size=embedding_size, seq_len=seq_len, k=50, num_samples=-1)

    train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iter in range(epocs):
            total_loss = 0
            for u in range(len(loc_seq_index)):
                X, Y, length, max_length = get_batch(loc_seq_index[u][1], batch_size, max_seq_len)
                _, loss_value = sess.run([train_op,loss], feed_dict={seq_input: X, class_output: Y, seq_len: length,
                                                  weight_mask: sequence_mask(length, max_length)})
                total_loss += loss_value
            print total_loss
            ndcg = 0
            acc = 0
            total = 0
            for batch_index in range(len(test)):
                X, Y, length, max_length = test[batch_index]
                top_k_loc = sess.run(top_k, feed_dict={seq_input: X, seq_len: length,
                                                   weight_mask: sequence_mask(length, max_length)})
                ndcg_batch, acc_batch = computing_metric(Y, top_k_loc)
                ndcg += ndcg_batch
                acc += acc_batch
                total += len(length)
            ndcg /= total
            acc /= float(total)
            print(ndcg, acc)

if __name__ == "__main__":
    #processing('E:/data/gowalla/Gowalla_totalCheckins.txt')
    main()




