import tensorflow as tf
from data import processing,get_test
from data import sequence_mask
from lang_model import embed_seq
import numpy as np


def prepare_batches(loc_seq_index, rare_loc, batch_size, max_seq_len):
    rowcol = []
    for row, (u, time_loc) in enumerate(loc_seq_index):
        rowcol.extend([(row, col, min(col, max_seq_len)) for col, (t, l) in enumerate(time_loc[:-1]) if l != rare_loc and col > 0])
    rowcol = sorted(rowcol, key=lambda e:(e[2],np.random.randn(1,1)))
    batches = []
    num_batches = (len(rowcol)+batch_size-1)/batch_size
    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index+1)*batch_size, len(rowcol))
        batches.append(rowcol[batch_start:batch_end])

    return batches

def get_batch(loc_seq_index, batch):
    instances = []
    target=[]
    seq_len=[]
    for row, col, sub in batch:
        target.append(loc_seq_index[row][1][col][1])
        instances.append([l for t,l in loc_seq_index[row][1][col-sub:col]])
        seq_len.append(sub)

    max_seq_len = max(seq_len)
    for batch_index in range(len(instances)):
        instances[batch_index] = instances[batch_index] + [0]*(max_seq_len-len(instances[batch_index]))
    weight = sequence_mask(seq_len, max_seq_len)
    return instances, target, seq_len, weight


# states : tensor of size [batch][time][att_size]
def attention(query, states):
    att_size = states.get_shape()[2].value
    hidden = tf.layers.dense(states, att_size, use_bias=False)
    v = tf.Variable(tf.zeros([att_size]))
    if query is not None:
        y = tf.layers.dense(query, att_size, use_bias=False)
        s = tf.reduce_sum(v * tf.tanh(hidden + tf.reshape(y, [-1, 1, att_size])), [2])
    else:
        s = tf.reduce_sum(v * tf.tanh(hidden), [2])

    s = tf.nn.softmax(s)
    return s ### batch x time


# seq:  tensor, of size [batch][time]
# labels: tensor, of size [batch]
# seq_len: tensor of [batch], indicating length of each sequence
def classifier_seq(seq, labels, weight_mask, num_loc, embed_size, seq_len, num_samples=-1, k=1, keep_prob=1.0):
    seq_embed = embed_seq(seq, num_loc, embed_size)
    fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(embed_size),input_keep_prob=keep_prob)
    bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(embed_size),input_keep_prob=keep_prob)
    attention_state, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, seq_embed, dtype=tf.float32, sequence_length=seq_len, time_major=False)
    attention_state = tf.concat(attention_state, -1)
    weight = attention(None, attention_state)
    weight *= weight_mask
    weight /= tf.reduce_sum(weight, [-1], keep_dims=True)
    attention_out = tf.reduce_sum(attention_state * tf.expand_dims(weight, [-1]), [1])
    W = tf.Variable(tf.zeros([embed_size*2, num_loc]))
                                #stddev=1.0 / math.sqrt(embed_size))
    b = tf.Variable(tf.zeros(num_loc))
    logit = tf.matmul(attention_out, W) + b
    _, pred_k_loc = tf.nn.top_k(logit, k)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred_k_loc[:,0], labels), tf.float32))
    if num_samples > 0:
        W_t = tf.transpose(W)
        step_loss = tf.nn.nce_loss(weights=W_t, biases=b,
                                               labels=tf.expand_dims(labels,axis=[-1]), inputs=attention_out,
                                               num_sampled=num_samples, num_classes=num_loc)
    else:
        step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logit)
    loss = tf.reduce_sum(step_loss)
    tf.summary.scalar('cross_entropy', loss)
    tf.summary.scalar('accuracy', acc)

    return loss, acc, pred_k_loc



def main():
    import os.path
    import json
    filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
    loc_seq_index = processing(filename)
    loc_seq_index = loc_seq_index[:1000]
    num_locations = max(l for (u, time_loc) in loc_seq_index for t,l in time_loc) + 1
    print('{0} locations, {1} users'.format(num_locations, len(loc_seq_index)))
    batch_size = 64
    max_seq_len = 10
    epocs = 50
    embedding_size = 50
    learning_rate = 0.1
    print('embed_size:{0}, max sequence length:{1}, batch size:{2}, learn_rate:{3}'.format(embedding_size, max_seq_len, batch_size, learning_rate))

    test = get_test(loc_seq_index, max_seq_len)
    batches = prepare_batches(loc_seq_index, -1, batch_size, max_seq_len)

    seq_input = tf.placeholder(tf.int32, shape=[None, None], name='input_seq')
    class_output = tf.placeholder(tf.int32, shape=[None], name='output_class')
    seq_len = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
    weight_mask = tf.placeholder(tf.float32, shape=[None, None], name='weight_mask')
    keep_prob = tf.placeholder(tf.float32)
    loss, acc_op, pred_top_op = classifier_seq(seq=seq_input, labels=class_output, weight_mask=weight_mask, num_loc=num_locations,
                             embed_size=embedding_size, seq_len=seq_len, k=50, num_samples=-1, keep_prob=keep_prob)
    merged = tf.summary.merge_all()

    train_op = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('/home/dlian/data/location_prediction/gowalla/train', sess.graph)
        test_writer = tf.summary.FileWriter('/home/dlian/data/location_prediction/gowalla/test')
        sess.run(tf.global_variables_initializer())
        total_loss = 0
        for iter in range(3):
            summary = None
            for batch_index in range(len(batches)):
                batch = batches[batch_index]
                X, Y, length, weight = get_batch(loc_seq_index, batch)
                _, loss_value, summary = sess.run([train_op,loss, merged], feed_dict={seq_input: X, class_output: Y, seq_len: length,
                                                  weight_mask: weight, keep_prob:0.5})
                total_loss += loss_value

            train_writer.add_summary(summary)

            X, Y, length, weight = test
            acc, pred, summary = sess.run([acc_op,pred_top_op, merged], feed_dict={seq_input: X, class_output: Y, seq_len: length,
                                               weight_mask: weight, keep_prob:1})
            test_writer.add_summary(summary)
            print total_loss, acc
            total_loss = 0

            with open('/home/dlian/data/location_prediction/gowalla/pred{0}.txt'.format(iter), 'w') as fout:
                for ii, (x, p, y) in enumerate(zip(X, pred[:,0], Y)):
                    if p != y:
                        fout.writelines('{3}, {0}, {1}, {2}\n'.format(y, p, x, ii))
        train_writer.close()
        test_writer.close()





if __name__ == "__main__":
    main()
