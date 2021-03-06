import tensorflow as tf
from data import processing,get_test
from eval import computing_metric
from data import sequence_mask
from lang_model import embed_seq
import numpy as np
## train_input_seq, train_out_seq, test_intput_seq and train_weight is placeholder

def get_batch(loc_seq, batch_size, max_seq_len, rare_loc):
    instances = []
    target=[]
    seq_len=[]
    index = [ind for ind in range(1,len(loc_seq)-1) if loc_seq[ind][1] != rare_loc]
    for batch_pos in np.random.choice(index, size=(batch_size), replace=True):
        start = max(batch_pos - max_seq_len,0)
        sub = [l for t,l in loc_seq[start:batch_pos]]
        instances.append(sub)
        seq_len.append(len(sub))
        target.append(loc_seq[batch_pos][1])
    max_seq_len = min(max_seq_len, max(seq_len))
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
def classifier_seq(seq, labels, weight_mask, num_loc, embed_size, seq_len, num_samples=-1, k=1):
    seq_embed = embed_seq(seq, num_loc, embed_size)
    fw_cell = tf.contrib.rnn.LSTMCell(embed_size)
    bw_cell = tf.contrib.rnn.LSTMCell(embed_size)
    attention_state, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, seq_embed, dtype=tf.float32, sequence_length=seq_len, time_major=False)
    attention_state = tf.concat(attention_state, -1)
    weight = attention(None, attention_state)
    weight *= weight_mask
    weight /= tf.reduce_sum(weight, [-1], keep_dims=True)
    attention_out = tf.reduce_sum(attention_state * tf.expand_dims(weight, [-1]), [1])

    W = tf.Variable(tf.zeros([embed_size*2, num_loc]))
    b = tf.Variable(tf.zeros(num_loc))
    logit = tf.matmul(attention_out, W) + b
    _, pred_k_loc = tf.nn.top_k(logit, k)
    compare = tf.cast(tf.equal(tf.expand_dims(labels,axis=[-1]), pred_k_loc), tf.float32)
    coefficient = tf.constant(1/np.log2(np.array(range(1,k+1), np.float32, ndmin=2)+1))
    compare *= coefficient
    ndcg = tf.reduce_mean(tf.reduce_sum(compare,axis=1))
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


    return loss, ndcg, acc



def main():
    import os.path
    import json
    filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
    loc_seq_index = processing(filename)
    loc_seq_index = loc_seq_index[:1000]
    num_locations = max(l for (u, time_loc) in loc_seq_index for t,l in time_loc) + 1
    batch_size = 60
    max_seq_len = 10
    epocs = 50
    embedding_size = 50

    test = get_test(loc_seq_index, max_seq_len)

    seq_input = tf.placeholder(tf.int32, shape=[None, None], name='input_seq')
    class_output = tf.placeholder(tf.int32, shape=[None], name='output_class')
    seq_len = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
    weight_mask = tf.placeholder(tf.float32, shape=[None, None], name='weight_mask')
    loss, ndcg_op, acc_op = classifier_seq(seq=seq_input, labels=class_output, weight_mask=weight_mask, num_loc=num_locations,
                             embed_size=embedding_size, seq_len=seq_len, k=50, num_samples=-1)

    train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iter in range(epocs):
            total_loss = 0
            for u in range(len(loc_seq_index)):
                X, Y, length, weight = get_batch(loc_seq_index[u][1], batch_size, max_seq_len, num_locations-1)
                _, loss_value = sess.run([train_op,loss], feed_dict={seq_input: X, class_output: Y, seq_len: length,
                                                  weight_mask: weight})
                total_loss += loss_value
            print total_loss

            X, Y, length, weight = test
            ndcg, acc = sess.run([ndcg_op, acc_op], feed_dict={seq_input: X, class_output: Y, seq_len: length,
                                               weight_mask: weight})

            print(ndcg, acc)

if __name__ == "__main__":
    main()
