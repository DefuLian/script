import tensorflow as tf
from keras.layers import Dot, Input, Lambda, multiply, dot
import numpy as np
from keras import backend as K
tf.reset_default_graph()
PAD = 0
EOS = 1
batch_size = 5
seq_len_val = 20
num_locations = 10
embedding_size = 50
hidden_units = embedding_size
x = np.random.randint(1, num_locations, size=(batch_size, seq_len_val))
#y = np.vstack((x.transpose(), [EOS] * batch_size))
#y = y.transpose()[:, 1:]
y = x.copy()



seq_len = tf.placeholder(tf.int32, name='sequence_length')
input_seq = tf.placeholder(tf.int32, shape=(None, None), name='input_sequence')
output_seq = tf.placeholder(tf.int32, shape=(None, None), name='output_sequence')
embeddings = tf.Variable(tf.random_uniform([num_locations, embedding_size], -1.0, 1.0), dtype=tf.float32)

input_embedding = tf.nn.embedding_lookup(embeddings, input_seq)
output_embedding = tf.nn.embedding_lookup(embeddings, output_seq)

rnn_cell = tf.contrib.rnn.LSTMCell(hidden_units)

rnn_out, rnn_state = tf.nn.dynamic_rnn(
    rnn_cell, input_embedding,
    dtype=tf.float32, time_major=False,
)

W = tf.Variable(tf.zeros([hidden_units, num_locations]))
W_t = tf.Variable(tf.zeros([num_locations, hidden_units]))
b = tf.Variable(tf.zeros(num_locations))
rnn_out_1 = tf.matmul(tf.reshape(rnn_out, (-1, hidden_units)), W)
rnn_out_2 = tf.reshape(rnn_out_1, (-1, seq_len, num_locations))
step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_seq, logits=rnn_out_2)
output_seq_reshape = tf.reshape(output_seq, [-1,1])
rnn_out_reshape = tf.reshape(rnn_out, [-1, hidden_units])
#step_loss = tf.nn.sampled_softmax_loss(W_t, b, output_seq_reshape , rnn_out_reshape, 500, num_locations)
#step_loss_reshape = tf.reshape(step_loss, [-1, seq_len])
#loss = tf.reduce_mean(tf.reduce_mean(step_loss_reshape, axis=[1]))
loss = tf.reduce_mean(step_loss)
train_op = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(20):
        loss_val, _ = sess.run([loss, train_op],
                               feed_dict={input_seq: x, output_seq: y, seq_len: seq_len_val})
        print(loss_val)

