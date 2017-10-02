import tensorflow as tf
import math
from data import processing
import numpy as np

def get_train_dataset(loc_seq_index):
    uid = []
    loc = []
    target = []
    for u, time_loc in loc_seq_index:
        train_seq = [l for _, l in time_loc[:-1]]
        for c, n in zip(train_seq, train_seq[1:]):
            uid.append(u)
            loc.append(c)
            target.append(n)
    return tf.contrib.data.Dataset.from_tensor_slices((uid, loc, target))

def get_test_dataset(loc_seq_index):
    uid = []
    loc = []
    target = []
    for u, time_loc in loc_seq_index:
        uid.append(u)
        loc.append(time_loc[-2][1])
        target.append(time_loc[-1][1])
    return tf.contrib.data.Dataset.from_tensor_slices((uid, loc, target))

filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
loc_seq_index = processing(filename)
num_loc = max(l for (u, time_loc) in loc_seq_index for t,l in time_loc) + 1
num_users = len(loc_seq_index)
num_sampled = 20
batch_size = 128
embedding_size = 128
epocs = 20
k = 50



graph = tf.Graph()
with graph.as_default():

    train_set = get_train_dataset(loc_seq_index)
    test_set = get_test_dataset(loc_seq_index)

    test = test_set.batch(num_users)
    test_iterator = test.make_initializable_iterator()
    test_next_ele = test_iterator.get_next()

    train = train_set.repeat(epocs)
    train = train.batch(batch_size)
    train_iterator = train.make_initializable_iterator()
    train_next_ele = train_iterator.get_next()


    user_input = tf.placeholder(tf.int32, shape=[None], name='uid')
    loc_input = tf.placeholder(tf.int32, shape=[None], name='loc')
    labels = tf.placeholder(tf.int32, shape=[None], name='target')

    # Ops and variables pinned to the CPU because of missing GPU implementation
    # Look up embeddings for inputs.
    embeddings_users = tf.Variable(tf.random_uniform([num_users, embedding_size], -1.0, 1.0))
    embed_user = tf.nn.embedding_lookup(embeddings_users, user_input)

    embeddings_item = tf.Variable(tf.random_uniform([num_loc, embedding_size], -1.0, 1.0))
    embed_item = tf.nn.embedding_lookup(embeddings_item, loc_input)
    #embed = tf.layers.dense(tf.concat([embed_user, embed_item], -1),embedding_size, activation=tf.nn.tanh)
    embed = tf.concat([embed_user, embed_item], -1)
    nce_weights = tf.Variable(
        tf.zeros([num_loc, embedding_size * 2]))
                            #stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([num_loc]))

    logit = tf.add(tf.matmul(embed, tf.transpose(nce_weights)), nce_biases)
    _, pred_k_loc = tf.nn.top_k(logit, k)
    compare = tf.cast(tf.equal(tf.expand_dims(labels, [-1]), pred_k_loc), tf.float32)
    coefficient = tf.constant(1/np.log2(np.array(range(1,k+1), np.float32, ndmin=2)+1))
    compare *= coefficient
    ndcg_op = tf.reduce_mean(tf.reduce_sum(compare,axis=1))
    acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_k_loc[:,0], labels), tf.float32))

    if num_sampled > 0:
        loss = tf.reduce_sum(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=tf.expand_dims(labels,[-1]),
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=num_loc))
    else:
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit))

    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    test_uid_value, test_loc_value, test_target_value = sess.run(test_next_ele)
    step = 0
    avg_loss = 0
    while True:
        step += 1
        try:
            uid_value, loc_value, target_value = sess.run(train_next_ele)
            _, loss_value = sess.run([optimizer, loss], feed_dict={user_input:uid_value, loc_input:loc_value, labels:target_value})
            avg_loss += loss_value
            if step % 2000 == 0:
                avg_loss /= 2000
                ndcg, acc = sess.run([ndcg_op, acc_op], feed_dict={user_input:test_uid_value, loc_input:test_loc_value, labels:test_target_value})
                print('Average loss at step:{0}, average_train_loss: {1}, testing_ndcg:{2}, test_acc:{3}'.format(step, avg_loss, ndcg, acc))
                avg_loss = 0

        except tf.errors.OutOfRangeError:
            break

