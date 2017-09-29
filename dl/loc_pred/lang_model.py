import tensorflow as tf
import numpy as np


def embed_seq(input_seq, num_locations, embedding_size):
    embeddings = tf.Variable(tf.random_uniform([num_locations, embedding_size], -1.0, 1.0), dtype=tf.float32)
    input_embedding = tf.nn.embedding_lookup(embeddings, input_seq)
    return input_embedding

def seq_classifer(source,
                  target,
                  weight_mask,
                  num_loc,
                  embed_size,
                  seq_len,
                  num_samples=-1,
                  k=1):
    input_embedding = embed_seq(source, num_loc, embed_size)
    cell = tf.contrib.rnn.LSTMCell(embed_size)

    outputs, state = tf.nn.dynamic_rnn(
        cell, input_embedding,
        dtype=tf.float32, sequence_length=seq_len)


    W = tf.Variable(tf.zeros([embed_size, num_loc]))
    b = tf.Variable(tf.zeros(num_loc))

    outputs_flat = tf.reshape(outputs, [-1, embed_size])
    logit_flat = tf.matmul(outputs_flat, W) + b

    target_flat = tf.reshape(target, [-1])

    if num_samples>0:
        W_t = tf.transpose(W)
        step_loss = tf.nn.sampled_softmax_loss(weights=W_t, biases=b,
                                               labels=tf.expand_dims(target_flat,[-1]), inputs=outputs_flat,
                                               num_sampled=num_samples, num_classes=num_loc)
    else:
        step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_flat, logits=logit_flat)

    step_loss *= tf.reshape(weight_mask, [-1])
    loss = tf.reduce_sum(step_loss)

    test_target = target[:,-1]
    test_output = outputs[:,-1,:]

    test_logit = tf.matmul(test_output, W) + b
    _, pred_k_loc = tf.nn.top_k(test_logit, k)
    compare = tf.cast(tf.equal(tf.expand_dims(test_target, axis=[-1]), pred_k_loc), tf.float32)
    coefficient = tf.constant(1/np.log2(np.array(range(1,k+1), np.float32, ndmin=2)+1))
    compare *= coefficient
    ndcg = tf.reduce_mean(tf.reduce_sum(compare,axis=1))
    acc = tf.reduce_mean(tf.cast(tf.equal(pred_k_loc[:,0], test_target), tf.float32))

    return loss, ndcg, acc

def get_train_batches(loc_seq, batch_size):
    from data import sequence_mask
    num_seq = len(loc_seq)
    num_batches = (num_seq + batch_size - 1) / batch_size
    batches = []
    for batch_index in range(num_batches):
        batch_start = batch_size * batch_index
        batch_end = min(batch_size * (batch_index + 1), num_seq)
        seq = loc_seq[batch_start:batch_end]
        seq = [[l for _, l in time_loc[:-1]] for (u, time_loc) in seq]
        seq_len = [len(e)-1 for e in seq]
        max_len = max(seq_len)
        seq = [(s[:-1] + [0]*(max_len-len(s)+1), s[1:] + [0]*(max_len-len(s)+1)) for s in seq]
        x, y = zip(*seq)
        weight = sequence_mask(seq_len, max_len)
        batches.append((list(x),list(y),weight,seq_len))
    return batches


def get_test_batches(loc_seq, batch_size):
    from data import sequence_mask
    num_seq = len(loc_seq)
    num_batches = (num_seq + batch_size - 1) / batch_size
    batches = []
    for batch_index in range(num_batches):
        batch_start = batch_size * batch_index
        batch_end = min(batch_size * (batch_index + 1), num_seq)
        seq = loc_seq[batch_start:batch_end]
        source = [[l for _, l in time_loc[:-1]] for (u, time_loc) in seq]
        target = [[time_loc[-1][1]] for (u, time_loc) in seq]
        seq_len = [len(e) for e in source]
        max_len = max(seq_len)
        weight = sequence_mask(seq_len, max_len)
        source = [s + [0]*(max_len-len(s)) for s in source]
        batches.append((source,target, weight, seq_len))
    return batches

def main():
    import os.path
    import json
    from data import processing
    filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
    loc_seq_index = processing(filename)
    loc_seq_index = sorted(loc_seq_index, key=lambda e:len(e[1]))
    loc_seq_index = loc_seq_index[:100]
    num_locations = max(l for (u, time_loc) in loc_seq_index for t,l in time_loc) + 1
    batch_size = 1
    epocs = 100
    embedding_size = 100



    train_batches = get_train_batches(loc_seq_index, batch_size)
    test_batches = get_test_batches(loc_seq_index, 100)

    seq_input = tf.placeholder(tf.int32, shape=[None, None], name='x')
    class_output = tf.placeholder(tf.int32, shape=[None, None], name='y')
    weight_mask = tf.placeholder(tf.float32, shape=[None, None], name='weight')
    seq_len = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
    loss, ndcg_op, acc_op = seq_classifer(source=seq_input, target=class_output, weight_mask=weight_mask, num_loc=num_locations,
                             seq_len=seq_len, embed_size=embedding_size, k=50, num_samples=-1)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tf.train.AdagradOptimizer(learning_rate=0.2).minimize(loss, global_step=global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for iter in range(epocs):
            loss_value_total = 0
            for batch_index in range(len(train_batches)):
                X, Y, weight, length = train_batches[batch_index]
                #print length
                _, loss_value = sess.run([train_op, loss], feed_dict={seq_input: X, class_output: Y, weight_mask: weight, seq_len:length})
                #print('{0}th batch, loss:{1}, batch_size:{2}'.format(batch_index, loss_value, len(length)))
                loss_value_total += loss_value
            print loss_value_total

            count = 0
            ndcg_total = 0
            acc_total = 0

            for batch_index in range(len(test_batches)):
                X, Y, weight, length = test_batches[batch_index]
                ndcg,acc = sess.run([ndcg_op, acc_op], feed_dict={seq_input: X, class_output: Y, weight_mask: weight, seq_len:length})
                ndcg_total += ndcg
                acc_total += acc
                count += 1
            print(ndcg_total, acc_total)



if __name__ == "__main__":
    main()
