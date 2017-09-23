import tensorflow as tf

def basic_lm_loss(input_seq,
                  output_seq,
                  num_locations,
                  embedding_size,
                  weight,
                  num_samples=-1):

    embeddings = tf.Variable(tf.random_uniform([num_locations, embedding_size], -1.0, 1.0), dtype=tf.float32)
    input_embedding = tf.nn.embedding_lookup(embeddings, input_seq)
    rnn_cell = tf.contrib.rnn.LSTMCell(embedding_size)

    rnn_out, rnn_state = tf.nn.dynamic_rnn(
        rnn_cell, input_embedding,
        dtype=tf.float32, time_major=False,
    )
    W = tf.Variable(tf.zeros([embedding_size, num_locations]))
    b = tf.Variable(tf.zeros(num_locations))

    max_seq_len = tf.shape(input_seq)[1]
    if num_samples>0:
        W_t = tf.transpose(W)
        output_seq_flat = tf.reshape(output_seq, [-1])
        rnn_out_flat = tf.reshape(rnn_out, [-1, embedding_size])
        step_loss = tf.nn.sampled_softmax_loss(weights=W_t, biases=b,
                                               labels=output_seq_flat, inputs=rnn_out_flat,
                                               num_sampled=num_samples, num_classes=num_locations)
    else:
        rnn_out_1 = tf.matmul(tf.reshape(rnn_out, (-1, embedding_size)), W) + b
        rnn_out_2 = tf.reshape(rnn_out_1, (-1, max_seq_len, num_locations))
        step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_seq, logits=rnn_out_2)

    step_loss_reshape = tf.reshape(step_loss, [-1, max_seq_len])
    step_loss_reshape *= weight
    loss = tf.reduce_mean(tf.reduce_sum(step_loss, axis=[1]))


    #extract prediction part : here we only predict the last element

    test_input_seq = tf.stack([input_seq[row, e-1] for (row, e) in enumerate(tf.unstack(seq_length))])
    test_output_seq = tf.stack([input_seq[row, e-1] for (row, e) in enumerate(tf.unstack(seq_length))])
    test_input_embedding = tf.nn.embedding_lookup(embeddings, test_input_seq)
    test_rnn_out, _ = tf.nn.dynamic_rnn(rnn_cell, test_input_embedding, initial_state=rnn_state, dtype=tf.float32, time_major=False)
    test_pred_score = tf.matmul(test_rnn_out, W) + b
    test_pred_loc = tf.argmax(test_pred_score, 1)
    acc = tf.reduce_mean(tf.case(tf.equal(test_output_seq, test_pred_loc), tf.float32))
    return loss, acc