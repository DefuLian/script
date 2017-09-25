import tensorflow as tf

## train_input_seq, train_out_seq, test_intput_seq and train_weight is placeholder

def lm(train_input_seq, train_out_seq, test_input_seq, num_locations, embedding_size, train_weight, num_samples=-1):
    train_input_embedding = embed_seq(train_input_seq, num_locations, embedding_size)
    test_input_embedding = embed_seq(test_input_seq, num_locations, embedding_size)
    lm_loss(train_input_embedding, train_out_seq, test_input_embedding,
            num_locations, embedding_size, train_weight, num_samples)

def embed_seq(input_seq, num_locations, embedding_size):
    embeddings = tf.Variable(tf.random_uniform([num_locations, embedding_size], -1.0, 1.0), dtype=tf.float32)
    input_embedding = tf.nn.embedding_lookup(embeddings, input_seq)
    return input_embedding

def lm_loss(train_input_embedding,
                  train_output_seq,
                  test_input_embedding,
                  num_locations,
                  embedding_size,
                  train_weight,
                  num_samples=-1):

    rnn_cell = tf.contrib.rnn.LSTMCell(embedding_size)
    train_rnn_out, train_rnn_state = tf.nn.dynamic_rnn(
        rnn_cell, train_input_embedding,
        dtype=tf.float32, time_major=False,
    )

    test_rnn_out, test_rnn_out = tf.nn.dynamic_rnn(
        rnn_cell,test_input_embedding,
        initial_state=train_rnn_state, dtype=tf.float32, time_major=False
    )
    W = tf.Variable(tf.zeros([embedding_size, num_locations]))
    b = tf.Variable(tf.zeros(num_locations))
    train_output_seq_flat = tf.reshape(train_output_seq, [-1])
    train_rnn_out_flat = tf.reshape(train_rnn_out, [-1, embedding_size])
    if num_samples>0:
        W_t = tf.transpose(W)
        step_loss = tf.nn.sampled_softmax_loss(weights=W_t, biases=b,
                                               labels=train_output_seq_flat, inputs=train_rnn_out_flat,
                                               num_sampled=num_samples, num_classes=num_locations)
    else:
        train_logit_flat = tf.matmul(train_rnn_out_flat, W) + b
        step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_output_seq_flat, logits=train_logit_flat)

    step_loss *= tf.reshape(train_weight, [-1])
    loss = tf.reduce_sum(step_loss)

    test_rnn_out_flat = tf.reshape(test_rnn_out, [-1, embedding_size])
    test_logit_flat = tf.matmul(test_rnn_out_flat, W) + b
    _, test_pred_loc = tf.nn.top_k(test_logit_flat, k=10)

    return loss, test_pred_loc

def embed_list_tensor(input_seq, num_locations, embedding_size):
    embeddings = tf.Variable(tf.random_uniform([num_locations, embedding_size], -1.0, 1.0), dtype=tf.float32)
    input_embedding = [tf.nn.embedding_lookup(embeddings, e) for e in input_seq]
    return input_embedding



# seq:  list of tensor, of size [batch], [time][batch][1]
# labels: tensor, of size [batch]
# seq_len: tensor of [batch], indicating length of each sequence


def attention(query, states):
    from keras.layers import Dense
    att_size = states.get_shape()[2].value
    batch_size = tf.shape(states)[0]
    if query is not None:
        y = Dense(att_size, activation='linear',use_bias=False)(query)
    else:
        y = tf.constant(tf.zeros(batch_size, att_size))
    hidden = Dense(att_size, activation='linear',use_bias=False)(states)
    v = tf.Variable(tf.zeros([att_size]))
    s = tf.reduce_sum(v * tf.tanh(hidden + tf.reshape(y, [-1, 1, att_size])), [2])
    s = tf.nn.softmax(s)
    return s ### batch x time



def classifier_seq(seq, labels, num_loc, embed_size, seq_len, num_samples=-1, k=1):
    T = len(seq)
    seq_embed = embed_list_tensor(seq, num_loc, embed_size)
    fw_cell = tf.contrib.rnn.LSTMCell(embed_size)
    bw_cell = tf.contrib.rnn.LSTMCell(embed_size)
    seq_output, fw_state, bw_state = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, seq_embed, dtype=tf.float32, sequence_length=seq_len)
    attention_state = [tf.reshape(e, [-1, 1, embed_size*2])for e in seq_output]
    attention_state = tf.concat(attention_state, axis=1) # tensor of size batch x time x embed_size
    weight = attention(None, attention_state)
    weight_mask = tf.cast(tf.sequence_mask(seq_len, T), dtype=tf.float32)
    weight *= weight_mask
    weight /= tf.reduce_sum(weight, [-1], keep_dims=True)
    attention_out = tf.reduce_sum(attention_state * tf.reshape(weight, [-1, T, 1]), [1])

    W = tf.Variable(tf.zeros([embed_size, num_loc]))
    b = tf.Variable(tf.zeros(num_loc))

    if num_samples > 0:
        W_t = tf.transpose(W)
        step_loss = tf.nn.sampled_softmax_loss(weights=W_t, biases=b,
                                               labels=labels, inputs=attention_out,
                                               num_sampled=num_samples, num_classes=num_loc)
    else:
        step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=attention_out)
    loss = tf.reduce_sum(step_loss)
    logit = tf.matmul(attention_out, W) + b
    _, pred_k_loc = tf.nn.top_k(logit, k)

    return loss, pred_k_loc

