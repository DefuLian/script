import tensorflow as tf
def training():
    tf.reset_default_graph()
    weight_input = tf.placeholder(tf.int32, shape=(None,), name='example_weight')
    train_input_seq = tf.placeholder(tf.int32, shape=(None, None), name='input_sequence')
    train_output_seq = tf.placeholder(tf.int32, shape=(None, None), name='output_sequence')
    test_input_seq = tf.placeholder(tf.int32, shape=(None, None), name='input_sequence')
    loss, test_pred = lm(train_input_seq, train_output_seq, test_input_seq, num_locations, embedding_size, weight_input)
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iter in range(20):
            loss_val, _ = sess.run([loss, train_op],
                                   feed_dict={train_input_seq: x, train_output_seq: y, weight_input:weight, test_input_seq: [x[:-1]]})
            print(loss_val)