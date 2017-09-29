import tensorflow as tf
import math
import numpy as np
import collections
import random
data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
      #buffer[:] = data[:span]
      buffer.extend(data[:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


def word2vec(train_inputs, train_labels, embedding_size, num_sampled, vocabulary_size):

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  return loss, embeddings

def gen_data(loc_seq_index):
    num_locations = max(l for (u, time_loc) in loc_seq_index for t,l in time_loc) + 1
    data = []
    for u, time_loc in loc_seq_index:
        seq = [l for t, l in time_loc]
        data.extend(seq)
        data.append(num_locations)
    return data, num_locations + 1

def main():
    from data import processing
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.
    num_sampled = 64 * 10
    num_epocs = 4
    filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
    loc_seq_index = processing(filename)
    data, vocabulary_size = gen_data(loc_seq_index)


    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    loss, embed = word2vec(train_inputs,train_labels, embedding_size, num_sampled, vocabulary_size)
    train_op = tf.train.AdagradOptimizer(1.0).minimize(loss)

    num_steps = num_epocs * len(data) // batch_size

    saver = tf.train.Saver({embed.name:embed})
    with tf.Session() as session:
      # We must initialize all variables before we use them.
      tf.global_variables_initializer().run()
      print('Initialized')

      average_loss = 0
      for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(data,
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = session.run([train_op, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ' out of ', num_steps, ' : ', average_loss)
          average_loss = 0

      saver.save(session, '/home/dlian/data/location_prediction/gowalla/logdir/model.ckpt')

if __name__ == "__main__":
    main()