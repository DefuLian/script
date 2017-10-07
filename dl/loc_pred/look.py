
def processing(filename):
    import os.path
    import json
    import pickle
    from itertools import groupby
    from collections import Counter
    from datetime import datetime
    time_format = '%Y-%m-%dT%H:%M:%SZ'
    def select_users(threshold):
        clean_file = os.path.join(os.path.dirname(filename), 'user_sequence.txt')
        if os.path.isfile(clean_file):
            visit_list = json.load(open(clean_file))
        else:
            lines = [line.strip().split('\t') for line in open(filename)]
            cnt = Counter(uid for (uid, _, _, _, _) in lines)
            users = set(uid for uid, count in cnt.items() if count >= threshold)
            records = [(uid, datetime.strptime(time, time_format), locid) for (uid, time, _, _, locid) in lines if uid in users]
            visit_list = {}
            for uid, group in groupby(sorted(records, key=lambda x:x[0]), key=lambda x:x[0]):
                time_loc = [(t.weekday()*24 + t.hour, l) for _, t, l in sorted(group, key=lambda x:x[1])]
                visit_list[uid] = time_loc

            json.dump(visit_list, open(clean_file, 'w'))
        return visit_list

    def select_locations(loc_sequence, threshold):
        uid_locid = ((uid, loc) for (uid, time_loc) in loc_sequence.items() for time, loc in time_loc)
        cnt = Counter(locid for uid, locid in uid_locid)
        locations = set(cnt.keys())
        frequent = set(locid for locid, count in cnt.items() if count >= threshold)
        rare = locations - frequent
        return frequent, rare


    clean_index_file = os.path.join(os.path.dirname(filename), 'user_sequence_index')
    if os.path.isfile(clean_index_file):
        sequence = pickle.load(open(clean_index_file, 'rb'))
    else:
        user_threshold = 40
        loc_threshold = 40
        loc_seq = select_users(threshold=user_threshold)
        locations, rare_loc = select_locations(loc_seq, loc_threshold)
        loc2index = dict(zip(locations, range(1, len(locations) + 1)))
        sequence = {}
        for user_id, visit in loc_seq.items():
            new_visit = [(t, loc2index[l]) for t, l in visit if l in loc2index]
            if len(new_visit) > user_threshold:
                sequence[int(user_id)] = new_visit
        pickle.dump(sequence, open(clean_index_file, 'wb'))

    return sequence

def split(loc_seq, ratio):
    train = {}
    test = {}
    for user, visit in loc_seq.items():
        train_len = int(round(len(visit) * ratio))
        train[user] = visit[:train_len]
        test[user] = visit[train_len:]
    return train, test

def gen_data(loc_seq, max_seq_len):
    instances = []
    for uid, visit in enumerate(loc_seq.values()):
        for pos in range(1, len(visit)):
            x = visit[max(pos-max_seq_len, 0):pos]
            if len(x) < max_seq_len:
                x = x + [(0,0)]*(max_seq_len-len(x))
            _, x = zip(*x)
            _, y = visit[pos]
            instances.append((list(x), y))
    return instances



def main(train, test, num_loc, max_seq_len = 10):
    from keras.models import Model
    from keras.layers import Dense, Embedding, LSTM, Input
    from keras.optimizers import Adagrad
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.callbacks import Callback

    import tensorflow as tf

    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)

    train_set = tf.contrib.data.Dataset.from_tensor_slices((list(x_train), list(y_train)))
    train_set.batch(batch_size = 128)
    test_set = tf.contrib.data.Dataset.from_tensor_slices((list(x_test), list(y_test)))
    test_set.batch(batch_size=1000)
    iterator = tf.contrib.data.Iterator.from_structure(train_set.output_types,
                                   train_set.output_shapes)
    next_element = iterator.get_next()
    train_init_op = iterator.make_initializer(train_set)
    test_init_op = iterator.make_initializer(test_set)

    embeding_size = 200
    hidden_units = 100
    learning_rate = 0.1
    max_epochs = 20

    loc_seq = tf.placeholder(tf.int32, shape=[None, max_seq_len])
    target = tf.placeholder(tf.int32, shape=[None,])
    loc_seq_embed = Embedding(output_dim=embeding_size, input_dim=num_loc, mask_zero=True, input_length=max_seq_len)(loc_seq)

    loc_seq_lstm = LSTM(hidden_units, dropout=0.3, recurrent_dropout=0, return_sequences=False)(loc_seq_embed)
    loc_seq_logit = Dense(num_loc, activation='linear')(loc_seq_lstm)
    step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=loc_seq_logit, labels=)
    loss = tf.reduce_mean(step_loss)
    train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)



if __name__ == "__main__":
    import numpy as np
    import pickle
    loc_seq = processing('/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt')
    max_seq_len = 10
    num_loc = max(l for u, time_loc in loc_seq.items() for t, l in time_loc) + 1
    print('{0} users, {1} locations'.format(len(loc_seq), num_loc))
    train, test = split(loc_seq, 0.8)
    train_set = gen_data(train, max_seq_len)
    np.random.shuffle(train_set)
    test_set = gen_data(test, max_seq_len)
    main(train_set, test_set, num_loc, max_seq_len)
