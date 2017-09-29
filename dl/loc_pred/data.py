import numpy as np
import os.path
import json
import tensorflow as tf
def processing(filename):
    from itertools import groupby
    from collections import Counter
    from datetime import datetime
    time_format = '%Y-%m-%dT%H:%M:%SZ'
    def select_users(threshold):
        clean_file = os.path.join(os.path.dirname(filename), 'user_sequence.txt')
        if os.path.isfile(clean_file):
            visit_list = []
            for line in open(clean_file):
                visit_list.append(json.loads(line.strip()))
        else:
            lines = [line.strip().split('\t') for line in open(filename)]
            cnt = Counter(uid for (uid, _, _, _, _) in lines)
            users = set(uid for uid, count in cnt.items() if count >= threshold)
            records = [(uid, datetime.strptime(time, time_format), locid) for (uid, time, _, _, locid) in lines if uid in users]
            visit_list = []
            for uid, group in groupby(sorted(records, key=lambda x:x[0]), key=lambda x:x[0]):
                time_loc = [(t.weekday()*24 + t.hour, l) for _, t, l in sorted(group, key=lambda x:x[1])]
                visit_list.append((uid, time_loc))
            with open(clean_file, 'w') as out_file:
                for (uid, time_loc) in visit_list:
                    out_file.write(json.dumps((uid, time_loc))+'\n')
        return visit_list



    def select_locations(loc_sequence, threshold):
        uid_locid = ((uid, loc) for (uid, time_loc) in loc_sequence for time, loc in time_loc)
        cnt = Counter(locid for uid, locid in uid_locid)
        locations = set(cnt.keys())
        frequent = set(locid for locid, count in cnt.items() if count >= threshold)
        rare = locations - frequent
        return frequent, rare

    clean_index_file = os.path.join(os.path.dirname(filename), 'user_sequence_index.txt')
    if os.path.isfile(clean_index_file):
        loc_sequence_index = []
        for line in open(clean_index_file):
            loc_sequence_index.append(json.loads(line.strip()))
    else:
        loc_sequence = select_users(threshold=50)
        user2ind = dict((u,ind) for (ind, (u, _)) in enumerate(loc_sequence))
        locations, rare = select_locations(loc_sequence, threshold=10)
        loc2index = dict((loc, index) for index, loc in enumerate(locations))
        next_loc = len(loc2index)
        for loc in rare:
            loc2index[loc] = next_loc

        loc_sequence_index = []
        for (uid, time_loc) in loc_sequence:
            seq = [(time, loc2index[loc]) for time, loc in time_loc]
            loc_sequence_index.append((user2ind[uid], seq))

        with open(clean_index_file, 'w') as out_file:
            for (uid, time_loc) in loc_sequence_index:
                out_file.write(json.dumps((uid, time_loc))+'\n')

    return loc_sequence_index

def sequence_mask(seq_len, max_seq_len):
    mask = np.zeros([len(seq_len), max_seq_len])
    for (b,v) in enumerate(seq_len):
        for i in range(v):
            mask[b, i] = 1
    return mask

def get_test(loc_seq_index, max_seq_len):
    instances = [time_loc[-max_seq_len-1:-1] for (u, time_loc) in loc_seq_index]
    instances = [[i[1] for i in inst] for inst in instances]
    target = [time_loc[-1][1] for (u, time_loc) in loc_seq_index]
    seq_len = [len(inst) for inst in instances]
    weight = sequence_mask(seq_len, max(seq_len))

    return instances, target, seq_len, weight



def get_batch(loc_seq, batch_size, max_seq_len):
    instances = []
    target=[]
    seq_len=[]
    for batch_pos in np.random.choice(range(1,len(loc_seq)-1), size=(batch_size), replace=True):
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


def get_test_dataset(loc_seq_index):
    source = [np.array([l for t,l in time_loc[:-1]], np.int32).tostring() for _, time_loc in loc_seq_index]
    target = [[time_loc[-1][1]] for uid, time_loc in loc_seq_index]
    dataset = tf.contrib.data.Dataset.from_tensor_slices((source, target))
    def gen_ele(x, y):
        x = tf.decode_raw(tf.convert_to_tensor(x),tf.int32)
        weight = tf.cast(tf.greater(x,-1), tf.float32)
        return x,y,weight,[tf.shape(x)[0]]
        #return {'x':x, 'y':y, 'weight':weight}
    dataset = dataset.map(gen_ele)
    #iterator = dataset.make_initializable_iterator()
    #next_element = iterator.get_next()
    #return iterator, next_element
    return dataset

def get_train_dataset(loc_seq_index):
    loc_seq_str = [np.array([l for t,l in time_loc[:-1]], np.int32).tostring() for _, time_loc in loc_seq_index]
    uid_seq = [uid for uid, time_loc in loc_seq_index]
    dataset = tf.contrib.data.Dataset.from_tensor_slices((uid_seq, loc_seq_str))
    def gen_ele(u, e):
        x = tf.decode_raw(tf.convert_to_tensor(e),tf.int32)
        y = x[1:]
        x = x[:-1]
        weight = tf.cast(tf.greater(x,-1), tf.float32)
        return x,y,weight,[tf.shape(x)[0]]
        #return {'x':x, 'y':y, 'weight':weight}
    dataset = dataset.map(gen_ele)
    #iterator = dataset.make_initializable_iterator()
    #next_element = iterator.get_next()
    #return iterator, next_element
    return dataset

def gen_train_test(loc_seq_index):
    train = []
    test = []
    for u, time_loc in loc_seq_index:
        for t in range(1, len(time_loc)):
            target_time, target_loc = time_loc[t]
            seq = time_loc[:t]
            time_seq, loc_seq = zip(*seq)
            if t < len(time_loc) - 1:
                train.append((target_loc,loc_seq))
            else:
                test.append((target_loc, loc_seq))
    return train, test

if __name__ == "__main__":
    import os.path
    import json
    filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
    loc_seq_index = processing(filename)
    batch_size = 3
    max_seq_len = 5


    with tf.Session() as sess:
        for u in range(2):
            X, Y = get_batch(loc_seq_index[u][1], batch_size, max_seq_len)




