
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
            instances.append((uid, list(x), y))
    return instances



def main(train, test, num_loc, num_user, max_seq_len = 10):
    from keras.models import Model
    from keras.layers import Dense, Embedding, LSTM, Input, concatenate, Reshape, Dropout, BatchNormalization, Add
    from keras.optimizers import Adagrad
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.callbacks import Callback

    class TestCallback(Callback):
        def __init__(self, x, y):
            super(TestCallback, self).__init__()
            self.x = x
            self.y = y
        def on_epoch_end(self, epoch, logs=None):
            loss, acc = self.model.evaluate(x=self.x, y=self.y, batch_size=1000, verbose=0)
            print('testing loss:{0}, acc:{1}\n'.format(loss, acc))

    user_train, x_train, y_train = zip(*train)
    user_train, x_train, y_train = np.array(list(user_train)), np.array(list(x_train)), np.array(list(y_train))
    user_test, x_test, y_test = zip(*test)
    user_test, x_test, y_test = np.array(list(user_test)), np.array(list(x_test)), np.array(list(y_test))

    embeding_size = 200
    hidden_units = 100
    learning_rate = 0.1
    batch_size = 64
    max_epochs = 20
    is_batch_norm = True
    print embeding_size, hidden_units, is_batch_norm
    user = Input(shape=(1,), dtype='int32', name='user_id')
    user_embed = Embedding(output_dim=embeding_size, input_dim=num_user, input_length=1)(user)
    user_embed = Reshape((embeding_size,))(user_embed)
    user_embed = Dense(hidden_units, activation='tanh')(user_embed)
    if is_batch_norm:
        user_embed = BatchNormalization()(user_embed)

    loc_seq = Input(shape=(max_seq_len,), dtype='int32', name='loc_sequence')
    loc_seq_embed = Embedding(output_dim=embeding_size, input_dim=num_loc, mask_zero=True, input_length=max_seq_len)(loc_seq)
    loc_seq_lstm = LSTM(hidden_units, dropout=0, recurrent_dropout=0, return_sequences=False)(loc_seq_embed)
    if is_batch_norm:
        loc_seq_lstm = BatchNormalization()(loc_seq_lstm)

    user_loc_seq_concate = concatenate([loc_seq_lstm, user_embed],axis=-1)
    user_loc_seq_concate = Dropout(0.5)(user_loc_seq_concate)
    user_loc_seq_concate = Dense(hidden_units, activation='relu')(user_loc_seq_concate)
    if is_batch_norm:
        user_loc_seq_concate = BatchNormalization()(user_loc_seq_concate)

    #user_loc_seq_concate = Add()([loc_seq_lstm, user_loc_seq_concate, user_embed])

    user_loc_seq_concate = Dropout(0.5)(user_loc_seq_concate)
    loc_seq_softmax = Dense(num_loc, activation='softmax')(user_loc_seq_concate)

    optimizer = Adagrad(lr=learning_rate)
    model = Model(inputs=[user, loc_seq], outputs=loc_seq_softmax)
    model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
    model.fit(x=[user_train, x_train], y=y_train, batch_size=batch_size, epochs=max_epochs,
              validation_split=0, callbacks=[TestCallback([user_test, x_test], y_test)])
    score = model.evaluate(x=[user_test, x_test], y=y_test, batch_size=batch_size*10)
    print(score)

def Data_Clean_Dove_Code(data):
    # f = open('../Data/{0}_totalCheckins.txt'.format(dataset))
    def num2weekhour(deltasec):
        from datetime import datetime, timedelta
        basetime = datetime(2006, 1, 1)
        newtime=basetime+timedelta(seconds=deltasec)
        weekid=newtime.weekday()
        hournum=weekid*24+newtime.hour
        return hournum
    newdata = []
    for eachuser in data:
        m,n=data[eachuser].shape
        tmpall=[]
        tmpall.append(eachuser)
        tmp = []
        for i in range(m):
            weekhour=num2weekhour(int(data[eachuser][i,1]))
            tmp.append((weekhour, int(data[eachuser][i,-1])))
        tmpall.append(tmp)
        tmpall=tuple(tmpall)
        newdata.append(tmpall)
    return newdata

if __name__ == "__main__":
    import numpy as np
    import pickle
    loc_seq = processing('/home/dove/data/Gowalla_totalCheckins.txt')
    #loc_seq = processing('/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt')
    #loc_seq = dict(loc_seq.items()[:2000])
    #loc_seq = dict(Data_Clean_Dove_Code(pickle.load(open('/home/dlian/data/location_prediction/Gowalla_check_40_poi_40_filter','rb'))))
    max_seq_len = 10
    num_loc = max(l for u, time_loc in loc_seq.items() for t, l in time_loc) + 1
    print('{0} users, {1} locations'.format(len(loc_seq), num_loc))
    train, test = split(loc_seq, 0.8)
    train_set = gen_data(train, max_seq_len)
    np.random.shuffle(train_set)
    test_set = gen_data(test, max_seq_len)
    main(train_set, test_set, num_loc, len(loc_seq), max_seq_len)
