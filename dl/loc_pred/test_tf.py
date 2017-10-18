
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
            new_visit = [(t+1, loc2index[l]) for t, l in visit if l in loc2index]
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
                x = [(0,0)]*(max_seq_len-len(x)) + x
            x_t, x = zip(*x)
            y_t, y = visit[pos]
            instances.append((uid, list(x_t), list(x), y_t, y))
    return instances



def main(train, vali, test, num_loc, num_user, num_time, max_seq_len = 10):
    from keras.models import Model
    from keras.layers import Dense, Embedding, LSTM, Input, Concatenate, Reshape, Dropout, BatchNormalization, Add, Lambda, Bidirectional
    from keras.optimizers import Adagrad
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.callbacks import Callback, EarlyStopping,TensorBoard
    from attention import Attention, SimpleAttention
    import keras.backend as K
    from keras.utils import plot_model
    #K.set_learning_phase(1)
    class TestCallback(Callback):
        def __init__(self, x, y):
            super(TestCallback, self).__init__()
            self.x = x
            self.y = y
        def on_epoch_end(self, epoch, logs=None):
            score = self.model.evaluate(x=self.x, y=self.y, batch_size=1000, verbose=0)
            print('\t')
            print(','.join(['{0}:{1}'.format(n,s) for n, s in zip(model.metrics_names, score)]))
    class AttentionCallback(Callback):
        def __init__(self):
            super(AttentionCallback, self).__init__()
            self.layer2weight = {}
        def on_batch_end(self, batch, logs=None):
            l2w = [(layer.name, layer.weight) for layer in self.model.layers if 'attention' in layer.name]
            for name, weight in l2w:
                w = np.mean(weight, axis=0)
                if name in self.layer2weight:
                    self.layer2weight[name].append(w)
                else:
                    self.layer2weight[name] = [w]
        def on_epoch_end(self, epoch, logs=None):
            for name, weight_list in self.layer2weight:
                weight = np.mean(weight_list, axis=0)
                print('{}  {}\n'.format(name, '  '.join(['{:.3f}'.format(w) for w in weight])))
            self.layer2weight.clear()

    xy_train = [np.array(list(e)) for e in zip(*train)]
    xy_vali = [np.array(list(e)) for e in zip(*vali)]
    xy_test = [np.array(e) for e in zip(*test)]
    x_train, y_train = xy_train[:3], xy_train[3:]
    x_vali, y_vali = xy_vali[:3], xy_vali[3:]
    x_test, y_test = xy_test[:3], xy_test[3:]

    embeding_size = 200
    hidden_units = 100
    learning_rate = 0.1
    batch_size = 64
    max_epochs = 30
    batch_norm = True
    time_dropout_rate = 0.5
    loc_dropout_rate = time_dropout_rate
    time_pred_weight = 1
    activation = 'relu'
    att_method = 'lba' #'lba'#'lba' # 'ga' 'cba'
    attention = True

    print embeding_size, hidden_units, batch_norm, time_dropout_rate, loc_dropout_rate, time_pred_weight, activation, att_method, attention

    user = Input(shape=(1,), dtype='int32', name='user_id')
    user_embed = Embedding(output_dim=embeding_size, input_dim=num_user, input_length=1)(user)
    user_embed = Reshape((embeding_size,))(user_embed)
    user_embed = Dense(hidden_units, activation=activation)(user_embed)
    if batch_norm:
        user_embed = BatchNormalization()(user_embed)

    time_seq = Input(shape=(max_seq_len,), dtype='int32', name='time_sequence')
    time_embedding = Embedding(output_dim=embeding_size, input_dim=num_time, mask_zero=True)
    time_seq_embed = time_embedding(time_seq)
    if not attention:
        time_seq_lstm = Bidirectional(LSTM(hidden_units, return_sequences=False))(time_seq_embed)
    else:
        time_seq_lstm_seq = Bidirectional(LSTM(hidden_units, return_sequences=True))(time_seq_embed)
        #time_seq_lstm = Attention(att_method)([time_seq_lstm_seq, user_embed])
        time_seq_lstm = SimpleAttention(att_method)(time_seq_lstm_seq)
    if batch_norm:
        time_seq_lstm = BatchNormalization()(time_seq_lstm)

    loc_seq = Input(shape=(max_seq_len,), dtype='int32', name='loc_sequence')
    loc_seq_embed = Embedding(output_dim=embeding_size, input_dim=num_loc, mask_zero=True, input_length=max_seq_len)(loc_seq)
    loc_time_seq_embed = Concatenate(axis=-1)([loc_seq_embed, time_seq_embed])

    if not attention:
        time_pred_feat = loc_pred_feat = Bidirectional(LSTM(hidden_units, return_sequences=False))(loc_time_seq_embed)
    else:
        loc_time_seq_lstm_seq = Bidirectional(LSTM(hidden_units, return_sequences=True))(loc_time_seq_embed)
        #time_pred_feat = Attention(att_method)([loc_time_seq_lstm_seq, user_embed])
        time_pred_feat = SimpleAttention(att_method)(loc_time_seq_lstm_seq)

    if batch_norm:
        time_pred_feat = BatchNormalization()(time_pred_feat)

    time_pred_feat = Concatenate(axis=-1, name='time_pred_feat')([time_pred_feat, user_embed, time_seq_lstm])
    time_pred_feat = Dropout(time_dropout_rate)(time_pred_feat)
    time_pred_feat = Dense(hidden_units, activation=activation)(time_pred_feat)
    if batch_norm:
        time_pred_feat = BatchNormalization()(time_pred_feat)

    time_pred_softmax = Dense(num_time, activation='softmax', name='time')(Dropout(time_dropout_rate)(time_pred_feat))
    pred_time_embed = Lambda(lambda inputs:time_embedding(K.argmax(inputs)), lambda shape: (shape[0], embeding_size), name='pred_time_embed')(time_pred_softmax)
    pred_time_embed = Dense(hidden_units, activation=activation)(pred_time_embed)
    if batch_norm:
        pred_time_embed = BatchNormalization()(pred_time_embed)

    user_time_embed = Concatenate(axis=-1)([user_embed, pred_time_embed])

    if attention:
        #loc_pred_feat = Attention(att_method)([loc_time_seq_lstm_seq, user_time_embed])
        loc_pred_feat = SimpleAttention(att_method)(loc_time_seq_lstm_seq)

    if batch_norm:
        loc_pred_feat = BatchNormalization()(loc_pred_feat)

    loc_pred_feat = Concatenate(axis=-1, name='loc_pred_feat')([loc_pred_feat, user_time_embed])
    loc_pred_feat = Dropout(loc_dropout_rate)(loc_pred_feat)
    loc_pred_feat = Dense(hidden_units, activation=activation)(loc_pred_feat)
    if batch_norm:
        loc_pred_feat = BatchNormalization()(loc_pred_feat)
    loc_pred_softmax = Dense(num_loc, activation='softmax', name='loc')(Dropout(loc_dropout_rate)(loc_pred_feat))



    optimizer = Adagrad(lr=learning_rate)
    model = Model(inputs=[user, time_seq, loc_seq], outputs=[time_pred_softmax, loc_pred_softmax])
    model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, loss_weights=[time_pred_weight, 1],
                  metrics={'loc':sparse_categorical_accuracy,'time':sparse_categorical_accuracy})
    #earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    tensorboard = TensorBoard('/home/dlian/data/location_prediction/gowalla/logs/', histogram_freq=1, embeddings_freq=1)
    model.metrics_names = ['loss', 'time_loss','loc_loss','time_acc','loc_acc']
    plot_model(model, to_file='/home/dlian/model.png', show_shapes=True)
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=max_epochs, validation_data=(x_vali, y_vali))#, callbacks=[AttentionCallback()])

    score = model.evaluate(x=x_test, y=y_test, batch_size=batch_size*10)
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
    #loc_seq = processing('/home/dove/data/Gowalla_totalCheckins.txt')
    loc_seq = processing('/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt')
    loc_seq = dict(loc_seq.items()[:100])
    #loc_seq = dict(Data_Clean_Dove_Code(pickle.load(open('/home/dlian/data/location_prediction/Gowalla_check_40_poi_40_filter','rb'))))
    max_seq_len = 10
    num_loc = max(l for u, time_loc in loc_seq.items() for t, l in time_loc) + 1
    num_time = max(t for u, time_loc in loc_seq.items() for t, l in time_loc) + 1
    print('{0} users, {1} locations, {2} time'.format(len(loc_seq), num_loc, num_time))
    train, test = split(loc_seq, 0.8)
    #train, vali = split(train, 0.9)
    train_set = gen_data(train, max_seq_len)
    np.random.shuffle(train_set)
    #vali_set = gen_data(vali, max_seq_len)
    test_set = gen_data(test, max_seq_len)
    main(train_set, test_set, test_set, num_loc, len(loc_seq), num_time, max_seq_len)
