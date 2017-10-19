def flat_gps_points(points):
    def clip(v, min_v, max_v):
        return min(max(v, min_v), max_v)
    def flat_point(p):
        import math
        lat, lon = p
        lat = clip(lat, -85.05112878, 85.05112878)
        lon = clip(lon, -180, 180)
        x = (lon + 180) / 360
        sinLatitude = math.sin(lat * math.pi / 180)
        y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)
        return [x, y]
    return [flat_point(p) for p in points]
def processing(filename):
    import os.path
    import json
    import pickle
    from itertools import groupby
    from collections import Counter
    from datetime import datetime
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans
    from geopy.distance import vincenty
    time_format = '%Y-%m-%dT%H:%M:%SZ'
    def getdb():
        from itertools import groupby,islice
        import numpy as np
        import pickle
        import os.path
        dbfile = os.path.join(os.path.dirname(filename), 'locdb.dat')
        if os.path.isfile(dbfile):
            loc2latlon = pickle.load(open(dbfile, 'rb'))
        else:
            lines = (line.strip().split('\t') for line in open(filename))
            data = set((loc, lat, lon) for u, t, lat, lon, loc in lines)
            loc2latlon = {}
            for key, group in groupby(sorted(data, key=lambda a:a[0]), key=lambda a:a[0]):
                latlon = [[float(lat), float(lon)] for _, lat, lon in group]
                loc2latlon[key] = np.mean(latlon, axis=0)
            pickle.dump(loc2latlon, open(dbfile, 'wb'))
        return loc2latlon
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
        frequent = dict((locid, count) for locid, count in cnt.items() if count >= threshold)
        return frequent


    clean_index_file = os.path.join(os.path.dirname(filename), 'user_sequence_index')
    if os.path.isfile(clean_index_file):
        sequence = pickle.load(open(clean_index_file, 'rb'))
    else:
        user_threshold = 40
        loc_threshold = 40
        loc_seq = select_users(threshold=user_threshold)
        loc2count = select_locations(loc_seq, loc_threshold)
        loc2index = dict(zip(loc2count.keys(), range(0, len(loc2count))))
        sequence = {}
        ind2loc = [None] * len(loc2count)
        ind2count = [0] * len(loc2count)
        for loc, ind in loc2index.items():
            ind2loc[ind] = loc
            ind2count[ind] = loc2count[loc]
        locdb = getdb()
        points = [locdb[loc] for loc in ind2loc]
        points = flat_gps_points(points)
        db = cKDTree(points)
        def get_nearby_popular(location_index):
            query = points[location_index]
            _, index = db.query(query, k=20)
            p = [ind2count[i] if i != location_index else 0 for i in index]
            p_sum = sum(p)
            p = [v/float(p_sum) for v in p]
            index_new = np.random.choice(index, 3, replace=False, p=p)
            return [i + 1 for i in index_new]
        for user_id, visit in loc_seq.items():
            visit = [(t, l) for t, l in visit if l in loc2index]
            if len(visit) <= user_threshold:
                continue
            all_points = flat_gps_points([locdb[l] for t, l in visit])
            model = KMeans(n_clusters=2, random_state=0).fit(all_points)
            labels_sorted = [label for label, _ in Counter(model.labels_).most_common(2)]
            presentation = model.transform(all_points)
            presentation = presentation[:,labels_sorted]
            new_visit = [((t % 24 + 1), loc2index[l]+1, p, get_nearby_popular(loc2index[l])) for (t, l), p in zip(visit, presentation)]
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
    from collections import Counter
    instances = []
    for uid, visit in enumerate(loc_seq.values()):
        dd = Counter(l for _, l, _, _ in visit[:2])
        for pos in range(2, len(visit)):
            mc = [l for l, c in dd.most_common(2)]
            if len(mc) < 2:
                mc = [0] + mc
            y_t, y, _, _ = visit[pos]
            dd[y] += 1
            x = visit[max(pos-max_seq_len, 0):pos]
            if len(x) < max_seq_len:
                x = [(0, 0, [0.,0.], [0,0,0])]*(max_seq_len-len(x)) + x
            x_t, x, x_p, x_n = zip(*x)
            instances.append((uid, list(x_t), list(x), list(x_p), list(x_n), mc, y_t, y))
    return instances



def main(train, vali, test, num_loc, num_user, num_time, max_seq_len = 10):
    from keras.models import Model
    from keras.layers import Dense, Embedding, LSTM, Input, Concatenate, Reshape, Dropout, BatchNormalization
    from keras.layers import Add, Lambda, Bidirectional, TimeDistributed, Activation
    from keras.optimizers import Adagrad
    from keras.losses import sparse_categorical_crossentropy
    from keras.metrics import sparse_categorical_accuracy
    from keras.callbacks import Callback, EarlyStopping,TensorBoard
    from attention import Attention, SimpleAttention
    import keras.backend as K
    from keras.utils import plot_model
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
        def __init__(self, x, y):
            super(AttentionCallback, self).__init__()
            self.x = x
            self.y = y

        def on_epoch_end(self, epoch, logs=None):
            self.att_output = [layer.output[-1] for layer in self.model.layers if 'attention' in layer.name]
            self.functor = K.function(self.model.input, self.att_output)
            att = self.functor(self.x)
            names = [layer.name for layer in self.model.layers if 'attention' in layer.name]
            np.set_printoptions(precision=3)
            print('\n')
            for n, a in zip(names, att):
                print(n +'\t'+ str(np.mean(a,axis=0)))

    xy_train = [np.array(list(e)) for e in zip(*train)]
    xy_vali = [np.array(list(e)) for e in zip(*vali)]
    xy_test = [np.array(list(e)) for e in zip(*test)]
    x_train, y_train = xy_train[:6], xy_train[6:]
    x_vali, y_vali = xy_vali[:6], xy_vali[6:]
    x_test, y_test = xy_test[:6], xy_test[6:]

    embeding_size = 200
    hidden_units = embeding_size
    learning_rate = 0.1
    batch_size = 64
    max_epochs = 30
    batch_norm = True
    time_dropout_rate = 0.8
    loc_dropout_rate = time_dropout_rate
    time_pred_weight = 1
    activation = 'relu'
    att_method = 'lba' #'lba'#'lba' # 'ga' 'cba'


    print embeding_size, hidden_units, batch_norm, time_dropout_rate, loc_dropout_rate, time_pred_weight, activation, att_method

    user = Input(shape=(1,), dtype='int32', name='user_id')
    user_embed = Embedding(output_dim=embeding_size, input_dim=num_user, input_length=1)(user)
    user_embed = Reshape((embeding_size,))(user_embed)
    user_embed = Activation(activation=activation)(user_embed)

    time_seq = Input(shape=(max_seq_len,), dtype='int32', name='time_sequence')
    time_embedding = Embedding(output_dim=embeding_size, input_dim=num_time, mask_zero=True)
    time_seq_embed = time_embedding(time_seq)

    loc_embedding = Embedding(output_dim=embeding_size, input_dim=num_loc)
    most_freq = Input(shape=(2,), dtype='int32', name='most_freq')
    most_freq_embed = loc_embedding(most_freq)
    most_freq_embed = Reshape([embeding_size*2])(most_freq_embed)

    loc_seq = Input(shape=(max_seq_len,), dtype='int32', name='loc_sequence')
    nearby_seq = Input(shape=(max_seq_len, 3), dtype='int32', name='nearby_sequence')
    loc_nearby_seq = Concatenate(-1)([nearby_seq, Lambda(lambda inputs:K.expand_dims(inputs))(loc_seq)])
    loc_nearby_seq_embed = loc_embedding(loc_nearby_seq)
    loc_nearby_seq_embed = SimpleAttention('cba')(loc_nearby_seq_embed)

    spatial_seq = Input(shape=(max_seq_len, 2), dtype='float32', name='spatial_sequence')
    spatial_seq_embed = Dense(embeding_size, use_bias=False, activation='relu')(spatial_seq)

    spatial_temporal_embeding = Lambda(lambda inputs:K.concatenate(inputs,axis=-1), mask=lambda input, mask: mask[0],
           output_shape=lambda shape:(shape[0][0],shape[0][1], sum(s[2] for s in shape)))([time_seq_embed, loc_nearby_seq_embed, spatial_seq_embed])


    spatial_temporal_lstm = LSTM(hidden_units, return_sequences=True)(spatial_temporal_embeding)
    pred_feat = SimpleAttention()(spatial_temporal_lstm)

    pred_feat = Concatenate(axis=-1, name='time_pred_feat')([pred_feat, user_embed, most_freq_embed])

    time_pred_softmax = Dense(num_time, activation='softmax', name='time')(Dropout(time_dropout_rate)(pred_feat))
    pred_time_embed = Lambda(lambda inputs:time_embedding(K.argmax(inputs)), lambda shape: (shape[0], embeding_size), name='pred_time_embed')(time_pred_softmax)
    pred_time_embed = Activation(activation=activation)(pred_time_embed)


    loc_pred_feat = Concatenate(axis=-1, name='loc_pred_feat')([pred_time_embed, pred_feat])
    loc_pred_softmax = Dense(num_loc, activation='softmax', name='loc')(Dropout(loc_dropout_rate)(loc_pred_feat))



    optimizer = Adagrad(lr=learning_rate)
    model = Model(inputs=[user, time_seq, loc_seq, spatial_seq, nearby_seq, most_freq], outputs=[time_pred_softmax, loc_pred_softmax])
    model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, loss_weights=[time_pred_weight, 1],
                  metrics={'loc':sparse_categorical_accuracy,'time':sparse_categorical_accuracy})
    #earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    tensorboard = TensorBoard('/home/dlian/data/location_prediction/gowalla/logs/', histogram_freq=1, embeddings_freq=1)
    model.metrics_names = ['loss', 'time_loss','loc_loss','time_acc','loc_acc']
    plot_model(model, to_file='/home/dlian/model.png', show_shapes=True)
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=max_epochs, validation_data=(x_vali, y_vali))#, callbacks=[AttentionCallback(x_vali,y_vali)])

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
    loc_seq = dict(loc_seq.items()[:2000])
    #loc_seq = dict(Data_Clean_Dove_Code(pickle.load(open('/home/dlian/data/location_prediction/Gowalla_check_40_poi_40_filter','rb'))))
    max_seq_len = 10
    num_loc = max(l for u, time_loc in loc_seq.items() for t, l, _, _ in time_loc) + 1
    num_time = max(t for u, time_loc in loc_seq.items() for t, l, _, _ in time_loc) + 1
    print('{0} users, {1} locations, {2} time'.format(len(loc_seq), num_loc, num_time))
    train, test = split(loc_seq, 0.8)
    #train, vali = split(train, 0.9)
    train_set = gen_data(train, max_seq_len)
    np.random.shuffle(train_set)
    #vali_set = gen_data(vali, max_seq_len)
    test_set = gen_data(test, max_seq_len)
    main(train_set, test_set, test_set, num_loc, len(loc_seq), num_time, max_seq_len)
