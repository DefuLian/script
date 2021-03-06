def markov_model(seq_all_users, ratio):
    import numpy as np
    from eval import computing_metric
    from collections import Counter
    k = 50
    def suff_estimate(seq):
        dictd = {}
        for (p,c) in zip(seq,seq[1:]):

            if p in dictd:
                if c in dictd[p]:
                    dictd[p][c] += 1
                else:
                    dictd[p][c] = 1
            else:
                dictd[p] = {}
                dictd[p][c] = 1
        return dictd
    y = []
    x = []
    for uid, user_seq in seq_all_users.items():
        train_len = int(round(len(user_seq) * ratio))
        seq_train = [l for (t,l,_,_) in user_seq[:train_len]]
        #locations = list(set(seq))
        cnt = Counter(seq_train)
        dictd = suff_estimate(seq_train)

        for pos in range(train_len, len(user_seq)):
            t, l, _,_ = user_seq[pos]
            y.append(l)
            prev = user_seq[pos-1][1]
            if prev in dictd:
                pred = sorted(dictd[prev].items(), key=lambda v: -v[1])[:k]
                pred = [key for (key,val) in pred]
            else:
                pred = [loc for loc, count in cnt.most_common(k)]
                #if k < len(locations):
                #    pred = np.random.choice(locations, replace=False, size=[k])
                #else:
                #    pred = np.random.permutation(locations)
            x.append(pred)
    #with open('/home/dlian/data/location_prediction/gowalla/mktest.txt', 'w') as fout:
    #    lines = (line for line in open('/home/dlian/data/location_prediction/gowalla/pred12.txt'))
    #    lines = ((int(line.strip().split(',')[0]), line) for line in lines)
    #    ids = dict(lines)
    #    fout.write('\n'.join(['{2}, {1}, {0}, {3}'.format(e1[0],e2, ii, ids[ii]) for ii,(e1,e2) in enumerate(zip(x, y)) if ii in ids and e2 == e1[0]]))
    ndcg, acc = computing_metric(y, x)
    ndcg /= len(y)
    acc /= len(y)+0.0
    return ndcg, acc

if __name__ == "__main__":
    from test_keras import processing
    from test_keras import Data_Clean_Dove_Code
    import pickle
    #processing('E:/data/gowalla/Gowalla_totalCheckins.txt')
    #filename = '/home/dove/data/Gowalla_totalCheckins.txt'
    filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
    loc_seq_index = processing(filename)
    #loc_seq_index = dict(loc_seq_index.items()[:2000])
    #loc_seq_index = dict(Data_Clean_Dove_Code(pickle.load(open('/home/dlian/data/location_prediction/Gowalla_check_40_poi_40_filter','rb'))))
    ndcg,acc = markov_model(loc_seq_index, 0.8)
    print(ndcg, acc)