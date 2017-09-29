def markov_model(seq_all_users):
    import numpy as np
    from eval import computing_metric
    from collections import Counter
    k = 50
    def suff_estimate(seq):
        dict = {}
        for (p,c) in zip(seq,seq[1:]):
            if p in dict:
                if c in dict[p]:
                    dict[p][c] += 1
                else:
                    dict[p][c] = 1
            else:
                dict[p] = {}
                dict[p][c] = 1
        return dict
    y = []
    x = []
    for uid, user_seq in seq_all_users:
        seq = [l for (t,l) in user_seq[:-1]]
        #locations = list(set(seq))
        cnt = Counter(seq)
        dict = suff_estimate(seq)
        prev = user_seq[-2][1]
        target = user_seq[-1][1]
        y.append(target)
        if prev in dict:
            pred = sorted(dict[prev].items(), key=lambda v: -v[1])[:k]
            pred = [key for (key,val) in pred]
        else:
            pred = [loc for loc, count in cnt.most_common(k)]
            #if k < len(locations):
            #    pred = np.random.choice(locations, replace=False, size=[k])
            #else:
            #    pred = np.random.permutation(locations)



        x.append(pred)
    ndcg, acc = computing_metric(y, x)
    ndcg /= len(y)
    acc /= len(y)+0.0
    return ndcg, acc

if __name__ == "__main__":
    from data import processing
    #processing('E:/data/gowalla/Gowalla_totalCheckins.txt')
    filename = '/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt'
    loc_seq_index = processing(filename)
    loc_seq_index = loc_seq_index[:2000]
    ndcg,acc = markov_model(loc_seq_index)
    print(ndcg, acc)