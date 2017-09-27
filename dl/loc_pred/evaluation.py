def computing_metric(Y, pred):
    import math
    import numpy as np
    def ndcg():
        dcg = 0
        for ind in range(len(Y)):
            y = Y[ind]
            if isinstance(pred, list):
                y_ = pred[ind]
            else:
                y_ = pred[ind,:]
            for p in range(len(y_)):
                if y_[p] == y:
                    dcg += 1/math.log(1+p+1, 2)
        return dcg
    def acc():
        acc = 0
        for ind in range(len(Y)):
            y = Y[ind]
            if isinstance(pred, list):
                y_ = set(pred[ind])
            else:
                y_ = set(pred[ind,:])
            if y in y_:
                acc += 1
        return acc
    return ndcg(), acc()