#!/usr/bin/evn python

import numpy as np
from itertools import groupby, chain
from fastFM import als
from numpy import random
from collections import defaultdict
import time
import scipy.sparse as sp
import heapq

data_dir = "/home/dove/dataset/location/"
dataset = "Beijing"
file_suffix = ".txt"



def getFeatFromInstance(i, instance, M, N):
    u,cur,nex = instance
    return [(i, u), (i, cur+M), (i, nex+M+N)]

def getFeatFromInstances(start_i, instances, M, N):
    return [e for (i, instance) in enumerate(instances) for e in getFeatFromInstance(start_i + i, instance, M, N)]

def sampleNegative(train_data, nearby):
    result = []
    for(u, values) in train_data.iteritems():
        for (cur, nextset) in values.iteritems():
            result.extend((u,cur,l, 1) for l in nextset)
            trueNeg = [l for l in nearby[cur] if l not in nextset]
            if(len(trueNeg)>0):
                if(len(trueNeg) > len(nextset)):
                    neg = [(u, cur, n, -1) for n in random.choice(trueNeg, len(nextset), replace=False)]
                    result.extend(neg)
                else:
                    result.extend((u, cur, n, -1) for n in trueNeg)
    return result
num_loc = 57122
num_user = 6703

def transForm(train_data, nearby):
    train_data = sampleNegative(train_data, nearby)
    y = [v for (u, i, j, v) in train_data]
    X = [(u,i,j) for (u, i, j, v) in train_data]

    ll = getFeatFromInstances(0, X, num_user, num_loc)
    rows = [a for (a,b) in ll]
    cols = [b for (a,b) in ll]

    XX = sp.coo_matrix((np.ones_like(cols, dtype=np.float64),(rows, cols)),shape=(len(train_data),num_user+num_loc+num_loc))

    return (sp.csc_matrix(XX), np.array(y))



def read_train_data():
    train_data = {}
    count = 0
    train_data_file = open(data_dir + dataset + "_TrainData" + file_suffix, 'r')
    for eachline in train_data_file:
        count = count + 1
        raw_data = eachline.strip().split('\t')
        u, lc, li = int(raw_data[0]) - 1, int(raw_data[1]) - 1, int(raw_data[2]) - 1
        if u not in train_data:
            train_data[u] = {}
        if lc not in train_data[u]:
            train_data[u][lc] = set()
        train_data[u][lc].add(li)
    print count
    return train_data

def get_nearby_pois():
    nearby_pois = defaultdict(set)
    geo_nn_file = open(data_dir + dataset + "_Nearbypoi_2km.txt", 'r')
    i = 0
    for eachline in geo_nn_file:
        #print i,eachline
        if(len(eachline.strip())>0):
            data = raw_data = eachline.strip().split('\t')
            nearby_pois[i] = map(lambda k: int(k), data)
        else:
            nearby_pois[i] = [0]*0
        i+=1
    return nearby_pois

def read_test_data():
    test_data = {}
    train_data_file = open(data_dir + dataset + "_TestData" + file_suffix, 'r')
    for eachline in train_data_file:
        raw_data = eachline.strip().split('\t')
        u, lc, li = int(raw_data[0]) - 1, int(raw_data[1]) - 1, int(raw_data[2]) - 1
        if u not in test_data:
            test_data[u] = {}
        if lc not in test_data[u]:
            test_data[u][lc] = set()
        test_data[u][lc].add(li)
    return test_data

def read_visits():
    visits = set()
    train_data_file = open(data_dir + dataset + "_TrainData" + file_suffix, 'r')
    for eachline in train_data_file:
        raw_data = eachline.strip().split('\t')
        u, lc, li = int(raw_data[0]) - 1, int(raw_data[1]) - 1, int(raw_data[2]) - 1
        #visits.add((u, lc, li))
        visits.add((u,lc))
        visits.add((u,li))
    return visits

def evaluation(test_data, visits, nearby_pois, fm):
    top_n =100
    all_precision, all_recall = [], []
    for cnt_u, u in enumerate(test_data):
        print cnt_u
        precision, recall = 0.0, 0.0
        utotal = 0.0
        hits = [0.0]*top_n
        for cnt, (lc, actuals) in enumerate(test_data[u].items()):

            #print cnt
            utotal += len(actuals)
            if (lc >= num_loc):
                continue
            pred_location = [(u, lc, lj) for lj in nearby_pois[lc] if (u, lj) not in visits and lj < num_loc]

            if len(pred_location)==0:
                continue

            ll = getFeatFromInstances(0, pred_location, num_user, num_loc)

            #for e in ll:
            #    print e[0],e[1]
            #print len(pred_location)
            #return;
            rows = [a for (a, b) in ll]
            cols = [b for (a, b) in ll]
            XX = sp.coo_matrix((np.ones_like(cols, dtype=np.float64), (rows, cols)),shape=(len(pred_location),num_user+num_loc+num_loc))
            y = fm.predict_proba(sp.csc_matrix(XX))
            poi_latent_dis = [(a[2], b) for(a, b) in zip(pred_location, y)]

            recommendations = [dd[0] for dd in heapq.nlargest(top_n, poi_latent_dis, lambda x:x[1])]
            for (ir, rec) in enumerate(recommendations):
                if (rec in actuals):
                    for jr in range(ir, top_n):
                        hits[jr] += 1.0
        precision = [h/len(test_data[u])/(k+1.0) for (k,h) in enumerate(hits)]
        recall = [h/utotal for (k,h) in enumerate(hits)]
        all_precision.append(precision)
        all_recall.append(recall)
    precision, recall = [0.0] * top_n, [0.0] * top_n
    for k in range(0, top_n):
        for p in all_precision:
            precision[k] += p[k] / (cnt_u+1)
        for r in all_recall:
            recall[k] += r[k] / (cnt_u+1)
    print ",".join(str(r) for r in recall)
    print "\n"
    print ",".join(str(r) for r in precision)
    #print(cnt_u, np.mean(all_precision), np.mean(all_recall))



if __name__== '__main__':
    import sys
    import pickle
    t = time.time()
    train_data = read_train_data()
    nearby_pois = get_nearby_pois()
    test_data = read_test_data()
    visits = read_visits()
    print("Data Loaded... Elapsed", time.time() - t)

    X, y = transForm(train_data, nearby_pois)

    reg = float(sys.argv[1])
    print reg
    fm = als.FMClassification(rank=50,l2_reg_V=reg)
    fm.fit(X, y)

    print("Train finished... Elapsed", time.time() - t)

    evaluation(test_data,visits,nearby_pois,fm)

    print X[0,0], X[0,6703], X[0,63824], y[0], y[1]
