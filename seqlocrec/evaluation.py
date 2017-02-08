import time
import numpy as np
import heapq
from util import Util
from collections import defaultdict




#dataset = "Gowalla"
file_suffix = ".txt"

time_format = "%Y/%m/%d %H:%M:%S"


alpha = 0.2       # linear combination

inf = 0x7fffffff

top_n = 100


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


def get_locations():
    locations = {}
    poi_data_file = open(data_dir + dataset + "_Poi_LatLon.txt", 'r')
    cnt = 0
    for eachline in poi_data_file:
        raw_data = eachline.strip().split('\t')
        locations[cnt] = [float(raw_data[0]), float(raw_data[1])]
        cnt += 1
    return locations

    
def get_nearby_pois_tuple():
    import itertools
    lines = (line.strip().split(',') for line in open(data_dir + dataset + "_Nearbypoi_2km.txt", 'r'))
    tupes = [(int(a[0])-1, int(a[1])-1) for a in lines]
    return dict((key, [l[1] for l in group]) for (key, group) in itertools.groupby(sorted(tupes, key=lambda t: t[0]), lambda t: t[0]));
    
def get_nearby_pois():
    nearby_pois = defaultdict(set)
    geo_nn_file = open(data_dir + dataset + "_Nearbypoi_2km.txt", 'r')
    i = 0;
    for eachline in geo_nn_file:
        #print i,eachline
        if(len(eachline.strip())>0):
            data = raw_data = eachline.strip().split('\t')
            nearby_pois[i] = map(lambda k: int(k), data)
        else:
            nearby_pois[i] = [0]*0;
        i+=1
    return nearby_pois

#def get_nearby_pois():
#    nearby_pois = defaultdict(set)
#    geo_nn_file = open(data_dir + dataset + "_GeoNN_2km.txt", 'r')
#    for eachline in geo_nn_file:
#        data = raw_data = eachline.strip().split('\t')
#        l = int(data[0])
#        nearby_pois[l] = map(lambda k: int(k), data[1:])
#        print len(nearby_pois[l])
#    return nearby_pois


def evaluation(test_data, locations, visits, nearby_pois):
    import random
    t = time.time()
    print("Loading matrics...")
    UP = np.load(model_dir + "UP.npy")
    LS = np.load(model_dir + "LS.npy")
    LP = np.load(model_dir + "LP.npy")
    print("Done. Elapsed", time.time() - t)

    all_precision, all_recall = [], []
    for cnt_u, u in enumerate(test_data):
        print cnt_u
        precision, recall = 0.0, 0.0
        utotal = 0.0;
        hits = [0.0]*top_n;
        for cnt, (lc, actuals) in enumerate(test_data[u].items()):
            #print cnt
            utotal += len(actuals);
            poi_latent_dis = [(lj, ((1.0 + util.dist(locations[lc], locations[lj])) ** dis_coef) *
                              (alpha * np.linalg.norm(UP[u] - LP[lj]) ** 2 + (1 - alpha) * np.linalg.norm(LS[lc] - LS[lj])))
                              for lj in nearby_pois[lc] if (u, lj) not in visits]

            recommendations = [dd[0] for dd in heapq.nsmallest(top_n, poi_latent_dis, lambda x:x[1])]
            #recommendations = [nearby_pois[lc][dd] for dd in sorted(poi_latent_dis)[:top_n]]
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

def main():
    t = time.time()
    visits = read_visits()
    test_data = read_test_data()
    locations = get_locations()
    nearby_pois = get_nearby_pois()
    print("Data Loaded... Elapsed", time.time() - t)
    #print sum(len(val) for (key,val) in nearby_pois.iteritems())
    evaluation(test_data, locations, visits, nearby_pois)


if __name__ == '__main__':
    import sys;
    dis_coef = float(sys.argv[2])   # distance coefficient
    dataset = sys.argv[1]
    data_dir = "../data/"
    result_dir = "./result/"
    model_dir = "./model_"+dataset+("_wgeo/" if dis_coef>1e-10 else "_wogeo/")
    util = Util()
    main()