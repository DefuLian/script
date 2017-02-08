import time
import random
import numpy as np

from util import Util


data_dir = "D:/PRME/data/"
result_dir = "D:/PRME/result/"
model_dir = "D:/PRME/model/"

dataset = "Gowalla"
file_suffix = "User120Poi5.txt"

time_format = "%Y/%m/%d %H:%M:%S"

# Beijing
# user_num = 6703
# poi_num = 57122
# K = 60            # latent dimensionality

# tau = 3600 * 6    # time difference threshold
# gamma = 0.0005      # learning rate
# lamda = 0.03      # regularization
# alpha = 0.2       # linear combination
# dis_coef = 0.25   # distance coefficient

# max_iters = 30   # max iterations

user_num = 13003
poi_num = 255691
K = 60            # latent dimensionality

tau = 3600 * 6    # time difference threshold
gamma = 0.05      # learning rate
lamda = 0.03      # regularization
alpha = 0.2       # linear combination
dis_coef = 0.25   # distance coefficient

max_iters = 100   # max iterations


def read_training_data():
    train_data = []
    visits = set()
    locations = {}

    train_data_file = open(data_dir + dataset + "_TrainData_" + file_suffix, 'r')
    for eachline in train_data_file:
        raw_data = eachline.strip().split('\t')
        u, lc, li = int(raw_data[0]) - 1, int(raw_data[1]) - 1, int(raw_data[5]) - 1

        current_time = util.date2time(raw_data[4], time_format)
        next_time = util.date2time(raw_data[8], time_format)
        time_irrelevance = (next_time - current_time) > tau

        train_data.append([u, lc, li, time_irrelevance])
        visits.add((u, lc, li))
        locations[lc] = [float(raw_data[2]), float(raw_data[3])]
        locations[li] = [float(raw_data[6]), float(raw_data[7])]
    return train_data, visits


def get_locations():
    locations = {}
    train_data_file = open(data_dir + dataset + "_TrainData_" + file_suffix, 'r')
    for eachline in train_data_file:
        raw_data = eachline.strip().split('\t')
        lc, li = int(raw_data[1]) - 1, int(raw_data[5]) - 1
        locations[lc] = [float(raw_data[2]), float(raw_data[3])]
        locations[li] = [float(raw_data[6]), float(raw_data[7])]
    test_data_file = open(data_dir + dataset + "_TestData_" + file_suffix, 'r')
    for eachline in test_data_file:
        raw_data = eachline.strip().split('\t')
        lc, li = int(raw_data[1]) - 1, int(raw_data[5]) - 1
        locations[lc] = [float(raw_data[2]), float(raw_data[3])]
        locations[li] = [float(raw_data[6]), float(raw_data[7])]
    return locations


def learning(train_data, visits, locations, cont=True):
    # load matrix from ./model/ to continue the training...

    if not cont:
        UP = np.random.normal(0.0, 0.01, (user_num, K))
        LS = np.random.normal(0.0, 0.01, (poi_num, K))
        LP = np.random.normal(0.0, 0.01, (poi_num, K))
    else:
        UP = np.load(model_dir + "UP.npy")
        LS = np.load(model_dir + "LS.npy")
        LP = np.load(model_dir + "LP.npy")

    poi_ids = range(poi_num)
    try:
        for iteration in range(max_iters):
            log_likelihood = 0.0
            random.shuffle(train_data)

            t = time.time()
            for each_data in train_data:
                try:
                    u, lc, li, time_irrelevance = each_data
                    lj = li
                    while (u, lc, lj) in visits:
                        lj = random.sample(poi_ids, 1)[0]

                    if time_irrelevance:
                        Di = np.linalg.norm(UP[u] - LP[li]) ** 2
                        Dj = np.linalg.norm(UP[u] - LP[lj]) ** 2
                        z = Dj - Di
                        log_likelihood += np.log(util.sigmoid(z))

                        UP[u] += gamma * ((1 - util.sigmoid(z)) * 2 * (LP[li] - LP[lj]) - 2 * lamda * UP[u])
                        LP[li] += gamma * ((1 - util.sigmoid(z)) * 2 * (UP[u] - LP[li]) - 2 * lamda * LP[li])
                        LP[lj] += gamma * (- (1 - util.sigmoid(z)) * 2 * (UP[u] - LP[lj]) - 2 * lamda * LP[lj])

                    else:
                        wci = (1.0 + util.dist(locations[lc], locations[li])) ** dis_coef
                        wcj = (1.0 + util.dist(locations[lc], locations[lj])) ** dis_coef
                        Di = wci * (alpha * np.linalg.norm(UP[u] - LP[li]) ** 2 + (1 - alpha) * np.linalg.norm(LS[lc] - LS[li]))
                        Dj = wcj * (alpha * np.linalg.norm(UP[u] - LP[lj]) ** 2 + (1 - alpha) * np.linalg.norm(LS[lc] - LS[lj]))
                        z = Dj - Di
                        log_likelihood += np.log(util.sigmoid(z))

                        UP[u] += gamma * ((1 - util.sigmoid(z)) *
                                          2 * alpha * ((wci - wcj) * UP[u] + (wci * LP[li] - wcj * LP[lj])) -
                                          2 * lamda * UP[u])
                        LP[li] += gamma * ((1 - util.sigmoid(z)) * 2 * alpha * wci * (UP[u] - LP[li]) - 2 * lamda * LP[li])
                        LP[lj] += gamma * (- (1 - util.sigmoid(z)) * 2 * alpha * wcj * (UP[u] - LP[lj]) - 2 * lamda * LP[lj])
                        LS[lc] += gamma * ((1 - util.sigmoid(z)) *
                                           2 * (1 - alpha) * ((wci - wcj) * LS[lc] + (wci * LS[li] - wcj * LS[lj])) -
                                           2 * lamda * LS[lc])
                        LS[li] += gamma * ((1 - util.sigmoid(z)) * 2 * (1 - alpha) * wci * (LS[lc] - LS[li]) - 2 * lamda * LS[li])
                        LS[lj] += gamma * (- (1 - util.sigmoid(z)) * 2 * (1 - alpha) * wcj * (LS[lc] - LS[lj]) - 2 * lamda * LS[lj])

                except OverflowError:
                    print("Calculation failed.")
            print("Iter: %d    likelihood: %f    elapsed: %fs" % (iteration, log_likelihood, time.time() - t))
    finally:
        np.save(model_dir + "UP", UP)
        np.save(model_dir + "LP", LP)
        np.save(model_dir + "LS", LS)
        print("Model saved...")


def main():
    t = time.time()
    train_data, visits = read_training_data()
    locations = get_locations()
    print("Data Loaded... Elapsed", time.time() - t)

    learning(train_data, visits, locations, cont=False)


if __name__ == '__main__':
    util = Util()
    main()