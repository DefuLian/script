import numpy as np
import os.path
import json
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
    loc_sequence = select_users(threshold=50)
    return
    def select_locations(threshold):
        lines = (line.strip().split('\t') for line in open(filename))
        uid_locid = [(locid, uid) for (uid, time, lat, lon, locid) in lines if uid in users]
        cnt = Counter(locid for locid, uid in uid_locid)
        locations = set(locid for locid, count in cnt.items() if count >= threshold)
        rare = set(locid for locid, count in cnt.items() if count < threshold)
        return locations, rare

    locations, rare = select_locations(threshold=10)

    def get_loc_db():
        lines = (line.strip().split('\t') for line in open(filename))
        locdb = [(locid, lat, lon) for (uid, time, lat, lon, locid) in lines if uid in users]
        cnt = Counter(locid for (locid, _, _ ) in locdb)
        locations = set(locid for locid, count in cnt.items() if count>=20)
        locdb = set(locdb)
        loc2latlon = {}
        for (key, group) in groupby(sorted(locdb, key=lambda x: x[0]), key=lambda x: x[0]):
            lat_avg, lon_avg = np.mean([[float(lat),float(lon)] for (_, lat, lon) in group], axis=0)
            loc2latlon[key] = (lat_avg, lon_avg)
        id2loc = list(loc2latlon.keys())
        id2point = [loc2latlon[k] for k in id2loc]
        loc2id = dict((locid, id) for (id, locid) in enumerate(id2loc))
        return (id2loc, loc2id, id2point)
    id2loc, loc2id, id2point = get_loc_db()
    print(id2loc)

if __name__ == "__main__":
    processing('E:/data/gowalla/Gowalla_totalCheckins.txt')