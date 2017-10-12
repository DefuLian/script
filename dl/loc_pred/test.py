
def getdb(filename):
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

if __name__ == "__main__":
    locdb = getdb('/home/dlian/data/location_prediction/gowalla/Gowalla_totalCheckins.txt')