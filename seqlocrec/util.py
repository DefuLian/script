import math
import time
import datetime


class Util(object):
    def __init__(self):
        pass

    def dist(self, loc1, loc2):
        lat1, long1 = loc1[0], loc1[1]
        lat2, long2 = loc2[0], loc2[1]
        if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
            return 0.0
        degrees_to_radians = math.pi/180.0
        phi1 = (90.0 - lat1)*degrees_to_radians
        phi2 = (90.0 - lat2)*degrees_to_radians
        theta1 = long1*degrees_to_radians
        theta2 = long2*degrees_to_radians
        cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
               math.cos(phi1)*math.cos(phi2))
        arc = math.acos( cos )
        earth_radius = 6371
        return arc * earth_radius

    def date2time(self, date, format):
        date = datetime.datetime.strptime(date, format)
        timestamp = time.mktime(date.timetuple())
        return timestamp

    def sigmoid(self, x):
        return 1.0 / (1 + math.exp(-x))
	
    def recallk(self, act, rec):
        return 1.0 * len(act.intersection(rec))/len(act)
		
    def precisionk(self, act, rec):
        return 1.0 * len(act.intersection(rec))/len(rec)
		