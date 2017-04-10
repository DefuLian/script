import math
EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180

def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def map_size(levelOfDetail):
    return 256 << levelOfDetail


def ground_resolution(latitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    return math.cos(latitude * math.pi / 180) * 2 * math.pi * EarthRadius / map_size(levelOfDetail)

def map_scale(latitude, levelOfDetail, screenDpi):
    return ground_resolution(latitude, levelOfDetail) * screenDpi / 0.0254


def latlon2pxy(latitude, longitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY

def pxy2latlon(pixelX, pixelY, levelOfDetail):
    mapSize = float(map_size(levelOfDetail))
    x = (clip(pixelX, 0, mapSize - 1) / mapSize) - 0.5
    y = 0.5 - (clip(pixelY, 0, mapSize - 1) / mapSize)
    latitude = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
    longitude = 360 * x
    return latitude, longitude

def pxy2txy(pixelX, pixelY):
    tileX = pixelX / 256
    tileY = pixelY / 256
    return tileX, tileY
def txy2pxy(tileX, tileY):
    pixelX = tileX * 256
    pixelY = tileY * 256
    return pixelX, pixelY

def latlon2txy(latitude, longitude, levelOfDetail):
    pixelX, pixelY = latlon2pxy(latitude, longitude, levelOfDetail)
    return pxy2txy(pixelX, pixelY)



#def get_retangle(lat, lon, distance, level):
#    deltaLatPixel = math.floor(distance * 1000 / ground_resolution(lat, level))
#    xpixel, ypixel = latlon2pxy(Lat, Lon, level, );
#    int highTileX, highTileY, lowTileX, lowTileY;
#    TileSystem.PixelXYToTileXY(xpixel + deltaLatPixel, ypixel + deltaLatPixel, out highTileX, out highTileY);
#    TileSystem.PixelXYToTileXY(xpixel - deltaLatPixel, ypixel - deltaLatPixel, out lowTileX, out lowTileY);
#    return new int[] { lowTileX, lowTileY, highTileX, highTileY };



def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)




def quadkey2txy(quadKey):
    tileX = tileY = 0
    levelOfDetail = len(quadKey)
    #for (int i = levelOfDetail; i > 0; i--):
    for i in range(levelOfDetail, 0, -1):
        mask = 1 << (i - 1)
        key = quadKey[levelOfDetail - i]
        if key == '0':
            continue
        elif key == '1':
            tileX |= mask
        elif key == '2':
            tileY |= mask
        elif key == '3':
            tileX |= mask
            tileY |= mask
        else:
            raise Exception("Invalid QuadKey digit sequence.")
    return tileX, tileY, levelOfDetail

def latlon2quadkey(lat,lon,level):
    pixelX, pixelY = latlon2pxy(lat, lon, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY,level)



def tile_size(lat, level):
    return 256 * ground_resolution(lat, level)

def QuadKeyToRectangle(key):
    tilex, tiley, level = quadkey2txy(key)
    ulx,uly = txy2pxy(tilex, tiley)
    highlat, lowlon = pxy2latlon(ulx, uly, level)
    ulx, uly = txy2pxy(tilex + 1, tiley + 1)
    lowlat, highlon = pxy2latlon(ulx, uly, level, )
    return [lowlat, lowlon, highlat, highlon]


def quadkey2center(key):
    tilex, tiley, level = quadkey2txy(key)
    ulx, uly = txy2pxy(tilex, tiley)
    highlat, lowlon= pxy2latlon(ulx, uly, level)
    ulx, uly = txy2pxy(tilex + 1, tiley + 1)
    lowlat, highlon = pxy2latlon(ulx, uly, level)
    return (lowlat + highlat) / 2, (lowlon + highlon) / 2



#def get_near_tiles(quad, l):
#    tx,ty,level = quadkey2txy(quad)
#    items = (txy2quadkey(x, y, level) for x in range(tx - l, tx + l + 1) for y in range(ty - l, ty + l + 1))
#    return [item for item in items if item != quad]

def get_near_tiles(lat, lon, level, distance, ins=None):

    deltaLatPixel = int(distance * 1000 / ground_resolution(lat, level))
    xpixel, ypixel = latlon2pxy(lat, lon, level)
    highTileX, highTileY = pxy2txy(xpixel + deltaLatPixel, ypixel + deltaLatPixel)
    lowTileX, lowTileY = pxy2txy(xpixel - deltaLatPixel, ypixel - deltaLatPixel)
    ret = []
    for i in range(lowTileX, highTileX+1):
        for j in range(lowTileY, highTileY+1):
            key = txy2quadkey(i, j, level)
            lat_cen, lon_cen = quadkey2center(key)
            d = compute_distance(lat, lon, lat_cen, lon_cen)
            if d < distance and (ins is None or (key in ins)):
                ret.append((key, d))

    return ret

def compute_distance(lat1, lon1, lat2, lon2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lon1) - rad(lon2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s = s * EarthRadius / 1000
    s = round(s * 10000) / 10000.0
    return s

def rad(d):
    return d * math.pi / 180.0


