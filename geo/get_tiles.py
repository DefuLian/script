import tile_system
count = 0
print tile_system.ground_resolution(36.677124920118, 16) * 256



def get_tiles_items(dataset, level):
    tile_dic = dict()
    with open('/home/dlian/data/Data/%s/item_grids_%d.txt' % (dataset, level), 'w') as out_file:
        for line in open('/home/dlian/data/Data/%s/items.txt' % dataset):
            id, lat, lon, oid = line.strip().split()
            tiles = tile_system.get_near_tiles(float(lat), float(lon), level, 1)
            for tile, dis in tiles:
                if tile not in tile_dic:
                    tile_dic[tile] = len(tile_dic)
                tile_id = tile_dic[tile]
                out_file.write("%s\t%d\t%.4f\t%s\n" % (id, tile_id, dis, tile))



if __name__ == "__main__":
    level = 16
    get_tiles_items('Beijing', level)
    get_tiles_items('Shanghai', level)
    get_tiles_items('Gowalla', level)
    level = 17
    get_tiles_items('Beijing', level)
    get_tiles_items('Shanghai', level)
    get_tiles_items('Gowalla', level)