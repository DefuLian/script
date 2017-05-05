import tile_system
count = 0
print tile_system.ground_resolution(36.677124920118, 16) * 256



def get_tiles_items(dataset, level):
    tile_dic = dict()
    with open('/home/dlian/data/checkin/%s/item_grids6_%d.txt' % (dataset, level), 'w') as out_file:
        for line in open('/home/dlian/data/checkin/%s/items.txt' % dataset):
            id, lat, lon, oid = line.strip().split()
            tiles = tile_system.get_near_tiles(float(lat), float(lon), level, 1)
            for tile, dis in tiles:
                if tile not in tile_dic:
                    tile_dic[tile] = len(tile_dic)
                tile_id = tile_dic[tile]
                out_file.write("%s\t%d\t%.4f\n" % (id, tile_id, dis))
    with open('/home/dlian/data/checkin/%s/grids_%d.txt' % (dataset, level), 'w') as out_file:
        for (tile, id) in sorted(tile_dic.items(), key=lambda x:x[1]):
            tile_lat, tile_lon = tile_system.quadkey2center(tile)
            out_file.write('%d\t%.5f\t%.5f\t%s\n' % (id, tile_lat, tile_lon, tile))

def gen_sim_matrix(dataset):
    from scipy.sparse import csr_matrix,coo_matrix
    import math
    lines = [line.strip().split('\t') for line in open('/home/dlian/data/checkin/%s/item_grids_%d.txt' % (dataset, 16))]
    items = [line.strip().split('\t') for line in open('/home/dlian/data/checkin/%s/items.txt' % dataset)]
    items = dict((int(att[0]), (float(att[1]), float(att[2]))) for att in items)
    I = [int(line[0]) for line in lines]
    J = [int(line[1]) for line in lines]
    data = [float(line[2]) for line in lines]
    mat = csr_matrix((data, (I,J)))
    sim = mat * mat.transpose()
    sim_coo = sim.tocoo()
    print sim_coo.nnz
    with open('/home/dlian/data/checkin/%s/item_sim.txt' % dataset, 'w') as outfile:
        for i in range(0,len(sim_coo.col)):
            col,row = sim_coo.col[i], sim_coo.row[i]
            if col == row:
                continue
            dis = tile_system.compute_distance(items[row][0],items[row][1], items[col][0], items[col][1])
            outfile.write('%d\t%d\t%.4f\n' % (row, col, math.exp(-dis*dis/0.5791)))



def tmp():

    #lines = [l.strip().split('\t') for l in open('/home/dlian/data/checkin/Beijing/items.txt')]
    #grid2loc = dict((l[0], (float(l[1]), float(l[2]))) for l in lines);
    lines = [l.strip().split('\t') for l in open('/home/dlian/data/checkin/Beijing/grids_17.txt')]
    grid2loc = dict((l[0], (float(l[1]), float(l[2]))) for l in lines)
    lines = [l.strip().split('\t') for l in open('/home/dlian/data/checkin/Beijing/test/x1.txt')]
    lines = [(grid2loc[l[0]], float(l[1])) for l in lines]
    with open('/home/dlian/data/checkin/Beijing/test/result_trans.html', 'w') as outfile:
        with open('/home/dlian/data/checkin/Beijing/test/head') as head:
            outfile.write(head.read())

        for ((lat, lon), weight) in lines:
            outfile.write('    {location: new google.maps.LatLng(%f,%f), weight:%f},\n' % (lat, lon, weight))

        with open('/home/dlian/data/checkin/Beijing/test/tail') as tail:
            outfile.write(tail.read())



if __name__ == "__main__":
    tmp()
    #gen_sim_matrix("Beijing")
    #gen_sim_matrix("Gowalla")
    #gen_sim_matrix("Shanghai")
    #level = 16
    #get_tiles_items('Beijing', level)
    #get_tiles_items('Shanghai', level)
    #get_tiles_items('Gowalla', level)
    #level = 17
    #get_tiles_items('Beijing', level)
    #get_tiles_items('Shanghai', level)
    #get_tiles_items('Gowalla', level)