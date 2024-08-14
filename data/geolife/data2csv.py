import pandas as pd
import numpy as np
import argparse

def get_gps(gps_file):
    X = []
    Y = []
    with open(gps_file) as f:
        gpss = f.readlines()
        for gps in gpss:
            x, y = float(gps.split()[0]), float(gps.split()[1])
            X.append(x)
            Y.append(y)
    return X, Y

def read_data_from_file(fp):
    """
    read a bunch of trajectory data from txt file
    :param fp:
    :return:
    """
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[int(t) for t in tmp]]
    return np.asarray(dat, dtype='int64')


def data_conversion(data, seq_len):
    """Converts input array to pandas dataframe"""
    lon = []
    lat = []
    tid=[]
    for traj, j in zip(data, range(1, len(data)+1)):
        for i in range(seq_len - 1):
            lat.append(X[traj[i]])
            lon.append(Y[traj[i]])
            tid.append(j)

    converted_data = pd.DataFrame([], columns=['tid', 'lat', 'lon'])
    converted_data['tid'] = tid
    converted_data['lat'] = lat
    converted_data['lon'] = lon
    return converted_data
    
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="real.data")
    parser.add_argument("--save_path", type=str, default="train_latlon.csv")
    parser.add_argument("--tid_col", type=str, default="tid")
    args = parser.parse_args()
    dataset = 'geolife'
    seq_len = 48
    X, Y = get_gps(f'data/{dataset}/gps')
    data = read_data_from_file(f'data/{dataset}/{args.load_path}')
    
    converted_data = data_conversion(data, seq_len)
    converted_data.to_csv(f'data/{dataset}/{args.save_path}', index=False)