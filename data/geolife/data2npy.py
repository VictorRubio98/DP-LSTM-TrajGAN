"""Convert encoded csv files to one-hot-encoded npy files."""

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
    """Converts input array to one-hot-encoded Numpy array (locations are still in float)."""
    
    x = [[] for i in ['lat_lon', 'day', 'hour', 'category', 'mask', '']]
    for traj in data:
        trajectory=[]
        for i in range(seq_len - 1):
            lat = X[traj[i]]
            lon = Y[traj[i]]
            trajectory.append(np.array([lat, lon], dtype=np.float64))

        x[0].append(trajectory)
        x[1].append(np.eye(7))
        x[2].append(np.eye(24))
        x[3].append(np.eye(10))
        x[4].append(np.ones(shape=(seq_len-1, 1)))
    
    converted_data = np.array([np.array(f) for f in x])
    converted_data = converted_data[0:5]
    return converted_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="real.data")
    parser.add_argument("--save_path", type=str, default="train_encoded.npy")
    parser.add_argument("--tid_col", type=str, default="tid")
    args = parser.parse_args()
    dataset = 'geolife'
    seq_len = 48
    X, Y = get_gps(f'data/{dataset}/gps')
    data = read_data_from_file(f'data/{dataset}/{args.load_path}')
    
    converted_data = data_conversion(data, seq_len)
    np.save(f'data/{dataset}/{args.save_path}', converted_data)