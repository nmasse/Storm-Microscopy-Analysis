import numpy as np
import csv
import time
import pickle
import os
import itertools
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class NN():

    def __init__(self, data_dir):

        self.data_dir = data_dir

    def search_dirs(self):

        # We assume that all CSV files are contained in subdirectories
        dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        print(dirs)

        NN_dist = []
        subdirs = []

        for d in dirs:
            if 'MAP2' in d:
                # don't know how to work with these files
                continue

            subdir = os.path.join(self.data_dir, d)
            files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f)) and 'csv' in f]

            print('Current directory ', d)
            coords = []
            BIN = []
            iter = []
            # load and clean all the coordinates from all files within subdirector
            for f in files:
                BIN.append(True if ('BIN' in f or 'Bin' in f) else False)
                k = f.find('.')
                iter.append(int(f[k-1]))
                coords.append(self.load_coords(os.path.join(subdir, f)))

            dist, mean_num_synaptic_coords = self.calculate_distance_from_BIN(coords, BIN, iter)
            NN_dist.append(dist)
            subdirs.append(d)
            print(d, ' mean NN dist = ', np.mean(dist), ' num synaptic markers ', mean_num_synaptic_coords)

        return NN_dist, subdirs

    def calculate_distance_from_BIN(self, coords, BIN, iter):

        dist = []
        num_synaptic_coords = []
        for i in np.unique(iter):
            ind = np.where(iter == i)[0]
            if not len(ind)==2:
                error('Issue ', ind)

            # for each BIN coordinate, we will find the nearest distance to the pre/post synaptic marker
            if BIN[ind[0]]:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[ind[0]])
                current_dist, _ = nbrs.kneighbors(coords[ind[1]])
                num_synaptic_coords.append(coords[ind[1]].shape[0])
            else:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[ind[1]])
                current_dist, _ = nbrs.kneighbors(coords[ind[0]])
                num_synaptic_coords.append(coords[ind[0]].shape[0])

            dist.append(current_dist)

        return list(itertools.chain(*dist)), np.mean(num_synaptic_coords)

    def load_coords(self, filename):

        s1 = []
        with open(filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                s1.append(row)
            s1 = np.stack(s1,axis=0)
        k = int(np.where(s1[0,:] == 'x [nm]')[0])
        coords = np.float32(s1[1:,k:k+3])
        coords = self.eliminate_isolated_coords(coords)
        return coords

    def eliminate_isolated_coords(self, coords):

        th = 10 # coordinates must be within this distance of another coordinate
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(coords)
        dist, _ = nbrs.kneighbors(coords)

        ind = np.where(dist[:,1] < th)[0]
        print('Percentage of coords kept = ', len(ind)/coords.shape[0])
        return coords[ind, :]
