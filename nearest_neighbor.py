import numpy as np
import csv
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import dbscan

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
            cluster_center_mass = []
            BIN = []
            iter = []
            # load and clean all the coordinates from all files within subdirector
            for f in files:
                BIN.append(True if ('BIN' in f or 'Bin' in f) else False)
                k = f.find('.')
                iter.append(int(f[k-1]))
                x = self.load_coords(os.path.join(subdir, f))
                _, cm = self.find_clusters(x)
                cm = np.stack(cm, axis = 0)
                print('cm shape ', cm.shape)
                cluster_center_mass.append(cm)

            dist, mean_num_synaptic_coords = self.calculate_distance_from_BIN(cluster_center_mass, BIN, iter)
            NN_dist.append(dist)
            subdirs.append(d)
            print(d, ' mean NN dist = ', np.mean(dist), ' num synaptic markers ', mean_num_synaptic_coords)

        return NN_dist, subdirs

    def find_clusters(self, coords):

        max_dist = 100
        min_samples = 3
        min_diameter = 10
        max_diameter = 750
        min_samples_per_cluster = 10
        max_samples_per_cluster = 25000
        metric = 'euclidean'
        algo = 'kd_tree'

        cluster_coords = []
        cluster_center_mass = []

        core_sample, labels = dbscan(coords, max_dist, min_samples, metric, algorithm = algo)
        for i in np.unique(labels):
            ind = np.where(labels == i)[0]
            if len(ind) < min_samples_per_cluster or len(ind) > max_samples_per_cluster:
                continue

            pair_dist = np.zeros((len(ind), len(ind)), dtype = np.float32)
            for j in range(3):
                pair_dist += (np.tile(np.reshape(coords[ind,j],(1,len(ind))), (len(ind), 1)) - \
                    np.tile(np.reshape(coords[ind,j],(len(ind), 1)), (1, len(ind))))**2

            if np.max(pair_dist) < min_diameter**2 or np.max(pair_dist) > max_diameter**2:
                continue

            cluster_coords.append(coords[ind,:])
            cluster_center_mass.append(np.mean(coords[ind,:], axis = 0))
        return cluster_coords, cluster_center_mass


    def calculate_distance_from_BIN(self, coords, BIN, iter):

        max_distance = 250
        dist = []
        num_synaptic_coords = []
        for i in np.unique(iter):
            ind = np.where(iter == i)[0]
            if not len(ind)==2:
                error('Issue ', ind)

            # for each BIN coordinate, we will find the nearest distance to the pre/post synaptic marker
            #plt.figure(figsize=(15,15))

            if BIN[ind[0]]:
                print('BIN coords shape ',coords[ind[0]].shape)
                print('NON BIN coords shape ',coords[ind[1]].shape)
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[ind[0]])
                current_dist, _ = nbrs.kneighbors(coords[ind[1]])
                num_synaptic_coords.append(coords[ind[1]].shape[0])
                ind1 = np.where(current_dist < max_distance)[0]
                #plt.plot(coords[ind[0]][ind1, 0], coords[ind[0]][ind1, 1], 'b.')
                #plt.plot(coords[ind[1]][:, 0], coords[ind[1]][:, 1], 'r.')
            else:
                print('BIN coords shape ',coords[ind[1]].shape)
                print('NON BIN coords shape ',coords[ind[0]].shape)
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[ind[1]])
                current_dist, _ = nbrs.kneighbors(coords[ind[0]])
                num_synaptic_coords.append(coords[ind[0]].shape[0])
                ind1 = np.where(current_dist < max_distance)[0]
                #plt.plot(coords[ind[1]][ind1, 0], coords[ind[1]][ind1, 1], 'b.')
                #plt.plot(coords[ind[0]][:, 0], coords[ind[0]][:, 1], 'r.')

            #plt.xlim([10000, 15000])
            #plt.ylim([7500, 12500])
            #plt.show()

            dist.append(current_dist[ind1])

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
        #coords = self.eliminate_isolated_coords(coords)
        return coords

    def eliminate_isolated_coords(self, coords):

        th = 10 # coordinates must be within this distance of another coordinate
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(coords)
        dist, _ = nbrs.kneighbors(coords)

        ind = np.where(dist[:,1] < th)[0]
        print('Percentage of coords kept = ', len(ind)/coords.shape[0])
        return coords[ind, :]
