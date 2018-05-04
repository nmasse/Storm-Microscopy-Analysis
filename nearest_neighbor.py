import numpy as np
import csv
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import dbscan
from itertools import product

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

            if not 'Bassoon' in d and not 'PSD' in d:
                pass

            subdir = os.path.join(self.data_dir, d)
            files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f)) and 'csv' in f]

            print('Current directory ', d)
            labels = []
            coords = []
            BIN = []
            iteration = []
            cluster_center_mass = []
            cluster_labels = []
            interaction_prob = []
            # load and clean all the coordinates from all files within subdirector
            for f in files:
                BIN.append(True if ('BIN' in f or 'Bin' in f) else False)
                k = f.find('.')
                iteration.append(int(f[k-1]))
                x = self.load_coords(os.path.join(subdir, f))
                lbs, cluster_cm, cluster_lbs = self.db_scan(x)
                ind = np.where(lbs > -1)[0] # only use coords belonging to a cluster
                coords.append(x[ind, :])
                labels.append(lbs[ind])
                cluster_center_mass.append(cluster_cm)
                cluster_labels.append(cluster_lbs)
                #coords.append(x)


            dist, mean_num_synaptic_coords, intearact_prob = self.calculate_BIN_interaction(coords, labels, \
                BIN, iteration, cluster_center_mass, cluster_labels)
            #dist, mean_num_synaptic_coords = self.calculate_NN_distribution(coords, BIN, iteration)
            NN_dist.append(dist)
            subdirs.append(d)
            interaction_prob.append(intearact_prob)
            print(d, ' mean NN dist = ', np.mean(dist), ' num synaptic markers ', mean_num_synaptic_coords)
            print( ' pct ', intearact_prob)

        return NN_dist, subdirs, intearact_prob

    def db_scan(self, coords):

        max_dist = 50
        min_samples = 3
        min_samples_per_cluster = 50
        max_samples_per_cluster = 10000

        min_diameter = 25
        max_diameter = 1000

        metric = 'euclidean'
        algo = 'kd_tree'
        _, labels = dbscan(coords, max_dist, min_samples, metric, algorithm = algo)
        cluster_center_mass = []
        cluster_labels = []

        for i in np.unique(labels):

            ind = np.where(labels == i)[0]
            if len(ind) < min_samples_per_cluster or len(ind) > max_samples_per_cluster:
                labels[ind] = -1
                continue

            pair_dist = np.zeros((len(ind), len(ind)), dtype = np.float32)
            for j in range(3):
                pair_dist += (np.tile(np.reshape(coords[ind,j],(1,len(ind))), (len(ind), 1)) - \
                    np.tile(np.reshape(coords[ind,j],(len(ind), 1)), (1, len(ind))))**2

            if np.max(pair_dist) < min_diameter**2 or np.max(pair_dist) > max_diameter**2:
                labels[ind] = -1
                continue

            cluster_center_mass.append(np.mean(coords[ind,:], axis = 0))
            cluster_labels.append(i)


        return labels, cluster_center_mass, cluster_labels

    def find_clusters(self, coords):

        max_dist = 50
        min_samples = 3
        min_diameter = 20
        max_diameter = 1000
        min_samples_per_cluster = 50
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

    def calculate_NN_distribution(self, coords, BIN, iteration):

        dist = []
        num_synaptic_coords = []

        for i in np.unique(iteration):
            ind = np.where(iteration == i)[0]
            if not len(ind)==2:
                error('Issue ', ind)

            if BIN[ind[0]]:
                bin_ind = ind[0]
                syn_ind = ind[1]
            else:
                bin_ind = ind[1]
                syn_ind = ind[0]

            nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(coords[bin_ind])
            current_dist, _ = nbrs.kneighbors(coords[syn_ind])
            print(len(current_dist))
            print(current_dist.shape)
            1/0
            num_synaptic_coords.append(coords[syn_ind].shape[0])
            dist.append(current_dist)

        return list(itertools.chain(*dist)), np.mean(num_synaptic_coords)

    def calculate_BIN_interaction(self, coords, labels, BIN, iteration, cluster_center_mass, cluster_labels):

        dist = []
        num_synaptic_coords = []
        num_bin = []
        num_synaptic_marker = []
        interactions = []
        interactions_u = []
        dist_threshold = 500
        interaction_threshold = 10
        contact_points = 5

        interaction_prob = []

        for i in np.unique(iteration):
            ind = np.where(iteration == i)[0]
            if not len(ind)==2:
                error('Issue ', ind)

            hits = 0
            total_possible = 0
            n_bin = 0

            # for each BIN coordinate, we will find the nearest distance to the pre/post synaptic marker
            #plt.figure(figsize=(15,15))

            if BIN[ind[0]]:
                bin_ind = ind[0]
                syn_ind = ind[1]
            else:
                bin_ind = ind[1]
                syn_ind = ind[0]


            for j in range(len(cluster_labels[bin_ind])):

                for k in range(len(cluster_labels[syn_ind])):
                    d = np.sqrt(np.sum((cluster_center_mass[bin_ind][j] - cluster_center_mass[syn_ind][k])**2))
                    if d < dist_threshold:
                        total_possible += 1
                        bin_cluster = np.where(labels[bin_ind] == cluster_labels[bin_ind][j])[0]
                        syn_cluster = np.where(labels[syn_ind] == cluster_labels[syn_ind][k])[0]
                        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[bin_ind][bin_cluster])
                        current_dist, _ = nbrs.kneighbors(coords[syn_ind][syn_cluster])
                        num_synaptic_coords.append(coords[syn_ind].shape[0])
                        if np.sum(current_dist < interaction_threshold) >= contact_points:
                            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[syn_ind][syn_cluster])
                            current_dist, _ = nbrs.kneighbors(coords[bin_ind][bin_cluster])
                            if np.sum(current_dist < interaction_threshold) >= contact_points:
                                hits+=1
                                num_bin.append(len(bin_cluster))
                                num_synaptic_marker.append(len(syn_cluster))
                                interactions.append(np.sum(current_dist < interaction_threshold))
                                interactions_u.append(np.mean(current_dist < interaction_threshold))


            interaction_prob.append(hits/total_possible)
            dist.append(current_dist)
            #print('hits ', hits, 'misses ', misses, 'number BIN', n_bin, 'Hit cluster size ', np.mean(num_bin), np.mean(num_synaptic_marker))
            #print('sum and mean interactions per hit', np.mean(interactions), np.mean(interactions_u))
        return list(itertools.chain(*dist)), np.mean(num_synaptic_coords), interaction_prob

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
