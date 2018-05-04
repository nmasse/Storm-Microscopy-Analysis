import numpy as np
import csv
import time
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import dbscan
import matplotlib.pyplot as plt

class Cluster():

    def __init__(self, filename):

        self.filename = filename
        self.load_coords()
        self.N = self.coords.shape[0] # number of coorinates
        self.K = 250 # number of nearest neighbors to use in chigirev_dim_reduction calculation

    def return_coords(self):
        return self.coords


    def db_scan(self):

        max_dist = 50
        min_samples = 3
        min_samples_per_cluster = 5
        max_samples_per_cluster = 10000

        min_diameter = 25
        max_diameter = 1000

        metric = 'euclidean'
        algo = 'kd_tree'
        _, labels = dbscan(self.coords, max_dist, min_samples, metric, algorithm = algo)

        for i in np.unique(labels):
            #continue

            ind = np.where(labels == i)[0]
            if len(ind) < min_samples_per_cluster or len(ind) > max_samples_per_cluster:
                labels[ind] = -1
                continue

            pair_dist = np.zeros((len(ind), len(ind)), dtype = np.float32)
            for j in range(3):
                pair_dist += (np.tile(np.reshape(self.coords[ind,j],(1,len(ind))), (len(ind), 1)) - \
                    np.tile(np.reshape(self.coords[ind,j],(len(ind), 1)), (1, len(ind))))**2

            if np.max(pair_dist) < min_diameter**2 or np.max(pair_dist) > max_diameter**2:
                labels[ind] = -1


        return labels

    def load_coords(self):

        s1 = []
        with open(self.filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                s1.append(row)
            s1 = np.stack(s1,axis=0)
        k = int(np.where(s1[0,:] == 'x [nm]')[0])
        self.coords = np.float32(s1[1:,k:k+3])

    def chigirev_dim_reduction(self):

        epsilon = 1e-16
        # initialize the cluster points on top of the coordinates
        y = np.array(self.coords)
        # uniform probability
        p_y = np.ones((self.N), dtype = np.float32)/self.N
        # intialize search tree

        alpha = 2*(250**2)

        for i in range(20):

            D = 0
            I = 0

            nbrs = NearestNeighbors(n_neighbors=self.K, algorithm='kd_tree').fit(y)

            dist, ind = nbrs.kneighbors(self.coords)
            dist = dist**2
            negexpdist = np.exp(-dist/alpha)
            p_new_y = np.zeros_like(p_y)
            new_y  = np.zeros_like(y)

            # cycle through coords
            p_y_x_collection = []
            for j in range(self.N):

                # equation 14
                p_y_x = p_y[ind[j,:]]*negexpdist[j,:] + epsilon
                p_y_x = p_y_x/np.sum(p_y_x)
                p_y_x_collection.append(p_y_x)

                D += np.sum(dist[j,:]*p_y_x)/self.N

                # equation 12
                for k in range(3):
                    new_y[ind[j,:], k] += p_y_x*self.coords[j, k]/self.N

                p_new_y[ind[j,:]] += p_y_x

            #p_new_y += epsilon
            p_y = p_new_y/np.sum(p_new_y)
            for k in range(3):
                new_y[:,k] /= p_y
            y = np.array(new_y)

            # calculate I
            for j in range(self.N):
                I += np.sum(p_y_x_collection[j]*np.log(p_y_x_collection[j]/p_y[ind[j,:]])/self.N)

            print('Iteration ', i, ' D ', D, ' I ', I, ' Loss ', D + alpha*I)

        return y, p_y, p_y_x_collection
