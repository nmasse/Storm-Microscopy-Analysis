import numpy as np
import csv
import time
import pickle
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class Cluster():

    def __init__(self, data_file):

        data_dir = './Data/'
        self.filename = data_dir + data_file
        self.load_coords()
        self.N = self.coords.shape[0] # number of coorinates
        self.K = 1000 # number of nearest neighbors to use in chigirev_dim_reduction calculation

    def return_coords(self):
        return self.coords

    def load_coords(self):

        s1 = []
        with open(self.filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                s1.append(row)
            s1 = np.stack(s1,axis=0)
        self.coords = np.float32(s1[1:,1:4])
        print('Coordinate shape ', self.coords.shape)

    def chigirev_dim_reduction(self):

        epsilon = 1e-16
        # initialize the cluster points on top of the coordinates
        y = np.array(self.coords)
        # uniform probability
        p_y = np.ones((self.N), dtype = np.float32)/self.N
        # intialize search tree

        alpha = 20000

        for i in range(50):

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

        return y
