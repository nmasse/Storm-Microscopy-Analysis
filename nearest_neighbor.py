
import numpy as np # for math
import pickle # for loading
import csv # for spreadsheetd
import os # operating systems, file loading
import itertools # for combining lists
import time
from itertools import product # for combining lists
import matplotlib # plotting
import scipy.stats # for stats
import matplotlib.pyplot as plt # plotting

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"

from sklearn.neighbors import NearestNeighbors # for stats
from sklearn.cluster import dbscan # for stats, cluster algorithm


class NN():

    def __init__(self, data_dir = '/home/masse/Storm-Microscopy-Analysis/Data/'):

        self.data_dir = data_dir
        self.save_fn = '_results_dist50_max_diam1000_min_samples25_noise250_'

        # dbscan parameters
        self.dbscan_max_dist = 50
        self.dbscan_min_samples = 3
        self.dbscan_min_samples_per_cluster = 25
        self.dbscan_max_samples_per_cluster = 10000
        self.dbscan_min_diameter = 25
        self.dbscan_max_diameter = 1000

        # interaction parameters
        self.bin_interact_dist_threshold = 50
        self.bin_interact_contact_points = 1
        self.reciprocal_interaction = False

        self.num_shuffle_repeats = 100
        self.shuffle_noise = 250


    def run_analysis(self, marker = 'Amph1'):

        # sweeps different number of interact_contact_points
        for m in [1,2,5,10]:

            self.bin_interact_contact_points = m

            save_fn = marker + self.save_fn + str(m) + 'pts.pkl'
            results = {'inter_p': [], 'shuffle_inter_p': [], 'bin_file': [], 'syn_file': [], 'bin_clusters': [], 'syn_clusters': []}

            bin_files, syn_files = self.search_dirs()
            for n in range(len(bin_files)):

                if not marker in syn_files[n]:
                    continue

                bin_data = self.load_and_cluster_data(bin_files[n])
                syn_data = self.load_and_cluster_data(syn_files[n])

                p = self.determine_interaction(syn_data, bin_data, cluster_centers = False)
                print('Analyzing file ', bin_files[n])

                p_shuffle = np.zeros((self.num_shuffle_repeats))
                for i in range(self.num_shuffle_repeats):
                    syn_data_shuffled = self.shuffle_coords(syn_data)
                    p_shuffle[i] = self.determine_interaction(syn_data_shuffled, bin_data, cluster_centers = False)
                u = np.mean(p_shuffle)
                sd = 1e-6 + np.std(p_shuffle)
                z = (p-u)/sd
                print('z-score ', z)

                results['inter_p'].append(p)
                results['shuffle_inter_p'].append(p_shuffle)
                results['bin_clusters'].append(len(bin_data['cluster_labels']))
                results['syn_clusters'].append(len(syn_data['cluster_labels']))
                results['bin_file'].append(bin_files[n])
                results['syn_file'].append(syn_files[n])
                pickle.dump(results, open(save_fn, 'wb'))


    def load_and_cluster_data(self, filename):

        coords = self.load_coords(filename)
        coords, coord_labels, cluster_center_mass, cluster_labels, coord_index = self.db_scan(coords)

        cluster_data = {'coords': coords, 'coord_labels': coord_labels, 'coord_index': coord_index, \
            'cluster_center_mass': cluster_center_mass, 'cluster_labels':cluster_labels}

        return cluster_data

    def shuffle_coords(self, data):

        new_data = {}
        for k, v in data.items():
            new_data[k] = np.array(v)

        for i in range(len(data['cluster_labels'])):
            ind_coords = data['coord_index'][i]
            noise = np.random.uniform(-self.shuffle_noise, self.shuffle_noise, size = [1, 3])
            new_data['coords'][ind_coords, :] += noise
            new_data['cluster_center_mass'][i, :] += np.reshape(noise, (3))

        return new_data

    def plot_results_from_files(self):

        labels = ['GluA1', 'PSD', 'Syn', 'SYP', 'Bassoon','Amph1','Vgat']

        c = 0.1 # for plotting
        f = plt.figure(figsize=(10,4))
        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
        ax = f.add_subplot(1,1,1)
        ax0 = plt.subplot(gs[0])

        post_scores = []
        pre_scores = []
        inh_scores = []

        for j, label in enumerate(labels):
            fn = label + '_results_dist50_max_diam1000_min_samples25_noise250_' + str(5) + 'pts_v2.pkl'
            #fn = label + '_results_dist50_max_diam1000_min_samples25_noise250_flat_z_' + str(2) + 'pts.pkl'
            if not os.path.isfile(fn):
                continue
            x = pickle.load(open(fn,'rb'))
            z_score = []
            for i in range(len(x['inter_p'])):
                if x['bin_clusters'][i] < 5 or x['syn_clusters'][i] < 5:
                    continue
                u = np.mean(x['shuffle_inter_p'][i])
                sd = 1e-6 + np.std(x['shuffle_inter_p'][i])
                z = np.maximum(0., (x['inter_p'][i] - u)/sd)
                z_score.append(z)

            if 'PSD' in label or 'GluA1' in label or 'CaMKII' in label:
                col = 'r'
                post_scores.append(z_score)
            elif 'Vgat' in label or 'Amph' in label:
                col = 'm'
                inh_scores.append(z_score)
            else:
                col = 'b'
                pre_scores.append(z_score)
            u = np.linspace(j+1-c, j+1+c, len(z_score))
            #z_score = np.stack(z_score)

            ax0.plot(u, np.stack(z_score), '.', color = col, markersize=8)
            ax0.plot([j+1-0.45, j+1+0.45],[np.median(z_score),np.median(z_score)],'k')

        ax0.set_xticks([1,2,3,4,5,6,7])
        ax0.set_xticklabels(labels)
        ax0.set_ylabel('z-score')

        ax1 = plt.subplot(gs[1])
        post_scores = list(itertools.chain(*post_scores))
        pre_scores = list(itertools.chain(*pre_scores))
        inh_scores = list(itertools.chain(*inh_scores))

        c = 0.75
        u = np.linspace(1-c, 1+c, len(post_scores))
        ax1.plot(u, post_scores,'r.',markersize=8)
        ax1.plot([1-0.9, 1+0.9],[np.median(post_scores),np.median(post_scores)],'k')

        u = np.linspace(3-c, 3+c, len(pre_scores))
        ax1.plot(u, pre_scores,'b.',markersize=8)
        ax1.plot([3-0.9, 3+0.9],[np.median(pre_scores),np.median(pre_scores)],'k')

        u = np.linspace(5-c, 5+c, len(inh_scores))
        ax1.plot(u, inh_scores,'m.',markersize=8)
        ax1.plot([5-0.9, 5+0.9],[np.median(inh_scores),np.median(inh_scores)],'k')

        ax1.set_xticks([1,3,5])
        ax1.set_xticklabels(['Post-synaptic', 'Pre-synaptic', 'Inhibitory'])
        #ax1.set_ylim([0,0.07])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        t1,p1 = scipy.stats.ttest_ind(post_scores, pre_scores)
        print('t-test p = ', p1)

        t1,p1 = scipy.stats.ranksums(post_scores, pre_scores)
        print('ranksum p = ', p1)

        plt.show()



    def search_dirs(self):

        bin_files = []
        syn_files = []

        # We assume that all CSV files are contained in subdirectories
        dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        for d in dirs:
            subdir = os.path.join(self.data_dir, d)
            files = [os.path.join(subdir,f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f)) and 'csv' in f] # find all csv files

            file_numbers = []
            for f in files:
                k = f.find('.')
                if f[k-2].isdigit():
                    file_numbers.append(int(f[k-2:k]))
                else:
                    file_numbers.append(int(f[k-1]))

            file_numbers = np.stack(file_numbers)
            for n in np.unique(file_numbers):
                ind = np.where(file_numbers == n)[0]
                try:
                    assert(len(ind) == 2)
                except AssertionError:
                    raise Exception('There are not 2 files with the same run number')
                if 'BIN' in files[ind[0]]:
                    bin_files.append(files[ind[0]])
                    syn_files.append(files[ind[1]])
                else:
                    bin_files.append(files[ind[1]])
                    syn_files.append(files[ind[0]])

        return bin_files, syn_files



    def db_scan(self, coords):

        metric = 'euclidean'
        algo = 'kd_tree'
        # label, one for each coordinate, specifies what cluster it belongs to
        _, coord_labels = dbscan(coords, self.dbscan_max_dist, self.dbscan_min_samples, metric, algorithm = algo)
        cluster_center_mass = []
        cluster_labels = []

        # given all the clusters, will now discard clusters that are too big/small, not enough points
        for i in np.unique(coord_labels):

            ind = np.where(coord_labels == i)[0]

            if len(ind) < self.dbscan_min_samples_per_cluster or len(ind) > self.dbscan_max_samples_per_cluster:
                coord_labels[ind] = -1
                continue

            pair_dist = np.zeros((len(ind), len(ind)), dtype = np.float32)
            for j in range(3):
            #for j in range(2):
                pair_dist += (np.tile(np.reshape(coords[ind,j],(1,len(ind))), (len(ind), 1)) - \
                    np.tile(np.reshape(coords[ind,j],(len(ind), 1)), (1, len(ind))))**2

            if np.max(pair_dist) < self.dbscan_min_diameter**2 or np.max(pair_dist) > self.dbscan_max_diameter**2:
                coord_labels[ind] = -1
                continue

            cluster_center_mass.append(np.mean(coords[ind,:], axis = 0))
            cluster_labels.append(i)

        cluster_center_mass = np.stack(cluster_center_mass)
        cluster_labels = np.stack(cluster_labels)

        ind_coords = np.where(coord_labels>=0)[0]
        coords = coords[ind_coords, :]
        coord_labels = coord_labels[ind_coords]

        coord_index = []
        for label in cluster_labels:
            ind = np.where(coord_labels == label)[0]
            coord_index.append(ind)


        return coords, coord_labels, cluster_center_mass, cluster_labels, coord_index


    def calculate_multiple_distance(self, x, y):

        m = x.shape[0]
        n = y.shape[0]
        d = np.zeros((m, n), dtype = np.float32)
        for i in range(3):
            d += (np.tile(x[:, i:i+1],(1, n)) - np.tile(np.transpose(y[:,i:i+1]),(m,1)))**2

        return d


    def determine_interaction(self, x_data, y_data, cluster_centers = False):

        if cluster_centers:
            d = self.calculate_multiple_distance(x_data['cluster_center_mass'], y_data['cluster_center_mass'])
            prob_interact = np.sum(d < self.bin_interact_dist_threshold**2, axis = 1)
            return np.mean(prob_interact)

        proximity = []
        for i in range(len(x_data['cluster_labels'])):
            ind_coords_x = np.array(x_data['coord_index'][i])
            x_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_data['coords'][ind_coords_x, :])
            proximity.append(0)
            for j in range(len(y_data['cluster_labels'])):
                center_distance = np.sum((x_data['cluster_center_mass'][i,:] - y_data['cluster_center_mass'][j,:])**2)
                # if cluster centers are more than twice the allowed diameter apart, then skip
                if center_distance > (2*self.dbscan_max_diameter)**2:
                    continue

                ind_coords_y = np.array(y_data['coord_index'][j])
                current_dist, _ = x_nbrs.kneighbors(y_data['coords'][ind_coords_y, :])
                if np.sum(current_dist < self.bin_interact_dist_threshold) >= self.bin_interact_contact_points:
                    if self.reciprocal_interaction:
                        y_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(y_data['coords'][ind_coords_y, :])
                        current_dist, _ = y_nbrs.kneighbors(x_data['coords'][ind_coords_x, :])
                        if np.sum(current_dist < self.bin_interact_dist_threshold) >= self.bin_interact_contact_points:
                            # positive interaction
                            proximity[-1] = 1
                            break
                    else:
                        proximity[-1] = 1
                        break

        return np.mean(proximity)


    def load_coords(self, filename):

        z_correction = 0.875
        s = []
        with open(filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                s.append(row)
            s = np.stack(s,axis=0)
        # finds what column is the x-position, assumes that y and z-positions are next to it
        k = int(np.where(s[0,:] == 'x [nm]')[0])
        coords = np.float32(s[1:,k:k+3])
        coords[:, -1] *= z_correction

        return coords
