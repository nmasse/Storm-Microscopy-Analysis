
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

        # dbscan parameters
        self.dbscan_max_dist = 50
        self.dbscan_min_samples = 3
        self.dbscan_min_samples_per_cluster = 25
        self.dbscan_max_samples_per_cluster = 25000
        self.dbscan_min_diameter = 25
        self.dbscan_max_diameter = 1000

        # interaction parameters
        self.bin_interact_dist_threshold = 30
        self.bin_interact_contact_points = 1

        self.num_shuffle_repeats = 50


    def run_analysis(self, marker = 'Amph1'):

        for m in [1,2,5,10]:

            self.bin_interact_contact_points = m

            save_fn = marker + '_results_dist30_max_diam1000_min_samples25_noise250_flat_z_' + str(m) + 'pts.pkl'
            results = {'inter_p': [], 'shuffle_inter_p': [], 'bin_file': [], 'syn_file': []}

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
                sd = 1e-6 +np.std(p_shuffle)
                z = (p-u)/sd
                print('z-score ', z)

                results['inter_p'].append(p)
                results['shuffle_inter_p'].append(p_shuffle)
                results['bin_file'].append(bin_files[n])
                results['syn_file'].append(syn_files[n])
                pickle.dump(results, open(save_fn, 'wb'))


    def load_and_cluster_data(self, filename):

        coords = self.load_coords(filename)
        coords, coord_labels, cluster_center_mass, cluster_labels, coord_index = self.db_scan(coords)

        cluster_data = {'coords': coords, 'coord_labels': coord_labels, 'coord_index': coord_index, \
            'cluster_center_mass': cluster_center_mass, 'cluster_labels':cluster_labels}
        #cluster_data = {'cluster_center_mass': np.float32(cluster_center_mass), 'cluster_labels': np.int32(cluster_labels)}

        return cluster_data

    def shuffle_coords(self, data):

        new_data = {}
        for k, v in data.items():
            new_data[k] = np.array(v)

        for i in range(len(data['cluster_labels'])):
            ind_coords = data['coord_index'][i]
            noise = np.random.uniform(-250, 250, size = [1, 3])
            new_data['coords'][ind_coords, :] += noise
            new_data['cluster_center_mass'][i, :] += np.reshape(noise, (3))

        return new_data

    def plot_supplemental_figure(self):

        labels = ['GluA1','PSD95','CaMKII', 'Syn','Syp','Amph1','Bassoon']
        order = [5,4,3,0,2,1,6]
        c = 0.1

        min_samples = [10, 25, 50, 10, 25, 50]
        dist_threshold = [10, 10, 10, 20, 20, 20]

        for i in range(len(min_samples)):
            ym = 0.
            self.dbscan_min_samples_per_cluster = min_samples[i]
            self.bin_interact_contact_points = dist_threshold[i]
            NN_dist, subdirs, intearact_prob = self.search_dirs()

            f = plt.figure(figsize=(6,4))
            ax = f.add_subplot(2,3,i+1)

            for j, k in enumerate(order):
                u = np.linspace(j+1-c, j+1+c, len(intearact_prob[k]))
                if i==3 or i==4 or i==5:
                    col = 'r'
                else:
                    col = 'b'

                plt.plot(u, intearact_prob[k],'.', color = col,markersize=8)
                plt.plot([j+1-0.45, j+1+0.45],[np.median(intearact_prob[k]),np.median(intearact_prob[k])],'k')
                ym = np.maximum(ym, np.max(np.array(intearact_prob[k])))

            ax.set_xticks([1,2,3,4,5,6,7])
            ax.set_xticklabels(labels)
            ax.set_ylabel('Proportion')
            ax.set_ylim([0, 1.05*ym])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.savefig('sup_fig.pdf', format='pdf')
        plt.show()


    def plot_results_from_files(self):

        labels = ['GluA1', 'PSD', 'Syn', 'SYP', 'Bassoon','Amph1','Vgat']

        pre_synaptic_scores = []
        post_synaptic_scores = []
        inhibitory_scores = []

        c = 0.1 # for plotting
        f = plt.figure(figsize=(10,4))
        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
        ax = f.add_subplot(1,1,1)
        ax0 = plt.subplot(gs[0])

        post_scores = []
        pre_scores = []
        inh_scores = []

        for j, label in enumerate(labels):
            f#n = label + '_results_dist30_max_diam1000_min_samples10_noise250_' + str(5) + 'pts.pkl'
            fn = label + '_results_dist50_max_diam1000_min_samples25_noise250_' + str(5) + 'pts.pkl'
            if not os.path.isfile(fn):
                continue
            x = pickle.load(open(fn,'rb'))
            z_score = []
            for i in range(len(x['inter_p'])):
                u = np.mean(x['shuffle_inter_p'][i])
                sd = 1e-6 + np.std(x['shuffle_inter_p'][i])
                z = (x['inter_p'][i] - u)/sd
                r = (x['inter_p'][i] - u)/(x['inter_p'][i] + u)
                d = x['inter_p'][i] - u
                pct = np.mean(x['inter_p'][i]>x['shuffle_inter_p'][i])

                z_score.append(np.maximum(-0.,z))
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

        post_synaptic_scores.append(post_scores)
        pre_synaptic_scores.append(pre_scores)
        inhibitory_scores.append(inh_scores)

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
        #ax1.set_title('Pre vs. Post: P = ', p1)
        print('t-test p = ', p1)

        t1,p1 = scipy.stats.ranksums(post_scores, pre_scores)
        print('ranksum p = ', p1)

        plt.show()


    def plot_main_results(self, labels = ['GluA1','PSD95','CaMKII', 'Syn','Syp','Amph1','Bassoon']):

        NN_dist, subdirs, intearact_prob, center_of_mass_dist = self.search_dirs()

        order = []
        for l in labels:
            for i, sd in enumerate(subdirs):
                if l in sd:
                    order.append(i)

        print(order)

        th = str(self.bin_interact_dist_threshold)
        titles = ['Cluster centers <' + th + 'nm', 'Median pairwise <' + th + 'nm', '10th lowest pairwise <' + th + 'nm', \
            '10 pts <' + th + 'nm','Number of cluster pairs']

        pre_synaptic_scores = []
        post_synaptic_scores = []
        inhibitory_scores = []

        subdir_names = []
        scores = [[] for _ in range(5)]


        for m in range(5):

            c = 0.1 # for plotting
            f = plt.figure(figsize=(10,4))
            gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[2.5, 1])
            ax = f.add_subplot(1,1,1)

            post_scores = []
            pre_scores = []
            inh_scores = []


            ax0 = plt.subplot(gs[0])
            for j, i in enumerate(order):
                u = np.linspace(j+1-c, j+1+c, len(intearact_prob[i][m]))
                if 'PSD' in subdirs[i] or 'GluA1' in subdirs[i] or 'CaMKII' in subdirs[i]:
                    col = 'r'
                    post_scores.append(intearact_prob[i][m])
                elif 'Vgat' in subdirs[i] or 'Amph' in subdirs[i]:
                    col = 'm'
                    inh_scores.append(intearact_prob[i][m])
                else:
                    col = 'b'
                    pre_scores.append(intearact_prob[i][m])
                if m == 0:
                    subdir_names.append(subdirs[i])
                scores[m].append(intearact_prob[i][m])

                ax0.plot(u, intearact_prob[i][m], '.', color = col, markersize=8)
                ax0.plot([j+1-0.45, j+1+0.45],[np.median(intearact_prob[i][m]),np.median(intearact_prob[i][m])],'k')

            ax0.set_xticks([1,2,3,4,5,6,7])
            ax0.set_xticklabels(labels)
            ax0.set_ylabel('Proportion')
            #ax0.set_ylim([0,0.07])
            ax0.spines['top'].set_visible(False)
            ax0.spines['right'].set_visible(False)

            ax0.set_title(titles[m])


            ax1 = plt.subplot(gs[1])
            post_scores = list(itertools.chain(*post_scores))
            pre_scores = list(itertools.chain(*pre_scores))
            inh_scores = list(itertools.chain(*inh_scores))

            post_synaptic_scores.append(post_scores)
            pre_synaptic_scores.append(pre_scores)
            inhibitory_scores.append(inh_scores)

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
            #ax1.set_title('Pre vs. Post: P = ', p1)
            print('t-test p = ', p1)

            t1,p1 = scipy.stats.ranksums(post_scores, pre_scores)
            print('ranksum p = ', p1)


            plt.savefig('fig' + str(m) + '.pdf', format='pdf')
            plt.show()

        d = self.bin_interact_dist_threshold
        results = {'post_synaptic_scores': post_synaptic_scores, 'pre_synaptic_scores': pre_synaptic_scores,'NN_dist':NN_dist,'center_of_mass_dist':center_of_mass_dist, \
            'inhibitory_scores': inhibitory_scores,'intearact_prob': intearact_prob, 'subdirs': subdirs, 'intearact_prob': intearact_prob}

        pickle.dump(results, open('results' + str(d) + '_X.pkl', 'wb'))

        results = {'subdir_names': subdir_names, 'scores': scores}
        pickle.dump(results, open('results_simple' + str(d) + '_X.pkl', 'wb'))



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
            coords[ind,-1] = np.mean(coords[ind,-1])

            if len(ind) < self.dbscan_min_samples_per_cluster or len(ind) > self.dbscan_max_samples_per_cluster:
                coord_labels[ind] = -1
                continue

            if len(ind) > 20000:
                coord_labels[ind[::2]] = -1
                ind = ind[1::2]

            pair_dist = np.zeros((len(ind), len(ind)), dtype = np.float32)
            for j in range(3):
            #for j in range(2):
                pair_dist += (np.tile(np.reshape(coords[ind,j],(1,len(ind))), (len(ind), 1)) - \
                    np.tile(np.reshape(coords[ind,j],(len(ind), 1)), (1, len(ind))))**2

            if np.max(pair_dist) < self.dbscan_min_diameter**2 or np.max(pair_dist) > self.dbscan_max_diameter**2:
                coord_labels[ind] = -1
                continue
            """
            if len(ind) > 2000:
                # for computational efficiency
                coord_labels[ind[::2]] = -1
            """


            cluster_center_mass.append(np.mean(coords[ind,:], axis = 0))
            cluster_labels.append(i)

        #coord_labels = np.stack(coord_labels)
        cluster_center_mass = np.stack(cluster_center_mass)
        cluster_labels = np.stack(cluster_labels)

        ind_coords = np.where(coord_labels>=0)[0]
        ind_clusters = np.where(cluster_labels>=0)[0]

        coords = coords[ind_coords, :]
        coord_labels = coord_labels[ind_coords]
        cluster_center_mass = cluster_center_mass[ind_clusters, :]
        cluster_labels = cluster_labels[ind_clusters]

        coord_index = []

        for i, label in enumerate(cluster_labels):
            ind = np.where(coord_labels == label)[0]
            coord_index.append(ind)


        return coords, coord_labels, cluster_center_mass, cluster_labels, coord_index

    def calculate_BIN_interaction_cluster_center(self, coords, labels, BIN, iteration, cluster_center_mass, cluster_labels):

        bins = np.arange(0,20000, 10)
        dist_hist = np.zeros((len(bins)-1))
        dist_hist_shuffled = np.zeros((len(bins)-1))

        for i in np.unique(iteration):

            # find the two files associated with specifci iteration
            ind = np.where(iteration == i)[0]
            if not len(ind)==2:
                error('Issue ', ind)

            # which file is the BIN file
            # find which clusters belong to each
            if BIN[ind[0]]:
                bin_ind = ind[0]
                syn_ind = ind[1]
            else:
                bin_ind = ind[1]
                syn_ind = ind[0]
            print('number of bin clusters ', i, len(cluster_labels[bin_ind]))

            bin_cent = np.stack(cluster_center_mass[bin_ind])
            syn_cent = np.stack(cluster_center_mass[syn_ind])
            min_vals = np.min(np.vstack((bin_cent, syn_cent)), axis = 0)
            bin_cent -= min_vals
            syn_cent -= min_vals
            max_vals = np.max(np.vstack((bin_cent, syn_cent)), axis = 0)
            print('max vals', max_vals)
            d = self.calculate_multiple_distance(bin_cent, syn_cent)
            print('NUMBER UNDER 30 ', np.sum(d<30))
            dist_hist += np.histogram(d, bins)[0]

            d_shuffled = []
            for k in range(5000):
                syn_cent_shuffled = np.zeros_like(syn_cent)
                for j in range(3):
                    syn_cent_shuffled[:, j] = max_vals[j]*np.random.rand(syn_cent.shape[0])
                d1 = self.calculate_multiple_distance(bin_cent, syn_cent_shuffled)
                dist_hist_shuffled += np.histogram(d1, bins)[0]

        return dist_hist, dist_hist_shuffled

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

        #d = self.calculate_multiple_distance(x_data['cluster_center_mass'], y_data['cluster_center_mass'])
        #prob_interact = np.sum(d < self.bin_interact_dist_threshold**2, axis = 1)
        #print('INTERACT PROB ', np.mean(prob_interact))

        #print('LEN ', len(x_data['cluster_labels']), len(y_data['cluster_labels']))
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
                    proximity[-1] = 1
                    break
                    y_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(y_data['coords'][ind_coords_y, :])
                    current_dist, _ = y_nbrs.kneighbors(x_data['coords'][ind_coords_x, :])
                    if np.sum(current_dist < self.bin_interact_dist_threshold) >= self.bin_interact_contact_points:
                        # positive interaction
                        proximity[-1] = 1
                        break

        #print('LEN ', len(proximity), len(x_data['cluster_labels']), len(y_data['cluster_labels']))
        #print('MEAN PROX ',np.mean(proximity))
        return np.mean(proximity)

    def calculate_BIN_interaction(self, coords, labels, BIN, iteration, cluster_center_mass, cluster_labels):

        dist = []
        num_synaptic_coords = []
        num_bin = []
        num_synaptic_marker = []
        interactions = []
        interactions_u = []
        center_of_mass_dist = []

        """
        Using 5 different metric to quantify the interaction between protein pairs
        1 - Center of mass between two clusters is less than self.bin_interact_dist_threshold
        2 - Median of pairwise diatances between two clusters is less than self.bin_interact_dist_threshold
        3 - 10th lowest  pairwise diatances between two clusters is less than self.bin_interact_dist_threshold
        4 - Both clusters have at least self.bin_interact_contact_points points that are within self.bin_interact_dist_threshold of other cluster
        5 - (Control) Total number of cluster pairs
        """

        interaction_prob = [[] for _ in range(5)]

        for i in np.unique(iteration):

            # find the two files associated with specifci iteration
            ind = np.where(iteration == i)[0]
            if not len(ind)==2:
                error('Issue ', ind)

            total_possible = 0
            n_bin = 0
            hits = [[] for _ in range(5)]
            histogram = [[] for _ in range(4)] # distribution of cluster pair "distances", based on metric above

            # which file is the BIN file
            # find which clusters belong to each
            if BIN[ind[0]]:
                bin_ind = ind[0]
                syn_ind = ind[1]
            else:
                bin_ind = ind[1]
                syn_ind = ind[0]
            print('number of bin clusters ', i, len(cluster_labels[bin_ind]))

            # loop through pairs of clusters
            for j in range(len(cluster_labels[bin_ind])):
                syn_marker_nearby = False
                for m in range(5):
                    # hits is used to indictate whether a pair is interacting
                    hits[m].append(0)

                hits[4].append(1)


                for k in range(len(cluster_labels[syn_ind])):

                    d = np.sum((cluster_center_mass[bin_ind][j] - cluster_center_mass[syn_ind][k])**2)
                    center_of_mass_dist.append(np.sqrt(d))
                    if d > 3000**2: # we assume that clusters with centers 3000 nm apart are not interacting
                        for m in range(4):
                            histogram[m].append(1000)
                        continue

                    # hit if center of clusters are less than 20 nm apart
                    if d < self.bin_interact_dist_threshold**2:
                        hits[0][-1] = 1
                        print('HIT UNDER 30 ', np.sum(hits[0]))

                    bin_cluster = np.where(labels[bin_ind] == cluster_labels[bin_ind][j])[0]
                    syn_cluster = np.where(labels[syn_ind] == cluster_labels[syn_ind][k])[0]

                    # find nearest points in other cluster
                    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[bin_ind][bin_cluster])
                    current_dist, _ = nbrs.kneighbors(coords[syn_ind][syn_cluster])

                    if np.median(current_dist) < self.bin_interact_dist_threshold:
                        # hit if median of pairwise distances are less than 20 nm apart
                        hits[1][-1] = 1
                        pass


                    # assorted pairwise distance distributions
                    #print(coords[syn_ind][syn_cluster].shape, coords[bin_ind][bin_cluster].shape, current_dist.shape)
                    #histogram.append(np.min(current_dist))
                    sorted_dist = np.sort(current_dist[:,0])
                    #print(sorted_dist.shape)
                    histogram[0].append(sorted_dist[0])
                    histogram[1].append(sorted_dist[4])
                    histogram[2].append(sorted_dist[9])
                    histogram[3].append(sorted_dist[24])


                    if sorted_dist[9] < self.bin_interact_dist_threshold:
                        # hit if 10th lowest pairwise distances is less than 20 nm
                        hits[2][-1] = 1
                        pass
                    #histogram.append(np.sqrt(d))


                    num_synaptic_coords.append(coords[syn_ind].shape[0])
                    if np.sum(current_dist < self.bin_interact_dist_threshold) >= self.bin_interact_contact_points:
                        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[syn_ind][syn_cluster])
                        current_dist, _ = nbrs.kneighbors(coords[bin_ind][bin_cluster])
                        if np.sum(current_dist < self.bin_interact_dist_threshold) >= self.bin_interact_contact_points:
                            hits[3][-1] = 1
                            #break
                            #num_bin.append(len(bin_cluster))
                            #num_synaptic_marker.append(len(syn_cluster))
                            #interactions.append(np.sum(current_dist < self.bin_interact_dist_threshold))
                            #interactions_u.append(np.mean(current_dist < self.bin_interact_dist_threshold))


            for m in range(4):
                interaction_prob[m].append(np.mean(hits[m]))
            interaction_prob[4].append(np.sum(hits[4]))
            dist.append(current_dist)

        return list(itertools.chain(*dist)), np.mean(num_synaptic_coords), interaction_prob, num_bin,\
            np.stack(histogram[0]), np.stack(histogram[1]), np.stack(histogram[2]), np.stack(histogram[3]), center_of_mass_dist

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
