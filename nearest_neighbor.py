
import numpy as np # for math
import pickle # for loading
import csv # for spreadsheetd
import os # operating systems, file loading
import itertools # for combining lists
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

    def __init__(self, data_dir):

        self.data_dir = data_dir

        # dbscan parameters
        self.dbscan_max_dist = 50
        self.dbscan_min_samples = 3
        self.dbscan_min_samples_per_cluster = 50
        self.dbscan_max_samples_per_cluster = 10000
        self.dbscan_min_diameter = 25
        self.dbscan_max_diameter = 1000

        # interaction parameters
        self.bin_interact_dist_threshold = 50
        self.bin_interact_contact_points = 10


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



    def plot_main_results(self, labels = ['GluA1','PSD95','CaMKII', 'Syn','Syp','Amph1','Bassoon']):

        NN_dist, subdirs, intearact_prob = self.search_dirs()

        order = []
        for l in labels:
            for i, sd in enumerate(subdirs):
                if l in sd:
                    order.append(i)

        print(order)

        th = str(self.bin_interact_dist_threshold)
        titles = ['Cluster centers <' + th + 'nm', 'Median pairwise <' + th + 'nm', '10th lowest pairwise <' + th + '0nm', \
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

        results = {'post_synaptic_scores': post_synaptic_scores, 'pre_synaptic_scores': pre_synaptic_scores, \
            'inhibitory_scores': inhibitory_scores,'intearact_prob': intearact_prob, 'subdirs': subdirs}

        pickle.dump(results, open('results50.pkl', 'wb'))

        results = {'subdir_names': subdir_names, 'scores': scores}
        pickle.dump(results, open('results_simple50.pkl', 'wb'))



    def search_dirs(self):

        # We assume that all CSV files are contained in subdirectories
        dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        print('Working directories...', dirs)

        NN_dist = []
        subdirs = []
        interaction_prob = []

        for d in dirs:
            #if 'MAP2' in d or (not 'Bassoon' in d and not 'SYP' in d and not 'Vgat' in d and not 'Syn' in d):
            #if 'MAP2' in d or (not 'Bassoon_BPB' in d ):
            if 'MAP2' in d: # specific for this experiment!!!
                # don't know how to work with these files
                continue

            subdir = os.path.join(self.data_dir, d)
            files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f)) and 'csv' in f] # find all csv files

            print('Current directory ', d)
            labels = []
            coords = []
            BIN = []
            iteration = []
            cluster_center_mass = []
            cluster_labels = []
            num_BIN_clusters = []

            # load and clean all the coordinates from all files within subdirector
            for f in files:
                BIN.append(True if ('BIN' in f or 'Bin' in f) else False)

                # iteration number, assuming iteration number is right before '.' before file extension
                k = f.find('.')
                if f[k-2].isdigit():
                    iteration.append(int(f[k-2:k]))
                else:
                    iteration.append(int(f[k-1]))

                # load coordinates
                x = self.load_coords(os.path.join(subdir, f))

                print('Current file ', f, ' iter ', iteration[-1], ' Bin ', BIN[-1])

                # clusters data dbscan
                lbs, cluster_cm, cluster_lbs = self.db_scan(x)


                ind = np.where(lbs > -1)[0] # only use coords belonging to a cluster
                print('Numer of pts, cluster pts, and pct ', len(lbs), len(ind), len(ind)/len(lbs))
                coords.append(x[ind, :])
                labels.append(lbs[ind])
                cluster_center_mass.append(cluster_cm)
                cluster_labels.append(cluster_lbs)
                #coords.append(x)


            dist, mean_num_synaptic_coords, intearact_prob , n_bin_clusters, h0, h1, h2, h3 = \
                self.calculate_BIN_interaction(coords, labels, BIN, iteration, cluster_center_mass, cluster_labels)



            NN_dist.append(dist)
            subdirs.append(d)
            interaction_prob.append(intearact_prob)
            print(d, ' mean NN dist = ', np.mean(dist), ' num synaptic markers ', mean_num_synaptic_coords)
            print( ' pct ', intearact_prob, 'num_BIN_clusters', n_bin_clusters)

        return NN_dist, subdirs, interaction_prob

    def db_scan(self, coords):

        metric = 'euclidean'
        algo = 'kd_tree'
        # label, one for each coordinate, specifies what cluster it belongs to
        _, labels = dbscan(coords, self.dbscan_max_dist, self.dbscan_min_samples, metric, algorithm = algo)
        cluster_center_mass = []
        cluster_labels = []





        # given all the clusters, will now discard clusters that are too big/small, not enough points
        for i in np.unique(labels):

            ind = np.where(labels == i)[0]
            if len(ind) < self.dbscan_min_samples_per_cluster or len(ind) > self.dbscan_max_samples_per_cluster:
                labels[ind] = -1
                continue

            pair_dist = np.zeros((len(ind), len(ind)), dtype = np.float32)
            for j in range(3):
            #for j in range(2):
                pair_dist += (np.tile(np.reshape(coords[ind,j],(1,len(ind))), (len(ind), 1)) - \
                    np.tile(np.reshape(coords[ind,j],(len(ind), 1)), (1, len(ind))))**2

            if np.max(pair_dist) < self.dbscan_min_diameter**2 or np.max(pair_dist) > self.dbscan_max_diameter**2:
                labels[ind] = -1
                continue

            cluster_center_mass.append(np.mean(coords[ind,:], axis = 0))
            cluster_labels.append(i)


        return labels, cluster_center_mass, cluster_labels



    def calculate_BIN_interaction(self, coords, labels, BIN, iteration, cluster_center_mass, cluster_labels):

        dist = []
        num_synaptic_coords = []
        num_bin = []
        num_synaptic_marker = []
        interactions = []
        interactions_u = []

        """
        Using 5 different metric to quantify the interaction between protein pairs
        1 - Center of mass between two clusters is less than self.bin_interact_dist_threshold
        2 - Median of pairwise diatances between two clusters is less than self.bin_interact_dist_threshold
        3 - 10th lowest  pairwise diatances between two clusters is less than self.bin_interact_dist_threshold
        4 - Both clusters have at least self.bin_interact_contact_points points that are within self.bin_interact_dist_threshold of other cluster
        5 - (Control) Total number of cluster pairs
        """
        hits = [[] for _ in range(5)]
        histogram = [[] for _ in range(4)]
        interaction_prob = [[] for _ in range(5)]


        print('len iteration', len(iteration))

        for i in np.unique(iteration):
            ind = np.where(iteration == i)[0]
            if not len(ind)==2:
                error('Issue ', ind)

            total_possible = 0
            n_bin = 0

            for r in [0]:

                if BIN[ind[r]]:
                    bin_ind = ind[0]
                    syn_ind = ind[1]
                else:
                    bin_ind = ind[1]
                    syn_ind = ind[0]
                print('number of bin clusters ', i, len(cluster_labels[bin_ind]))


                for j in range(len(cluster_labels[bin_ind])):
                    syn_marker_nearby = False
                    for k in range(len(cluster_labels[syn_ind])):

                        for m in range(5):
                            hits[m].append(0)

                        hits[4].append(1)

                        d = np.sum((cluster_center_mass[bin_ind][j] - cluster_center_mass[syn_ind][k])**2)
                        if d > 3000**2: # we assume that clusters with centers 3000 nm apart are not interacting
                            for m in range(4):
                                histogram[m].append(1000)
                            continue

                        # hit if center of clusters are less than 20 nm apart
                        if d < self.bin_interact_dist_threshold**2:
                            hits[0][-1] = 1


                        bin_cluster = np.where(labels[bin_ind] == cluster_labels[bin_ind][j])[0]
                        syn_cluster = np.where(labels[syn_ind] == cluster_labels[syn_ind][k])[0]


                        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(coords[bin_ind][bin_cluster])
                        current_dist, _ = nbrs.kneighbors(coords[syn_ind][syn_cluster])

                        if np.median(current_dist) < self.bin_interact_dist_threshold:
                            # hit if median of pairwise distances are less than 20 nm apart
                            hits[1][-1] = 1
                            pass

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
            np.stack(histogram[0]), np.stack(histogram[1]), np.stack(histogram[2]), np.stack(histogram[3])

    def load_coords(self, filename):

        s1 = []
        with open(filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                s1.append(row)
            s1 = np.stack(s1,axis=0)
        # finds what column is the x-position, assumes that y and z-positions are next to it
        k = int(np.where(s1[0,:] == 'x [nm]')[0])
        coords = np.float32(s1[1:,k:k+3])

        return coords
