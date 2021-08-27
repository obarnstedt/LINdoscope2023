# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:14:03 2021

@author: kluxem
"""

# Create graph for WT and TG animals

import sys
import numpy as np
import networkx as nx
from scipy.linalg import eig

from scipy.stats import entropy 
from scipy.stats import wasserstein_distance as wd


from pathlib import Path
from matplotlib import pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

sys.path.append("C://Research//Lindoscope//")
from utils import behavioral_graph, load_labels, P_transition_matrix, find_zero_labels, consecutive

from vame.util.auxiliary import read_config

def markov_graph_analysis(config, param='hmm', threshold=0.00):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    
    files = []
    for file in cfg['video_sets']:
        files.append(file)
    
    num_animals = len(files)
        
    group1_label, _ = load_labels(cfg, files, num_animals, n_cluster, model_name, param)
    
    group_lables = [group1_label]
    
    # compute graph for each group
    group_ptrans = []
    file_usage = []
    for i, labels in enumerate(group_lables):
        _, trans_mat, _ = P_transition_matrix(labels, n_cluster=n_cluster, threshold=threshold)
        
        # Directed Markov Graph
        motif_usage = np.unique(labels, return_counts=True)
        
        if len(motif_usage[0]) < n_cluster:
            motif_usage = find_zero_labels(motif_usage, n_cluster)
        else: 
            motif_usage = motif_usage[1]
            
        norm_usage = motif_usage / np.sum(motif_usage)
        
        behavioral_graph(trans_mat, norm_usage, title_idx=i+1)
        
        group_ptrans.append(trans_mat)
        file_usage.append(motif_usage)
        
    plt.figure(figsize=(10,5))
    plt.imshow(group_ptrans[0], cmap='jet')
    plt.colorbar()
    plt.title("Transition probability matrix")
        
   
def community_ordered_transition_probabilities(config, community_order, param='hmm', threshold=0.0):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    
    files = []
    for file in cfg['video_sets']:
        files.append(file)
    
    num_animals = len(files)
        
    group1_label, _ = load_labels(cfg, files, num_animals, n_cluster, model_name, param)
    
    group_lables = [group1_label]
    
    # compute graph for each group
    group_ptrans = []
    for i, labels in enumerate(group_lables):
        _, trans_mat, _ = P_transition_matrix(labels, n_cluster=n_cluster, threshold=threshold)
                                
        group_ptrans.append(trans_mat)
    
    group_comm_trans = []
    for mat in group_ptrans:
        comm_trans = np.zeros_like(mat)
        
        for i in range(len(comm_trans)):
            for j in range(len(comm_trans)):
                comm_trans[i,j] = mat[community_order[i], community_order[j]]
                
        group_comm_trans.append(comm_trans)
    
    plt.figure(figsize=(10,5))
    plt.imshow(group_comm_trans[0], vmin=0, vmax=1, cmap='jet')
    plt.colorbar()
    plt.title("Community ordered transition probability matrix")
        
        
        
        
        
        
        
        

