# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:25:11 2021

@author: kluxem
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from sklearn.cluster import KMeans

sys.path.append("C://Research//Lindoscope//")
from utils import find_zero_labels

from pathlib import Path
from vame.util.auxiliary import read_config


def motif_dist_norm(cfg, files, model_name, n_cluster):    
    motif_list = []
    motif_list_norm = []
    for idx, file in enumerate(files):
        print(file)
        
        path_to_file=cfg['project_path']+'\\results\\'+str(file)+'\\'+model_name+'\\'+'hmm-'+str(n_cluster)+'\\'+str(n_cluster)+'_km_label_'+file+'.npy'

        label=np.load(path_to_file)
        motif_usage = np.unique(label, return_counts=True)
        if len(motif_usage[0]) < n_cluster:
            motif_usage = find_zero_labels(motif_usage, n_cluster)
        else: 
            motif_usage = motif_usage[1]
        motif_usage_colsum = motif_usage.sum(axis=0)
        motif_norm = motif_usage/motif_usage_colsum
        motif_list_norm.append(motif_norm)
        motif_list.append(motif_usage)
    
    return motif_list_norm, motif_list

def group_distribution(files, motif_norm_list, n_cluster, num_animals):
    group_dist = np.zeros((n_cluster,1))  
    
    for idx, file in enumerate(files):
        group_dist[:,0] += motif_norm_list[idx]
        
    return group_dist

def barplot(n_cluster, group_dist, ylabel, title):
    x = np.arange(n_cluster)  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(20,10))
    rects1 = ax.bar(x, group_dist[:,0], width, label="Animal")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()

def behavior_distribution(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']

    files = []
    for file in cfg['video_sets']:
        files.append(file)
            
    num_animals = len(files)
        
    # load and normalize motifs; save in a list
    motif_list_norm, motif_list = motif_dist_norm(cfg, files, model_name, n_cluster=n_cluster)
    
    group_dist = group_distribution(files, motif_list_norm, n_cluster, num_animals)
    
    # compute std for motifs
    barplot(n_cluster, group_dist, ylabel="% Normalized Usage",title="Norm Motif Distribution")
    
    