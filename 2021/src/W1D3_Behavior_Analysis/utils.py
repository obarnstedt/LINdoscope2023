# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:41:31 2021

@author: kluxem
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

def behavioral_graph(trans_mat, norm_usage, title_idx):
    
    plt.figure(figsize=(20,10))
    G = nx.from_numpy_matrix(trans_mat,create_using=nx.MultiDiGraph())
    pos = nx.drawing.layout.fruchterman_reingold_layout(G)     
    # pos = nx.drawing.layout.circular_layout(G)
    # pos = nx.drawing.layout.spectral_layout(G)
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    
    nx.draw_networkx(G, pos = pos, with_labels=True, node_size=norm_usage*8000, edgelist=edges, edge_color=weights, 
                      width=2.0, edge_cmap=plt.cm.Reds)
    
    plt.title("Behavioral Graph")


def P_transition_matrix(labels, n_cluster, threshold = 0.0):
    temp_matrix = np.zeros((n_cluster,n_cluster), dtype=np.float64)
    adjacency_matrix = np.zeros((n_cluster,n_cluster), dtype=np.float64)
    cntMat = np.zeros((n_cluster))
    steps = len(labels)
    
    for i in range(n_cluster):
        for k in range(steps-1):
            idx = labels[k]
            if idx == i:
                idx2 = labels[k+1]
                if idx == idx2:
                    continue
                else:
                    cntMat[idx2] = cntMat[idx2] +1
        temp_matrix[i] = cntMat
        cntMat = np.zeros((n_cluster))
    
    for k in range(steps-1):
        idx = labels[k]
        idx2 = labels[k+1]
        
        if idx == idx2:
            continue
        adjacency_matrix[idx,idx2] = 1
        adjacency_matrix[idx2,idx] = 1
        
    transition_matrix = get_transition_matrix(temp_matrix, threshold, eps=1e-10)
    
    return adjacency_matrix, transition_matrix, temp_matrix


def get_transition_matrix(adjacency_matrix, threshold, eps=1e-10):
    row_sum=adjacency_matrix.sum(axis=1)
    transition_matrix = adjacency_matrix/(row_sum[:,np.newaxis] + eps)
    transition_matrix[transition_matrix <= (threshold + eps)] = 0
    if np.any(np.isnan(transition_matrix)):
            transition_matrix=np.nan_to_num(transition_matrix)
    return transition_matrix


def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def find_zero_labels(motif_usage, n_cluster):
    cons = consecutive(motif_usage[0])
    if len(cons) != 1:
        print("Go")
        usage_list = list(motif_usage[1])
        for i in range(len(cons)-1):
            a = cons[i+1][0]
            b = cons[i][-1]
            d = (a-b)-1
            for j in range(1,d+1):
                index = cons[i][-1]+j
                usage_list.insert(index,0)
        if len(usage_list) < n_cluster:
            usage_list.insert(n_cluster,0)
            
    elif cons[0][-1] != n_cluster:
        # diff = n_cluster - cons[0][-1]
        usage_list = list(motif_usage[1])
        usage_list.insert(n_cluster-1,0)
        
    usage = np.array(usage_list)
    return usage


def load_labels(cfg, files, num_animals, n_cluster, model_name, param):
    if num_animals > 1:
        
        group1_labels = []
        group2_labels = []
        group_files = []
         
        for idx, file in enumerate(files):
            
            path_to_file=os.path.join(cfg['project_path'],'results',+str(file),model_name,param+'-'+str(n_cluster),str(n_cluster)+'_km_label_'+file+'.npy')

            if idx < (int(num_animals / 2)):
                print("Group 1: "+file)
                lbl = np.load(path_to_file)
                group1_labels.append(lbl)
                group_files.append(lbl)
            else:
                print("Group 2: "+file)
                lbl = np.load(path_to_file)
                group2_labels.append(lbl)
                group_files.append(lbl)
        
        group1 = np.concatenate(group1_labels)
        group2 = np.concatenate(group2_labels)
        
        return group1, group2, group_files
    
    else:

        file = files[0]
        
        path_to_file=os.path.join(cfg['project_path'],'results',str(file),model_name,param+'-'+str(n_cluster),str(n_cluster)+'_km_label_'+file+'.npy')

        lbl = np.load(path_to_file)
        
        return lbl, 0

