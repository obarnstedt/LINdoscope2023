# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:01:04 2021

@author: kluxem
"""

# create communities for both and combined groups

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("C://Research//Lindoscope//")
from utils import load_labels, P_transition_matrix, find_zero_labels

from pathlib import Path
from vame.util.auxiliary import read_config
from tree_creation import graph_to_tree, draw_tree, traverse_tree_cutline


def create_tree_representation(config, param='hmm'):
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

    group_trees = []
    for i, lbl in enumerate(group_lables):
        _, trans_mat, _ = P_transition_matrix(lbl, n_cluster=n_cluster, threshold=0.0)        
        _, usage = np.unique(lbl, return_counts=True)
        T = graph_to_tree(usage, trans_mat, n_cluster, merge_sel=1) 
        group_trees.append(T)
    
    draw_tree(group_trees[0])
    
    
    
    
    
    
    
    
    
    
    
    