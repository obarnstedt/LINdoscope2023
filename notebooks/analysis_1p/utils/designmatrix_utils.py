#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:39:25 2023

@author: compraka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
import os
import glob


def create_design_matrix(annotated_events_path, save_df_path, 
                       spikes, n_beh_events=['event_1'], 
                       add_forward_jitter=True, forward=5,
                       add_back_jitter=True, backward=5,
                       fps=30, to_use='Regrouped Behaviors'):
    
    frames = len(spikes[0])
    manual_annotation = pd.read_csv(annotated_events_path)
    
    start_time = manual_annotation[['Start (s)']].to_numpy() * fps
    
    event_matrix = np.zeros((frames, len(n_beh_events)))
        
    for event in range(0, len(n_beh_events)):
        event_idx = np.where(manual_annotation[[to_use]] == n_beh_events[event])[0]
        start_idx = start_time[event_idx]
        if add_forward_jitter == True:
            for start in start_idx:
                idx = np.arange(int(start), int(start + forward))
                event_matrix[idx.astype(int), event] = 1
        if add_back_jitter == True:
            for start in start_idx:
                idx = np.arange(int(start - backward), int(start))
                event_matrix[idx.astype(int), event] = 1
        if add_forward_jitter == True and add_back_jitter == True:
            for start in start_idx:
                idx = np.arange(int(start - backward), int(start + forward))
                event_matrix[idx.astype(int), event] = 1
        else:
            for start in start_idx:
                event_matrix[int(start), event] = 1
    
    behavior_dicts =  {} 
    
    for event_mat_row in range(0, len(event_matrix[0])):
        behavior_dicts[n_beh_events[event_mat_row]] = event_matrix[:, event_mat_row]
    
    behavior_matrix = pd.DataFrame.from_dict(behavior_dicts, orient = "columns")
    
    return behavior_matrix


def qr_decomp_matrix(design_matrix, save_file_path, name):
    # The resulting plot ranges from 0 to 1 for each regressor, with 1 being fully orthogonal to 
    # all preceeding regressors in the matrix and 0 being fully redundant. QR decomposition is more numerically stable
    # In Musall et. al 2019 they normalize the design matrix to get values between 0 and 1
    normed = design_matrix / np.sqrt(np.sum(design_matrix**2, axis=0))
    normed_zero = normed.fillna(0)
    q, r = np.linalg.qr(normed_zero)
    
    save_csv_path = save_file_path + '/qr_' + name[:52] + '_regressor_orthogonality.csv'
    qr_data = pd.DataFrame()
    qr_data["feature"] = design_matrix.columns
    qr_data["Norm.vector angle"] = abs(np.diag(r))
    qr_data.to_csv(save_csv_path)
    
    save_fig_path = save_file_path + '/qr_' + name[:52] + '_regressor_orthogonality.pdf'
    plt.plot(abs(np.diag(r)))
    plt.xlabel('Regressors')
    plt.ylim([0, 1.1])
    plt.ylabel('Norm.vector angle')
    plt.title('Regressor orthogonality')
    plt.savefig(save_fig_path)
    plt.close()
    
    #Rank of a matrix is equal to the number of non-zero singular values
    keep_idx = []
    if matrix_rank(np.nan_to_num(r)) < len(r[0]):
        temp = ~(abs(np.diag(np.nan_to_num(r))) > np.max(np.nan_to_num(r[1].shape)) * np.spacing(r[1]))
        keep_idx_ = np.where(temp == False)[0]
        keep_idx.append(keep_idx_)
        print('Design matrix is rank defficient. Removing %d%d additional regressors. \n', sum(temp), len(keep_idx))
    
    if len(keep_idx) == 0 :
        design_matrix_ = design_matrix
    else:
        design_matrix_ = design_matrix.iloc[:, list(keep_idx[0])] 
        
        normed = design_matrix_ / np.sqrt(np.sum(design_matrix_**2, axis=0))
        normed_zero = normed.fillna(0)
        q, r = np.linalg.qr(normed_zero)
        
        save_csv_path = save_file_path + '/qr_' + name[:52] + '_regressor_orthogonality_removed_reg.csv'
        qr_data = pd.DataFrame()
        qr_data["feature"] = design_matrix_.columns
        qr_data["Norm.vector angle"] = abs(np.diag(r))
        qr_data.to_csv(save_csv_path)
        
        save_fig_path = save_file_path + '/qr_' + name[:52] + '_regressor_orthogonality_removed_reg.pdf'
        plt.plot(abs(np.diag(r)))
        plt.xlabel('Regressors')
        plt.ylim([0, 1.1])
        plt.ylabel('Norm.vector angle')
        plt.title('Regressor orthogonality')
        plt.savefig(save_fig_path)
        plt.close()
    
    return design_matrix_


if __name__ == '__main__':
    
    spikes_dir = r'/home/compraka/Documents/Projects/prosocial_behavior/results_all/caiman_processed'
    spikes_paths = sorted(glob.glob(os.path.join(spikes_dir, '*_spikes.npy*')))
    manual_annotations_dir = r'/home/compraka/Documents/Projects/prosocial_behavior/results_all/manual_annotations/dHC'
    
    days = ['day_0', 'day_2', 'test', 'pool', 'toy', 'stranger']
    manual_annotation_day = []
    for day in days:
        temp_annotations = sorted(glob.glob(os.path.join(manual_annotations_dir + '/' + day, '*_regrouped.csv*')))
        temp_annotations.sort()
        manual_annotation_day.append(temp_annotations)
    manual_annotation_paths = sum(manual_annotation_day, [])
    
    save_df_path = r'/home/compraka/Documents/Projects/prosocial_behavior/results_all/decoder_logistic_regression/behavior_matrices'
    
    # behaviors = ['grooming', 'alone', 'jumping against the wall', 
    #                 'huddled', 'streched posture', 
    #                 'rearing and heading up', 'rearing and heading down',
    #                 'exploratory sniffing', 'heading down',
    #                 'heading up', 'sniffing near the door', 
    #                 'sniffing the door', 'sniffing the mouse',
    #                 'climbing the door', 'animal is pushing the door/lever', 'animal is climbing at the door',
    #                 'anogenital sniffing', 'contact nose-nose',  'approaching the mouse in its social range',
    #                 'following a mouse within its path', 'animal is head butting the door',
    #                 'contact side by side in the opposite way', 'contact side by side in the same way',
    #                 'rearing in contact', 'moving away from a mouse in its social range',
    #                 'moving in contact', 'approaching the mouse in rearing in its social range',
    #                 'stopped in contact']
    
    behaviors = ['exploratory behaviors', 'task behaviors', 
                 'appraisel behaviors', 'defensive behaviors',
                 'prosocial behaviors']
    
    for i in range(len(spikes_paths)):
        spikes = np.load(spikes_paths[i])
        path, name = os.path.split(manual_annotation_paths[i])
        
        design_matrix = create_design_matrix(manual_annotation_paths[i], save_df_path, 
                                             spikes, n_beh_events=behaviors, 
                                             add_forward_jitter=True, forward=5,
                                             add_back_jitter=True, backward=5,
                                             fps=30, to_use='Regrouped Behaviors')
        
        #design_matrix_ = qr_decomp_matrix(design_matrix, save_df_path, name)
        design_matrix_path = save_df_path + '/' + name[:52] + '_design_matrix_unqr_regrouped.csv'
        design_matrix.to_csv(design_matrix_path)
        
        


