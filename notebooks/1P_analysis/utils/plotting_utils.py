#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:48:29 2023

@author: compraka
"""
import matplotlib.pyplot as plt
import seaborn as sns

def plot_smat(spikes, cell_idx=10):
    
    plt.figure()
    plt.plot(spikes[cell_idx, :])
    sns.despine()
    
    
def plot_design_matrix(design_matrix):

    plt.figure()
    sns.heatmap(design_matrix.T, vmin=0, vmax=1)
    
    
def plot_binned_data(data, binned_data, cell_idx=10):
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(data[:, cell_idx], label='raw data')
    ax1.legend()
    ax2.plot(binned_data[:, cell_idx], label='binned data')
    ax2.legend()
    sns.despine()
    

def plot_train(X_train, Y_train, Y_fit_train, cell_idx=10):
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(X_train[:, cell_idx], '-b', label='Observed spikes')
    ax2.plot(Y_fit_train[cell_idx, :], '-r', label='Fitted events')
    ax2.plot(Y_train, label='Behavior events')
    sns.despine()
    ax1.legend()
    ax2.legend()
    
    
def plot_test(X_test, Y_test, Y_fit, cell_idx=10):
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(X_test[:, cell_idx], '-b', label='Observed spikes')
    ax2.plot(Y_fit[cell_idx, :], '-r', label='Fitted events')
    ax2.plot(Y_test, label='Behavior events')
    sns.despine()
    ax1.legend()
    ax2.legend()
    