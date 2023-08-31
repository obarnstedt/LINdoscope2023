#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:37:05 2023

@author: compraka

Utils file contains all the functions to run the logictic regression decoder. 
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score, RocCurveDisplay
from sklearn.metrics import recall_score
import tqdm
import random


def plot_design_matrix(design_matrix, save_path, name):

    fig, ax = plt.subplots()
    sns.heatmap(design_matrix.T, vmin=0, vmax=1)
    plt.savefig(save_path + '/' + name + '_design_matrix.pdf', bbox_inches='tight')
    plt.close()


def bin_data(data, bin_step, bin_start, bin_stop, func=np.mean):

    bins = np.arange(bin_start, bin_stop, bin_step)
    num_bins = bins.shape[0] - 1
    binned_data = np.zeros([len(bins), data.shape[1]])

    for i in range(num_bins - 1):
        binned_data[i, :] = func(data[bins[i]:bins[i + 1], :], axis=0)


    binned_data = np.where(np.isnan(binned_data), 0, binned_data)
    binned_data = binned_data.astype(int)

    return binned_data


def data_splitter(X, Y):

    idx_class1 = []
    idx_class2 = []

    samples_n = int(len(Y) * 0.2)

    for i in range(0, len(Y[0])):
        for j in range(len(Y)):
            if Y[j, i] == 0 and len(idx_class1) < samples_n:
                idx_class1.append([j])
            elif Y[j, i] == 1 and len(idx_class2) < 2:
                idx_class2.append([j])

    idx_test = idx_class1 + idx_class2
    idx_test = shuffle(idx_test, random_state=0)

    X_test = []
    Y_test = []

    for m in idx_test:
        X_test_temp = X[m]
        Y_test_temp = Y[m]
        X_test.append(X_test_temp)
        Y_test.append(Y_test_temp)

    X_test = np.vstack(X_test)
    Y_test = np.vstack(Y_test)

    X_train = np.delete(X, idx_test, axis=0)
    Y_train = np.delete(Y, idx_test, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(X_train[:, 0])
    ax2.plot(Y_train[:, 0])
    ax1.set_title('Train split')
    sns.despine()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(X_test[:, 0])
    ax2.plot(Y_test[:, 0])
    ax1.set_title('Test split')
    sns.despine()

    return X_train, X_test, Y_train, Y_test


def oversampler(X, Y, sampling_strategy=0.4):

    counter = Counter(Y.flatten())
    print(counter)

    if counter[1] > 1:
        sampler_smote = SMOTE(k_neighbors=3)
        X, Y = sampler_smote.fit_resample(X, Y.flatten())
    else:
        print('Too few samples in one class to create synthetic copies with SMOTE')

    return X, Y


def logistic_regression_decoder(X_train, Y_train, X_test, Y_test):
    """
    Models the probability of a behavior event taking place based
    on the independent variables, here the Spike inferred activity 

    """
    Y_pred = []
    Y_pred_train = []
    accuracy_log_reg = []
    recall_score_log_reg = []
    score_f1_log_reg = []
    auc_score = []
    correlation_val = []

    for i in range(X_train.shape[1]):

        log_reg_classifier = LogisticRegression(penalty=None).fit(
            X_train[:, i].reshape(-1, 1), Y_train.ravel())
        Y_pred_train_ = log_reg_classifier.predict(
            X_train[:, i].reshape(-1, 1))
        Y_pred_ = log_reg_classifier.predict(X_test[:, i].reshape(-1, 1))

        accuracy_log_reg_ = (Y_test == Y_pred_.reshape(-1, 1)).mean()
        recall_score_log_reg_ = recall_score(Y_test, Y_pred_)
        score_f1_log_reg_ = f1_score(Y_test, Y_pred_, average='binary')
        auc_score_ = roc_auc_score(Y_test, Y_pred_)
        corr_val = np.corrcoef(Y_test, Y_pred_, rowvar=False)

        Y_pred.append(Y_pred_)
        Y_pred_train.append(Y_pred_train_)
        accuracy_log_reg.append(accuracy_log_reg_)
        recall_score_log_reg.append(recall_score_log_reg_)
        score_f1_log_reg.append(score_f1_log_reg_)
        auc_score.append(auc_score_)
        correlation_val.append(corr_val[0, 1])

    return np.array(Y_pred_train), np.array(Y_pred), np.array(accuracy_log_reg), np.array(recall_score_log_reg), np.array(score_f1_log_reg), np.array(auc_score), np.array(correlation_val)


def get_R2(Y_test, Y_pred):
    """
    Function that retruns the goodness of fit for each output for model
    R2 tells us how the predictor variables can explain the variation in the 
    response variable 
    Y_test: test data
    Y_pred: the prediction from model
    """
    R2_list = []
    for i in range(Y_pred.shape[0]):
        Y_mean = np.mean(Y_test[:, 0])
        R2 = 1 - np.sum((Y_pred[i, :].reshape(-1, 1) - Y_test[:, 0])
                        ** 2) / np.sum((Y_test[:, 0] - Y_mean) ** 2)
        R2_list.append(R2)

    R2_array = np.array(R2_list)

    return R2_array


def compute_pct(score_f1_log_reg_shuffled, score_f1_log_reg):

    pct_log_reg = np.zeros(len(score_f1_log_reg), dtype=object)

    for c in range(0, len(score_f1_log_reg)):
        n_less = np.sum(score_f1_log_reg_shuffled[:, c] < score_f1_log_reg[c])
        n_equal = np.sum(
            score_f1_log_reg_shuffled[:, c] == score_f1_log_reg[c])
        pct_log_reg[c] = (n_less + n_equal) / (len(score_f1_log_reg_shuffled))

    return pct_log_reg


def shuffle_log_reg(X_train, Y_train, X_test, Y_test, n_shuffle=100, shuffle_strategy='fit_shuffle'):
    """
    shuffle_strategy: default is 'fit_shuffle'. 
                        fit_shuffle: the model is train using unshuffled X_train and Y_train. 
                        And then model is fit using shuffled training data.
                        smat_shuffle: Events bins are shuffled but the behavior bins are left
                        unshuffled in the training and test data.

    n_shuffle: The number of times to shuffle the data

    """
    i = 0
    score_f1_log_reg_shuffled = []
    recall_score_log_reg_shuffled = []
    accuracy_log_reg_shuffled = []
    auc_score_shuffled = []
    corr_vals_shuffled = []

    for i in tqdm.tqdm(range(n_shuffle)):
        if shuffle_strategy == 'fit_shuffle':
            # What has been done in the Kennedy et. al 2017
            # Does not make sense to me. If both X_test and Y_test are shuffled
            # and the model is trained on unshuffled data, you get the same F1 score because
            # the model learns the relationship for each bin and this does not change if both
            # are shuffled.

            shuffle_idx = np.arange(X_test.shape[0])
            np.random.shuffle(shuffle_idx)
            X_test_temp = X_test[shuffle_idx]
            Y_test_temp = Y_test[shuffle_idx]
            _, Y_fit_temp, accuracy_log_reg_temp, recall_score_log_reg_temp, score_f1_log_reg_temp, auc_score_temp, corr_val_temp = logistic_regression_decoder(
                X_train, Y_train, X_test_temp, Y_test_temp)

            # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            # ax1.plot(X_test_temp[3])
            # ax1.plot(X_test[3])
            # ax2.plot(Y_test_temp)
            # ax2.plot(Y_test)
            # ax3.plot(score_f1_log_reg_temp)
            # ax3.set_xlabel('Bins')
            # ax3.set_ylabel('F1 scores')

            score_f1_log_reg_shuffled.append(score_f1_log_reg_temp)
            recall_score_log_reg_shuffled.append(recall_score_log_reg_temp)
            accuracy_log_reg_shuffled.append(accuracy_log_reg_temp)
            auc_score_shuffled.append(auc_score_temp)
            corr_vals_shuffled.append(corr_val_temp)

        elif shuffle_strategy == 'smat_shuffle':

            shuffle_idx = np.arange(Y_test.shape[0])
            np.random.shuffle(shuffle_idx)
            Y_test_temp = Y_test[shuffle_idx]
            _, Y_fit_temp, accuracy_log_reg_temp, recall_score_log_reg_temp, score_f1_log_reg_temp, auc_score_temp, corr_val_temp = logistic_regression_decoder(
                X_train, Y_train, X_test, Y_test_temp)

            # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            # ax1.plot(X_test[3])
            # ax2.plot(Y_test_temp)
            # ax2.plot(Y_test)
            # ax3.plot(score_f1_log_reg_temp)
            # ax3.set_xlabel('Bins')
            # ax3.set_ylabel('F1 scores')

            score_f1_log_reg_shuffled.append(score_f1_log_reg_temp)
            recall_score_log_reg_shuffled.append(recall_score_log_reg_temp)
            accuracy_log_reg_shuffled.append(accuracy_log_reg_temp)
            auc_score_shuffled.append(auc_score_temp)
            corr_vals_shuffled.append(corr_val_temp)

        else:
            print('Check the shuffle strategy you have provided')

    i += 1

    return np.array(score_f1_log_reg_shuffled), np.array(recall_score_log_reg_shuffled), np.array(accuracy_log_reg_shuffled), np.array(auc_score_shuffled), np.array(corr_vals_shuffled)


def verify_pct_vals(pct_log_reg, Y_fit, X_test, Y_test, score_f1_log_reg, save_plots_path):

    cell_idx = np.where(pct_log_reg >= 0.95)[0]

    for i in cell_idx:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(X_test[:, i], label='Spike events')
        ax2.plot(Y_test, label='GT')
        ax2.plot(Y_fit[i, :], '--r', label='Prediction')
        ax1.legend()
        ax2.legend()
        ax1.set_title('F1 score value:{0:.2f}'.format(score_f1_log_reg[i]))
        sns.despine()

        plots_path = save_plots_path + \
            '/shuffle_validation_cell_' + str(i) + '.png'
        plt.savefig(plots_path)
        plt.close()


def run_logistic_regression(sim_spikes, sim_design_matrix, save_plots_path, name,
                            bin_step=30, behavior_decode='animal is pushing the door/lever',
                            compute_shuffle_pct=False):

    stimulus = sim_design_matrix[behavior_decode].values
    stimulus = stimulus.reshape(-1, 1)
    # Binning the input and output data
    bin_start = 0
    bin_stop = stimulus.shape[0]

    binned_spikes = bin_data(sim_spikes, bin_step,
                             bin_start, bin_stop, func=np.max)
    binned_stimulus = bin_data(
        stimulus, bin_step, bin_start, bin_stop, func=np.max)

    binned_spikes = np.where(np.isnan(binned_spikes), 0, binned_spikes)
    binned_spikes = binned_spikes.astype(int)
    binned_stimulus = np.where(np.isnan(binned_stimulus), 0, binned_stimulus)

    X_train, X_test, Y_train, Y_test = data_splitter(
        binned_spikes, binned_stimulus)
    counter = Counter(Y_train.flatten())

    if counter[0] and counter[1] > 4:
        X_train, Y_train = oversampler(X_train, Y_train, sampling_strategy=0.4)
    else:
        print('Not enough values to create synthetic values')

    if counter[0] and counter[1] > 1:

        print('Training and fitting model')
        Y_fit_train, Y_fit, accuracy_log_reg, recall_score_log_reg, score_f1_log_reg, auc_score, corr_vals = logistic_regression_decoder(
            X_train, Y_train, X_test, Y_test)

        responding_cell_idx = np.where(np.array(score_f1_log_reg) > 0.3)[0]

        for cell_idx in range(len(responding_cell_idx)):

            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(X_train[:, cell_idx], '-b', label='Observed spikes')
            ax2.plot(Y_fit_train[cell_idx, :], '-r', label='Fitted events')
            ax2.plot(Y_train, label='Behavior events')
            sns.despine()
            ax1.legend()
            ax2.legend()

            save_fig1_file = save_plots_path + '/' + name + '_' + \
                str(bin_step) + '_validation_train_' + \
                str(responding_cell_idx[cell_idx]) + '.png'
            plt.savefig(save_fig1_file)
            plt.close()

            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(X_test[:, cell_idx], '-b', label='Observed spikes')
            ax2.plot(Y_fit[cell_idx, :], '-r', label='Fitted events')
            ax2.plot(Y_test, label='Behavior events')
            ax1.set_title('F1 score value:{0:.2f}'.format(
                score_f1_log_reg[responding_cell_idx[cell_idx]]))
            sns.despine()
            ax1.legend()
            ax2.legend()

            save_fig2_file = save_plots_path + '/' + name + '_' + \
                str(bin_step) + '_validation_' + \
                str(responding_cell_idx[cell_idx]) + '.png'
            plt.savefig(save_fig2_file)
            plt.close()

        plt.figure()
        plt.plot(score_f1_log_reg)
        plt.axhline(0.4, color='r')
        plt.xlabel('Neurons')
        plt.ylabel('F1 Score')
        sns.despine()
        plt.savefig(save_plots_path + '/' + name + '_' +
                    str(bin_step) + '_f1_score.pdf')
        plt.close()

        plt.figure()
        plt.plot(accuracy_log_reg)
        plt.axhline(0.7, color='r')
        plt.xlabel('Neurons')
        plt.ylabel('Accuracy')
        sns.despine()
        plt.savefig(save_plots_path + '/' + name + '_' +
                    str(bin_step) + '_accuracy.pdf')
        plt.close()

        plt.figure()
        plt.plot(recall_score_log_reg)
        plt.axhline(0.7, color='r')
        plt.xlabel('Neurons')
        plt.ylabel('Recall')
        sns.despine()
        plt.savefig(save_plots_path + '/' + name +
                    '_' + str(bin_step) + '_recall.pdf')
        plt.close()

        plt.figure()
        plt.plot(auc_score)
        plt.axhline(0.7, color='r')
        plt.xlabel('Neurons')
        plt.ylabel('AUC score')
        sns.despine()
        plt.savefig(save_plots_path + '/' + name + '_' +
                    str(bin_step) + '_auc_score.pdf')
        plt.close()

        plt.figure()
        plt.plot(corr_vals)
        plt.axhline(0.7, color='r')
        plt.xlabel('Neurons')
        plt.ylabel('Correlation value')
        sns.despine()
        plt.savefig(save_plots_path + '/' + name + '_' +
                    str(bin_step) + '_corr_vals.pdf')
        plt.close()

        # rsqr_array = get_R2(binned_spikes, Y_fit)

        # rsqr_cell_idx = np.where(rsqr_array >= 0.8)[0]

        # for rsqr_idx in  range(len(rsqr_cell_idx)):
        #     fig, (ax1, ax2) = plt.subplots(2, 1)
        #     ax1.plot(binned_spikes[:, rsqr_idx], '-b', label='Observed spikes')
        #     ax2.plot(Y_fit[:, rsqr_idx], '-r', label='Fitted events')
        #     ax2.plot(binned_stimulus, label='Behavior events')
        #     ax1.set_title('R2 value:{0:.2f}'.format(rsqr_array[rsqr_array[rsqr_idx]]))
        #     sns.despine()
        #     ax1.legend()
        #     ax2.legend()
        #     save_fig2_file = save_plots_path + '/' + name + '_' + str(bin_step) + '_validation_' + str(responding_cell_idx[rsqr_idx]) + '_rsqr.pdf'
        #     plt.savefig(save_fig2_file)
        #     plt.close()

        # plt.figure()
        # plt.plot(rsqr_array)
        # plt.axhline(0.7, color='r')
        # plt.xlabel('Neurons')
        # plt.ylabel('R2')
        # sns.despine()
        # plt.savefig(save_plots_path + '/' + name + '_' + str(bin_step) + '_rsqr.pdf')

        data_dict = {'F1 score': score_f1_log_reg,
                     'Accuracy': accuracy_log_reg,
                     'Recall': recall_score_log_reg,
                     'AUC score': auc_score,
                     'Correlation value': corr_vals,
                     }

        if compute_shuffle_pct == True:
            print('Computing the shuffled F1 scores')
            # Two different strategies can be used to compute the shuffled F1 scores
            score_f1_log_reg_shuffled, recall_score_log_reg_shuffled, accuracy_log_reg_shuffled, auc_score_shuffled, corr_vals_shuffled = shuffle_log_reg(
                X_train, Y_train, X_test, Y_test, n_shuffle=100, shuffle_strategy='smat_shuffle')

            print('Saving shuffled scores')
            save_shuffled_path = save_plots_path + \
                '/' + name + '_' + str(bin_step)
            np.save(save_shuffled_path + 'shuffled_accuracy.npy',
                    accuracy_log_reg_shuffled)
            np.save(save_shuffled_path + 'shuffled_recall.npy',
                    recall_score_log_reg_shuffled)
            np.save(save_shuffled_path + 'shuffled_f1_score.npy',
                    score_f1_log_reg_shuffled)
            np.save(save_shuffled_path +
                    'shuffled_auc_score.npy', auc_score_shuffled)
            np.save(save_shuffled_path +
                    'shuffled_corr_score.npy', corr_vals_shuffled)

            pct_log_reg = compute_pct(
                score_f1_log_reg_shuffled, score_f1_log_reg)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax1.hist(pct_log_reg, bins=10)
            ax1.axvline(0.95, color='r')
            ax1.set_xlabel('Neuron No.')
            ax1.set_ylabel('No. of Neurons')
            ax2.hist(score_f1_log_reg_shuffled[0], bins=10)
            ax3.hist(score_f1_log_reg_shuffled[10], bins=10)
            ax2.set_ylabel('Shuffle 0 F1 scores')
            ax3.set_ylabel('Shuffle 10 F1 scores')
            sns.despine()
            plt.savefig(save_plots_path + '/' + name + '_' +
                        str(bin_step) + '_histograms_f1_scores_shuffled_data.pdf')
            plt.close()

            data_dict['Percentile score'] = pct_log_reg

        print('Saving results')
        results_df = pd.DataFrame.from_dict(data_dict, orient="columns")
        csv_file_path = save_plots_path + '/' + \
            name + '_' + str(bin_step) + 'results.csv'
        results_df.to_csv(csv_file_path)

        #print('Verifying results')
        #verify_pct_vals(pct_log_reg, Y_fit, X_test, Y_test, score_f1_log_reg, save_plots_path)

    else:
        print('Behavior does not exist in this session')
