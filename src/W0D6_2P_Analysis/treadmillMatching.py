# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:33:10 2021

@author: pbauer
"""


import pandas as pd
from nptdms import TdmsFile
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy.signal import hilbert, butter, filtfilt, find_peaks
#from statsmodels.nonparametric.smoothers_lowess import lowess
import time
import matplotlib.pyplot as plt 

#caiman
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as cmparams
from caiman.source_extraction.cnmf.cnmf import load_CNMF

#ptm
from ptm_treadmill import NIposition_processing
from ptm_treadmill import triggerfinder
from ptm_treadmill import lickometer_processing
from ptm_treadmill import resotriggerfinder
from ptm_treadmill import Kalman_Filt_v


analog_file = '2021-08-17_11-27-59_analog_log.tdms'
counter_file = '2021-08-17_11-27-59_counter_log.tdms'

#read in analog file
tdms_analog_file = TdmsFile(analog_file)
analog_group=tdms_analog_file['analog_log_task']

#read counter file
tdms_position_file = TdmsFile(counter_file)
counter_group=tdms_position_file['counter_log_task']

# read into a pandas dataframe, check configuration xls file for specific device and channel IDs!
treadmill_raw = pd.DataFrame()
treadmill_raw['position'] = tdms_position_file['counter_log_task']['Dev5/ctr1'].data
treadmill_raw['time'] = tdms_position_file['counter_log_task']['Dev5/ctr1'].time_track()
treadmill_raw['resotrigger'] = tdms_analog_file['analog_log_task']['Dev5/ai1'].data
treadmill_raw['licking'] = tdms_analog_file['analog_log_task']['Dev5/ai7'].data
treadmill_raw['laptrigger'] = tdms_analog_file['analog_log_task']['Dev5/ai5'].data
treadmill_raw['camtrigger'] = tdms_analog_file['analog_log_task']['Dev5/ai2'].data
treadmill_raw['pump'] = tdms_analog_file['analog_log_task']['Dev5/ai0'].data

#parameters
treadmill_data = pd.DataFrame()
params = {'treadmill': {'NI_Hz': 10000,
                        'BeltMinMax' : str('0:360')},
          'xlsmeta': {'Belt_cm': '360'}}

#oops! position should not jump backward in the beginning
treadmill_raw['position'][0:40000]=0

#processed data dataframe
treadmill_data['Position'], treadmill_data['Velocity'], treadmill_data['Lap'] = NIposition_processing(treadmill_raw['position'], treadmill_raw['laptrigger'], params)
treadmill_data['CamTrigger'], camtriggertimepoints = triggerfinder(treadmill_raw['camtrigger'], .9, 10)
treadmill_data['Licking'], treadmill_data['Licks'] = lickometer_processing(treadmill_raw['licking'], params)
treadmill_data['Pump'] = treadmill_raw['pump'] > 0.4
treadmill_data['ResoTrigger'], resotriggertimepoints = triggerfinder(treadmill_raw['resotrigger'], 2.4, 20)
treadmill_data['ResoTriggerBlocks'], resotriggerblocks = resotriggerfinder(treadmill_raw['resotrigger'], 2.4, 10000, 250, params['treadmill']['NI_Hz']) #fails!

#load CaiMan results
cnm = load_CNMF('analysis_results.hdf5')
dFF = cnm.estimates.F_dff.T
C = cnm.estimates.C.T
S = cnm.estimates.S.T
good_components= cnm.estimates.idx_components
num_comps=C.shape[1]

#create dFF_samp
dFF_samp = np.zeros((num_comps,len(resotriggertimepoints)))
dFF_samp = C

# only need downsampling pos and lap


#what we need:
#1) dFF_samp

#2) pos = posSamp
pos = treadmill_data['Position'].to_numpy()
#3) lap = lapSamp
lap=treadmill_data['Lap']