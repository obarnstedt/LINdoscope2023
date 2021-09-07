#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyTreadMouse functions to analyse treadmill data

Created on Wed May  8 14:27:32 2019
@author: Oliver Barnstedt, with contributions from Dennis Daluegge (licking), Pavol Bauer and Daniel Justus (FPH processing)
"""

import pandas as pd
from nptdms import TdmsFile
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
#from igor import binarywave as bw
from scipy.signal import hilbert, butter, filtfilt, find_peaks
#from statsmodels.nonparametric.smoothers_lowess import lowess
import time


#  1/3) READ TDMS FILES
def tdmsreader(params):
    treadmill_raw = pd.DataFrame()
    print("Loading TDMS position/counter file...")
    tdms_position_file = TdmsFile(params['paths']['Data_NI_position'])
    treadmill_raw['position'] = tdms_position_file.object('counter_log_task', 'Dev1/ctr1').data
    treadmill_raw['time'] = tdms_position_file.object('counter_log_task', 'Dev1/ctr1').time_track()

    print("Loading TDMS analogue file...")
    tdms_analog_file = TdmsFile(params['paths']['Data_NI_analog'])

    # Setting NI channel defaults
    if 'lick_channel' not in params['xlsmeta']:
        params['xlsmeta']['lick_channel'] = 0
    lick_channel = str('Dev1/ai'+str(int(params['xlsmeta']['lick_channel'])))

    if 'resotrigger_channel' not in params['xlsmeta']:
        params['xlsmeta']['resotrigger_channel'] = 1
    resotrigger_channel = str('Dev1/ai'+str(int(params['xlsmeta']['resotrigger_channel'])))

    if 'camtrigger_channel' not in params['xlsmeta']:
        params['xlsmeta']['camtrigger_channel'] = 2
    camtrigger_channel = str('Dev1/ai'+str(int(params['xlsmeta']['camtrigger_channel'])))

    if 'pump_channel' not in params['xlsmeta']:
        params['xlsmeta']['pump_channel'] = 3
    pump_channel = str('Dev1/ai'+str(int(params['xlsmeta']['pump_channel'])))

    if 'laptrigger_channel' not in params['xlsmeta']:
        params['xlsmeta']['laptrigger_channel'] = 4
    laptrigger_channel = str('Dev1/ai'+str(int(params['xlsmeta']['laptrigger_channel'])))

    if 'brakepuff_channel' not in params['xlsmeta']:
        params['xlsmeta']['brakepuff_channel'] = 5
    brakepuff_channel = str('Dev1/ai'+str(int(params['xlsmeta']['brakepuff_channel'])))

    if 'optotrigger_channel' not in params['xlsmeta']:
        params['xlsmeta']['optotrigger_channel'] = 6
    optotrigger_channel = str('Dev1/ai'+str(int(params['xlsmeta']['optotrigger_channel'])))

    treadmill_raw['licking'] = tdms_analog_file.object('analog_log_task', lick_channel).data
    if 'Data_Imaging_Dir' in params['paths']:
        treadmill_raw['resotrigger'] = tdms_analog_file.object('analog_log_task', resotrigger_channel).data
        print("Loading resonant triggers...")
    else:
        print("No resonant triggers found; continuing without...")
    treadmill_raw['camtrigger'] = tdms_analog_file.object('analog_log_task', camtrigger_channel).data
    treadmill_raw['pump'] = tdms_analog_file.object('analog_log_task', pump_channel).data
    treadmill_raw['laptrigger'] = tdms_analog_file.object('analog_log_task', laptrigger_channel).data
    try:
        treadmill_raw['brakepuff'] = tdms_analog_file.object('analog_log_task', brakepuff_channel).data
    except:
        print("No brakepuff signal found; continuing without...")
    try:
        treadmill_raw['optotrigger'] = tdms_analog_file.object('analog_log_task', optotrigger_channel).data
    except:
        print("No optotrigger found; continuing without...")

    if params['treadmill']['save_raw_treadmill']:
        print("Saving NI raw data to " + str(params['paths']['Results_NI_raw']) + '...')
        treadmill_raw.to_hdf(path_or_buf=params['paths']['Results_NI_raw'], key='treadmill_raw')

    return treadmill_raw

# 1b: IGOR READER
def igorreader(params):
    startsweep = int(params['xlsmeta']['Sweeps'].split('_')[0])
    for idx in range(len(params['paths']['Data_IGOR_Position'])):
        print("Loading IGOR files for sweep {}/{}...".format(idx + startsweep, int(params['xlsmeta']['Sweeps'].split('_')[-1])))
        treadmill_raw = pd.DataFrame()
        treadmill_raw['position'] = bw.load(params['paths']['Data_IGOR_Position'][idx])['wave']['wData']
        treadmill_raw['time'] = np.linspace(start=0, stop=params['treadmill']['IGOR_Length_sec'][idx], num=len(treadmill_raw))
        treadmill_raw['licking'] = bw.load(params['paths']['Data_IGOR_Licking'][idx])['wave']['wData']
        treadmill_raw['pump'] = bw.load(params['paths']['Data_IGOR_Pump'][idx])['wave']['wData']
        treadmill_raw['fph'] = bw.load(params['paths']['Data_IGOR_FPH'][idx])['wave']['wData']
        treadmill_raw['fphlp'] = bw.load(params['paths']['Data_IGOR_FPHlp'][idx])['wave']['wData']
        treadmill_raw['camtrigger'] = bw.load(params['paths']['Data_IGOR_CamTriggers'][idx])['wave']['wData']
        treadmill_raw['optotrigger'] = bw.load(params['paths']['Data_IGOR_Stim'][idx])['wave']['wData']
        print("Saving IGOR raw data to {} as '/w{:02d}/'...".format(params['paths']['Results_NI_raw'], idx + startsweep))
        treadmill_raw.to_hdf(path_or_buf=params['paths']['Results_NI_raw'], key='w{:02d}'.format(idx + startsweep))

# 2/3) PROCESS TREADMILL DATA
def treadmill_processing(params, treadmill_raw, hz):
    treadmill_data = pd.DataFrame()
    treadmill_data['Time_s'] = treadmill_raw['time']
    #    treadmill_data.index = pd.to_datetime(treadmill_raw['time'], unit='s', origin = paths['NI_startdate']).astype('datetime64[ms]')
    #    treadmill_data.index.name = 'Time_abs'
    if 'FPH' in params['xlsmeta']['Setup']:
        treadmill_data['Position'], treadmill_data['Velocity'], treadmill_data['Lap'] = IGORposition_processing(treadmill_raw['position'], hz, params)
        treadmill_data['FPH'] = FPH_processing(treadmill_raw['fphlp'], hz, params)
    else:
        treadmill_data['Position'], treadmill_data['Velocity'], treadmill_data['Lap'] = NIposition_processing(treadmill_raw['position'], treadmill_raw['laptrigger'], params)
    treadmill_data['CamTrigger'], camtriggertimepoints = triggerfinder(treadmill_raw['camtrigger'], .9, 10)
    treadmill_data['Licking'], treadmill_data['Licks'] = lickometer_processing(treadmill_raw['licking'], hz, params)
    treadmill_data['Pump'] = treadmill_raw['pump'] > 0.4
    if 'optotrigger' in treadmill_raw:
        if max(treadmill_raw['optotrigger']) > 0.01:
            treadmill_data['OptoTrigger'] = False
            if 'FPH' in params['xlsmeta']['Setup']:
                treadmill_data['OptoTrigger'][treadmill_raw['optotrigger'] > .95] = True
            else:
                treadmill_data['OptoTrigger'][treadmill_raw['optotrigger'] > 1.15] = True
    if 'Data_Imaging_Dir' in params['paths']:
        if params['treadmill']['NI_Hz'] < 2000:  # if NI data collected slower than 2 kHz
            treadmill_data['ResoTrigger'], resotriggertimepoints = triggerfinder(treadmill_raw['resotrigger'], 4, 20)
            treadmill_data['ResoTriggerBlocks'], resotriggerblocks = resotriggerfinder(treadmill_raw['resotrigger'], 4, 1000, 250, params['treadmill']['NI_Hz'])
        else:  # if NI data collected faster than 2 kHz
            treadmill_data['ResoTrigger'], resotriggertimepoints = triggerfinder(treadmill_raw['resotrigger'], 2.4, 20)
            treadmill_data['ResoTriggerBlocks'], resotriggerblocks = resotriggerfinder(treadmill_raw['resotrigger'], 2.4, 10000, 250, params['treadmill']['NI_Hz'])

    return treadmill_data

def FPH_processing(fphlp, hz, params):
    ds_factor = 10
    fph_ds = fphlp[::ds_factor]
    polydegree = 2
    x = np.arange(len(fph_ds))
    p = np.polyfit(x, fph_ds, polydegree)
    y = np.polyval(p, x)
    dff = (fph_ds - y) / fph_ds
    print(time.ctime()+" Applying LOESS filter on {}x downsampled signal...".format(ds_factor))
    dff_smoothed = lowess(dff, x, (.005*ds_factor)/(hz/1000))[:, 1]
    print(time.ctime()+" Finished LOESS filter.")
    zscore = dff_smoothed / np.std(dff_smoothed)
    zscore_upsampled = zscore.repeat(ds_factor)

    return zscore_upsampled


def NIposition_processing(position_raw, laptrigger, params):
    # CONCATENATING NI POSITION SIGNAL
    position_flattened = position_raw.copy()
    lap = pd.Series([1] * len(laptrigger))
    NIpositionjumps = position_raw.index[position_raw.diff() < -20000].tolist()
    for posindex in NIpositionjumps:
        position_flattened[posindex:] = position_flattened[posindex:] + 65535

    # KALMAN VELOCITY
    print("Using Kalman Filter to calculate velocity...")
    posF, velocity = Kalman_Filt_v(position_flattened, 1 / params['treadmill']['NI_Hz'])

    # LAPS
    beltpositionjumps = position_flattened.index[laptrigger.diff() > 1].tolist()
    position_laps = position_flattened.copy()
    for idx, posindex in enumerate(beltpositionjumps):
        position_laps[posindex:] = position_laps[posindex:] - position_laps[posindex]
        lap[posindex:] = lap[posindex:] + 1
    if beltpositionjumps:  # if at least one lap signal has been triggered
        position_laps[:beltpositionjumps[0]] = position_laps[:beltpositionjumps[0]] + (
                max(position_laps) - max(position_laps[0:beltpositionjumps[0]]))

    # POSITION SCALING
    BeltMinMax = tuple([int(params['treadmill']['BeltMinMax'].split(':')[0]), int(params['treadmill']['BeltMinMax'].split(':')[1])])
    scaler = MinMaxScaler(feature_range=BeltMinMax)
    position = scaler.fit_transform(position_laps.to_frame()).flatten()

    # VELOCITY CALIBRATION
    total_distance_run_cm = (max(position_flattened) - min(position_flattened)) / 65535 * int(params['xlsmeta']['Belt_cm'])
    total_time = int(len(position_raw) / params['treadmill']['NI_Hz'])
    avg_velocity_real = total_distance_run_cm / total_time
    avg_velocity_measured = np.mean(velocity)
    velocity_calibrated = velocity * (avg_velocity_real / avg_velocity_measured)

    return position, velocity_calibrated, lap


def IGORposition_processing(position_raw, Hz, params):
    lap = pd.Series([1] * len(position_raw))
    IGORpositionjumps = position_raw.index[position_raw.diff() < -5].tolist()
    position_flattened = position_raw.copy()
    for posindex in IGORpositionjumps:
        position_flattened[posindex:] = position_flattened[posindex:] + 9

    # KALMAN VELOCITY
    print("Using Kalman Filter to calculate velocity...")
    posF, velocity = Kalman_Filt_v(position_flattened, 1 / Hz)

    # LAPS
    for idx, posindex in enumerate(IGORpositionjumps):
        lap[posindex:] = lap[posindex:] + 1

    # POSITION SCALING
    BeltMinMax = tuple([int(params['treadmill']['BeltMinMax'].split(':')[0]), int(params['treadmill']['BeltMinMax'].split(':')[1])])
    beltscale = abs(BeltMinMax[0]-BeltMinMax[1]) / 9
    position = (position_raw+4.5)*beltscale+BeltMinMax[0]

    # VELOCITY CALIBRATION
    total_distance_run_cm = (max(position_flattened) - min(position_flattened)) / 9 * int(params['xlsmeta']['Belt_cm'])
    total_time = int(len(position_raw)/Hz)
    avg_velocity_real = total_distance_run_cm / total_time
    avg_velocity_measured = np.mean(velocity)
    velocity_calibrated = velocity * (avg_velocity_real / avg_velocity_measured)

    return position, velocity_calibrated, lap

def triggerfinder(triggers_raw, thresh_diff, thresh_frames):
    triggers = np.where(triggers_raw.diff() > thresh_diff)[0]
    truetriggers = np.diff(triggers) > thresh_frames
    truetriggers = np.append(truetriggers, [True])
    triggers_output = pd.Series([False] * len(triggers_raw))
    triggers_output[triggers[truetriggers]] = True
    return triggers_output, triggers[truetriggers]


def resotriggerfinder(triggers_raw, thresh_diff, thresh_frames, min_s, NI_Hz):
    triggers = np.where(triggers_raw.diff() > thresh_diff)[0]
    blocks = pd.DataFrame()
    blocks['start'] = triggers[np.where(np.diff(triggers) > thresh_frames)[0] + 1]
    # blocks['end'] = math.nan
    if sum(np.diff(triggers) > thresh_frames) > 1:
        blocks['end'] = pd.Series(triggers[np.where(np.diff(triggers) > thresh_frames)[0]][1:])
        blocks['end'].iloc[-1] = triggers[-1]
    else:
        blocks['end'] = triggers[-1]+1
    blocks['length'] = blocks['end'] - blocks['start']
    blocks = blocks[blocks['length'] > min_s * NI_Hz].astype(int).reset_index(drop=True)
    blockframes = pd.Series([math.nan] * len(triggers_raw))
    for blockidx, blockrow in blocks.iterrows():
        blockframes[blockrow['start']: blockrow['end']] = blockidx
    return blockframes, blocks



def lickometer_processing(licking_raw, params):
    """
    Created on Fri Jan 24 16:08:42 2020
    @author: Dennis Daluegge, modified by Oliver Barnstedt
    """
    SampFreq=params['treadmill']['NI_Hz']
    LickingZeroed = licking_raw - np.mean(licking_raw)

    # Hilbert-Transform for Envelope:
    Analytic_signal = hilbert(LickingZeroed)
    Licking_Envelope = np.abs(Analytic_signal)

    # Low-Pass Butter filter:
    fc = 2  # Cut-off frequency of the filter
    w = fc / (SampFreq / 2)  # Normalize the frequency to SamplingFrequency
    b, a = butter(5, w, 'low')
    LickingEnvFilt = filtfilt(b, a, Licking_Envelope)
    LickingEnvFilt[0:int(1.5 * SampFreq)] = 0

    # Find Lick-Events:
    if 'LickAmpThresholdAbs' in params['treadmill']:
        if params['treadmill']['LickAmpThresholdAbs']:
            LickAmpThreshold = params['treadmill']['LickAmpThresholdAbs']
    elif 'LickAmpThresholdStd' in params['treadmill']:
        if params['treadmill']['LickAmpThresholdStd']:
            LickAmpThreshold = np.std(LickingEnvFilt) * params['treadmill']['LickAmpThresholdStd']
        else:
            LickAmpThreshold = np.std(LickingEnvFilt) * 1.5
    else:
        LickAmpThreshold = np.std(LickingEnvFilt) * 1.5
    Lickevents, LickAmpsDic = find_peaks(LickingEnvFilt,
                                         height=LickAmpThreshold)  # ,ThresholdAmp=LickAmpThreshold,ThresholdNumFrames=10)
    LickAmps = LickAmpsDic['peak_heights']

    # Control in Time Domain (Window in which two licks can occure):
    WindowSize = 1 * SampFreq
    i = 0
    TrueLickEvents = []
    while i < len(Lickevents):
        #        print(i)
        DistanceForSingle = np.absolute(Lickevents - Lickevents[i])
        #        print(DistanceForSingle)
        Possible = np.where(DistanceForSingle < WindowSize)
        #        print(Possible)
        if len(Possible[0]) > 1:
            # print('First')
            MaxInt = np.argmax(LickAmps[Possible[0]])
            MaxIdx = Possible[0][MaxInt]
            TrueLickEvents.append(Lickevents[MaxIdx])
            i = i + len(Possible[0]) - 1
        else:
            # print('Second')
            TrueLickEvents.append(Lickevents[i])
            i += 1

    # Get Array for LickEvents
    LickingColumn = np.zeros(shape=(len(licking_raw), 1))
    i = 0
    while i < len(TrueLickEvents):
        LickingColumn[TrueLickEvents[i]] = 1
        i += 1
    # As DataFrame:
    LickingColumn = pd.DataFrame(columns=['Licks'], data=LickingColumn)

    return LickingEnvFilt, LickingColumn



def Kalman_Filt_v(pos, dt):
    measurements = pos

    #  initialize
    #    x  x'
    x = np.array([[pos[0]], [0]], ndmin=2)  # Initial State (Location and velocity and acceleration)
    P = np.array([[1, 0], [0, 100]])  # Initial Uncertainty
    A = np.array([[1, dt], [0, 1]])  # Transition Matrix

    # Measurement function
    H = np.array([1, 0], ndmin=2)

    # measurement noise covariance; 1e-3 recommended, smaller values for precise onset, larger for smooth velocity
    R = 1e-3

    # Process Noise Covariance
    Q = np.array([[1 / 4 * dt ** 4, 1 / 2 * dt ** 3], [1 / 2 * dt ** 3, dt ** 2]])

    # Identity matrix
    I = np.identity(2)

    #  compare to datapoints
    posF = []
    vF = []

    with tqdm(total=len(measurements)) as pbar:
        for n, measurement in enumerate(measurements):
            # Prediction
            x = np.matmul(A, x)  # predicted State
            P = A @ P @ A.transpose() + Q  # predicted Covariance

            # Correction
            Z = measurement
            y = Z - np.matmul(H, x)  # Innovation from prediction and measurement
            S = H @ P @ H.transpose() + R  # Innovation-covariance
            K = np.matmul(P, H.transpose()) / S  # Filter-Matrix (Kalman-Gain)

            x = x + (K * y)  # recalculation of system state
            # print(x)
            posF.append(np.float64(x[0]))
            vF.append(np.float64(x[1]))

            P = np.matmul(I - (np.matmul(K, H)), P)  # recalculation of covariance
            pbar.update(1)

    return pd.Series(posF), pd.Series(vF)
