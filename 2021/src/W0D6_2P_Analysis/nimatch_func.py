# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 12:33:10 2021

@author: pbauer
"""


import pandas as pd
from nptdms import TdmsFile
from caiman.source_extraction.cnmf.cnmf import load_CNMF

#ptm
from ptm_treadmill import NIposition_processing
from ptm_treadmill import triggerfinder
from ptm_treadmill import lickometer_processing
from ptm_treadmill import resotriggerfinder

def nimatch(analog_file,counter_file,cnmf_file):
    #read in analog file
    tdms_analog_file = TdmsFile(analog_file)
    
    #read counter file
    tdms_position_file = TdmsFile(counter_file)
    
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

    
    #processed data dataframe
    treadmill_data['Position'], treadmill_data['Velocity'], treadmill_data['Lap'] = NIposition_processing(treadmill_raw['position'], treadmill_raw['laptrigger'], params)
    treadmill_data['CamTrigger'], camtriggertimepoints = triggerfinder(treadmill_raw['camtrigger'], .9, 10)
    treadmill_data['Licking'], treadmill_data['Licks'] = lickometer_processing(treadmill_raw['licking'], params)
    treadmill_data['Pump'] = treadmill_raw['pump'] > 0.4
    treadmill_data['ResoTrigger'], resotriggertimepoints = triggerfinder(treadmill_raw['resotrigger'], 2.4, 20)
    treadmill_data['ResoTriggerBlocks'], resotriggerblocks = resotriggerfinder(treadmill_raw['resotrigger'], 2.4, 10000, 250, params['treadmill']['NI_Hz']) #fails!
    
    #load CaiMan results
    cnm = load_CNMF(cnmf_file)
    dFF_all = cnm.estimates.F_dff
    C_all = cnm.estimates.C
    S_all = cnm.estimates.S
    good_idx= cnm.estimates.idx_components
    
    #create dFF_samp from dFF of good components
    dFF = dFF_all[good_idx]
    S = S_all[good_idx]
    C = C_all[good_idx]
    
    #align position to frame trigger
    pos_full = treadmill_data['Position'].to_numpy()
    pos = pos_full[resotriggertimepoints]
    #3) lap = lapSamp
    lap_full=treadmill_data['Lap'].to_numpy()
    lap = lap_full[resotriggertimepoints]
    v_full=treadmill_data['Velocity'].to_numpy()
    v = v_full[resotriggertimepoints]
    
    return dFF,S,C,pos,lap,v
    
