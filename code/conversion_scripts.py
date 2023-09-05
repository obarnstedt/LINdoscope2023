import pandas as pd
import h5py
import numpy as np

behaviour = pd.read_hdf('/Users/Oliver/Google Drive/LINdoscope2023_analysis/sample_data/2P_dCA1_treadmill/behaviour_ft.h5')
behaviour.columns = pd.MultiIndex.from_product([['behaviour'], list(behaviour.columns)])

with h5py.File('/Users/Oliver/Google Drive/LINdoscope2023_analysis/sample_data/2P_dCA1_treadmill/OB_167_green_caiman_results3.hdf5', "r") as f:
    calcium = pd.DataFrame(np.array(f['estimates']['C']).T)
calcium.columns = pd.MultiIndex.from_product([['cell'], list(calcium.columns)])

bodydlc = pd.read_hdf('/Users/Oliver/Google Drive/LINdoscope2023_analysis/sample_data/2P_dCA1_treadmill/Basler acA780-75gm (22611479)_20190218_151620616DeepCut_resnet50_MouseBodySep9shuffle1_800000.h5')
bodydlc.columns = bodydlc.columns.droplevel(0)

facedlc = pd.read_hdf('/Users/Oliver/Google Drive/LINdoscope2023_analysis/sample_data/2P_dCA1_treadmill/Basler acA780-75gm (22611477)_20190218_151617709DeepCut_resnet50_MouseFaceAug21shuffle1_1030000.h5')
facedlc.columns = facedlc.columns.droplevel(0)

calcium = calcium.join(facedlc.iloc[behaviour.behaviour.camframes].reset_index(drop=True))
calcium = calcium.join(bodydlc.iloc[behaviour.behaviour.camframes].reset_index(drop=True)).join(behaviour)
calcium

calcium.to_hdf('/Users/Oliver/Google Drive/LINdoscope2023_analysis/sample_data/2P_dCA1_treadmill/calcium_behaviour.h5', 'all')