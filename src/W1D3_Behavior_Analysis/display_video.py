# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:58:25 2021

@author: kluxem
"""

import os
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch.utils.data as Data

from vame.util.auxiliary import read_config
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

def play_motif_video(config, motif=0, parameterization="hmm"):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    
    files = []
    if cfg['all_data'] == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to train on " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        for file in cfg['video_sets']:
            files.append(file)
    
    file = files[0]
    
    path_to_file=os.path.join(cfg['project_path'],"results",file,model_name,parameterization+'-'+str(n_cluster),"cluster_videos",file+'-motif_'+str(motif)+'.avi')

    
    cap = cv2.VideoCapture(path_to_file)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
    
        # Display the resulting frame
        cv2.imshow('Frame',frame)
    
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    
      # Break the loop
      else: 
        break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()