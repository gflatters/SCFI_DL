#!/usr/bin/env python
# coding: utf-8

# In[2]

#functions used to get fluctuations, lengths, etc

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import os
import pickle
import pandas as pd
import Centre_finder_vitamica

# In[5]:


def load_video_to_array(path):
    #from a path, find all images which begin /crop, normalise, and load into array
    frames = glob.glob(path+'/crop*')
    loaded_frame_list = []
    for fr in range(len(frames)):
        frame = plt.imread(frames[fr]).astype(np.uint16)/65535.
        loaded_frame_list.append(frame)
    return np.array(loaded_frame_list)

def findstd(video,threshold):
    #find standard deviation of video pixels in time, 0 those below a threshold
    stds = np.std(video,axis=0)
    stds[stds<threshold]=0
    return stds

def getfluctuations(path, controlpath,suspath,thresh):
    #for path/control path and path/suspath, find all 128x128 cell videos, take the standard deviation in time
    #return average standard deviation, average image intensity and a label
    result = dict()
    for i in glob.glob(path+suspath+'sample*'):
        print(i)
        for j in glob.glob(i+'/cell[0-9]*'):
            video = load_video_to_array(j+'/cropped_video128')
            stds = findstd(video,thresh)
            result[j] = (float(np.average(stds)),float(np.average(video)),1)
    
    for i in glob.glob(path+controlpath+'sample*'):
        print(i)
        for j in glob.glob(i+'/cell[0-9]*'):
            video = load_video_to_array(j+'/cropped_video128')
            stds = findstd(video,thresh)
            result[j] = (float(np.average(stds)),float(np.average(video)),0)
            
    return result

def flucs_from_csv(path):
    #given a path, find all cell fluctuations in the included .csv files 
    flucs_dict = {}
    for sample in glob.glob(path+'sample*'): #iterate through all samples
        
        csv_path = glob.glob(sample+'/out/*checked.csv')#read csv
        flucs = np.array(pd.read_csv(csv_path[0])['fluctuations'])#get flucs
        paths = np.array(pd.read_csv(csv_path[0])['video'])#get paths
        for index, path in enumerate(paths):
            cell_number = path[-3:]#get cell number from path (assuming videonum = cell num)
            correct_path = sample+'/cell'+cell_number #turn this into my file system ordering
            flucs_dict[correct_path] = flucs[index] #fill up dict
    return flucs_dict

def gettargets(controlpath,suspath):
    #get dict of {paths: labels}
    targets = {}
    for i in glob.glob(controlpath+'sample*'):
        for j in glob.glob(i+'/cell[0-9]*'):
            targets[j]=0
    for i in glob.glob(suspath+'sample*'):
        for j in glob.glob(i+'/cell[0-9]*'):
            targets[j]=1
    return targets


def getlengths(path_to_samples):
    #use vitamica code to get lenghts+centres
    lengths = []
    for path in glob.glob(path_to_samples+'sample*'):
        lengths.append(Centre_finder_vitamica.find_parameters(sample_path=path))
    
    return [item for sublist in lengths for item in sublist]



