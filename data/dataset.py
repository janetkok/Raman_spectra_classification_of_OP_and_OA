from re import X
import numpy as np
import torch
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from scipy.interpolate import UnivariateSpline
from pybaselines import whittaker, spline, morphological, misc, polynomial, smooth, classification, spline, optimizers

class RamanDataset(torch.utils.data.Dataset):
    """raman dataset 
    Arguments:
        ds: dataset containing path to images and their respective labels
        split_idx: indices of samples to be used
        cnn: indicate if network used is cnn
    """
    def __init__(self, 
                ds, 
                split_idx,cnn=False):
        scaler = StandardScaler()


        ds.iloc[:,1:] = scaler.fit_transform(ds.iloc[:,1:])

        ds = ds.iloc[split_idx, :].copy()
        ds.reset_index(drop=True,inplace=True)
    
        self.ds = ds  
        self.cnn= cnn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index): 
        x = np.array(self.ds.iloc[index,1:])
        if self.cnn:
            x = np.expand_dims(x, axis=0) # remove if cnn
        y = np.array(self.ds.iloc[index,0])
        return x,y

class RamanMultiChannelDataset(torch.utils.data.Dataset):
    """raman dataset 
    Arguments:
        ds: dataset containing path to images and their respective labels
        split_idx: indices of samples to be used
        channel: list of channel names used 
    """
    def __init__(self, 
                ds, 
                split_idx, channel=[]):
        scaler = StandardScaler()
 
        ds = np.array(ds)
        ds = np.expand_dims(ds, axis=1)
        dsOri = ds.copy()
        for i in range(len(channel)):
            ds = np.concatenate((ds,dsOri),axis=1)
        dsTemp = ds.copy(order='C')

        
        for i in range(ds.shape[0]):
            j=1

            if 'jbcd' in channel:
                ds[i,j,1:] = morphological.jbcd(dsTemp[i,0,1:])[1]['signal']
                j+=1
            if 'beads' in channel:
                ds[i,j,1:] = misc.beads(dsTemp[i,0,1:])[1]['signal']
                j+=1
            if 'pspline_aspls' in channel:
                ds[i,j,1:] = ds[i,0,1:]-spline.pspline_aspls(dsTemp[i,0,1:])[0]
                j+=1
            if 'ria' in channel:
                ds[i,j,1:] = ds[i,0,1:]-smooth.ria(dsTemp[i,0,1:])[0]
                j+=1
            if 'fabc' in channel:
                ds[i,j,1:] = ds[i,0,1:]-classification.fabc(dsTemp[i,0,1:])[0]
                j+=1
            if 'adaptive_minmax' in channel:
                ds[i,j,1:] = ds[i,0,1:]-optimizers.adaptive_minmax(dsTemp[i,0,1:])[0]
                j+=1
            if 'goldindec' in channel:
                ds[i,j,1:] = ds[i,0,1:]-polynomial.goldindec(dsTemp[i,0,1:])[0]
                j+=1
            # if 'loess' in channel:
            #     ds[i,j,1:] = ds[i,0,1:]-polynomial.loess(dsTemp[i,0,1:])[0]
            #     j+=1
            # if 'mpspline' in channel:
            #     ds[i,j,1:] = ds[i,0,1:]-morphological.mpspline(dsTemp[i,0,1:])[0]
            #     j+=1
            # if 'golotvin' in channel:
            #     ds[i,j,1:] = ds[i,0,1:]-classification.golotvin(dsTemp[i,0,1:])[0]
            #     j+=1
            # if 'aspls' in channel:
            #     ds[i,j,1:] = ds[i,0,1:]-whittaker.aspls(dsTemp[i,0,1:])[0] 
            #     j+=1
            # if 'derpsalsa' in channel:
            #     ds[i,j,1:] = ds[i,0,1:]-whittaker.derpsalsa(dsTemp[i,0,1:])[0] 
            #     j+=1
        
        for n in range(ds.shape[1]):
            ds[:,n,1:] = scaler.fit_transform(ds[:,n,1:])

        ds = ds[split_idx, :,:]
     
        self.ds = ds  


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index): 
        x = self.ds[index,:,1:]
        y = int(self.ds[index,0,0])

        return x,y
