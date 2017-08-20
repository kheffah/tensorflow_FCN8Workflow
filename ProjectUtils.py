#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:08:16 2017

@author: mohamedt

Workflow utilities
"""

import os
import numpy as np
import logging
import subprocess

#%%==============================================================================
# Determine if current device is a GPU device
#==============================================================================
def isGPUDevice():
    '''
    Determine if device is an NVIDIA GPU device
    '''  
    
    success = os.system("nvidia-smi")
    
    return success == 0

#%%============================================================================
# GPU allocation management - restrict to set number
#==============================================================================

def AllocateGPU(N_GPUs = 2, N_trials=0):    
    
    '''
    Restricts GPU use to a set number based on memory use
    '''
    
    # only restrict if not a GPU machine or already restricted
    isGPU = isGPUDevice()
    
    try:
        AlreadyRestricted = os.environ["CUDA_VISIBLE_DEVICES"] is not None
    except KeyError:
        AlreadyRestricted = False
    
    if isGPU and (not AlreadyRestricted):
        
        try:
            print("Restricting GPU use to {} GPUs ...".format(N_GPUs))
            
            # Get processes from nvidia-smi command
            gpuprocesses = str(subprocess.check_output("nvidia-smi", shell=True))\
                              .split('\\n')
            # Parse out numbers, representing GPU no, PID and memory use
            gpuprocesses = gpuprocesses[24:len(gpuprocesses)-2]
            gpuprocesses = [j.split('MiB')[0] for i,j in enumerate(gpuprocesses)]
            
            PIDs = ['',]
            
            for p in range(len(gpuprocesses)):
                pid = [int(s) for s in gpuprocesses[p].split() if s.isdigit()]
                PIDs.append(pid)
                
            PIDs.pop(0)
            PIDs = np.array(PIDs)
            
            # Add "fake" zero-memory processes to ensure all GPUs are represented
            extrapids = np.zeros([4, 3])
            extrapids[:,0] = np.arange(4)
            PIDs = np.concatenate((PIDs, extrapids), axis=0)
            
            # Get GPUs memory consumption
            memorycons = np.zeros([4,2])
            for gpuidx in range(4):
                thisgpuidx = 1 * np.array(PIDs[:,0] == gpuidx)
                thisgpu = PIDs[thisgpuidx==1, :]
                memorycons[gpuidx,0] = gpuidx
                memorycons[gpuidx,1] = np.sum(thisgpu[:,2])
                
            # sort and get GPU's with lowest consumption
            memorycons = memorycons[memorycons[:,1].argsort()]
            GPUs_to_use = list(np.int32(memorycons[0:N_GPUs,0]))
            
            # Now restrict use to available GPUs
            GPUs_to_use = str(GPUs_to_use).split('[')[1].split(']')[0]
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = GPUs_to_use
                      
            print("Successfully Restricted GPU use to GPUs: "+ GPUs_to_use)
        
        except ValueError:
            
            if N_trials < 10:
                print("Got value error, trying again ...")
                N = N_trials + 1
                AllocateGPU(N_GPUs=N_GPUs, N_trials=N)
            else:
                raise ValueError(\
                "Something is wrong, tried too many times and failed.")
                  
    else:
        if isGPU:
            print("No GPU allocation done.")
        if AlreadyRestricted:
            print("GPU devices already allocated.")

#%%============================================================================
# Log and print
#==============================================================================

def Log_and_print(Msg, loglevel="info", Monitor=True):
    
    """Logs information +/- prints it to screen"""
    
    if loglevel == "debug":
        logging.debug(Msg)
    elif loglevel == "info":
        logging.info(Msg)
    elif loglevel == "warning":
        logging.warning(Msg)
    elif loglevel == "error":
        logging.error(Msg)
    elif loglevel == "critical":
        logging.critical(Msg)
    
    if Monitor:
        print(Msg)
        
#%%============================================================================
# Create a subdirectory        
#==============================================================================

def makeSubdir(dirPath, dirName):
    
    """creates a subdirectory if non-existent"""
    
    dirs = os.listdir(dirPath)
    if dirName not in dirs:
        success = os.system('mkdir ' + dirPath + dirName)
        if success != 0:
            raise Exception("Failed to create output subdirectory, " + \
                            "check user permissions!")
            
#%%============================================================================
# Merge default and given dict
#==============================================================================

def Merge_dict_with_default(dict_given, dict_default, keys_Needed=[]):
    
    """Sets default values of dict keys not given"""
    
    keys_default = list(dict_default.keys())
    keys_given = list(dict_given.keys())
    
    # Optional: force user to unput some keys (eg. those without defaults)
    if len(keys_Needed) > 0:
        for j in keys_Needed:
            if j not in keys_given:
                raise KeyError("Please provide the following key: " + j)
    
    keys_Notgiven = [j for j in keys_default if j not in keys_given]
    
    for j in keys_Notgiven:
        dict_given[j] = dict_default[j]
        
    return dict_given


#%%============================================================================
# Test methods
#==============================================================================

if __name__ == '__main__':
    
    print("isGPUDevice: " + str(isGPUDevice()))
    AllocateGPU(N_GPUs=2)
    
