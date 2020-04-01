'''
        
        CS 475 Spring 2020 
        Molly O'Brien
        
        Helper functions to read in the JIGSAWS data 
        
'''
import os
import sys
import pdb
import numpy as np

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_cnn_window(data, labels, window_size, batch_size):
    ''' split temporal input data of variable length into windows of fixed legth '''
    window_data   = np.zeros((batch_size, 6, window_size))
    window_labels = np.zeros((batch_size, 1))
    
    for row in range(batch_size):
        idx = np.random.randint(len(data))
        t = np.random.randint(len(data[idx]) - window_size)
        window_data[row, :, :] = data[idx][t:t+window_size, :].T
        window_labels[row] = labels[idx]
    
    return window_data, window_labels

def get_window(data, labels, window_size, batch_size):
    ''' split temporal input data of variable length into windows of fixed legth '''
    window_data   = np.zeros((batch_size, 6*window_size))
    window_labels = np.zeros((batch_size, 1))
    
    for row in range(batch_size):
        idx = np.random.randint(len(data))
        t = np.random.randint(len(data[idx]) - window_size)
        window_data[row, :] = data[idx][t:t+window_size, :].flatten()
        window_labels[row] = labels[idx]
    
    return window_data, window_labels


def read_jigsaws_data(path):
    ''' Read in the left master tool tip xyz positions and the right master tool tip xyz positions. 
    
        Args: 
            path: path to the JIGSAWS Suturing folder (the readme.txt should be in this directory)
        Returns     data: a list. Each list item is a Tx6 ndarray, where T is the number of frames in the trial,
                    labels: binary skill labels. 0: novice surgeon. 1: intermediate or expert surgeon, 
                    names: a list of file names.
            
            We separate the train/test data in this function so the data returned is 
            
            * Train: [train_data, train_labels, train_names]
            * Test: [test_data, test_labels, test_names]
            
            where train_*** or test_*** is a list described above. 
            
    '''
    # columns with left master tool tip xyz, 0:3
    left  = [0, 3]
    # columns with right master tool tip xyz, 19:22
    right = [19, 22]
    
    # add the file names to this list
    names = []
    # add the tool tip positions to this list
    data  = []

    kin_root = os.path.join(path, 'Suturing', 'kinematics', 'AllGestures')
    for fil in os.listdir(kin_root):
        if(fil == 'Suturing_H002.txt'):
            # we skip this file bc it doesn't have a skill label
            pass
        else:
            dat = read_jigsaws_file(os.path.join(kin_root, fil)).T
            tool_tip = np.concatenate([dat[:, left[0]:left[1]], dat[:, right[0]:right[1]]], axis=1)
            names.append(fil.split('.')[0])
            data.append(tool_tip)
        
    # read in the meta data
    meta_path = os.path.join(path, 'Suturing', 'meta_file_Suturing.txt')
    meta_data = read_file(meta_path, delimiter='/t', num_flag=False, header_flag=False)
    
    
    # find the skill labels for each trial
    labels = []
    for fil in names:
        row = np.where(meta_data[:, 0] == fil)[0][0]
        labels.append(int(not(meta_data[row, 2]=='N')))
    
    
    Train_Names = ['Suturing_B001',
                     'Suturing_B002',
                     'Suturing_B003',
                     'Suturing_B004',
#                      'Suturing_B005',
                     ##'Suturing_C001',
                     'Suturing_C002',
                     'Suturing_C003',
                     'Suturing_C004',
#                      'Suturing_C005',
                     'Suturing_D001',
                     'Suturing_D002',
                     'Suturing_D003',
                     'Suturing_D004',
#                      'Suturing_D005',
                     'Suturing_E001',
                     'Suturing_E002',
                     'Suturing_E003',
#                      'Suturing_E004',
#                      'Suturing_E005',
                     'Suturing_F001',
                     'Suturing_F002',
                     'Suturing_F003',
                     'Suturing_F004',
#                      'Suturing_F005',
                     'Suturing_G001',
                     'Suturing_G002',
                     'Suturing_G003',
                     'Suturing_G004',
#                      'Suturing_G005',
                     'Suturing_H001',
                     'Suturing_H003',
#                      'Suturing_H004',
#                      'Suturing_H005',
                     'Suturing_I001',
                     'Suturing_I002',
                     'Suturing_I003',
                     'Suturing_I004']
#                      'Suturing_I005']
    
   
    
    ''' split data into training and testing '''
    train_data = [];     train_labels = [];     train_names = []
    test_data  = [];     test_labels  = [];     test_names  = []
    
    for i in range(len(data)):
        if(names[i] in Train_Names):
            train_data.append(data[i])
            train_labels.append(labels[i])
            train_names.append(names[i])
        else:
            test_data.append(data[i])
            test_labels.append(labels[i])
            test_names.append(names[i])
        
    return [train_data, train_labels, train_names], [test_data, test_labels, test_names]
    
    

def read_file(path, delimiter, num_flag, header_flag):
    """
            given a path to a file, open it, split lines by delimiter. Save in an array

            num_flag: can we convert all the data into numbers?
            header_flag: are there column headers?
    """

    file_info = open(path, 'r')
    info_list = []
    # read off the 1st line with labels
    if(header_flag):
        file_info.readline()

    for line in file_info:
        line_parts = line.strip('\n').split(delimiter)
        float_parts = []
        for part in line_parts:
            if(num_flag):
                float_parts.append([np.float(pt) for pt in part.split('\t')])
            else:
                float_parts.append(part.strip('\n').split('\t'))
        info_list.append(float_parts)

    file_info.close()

    return np.squeeze(np.array(info_list))


def save_file(info, path, delimiter):
    """
            save all the data in info into a text file

            info: array with data
            path: file path
            delimiter: datum seperator
    """
    out = open(path, 'w')

    for row in range(info.shape[0]):
        line = ''
        N = info.shape[1]
        for col in range(N - 1):
            line = line + str(info[row, col]) + delimiter
        line = line + str(info[row, N-1]) + '\n'
        out.write(line)

    out.close()
    
# def read_jigsaws_data(in_path, out_path): 
#     ''' read in the kinematics '''
#     kin_path = os.path.join(path, 'kinematics')
    
#     kin_names = []
#     kin_data  = []
#     for fil in os.listdir(kin_path):
#         dat = read_file(os.path.join(kin_path, fil), '\t', num_flag=True, header_flag=False)
#         kin_names.append(fil.split('.')[0])
#         kin_data.append(dat)
    
#     ''' read in the data labels '''
#     meta_path = os.path.join(path, 'meta_file.txt')
#     meta_data = read_file(meta_path, delimiter='\t', num_flag=False, header_flag=False)

#     kin_labels = []
#     for fil in kin_names:
#         row = np.where(meta_data[:, 0] == fil)
#         kin_labels.append(meta_data[row, 1][0][0])
        
#     ''' return list of kinematics and '''
#     return kin_data, np.array(kin_labels)

#_________________________________________________________
#
# Function: readJIGSAWS_file
#
#			read in kinematic data from JIGSAWS data set
#
# input: 	* path: string 		path to kinematic file
# output: 	* data: array		76xNumFrames array
#				each column is the info from one time stamp
#				meaning of each row follows the input file format
#---------------------------------------------------------
def read_jigsaws_file(path):
	# open file from path
	kin_file = open(path, "r")
	# number of values per time stamp
	N = 76
	# declary array 76xT, each column is info for one time stamp
	kin_data = []
	# count number of lines in file
	line_count = 0

	for line in kin_file:
		# split line by white space
		split_line = line.split()
		# declare vector to store this time stamps' info
		kin_vect = np.zeros((N,1))
		# add the elements from the first line
		for idx in range(len(split_line)):
			try:
				kin_vect[idx, 0] = float(split_line[idx])
			except:
				print("ERROR converting " + split_line[idx] +" to a float.")

		# add kin_vect to kin_data
		kin_data.append(kin_vect)
		# increment line_count
		line_count = line_count + 1
	# move data into a 2D array
	data = np.zeros((N, line_count))
	for line in range(line_count):
		data[:, line:line+1] = kin_data[line]

	# return array of data
	return data


def compute_accuracy(data, labels, model, window_size):
    '''
        Compute the 
    '''
    model.eval()

    Predictions = [] 
    P = []
    for dat, lab in zip(data, labels): 
        predictions = []
        t = 0 
        while(t < len(dat) - window_size):
            win = torch.Tensor(dat[t:t+window_size,:].flatten())
            pred = model.predict(win.reshape(1, -1))
            #val = model(win.reshape(1, -1))
            predictions.append(pred.data.numpy())
            t += window_size

        Predictions.append(predictions)
        P.append((np.mean(predictions)>=0.5).astype(int))
        
    accuracy = np.mean(np.array(labels) == np.array(P))
    
    return accuracy, P

def compute_cnn_accuracy(data, labels, model, window_size):
    '''
        Compute the 
    '''
    model.eval()

    Predictions = [] 
    P = []
    for dat, lab in zip(data, labels): 
        predictions = []
        t = 0 
        while(t < len(dat) - window_size):
            win = torch.Tensor(dat[t:t+window_size,:].T)
            pred = model.predict(win.reshape(1, 6, window_size), 1)
            #val = model(win.reshape(1, -1), 1)
            predictions.append(pred.data.numpy())
            t += window_size

        Predictions.append(predictions)
        P.append((np.mean(predictions)>=0.5).astype(int))
        
    accuracy = np.mean(np.array(labels) == np.array(P))
    
    return accuracy, P



#=============================================
	#		----- Data Key -----
	# row in data
	# 1-3    (3) : Master left tooltip xyz
	# 4-12   (9) : Master left tooltip R
	# 13-15  (3) : Master left tooltip trans_vel x', y', z'
	# 16-18  (3) : Master left tooltip rot_vel
	# 19     (1) : Master left gripper angle
	# 20-22  (3) : Master right tooltip xyz
	# 23-31  (9) : Master right tooltip R
	# 32-34  (3) : Master right tooltip trans_vel x', y', z'
	# 35-37  (3) : Master right tooltip rot_vel
	# 38     (1) : Master right gripper angle
	# 39-41  (3) : Slave left tooltip xyz
	# 42-50  (9) : Slave left tooltip R
	# 51-53  (3) : Slave left tooltip trans_vel x', y', z'
	# 54-56  (3) : Slave left tooltip rot_vel
	# 57     (1) : Slave left gripper angle
	# 58-60  (3) : Slave right tooltip xyz
	# 61-69  (9) : Slave right tooltip R
	# 70-72  (3) : Slave right tooltip trans_vel x', y', z'
	# 73-75  (3) : Slave right tooltip rot_vel
	# 76     (1) : Slave right gripper angle
#_________________________________________________________