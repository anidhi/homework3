'''
    
        We are using this file to do some "behind-the-scenes" work. You do NOT need to modify or understand the code in this file. 
        
        CS 475: Intro to Machine Learning 
        Spring 2020 
        
        Modified 03/04/2020 by Molly O'Brien
    

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import torch 
from torch.autograd import Variable 

''' Part 1: Gradient Descent and Non-Convex Objective Functions '''
def loss(w, Torch=False):
    ''' compute the loss using our current weight values '''
    omega_1 = 5
    omega_2 = 2
    if(not Torch):
        err = (2*np.sin(w[0]*omega_1)*np.cos(w[1]/100.0) + 4*np.cos(w[1]*omega_2))*((w[0]+w[1])/(w[0]**2 + w[1]**2 + 100))
    else: 
        err = (2*torch.sin(w[0]*omega_1)*torch.cos(w[1]/100.0) + 4*torch.cos(w[1]*omega_2))*((w[0]+w[1])/(w[0]**2 + w[1]**2 + 100))
    return err 


def part1():
    Points = []
    for X in range(-30, 30):
        x = X/10.0
        for Y in range(-25, 30):
            y = Y/10.0
            z = loss([x, y])
            Points.append([x, y, z])
            
    return np.array(Points)

def plot_part1(Points, fig):
    ax = Axes3D(fig)

    ax.plot_trisurf(Points[:, 0], Points[:, 1], Points[:, 2], cmap=cm.get_cmap('plasma'))
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('Loss(w1, w2)')
    return fig