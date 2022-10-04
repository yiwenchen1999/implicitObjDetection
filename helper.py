from tkinter import N
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def seg(image, n_segments):
    seg = pow(2,n_segments)
    print("shape:", image.shape, type(image))
    w = int(image.shape[0]/seg)
    h = int(image.shape[1]/seg)
    print("w,h:",w,h)
    
    plate = np.zeros(image.shape)
    n = 0
    for i in range(seg):
        for j in range(seg):
            if n != 0:
                mask = n*np.ones((w,h))
                print("mask")
                print(mask)
                plate[i*w:(i+1)*w, j*h:(j+1)*h] = mask
                print("plate:")
                print(plate)
