from tkinter import N
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def seg(image, n_segments, window_size = 5):
    seg = pow(2,n_segments)
    # print("shape:", image.shape, type(image))
    w = int(image.shape[0]/seg)
    h = int(image.shape[1]/seg)
    # print("w,h:",w,h)
    window_size = window_size
    r  = int((window_size-1)/2)
    plate = np.zeros((image.shape[0],image.shape[1]))
    areas = []
    n = 0
    for i in range(seg):
        for j in range(seg):
            if n != 0:
                mask = n*np.ones((w,h))
                plate[i*w:(i+1)*w, j*h:(j+1)*h] = mask
            canvas = np.zeros((image.shape[0],image.shape[1]))
            # for x in range(window_size):
                # for y in range (window_size):
            canvas[max(0,(i-r)*w):min(image.shape[0],(i+r+1)*w), max(0,(j-r)*h):min(image.shape[1],(j+r+1)*h)] = n
            # print("plate:")
            # print(plate)
            # print("canvas:")
            # print(canvas)
            areas.append(canvas)
            n = n + 1

    return plate, areas

def segPerPixel(image, window_size = 5):
    w = image.shape[0]
    h = image.shape[1]
    window_size = window_size
    r  = int((window_size-1)/2)
    areas = []
    for i in range(w):
        for j in range(h):
            canvas = np.zeros((image.shape[0],image.shape[1]))
            canvas[max(0,i-r):min(w, i+r), max(0, j - r): min(h, j + r)] = 1
            areas.append(canvas)
    return areas

def cropPerPixel(image, windowsize = 5):
    w = image.shape[0]
    h = image.shape[1]
    window_size = window_size
    r  = int((window_size-1)/2)
    areas = []
    for i in range(w):
        for j in range(h):
            canvas = np.zeros((image.shape[0],image.shape[1]))
            canvas = image.crop((max(0,i-r), j - r, min(w, i+r),  min(h, j + r)))
            # canvas[max(0,i-r):min(w, i+r), max(0, j - r): min(h, j + r)] = 1
            areas.append(canvas)
    return areas
