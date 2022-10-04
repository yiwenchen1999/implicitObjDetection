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
    mask = np.ones((w,h))
    plate = np.zeros(image.shape)