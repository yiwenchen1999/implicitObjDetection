import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

def oct_seg(image, n_segments):
    seg = pow(2,n_segments)
    print("shape:", image.shape, type(image))
    w = image.shape[0]/seg
    h = image.shape[1]/seg
    mask = np.ones((w,h))
    image = np.zeros(seg.shape)