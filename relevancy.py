import argparse
import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
from models.slic_vit import SLICViT
from models.ss_baseline import SSBaseline
from models.resnet_high_res import ResNetHighRes
from utils.zsg_data import FlickrDataset, VGDataset
from utils.grounding_evaluator import GroundingEvaluator
from PIL import Image
import os




def getHeatmap(model, im, text):
    _, heatmap  = model(im, text)
    return heatmap


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='vit14')
parser.add_argument('--num_samples', type=int,
                    default=500)  # 0 to test all samples
parser_args = parser.parse_args()


if parser_args.model == 'vit14':
    model = SLICViT
elif parser_args.model == 'rn50':
    model = ResNetHighRes
elif parser_args.model == 'ssbaseline':
    model = SSBaseline
    args = {}
else:
    assert False

model = model(**args).cuda()

if __name__=='__main__':
    path = '/gpfs/data/ssrinath/ychen485/implicitSearch/adaptingCLIPtesting/toybox-13/0/'
    directories = os.listdir( path )
    for filename in directories:
        if filename[0:4] == 'rgba':
            img_path = path + filename
            im = np.array(Image.open(img_path).convert("RGB"))
            heatmap = getHeatmap(model, im , "chair")
            heatimg = heatmap*800
            # print(heatimg)
            o_im = Image.fromarray(im).convert ('RGB')
            h_im = Image.fromarray(heatimg).convert ('RGB')
            o_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/adaptingCLIPtesting/output0/"+filename+".png")
            h_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/adaptingCLIPtesting/output0/"+filename+"_heat.png")
            print(filename+" saved")
