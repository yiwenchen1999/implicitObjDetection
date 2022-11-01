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
    # _, heatmap  = model(im, text)
    heatmap = model.get_heatmap_perpixel(im, text, att = False)
    return heatmap

def getclipmap(model, im):
    clipmap = model.get_clipmap(im)
    return clipmap


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='vit14')
parser.add_argument('--num_samples', type=int,
                    default=500)  # 0 to test all samples
parser_args = parser.parse_args()


if parser_args.model == 'vit14':
    model = SLICViT
    args = {
        'model': 'vit14',
        'alpha': 0.75,
        'aggregation': 'mean',
        # 'n_segments': list(range(100, 601, 50)),
        'n_segments': [7],
        'temperature': 0.02,
        'upsample': 2,
        'start_block': 0,
        'compactness': 50,
        'sigma': 0,
    }

elif parser_args.model == 'rn50':
    model = ResNetHighRes
elif parser_args.model == 'ssbaseline':
    model = SSBaseline
    args = {}
else:
    assert False

model = model(**args).cuda()
# model = model(**args)

if __name__=='__main__':
    # print("runnning main fuction")
    path = '/gpfs/data/ssrinath/toybox-13/0/'
    # path = '/gpfs/data/ssrinath/ychen485/implicitSearch/NiceSlamTesting/Datasets/Demo/frames/color/'
    # path = '/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/nerf/data/nerf_synthetic/chair/train/'
    directories = os.listdir( path )
    i = 0
    for filename in directories:
        # if filename[5] == "." or filename[4] == "." or filename[3] == ".":
        if filename[0:4] == 'rgba':
        # if True:
            img_path = path + filename
            im = np.array(Image.open(img_path).convert("RGB"))
            # print("image shae inspection: ")
            # print(im.shape, im)

            heatmap = getHeatmap(model, im , "an air plane")
            heatimg = heatmap*200
            # print(heatimg)
            o_im = Image.fromarray(im).convert ('RGB')
            h_im = Image.fromarray(heatimg).convert ('RGB')
            o_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/perpixel3/"+filename)
            h_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/perpixel3/"+filename[:-4]+"_heat_testair51.png")
            np.save("/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/perpixel3/"+filename[:-4]+"_heat_testair51", heatmap)
            # h_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/outputChair/"+filename[:-4]+"_heat.png")
            # np.save("/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/outputChair/"+filename[:-4]+"_heat", heatmap)
            
            # clipmap = getclipmap(model, im)
            # print(filename+" clipmap has shape: ", clipmap.shape)
            print(filename+" saved")
            i = i + 1
            if i > 10:
                break
