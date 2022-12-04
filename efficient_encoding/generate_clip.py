import numpy as np
from models.image_clip import Image_CLIP
from models.slic_vit import SLICViT
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt


if __name__=='__main__':
    args = {
        'model': 'vit14',
        'alpha': 0.75,
        'aggregation': 'mean',
        'n_segments': [5],
        'temperature': 0.02,
        'upsample': 2,
        'start_block': 0,
        'compactness': 50,
        'sigma': 0,
    }
    model = SLICViT(**args).cuda()

    #root_path = '/users/aren10/data/'
    #data_path = root_path + '0/'
    # root_path = '/gpfs/data/ssrinath/ychen485/implicitSearch/room_0/Sequence_2/rgb/'
    # # data_path = "/gpfs/data/ssrinath/ychen485/toybox-13/0/"
    #replica dataset
    # data_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/room_0/Sequence_2/rgb/"
    # root_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/results/replica/room0/frames/"
    #cups schemes
    data_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/cups/"
    root_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/cups_score/"


    directories = os.listdir(data_path)
    for filename in directories:
        # if filename[0:4] == 'rgba':
        if True:
            img_path = data_path + filename
            im = np.array(Image.open(img_path).convert("RGB")) #im shape is (256, 256, 3)
            o_im = Image.fromarray(im).convert ('RGB')
            o_im.save(root_path + filename)
            image_clip_feature = torch.tensor(model.get_clipmap(im)) #image_clip_feature's size is torch.Size([1, 768, 1])
            image_clip_feature_normalized = image_clip_feature
            np.save(root_path + filename[:-4]+"_image_clip_feature", image_clip_feature_normalized)
            print(filename+" saved")

            
            # image_id = "00080"
            # image_clip_feature_normalized = torch.tensor(np.load(root_path + image_id + "_image_clip_feature.npy")).cuda() #[256, 256, 768]
            #print(image_clip_feature_normalized)
            #image_clip_feature_normalized = (image_clip_feature_normalized - torch.unsqueeze(torch.min(image_clip_feature_normalized, dim = -1)[0], dim = -1)) / (torch.unsqueeze(torch.max(image_clip_feature_normalized, dim = -1)[0], dim = -1) - torch.unsqueeze(torch.min(image_clip_feature_normalized, dim = -1)[0], dim = -1))
            query_map = model.verify(image_clip_feature_normalized, "the handle of a cup", root_path).cpu().float().numpy()
            # #plt.imshow(query_map)
            # #plt.show()
            query_map_scores = np.squeeze(query_map)
            max = np.max(query_map)
            min = np.min(query_map)
            print(filename+" max score: "+str(max) + ", min score: "+str(min))
            # query_map_remapped = (query_map - np.min(query_map)) / (np.max(query_map) - np.min(query_map))
            # r,c = np.shape(query_map_remapped)
            # query_map_3d = np.zeros((r,c,3))
            # query_map_3d[:,:,0] = query_map_remapped
            # query_map_3d[:,:,1] = query_map_remapped
            # query_map_3d[:,:,2] = query_map_remapped

            # query_map = query_map.cpu().detach().numpy()
            query_map = query_map.reshape(query_map.shape[0], query_map.shape[1])
            plt.imshow(query_map)
            # plt.imshow(query_map_3d)
            plt.imsave(root_path + filename[:-4]+"_heat.png", query_map)
            # exit(0)
            
            
            
