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
    #cups schemes/
    # data_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/mug1/"
    # root_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/mug1"
    # data_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/test_clip/cups/"
    # root_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/test_results/cups/"
    data_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/room_nerf/logs/seg_all_clip/renderonly_path_143999/"
    root_path = "/gpfs/data/ssrinath/ychen485/implicitSearch/implicitObjDetection/room_nerf/logs/seg_all_clip/renderonly_path_143999/"


    def save_query(text, image_clip_feature_normalized):
        query_map = model.verify(image_clip_feature_normalized, text, root_path)
        query_map_scores = np.squeeze(query_map)
        # max = np.max(query_map)
        # min = np.min(query_map)
        # print(filename+" max score: "+str(max) + ", min score: "+str(min))
        query_map_remapped = (query_map_scores - np.min(query_map_scores)) / (np.max(query_map_scores) - np.min(query_map_scores))
        np.save(root_path + filename[:-4]+ text, query_map_remapped)
        query_map = query_map.reshape(query_map.shape[0], query_map.shape[1])
        plt.imshow(query_map, cmap = 'plasma')
        # plt.imshow(query_map_3d)
        plt.imsave(root_path + filename[:-4]+ text + "_heat.png", query_map)



    directories = os.listdir(data_path)
    for filename in directories:
        # if filename[0:4] == 'rgba':
        if filename[-4:] == '.npy'and filename[3] == '.':
        # if True:
            clip_path = data_path + filename
            feature_map = np.load(clip_path)
             #image_clip_feature's size is torch.Size([1, 768, 1])
            image_clip_feature_normalized = torch.from_numpy(feature_map)
            print(image_clip_feature_normalized.shape, image_clip_feature_normalized.size)
            print(filename+"loaded")
            # query_map = model.verify(image_clip_feature_normalized, "a chair", root_path).cpu().float().numpy()


            save_query("the mic", image_clip_feature_normalized)
            save_query("stand", image_clip_feature_normalized)
            save_query("wires", image_clip_feature_normalized)
            save_query("head", image_clip_feature_normalized)
            # save_query("the dog", image_clip_feature_normalized)
            # save_query("the ears", image_clip_feature_normalized)
            # save_query("the head", image_clip_feature_normalized)
            # save_query("legs", image_clip_feature_normalized)
            # save_query("legs of a chair", image_clip_feature_normalized, 3)
            # save_query("back of a chair", image_clip_feature_normalized, 3)
            # save_query("swivel chair", image_clip_feature_normalized, 3)
            # save_query("a chair", image_clip_feature_normalized, 3)
            # # image_id = "00080"
            # # image_clip_feature_normalized = torch.tensor(np.load(root_path + image_id + "_image_clip_feature.npy")).cuda() #[256, 256, 768]
            # #print(image_clip_feature_normalized)
            # #image_clip_feature_normalized = (image_clip_feature_normalized - torch.unsqueeze(torch.min(image_clip_feature_normalized, dim = -1)[0], dim = -1)) / (torch.unsqueeze(torch.max(image_clip_feature_normalized, dim = -1)[0], dim = -1) - torch.unsqueeze(torch.min(image_clip_feature_normalized, dim = -1)[0], dim = -1))
            # query_map = model.verify(image_clip_feature_normalized, "the handle", root_path).cpu().float().numpy()
            # # #plt.imshow(query_map)
            # # #plt.show()
            # query_map_scores = np.squeeze(query_map)
            # max = np.max(query_map)
            # min = np.min(query_map)
            # print(filename+" max score: "+str(max) + ", min score: "+str(min))
            # # query_map_remapped = (query_map - np.min(query_map)) / (np.max(query_map) - np.min(query_map))
            # # r,c = np.shape(query_map_remapped)
            # # query_map_3d = np.zeros((r,c,3))
            # # query_map_3d[:,:,0] = query_map_remapped
            # # query_map_3d[:,:,1] = query_map_remapped
            # # query_map_3d[:,:,2] = query_map_remapped

            # # query_map = query_map.cpu().detach().numpy()
            # query_map = query_map.reshape(query_map.shape[0], query_map.shape[1])
            # plt.imshow(query_map)
            # # plt.imshow(query_map_3d)
            # plt.imsave(root_path + filename[:-4]+"_heat.png", query_map)
            # exit(0)
            
            
            
