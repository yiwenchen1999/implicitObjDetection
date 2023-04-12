from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
from PIL import Image
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_clip_model, preprocess_img = clip.load("ViT-B/32", device=device)
pretrained_clip_model = pretrained_clip_model.to(device)
def show_mask(i, mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.savefig("mask_" + str(i))
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
def compute_logits(image_features, text_features):
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # cosine similarity as logits
    logit_scale = 100
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text


token = "all"
if token != "all":
    predictor = SamPredictor(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
    image = cv2.imread("notebooks/images/car2.jpg") 
    #image = cv2.imread("segment_anything/Reproject_CLIP/flickr8k/Images/667626_18933d713e.jpg") 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    if token == "point":
        input_point = np.array([[500, 375], [1125, 625]])
        input_label = np.array([1, 1])
        masks, scores, logits = predictor.predict_text(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(i, mask, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
    elif token == "text":
        input_text = np.array(["car"])
        masks, scores, logits = predictor.predict_text(
            texts = input_text,
            multimask_output=True,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(i, mask, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
    

def generate_trainSet_clip_imgFeatures(image_dic_path, output, mask_number_flag, scale_percent):
    if output == "mac":
        output_path = "output_data/"
    elif output == "linux":
        output_path = "/users/aren10/data/datasets/seg_all_2DCLIP_gt/"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    """
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    """
    image_dic = os.listdir(image_dic_path)
    for filename in image_dic:
        print(filename)
        #image = cv2.imread("notebooks/images/dog.jpg")  #image = cv2.imread("segment_anything/Reproject_CLIP/flickr8k/Images/667626_18933d713e.jpg") 
        image = cv2.imread(image_dic_path + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
        dim = (width, height)
        image = cv2.resize(image, dim)
        masks = mask_generator.generate(image) #can sort the masks by area, and disregard any mask smaller than certain area
        print(len(masks))
        clip2d_gt = torch.zeros((image.shape[0], image.shape[1], 512))
        clip2d_gt = clip2d_gt.to(device)
        if mask_number_flag == 0:
            mask_number = len(masks)
        else:
            mask_number = mask_number_flag
        for i in range(mask_number):
            print(i)
            #print(masks[i])
            start_time = time.time()
            masked_image = torch.zeros(image.shape)
            masked_image = torch.from_numpy(image) * torch.unsqueeze(torch.from_numpy(masks[i]["segmentation"]), -1) # multiply both by bbx XYWH
            masked_image = masked_image.type(dtype=torch.uint8)
            print(time.time() - start_time)
            #cv2.imwrite('image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            #cv2.imwrite('masked_image.png', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
            #clip image 1D vector
            start_time = time.time()
            masked_image = cv2.cvtColor(masked_image.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
            masked_image = Image.fromarray(masked_image)
            print(time.time() - start_time)
            start_time = time.time()
            masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
            print(time.time() - start_time)
            start_time = time.time()
            with torch.no_grad():
                masked_image_features = pretrained_clip_model.encode_image(masked_image)
            print(time.time() - start_time)
            #clip image 2D map
            start_time = time.time()
            clip2d_gt += (torch.unsqueeze(torch.from_numpy(masks[i]["segmentation"]).to(device), -1) * torch.unsqueeze(masked_image_features, 0))
            print(time.time() - start_time)
        torch.save(clip2d_gt.detach(), output_path + filename[:-4] + "_clip2d_gt.pt")
        clip2d_gt_partial = (clip2d_gt[:,:,0:3] * 255).type(dtype=torch.uint8)
        cv2.imwrite(output_path + filename[:-4] + "_clip2d_gt_partial.png", cv2.cvtColor(clip2d_gt_partial.detach().cpu().numpy(), cv2.COLOR_RGB2BGR))



#test: Top Left Corner is (0,0), xnview(y,x)
def test(load, file, image_features1_x, image_features1_y, image_features2_x, image_features2_y, text):
    if load == "mac":
        load_path = "output_data/" + file
    elif load == "linux":
        load_path = "/users/aren10/data/datasets/seg_all_2DCLIP_gt/" + file
    clip2d_gt = torch.load(load_path, map_location=torch.device('cpu'))
    print(clip2d_gt.shape)
    image_features1 = torch.unsqueeze(clip2d_gt[image_features1_x,image_features1_y], 0).to(torch.float32).to(device)
    image_features2 = torch.unsqueeze(clip2d_gt[image_features2_x,image_features2_y], 0).to(torch.float32).to(device)
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = pretrained_clip_model.encode_text(text)
        logits_per_image, logits_per_text = compute_logits(image_features1.float(), text_features.float())
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)
        logits_per_image, logits_per_text = compute_logits(image_features2.float(), text_features.float())
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)



if __name__ == "__main__":
    #generate_trainSet_clip_imgFeatures("notebooks/images/mic/train/", "linux", 0, 0.4)
    test("linux", "r_16_clip2d_gt.pt", 131, 168, 217,183, ["mic", "line", "iron stand"]) # Size = 4 * 800 * 800 * 512 = 1.31 Gb vs Size = 4 * 320 * 320 * 768 = 0.20 Gb













"""
plt.figure(figsize=(20,20))
plt.imshow(masks[0]["segmentation"])
plt.show()
"""
"""
plt.figure(figsize=(20,20))
plt.imshow(image)
#show_anns(masks)
plt.axis('off')
plt.show() 
"""
"""
# More deatiled masks
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks2 = mask_generator_2.generate(image)
len(masks2)
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show() 
"""
