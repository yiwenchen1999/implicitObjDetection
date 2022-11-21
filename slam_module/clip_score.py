import os
import torch
import numpy as np
import clip
from PIL import Image
import string


def calculate_clip_loss(image_token, text_token, model):
    with torch.no_grad():
        # print(type(model))
        image_features =torch.nn.functional.normalize(model.encode_image(image_token))
        image_features =(model.encode_image(image_token))
        # print((image_features[0][0:10]))
        text_features = torch.nn.functional.normalize((model.encode_text(text_token)))
        text_features =(model.encode_text(text_token))
        # print((text_features[0][0:10]))
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        dist = cos(image_features,text_features) 
        print(dist)
    return dist

def evaluateFrame(frame,text,model):
    image = frame
    score = calculate_clip_loss(image, text, model)
    return score.clone().cpu()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    input_folder = 'test_render_ours'
    file = "Datasets/Replica/office1/results/frame000637.jpg"
    input = preprocess(Image.open(file)).unsqueeze(0).to(device)
    text = ["a table", "a computer", "a forest","a plant", "a bed", "a whiteboard"]
    evaluateFrame(input, clip.tokenize(text).to(device),model_clip)
