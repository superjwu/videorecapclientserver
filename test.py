import torch
from torchvision import models, transforms
import os
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from moviepy.editor import *
#from transformers import AutoTokenizer
#from tqdm import tqdm
#import argparse

from src.data.video_transforms import Permute
from src.models.video_recap import VideoRecap
from src.data.datasets import VideoCaptionDataset, CaptionDataCollator
from src.models.timesformer import SpaceTimeTransformer
#from src.models.openai_model import QuickGELU
#from src.configs.defaults import defaultConfigs
#from PIL import Image
#import pickle
#from io import BytesIO

import requests
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import torch
from torchvision import models, transforms
import os
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from moviepy.editor import *
#from transformers import AutoTokenizer
#from tqdm import tqdm
#import argparse

from src.data.video_transforms import Permute
from src.models.video_recap import VideoRecap
from src.data.datasets import VideoCaptionDataset, CaptionDataCollator
from src.models.timesformer import SpaceTimeTransformer
#from src.models.openai_model import QuickGELU
#from src.configs.defaults import defaultConfigs
#from PIL import Image
#import pickle
#from io import BytesIO

import requests
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
# Create model and tokenizer
ckpt_path = 'pretrained_models/videorecap/videorecap_clip.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
old_args = ckpt['args']
old_args.video_feature_type = 'pixel'  
old_args.num_video_feat=4                     # number of frames per clip caption
crop_size = 224
transform = transforms.Compose([
        Permute([3, 0, 1, 2]),  # T H W C -> C T H W
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]),
    ])
#tokenizer = AutoTokenizer.from_pretrained(old_args.decoder_name)
state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    state_dict[k.replace('module.', '')] = v

print("=> Creating model")
model = VideoRecap(old_args, eval_only=True)
model = model.cuda()
model.load_state_dict(state_dict, strict=True)
print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))
# Create dataset from the video
video = VideoFileClip(video_file)
print('Video length', video.duration, 'seconds')

video_length = video.duration
caption_duration = 4                              # Extract clip caption at each 4 seconds
old_args.video_loader_type='moviepy'
old_args.chunk_len = -1                           # load from raw video
old_args.video_feature_path = 'assets'            # path to the video folder 
metadata = []  
for i in np.arange(0, video_length, caption_duration):
    metadata.append([vid, i, min(i + caption_duration, video_length)])    # video name is example.mp4 so assuming video id=example
print('number of captions', len(metadata))
print(metadata)

old_args.metadata = metadata
dataset = VideoCaptionDataset(old_args, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, 
                                        num_workers=8, pin_memory=True, drop_last=False)
print(len(dataset), len(data_loader))
print(data_loader)
jj = 0
with torch.no_grad():
    for data_iter, samples in enumerate(data_loader):
        indices = samples['index']
        if hasattr(model, "vision_model"):
            image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().cuda()  # BCTHW -> BTCHW
            samples["video_features"] = model.vision_model.forward_features(image, use_checkpoint=old_args.use_checkpoint, cls_at_last=False)  # NLD
            #print(samples["video_features"])
            
            # Get tensor and its shape
            tensor = samples["video_features"]
            tensor_shape = tensor.shape

            # Serialize the tensor to bytes and then encode it in base64 to make it JSON-compatible
            tensor_bytes = tensor.cpu().numpy().tobytes()
            tensor_base64 = base64.b64encode(tensor_bytes).decode('utf-8')  # base64 encoded string

            # Get tensor and its shape
            index_tensor = samples['index']
            index_tensor_shape = index_tensor.shape

            # Serialize the tensor to bytes and then encode it in base64 to make it JSON-compatible
            index_tensor_bytes = index_tensor.cpu().numpy().tobytes()
            index_tensor_base64 = base64.b64encode(index_tensor_bytes).decode('utf-8')  # base64 encoded string 
            
            # Prepare the data to send
            data = {
                "feature_tensor": tensor_base64,
                "feature_tensor_shape": list(tensor_shape),
                "index_tensor": index_tensor_base64,
                "index_tensor_shape": list(index_tensor_shape),
                "jj": jj
            }

            #Send the tensor and shape to the server in the body of a POST request
            response = requests.post(
                "http://127.0.0.1:8000/upload_tensor",
                json=data  # send as JSON body
            )

            print(response.json())
            jj+=1
        