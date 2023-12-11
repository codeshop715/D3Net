import argparse
import datetime
import itertools
import math
import os
import random
import sys
import time
from math import log10
import ssl
import torchvision.transforms as T
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.dataloader import *
from models.base_model import * 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import cv2
from matplotlib import pyplot as plt
import setproctitle
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor
from PIL import Image
from utils.metrics import *
import pandas as pd
setproctitle.setproctitle("python")


def image_to_tensor(image):
    return ToTensor()(image).unsqueeze(0)

parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser.add_argument("--image_path", type=str,default='./input_test/', help="Path to image")
parser.add_argument("--save_path", type=str,default='./output', help="test to save")
parser.add_argument('-e',"--epoch", type=str,default=96, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--ADB_blocks", type=int, default=12, help="number of ADB blocks in the generator")
parser.add_argument("--concat", type=str, default='no', help="Compare inputs and outputs")
parser.add_argument("--img_width", type=int, default=128, help="img_width")
parser.add_argument("--img_height", type=int, default=128, help="img_height")
parser.add_argument("--model_folder", type=str, default='testModels', help="store folder name of trained model")
parser.add_argument("--target_data_dir", type=str, default='empty', help="target_data_dir")

opt = parser.parse_args()
print(opt)
img_width=opt.img_width
img_height=opt.img_height
image_path = opt.image_path
target_data_dir = opt.target_data_dir
save_path=opt.save_path
model_folder=opt.model_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define model and load model checkpoint
model=D3Net(3, 3, opt.ADB_blocks)
generator=model.cuda()
 
generator_checkpoint = torch.load(f"{model_folder}/generator_{opt.epoch}.pth")
      
generator.load_state_dict(generator_checkpoint['net'])  #load the generator

generator.eval()

for param in generator.parameters():
    param.requires_grad = False


ensure_size = RescaleIfSmallerThan(opt.img_height, opt.img_width)
random_crop = transforms.RandomCrop((opt.img_height, opt.img_width))
        

transforms_list = [
  ensure_size,       # Ensure image size
  random_crop,       # Random cropping
]

gt_transform = FixedSeedTransform(transforms.Compose(
  transforms_list + [
  transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize(mean, std), 
  ]
))

input_transform = FixedSeedTransform(transforms.Compose(
    transforms_list + [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ]
))

# Prepare input
test_set=sorted(glob.glob(opt.image_path+'/*.*'))

# print(opt.image_path+'*.*')
pic_number = len(test_set)

i=0
time_start = time.time()

psnr_total = 0.0
ssim_total = 0.0
num_samples = 0

fileList = os.listdir(image_path)

print('pic_number:  '+str(pic_number))
for idx, image_name in enumerate(fileList):

    test_data_dir = os.path.join(image_path, image_name)  # Degradation image path
    target_image_path = os.path.join(target_data_dir, image_name)  # Target Image Path
    
    print('test_data_dir:  '+str(test_data_dir))
       
    seed = torch.random.seed()
    image_tensor = Variable(input_transform(Image.open(test_data_dir), seed)).to(device).unsqueeze(0)
    target_tensor = Variable(gt_transform(Image.open(target_image_path), seed)).to(device).unsqueeze(0)


    with torch.no_grad():
        output_tensor = generator(image_tensor)
        output_tensor = torch.add(image_tensor, output_tensor) 
    
    # Save image
    image_tensor = denormalize(image_tensor)
    output_tensor = denormalize(output_tensor)
    target_tensor = denormalize(target_tensor)
    
    # Calculating PSNR and SSIM
    ssim_value = calc_ssim(output_tensor, target_tensor)
    fn = image_path.split("/")[-2]+"/"+image_path.split("/")[-1]
    
    # Convert output from tensor to image
    output_image = to_pil_image(output_tensor.squeeze(0))
    target_image = to_pil_image(target_tensor.squeeze(0))

    # Convert images to NumPy arrays
    output_np = np.array(output_image) 
    target_np = np.array(target_image)

    output_np = cv2.cvtColor(output_np, cv2.COLOR_BGR2YCrCb)
    target_np = cv2.cvtColor(target_np, cv2.COLOR_BGR2YCrCb)
    
    psnr = calc_psnr(output_np, target_np, 10)


    psnr_total += psnr
    ssim_total += ssim_value
    num_samples += 1

    #save images
    img_grid = torch.cat((image_tensor, output_tensor), -1)
    img_grid = torch.cat((img_grid, target_tensor), 3)
    os.makedirs (f"{save_path}_concat/"+ image_path.split("/")[-2],exist_ok=True)
        
    try:
        save_image(img_grid, f"{save_path}_concat/"+ image_path.split("/")[-2]+f"/{image_name}",normalize=False)
    except Exception as e:
            print("Error:", e)
     
    average_psnr = psnr_total / num_samples
    average_ssim = ssim_total / num_samples
    print(f"Result (dynamic update): ------ PSNR: {average_psnr:.2f} dB, SSIM: {average_ssim:.4f}-------")
    
torch.cuda.synchronize()
time_end = time.time()
time_sum = time_end - time_start
print(str(pic_number)+"pictures ,avg time:"+str(time_sum/pic_number)+"s")
print(save_path)