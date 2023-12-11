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
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor
from PIL import Image
import pandas as pd
from utils.metrics import *
setproctitle.setproctitle("D3Net")

def main():

  '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
  '''
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--launcher', default='pytorch', help='job launcher')
  parser.add_argument('--dist', default=False)
  parser.add_argument("--world_size", type=int, default=4, help="it is recommended to set to the number of Gpus")
  parser.add_argument("-e", "--epoch", type=int, default=0, help="epoch to start training from")
  parser.add_argument("--train_datasets", type=str, default="/Datasets", help="name of the dataset")
  parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
  parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
  parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
  parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
  parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
  parser.add_argument("--n_cpu", type=int, default=40, help="number of cpu threads to use during batch generation")
  parser.add_argument("--img_height", type=int, default=128, help="high res. image height") 
  parser.add_argument("--img_width", type=int, default=128, help="high res. image width")
  parser.add_argument("--channels", type=int, default=3, help="number of image channels")
  parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
  parser.add_argument("-c", "--checkpoint_interval", type=int, default=5000,
                      help="batch interval between model checkpoints")
  parser.add_argument("--ADB_blocks", type=int, default=12, help="number of ADB blocks in the generator")
  parser.add_argument("--netG", type=str, default='D3Net', help="The network structure of the generator")
  parser.add_argument("--test_every_epochs", type=int, default=10, help="test_every_epochs")
  parser.add_argument("--model_folder", type=str, default='./ckpt/test/', help="model_folder name")
  parser.add_argument('--local-rank', type=int, help='local rank for dist')

  os.makedirs("images/training", exist_ok=True)

  opt = parser.parse_args()

  # ----------------------------------
  # Rename operation
  # ----------------------------------

  test_every_epochs = opt.test_every_epochs

  dataSets = opt.train_datasets
  model_folder = opt.model_folder

  if model_folder == "empty":
    folder_name = os.path.basename(current_dir)
    model_folder = folder_name
    os.makedirs(model_folder, exist_ok=True)
  else:
    os.makedirs(model_folder, exist_ok=True)

  # ----------------------------------------
  # distributed settings
  # ----------------------------------------

  print("MASTER_ADDR:"+os.environ['MASTER_ADDR'])
  print("MASTER_PORT:"+os.environ['MASTER_PORT'])

  print("---------------------------")
  torch.cuda.set_device(opt.local_rank)
  print(opt.local_rank)
  torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
    )
  device = torch.device(f'cuda:{opt.local_rank}')
 
  '''
  # ----------------------------------------
  # Step--2 (creat dataloader)
  # ----------------------------------------
  '''
  output_shape = (opt.img_height, opt.img_width)
  input_shape = (opt.img_height , opt.img_width )

  transform = transforms.Compose([
    transforms.Resize ((opt.img_height, opt.img_width),interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)]
  )

  train_sampler = object()
  if opt.netG == "D3Net":
    train_dataset= ALL_Dataset(dataSets,
                        output_shape, input_shape)
                       
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

    dataloader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            drop_last=True, 
            pin_memory=True,
            num_workers=opt.n_cpu,
            sampler=train_sampler
        )
  
  '''
  # ----------------------------------------
  # Step--3 (init model)
  # ----------------------------------------
  '''
  model=D3Net(n_channels=3, out_channels=3, num_adb_blocks=opt.ADB_blocks)
  model=model.cuda()
  generator = torch.nn.parallel.DistributedDataParallel(model,  device_ids=[opt.local_rank], find_unused_parameters=True)
        
  print([opt.local_rank])
  
  #----------------- Loss----------------------------------
  criterion_pixel = torch.nn.L1Loss().to(device)
  #-----------------------------------------------------s----- 
  
  # Create Optimizers
  optimizer_G = torch.optim.Adam(itertools.chain( generator.parameters()), lr=opt.lr,
                                 betas=(opt.b1, opt.b2))
  Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
  
  # Set the cosine annealing learning rate scheduler
  lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=opt.n_epochs, eta_min=1e-6)
 
  if opt.epoch != 0:
      # Load pretrained models
      generator_checkpoint = torch.load(f"{model_folder}/generator_{opt.epoch}.pth")   
      generator.module.load_state_dict(generator_checkpoint['net'])  #load the generator of first part
 
      optimizer_G.load_state_dict(generator_checkpoint['optimizer'])
   
  prev_time = time.time()
  for epoch in range(opt.epoch, opt.n_epochs):
      # Synchronize all processes before each epoch starts
      torch.distributed.barrier()

      train_sampler.set_epoch(epoch)
      for i, imgs in enumerate(dataloader):
          
          batches_done = epoch * len(dataloader) + i
            
          imgs_input = Variable(imgs["input"].type(Tensor))
          imgs_output = Variable(imgs["gt"].type(Tensor))

          imgs_input = imgs_input.reshape(-1, 3, opt.img_height, opt.img_width)
          imgs_output = imgs_output.reshape(-1, 3, opt.img_height, opt.img_width)

          # ------------------
          #  Train D3Net
          # ------------------
  
          optimizer_G.zero_grad()

          decInformation = generator(imgs_input)
          output = torch.add(imgs_input, decInformation)
          
          loss_pixel = criterion_pixel(output, imgs_output) 

          # Total loss
          loss_G =  1 * loss_pixel         

          loss_G.backward()
          optimizer_G.step()
  
          torch.distributed.barrier()
  
          # --------------
          #  Log Progress
          # --------------
  
          batches_done = epoch * len(dataloader) + i
          batches_left = opt.n_epochs * len(dataloader) - batches_done
          time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
          prev_time = time.time()
          if opt.local_rank==0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f, pixel: %f] [G_lr %f] [ETS: %s]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_G.item(),
                    loss_pixel.item(),
                    optimizer_G.param_groups[0]['lr'],
                    time_left,
                )
            )
          
          if opt.local_rank==0:
            if batches_done % opt.sample_interval == 0:

                # Save image grid with upsampled inputs and outputs
                img_grid = torch.cat((imgs_input, output), -1)
                img_grid = torch.cat((img_grid, imgs_output), 3)
                img_grid = denormalize(img_grid)
                save_image(img_grid, f"images/training/{batches_done}.png", nrow=1, normalize=False)
       
          if opt.local_rank==0:
            if batches_done % opt.checkpoint_interval == 0:
                # Save model checkpoints
                generator_checkpoint = {
                  "net": generator.module.state_dict(),
                  'optimizer':optimizer_G.state_dict(),
                  "epoch": epoch
                }
                torch.save(generator_checkpoint, f"{model_folder}/generator_{epoch}.pth")
                                    
      # Update learning rates
      lr_scheduler_G.step()
      generator.eval()

if __name__ == '__main__':
    main()
