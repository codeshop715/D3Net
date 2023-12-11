import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import warnings
import torch.nn as nn
import lmdb
from torchvision.transforms import ColorJitter, RandomAffine, RandomPerspective, RandomCrop
import atexit
warnings.filterwarnings("ignore")


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)
    
class FixedSeedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, seed=None):
        if seed is not None:
            torch.random.manual_seed(seed)
        return self.transform(img)

class RescaleIfSmallerThan(object):
    def __init__(self, min_height, min_width):
        self.min_height = min_height
        self.min_width = min_width

    def __call__(self, image):
        width, height = image.size
        
        # Rescale if image is smaller than minimum size
        if width < self.min_width or height < self.min_height:
            scale_factor = max(self.min_width / width, self.min_height / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = transforms.functional.resize(image, (new_height, new_width))
        return image


class ALL_Dataset(Dataset):
    def __init__(self, roots, gt_shape, input_shape, target_samples=10000):
        input_height, input_width = input_shape
        gt_height, gt_width = gt_shape
        self.target_samples = target_samples

        ensure_size = RescaleIfSmallerThan(input_height, input_width)
        random_crop = transforms.RandomCrop((input_height, input_width))
        
        transforms_list = [
            ensure_size,     
            random_crop,     
        ]

        self.gt_transform = FixedSeedTransform(transforms.Compose(
            transforms_list + [
                transforms.Resize((gt_height, gt_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std), 
            ]
        ))
        self.input_transform = FixedSeedTransform(transforms.Compose(
            transforms_list + [
                transforms.Resize((input_height, input_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ))

        self.files_c = []
        self.files_n = []
        
        type_files_c = []
        for root in os.listdir(roots):
            print(root)
            if 'RESIDE' in root:
                # Adjust based on our exact paths and patterns
                print('RESIDE loading...')
                type_files_n = sorted(glob.glob(os.path.join(roots, 'RESIDE/OTS/haze/**/*.*'), recursive=True))
                base_names = [os.path.basename(f).split('_')[0] for f in type_files_n]
                type_files_c = [os.path.join(roots, 'RESIDE/OTS/clear/clear', str(bn)+'.jpg') for bn in base_names]
                print('RESIDE finishing...')
                
            elif 'GOPRO' in root:
                # Recursively get all images
                # print('No action')
                # continue
                print('GOPRO loading...')
                type_files_n = sorted(glob.glob(os.path.join(roots, 'GOPRO/train/**/blur/*.*'), recursive=True))
                type_files_c = sorted(glob.glob(os.path.join(roots, 'GOPRO/train/**/sharp/*.*'), recursive=True))
                print('GOPRO finishing...')
            
            elif 'Denoise' in root:
                print('WED and BSD loading...')
                type_files_n = []
                type_files_c = []

                for noise_level in ['noisy15', 'noisy25', 'noisy50']:
                    noisy_files = sorted(glob.glob(os.path.join(roots, 'Denoise/BSD', noise_level, '*.*')))
                    type_files_n.extend(noisy_files)
                    clean_files = [os.path.join(roots, 'Denoise/BSD/original', os.path.basename(f)[:-4]+'.jpg') for f in noisy_files]
                    type_files_c.extend(clean_files)

                # If total BSD samples are less than target, take the rest from WED dataset
                if len(type_files_n) < target_samples:
                    # Calculate how many samples we still need
                    remaining_samples = target_samples - len(type_files_n)

                    # Take the rest from WED dataset evenly from different noise levels
                    samples_per_noise = remaining_samples // 3  # 5 noise levels
    
                    for noise_level in [ 'noisy15', 'noisy25', 'noisy50']:
                        noisy_files = sorted(glob.glob(os.path.join(roots, 'Denoise/WED', noise_level, '*.*')))
                        type_files_n.extend(noisy_files[:samples_per_noise])
                        clean_files = [os.path.join(roots, 'Denoise/WED/original', os.path.basename(f)[:-4]+'.png') for f in noisy_files]
                        type_files_c.extend(clean_files[:samples_per_noise])

                print('WED and BSD finishing...')

            elif 'LOL' in root:
                print('LOL loading...')

                type_files_c = sorted(glob.glob(os.path.join(roots , "LOL/our485/high/*.*")))
                type_files_n = sorted(glob.glob(os.path.join(roots ,"LOL/our485/low/*.*")))

                print('LOL finishing...')

            elif 'Rain200L' in root:
                print('Rain200L loading...')
                type_files_c = sorted(glob.glob(os.path.join(roots, 'Rain200L/train/norain/*.*')))
                base_names = [os.path.basename(f).split('.')[0] for f in type_files_c]
                type_files_n = [os.path.join(roots, 'Rain200L/train/rain/', str(bn)+'x2.png') for bn in base_names]
                print('Rain200L finishing...')
           

            while len(type_files_c) < target_samples:
                type_files_c += type_files_c

            while len(type_files_n) < target_samples:
                type_files_n += type_files_n

            

            # Create a list of indexes
            index_list = list(range(len(type_files_c)))
            # Use random.sample to disrupt the indexed list
            shuffled_index_list = random.sample(index_list, len(index_list))

            # Reorder type_files_c and type_files_n according to the list of disrupted indexes
            type_files_c = [type_files_c[i] for i in shuffled_index_list]
            type_files_n = [type_files_n[i] for i in shuffled_index_list]

            self.files_c.append(type_files_c[:target_samples])
            self.files_n.append(type_files_n[:target_samples])

    def __getitem__(self, index):
        gts = []  
        inputs = []  
        
        for i in range(5):  # Rescale if image is smaller than minimum size
            type_idx = i

            img_c = Image.open(self.files_c[type_idx][index])
            img_n = Image.open(self.files_n[type_idx][index])

            if img_c.mode != 'RGB':
                    img_c = img_c.convert('RGB')
            if img_n.mode != 'RGB':
                    img_n = img_n.convert('RGB')

            # Get a random seed with this method      
            seed = torch.random.seed()

            # The method sets the random seed
            gts.append(self.gt_transform(img_c, seed))
            inputs.append(self.input_transform(img_n, seed))


        gts_tensor = torch.stack(gts)
        inputs_tensor = torch.stack(inputs)
 
        return {"gt": gts_tensor, "input": inputs_tensor}

    def __len__(self):
        # We return 5 images in each iteration, so divide the length by 5
        return len(self.files_c[0]) // 5
