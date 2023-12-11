from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
import torch
from torchvision import transforms,datasets
import cv2
from .lmdb import *



def create_lmdb_dataset(input_files, output_path):
    # Create a new lmdb environment
    env = lmdb.open(output_path, map_size=1099511627776, max_readers=60)  # map_size indicates the database size
    
    with env.begin(write=True) as txn:
        for idx, img_list in enumerate(input_files):
            for img_path in img_list:
                # Read and encode images into PNG format
                image = cv2.imread(img_path) 
                # Read and encode images into PNG format (other encodings such as JPEG can be selected)
                if len(image.shape) == 2:  # If it's a grayscale image
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)      
                _, buffer = cv2.imencode(".png", image) # Returns two values, the second value is the encoded image
                # Use image paths as keys
                txn.put(img_path.encode(), buffer.tobytes())

    env.close()


class LMDBSingleton:
    _instance = None
    _env_c = None
    _env_n = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LMDBSingleton, cls).__new__(cls)
            cls._env_c = lmdb.open("/files_c.lmdb", readonly=True, max_readers=60)
            cls._env_n = lmdb.open("/files_n.lmdb", readonly=True, max_readers=60)
        return cls._instance

    @staticmethod
    def get_env_c():
        return LMDBSingleton._env_c

    @staticmethod
    def get_env_n():
        return LMDBSingleton._env_n

    @staticmethod
    def close():
        if LMDBSingleton._env_c:
            LMDBSingleton._env_c.close()
        if LMDBSingleton._env_n:
            LMDBSingleton._env_n.close()

