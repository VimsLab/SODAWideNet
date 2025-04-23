import sys
import glob
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def random_brightness(inp_img):
    contrast = np.random.rand(1) + 0.5
    light = np.random.randint(-100, 100)
    inp_img = contrast * inp_img + light

    return np.clip(inp_img, 0, 255)

class SODLoaderAugmentNew(Dataset):
    def __init__(self):
    	self.inp_path = './AugmentedDUTS384/Image'
    	self.out_path = './AugmentedDUTS384/Mask'
    	self.contour_path = './AugmentedDUTS384/Contour'

    	self.augment_data = True
    	self.mode = mode

    	self.inp_files = sorted(glob.glob(self.inp_path + '/*'))
    	self.out_files = sorted(glob.glob(self.out_path + '/*'))
    	self.contour_files = sorted(glob.glob(self.contour_path + '/*'))

    def __getitem__(self, idx):
        inp_img = cv.imread(self.inp_files[idx])
	inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
	inp_img = inp_img.astype('float32')

	mask_img = cv.imread(self.out_files[idx], 0)
	mask_img = mask_img.astype('float32')
	mask_img /= np.max(mask_img)

   	contour_img = cv.imread(self.contour_files[idx], 0)
	contour_img = contour_img.astype('float32')
	contour_img /= np.max(contour_img)

	if self.augment_data:
	    inp_img = random_brightness(inp_img)

    	inp_img /= np.max(inp_img)
    	inp_img = np.transpose(inp_img, axes=(2, 0, 1))
    	inp_img = torch.from_numpy(inp_img).float()

    	mask_img = np.expand_dims(mask_img, axis=0)

    	contour_img = np.expand_dims(contour_img, axis=0)

    	return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(contour_img).float()

    def __len__(self):
    	return len(self.inp_files)
