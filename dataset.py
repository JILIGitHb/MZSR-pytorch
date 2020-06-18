import numpy as np
import random
from scipy.misc import imresize as resize

import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
from gkernel import *
from imresize import imresize
from source_target_transforms import *

META_BATCH_SIZE = 5
TASK_BATCH_SIZE = 8

def random_crop(hr,size):
    h, w = hr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    crop_hr = hr[y:y+size, x:x+size].copy()

    return crop_hr


def random_flip_and_rotate(im1):
    if random.random() < 0.5:
        im1 = np.flipud(im1)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)

    # have to copy before be called by transform function
    return im1.copy()


class preTrainDataset(data.Dataset):
    def __init__(self, path, patch_size=64, scale=4):
        super(preTrainDataset, self).__init__()
        self.patch_size = patch_size

        h5f = h5py.File(path, 'r')
        self.hr = [v[:] for v in h5f["HR"].values()]
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]
        
        h5f.close()

        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(self.patch_size),
            ToTensor()
        ])
    def __getitem__(self, index):
        item = [(self.hr[index], resize(self.lr[i][index], self.scale*100, interp='cubic')) for i, _ in enumerate(self.lr)]
        # return [(self.transform(hr), self.transform(imresize(lr, 400, interp='cubic'))) for hr, lr in item]
        return [self.transform(hr,lr) for hr, lr in item]

    def __len__(self):
        return len(self.hr)

class metaTrainDataset(data.Dataset):
    def __init__(self, path, patch_size=64, scale=4):
        super(metaTrainDataset, self).__init__()
        self.size = patch_size
        h5f = h5py.File(path, 'r')
        self.hr = [v[:] for v in h5f["HR"].values()]
        self.hr = random.sample(self.hr, TASK_BATCH_SIZE*2*META_BATCH_SIZE)
        h5f.close()

        # self.tansform = transforms.Compose([
        #     transforms.RandomCrop(64),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip()
        #     # transforms.ToTensor()
        # ])

    def __getitem__(self, index):
        item = [self.hr[index]/255.]
        item = [random_crop(hr,self.size) for hr in item]
        return [random_flip_and_rotate(hr) for hr in item]
    def __len__(self):
        return len(self.hr)

def make_data_tensor(scale, noise_std=0.0):
    label_train = metaTrainDataset('data/DIV2K_train.h5')

    input_meta = []
    label_meta = []

    for t in range(META_BATCH_SIZE):
        input_task = []
        label_task = []

        Kernel = generate_kernel(k1=scale*2.5, ksize=15)
        for idx in range(TASK_BATCH_SIZE*2):
            img_HR = label_train[t*TASK_BATCH_SIZE*2 + idx][-1]
            # add isotropic and anisotropic Gaussian kernels for the blur kernels 
            # and downsample 
            clean_img_LR = imresize(img_HR, scale=1./scale, kernel=Kernel)
            # add noise
            img_LR = np.clip(clean_img_LR + np.random.randn(*clean_img_LR.shape)*noise_std, 0., 1.)
            # used cubic upsample 
            img_ILR = imresize(img_LR,scale=scale, output_shape=img_HR.shape, kernel='cubic')

            input_task.append(img_ILR)
            label_task.append(img_HR)
        
        input_meta.append(np.asarray(input_task))
        label_meta.append(np.asarray(label_task))
    
    input_meta = np.asarray(input_meta)
    label_meta = np.asarray(label_meta)

    inputa = input_meta[:,:TASK_BATCH_SIZE,:,:]
    labela = label_meta[:,:TASK_BATCH_SIZE,:,:]
    inputb = input_meta[:,TASK_BATCH_SIZE:,:,:]
    labelb = label_meta[:,TASK_BATCH_SIZE:,:,:]

    return inputa, labela, inputb, labelb

if __name__ == '__main__':
    make_data_tensor(4)