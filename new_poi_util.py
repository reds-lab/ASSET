from ast import Not
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import cv2 as cv
import torch.nn as nn
from collections import OrderedDict
import copy
from PIL import Image
from tqdm import tqdm
import random

seed = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,  transform):
        self.indices = indices
        if not isinstance(self.indices, list):
            self.indices = list(self.indices)
        self.dataset = Subset(dataset, self.indices)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        if self.transform != None:
            # image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.indices)

class delete_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.indices = indices
        self.dataset = dataset
        if not isinstance(self.indices, list):
            self.indices = list(self.indices)
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.data = np.delete(self.data,indices,0)
        self.targets = np.delete(self.targets,indices)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        return (image, label)

    def __len__(self):
        return len(self.targets)
        

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
    
def apply_noise_patch(noise,images,offset_x=0,offset_y=0,mode='change',padding=20,position='fixed'):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, 512, 512)
    outputs: torch.Tensor(N, 3, 512, 512)
    '''
    length = images.shape[2] - noise.shape[2]
    if position == 'fixed':
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding,length-padding)
        ht = np.random.randint(padding,length-padding)
    if len(images.shape) == 3:
        noise_now = np.copy(noise[0,:,:,:])
        wr = length-wl
        hb = length-ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if(mode == 'change'):
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
            images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = noise_now
        else:
            images += noise_now
    else:
        for i in range(images.shape[0]):
            noise_now = np.copy(noise)
            wr = length-wl
            hb = length-ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if(mode == 'change'):
                images[i:i+1,:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = 0
                images[:,ht:ht+noise.shape[2],wl:wl+noise.shape[3]] = noise_now
            else:
                images[i:i+1] += noise_now
    return images

class noisy_label(Dataset):
    def __init__(self, dataset, indices, num_classes, transform, seed):
        set_seed(seed)
        print('Random seed is: ', seed)
        self.dataset = dataset
        self.indices = indices
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.num_classes = num_classes
        self.transform = transform

        allos_idx = []
        for i in range(num_classes):
            allowed_values = list(range(num_classes))
            allowed_values.remove(i)
            allos_idx.append(allowed_values)
        for i in range(len(indices)):
            tar_lab = self.targets[indices[i]]
            self.targets[indices[i]] = random.choice(allos_idx[tar_lab])

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, target)

    def __len__(self):
        return len(self.dataset)

class flipping_label(Dataset):
    def __init__(self, dataset, indices, tar_lab, transform, seed):
        set_seed(seed)
        self.dataset = dataset
        self.indices = indices
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.tar_lab = tar_lab
        for i in self.indices:
            self.targets[i] = self.tar_lab
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, target)

    def __len__(self):
        return len(self.dataset)

class change_label(Dataset):
    def __init__(self, dataset, tar_lab):
        set_seed(seed)
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.indices = np.where(np.array(self.targets)==tar_lab[0])[0]
        self.tar_lab = tar_lab

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        return (image, self.tar_lab[1])

    def __len__(self):
        return self.indices.shape[0]

class posion_image_nottar_label(Dataset):
    def __init__(self, dataset,indices,noise,lab):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.lab = lab

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        if idx in self.indices:
            image = torch.clamp(apply_noise_patch(self.noise,image,mode='add'),-1,1)
            label = self.lab
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_image_all2all(Dataset):
    def __init__(self, dataset,noise,poi_list,num_classes, transform):
        self.dataset = dataset
        self.data = dataset.data
        self.targets = self.dataset.targets
        self.poi_list = poi_list
        self.noise = noise
        self.num_classes = num_classes
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.targets[i] + 1
            if self.targets[i] == self.num_classes:
                self.targets[i] = 0

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if idx in self.poi_list:
            pat_size = 4
            image[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 255
        
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_noisy_all2all(Dataset):
    def __init__(self, dataset,noise,poi_list,num_classes, transform):
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.noise = noise
        self.num_classes = num_classes
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.targets[i] + 1
            if self.targets[i] == self.num_classes:
                self.targets[i] = 0

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if idx in self.poi_list:
            image = image.astype(int)
            image += self.noise
            image = np.clip(image,0,255)
        
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.dataset)

class posion_image_all2one(Dataset):
    def __init__(self, dataset,poi_list,tar_lab, transform):
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.tar_lab = tar_lab
        self.transform = transform
        for i in self.poi_list:
            self.targets[i] = self.tar_lab

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        poison = 0
        if idx in self.poi_list:
            poison = 1
            pat_size = 4
            image[32 - 1 - pat_size:32 - 1, 32 - 1 - pat_size:32 - 1, :] = 255
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label, poison)

    def __len__(self):
        return len(self.dataset)

class posion_noisy_all2one(Dataset):
    def __init__(self, dataset,poi_list,tar_lab, transform, noisy):
        self.dataset = dataset
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.poi_list = poi_list
        self.tar_lab = tar_lab
        self.transform = transform
        self.noisy = noisy
        for i in self.poi_list:
            self.targets[i] = self.tar_lab

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        poison = 0
        if idx in self.poi_list:
            poison = 1
            image = image.astype(int)
            image += self.noisy
            image = np.clip(image,0,255)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label, poison)

    def __len__(self):
        return len(self.dataset)

class posion_image(Dataset):
    def __init__(self, dataset,indices,noise, transform):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.data = copy.deepcopy(self.dataset.data) 
        self.targets = copy.deepcopy(self.dataset.targets)
        self.transform = transform

    def __getitem__(self, idx):
        poi = 0
        image = self.data[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        if idx in self.indices:
            poi = 1
            image += self.noise
        label = self.targets[idx]
        return (image, label, poi)

    def __len__(self):
        return len(self.dataset)
    
class posion_image_label(Dataset):
    def __init__(self, dataset,indices,noise,target,transform):
        self.dataset = dataset
        self.indices = indices
        self.noise = noise
        self.target = target
        self.targets = copy.deepcopy(self.dataset.targets)
        self.data = copy.deepcopy(self.dataset.data) 
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        if idx in self.indices:
            image = image.astype(int)
            image += self.noise
            image = np.clip(image,0,255)
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        #label = self.dataset[idx][1]
        return (image, self.target)

    def __len__(self):
        return len(self.indices)
    
class get_labels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][1]

    def __len__(self):
        return len(self.dataset)

class concoct_dataset(torch.utils.data.Dataset):
    def __init__(self, target_dataset,outter_dataset):
        self.idataset = target_dataset
        self.odataset = outter_dataset

    def __getitem__(self, idx):
        if idx < len(self.odataset):
            img = self.odataset[idx][0]
            labels = self.odataset[idx][1]
        else:
            img = self.idataset[idx-len(self.odataset)][0]
            #labels = torch.tensor(len(self.odataset.classes),dtype=torch.long)
            labels = len(self.odataset.classes)
        #label = self.dataset[idx][1]
        return (img,labels)

    def __len__(self):
        return len(self.idataset)+len(self.odataset)

def inverse_normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    for i in range(len(mean)):
        img[:,:,i] = img[:,:,i]*std[i]+mean[i]
    return img

def poi_dataset(Dataset, poi_methond='badnets', transform=None, tar_lab = 0, poi_rates = 0.2, random_seed = 0, noisy = None):
    set_seed(random_seed)
    label = Dataset.targets
    num_classes = len(np.unique(label))
    if poi_methond == 'backdoor_all2all':
        badnets_noise = np.zeros((1, 3, 32, 32))
        badnets_noise[0,:,26:31,26:31] = 255
        poi_idx = []
        for i in range(num_classes):
            current_label = np.where(np.array(label)==i)[0]
            samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
            poi_idx.extend(samples_idx)
        posion_dataset = posion_image_all2all(Dataset,badnets_noise, poi_idx, num_classes, transform)
        return posion_dataset, poi_idx
    elif poi_methond == 'noisy_label':
        poi_idx = []
        for i in range(num_classes):
            current_label = np.where(np.array(label)==i)[0]
            samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
            poi_idx.extend(samples_idx)
        posion_dataset = noisy_label(Dataset, poi_idx, num_classes, transform, random_seed)
        return posion_dataset, poi_idx
    elif poi_methond == 'flipping_label':
        poi_idx = []
        current_label = np.where(np.array(label)==tar_lab[0])[0]
        samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
        poi_idx.extend(samples_idx)
        posion_dataset = flipping_label(Dataset, poi_idx, tar_lab[1], transform, random_seed)
        return posion_dataset, poi_idx
    elif poi_methond == 'backdoor':
        current_label = np.where(np.array(label)!=tar_lab)[0]
        poi_idx = np.random.choice(current_label, size=int(len(Dataset) * poi_rates), replace=False)
        posion_dataset = posion_image_all2one(Dataset, poi_idx, tar_lab, transform)
        return posion_dataset, poi_idx
    elif poi_methond == 'clean_label_narcissus':
        current_label = np.where(np.array(label)==tar_lab)[0]
        poi_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
        if noisy is None:
            noisy = np.load('/home/minzhou/public_html/unlearnable/cifar10/result/resnet18_97.npy')[0]
        posion_dataset = posion_image(Dataset, poi_idx, noisy, transform)
        return posion_dataset, poi_idx
    elif poi_methond == 'noisy_all2one':
        current_label = np.where(np.array(label)!=tar_lab)[0]
        poi_idx = np.random.choice(current_label, size=int(len(Dataset) * poi_rates), replace=False)
        if noisy is None:
            noisy = (np.load('/home/minzhou/public_html/unlearnable/cifar10/best_universal.npy')[0]*255).astype(int)
        posion_dataset = posion_noisy_all2one(Dataset, poi_idx, tar_lab, transform, noisy)
        return posion_dataset, poi_idx
    if poi_methond == 'noisy_all2all':
        if noisy is None:
            noisy = (np.load('/home/minzhou/public_html/unlearnable/cifar10/best_universal.npy')[0]*255).astype(int)
        poi_idx = []
        for i in range(num_classes):
            current_label = np.where(np.array(label)==i)[0]
            samples_idx = np.random.choice(current_label, size=int(current_label.shape[0] * poi_rates), replace=False)
            poi_idx.extend(samples_idx)
        posion_dataset = posion_noisy_all2all(Dataset,noisy, poi_idx, num_classes, transform)
        return posion_dataset, poi_idx

import h5py
class h5_dataset(Dataset):
    def __init__(self, path, train, transform):
        f = h5py.File(path,'r') 
        if train:
            self.data = np.vstack((np.asarray(f['X_train']),np.asarray(f['X_val']))).astype(np.uint8)
            self.targets = list(np.argmax(np.vstack((np.asarray(f['Y_train']),np.asarray(f['Y_val']))),axis=1))
        else:
            self.data = np.asarray(f['X_test']).astype(np.uint8)
            self.targets = list(np.argmax(np.asarray(f['Y_test']),axis=1))
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.targets)

def get_result(model, dataset, poi_idx):
    poi_set = Subset(dataset, poi_idx)
    clean_idx = list(set(np.arange(len(dataset))) - set(poi_idx))
    clean_set = Subset(dataset,clean_idx)
    
    poiloader = torch.utils.data.DataLoader(poi_set, batch_size=512, shuffle=False, num_workers=4)
    cleanloader = torch.utils.data.DataLoader(clean_set, batch_size=512, shuffle=False, num_workers=4)
    full_ce = nn.CrossEntropyLoss(reduction='none')
    
    poi_res = []
    for i, (data, target,_) in enumerate(tqdm(poiloader)):
        data, target= data.cuda(), target.cuda()
        with torch.no_grad():
            poi_outputs = model(data)
            # poi_loss = torch.var(poi_outputs,dim=1)
            poi_loss = full_ce(poi_outputs, target)
            poi_res.extend(poi_loss.cpu().detach().numpy())
            
    clean_res = []
    model.eval()
    for i, (data, target,_) in enumerate(tqdm(cleanloader)):
        data, target= data.cuda(), target.cuda()
        with torch.no_grad():
            clean_outputs = model(data)
            # clean_loss = torch.var(clean_outputs,dim=1)
            clean_loss = full_ce(clean_outputs, target)
            clean_res.extend(clean_loss.cpu().detach().numpy())
            
    return poi_res, clean_res

from sklearn.mixture import GaussianMixture
def get_t(data, eps=1e-3):
    halfpoint = np.quantile(data, 0.5, interpolation='lower')
    lowerdata = np.array(data)[np.where(data<=halfpoint)[0]]
    f = np.ravel(lowerdata).astype(np.float)
    f = f.reshape(-1,1)
    g = GaussianMixture(n_components=1,covariance_type='full')
    g.fit(f)
    weights = g.weights_
    means = g.means_ 
    covars = np.sqrt(g.covariances_)
    return (covars*np.sqrt(-2*np.log(eps)*covars*np.sqrt(2*np.pi)) + means)/ weights


import statsmodels.api
def adjusted_outlyingness(series):
    _ao = []

    med = torch.median(series)
    q1, q3 = torch.quantile(series, torch.tensor([0.25, 0.75]).cuda())
    mc = torch.tensor(statsmodels.api.stats.stattools.medcouple(series.cpu().detach().numpy())).cuda()
    iqr = q3 - q1

    if mc > 0:
        w1 = q1 - (1.5 * torch.e ** (-4 * mc) * iqr)
        w2 = q3 + (1.5 * torch.e ** (3 * mc) * iqr)
    else:
        w1 = q1 - (1.5 * torch.e ** (-3 * mc) * iqr)
        w2 = q3 + (1.5 * torch.e ** (4 * mc) * iqr)

    for s in series:
        if s > med:
            _ao.append((s - med) / (w2 - med))
        else:
            _ao.append((med - s) / (med - w1))

    return torch.tensor(_ao).cuda()