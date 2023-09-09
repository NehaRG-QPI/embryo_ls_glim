#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
import skimage
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


class embhealth(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self,index):
        img_path=os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image=io.imread(img_path)
        image=(image-image.min())/(image.max()-image.min())
        name=self.annotations.iloc[index,0]
        y_label=torch.tensor(int(self.annotations.iloc[index,1]))
        
        if self.transform:
            image=self.transform(image)
            
        return(image,y_label,name)
            







