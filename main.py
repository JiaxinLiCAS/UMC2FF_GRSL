# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:48:00 2022

@author: 13572
"""


import torch
import torch.nn as nn
import time
import numpy as np
import hues
import os
import random
import scipy.io as sio
from model.config import args
from model.evaluation import MetricsCal



def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
        
setup_seed(28) 

#Step:1
from model.srf_psf_layer import Blind       #将退化函数看作一层的参数

blind = Blind(args)
lr_msi_fhsi_est, lr_msi_fmsi_est=blind.train() #device格式的1 C H W张量
blind.get_save_result() #保存PSF SRF
psf = blind.model.psf.data.cpu().detach().numpy()[0,0,:,:] #15 x 15  numpy
srf = blind.model.srf.data.cpu().detach().numpy()[:,:,0,0].T #46 x 8   numpy
psf_gt=blind.psf_gt #15 x 15   numpy
srf_gt=blind.srf_gt  #46 x 8   numpy


#Step:2 and 3

from model.spectral_SR_fusion_C2F import spectral_SR
    
#提供学习到的两个lrmsi
spectral_sr=spectral_SR(args,lr_msi_fhsi_est.clone().detach(), lr_msi_fmsi_est.clone().detach(),
                           blind.tensor_lr_hsi,blind.tensor_hr_msi,blind.gt,srf,blind.tensor_lr_msi_fmsi) 

est=spectral_sr.train() #四维tensor 在device上
    



