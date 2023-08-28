# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 18:50:45 2021

@author: 13572
"""

import argparse
import torch
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
###通用参数
parser.add_argument('--scale_factor',type=int,default=12, help='缩放尺度  DC=10 TG=12 Chikusei=16')
parser.add_argument('--sp_root_path',type=str, default='data/XINet/spectral_response/',help='光谱相应地址')
parser.add_argument('--default_datapath',type=str, default="data/XINet/",help='高光谱读取地址')
parser.add_argument('--data_name',type=str, default="TG",help='Whisper=8 DC=10 TG=12 Chikusei=16 ')
parser.add_argument("--gpu_ids", type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir',type=str, default='checkpoints',help='高光谱读取地址')
####



#训练参数
parser.add_argument("--lr_stage1", type=float, default=0.001,help='学习率6e-3 0.001')
parser.add_argument('--niter1', type=int, default=3000, help='# of iter at starting learning rate3000')
parser.add_argument('--niter_decay1', type=int, default=3000, help='# of iter to linearly decay learning rate to zero3000')

parser.add_argument("--lr_stage2_SPe", type=float, default=4e-3,help='学习率4e-3')
parser.add_argument('--niter2_SPe', type=int, default=7000, help='#7000 of iter at starting learning rate')
parser.add_argument('--niter_decay2_SPe', type=int, default=7000, help='# 7000of iter to linearly decay learning rate to zero')
    

  

#添加噪声
parser.add_argument('--noise', type=str, default="No", help='Yes ,No')
parser.add_argument('--nSNR', type=int, default=35)


args=parser.parse_args()

device = torch.device(  'cuda:{}'.format(args.gpu_ids)  ) if  torch.cuda.is_available() else torch.device('cpu') 
args.device=device
args.sigma = args.scale_factor / 2.35482

args.expr_dir=os.path.join('checkpoints', args.data_name+'_SF'+str(args.scale_factor)
                           )
