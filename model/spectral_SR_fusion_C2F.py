# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 19:38:23 2021

@author: 13572
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
from .evaluation import MetricsCal
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import scipy

#from.network import Spectral_upsample
from.network import def_refinement #def_lr_hsi_initial_feature
from.network import def_MSAF
from.network import def_SDG #def_SRF_Guided_conv

class spectral_SR():
    def __init__(self,args,lr_msi_fhsi,lr_msi_fmsi,lr_hsi,hr_msi,gt,srf,lr_msi_gt):
        self.args=args
        self.hs_band=lr_hsi.shape[1]  #torch.Size([1, 54, 20, 20])
        self.ms_band=lr_msi_fhsi.shape[1] #torch.Size([1, 8, 20, 20])
        
        self.lr_msi_fhsi=lr_msi_fhsi #torch.Size([1, 8, 20, 20]) 已经在device上的
        self.lr_msi_fmsi=lr_msi_fmsi #torch.Size([1, 8, 20, 20]) 已经在device上的
        self.lr_hsi=lr_hsi   #torch.Size([1, 54, 20, 20]) 已经在device上的    低空间分辨率的高光谱 四维tensor
        self.hr_msi=hr_msi   #torch.Size([1, 8, 240, 240]) 已经在device上的   高空间分辨率的多光谱 四维tensor
        self.gt=gt #H W C的numpy
        self.lr_msi_gt=lr_msi_gt #1 C H W tensor
        
        self.srf_est=srf #numpy 54X8
        self.index=self.srf_est.argmax(1) #numpy (54,)
        #self.index_modification=self.modify(self.index.copy()) # numpy (54,) 根据srf连续性，假如最强响应和左右不相同，则对其纠正
        self.index_statistics=self.cal_index_statistics(self.index)#统计每个MS波段对应的HS波段索引 字典形式
        
        #生成每个子网络
        
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch +  1 - self.args.niter2_SPe) / float(self.args.niter_decay2_SPe + 1)
            return lr_l

        self.optimizers=[]  #保存超分网络以及融合网络
        self.schedulers=[]  #保存超分网络以及融合网络
        
        #初始化融合模块
        self.msaf=def_MSAF(self.ms_band,self.args.device) #.to(self.args.device)
        optimizer_fusion = optim.Adam(self.msaf.parameters(), lr=self.args.lr_stage2_SPe)
        scheduler_fusion = lr_scheduler.LambdaLR(optimizer_fusion, lr_lambda=lambda_rule)
        self.optimizers.append(optimizer_fusion)
        self.schedulers.append(scheduler_fusion)
        
        
        #初始化C2F
        self.subnets=[]  #保存超分网络
        
        for i in range(self.ms_band):
            subnet=def_SDG(1,len(self.index_statistics[str(i)]),self.args.device)
            #subnet=Spectral_upsample(self.args,1,len(self.index_statistics[str(i)]))

            optimizer=optim.Adam(subnet.parameters(),lr=self.args.lr_stage2_SPe)
            scheduler=lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            
            self.subnets.append(subnet)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            
        self.refinement=def_refinement(self.hs_band,self.hs_band,self.args.device,1)
        #self.C2F=def_residual(self.hs_band,self.hs_band,self.args.device)
        optimizer=optim.Adam(self.refinement.parameters(),lr=self.args.lr_stage2_SPe)
        scheduler=lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        self.optimizers.append(optimizer)
        self.schedulers.append(scheduler)
       
    
    def modify(self,index):
        
        for i in range(len(index)):
            
            if i !=0 and i!=(len(index)-1):
                if index[i]!=index[i-1] and index[i] !=index[i+1]:
                    index[i]=index[i-1]
        return index
        
    def cal_index_statistics(self,index): #统计index里面每个波段对应的索引
        statistics={}
        for i in range(self.ms_band):
            statistics[str(i)]=np.where(index==i)[0]
        return statistics
    '''
    {'0': array([ 0,  1,  2,  3,  4, 41], dtype=int64),
     '1': array([ 5,  6,  7,  8,  9, 10, 11, 12], dtype=int64),
     '2': array([13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=int64),
     '3': array([22, 23, 24, 25, 26, 27], dtype=int64),
     '4': array([28, 29, 30, 31, 32, 33, 34], dtype=int64),
     '5': array([35, 36, 37, 38, 39, 40], dtype=int64),
     '6': array([42, 43, 44, 45, 46, 47, 48, 49, 50], dtype=int64),
     '7': array([51, 52, 53], dtype=int64)}
    '''
    
    def train(self):
        flag_best=[10,0,'data'] #第一个是SAM，第二个是PSNR,第三个为恢复的图像
        information_a=''
        information_b=''
        information_A=''
        weight=0
        L1Loss = nn.L1Loss(reduction='mean')
        
        for epoch in range(1,self.args.niter2_SPe + self.args.niter_decay2_SPe + 1):
       
            
            for optimizer in self.optimizers: #包含了融合和C2F网络
                optimizer.zero_grad()
            
            #实现lrmsi融合
            fused_lr_msi,W = self.msaf(self.lr_msi_fhsi,self.lr_msi_fmsi) #torch.Size([1, 8, 20, 20])
            
            #实现C2F网络
            lr_hsi_srf_est=torch.empty(1,self.hs_band,self.lr_hsi.shape[2],self.lr_hsi.shape[3]).to(self.args.device)
            for i in range(self.ms_band):
                #sub_input=fused_lr_msi[:,[i],:,:]
                lr_hsi_srf_est[:,self.index_statistics[str(i)],:,:]=self.subnets[i](fused_lr_msi[:,[i],:,:])
            
            lr_hsi_est=self.refinement(lr_hsi_srf_est)
            loss2=L1Loss(lr_hsi_est, self.lr_hsi)  
            loss2.backward()
           
          
            
            for optimizer in self.optimizers: #包含了融合和超分网络
                optimizer.step()
                
            for scheduler in self.schedulers: #包含了融合和超分网络
                scheduler.step()
                

            
            if epoch % 10 ==0:

                with torch.no_grad():
                    
                    print("____________________________________________")
                    print('epoch:{} lr:{}'.format(epoch,self.optimizers[0].param_groups[0]['lr']))
                    print('************')
                    
                    
                    #转为W H C的numpy 方便计算指标
                    lr_msi_gt_numpy=self.lr_msi_gt.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    lr_hsi_numpy=self.lr_hsi.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    lr_hsi_est_numpy=lr_hsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                    #gt_est=self.spectral_sr(self.hr_msi).detach().data.cpu().numpy()[0].transpose(1,2,0) #对msi上采样到hhsi
                    
                    #输出融合参数
                    print('W:{}'.format(W.data.cpu().detach().numpy()[0,:,0,0]))
                    
                    #融合后的lrmsi与真值计算指标
                    fused_lr_msi_numpy=fused_lr_msi.cpu().detach().numpy()[0].transpose(1,2,0)
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_msi_gt_numpy,fused_lr_msi_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( lr_msi_gt_numpy - fused_lr_msi_numpy ))
                    information0="fuse lrmsi与目标lrmsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information0) 
                    print('************')
                    
                 
                    #学习到的lrhsi与真值
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(lr_hsi_numpy,lr_hsi_est_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( lr_hsi_numpy - lr_hsi_est_numpy ))
                    information1="生成lrhsi与目标lrhsi\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information1) #监控训练过程
                    print('************')
                    
                    
                    #学习到的gt与真值
                    gt_srf_est=torch.empty(1,self.hs_band,self.hr_msi.shape[2],self.hr_msi.shape[3]).to(self.args.device)
                    for i in range(self.ms_band):
                        #sub_input=fused_lr_msi[:,[i],:,:]
                        gt_srf_est[:,self.index_statistics[str(i)],:,:]=self.subnets[i](self.hr_msi[:,[i],:,:])
                    gt_est=self.refinement(gt_srf_est)
                    gt_est_numpy=gt_est.data.cpu().detach().numpy()[0].transpose(1,2,0) #转为numpy
                    sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(self.gt,gt_est_numpy, self.args.scale_factor)
                    L1=np.mean( np.abs( self.gt - gt_est_numpy ))
                    information2="生成gt与目标gt\n L1 {} sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(L1,sam,psnr,ergas,cc,rmse,Ssim,Uqi)
                    print(information2)
                    print('************')
                    
                    
                    if sam < flag_best[0] and psnr > flag_best[1]:         #只有比之前的结果好 
                        flag_best[0]=sam
                        flag_best[1]=psnr
                        flag_best[2]=gt_est  #保存四维tensor
                        weight=W.data.cpu().detach().numpy()[0,:,0,0]
                        information_A=information0
                        information_a=information1
                        information_b=information2
                        
                        
                        
        #保存最好的结果
        scipy.io.savemat(os.path.join(self.args.expr_dir, 'Out.mat'), {'Out':flag_best[2].data.cpu().numpy()[0].transpose(1,2,0)})
        #保存精度   
        file_name = os.path.join(self.args.expr_dir, 'Stage2_Spe.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write(str(weight))
            opt_file.write('\n')
            opt_file.write(information_A)
            opt_file.write('\n')
            opt_file.write(information_a)
            opt_file.write('\n')
            opt_file.write(information_b)
            opt_file.write('\n')
                
        return flag_best[2] #返回的是四维tensor
        
