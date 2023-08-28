# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:26:59 2022

@author: 13572
"""
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
import os
import scipy


def init_weights(net, init_type, gain):
    print('in init_weights')
    def init_func(m):
        classname = m.__class__.__name__
        #print(classname,m,'_______')
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    
    print('Spectral_upsample initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net,device, init_type, init_gain,initializer):
    print('in init_net')
    net.to(device)  #gpu_ids[0] 是 gpu_ids列表里面的第一个int值
    if initializer :
        #print(2,initializer)
        init_weights(net,init_type, init_gain)
    else:
        print('Spectral_downsample with default initialize')
    return net

    


#########################  refinement subnetwork  #########################
def def_refinement(input_channel,output_channel,device,block_num,init_type='kaiming', init_gain=0.02,initializer=True):
    
    #input_channel：lr_hsi的波段数
    #output_channel：self.begin输出的波段数
    #endmember_num：端元个数
    net = refinement(input_channel,  output_channel, block_num)

    return init_net(net,device, init_type, init_gain,initializer)  #init_net(net, args.device, init_type, init_gain ,initializer)
    
class refinement(nn.Module):
    def __init__(self,input_channel,output_channel,block_num): 

        #input_channel：lr_hsi的波段数
        #output_channel：输出的波段数
       
        super().__init__()
        
        middle=30 #40
        
        self.begin=nn.Conv2d(in_channels=input_channel,out_channels=middle,kernel_size=1,stride=1) 
        
        layer = []
        for i in range(block_num):
            layer.append(
                                spectral_res_block(middle)
                              ) 
        self.middle=nn.Sequential(*layer)
        #self.middle=spectral_res_block(60) #不改变波段数
        
        ###最后输出的波段数量是int(endmember_num/4)
        self.end=nn.Conv2d(in_channels=middle,out_channels=output_channel,kernel_size=1,stride=1)
    
    def forward(self,input):
        output1=self.begin(input)
        #print("output1~",output1.shape) torch.Size([1, 60, 240, 240])
        output2=self.middle(output1)
        #print("output2~",output2.shape) torch.Size([1, 60, 240, 240])
        output3=self.end(output2)
        #print("output3~",output3.shape)   torch.Size([1, 100, 240, 240])
        
        
        return output3 #1 endnum  h w
    

class spectral_res_block(nn.Module):
    def __init__(self,input_channel): #input_channel 60
        super().__init__()
        self.one=nn.Sequential(
        #nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1,padding=0),
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel),kernel_size=1,stride=1,padding=0),

        nn.ReLU(inplace=True),
        #nn.Conv2d(in_channels=int(input_channel/3),out_channels=input_channel,kernel_size=1,stride=1,padding=0) ,
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel),kernel_size=1,stride=1,padding=0),

        nn.ReLU(inplace=True)
                                )
        
    def forward(self,input):
        identity_data = input
        output = self.one(input) # 60
        output = torch.add(output, identity_data) # 60
        #output = nn.ReLU(inplace=True)(output) # 60
        return output   

#########################  refinement subnetwork  #########################

#########################  MSAF  #########################
def def_MSAF(input_channel,device,init_type='kaiming', init_gain=0.02,initializer=True):
    
    #input_channel：输入lr_msi的波段数
    
    net = MSAF(input_channel)

    return init_net(net,device, init_type, init_gain,initializer)
    
    
class MSAF(nn.Module):
    def __init__(self,input_channel):
        super().__init__()
        
        self.three=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=(1,3),stride=1,padding=(0,1)) ,
        nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=(3,1),stride=1,padding=(1,0)) ,
        nn.ReLU(inplace=True)
                                )
        #padding=(x,y) x是在高度上增高,y是在宽度上变宽
        self.five=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=(1,5),stride=1,padding=(0,2)),
        nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=(5,1),stride=1,padding=(2,0)) ,
        nn.ReLU(inplace=True)
                                )
        
        self.seven=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=(1,7),stride=1,padding=(0,3)) ,
        nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=(7,1),stride=1,padding=(3,0)),
        nn.ReLU(inplace=True)
                                )
    
    
        self.point_wise1=nn.Sequential(
                    nn.Conv2d(in_channels=input_channel*2, out_channels=input_channel,kernel_size=1,stride=1,padding=0),             
                    nn.ReLU(inplace=True)
                    #nn.AdaptiveAvgPool2d(1)
                )
        
        self.point_wise2=nn.Sequential(
                    nn.Conv2d(in_channels=input_channel*2, out_channels=input_channel,kernel_size=1,stride=1,padding=0),             
                    nn.ReLU(inplace=True)
                    #nn.AdaptiveAvgPool2d(1)
                )
        
        self.point_wise3=nn.Sequential(
                    nn.Conv2d(in_channels=input_channel*2, out_channels=input_channel,kernel_size=1,stride=1,padding=0),             
                    nn.ReLU(inplace=True)
                    #nn.AdaptiveAvgPool2d(1)
                )

        self.point_wise4=nn.Sequential(
                    nn.Conv2d(in_channels=input_channel, out_channels=input_channel,kernel_size=1,stride=1,padding=0),             
                    nn.Sigmoid()
                    )
                   
                
    def forward(self,lr_msi_fhsi,lr_msi_fmsi):
        
        input=torch.add(lr_msi_fhsi,lr_msi_fmsi)
        
        scale3=self.three(input)
        scale5=self.five(input)
        scale7=self.seven(input)
        #print('scale3.shape {}  scale5.shape {} scale7.shape {}'.format(scale3.shape,scale5.shape,scale7.shape))
        pair_35=torch.cat((scale3,scale5),dim=1) 
        pair_57=torch.cat((scale5,scale7),dim=1) 
        pair_37=torch.cat((scale3,scale7),dim=1) 
            
        #print('pair_35.shape {}  pair_57.shape {} pair_37.shape {}'.format(pair_35.shape,pair_57.shape,pair_37.shape))

        out_wise1=self.point_wise1(pair_35)
        out_wise2=self.point_wise2(pair_37)
        out_wise3=self.point_wise3(pair_57)
        
        #print('out_wise1.shape {}  out_wise2.shape {} out_wise3.shape {}'.format(out_wise1.shape,out_wise2.shape,out_wise3.shape))

        
        spectral_1=torch.add(nn.AdaptiveAvgPool2d(1)(out_wise1), nn.AdaptiveMaxPool2d(1)(out_wise1))
        spectral_2=torch.add(nn.AdaptiveAvgPool2d(1)(out_wise2), nn.AdaptiveMaxPool2d(1)(out_wise2))
        spectral_3=torch.add(nn.AdaptiveAvgPool2d(1)(out_wise3), nn.AdaptiveMaxPool2d(1)(out_wise3))

        #print('spectral_1.shape {}  spectral_2.shape {} spectral_3.shape {}'.format(spectral_1.shape,spectral_2.shape,spectral_3.shape))


        spectral_sum=spectral_1 + spectral_2 + spectral_3
        
        #print('spectral_sum.shape {}  '.format(spectral_sum.shape))


        W = self.point_wise4(spectral_sum)
        
        #print('W.shape {}  '.format(W.shape))

        fused_lr_msi= W * lr_msi_fhsi + (1-W) * lr_msi_fmsi
        
        #print('W * lr_msi_fhsi.shape {}  '.format((W * lr_msi_fhsi).shape))
        #print('(1-W) * lr_msi_fmsi.shape {}  '.format(((1-W) * lr_msi_fmsi).shape))

        return fused_lr_msi,W
    
#########################  MSAF  #########################



#########################  SDG subnetwork  #########################
    
def def_SDG(input_channel,output_channel,device,init_type='kaiming', init_gain=0.02,initializer=True):
    
    #input_channel：lr_hsi的波段数
    #output_channel：self.begin输出的波段数
    #endmember_num：端元个数
    net = SDG(input_channel, output_channel)

    return init_net(net,device, init_type, init_gain,initializer)  #init_net(net, args.device, init_type, init_gain ,initializer)
    
class SDG(nn.Module):
    def __init__(self,input_channel,output_channel): 

        #input_channel：输入的波段数
        #output_channel：输出的波段数
       
        super().__init__()
        middle=5
        self.conv=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=middle,kernel_size=(1,1),stride=1) ,
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=middle,out_channels=output_channel,kernel_size=(1,1),stride=1) ,
                                )
    
    def forward(self,input):
        
        output=self.conv(input)
                
        return output

#########################  SDG subnetwork  #########################



    