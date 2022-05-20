"""
Networks of the Pix2PixHD Model
@author: Hyun-Jin Jeong (https://jeonghyunjin.com, jeong_hj@khu.ac.kr)
Reference:
1) https://github.com/JeongHyunJin/Pix2PixCC
2) https://arxiv.org/pdf/2204.12068.pdf
"""

#==============================================================================

import torch
import torch.nn as nn
from pix2pixCC_Utils import get_grid, get_norm_layer, get_pad_layer
import torch.nn.functional as F
import numpy as np


#==============================================================================
# [1] Generative Network

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        #----------------------------------------------------------------------        
        input_ch = opt.input_ch
        output_ch = opt.target_ch
        n_gf = opt.n_gf
        norm = get_norm_layer(opt.norm_type)
        act = Mish()
        pad = get_pad_layer(opt.padding_type)
        trans_conv = opt.trans_conv

        #----------------------------------------------------------------------
        model = []
        
        model += [pad(3), nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0), norm(n_gf), act]

        for _ in range(opt.n_downsample):
            model += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=5, padding=2, stride=2), norm(2 * n_gf), act]
            n_gf *= 2

        for _ in range(opt.n_residual):
            model += [ResidualBlock(n_gf, pad, norm, act)]

        for n_up in range(opt.n_downsample):
            #------------------------------------------------------------------
            if trans_conv == True:
                model += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                          norm(n_gf//2), act]
            else:
                model += [nn.UpsamplingBilinear2d(scale_factor=2)]
                model += [pad(1), nn.Conv2d(n_gf, n_gf//2, kernel_size=3, padding=0, stride=1), norm(n_gf//2), act]
            #------------------------------------------------------------------                        
            n_gf //= 2
        
        
        model += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)
        #----------------------------------------------------------------------
        
        print(self)
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    def forward(self, x):
        return self.model(x)

#------------------------------------------------------------------------------
        
class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


#------------------------------------------------------------------------------

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



#==============================================================================
# [2] Discriminative Network

class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        
        #----------------------------------------------------------------------
        if opt.ch_balance > 0:
            ch_ratio = np.float(opt.input_ch)/np.float(opt.target_ch)
            ch_ratio *= opt.ch_balance
            if ch_ratio > 1:
                input_channel = opt.input_ch + opt.target_ch*np.int(ch_ratio)                            
            elif ch_ratio < 1:
                input_channel = opt.input_ch*np.int(1/ch_ratio) + opt.target_ch
            else:
                input_channel = opt.input_ch + opt.target_ch
        else:
            input_channel = opt.input_ch + opt.target_ch
        
        #----------------------------------------------------------------------
        act = nn.LeakyReLU(0.2, inplace=True)
        n_df = opt.n_df
        norm = nn.InstanceNorm2d
        
        #----------------------------------------------------------------------
        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))
            
        #----------------------------------------------------------------------
        
        
    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input

#------------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        #----------------------------------------------------------------------
        for i in range(opt.n_D):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator(opt))
        self.n_D = opt.n_D

        #----------------------------------------------------------------------
        print(self)
        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
                
        return result



#==============================================================================
# [3] Objective (Loss) functions

class Loss(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:0' if opt.gpu_ids != -1 else 'cpu:0')
        self.dtype = torch.float16 if opt.data_type == 16 else torch.float32
        
        self.MSE = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = opt.n_D


    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0
        
        fake = G(input)
        
        
        #----------------------------------------------------------------------
        # [3-1] Get Real and Fake (Generated) pairs and features 
        
        if self.opt.ch_balance > 0:        
            
            real_pair = torch.cat((input, target), dim=1)
            fake_pair = torch.cat((input, fake.detach()), dim=1)
            
            ch_plus = 0
            ch_ratio = np.float(self.opt.input_ch)/np.float(self.opt.target_ch)
            ch_ratio *= self.opt.ch_balance
            if ch_ratio > 1:
                for dr in range(np.int(ch_ratio)-1):
                    real_pair = torch.cat((real_pair, target), dim=1)
                    fake_pair = torch.cat((fake_pair, fake.detach()), dim=1)
                    ch_plus += self.opt.target_ch                         
            
            elif ch_ratio < 1:                
                for _ in range(np.int(1/ch_ratio)-1):
                    real_pair = torch.cat((input, real_pair), dim=1)
                    fake_pair = torch.cat((input, fake_pair), dim=1)
                    ch_plus += self.opt.input_ch
                
            else:
                pass
            
            real_features = D(real_pair)
            fake_features = D(fake_pair)
        else:
            real_features = D(torch.cat((input, target), dim=1))
            fake_features = D(torch.cat((input, fake.detach()), dim=1))
        
        
        #----------------------------------------------------------------------
        # [3-2] Compute LSGAN loss for the discriminator
        
        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device, self.dtype)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device, self.dtype)

            loss_D += (self.MSE(real_features[i][-1], real_grid) +
                       self.MSE(fake_features[i][-1], fake_grid)) * 0.5
        
        
        #----------------------------------------------------------------------
        # [3-3] Compute LSGAN loss and Feature Matching loss for the generator
        
        if self.opt.ch_balance > 0:  
            fake_pair = torch.cat((input, fake), dim=1)
            
            if ch_ratio > 1:
                for _ in range(np.int(ch_ratio)-1):
                    fake_pair = torch.cat((fake_pair, fake), dim=1)
            elif ch_ratio < 1:
                for _ in range(np.int(1/ch_ratio)-1):
                    fake_pair = torch.cat((input, fake_pair), dim=1)
            else:
                pass
            
            fake_features = D(fake_pair)
        else:
            fake_features = D(torch.cat((input, fake), dim=1))
            
        
        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1], is_real=True).to(self.device, self.dtype)
            loss_G += self.MSE(fake_features[i][-1], real_grid) * 0.5 * self.opt.lambda_LSGAN
            
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())
                
            loss_G += loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM
        
        
        #----------------------------------------------------------------------
        # [3-4] Compute Correlation Coefficient loss for the generator
        
        for i in range(self.opt.n_CC):
            real_down = target.to(self.device, self.dtype)
            fake_down = fake.to(self.device, self.dtype)
            for _ in range(i):
                real_down = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(real_down)
                fake_down = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(fake_down)
            
            loss_CC = self.__Inspector(real_down, fake_down)
            loss_G += loss_CC * (1.0 / self.opt.n_CC) * self.opt.lambda_CC
        
        #----------------------------------------------------------------------
        return loss_D, loss_G, target, fake
    
    
    
#==============================================================================
# [4] Inspector
    
    def __Inspector(self, target, fake):
                
        rd = target - torch.nanmean(target)
        fd = fake - torch.nanmean(fake)
        
        r_num = torch.nansum(rd * fd)
        r_den = torch.sqrt(torch.nansum(rd ** 2)) * torch.sqrt(torch.nansum(fd ** 2))
        PCC_val = r_num/(r_den + self.opt.eps)
        
        #----------------------------------------------------------------------
        if self.opt.ccc == True:
            numerator = 2*PCC_val*torch.std(target)*torch.std(fake)
            denominator = (torch.var(target) + torch.var(fake)
                           + (torch.nanmean(target) - torch.nanmean(fake))**2)
            
            CCC_val = numerator/(denominator + self.opt.eps)
            loss_CC = (1.0 - CCC_val)
        
        else:
            loss_CC = (1.0 - PCC_val)
            
        #----------------------------------------------------------------------
        return loss_CC
    
    
#==============================================================================
