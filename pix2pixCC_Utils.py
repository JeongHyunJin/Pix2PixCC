"""
Utility programs of the Pix2PixCC Model
@author: Hyun-Jin Jeong (https://jeonghyunjin.com, jeong_hj@khu.ac.kr)
Reference:
1) https://github.com/JeongHyunJin/Pix2PixCC
2) https://arxiv.org/pdf/2204.12068.pdf
"""

#==============================================================================

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from functools import partial


#==============================================================================
# [1] True or False grid

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid



#==============================================================================
# [2] Set the Normalization method for the input layer

def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d,affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer



#==============================================================================
# [3] Set the Padding method for the input layer

def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer



#==============================================================================
# [4] Save or Report the model results 

class Manager(object):
    def __init__(self, opt):
        self.opt = opt
        self.dtype = opt.data_type

    #--------------------------------------------------------------------------      
    @staticmethod
    def report_loss(package):
        print("Epoch: {} [{:.{prec}}%] Current_step: {} D_loss: {:.{prec}}  G_loss: {:.{prec}}".
              format(package['Epoch'], package['current_step']/package['total_step'] * 100, package['current_step'],
                     package['D_loss'], package['G_loss'], prec=4))

    
    #--------------------------------------------------------------------------
    def adjust_dynamic_range(self, data, drange_in, drange_out):
        if drange_in != drange_out:
            if self.dtype == 32:
                scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                            np.float32(drange_in[1]) - np.float32(drange_in[0]))
                bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            elif self.dtype == 16:
                scale = (np.float16(drange_out[1]) - np.float16(drange_out[0])) / (
                            np.float16(drange_in[1]) - np.float16(drange_in[0]))
                bias = (np.float16(drange_out[0]) - np.float16(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    
    #--------------------------------------------------------------------------
    def tensor2image(self, image_tensor):
        np_image = image_tensor[0].squeeze().cpu().float().numpy()
        if len(np_image.shape) == 3:
            np_image = np.transpose(np_image, (1, 2, 0))  # HWC
        else:
            pass

        np_image = self.adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image
    
    
    #--------------------------------------------------------------------------
    def save_image(self, image_tensor, path):
        Image.fromarray(self.tensor2image(image_tensor)).save(path, self.opt.image_mode)

    
    #--------------------------------------------------------------------------        
    def save(self, package, image=False, model=False):
        if image:
            path_real = os.path.join(self.opt.image_dir, str(package['current_step']) + '_' + 'real.png')
            path_fake = os.path.join(self.opt.image_dir, str(package['current_step']) + '_' + 'fake.png')
            self.save_image(package['target_tensor'], path_real)
            self.save_image(package['generated_tensor'], path_fake)

        elif model:
            path_D = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'D.pt')
            path_G = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'G.pt')
            torch.save(package['D_state_dict'], path_D)
            torch.save(package['G_state_dict'], path_G)

    
    #--------------------------------------------------------------------------
    def __call__(self, package):
        if package['current_step'] % self.opt.display_freq == 0:
            self.save(package, image=True)

        if package['current_step'] % self.opt.report_freq == 0:
            self.report_loss(package)

        if package['current_step'] % self.opt.save_freq == 0:
            self.save(package, model=True)

    
    #--------------------------------------------------------------------------
    
    

#==============================================================================
# Set the initial conditions of weights

def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)
        
        
#==============================================================================
