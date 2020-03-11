# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import logging
import sys
import time
import math

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np


def initialize(m):
    """FUNCTION TO INITILIZE NETWORK PARAMETERS

    Arg:
        m (torch.nn.Module): torch nn module instance
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
        else:
            logging.info("ERROR: " + name)


class TwoSidedDilConv1d(nn.Module):
    """1D TWO-SIDED DILATED CONVOLUTION"""

    def __init__(self, in_dim=55, kernel_size=3, layers=2):
        super(TwoSidedDilConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.layers = layers
        self.rec_field = self.kernel_size**self.layers
        self.padding = int((self.rec_field-1)/2)
        self.conv = nn.ModuleList()
        for i in range(self.layers):
            if i > 0:
                self.conv += [nn.Conv1d(self.in_dim*(self.kernel_size**(i)), \
                    self.in_dim*(self.kernel_size**(i+1)), self.kernel_size, dilation=self.kernel_size**i)]
            else:
                self.conv += [nn.Conv1d(self.in_dim, self.in_dim*(self.kernel_size**(i+1)), \
                    self.kernel_size, padding=self.padding)]

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """
        x = self.conv[0](x)
        for i in range(1,self.layers):
            x = self.conv[i](x)

        return x


class CausalDilConv1d(nn.Module):
    """1D Causal DILATED CONVOLUTION"""

    def __init__(self, in_dim=11, kernel_size=2, layers=2):
        super(CausalDilConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.layers = layers
        self.padding_list = [self.kernel_size**(i+1)-self.kernel_size**(i) for i in range(self.layers)]
        logging.info(self.padding_list)
        self.padding = sum(self.padding_list)
        self.rec_field = self.padding + 1
        self.conv = nn.ModuleList()
        for i in range(self.layers):
            if i > 0:
                self.conv += [nn.Conv1d(self.in_dim*(sum(self.padding_list[:i])+1), \
                    self.in_dim*(sum(self.padding_list[:i+1])+1), self.kernel_size, \
                        dilation=self.kernel_size**i)]
            else:
                self.conv += [nn.Conv1d(self.in_dim, self.in_dim*(sum(self.padding_list[:i+1])+1), \
                    self.kernel_size, padding=self.padding)]

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """
        x = self.conv[0](x)
        for i in range(1,self.layers):
            x = self.conv[i](x)

        return x[:,:,:-self.padding]


def sampling_vae_laplace(param, lat_dim=None):
    if lat_dim is None:
        lat_dim = int(param.shape[1]/2)
    mu = param[:,:lat_dim]
    sigma = param[:,lat_dim:]
    eps = torch.empty(param.shape[0], lat_dim).cuda().uniform_(-0.4999,0.5)
    return mu - torch.exp(sigma) * eps.sign() * torch.log1p(-2*eps.abs()) # log_scale


def sampling_vae_laplace_batch(param, lat_dim=None):
    if lat_dim is None:
        lat_dim = int(param.shape[1]/2)
    mu = param[:,:,:lat_dim]
    sigma = param[:,:,lat_dim:]
    eps = torch.empty(param.shape[0], param.shape[1], lat_dim).cuda().uniform_(-0.4999,0.5)
    return mu - torch.exp(sigma) * eps.sign() * torch.log1p(-2*eps.abs()) # log_scale


def loss_vae_laplace(param, clip=False, lat_dim=None):
    if lat_dim is None:
        lat_dim = int(param.shape[1]/2)
    mu = param[:,:lat_dim]
    sigma = param[:,lat_dim:]
    #if clip and torch.min(sigma) < -10.708206508753178232789577606809: #1e-9
    if clip and torch.min(sigma) < -14.162084148244246758816564788835: #1e-12
        #sigma = torch.clamp(sigma,min=-7.2543288692621097067625904247823) #1e-6
        #sigma = torch.clamp(sigma,min=-10.708206508753178232789577606809) #1e-9
        sigma = torch.clamp(sigma,min=-14.162084148244246758816564788835) #1e-12
    mu_abs = mu.abs()
    scale = torch.exp(sigma)
    return torch.mean(torch.sum(-sigma+scale*torch.exp(-mu_abs/scale)+mu_abs-1,1)) # log_scale


class GRU_RNN(nn.Module):
    """GRU-RNN for FEATURE MAPPING

    Args:
        in_dim (int): input dimension
        out_dim (int): RNN output dimension
        hidden_units (int): GRU hidden units amount
        hidden_layers (int): GRU hidden layers amount
        kernel_size (int): kernel size for input convolutional layers
        dilation_size (int): dilation size for input convolutional layers
        do_prob (float): drop-out probability
        scale_in_flag (bool): flag to use input normalization layer
        scale_out_flag (bool): flag to use output de-normalization layer
        scale_in_out_flag (bool): flag to use output normalization layer for after performing input norm.
                                                                        (e.g., for Gaussian noise injection)
        [Weights & biases of norm/de-norm layers should be set with training data stats]
    """

    def __init__(self, in_dim=39, out_dim=35, hidden_units=1024, hidden_layers=1, kernel_size=3, \
            dilation_size=2, do_prob=0, scale_in_flag=True, scale_out_flag=True, \
                causal_conv=False, spk_dim=None):
        super(GRU_RNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.scale_in_flag = scale_in_flag
        self.scale_out_flag = scale_out_flag
        self.causal_conv = causal_conv
        self.spk_dim = spk_dim

        # Normalization layer
        if self.scale_in_flag:
            self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)

        # Dilated two-sides convolution layers: -4/0/+4 frames with kernel_size = 3 and dilation_size = 2
        # or Dilated causal convolution layers: -8/0 frames with kernel_size = 3 and dilation_size = 2
        if not self.causal_conv:
            self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                            layers=self.dilation_size)
        else:
            self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                                        layers=self.dilation_size)
        self.receptive_field = self.conv.rec_field
        self.tot_in_dim = self.in_dim*self.receptive_field+self.out_dim
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if self.do_prob > 0 and self.hidden_layers > 1:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, \
                                dropout=self.do_prob, batch_first=True)
        else:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, batch_first=True)

        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)
        self.out_1 = nn.Conv1d(self.hidden_units, self.out_dim, 1)

        # Denormalization layer
        if self.scale_out_flag:
            self.scale_out = nn.Conv1d(self.out_dim, self.out_dim, 1)

    def forward(self, x, y_in, h_in=None, do=False):
        """Forward calculation

        Args:
            x (Variable): float tensor variable with the shape  (T x C_in)

        Return:
            (Variable): float tensor variable with the shape (T x C_out)
        """
        if len(x.shape) > 2:
            batch_flag = True
            T = x.shape[1]
            # Normalization layer
            if self.scale_in_flag:
                x_in = self.scale_in(x.transpose(1,2)) # B x T x C -> B x C x T
            else:
                x_in = x.transpose(1,2) # B x T x C -> B x C x T
        else:
            batch_flag = False
            T = x.shape[0]
            # Normalization layer
            if self.scale_in_flag:
                x_in = self.scale_in(torch.unsqueeze(x.transpose(0,1),0)) # T x C -> C x T -> B x C x T
            else:
                x_in = torch.unsqueeze(x.transpose(0,1),0) # T x C -> C x T -> B x C x T

        # Dilated two-sides convolution layers: -4/0/+4 frames with kernel_size = 3 and dilation_size = 2
        # or Dilated causal convolution layers: -8/0 frames with kernel_size = 3 and dilation_size = 2
        if self.do_prob > 0 and do:
            x_conv = self.conv_drop(self.conv(x_in).transpose(1,2)) # T x C --> B x C x T --> B x T x C
        else:
            x_conv = self.conv(x_in).transpose(1,2) # T x C --> B x C x T --> B x T x C

        # GRU and AR layers
        # 1st frame
        if h_in is not None:
            out, h = self.gru(torch.cat((x_conv[:,:1],y_in),2), h_in) # B x T x C
        else:
            out, h = self.gru(torch.cat((x_conv[:,:1],y_in),2)) # B x T x C
        if self.do_prob > 0 and do:
            y_in = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
        else:
            y_in = self.out_1(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
        if self.spk_dim is not None:
            y_in = torch.cat((F.selu(y_in[:,:,:self.spk_dim]),y_in[:,:,self.spk_dim:]),2)
        trj = y_in
        # 2nd-Tth frame
        if self.spk_dim is None:
            if self.do_prob > 0 and do:
                for i in range(1,T):
                    out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                    y_in = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
                    trj = torch.cat((trj, y_in), 1)
            else:
                for i in range(1,T):
                    out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                    y_in = self.out_1(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                    trj = torch.cat((trj, y_in), 1)
        else:
            if self.do_prob > 0 and do:
                for i in range(1,T):
                    out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                    y_in = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
                    y_in = torch.cat((F.selu(y_in[:,:,:self.spk_dim]),y_in[:,:,self.spk_dim:]),2)
                    trj = torch.cat((trj, y_in), 1)
            else:
                for i in range(1,T):
                    out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                    y_in = self.out_1(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
                    y_in = torch.cat((F.selu(y_in[:,:,:self.spk_dim]),y_in[:,:,self.spk_dim:]),2)
                    trj = torch.cat((trj, y_in), 1)

        # Denormalization layer
        if self.scale_out_flag:
            if batch_flag:
                # B x T x C -> B x C x T -> B x T x C
                trj_out = self.scale_out(trj.transpose(1,2)).transpose(1,2)
            else:
                # B x T x C -> B x C x T -> T x C
                trj_out = torch.squeeze(self.scale_out(trj.transpose(1,2)).transpose(1,2),0)
            
            return trj_out, y_in, h
        else:
            if not batch_flag:
                trj = trj.view(-1,self.out_dim)
            return trj, y_in, h


class MCDloss(nn.Module):
    """ spectral loss based on mel-cepstrum distortion (MCD) """
    def __init__(self):
        super(MCDloss, self).__init__()
        self.frac10ln2 = (10.0/2.3025850929940456840179914546844)
        self.sqrt2 = 1.4142135623730950488016887242097
    
    def forward(self, x, y, twf=None, L2=False):
        """
            twf is time-warping function, none means exact same time-alignment
            L2 means using squared loss (L2-based loss), false means using abs./L1-based loss; default false
        """
        if twf is None:
            if not L2:
                mcd = self.frac10ln2*self.sqrt2*torch.sum(torch.abs(x-y),1)
            else:
                mcd = self.frac10ln2*torch.sqrt(2.0*torch.sum((x-y).pow(2),1))
        else:
            if not L2:
                mcd = self.frac10ln2*self.sqrt2*torch.sum(torch.abs(torch.index_select(x,0,twf)-y),1)
            else:
                mcd = self.frac10ln2*torch.sqrt(2.0*torch.sum((torch.index_select(x,0,twf)-y).pow(2),1))
        mcd_sum = torch.sum(mcd)
        mcd_mean = torch.mean(mcd)
        mcd_std = torch.std(mcd)
        return mcd_sum, mcd_mean, mcd_std


class GRU_RNN_STOCHASTIC(nn.Module):
    """STOCHASTIC GRU-RNN for FEATURE MAPPING

    Args:
        in_dim (int): input dimension
        out_dim (int): RNN output dimension
        hidden_units (int): GRU hidden units amount
        hidden_layers (int): GRU hidden layers amount
        kernel_size (int): kernel size for input convolutional layers
        dilation_size (int): dilation size for input convolutional layers
        do_prob (float): drop-out probability
        scale_in_flag (bool): flag to use input normalization layer
        scale_out_flag (bool): flag to use output de-normalization layer
        [Weights & biases of norm/de-norm layers should be set with training data stats]
    """

    def __init__(self, in_dim=55, out_dim=50, hidden_units=1024, hidden_layers=1, kernel_size=3, \
            dilation_size=2, do_prob=0, spk_dim=None, scale_in_flag=True, scale_out_flag=True, \
            causal_conv=False, arparam=True):
        super(GRU_RNN_STOCHASTIC, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.dilation_size = dilation_size
        self.do_prob = do_prob
        self.scale_in_flag = scale_in_flag
        self.scale_out_flag = scale_out_flag
        self.spk_dim = spk_dim
        self.causal_conv = causal_conv
        if self.spk_dim is not None:
            self.mu_dim = self.spk_dim+self.out_dim
        else:
            self.mu_dim = self.out_dim
        self.arparam = arparam

        # Normalization layer
        if self.scale_in_flag:
            self.scale_in = nn.Conv1d(self.in_dim, self.in_dim, 1)

        # Dilated two-sides convolution layers: -4/0/+4 frames with kernel_size = 3 and dilation_size = 2
        # or Dilated causal convolution layers: -8/0 frames with kernel_size = 3 and dilation_size = 2
        if not self.causal_conv:
            self.conv = TwoSidedDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                            layers=self.dilation_size)
        else:
            self.conv = CausalDilConv1d(in_dim=self.in_dim, kernel_size=self.kernel_size, \
                            layers=self.dilation_size)
        self.receptive_field = self.conv.rec_field
        if self.arparam:
            if self.spk_dim is not None:
                self.tot_in_dim = self.in_dim*self.receptive_field+self.out_dim*2+self.spk_dim
            else:
                self.tot_in_dim = self.in_dim*self.receptive_field+self.out_dim*2
        else:
            if self.spk_dim is not None:
                self.tot_in_dim = self.in_dim*self.receptive_field+self.out_dim+self.spk_dim
            else:
                self.tot_in_dim = self.in_dim*self.receptive_field+self.out_dim
        if self.do_prob > 0:
            self.conv_drop = nn.Dropout(p=self.do_prob)

        # GRU layer(s)
        if self.do_prob > 0 and self.hidden_layers > 1:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, dropout=self.do_prob, \
                                batch_first=True)
        else:
            self.gru = nn.GRU(self.tot_in_dim, self.hidden_units, self.hidden_layers, batch_first=True)

        if self.do_prob > 0:
            self.gru_drop = nn.Dropout(p=self.do_prob)
        if self.spk_dim is not None:
            self.out_1 = nn.Conv1d(self.hidden_units, self.spk_dim+self.out_dim*2, 1)
        else:
            self.out_1 = nn.Conv1d(self.hidden_units, self.out_dim*2, 1)

        # Denormalization layer
        if self.scale_out_flag:
            self.scale_out = nn.Conv1d(self.out_dim, self.out_dim, 1)

    def forward(self, x, y_in, h_in=None, noise=0, do=False, sampling=True):
        """Forward calculation

        Args:
            x (Variable): float tensor variable with the shape  (T x C_in) or (B x T x C_in)

        Return:
            (Variable): float tensor variable with the shape (T x C_out) or (B x T x C_out)
        """
        if len(x.shape) > 2:
            batch_flag = True
            T = x.shape[1]
            # Normalization layer
            if self.scale_in_flag:
                x_in = self.scale_in(x.transpose(1,2)) # B x T x C -> B x C x T
            else:
                x_in = x.transpose(1,2) # B x T x C -> B x C x T
        else:
            batch_flag = False
            T = x.shape[0]
            # Normalization layer
            if self.scale_in_flag:
                x_in = self.scale_in(torch.unsqueeze(x.transpose(0,1),0)) # T x C -> C x T -> B x C x T
            else:
                x_in = torch.unsqueeze(x.transpose(0,1),0) # T x C -> C x T -> B x C x T

        if noise > 0:
            x_noise = torch.normal(mean=0, \
                std=noise*torch.ones(x_in.shape[0],x_in.shape[1],x_in.shape[2])).cuda()
            x_in = x_in + x_noise # B x C x T

        # Dilated two-sides convolution layers: -4/0/+4 frames with kernel_size = 3 and dilation_size = 2
        # or Dilated causal convolution layers: -8/0 frames with kernel_size = 3 and dilation_size = 2
        if self.do_prob > 0 and do:
            x_conv = self.conv_drop(self.conv(x_in).transpose(1,2)) # T x C --> B x C x T --> B x T x C
        else:
            x_conv = self.conv(x_in).transpose(1,2) # T x C --> B x C x T --> B x T x C

        # GRU and AR layers
        # 1st frame
        if h_in is not None:
            out, h = self.gru(torch.cat((x_conv[:,:1],y_in),2), h_in) # B x T x C
        else:
            out, h = self.gru(torch.cat((x_conv[:,:1],y_in),2)) # B x T x C
        if self.do_prob > 0 and do:
            out = self.gru_drop(out)
        out = self.out_1(out.transpose(1,2)).transpose(1,2) # B x T x C -> B x C x T -> B x T x C
        if self.spk_dim is not None:
            out_param = torch.cat((F.selu(out[:,:,:self.spk_dim]),out[:,:,self.spk_dim:self.mu_dim],\
                                    F.logsigmoid(out[:,:,self.mu_dim:])),2)
            if not self.arparam:
                if sampling:
                    out = sampling_vae_laplace_batch(out_param[:,:,self.spk_dim:], lat_dim=self.out_dim)
                else:
                    out = out[:,:,self.spk_dim:self.mu_dim]
        else:
            out_param = torch.cat((out[:,:,:self.mu_dim],F.logsigmoid(out[:,:,self.mu_dim:])),2)
            if not self.arparam:
                if sampling:
                    out = sampling_vae_laplace_batch(out_param, lat_dim=self.out_dim)
                else:
                    out = out[:,:,:self.mu_dim]
        trj_out_param = out_param
        if not self.arparam:
            trj_out = out
        if self.arparam:
            y_in = out_param
        else:
            if self.spk_dim is not None:
                y_in = torch.cat((out_param[:,:,:self.spk_dim],out),2)
            else:
                y_in = out
        # 2nd-Tth frame
        if self.do_prob > 0 and do:
            if self.arparam:
                if self.spk_dim is not None:
                    for i in range(1,T):
                        out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                        out = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
                        out_param = torch.cat((F.selu(out[:,:,:self.spk_dim]),\
                                    out[:,:,self.spk_dim:self.mu_dim],F.logsigmoid(out[:,:,self.mu_dim:])),2)
                        trj_out_param = torch.cat((trj_out_param, out_param),1)
                        y_in = out_param
                else:
                    for i in range(1,T):
                        out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                        out = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
                        out_param = torch.cat((out[:,:,:self.mu_dim],F.logsigmoid(out[:,:,self.mu_dim:])),2)
                        trj_out_param = torch.cat((trj_out_param, out_param),1)
                        y_in = out_param
            else:
                if sampling:
                    if self.spk_dim is not None:
                        for i in range(1,T):
                            out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                            out = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
                            out_param = torch.cat((F.selu(out[:,:,:self.spk_dim]),\
                                            out[:,:,self.spk_dim:self.mu_dim],\
                                                F.logsigmoid(out[:,:,self.mu_dim:])),2)
                            out = sampling_vae_laplace_batch(out_param[:,:,self.spk_dim:],lat_dim=self.out_dim)
                            trj_out_param = torch.cat((trj_out_param, out_param),1)
                            trj_out = torch.cat((trj_out, out), 1)
                            y_in = torch.cat((out_param[:,:,:self.spk_dim], out),2)
                    else:
                        for i in range(1,T):
                            out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                            out = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
                            out_param = torch.cat((out[:,:,:self.mu_dim],\
                                            F.logsigmoid(out[:,:,self.mu_dim:])),2)
                            out = sampling_vae_laplace_batch(out_param, lat_dim=self.out_dim)
                            trj_out_param = torch.cat((trj_out_param, out_param),1)
                            trj_out = torch.cat((trj_out, out), 1)
                            y_in = out
                else:
                    if self.spk_dim is not None:
                        for i in range(1,T):
                            out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                            out = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
                            out_param = torch.cat((F.selu(out[:,:,:self.spk_dim]),\
                                            out[:,:,self.spk_dim:self.mu_dim],\
                                                F.logsigmoid(out[:,:,self.mu_dim:])),2)
                            out = out[:,:,self.spk_dim:self.mu_dim]
                            trj_out_param = torch.cat((trj_out_param, out_param),1)
                            trj_out = torch.cat((trj_out, out), 1)
                            y_in = torch.cat((out_param[:,:,:self.spk_dim], out),2)
                    else:
                        for i in range(1,T):
                            out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                            out = self.out_1(self.gru_drop(out).transpose(1,2)).transpose(1,2)
                            out_param = torch.cat((out[:,:,:self.mu_dim],\
                                            F.logsigmoid(out[:,:,self.mu_dim:])),2)
                            out = out[:,:,:self.mu_dim]
                            trj_out_param = torch.cat((trj_out_param, out_param),1)
                            trj_out = torch.cat((trj_out, out), 1)
                            y_in = out
        else:
            if self.arparam:
                if self.spk_dim is not None:
                    for i in range(1,T):
                        out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                        out = self.out_1(out.transpose(1,2)).transpose(1,2)
                        out_param = torch.cat((F.selu(out[:,:,:self.spk_dim]),\
                                        out[:,:,self.spk_dim:self.mu_dim],\
                                            F.logsigmoid(out[:,:,self.mu_dim:])),2)
                        trj_out_param = torch.cat((trj_out_param, out_param),1)
                        y_in = out_param
                else:
                    for i in range(1,T):
                        out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                        out = self.out_1(out.transpose(1,2)).transpose(1,2)
                        out_param = torch.cat((out[:,:,:self.mu_dim],F.logsigmoid(out[:,:,self.mu_dim:])),2)
                        trj_out_param = torch.cat((trj_out_param, out_param),1)
                        y_in = out_param
            else:
                if sampling:
                    if self.spk_dim is not None:
                        for i in range(1,T):
                            out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                            out = self.out_1(out.transpose(1,2)).transpose(1,2)
                            out_param = torch.cat((F.selu(out[:,:,:self.spk_dim]),\
                                            out[:,:,self.spk_dim:self.mu_dim],\
                                                F.logsigmoid(out[:,:,self.mu_dim:])),2)
                            out = sampling_vae_laplace_batch(out_param[:,:,self.spk_dim:],lat_dim=self.out_dim)
                            trj_out_param = torch.cat((trj_out_param, out_param),1)
                            trj_out = torch.cat((trj_out, out), 1)
                            y_in = torch.cat((out_param[:,:,:self.spk_dim], out),2)
                    else:
                        for i in range(1,T):
                            out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                            out = self.out_1(out.transpose(1,2)).transpose(1,2)
                            out_param = torch.cat((out[:,:,:self.mu_dim],\
                                            F.logsigmoid(out[:,:,self.mu_dim:])),2)
                            out = sampling_vae_laplace_batch(out_param, lat_dim=self.out_dim)
                            trj_out_param = torch.cat((trj_out_param, out_param),1)
                            trj_out = torch.cat((trj_out, out), 1)
                            y_in = out
                else:
                    if self.spk_dim is not None:
                        for i in range(1,T):
                            out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                            out = self.out_1(out.transpose(1,2)).transpose(1,2)
                            out_param = torch.cat((F.selu(out[:,:,:self.spk_dim]),\
                                            out[:,:,self.spk_dim:self.mu_dim],\
                                                F.logsigmoid(out[:,:,self.mu_dim:])),2)
                            out = out[:,:,self.spk_dim:self.mu_dim]
                            trj_out_param = torch.cat((trj_out_param, out_param),1)
                            trj_out = torch.cat((trj_out, out), 1)
                            y_in = torch.cat((out_param[:,:,:self.spk_dim], out),2)
                    else:
                        for i in range(1,T):
                            out, h = self.gru(torch.cat((x_conv[:,i:(i+1)],y_in),2), h)
                            out = self.out_1(out.transpose(1,2)).transpose(1,2)
                            out_param = torch.cat((out[:,:,:self.mu_dim],\
                                            F.logsigmoid(out[:,:,self.mu_dim:])),2)
                            out = out[:,:,:self.mu_dim]
                            trj_out_param = torch.cat((trj_out_param, out_param),1)
                            trj_out = torch.cat((trj_out, out), 1)
                            y_in = out

        # MAP/Latent-feat sampling
        if self.spk_dim is not None:
            trj_map = trj_out_param[:,:,self.spk_dim:self.mu_dim]
        else:
            trj_map = trj_out_param[:,:,:self.mu_dim]
        if self.arparam:
            if self.spk_dim is not None:
                trj_out = trj_out_param[:,:,self.spk_dim:]
            else:
                trj_out = trj_out_param
            if sampling:
                trj_out = sampling_vae_laplace_batch(trj_out, lat_dim=self.out_dim)
            else:
                trj_out = trj_out[:,:,:self.out_dim]

        # Denormalization layer
        if self.scale_out_flag:
            # B x T x C -> B x C x T -> B x T x C
            trj_out = self.scale_out(trj_out.transpose(1,2)).transpose(1,2)
            trj_map = self.scale_out(trj_map.transpose(1,2)).transpose(1,2)

        if not batch_flag:
            trj_out = torch.squeeze(trj_out,0) # B x T x C -> B x C x T -> T x C
            trj_out_param = torch.squeeze(trj_out_param,0) # B x T x C -> B x C x T -> T x C
            trj_map = torch.squeeze(trj_map,0) # B x T x C -> B x C x T -> T x C

        return trj_out, trj_out_param, y_in, h, trj_map
