#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
from dateutil.relativedelta import relativedelta
from distutils.util import strtobool
import logging
import itertools
import os
import sys
import time

import numpy as np
import six
import torch
from torch.autograd import Variable
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import find_files
from utils import read_hdf5
from utils import read_txt

from gru_vae import initialize
from gru_vae import GRU_RNN_STOCHASTIC, GRU_RNN
from gru_vae import MCDloss
from gru_vae import sampling_vae_laplace_batch, loss_vae_laplace

from dataset import FeatureDatasetMultMixTrainVAEClsStatCvExcit, FeatureDatasetMultMixEvalVAEClsStatCvExcit
from dataset import padding

from dtw_c import dtw_c as dtw

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)


def train_generator(dataloader, device, count_spk_pair, n_cyc=0, batch_size=0):
    while True:
        c_idx = 0
        # process over all of files
        for idx, batch in enumerate(dataloader):
            flens = batch['flen_src'].data.numpy()
            max_flen = np.max(flens) ## get max frame length
            flens_spc_src = batch['flen_spc_src'].data.numpy()
            max_flen_spc_src = np.max(flens_spc_src) ## get max frame length
            flens_src_trg = batch['flen_src_trg'].data.numpy()
            max_flen_src_trg = np.max(flens_src_trg) ## get max frame length
            flens_spc_src_trg = batch['flen_spc_src_trg'].data.numpy()
            max_flen_spc_src_trg = np.max(flens_spc_src_trg) ## get max frame length
            hs_src = batch['h_src'][:,:max_flen].to(device)
            src_codes = batch['src_code'][:,:max_flen].to(device)
            src_class_codes = batch['src_class_code'][:,:max_flen].to(device)
            spcidcs_src = batch['spcidx_src'][:,:max_flen_spc_src].to(device)
            hs_src_trg = batch['h_src_trg'][:,:max_flen_src_trg].to(device)
            spcidcs_src_trg = batch['spcidx_src_trg'][:,:max_flen_spc_src_trg].to(device)
            featfiles_src = batch['featfile_src']
            featfiles_src_trg = batch['featfile_src_trg']
            featfile_spk = batch['featfile_spk']
            pair_spk_list = batch['pair_spk_list']
            file_src_trg_flag = batch['file_src_trg_flag']
            n_batch_utt = hs_src.size(0)
            for i in range(n_batch_utt):
                for j in range(n_cyc):
                    count_spk_pair[j][featfile_spk[i]][pair_spk_list[j][i]] += 1
            trg_codes_list = [None]*n_cyc
            cvs_src_list = [None]*n_cyc
            trg_class_codes_list = [None]*n_cyc
            for i in range(n_cyc):
                trg_codes_list[i] = batch['src_trg_code_list'][i][:,:max_flen].to(device)
                cvs_src_list[i] = batch['cv_src_list'][i][:,:max_flen].to(device)
                trg_class_codes_list[i] = batch['trg_class_code_list'][i][:,:max_flen].to(device)

            # use mini batch
            if batch_size != 0:
                src_idx_s = 0
                src_idx_e = batch_size-1
                spcidcs_src_s_idx = np.repeat(-1,n_batch_utt)
                spcidcs_src_e_idx = np.repeat(-1,n_batch_utt)
                s_flag = np.repeat(False,n_batch_utt)
                e_flag = np.repeat(True,n_batch_utt)
                flen_acc = np.repeat(batch_size,n_batch_utt)
                for j in range(n_batch_utt):
                    for i in range(spcidcs_src_e_idx[j]+1,flens_spc_src[j]):
                        if not s_flag[j] and spcidcs_src[j,i] >= src_idx_s:
                            if spcidcs_src[j,i] > src_idx_e:
                                spcidcs_src_s_idx[j] = -1
                                break
                            spcidcs_src_s_idx[j] = i
                            s_flag[j] = True
                            e_flag[j] = False
                            if i == flens_spc_src[j]-1:
                                spcidcs_src_e_idx[j] = i
                                s_flag[j] = False
                                e_flag[j] = True
                                break
                        elif not e_flag[j] and (spcidcs_src[j,i] >= src_idx_e or i == flens_spc_src[j]-1):
                            if spcidcs_src[j,i] > src_idx_e:
                                spcidcs_src_e_idx[j] = i-1
                            else:
                                spcidcs_src_e_idx[j] = i
                            s_flag[j] = False
                            e_flag[j] = True
                            break
                select_utt_idx = [i for i in range(n_batch_utt)]
                yield hs_src, src_codes[:,src_idx_s:src_idx_e+1], trg_codes_list, hs_src_trg, cvs_src_list, \
                    src_idx_s, src_idx_e, spcidcs_src_s_idx, spcidcs_src_e_idx, c_idx, idx, spcidcs_src, \
                        spcidcs_src_trg, featfiles_src, featfiles_src_trg, flens, flens_src_trg, \
                            flens_spc_src, flens_spc_src_trg, select_utt_idx, flen_acc, n_batch_utt, \
                                pair_spk_list, src_class_codes, trg_class_codes_list, file_src_trg_flag
                while src_idx_e < max_flen-1:
                    src_idx_s = src_idx_e + 1
                    src_idx_e = src_idx_s+batch_size-1
                    if src_idx_e >= max_flen:
                        src_idx_e = max_flen-1
                    select_utt_idx  = []
                    for j in range(n_batch_utt):
                        if spcidcs_src_e_idx[j] < flens_spc_src[j]-1:
                            if src_idx_e >= flens[j]:
                                flen_acc[j] = flens[j]-src_idx_s
                            for i in range(spcidcs_src_e_idx[j]+1,flens_spc_src[j]):
                                if not s_flag[j] and spcidcs_src[j,i] >= src_idx_s:
                                    if spcidcs_src[j,i] > src_idx_e:
                                        spcidcs_src_s_idx[j] = -1
                                        break
                                    spcidcs_src_s_idx[j] = i
                                    s_flag[j] = True
                                    e_flag[j] = False
                                    if i == flens_spc_src[j]-1:
                                        spcidcs_src_e_idx[j] = i
                                        s_flag[j] = False
                                        e_flag[j] = True
                                        break
                                elif not e_flag[j] and (spcidcs_src[j,i] >= src_idx_e or \
                                                            i == flens_spc_src[j]-1):
                                    if spcidcs_src[j,i] > src_idx_e:
                                        spcidcs_src_e_idx[j] = i-1
                                    else:
                                        spcidcs_src_e_idx[j] = i
                                    s_flag[j] = False
                                    e_flag[j] = True
                                    break
                            select_utt_idx.append(j)
                    yield hs_src, src_codes[:,src_idx_s:src_idx_e+1], trg_codes_list, hs_src_trg, \
                        cvs_src_list, src_idx_s, src_idx_e, spcidcs_src_s_idx, spcidcs_src_e_idx, c_idx, idx, \
                            spcidcs_src, spcidcs_src_trg, featfiles_src, featfiles_src_trg, flens, \
                                flens_src_trg, flens_spc_src, flens_spc_src_trg, select_utt_idx, flen_acc, \
                                    n_batch_utt, pair_spk_list, src_class_codes, \
                                        trg_class_codes_list, file_src_trg_flag

            # use utterance batch
            else:
                yield hs_src, src_codes, trg_codes_list, hs_src_trg, cvs_src_list, c_idx, idx, spcidcs_src, \
                    spcidcs_src_trg, featfiles_src, featfiles_src_trg, flens, flens_src_trg, flens_spc_src, \
                        flens_spc_src_trg, n_batch_utt, pair_spk_list, src_class_codes, \
                            trg_class_codes_list, file_src_trg_flag

            c_idx += 1
            #if c_idx > 0:
            #if c_idx > 1:
            #if c_idx > 2:
            #    break

        if batch_size > 0:
            yield [], [], [], [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], \
                    [], [], [], [], []
        else:
            yield [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], []


def eval_generator(dataloader, device):
    while True:
        c_idx = 0
        # process over all of files
        for idx, batch in enumerate(dataloader):
            flens = batch['flen_src'].data.numpy()
            max_flen = np.max(flens) ## get max frame length
            flens_spc_src = batch['flen_spc_src'].data.numpy()
            max_flen_spc_src = np.max(flens_spc_src) ## get max frame length
            flens_src_trg = batch['flen_src_trg'].data.numpy()
            max_flen_src_trg = np.max(flens_src_trg) ## get max frame length
            flens_spc_src_trg = batch['flen_spc_src_trg'].data.numpy()
            max_flen_spc_src_trg = np.max(flens_spc_src_trg) ## get max frame length
            hs_src = batch['h_src'][:,:max_flen].to(device)
            src_codes = batch['src_code'][:,:max_flen].to(device)
            src_trg_codes = batch['src_trg_code'][:,:max_flen].to(device)
            src_class_codes = batch['src_class_code'][:,:max_flen].to(device)
            src_trg_class_codes = batch['src_trg_class_code'][:,:max_flen].to(device)
            cvs_src = batch['cv_src'][:,:max_flen].to(device)
            spcidcs_src = batch['spcidx_src'][:,:max_flen_spc_src].to(device)
            hs_src_trg = batch['h_src_trg'][:,:max_flen_src_trg].to(device)
            trg_codes = batch['trg_code'][:,:max_flen_src_trg].to(device)
            trg_src_codes = batch['trg_src_code'][:,:max_flen_src_trg].to(device)
            trg_class_codes = batch['trg_class_code'][:,:max_flen_src_trg].to(device)
            trg_src_class_codes = batch['trg_src_class_code'][:,:max_flen_src_trg].to(device)
            cvs_trg = batch['cv_trg'][:,:max_flen_src_trg].to(device)
            spcidcs_src_trg = batch['spcidx_src_trg'][:,:max_flen_spc_src_trg].to(device)
            featfiles_src = batch['featfile_src']
            featfiles_src_trg = batch['featfile_src_trg']
            file_src_trg_flag = batch['file_src_trg_flag']
            n_batch_utt = hs_src.size(0)

            # use utterance batch
            yield hs_src, src_codes, src_trg_codes, trg_codes, trg_src_codes, hs_src_trg, cvs_src, cvs_trg, \
                c_idx, idx, spcidcs_src, spcidcs_src_trg, featfiles_src, featfiles_src_trg, flens, \
                    flens_src_trg, flens_spc_src, flens_spc_src_trg, n_batch_utt, src_class_codes, \
                        src_trg_class_codes, trg_class_codes, trg_src_class_codes, file_src_trg_flag

            c_idx += 1
            #if c_idx > 0:
            #if c_idx > 1:
            #if c_idx > 2:
            #    break

        yield [], [], [], [], [], [], [], [], -1, -1, [], [], [], [], [], [], [], [], [], [], [], [], [], []


def save_checkpoint(checkpoint_dir, model_encoder, model_decoder, optimizer, numpy_random_state, \
                    torch_random_state, count_spk_pair, iterations):
    model_encoder.cpu()
    model_decoder.cpu()
    checkpoint = {
        "model_encoder": model_encoder.state_dict(),
        "model_decoder": model_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "numpy_random_state": numpy_random_state,
        "torch_random_state": torch_random_state,
        "count_spk_pair": count_spk_pair,
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    model_encoder.cuda()
    model_decoder.cuda()
    logging.info("%d-iter checkpoint created." % iterations)


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of training feat files")
    parser.add_argument("--feats_eval_list", required=True,
                        type=str, help="directory or list of evaluation feat files")
    parser.add_argument("--stats_list", required=True,
                        type=str, help="hdf5 file including speaker statistics")
    parser.add_argument("--stats_jnt", required=True,
                        type=str, help="hdf5 file including joint statistics")
    parser.add_argument("--spk_list", required=True,
                        type=str, help="list of speakers")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument('--batch_size_utt', default=24, type=int,
                        help='Batch size training')
    parser.add_argument('--batch_size_utt_eval', default=24, type=int,
                        help='Batch size development')
    parser.add_argument('--pad_len', default=4200, type=int,
                        help='Length for batch padding')
    parser.add_argument('--n_workers', default=2, type=int,
                        help='# of workers for dataset')
    parser.add_argument('--stdim', default=4, type=int,
                        help='stdim for mcep')
    # network structure setting
    parser.add_argument("--in_dim", default=54,
                        type=int, help="number of dimension of input features")
    parser.add_argument("--lat_dim", default=32,
                        type=int, help="number of dimension of latent features")
    parser.add_argument("--out_dim", default=50,
                        type=int, help="number of dimension of output features")
    parser.add_argument("--hidden_layers", default=1,
                        type=int, help="number of GRU hidden layers")
    parser.add_argument("--hidden_units", default=1024,
                        type=int, help="number of GRU hidden units")
    parser.add_argument("--kernel_size", default=3,
                        type=int, help="number of kernel for input conv. [both enc./dec.]")
    parser.add_argument("--dilation_size", default=2,
                        type=int, help="number of dilation layers for input conv. [both enc./dec.]")
    parser.add_argument("--kernel_size_enc", default=None,
                        type=int, help="number of kernel input conv. in enc.")
    parser.add_argument("--kernel_size_dec", default=None,
                        type=int, help="number of kernel input conv. in dec.")
    parser.add_argument("--dilation_size_enc", default=None,
                        type=int, help="number of dilation layers for input conv. in enc.")
    parser.add_argument("--dilation_size_dec", default=None,
                        type=int, help="number of dilation layers for input conv. in dec.")
    parser.add_argument("--n_cyc", default=2,
                        type=int, help="number of cycles")
    parser.add_argument("--do_prob", default=0.5,
                        type=float, help="dropout probability")
    parser.add_argument("--causal_conv", default=False,
                        type=strtobool, help="use causal convolution for input")
    parser.add_argument("--arparam", default=True,
                        type=strtobool, help="post. param./sampled latent feat. in enc. ar-flow")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=80,
                        type=int, help="frame-size mini-batch (if set 0, utterance batch will be used)")
    parser.add_argument("--epoch_count", default=400,
                        type=int, help="number of maximum training epochs")
    # other setting
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--GPU_device", default=None,
                        type=int, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    if args.GPU_device is not None:
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]  = str(args.GPU_device)

    # make experimental directory
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == "cpu":
        raise ValueError('ERROR: Training by CPU is not acceptable.')

    if args.n_cyc < 1:
        half_cyc = True
        args.n_cyc = 1
    else:
        half_cyc = False

    # save args as conf
    if args.kernel_size_enc is None:
        args.kernel_size_enc = args.kernel_size
    if args.dilation_size_enc is None:
        args.dilation_size_enc = args.dilation_size
    if args.kernel_size_dec is None:
        args.kernel_size_dec = args.kernel_size
    if args.dilation_size_dec is None:
        args.dilation_size_dec = args.dilation_size
    torch.save(args, args.expdir + "/model.conf")

    stdim = args.stdim
    stdim_ = stdim+1

    # define statistics src
    mean_jnt = torch.FloatTensor(read_hdf5(args.stats_jnt, "/mean_feat_org_lf0"))
    std_jnt = torch.FloatTensor(read_hdf5(args.stats_jnt, "/scale_feat_org_lf0"))
    mean_jnt_trg = torch.FloatTensor(read_hdf5(args.stats_jnt, "/mean_feat_org_lf0")[stdim:])
    std_jnt_trg = torch.FloatTensor(read_hdf5(args.stats_jnt, "/scale_feat_org_lf0")[stdim:])

    if torch.cuda.is_available():
        mean_jnt = mean_jnt.cuda()
        std_jnt = std_jnt.cuda()
        mean_jnt_trg = mean_jnt_trg.cuda()
        std_jnt_trg = std_jnt_trg.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")

    spk_list = args.spk_list.split('@')
    n_spk = len(spk_list)

    # define network
    model_encoder = GRU_RNN_STOCHASTIC(
        in_dim=args.in_dim,
        out_dim=args.lat_dim,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        kernel_size=args.kernel_size_enc,
        dilation_size=args.dilation_size_enc,
        spk_dim=n_spk,
        do_prob=args.do_prob,
        causal_conv=args.causal_conv,
        arparam=args.arparam,
        scale_out_flag=False)
    logging.info(model_encoder)
    model_decoder = GRU_RNN(
        in_dim=args.lat_dim+n_spk,
        out_dim=args.out_dim,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        kernel_size=args.kernel_size_dec,
        dilation_size=args.dilation_size_dec,
        do_prob=args.do_prob,
        causal_conv=args.causal_conv,
        scale_in_flag=False)
    logging.info(model_decoder)
    criterion_mcd = MCDloss()
    criterion_ce = nn.CrossEntropyLoss()

    # send to gpu
    if torch.cuda.is_available():
        model_encoder.cuda()
        model_decoder.cuda()
        criterion_mcd.cuda()
        criterion_ce.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)
    model_encoder.apply(initialize)
    model_encoder.train()
    model_decoder.apply(initialize)
    model_decoder.train()
    model_encoder.scale_in.weight = torch.nn.Parameter(torch.diag(1.0/std_jnt.data).unsqueeze(2))
    model_encoder.scale_in.bias = torch.nn.Parameter(-(mean_jnt.data/std_jnt.data))
    model_decoder.scale_out.weight = torch.nn.Parameter(torch.diag(std_jnt_trg.data).unsqueeze(2))
    model_decoder.scale_out.bias = torch.nn.Parameter(mean_jnt_trg.data)
    if args.resume is None:
        epoch_idx = 0
    else:
        checkpoint = torch.load(args.resume)
        model_encoder.load_state_dict(checkpoint["model_encoder"])
        model_decoder.load_state_dict(checkpoint["model_decoder"])
        count_spk_pair = checkpoint["count_spk_pair"]
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % (epoch_idx))

    if args.arparam:
        init_pp = np.zeros((args.batch_size_utt,1,args.lat_dim*2+n_spk))
    else:
        init_pp = np.zeros((args.batch_size_utt,1,args.lat_dim+n_spk))
    y_in_pp = torch.FloatTensor(init_pp).cuda()
    y_in_src = y_in_trg = \
        torch.unsqueeze(torch.unsqueeze((0-mean_jnt_trg)/std_jnt_trg,0),0).repeat(args.batch_size_utt,1,1)
    with torch.no_grad():
        if args.arparam:
            init_pp_eval = np.zeros((args.batch_size_utt_eval,1,args.lat_dim*2+n_spk))
        else:
            init_pp_eval = np.zeros((args.batch_size_utt_eval,1,args.lat_dim+n_spk))
        y_in_pp_eval = torch.FloatTensor(init_pp_eval).cuda()
        y_in_src_eval = y_in_trg_eval = torch.unsqueeze(\
                        torch.unsqueeze((0-mean_jnt_trg)/std_jnt_trg,0),0).repeat(args.batch_size_utt_eval,1,1)

    for param in model_encoder.parameters():
        param.requires_grad = True
    for param in model_decoder.parameters():
        param.requires_grad = True
    for param in model_encoder.scale_in.parameters():
        param.requires_grad = False
    for param in model_decoder.scale_out.parameters():
        param.requires_grad = False
    module_list = list(model_encoder.conv.parameters())
    module_list += list(model_encoder.gru.parameters()) + list(model_encoder.out_1.parameters())
    module_list += list(model_decoder.conv.parameters())
    module_list += list(model_decoder.gru.parameters()) + list(model_decoder.out_1.parameters())
    optimizer = torch.optim.Adam(module_list, lr=args.lr)
    if args.resume is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    parameters = filter(lambda p: p.requires_grad, model_encoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (encoder): %.3f million' % parameters)
    parameters = filter(lambda p: p.requires_grad, model_decoder.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters (decoder): %.3f million' % parameters)

    # define generator training
    if os.path.isdir(args.feats):
        feat_list_src = sorted(find_files(args.feats, "*.h5", use_dir_name=False))
    elif os.path.isfile(args.feats):
        feat_list_src = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)
    logging.info("number of training data = %d." % len(feat_list_src))

    feat_eval_src_list = args.feats_eval_list.split('@')
    assert(n_spk == len(feat_eval_src_list))
    stats_list = args.stats_list.split('@')
    assert(n_spk == len(stats_list))

    n_train_data = len(feat_list_src)
    mod_train_batch = n_train_data % args.batch_size_utt
    if mod_train_batch > 0:
        if args.arparam:
            init_pp_mod = np.zeros((mod_train_batch,1,args.lat_dim*2+n_spk))
        else:
            init_pp_mod = np.zeros((mod_train_batch,1,args.lat_dim+n_spk))
        y_in_pp_mod= torch.FloatTensor(init_pp_mod).cuda()
        y_in_src_mod = y_in_trg_mod = torch.unsqueeze(\
                        torch.unsqueeze((0-mean_jnt_trg)/std_jnt_trg,0),0).repeat(mod_train_batch,1,1)

    # define generator evaluation
    feat_list_eval_src_list = [None]*n_spk
    for i in range(n_spk):
        if os.path.isdir(feat_eval_src_list[i]):
            feat_list_eval_src_list[i] = sorted(find_files(feat_eval_src_list[i], "*.h5", use_dir_name=False))
        elif os.path.isfile(feat_eval_src_list[i]):
            feat_list_eval_src_list[i] = read_txt(feat_eval_src_list[i])
        else:
            logging.error("%s should be directory or list." % (feat_eval_src_list[i]))
            sys.exit(1)

    # data
    def zero_pad(x): return padding(x, args.pad_len, value=0.0)  # noqa: E704
    pad_transform = transforms.Compose([zero_pad])
    dataset = FeatureDatasetMultMixTrainVAEClsStatCvExcit(feat_list_src, pad_transform, spk_list, args.n_cyc, \
                    stats_list, stdim=args.stdim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_utt, shuffle=True, num_workers=args.n_workers)
    dataset_eval = FeatureDatasetMultMixEvalVAEClsStatCvExcit(feat_list_eval_src_list, pad_transform, \
                    spk_list, stats_list, stdim=args.stdim)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size_utt_eval, num_workers=args.n_workers)
    count_spk_pair_cv = dataset_eval.count_spk_pair_cv
    n_eval_data = len(dataset_eval.file_list_src)

    logging.info("number of evaluation data = %d." % n_eval_data)

    mod_eval_batch = n_eval_data % args.batch_size_utt_eval
    if mod_eval_batch > 0:
        with torch.no_grad():
            if args.arparam:
                init_pp_eval_mod = np.zeros((mod_eval_batch,1,args.lat_dim*2+n_spk))
            else:
                init_pp_eval_mod = np.zeros((mod_eval_batch,1,args.lat_dim+n_spk))
            y_in_pp_eval_mod = torch.FloatTensor(init_pp_eval_mod).cuda()
            y_in_src_eval_mod = y_in_trg_eval_mod = torch.unsqueeze(\
                                torch.unsqueeze((0-mean_jnt_trg)/std_jnt_trg,0),0).repeat(mod_eval_batch,1,1)

    # generator optimization instance
    if args.resume is None:
        count_spk_pair = [None]*args.n_cyc
        for k in range(args.n_cyc):
            count_spk_pair[k] = {}
            for i in range(n_spk):
                count_spk_pair[k][spk_list[i]] = {}
                for j in range(n_spk):
                    if i != j:
                        count_spk_pair[k][spk_list[i]][spk_list[j]] = 0
    generator_src = train_generator(dataloader, device, count_spk_pair, \
                                    n_cyc=args.n_cyc, batch_size=args.batch_size)

    # generator eval instance
    count_spk_pair_cv = dataset_eval.count_spk_pair_cv
    generator_eval = eval_generator(dataloader_eval, device)

    # natural GV stats
    gv_mean = [None]*n_spk
    for i in range(n_spk):
        gv_mean[i] = read_hdf5(stats_list[i], "/gv_range_mean")[1:]

    # initialize variable storage for stats and/or tensors
    batch_lat_src = [None]*args.n_cyc
    batch_mle_lat_src = [None]*args.n_cyc
    batch_latpost_src = [None]*args.n_cyc
    batch_scpost_src = [None]*args.n_cyc
    y_in_pp_src = [None]*args.n_cyc
    h_in_pp_src = [None]*args.n_cyc
    batch_trj_src_src = [None]*args.n_cyc
    y_in_src_src = [None]*args.n_cyc
    h_in_src_src = [None]*args.n_cyc
    batch_trj_src_trg = [None]*args.n_cyc
    y_in_src_trg = [None]*args.n_cyc
    h_in_src_trg = [None]*args.n_cyc
    batch_lat_src_trg = [None]*args.n_cyc
    batch_mle_lat_src_trg = [None]*args.n_cyc
    batch_latpost_src_trg = [None]*args.n_cyc
    batch_scpost_src_trg = [None]*args.n_cyc
    y_in_pp_src_trg = [None]*args.n_cyc
    h_in_pp_src_trg = [None]*args.n_cyc
    batch_trj_src_trg_src = [None]*args.n_cyc
    y_in_src_trg_src = [None]*args.n_cyc
    h_in_src_trg_src = [None]*args.n_cyc
    batch_lat_trg_ = [None]*args.n_cyc
    batch_mle_lat_trg_ = [None]*args.n_cyc
    batch_latpost_trg_ = [None]*args.n_cyc
    batch_scpost_trg_ = [None]*args.n_cyc
    batch_trj_trg_trg_ = [None]*args.n_cyc
    batch_trj_trg_src_ = [None]*args.n_cyc
    batch_lat_trg_src_ = [None]*args.n_cyc
    batch_mle_lat_trg_src_ = [None]*args.n_cyc
    batch_latpost_trg_src_ = [None]*args.n_cyc
    batch_scpost_trg_src_ = [None]*args.n_cyc
    batch_trj_trg_src_trg_ = [None]*args.n_cyc
    batch_lat_src_ = [None]*args.n_cyc
    batch_mle_lat_src_ = [None]*args.n_cyc
    batch_latpost_src_ = [None]*args.n_cyc
    batch_scpost_src_ = [None]*args.n_cyc
    batch_trj_src_src_ = [None]*args.n_cyc
    batch_trj_src_trg_ = [None]*args.n_cyc
    batch_lat_src_trg_ = [None]*args.n_cyc
    batch_mle_lat_src_trg_ = [None]*args.n_cyc
    batch_latpost_src_trg_ = [None]*args.n_cyc
    batch_scpost_src_trg_ = [None]*args.n_cyc
    batch_trj_src_trg_src_ = [None]*args.n_cyc
    batch_loss_mcd_trg_trg = [None]*args.n_cyc
    batch_loss_mcd_trg_src_trg = [None]*args.n_cyc
    batch_loss_mcd_trg_src = [None]*args.n_cyc
    batch_loss_mcd_src_src = [None]*args.n_cyc
    batch_loss_mcd_src_trg_src = [None]*args.n_cyc
    batch_loss_mcd_src_trg = [None]*args.n_cyc
    batch_loss_lat_src = [None]*args.n_cyc
    batch_loss_scpost_src = [None]*args.n_cyc
    batch_loss_lat_trg = [None]*args.n_cyc
    batch_loss_scpost_trg = [None]*args.n_cyc
    batch_loss_lat_src_cv = [None]*args.n_cyc
    batch_loss_scpost_src_cv = [None]*args.n_cyc
    batch_loss_lat_trg_cv = [None]*args.n_cyc
    batch_loss_scpost_trg_cv = [None]*args.n_cyc
    batch_gv_trg_trg = [None]*args.n_cyc
    batch_mcdpow_trg_trg = [None]*args.n_cyc
    batch_mcd_trg_trg = [None]*args.n_cyc
    batch_gv_trg_src_trg = [None]*args.n_cyc
    batch_mcdpow_trg_src_trg = [None]*args.n_cyc
    batch_mcd_trg_src_trg = [None]*args.n_cyc
    batch_gv_trg_src = [None]*args.n_cyc
    batch_mcdpow_trg_src = [None]*args.n_cyc
    batch_mcd_trg_src = [None]*args.n_cyc
    batch_lat_dist_trgsrc1 = [None]*args.n_cyc
    batch_lat_dist_trgsrc2 = [None]*args.n_cyc
    batch_lat_cdist_trgsrc1 = [None]*args.n_cyc
    batch_lat_cdist_trgsrc2 = [None]*args.n_cyc
    batch_gv_src_src = [None]*args.n_cyc
    batch_mcdpow_src_src = [None]*args.n_cyc
    batch_mcd_src_src = [None]*args.n_cyc
    batch_gv_src_trg_src = [None]*args.n_cyc
    batch_mcdpow_src_trg_src = [None]*args.n_cyc
    batch_mcd_src_trg_src = [None]*args.n_cyc
    batch_gv_src_trg = [None]*args.n_cyc
    batch_mcdpow_src_trg = [None]*args.n_cyc
    batch_mcd_src_trg = [None]*args.n_cyc
    batch_lat_dist_srctrg1 = [None]*args.n_cyc
    batch_lat_dist_srctrg2 = [None]*args.n_cyc
    batch_lat_cdist_srctrg1 = [None]*args.n_cyc
    batch_lat_cdist_srctrg2 = [None]*args.n_cyc
    loss = []
    loss_mcd_src_src = []
    loss_mcd_src_trg_src = []
    loss_mcd_src_trg = []
    loss_scpost_src = []
    loss_scpost_src_cv = []
    loss_lat_src = []
    loss_lat_src_cv = []
    gv_src_src = []
    mcdpow_src_src = []
    mcd_src_src = []
    gv_src_trg_src = []
    mcdpow_src_trg_src = []
    mcd_src_trg_src = []
    gv_src_trg = []
    mcdpow_src_trg = []
    mcd_src_trg = []
    lat_dist_srctrg1 = []
    lat_dist_srctrg2 = []
    for i in range(args.n_cyc):
        loss_mcd_src_src.append([])
        loss_mcd_src_trg_src.append([])
        loss_mcd_src_trg.append([])
        loss_scpost_src.append([])
        loss_scpost_src_cv.append([])
        loss_lat_src.append([])
        loss_lat_src_cv.append([])
        gv_src_src.append([])
        mcdpow_src_src.append([])
        mcd_src_src.append([])
        gv_src_trg_src.append([])
        mcdpow_src_trg_src.append([])
        mcd_src_trg_src.append([])
        gv_src_trg.append([])
        mcdpow_src_trg.append([])
        mcd_src_trg.append([])
        lat_dist_srctrg1.append([])
        lat_dist_srctrg2.append([])
        for j in range(n_spk):
            gv_src_src[i].append([])
            gv_src_trg_src[i].append([])
            gv_src_trg[i].append([])
    total = []
    n_ev_cyc = 1
    #if args.n_cyc > 1:    
    #    n_ev_cyc = 2
    #else:
    #    n_ev_cyc = 1
    eval_loss_mcd_src_src = [None]*n_ev_cyc
    eval_loss_mcd_src_trg_src = [None]*n_ev_cyc
    eval_loss_mcd_src_trg = [None]*n_ev_cyc
    eval_loss_scpost_src = [None]*n_ev_cyc
    eval_loss_scpost_src_cv = [None]*n_ev_cyc
    eval_loss_lat_src = [None]*n_ev_cyc
    eval_loss_lat_src_cv = [None]*n_ev_cyc
    eval_gv_src_src = [None]*n_ev_cyc
    eval_mcdpow_src_src = [None]*n_ev_cyc
    eval_mcd_src_src = [None]*n_ev_cyc
    eval_gv_src_trg_src = [None]*n_ev_cyc
    eval_mcdpow_src_trg_src = [None]*n_ev_cyc
    eval_mcd_src_trg_src = [None]*n_ev_cyc
    eval_gv_src_trg = [None]*n_ev_cyc
    eval_mcdpow_src_trg = [None]*n_ev_cyc
    eval_mcdpowstd_src_trg = [None]*n_ev_cyc
    eval_mcd_src_trg = [None]*n_ev_cyc
    eval_mcdstd_src_trg = [None]*n_ev_cyc
    eval_lat_dist_srctrg1 = [None]*n_ev_cyc
    eval_lat_dist_srctrg2 = [None]*n_ev_cyc
    prev_featfile_src = np.repeat("",args.batch_size_utt)
    iter_idx = 0 
    #iter_idx = 262063
    iter_count = 0 
    min_idx = -1
    min_eval_mcdpow_src_trg = 99999999.99
    min_eval_mcdpowstd_src_trg = 99999999.99
    min_eval_mcd_src_trg = 99999999.99
    min_eval_mcdstd_src_trg = 99999999.99
    min_eval_mcdpow_src_src = 99999999.99
    min_eval_mcd_src_src = 99999999.99
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        if args.batch_size > 0: # frame-size mini-batch
            # store the current batch data before generating new one, in case the next batch is a different
            # set of utterances, on which the current utterance-level accuracy will be calculated
            if iter_count > 0:
                featfile_src_ = featfile_src
                featfile_src_trg_ = featfile_src_trg
                spcidx_src_ = spcidx_src
                prev_flens_src = flens_src
                flens_spc_src_ = flens_spc_src
                batch_src_trg_ = batch_src_trg
                spcidx_src_trg_ = spcidx_src_trg
                flens_spc_src_trg_ = flens_spc_src_trg
                n_batch_utt_ = n_batch_utt
                file_src_trg_flag_ = file_src_trg_flag
            batch_src, batch_src_src_code, batch_src_trg_code, batch_src_trg, batch_cv_src, src_idx_s, \
            src_idx_e, spcidx_src_s_idx, spcidx_src_e_idx, c_idx_src, utt_idx_src, spcidx_src, \
            spcidx_src_trg, featfile_src, featfile_src_trg, flens_src, flens_src_trg, flens_spc_src, \
            flens_spc_src_trg, select_utt_idx, flen_acc, n_batch_utt, pair_spk_list, \
            batch_src_src_class_code, batch_src_trg_class_code_list, file_src_trg_flag = next(generator_src)
            if iter_count > 0 and (src_idx_s == 0 or c_idx_src < 0): # utterance-level acc. of prev. batch
                if pair_flag: # at least parallel one pair target conversion exists, generate target latent
                    with torch.no_grad():
                        if n_batch_utt_ == args.batch_size_utt:
                            _, _, _, _, trj_lat_srctrg = model_encoder(batch_src_trg_, y_in_pp)
                        else:
                            _, _, _, _, trj_lat_srctrg = model_encoder(batch_src_trg_, y_in_pp_mod)
                for i in range(n_batch_utt_): # iterate over utterances
                    if file_src_trg_flag_[i]: # calculate only if target pair parallel data exists
                        # MCD of spectral with 0th power
                        _, _, batch_mcdpow_src_trg[0], _ = dtw.dtw_org_to_trg(np.array(\
                            torch.index_select(trj_src_trg[i],0,spcidx_src_[i,\
                                :flens_spc_src_[i]]).cpu().data.numpy(), dtype=np.float64), \
                            np.array(torch.index_select(batch_src_trg_[i][:,stdim:],0,\
                            spcidx_src_trg_[i,:flens_spc_src_trg_[i]]).cpu().data.numpy(), dtype=np.float64))
                        # MCD of spectral w/o 0th power, i.e., [:,1:]
                        _, _, batch_mcd_src_trg[0], _ = dtw.dtw_org_to_trg(np.array(\
                            torch.index_select(trj_src_trg[i][:,1:],0,\
                            spcidx_src_[i,:flens_spc_src_[i]]).cpu().data.numpy(), dtype=np.float64), \
                            np.array(torch.index_select(batch_src_trg_[i][:,stdim_:],0,\
                            spcidx_src_trg_[i,:flens_spc_src_trg_[i]]).cpu().data.numpy(), dtype=np.float64))
                        # take latent feat. on speech frames only
                        trj_lat_srctrg_ = np.array(torch.index_select(trj_lat_srctrg[i],0,\
                            spcidx_src_trg_[i,:flens_spc_src_trg_[i]]).cpu().data.numpy(), dtype=np.float64)
                        trj_lat_src_ = np.array(torch.index_select(trj_lat_src[i],0,\
                            spcidx_src_[i,:flens_spc_src_[i]]).cpu().data.numpy(), dtype=np.float64)
                        # time-warping of latent source-to-target for RMSE
                        aligned_lat_srctrg1, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src_, trj_lat_srctrg_)
                        batch_lat_dist_srctrg1[0] = np.mean(np.sqrt(np.mean((\
                            aligned_lat_srctrg1-trj_lat_srctrg_)**2, axis=0)))
                        # Cos-sim of latent source-to-target
                        _, _, batch_lat_cdist_srctrg1[0], _ = dtw.dtw_org_to_trg(\
                            trj_lat_srctrg_, trj_lat_src_, mcd=0)
                        # time-warping of latent target-to-source for RMSE
                        aligned_lat_srctrg2, _, _, _ = dtw.dtw_org_to_trg(trj_lat_srctrg_, trj_lat_src_)
                        batch_lat_dist_srctrg2[0] = np.mean(np.sqrt(np.mean((\
                            aligned_lat_srctrg2-trj_lat_src_)**2, axis=0)))
                        # Cos-sim of latent target-to-source
                        _, _, batch_lat_cdist_srctrg2[0], _ = dtw.dtw_org_to_trg(\
                            trj_lat_src_, trj_lat_srctrg_, mcd=0)
                        # RMSE
                        batch_lat_dist_srctrg1[0] = (batch_lat_dist_srctrg1[0]+batch_lat_dist_srctrg2[0])/2
                        # Cos-sim
                        batch_lat_dist_srctrg2[0] = (batch_lat_cdist_srctrg1[0]+batch_lat_cdist_srctrg2[0])/2
                        # record spectral and latent acc. stats
                        mcdpow_src_trg[0].append(batch_mcdpow_src_trg[0])
                        mcd_src_trg[0].append(batch_mcd_src_trg[0])
                        lat_dist_srctrg1[0].append(batch_lat_dist_srctrg1[0])
                        lat_dist_srctrg2[0].append(batch_lat_dist_srctrg2[0])
                        logging.info("batch cv loss %s %s = %.3f dB %.3f dB , %.3f %.3f" % (
                            os.path.join(os.path.basename(os.path.dirname(featfile_src_[i])),\
                            os.path.basename(featfile_src_[i])),os.path.join(os.path.basename(\
                            os.path.dirname(featfile_src_trg_[i])),os.path.basename(featfile_src_trg_[i])),\
                            batch_mcdpow_src_trg[0], batch_mcd_src_trg[0], batch_lat_dist_srctrg1[0],\
                            batch_lat_dist_srctrg2[0]))
        else: # utterance-size mini-batch
            batch_src, batch_src_src_code, batch_src_trg_code, batch_src_trg, batch_cv_src, c_idx_src, \
            utt_idx_src, spcidx_src, spcidx_src_trg, featfile_src, featfile_src_trg, flens_src, \
            flens_src_trg, flens_spc_src, flens_spc_src_trg, n_batch_utt, pair_spk_list, \
            batch_src_src_class_code, batch_src_trg_class_code_list, file_src_trg_flag = next(generator_src)
        if c_idx_src < 0: # all utterances have been processed for current epoch, summarize and eval.
            # save state of random-generator, both numpy and torch, for training resume
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # save current epoch model
            save_checkpoint(args.expdir, model_encoder, model_decoder, optimizer, numpy_random_state, \
                torch_random_state, count_spk_pair, epoch_idx + 1)
            #for i in range(args.n_cyc): # logging speaker-pair count in conversion flow
            #    logging.info(count_spk_pair[i])
            if args.batch_size > 0: # frame-size
                for j in range(n_batch_utt_): # append GV stats. of the last batch
                    spk_src_ = os.path.basename(os.path.dirname(prev_featfile_src[j]))
                    for k in range(n_spk):
                        if spk_src_ == spk_list[k]:
                            #GV stat of reconst.
                            gv_src_src[0][k].append(torch.var(\
                                tmp_src_src[j,:prev_flens_src[j]], 0).cpu().data.numpy())
                            #GV stat of cyclic reconst.
                            gv_src_trg_src[0][k].append(torch.var(\
                                tmp_src_trg_src[j,:prev_flens_src[j]], 0).cpu().data.numpy())
                            break
                    spk_src_trg_ = prev_pair_spk[0][j] # find target pair
                    for k in range(n_spk):
                        if spk_src_trg_ == spk_list[k]:
                            #GV stat of converted
                            gv_src_trg[0][k].append(torch.var(\
                                tmp_src_trg[j,:prev_flens_src[j]], 0).cpu().data.numpy())
                            break
                # generate first batch of next epoch
                batch_src, batch_src_src_code, batch_src_trg_code, batch_src_trg, batch_cv_src, src_idx_s, \
                    src_idx_e, spcidx_src_s_idx, spcidx_src_e_idx, c_idx_src, utt_idx_src, spcidx_src, \
                    spcidx_src_trg, featfile_src, featfile_src_trg, flens_src, flens_src_trg, flens_spc_src, \
                    flens_spc_src_trg, select_utt_idx, flen_acc, n_batch_utt, pair_spk_list, \
                    batch_src_src_class_code, batch_src_trg_class_code_list, file_src_trg_flag \
                        = next(generator_src)
            else: # utterance-size
                # generate first batch of next epoch
                batch_src, batch_src_src_code, batch_src_trg_code, batch_src_trg, batch_cv_src, c_idx_src, \
                    utt_idx_src, spcidx_src, spcidx_src_trg, featfile_src, featfile_src_trg, flens_src, \
                    flens_src_trg, flens_spc_src, flens_spc_src_trg, n_batch_utt, pair_spk_list, \
                    batch_src_src_class_code, batch_src_trg_class_code_list, file_src_trg_flag \
                        = next(generator_src)
            # save current random generate stats before evaluating development set
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # report current epoch
            text_log = "%.3f ;; " % np.mean(loss)
            #for i in range(args.n_cyc):
            for i in range(n_ev_cyc):
                # summarizing GV stats
                tmp_gv_1 = []
                tmp_gv_2 = []
                tmp_gv_3 = []
                for j in range(n_spk):
                    if len(gv_src_src[i][j]) > 0:
                        tmp_gv_1.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_src[i][j], \
                                            axis=0))-np.log(gv_mean[j])))))
                    if len(gv_src_trg[i][j]) > 0:
                        tmp_gv_3.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg[i][j], \
                                            axis=0))-np.log(gv_mean[j])))))
                    if len(gv_src_trg_src[i][j]) > 0:
                        tmp_gv_2.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg_src[i][j], \
                                            axis=0))-np.log(gv_mean[j])))))
                eval_gv_src_src[i] = np.mean(tmp_gv_1)
                eval_gv_src_trg[i] = np.mean(tmp_gv_3)
                eval_gv_src_trg_src[i] = np.mean(tmp_gv_2)
                # logging epoch summary acc. stats
                if len(mcdpow_src_trg[i]) > 0: # at least one target parallel pair exists
                    text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ; "\
                        "%.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , "\
                            "%.6f %.3f dB (+- %.3f) %.6f dB (+- %.3f) , %.6f %.6f ;; " % ( \
                        i+1, np.mean(loss_mcd_src_src[i]), \
                        np.mean(loss_mcd_src_trg_src[i]), np.mean(loss_mcd_src_trg[i]), \
                        np.mean(loss_scpost_src[i]), np.mean(loss_scpost_src_cv[i]), \
                        np.mean(loss_lat_src[i]), np.mean(loss_lat_src_cv[i]), eval_gv_src_src[i], \
                        np.mean(mcdpow_src_src[i]), np.mean(mcd_src_src[i]), eval_gv_src_trg_src[i], \
                        np.mean(mcdpow_src_trg_src[i]), np.mean(mcd_src_trg_src[i]), eval_gv_src_trg[i], \
                        np.mean(mcdpow_src_trg[i]), np.std(mcdpow_src_trg[i]), np.mean(mcd_src_trg[i]), \
                        np.std(mcd_src_trg[i]), np.mean(lat_dist_srctrg1[i]), np.mean(lat_dist_srctrg2[i]))
                else: # no target parallel data exists
                    text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ; "\
                            "%.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , "\
                                "%.6f n/a dB (+- n/a) n/a dB (+- n/a) , n/a n/a ;; " % (\
                        i+1, np.mean(loss_mcd_src_src[i]), \
                        np.mean(loss_mcd_src_trg_src[i]), np.mean(loss_mcd_src_trg[i]), \
                        np.mean(loss_scpost_src[i]), np.mean(loss_scpost_src_cv[i]), \
                        np.mean(loss_lat_src[i]), np.mean(loss_lat_src_cv[i]), eval_gv_src_src[i], \
                        np.mean(mcdpow_src_src[i]), np.mean(mcd_src_src[i]), eval_gv_src_trg_src[i], \
                        np.mean(mcdpow_src_trg_src[i]), np.mean(mcd_src_trg_src[i]), eval_gv_src_trg[i])
            logging.info("(EPOCH:%d) average optimization loss = %s  (%.3f min., %.3f sec / batch)" % (\
                epoch_idx + 1, text_log, np.sum(total) / 60.0, np.mean(total)))
            logging.info("estimated training time until maximum epoch = "\
                "{0.days:02}:{0.hours:02}:{0.minutes:02}:{0.seconds:02}".format(\
                    relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * np.sum(total)))))
            # start evaluating development set
            if n_eval_data > 0: # only if there exists a development set
                model_encoder.eval()
                model_decoder.eval()
                for param in model_encoder.parameters():
                    param.requires_grad = False
                for param in model_decoder.parameters():
                    param.requires_grad = False
                loss = []
                loss_mcd_src_src = []
                loss_mcd_src_trg_src = []
                loss_mcd_src_trg = []
                loss_scpost_src = []
                loss_scpost_src_cv = []
                loss_lat_src = []
                loss_lat_src_cv = []
                gv_src_src = []
                mcdpow_src_src = []
                mcd_src_src = []
                gv_src_trg_src = []
                mcdpow_src_trg_src = []
                mcd_src_trg_src = []
                gv_src_trg = []
                mcdpow_src_trg = []
                mcd_src_trg = []
                lat_dist_srctrg1 = []
                lat_dist_srctrg2 = []
                for i in range(n_ev_cyc):
                    loss_mcd_src_src.append([])
                    loss_mcd_src_trg_src.append([])
                    loss_mcd_src_trg.append([])
                    loss_scpost_src.append([])
                    loss_scpost_src_cv.append([])
                    loss_lat_src.append([])
                    loss_lat_src_cv.append([])
                    gv_src_src.append([])
                    mcdpow_src_src.append([])
                    mcd_src_src.append([])
                    gv_src_trg_src.append([])
                    mcdpow_src_trg_src.append([])
                    mcd_src_trg_src.append([])
                    gv_src_trg.append([])
                    mcdpow_src_trg.append([])
                    mcd_src_trg.append([])
                    lat_dist_srctrg1.append([])
                    lat_dist_srctrg2.append([])
                    for j in range(n_spk):
                        gv_src_src[i].append([])
                        gv_src_trg_src[i].append([])
                        gv_src_trg[i].append([])
                total = []
                iter_count = 0
                logging.info("Evaluation data")
                with torch.no_grad():
                    while True:
                        start = time.time()
                        batch_src_, batch_src_src_code_, batch_src_trg_code_, batch_trg_trg_code_, \
                            batch_trg_src_code_, batch_src_trg_, batch_cv_src_, batch_cv_trg_, c_idx, \
                            utt_idx, spcidx_src_, spcidx_src_trg_, featfile_src_, featfile_src_trg_, \
                            flens_src_, flens_src_trg_, flens_spc_src_, flens_spc_src_trg_, n_batch_utt_, \
                            batch_src_src_class_code_, batch_src_trg_class_code_, batch_trg_trg_class_code_, \
                            batch_trg_src_class_code_, file_src_trg_flag_ = next(generator_eval)
                        batch_trg_ = batch_src_trg_
                        batch_trg_src_ = batch_src_
                        spcidx_trg_ = spcidx_src_trg_
                        spcidx_trg_src_ = spcidx_src_
                        featfile_trg_ = featfile_src_trg_
                        featfile_trg_src_ = featfile_src_
                        flens_trg_ = flens_src_trg_
                        flens_trg_src_ = flens_src_
                        flens_spc_trg_ = flens_spc_src_trg_
                        flens_spc_trg_src_ = flens_spc_src_
                        if c_idx < 0:
                            break
                        pair_flag = False
                        for i in range(n_batch_utt_): # check existence of parallel pair target
                            if file_src_trg_flag_[i]: # target parallel pair data exists
                                logging.info("%s %s %d %d %d %d" % (
                                    os.path.join(os.path.basename(os.path.dirname(featfile_src_[i])), \
                                    os.path.basename(featfile_src_[i])), os.path.join(os.path.basename(\
                                    os.path.dirname(featfile_src_trg_[i])), \
                                    os.path.basename(featfile_src_trg_[i])), flens_src_[i], \
                                    flens_src_trg_[i], flens_spc_src_[i], flens_spc_src_trg_[i]))
                                logging.info("%s %s %d %d %d %d" % (
                                    os.path.join(os.path.basename(os.path.dirname(featfile_trg_[i])), \
                                    os.path.basename(featfile_trg_[i])), os.path.join(os.path.basename(\
                                    os.path.dirname(featfile_trg_src_[i])), \
                                    os.path.basename(featfile_trg_src_[i])), flens_trg_[i], \
                                    flens_trg_src_[i], flens_spc_trg_[i], flens_spc_trg_src_[i]))
                                pair_flag = True
                            else: # no target parallel pair data
                                logging.info("%s %s %d %d" % (
                                    os.path.join(os.path.basename(os.path.dirname(featfile_src_[i])), \
                                    os.path.basename(featfile_src_[i])), os.path.basename(os.path.dirname(\
                                    featfile_src_trg_[i])), flens_src_[i], flens_spc_src_[i]))

                        if n_batch_utt_ == args.batch_size_utt_eval: # batch seq. size is the same as setting
                            y_in_pp_eval_ = y_in_pp_eval
                            y_in_trg_eval_ = y_in_trg_eval
                            y_in_src_eval_ = y_in_src_eval
                        else: # current batch size is reduced due to unrounded values of total_data/batch_size
                            y_in_pp_eval_ = y_in_pp_eval_mod
                            y_in_trg_eval_ = y_in_trg_eval_mod
                            y_in_src_eval_ = y_in_src_eval_mod

                        if pair_flag: # at least parallel one pair target conversion exists
                            _, _, _, _, trj_lat_srctrg = model_encoder(batch_src_trg_, y_in_pp_eval_,\
                                                            sampling=False)
                            _, _, _, _, trj_lat_trgsrc = model_encoder(batch_trg_src_, y_in_pp_eval_,\
                                                            sampling=False)

                        for i in range(n_ev_cyc):
                            batch_mcdpow_src_src[i] = []
                            batch_mcd_src_src[i] = []
                            batch_mcdpow_src_trg_src[i] = []
                            batch_mcd_src_trg_src[i] = []
                            batch_mcdpow_src_trg[i] = []
                            batch_mcd_src_trg[i] = []
                            batch_mcdpow_trg_trg[i] = []
                            batch_mcd_trg_trg[i] = []
                            batch_mcdpow_trg_src_trg[i] = []
                            batch_mcd_trg_src_trg[i] = []
                            batch_mcdpow_trg_src[i] = []
                            batch_mcd_trg_src[i] = []
                            batch_lat_dist_srctrg1[i] = []
                            batch_lat_dist_srctrg2[i] = []
                            batch_lat_dist_trgsrc1[i] = []
                            batch_lat_dist_trgsrc2[i] = []
                            if i > 0: # [2nd, 3rd, ..., Nth] cycle
                                if pair_flag: # at least one target pair parallel data exists
                                    # encoding input features
                                    batch_lat_trg_[i], batch_param, _, _, batch_mle_lat_trg_[i] = \
                                        model_encoder(torch.cat((batch_trg_[:,:,:stdim], \
                                            batch_trj_trg_src_trg_[i-1]),2), y_in_pp_eval_, sampling=False)
                                    batch_scpost_trg_[i] = batch_param[:,:,:n_spk]
                                    batch_latpost_trg_[i] = batch_param[:,:,n_spk:]

                                    # spectral reconst.
                                    batch_trj_trg_trg_[i], _, _ = \
                                        model_decoder(torch.cat((batch_trg_trg_code_, batch_lat_trg_[i]),2), \
                                            y_in_trg_eval_)

                                    # spectral conversion
                                    batch_trj_trg_src_[i], _, _ = \
                                        model_decoder(torch.cat((batch_trg_src_code_, batch_lat_trg_[i]),2), \
                                            y_in_src_eval_)

                                    # encoding converted features
                                    batch_lat_trg_src_[i], batch_param, _, _, batch_mle_lat_trg_src_[i] = \
                                        model_encoder(torch.cat((batch_cv_trg_, batch_trj_trg_src_[i]),2), \
                                            y_in_pp_eval_, sampling=False)
                                    batch_scpost_trg_src_[i] = batch_param[:,:,:n_spk]
                                    batch_latpost_trg_src_[i] = batch_param[:,:,n_spk:]

                                    # cyclic spectral reconst.
                                    batch_trj_trg_src_trg_[i], _, _ = \
                                        model_decoder(torch.cat((batch_trg_trg_code_, \
                                            batch_lat_trg_src_[i]),2), y_in_trg_eval_)

                                # encoding input features
                                batch_lat_src_[i], batch_param, _, _, batch_mle_lat_src_[i] = \
                                    model_encoder(torch.cat((batch_src_[:,:,:stdim], \
                                        batch_trj_src_trg_src_[i-1]),2), y_in_pp_eval_, sampling=False)
                                batch_scpost_src_[i] = batch_param[:,:,:n_spk]
                                batch_latpost_src_[i] = batch_param[:,:,n_spk:]

                                # spectral reconst.
                                batch_trj_src_src_[i], _, _ = \
                                    model_decoder(torch.cat((batch_src_src_code_, batch_lat_src_[i]),2), \
                                        y_in_src_eval_)

                                # spectral conversion
                                batch_trj_src_trg_[i], _, _ = \
                                    model_decoder(torch.cat((batch_src_trg_code_, batch_lat_src_[i]),2), \
                                        y_in_trg_eval_)

                                # encoding converted features
                                batch_lat_src_trg_[i], batch_param, _, _, batch_mle_lat_src_trg_[i] = \
                                    model_encoder(torch.cat((batch_cv_src_, batch_trj_src_trg_[i]),2), \
                                        y_in_pp_eval_, sampling=False)
                                batch_scpost_src_trg_[i] = batch_param[:,:,:n_spk]
                                batch_latpost_src_trg_[i] = batch_param[:,:,n_spk:]

                                # cyclic spectral reconst.
                                batch_trj_src_trg_src_[i], _, _ = \
                                    model_decoder(torch.cat((batch_src_src_code_,\
                                        batch_lat_src_trg_[i]),2), y_in_src_eval_)
                            else: # 1st cycle
                                if pair_flag: # at least one target pair parallel data exists
                                    # encoding input features
                                    batch_lat_trg_[0], batch_param, _, _, batch_mle_lat_trg_[0] = \
                                        model_encoder(batch_trg_, y_in_pp_eval_, sampling=False)
                                    batch_scpost_trg_[0] = batch_param[:,:,:n_spk]
                                    batch_latpost_trg_[0] = batch_param[:,:,n_spk:]

                                    # spectral reconst.
                                    batch_trj_trg_trg_[0], _, _ = \
                                        model_decoder(torch.cat((batch_trg_trg_code_, batch_lat_trg_[0]),2), \
                                            y_in_trg_eval_)

                                    # spectral conversion
                                    batch_trj_trg_src_[0], _, _ = \
                                        model_decoder(torch.cat((batch_trg_src_code_, batch_lat_trg_[0]),2), \
                                            y_in_src_eval_)

                                    # encoding converted features
                                    batch_lat_trg_src_[0], batch_param, _, _, \
                                        batch_mle_lat_trg_src_[0] = \
                                        model_encoder(torch.cat((batch_cv_trg_, batch_trj_trg_src_[0]),2), \
                                            y_in_pp_eval_, sampling=False)
                                    batch_scpost_trg_src_[0] = batch_param[:,:,:n_spk]
                                    batch_latpost_trg_src_[0] = batch_param[:,:,n_spk:]

                                    # cyclic spectral reconst.
                                    batch_trj_trg_src_trg_[0], _, _ = \
                                        model_decoder(torch.cat((batch_trg_trg_code_, \
                                            batch_lat_trg_src_[0]),2), y_in_trg_eval_)

                                # encoding input features
                                batch_lat_src_[0], batch_param, _, _, batch_mle_lat_src_[0] = \
                                    model_encoder(batch_src_, y_in_pp_eval_, sampling=False)
                                batch_scpost_src_[0] = batch_param[:,:,:n_spk]
                                batch_latpost_src_[0] = batch_param[:,:,n_spk:]

                                # spectral reconst.
                                batch_trj_src_src_[0], _, _ = \
                                    model_decoder(torch.cat((batch_src_src_code_, batch_lat_src_[0]),2), \
                                        y_in_src_eval_)

                                # spectral conversion.
                                batch_trj_src_trg_[0], _, _ = \
                                    model_decoder(torch.cat((batch_src_trg_code_, batch_lat_src_[0]),2), \
                                        y_in_trg_eval_)

                                # encoding converted features
                                batch_lat_src_trg_[0], batch_param, _, _, batch_mle_lat_src_trg_[0] = \
                                    model_encoder(torch.cat((batch_cv_src_, batch_trj_src_trg_[0]),2), \
                                        y_in_pp_eval_, sampling=False)
                                batch_scpost_src_trg_[0] = batch_param[:,:,:n_spk]
                                batch_latpost_src_trg_[0] = batch_param[:,:,n_spk:]

                                # cyclic spectral reconst.
                                batch_trj_src_trg_src_[0], _, _ = \
                                    model_decoder(torch.cat((batch_src_src_code_, \
                                        batch_lat_src_trg_[0]),2), y_in_src_eval_)

                                # calc acc.
                                for j in range(n_batch_utt_):
                                    spk_src_ = os.path.basename(os.path.dirname(featfile_src_[j]))
                                    for k in range(n_spk):
                                        if spk_src_ == spk_list[k]:
                                            #GV stat of reconst.
                                            gv_src_src[i][k].append(np.var(\
                                                batch_trj_src_src_[i][j,:flens_src_[j],\
                                                    1:].cpu().data.numpy(), axis=0))
                                            #GV stat of cyclic reconst.
                                            gv_src_trg_src[i][k].append(np.var(\
                                                batch_trj_src_trg_src_[i][j,:flens_src_[j],\
                                                    1:].cpu().data.numpy(), axis=0))
                                            break
                                    spk_trg_ = os.path.basename(os.path.dirname(featfile_src_trg_[j]))
                                    for k in range(n_spk):
                                        if spk_trg_ == spk_list[k]:
                                            #GV stat of converted
                                            gv_src_trg[i][k].append(np.var(\
                                                batch_trj_src_trg_[i][j,:flens_src_[j],\
                                                    1:].cpu().data.numpy(), axis=0))

                                    # target pair parallel data exists
                                    if file_src_trg_flag_[j]:
                                        #latent of converted
                                        trj_lat_srctrg_ = np.array(torch.index_select(\
                                            trj_lat_srctrg[j],0,spcidx_src_trg_[j,\
                                                :flens_spc_src_trg_[j]]).cpu().data.numpy(), dtype=np.float64)
                                        #latent of source
                                        trj_lat_src_ = np.array(torch.index_select(\
                                            batch_mle_lat_src_[0][j],0,spcidx_src_[j,\
                                                :flens_spc_src_[j]]).cpu().data.numpy(), dtype=np.float64)
                                        # time-warping of source-to-target for RMSE
                                        aligned_lat_srctrg1, _, _, _ = dtw.dtw_org_to_trg(trj_lat_src_, \
                                            trj_lat_srctrg_)
                                        tmp_batch_lat_dist_srctrg1 = np.mean(np.sqrt(np.mean((\
                                            aligned_lat_srctrg1-trj_lat_srctrg_)**2, axis=0)))
                                        # Cos-Sim source-to-target
                                        _, _, tmp_batch_lat_cdist_srctrg1, _ = dtw.dtw_org_to_trg(\
                                            trj_lat_srctrg_, trj_lat_src_, mcd=0)
                                        # time-warping of target-to-source for RMSE
                                        aligned_lat_srctrg2, _, _, _ = dtw.dtw_org_to_trg(trj_lat_srctrg_, \
                                            trj_lat_src_)
                                        tmp_batch_lat_dist_srctrg2 = np.mean(np.sqrt(np.mean((\
                                            aligned_lat_srctrg2-trj_lat_src_)**2, axis=0)))
                                        # Cos-Sim target-to-source
                                        _, _, tmp_batch_lat_cdist_srctrg2, _ = dtw.dtw_org_to_trg(\
                                            trj_lat_src_, trj_lat_srctrg_, mcd=0)

                                        # RMSE
                                        tmp_batch_lat_dist_srctrg1 = (tmp_batch_lat_dist_srctrg1 + \
                                            tmp_batch_lat_dist_srctrg2)/2
                                        # Cos-Sim
                                        tmp_batch_lat_dist_srctrg2 = (tmp_batch_lat_cdist_srctrg1 + \
                                            tmp_batch_lat_cdist_srctrg2)/2

                                        # record latent source acc. stats
                                        lat_dist_srctrg1[0].append(tmp_batch_lat_dist_srctrg1)
                                        lat_dist_srctrg2[0].append(tmp_batch_lat_dist_srctrg2)
                                        batch_lat_dist_srctrg1[0].append(tmp_batch_lat_dist_srctrg1)
                                        batch_lat_dist_srctrg2[0].append(tmp_batch_lat_dist_srctrg2)

                                        #latent of converted
                                        trj_lat_trgsrc_ = np.array(torch.index_select(\
                                            trj_lat_trgsrc[j],0,spcidx_trg_src_[j,\
                                                :flens_spc_trg_src_[j]]).cpu().data.numpy(), dtype=np.float64)
                                        #latent of target
                                        trj_lat_trg_ = np.array(torch.index_select(\
                                            batch_mle_lat_trg_[0][j],0,spcidx_trg_[j,\
                                                :flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64)
                                        # time-warping of target-to-source for RMSE
                                        aligned_lat_trgsrc1, _, _, _ = dtw.dtw_org_to_trg(trj_lat_trg_, \
                                            trj_lat_trgsrc_)
                                        tmp_batch_lat_dist_trgsrc1 = np.mean(np.sqrt(np.mean((\
                                            aligned_lat_trgsrc1-trj_lat_trgsrc_)**2, axis=0)))
                                        # Cos-Sim of target-to-source
                                        _, _, tmp_batch_lat_cdist_trgsrc1, _ = dtw.dtw_org_to_trg(\
                                            trj_lat_trgsrc_, trj_lat_trg_, mcd=0)
                                        # time-warping of source-to-target for RMSE
                                        aligned_lat_trgsrc2, _, _, _ = dtw.dtw_org_to_trg(trj_lat_trgsrc_, \
                                            trj_lat_trg_)
                                        tmp_batch_lat_dist_trgsrc2 = np.mean(np.sqrt(np.mean((\
                                            aligned_lat_trgsrc2-trj_lat_trg_)**2, axis=0)))
                                        # Cos-Sim of source-to-target
                                        _, _, tmp_batch_lat_cdist_trgsrc2, _ = dtw.dtw_org_to_trg(\
                                            trj_lat_trg_, trj_lat_trgsrc_, mcd=0)

                                        # RMSE
                                        tmp_batch_lat_dist_trgsrc1 = (tmp_batch_lat_dist_trgsrc1 + \
                                            tmp_batch_lat_dist_trgsrc2)/2
                                        # Cos-sim
                                        tmp_batch_lat_dist_trgsrc2 = (tmp_batch_lat_cdist_trgsrc1 + \
                                            tmp_batch_lat_cdist_trgsrc2)/2

                                        # record latent target acc. stats
                                        lat_dist_srctrg1[0].append(tmp_batch_lat_dist_trgsrc1)
                                        lat_dist_srctrg2[0].append(tmp_batch_lat_dist_trgsrc2)
                                        batch_lat_dist_trgsrc1[0].append(tmp_batch_lat_dist_trgsrc1)
                                        batch_lat_dist_trgsrc2[0].append(tmp_batch_lat_dist_trgsrc2)

                                        # time-warping function with speech frames to calc true MCD values
                                        batch_trg_spc_ = np.array(torch.index_select(\
                                            batch_trg_[j,:,stdim:],0,spcidx_trg_[j,\
                                                :flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64)
                                        batch_trg_spc__ = np.array(torch.index_select(\
                                            batch_trg_[j,:,stdim_:],0,spcidx_trg_[j,\
                                                :flens_spc_trg_[j]]).cpu().data.numpy(), dtype=np.float64)

                                        # MCD of reconst.
                                        tmp_batch_mcdpow_trg_trg, _ = dtw.calc_mcd(batch_trg_spc_, \
                                            np.array(torch.index_select(batch_trj_trg_trg_[i][j],0,\
                                            spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), \
                                            dtype=np.float64))
                                        tmp_batch_mcd_trg_trg, _ = dtw.calc_mcd(batch_trg_spc__, \
                                            np.array(torch.index_select(batch_trj_trg_trg_[i][j,:,1:],0,\
                                            spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), \
                                            dtype=np.float64))

                                        # MCD of cyclic reconst.
                                        tmp_batch_mcdpow_trg_src_trg, _ = dtw.calc_mcd(batch_trg_spc_, \
                                            np.array(torch.index_select(batch_trj_trg_src_trg_[i][j],0,\
                                            spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), \
                                            dtype=np.float64))
                                        tmp_batch_mcd_trg_src_trg, _ = dtw.calc_mcd(batch_trg_spc__, \
                                            np.array(torch.index_select(batch_trj_trg_src_trg_[i][j,:,1:],0,\
                                            spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), \
                                            dtype=np.float64))

                                        # MCD of conversion target-to-source
                                        _, _, tmp_batch_mcdpow_trg_src, _ = dtw.dtw_org_to_trg(\
                                            np.array(torch.index_select(batch_trj_trg_src_[i][j],0,\
                                                spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), \
                                                dtype=np.float64), \
                                            np.array(torch.index_select(batch_trg_src_[j,:,stdim:],0,\
                                                spcidx_trg_src_[j,:flens_spc_trg_src_[j]]).cpu().data.numpy(),\
                                                dtype=np.float64))
                                        _, _, tmp_batch_mcd_trg_src, _ = dtw.dtw_org_to_trg(\
                                            np.array(torch.index_select(batch_trj_trg_src_[i][j,:,1:],0,\
                                                spcidx_trg_[j,:flens_spc_trg_[j]]).cpu().data.numpy(), \
                                                dtype=np.float64), \
                                            np.array(torch.index_select(batch_trg_src_[j,:,stdim_:],0,\
                                                spcidx_trg_src_[j,:flens_spc_trg_src_[j]]).cpu().data.numpy(),\
                                                dtype=np.float64))

                                    # time-warping function with speech frames to calc true MCD values
                                    batch_src_spc_ = np.array(torch.index_select(batch_src_[j,:,stdim:],0,\
                                        spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), \
                                        dtype=np.float64)
                                    batch_src_spc__ = np.array(torch.index_select(batch_src_[j,:,stdim_:],0,\
                                        spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), \
                                        dtype=np.float64)

                                    # MCD of reconst.
                                    tmp_batch_mcdpow_src_src, _ = dtw.calc_mcd(batch_src_spc_, \
                                        np.array(torch.index_select(batch_trj_src_src_[i][j],0,\
                                        spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), \
                                        dtype=np.float64))
                                    tmp_batch_mcd_src_src, _ = dtw.calc_mcd(batch_src_spc__, \
                                        np.array(torch.index_select(batch_trj_src_src_[i][j,:,1:],0,\
                                        spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), \
                                        dtype=np.float64))

                                    # MCD of cyclic reconst.
                                    tmp_batch_mcdpow_src_trg_src, _ = dtw.calc_mcd(batch_src_spc_, \
                                        np.array(torch.index_select(batch_trj_src_trg_src_[i][j],0,\
                                            spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), \
                                            dtype=np.float64))
                                    tmp_batch_mcd_src_trg_src, _ = dtw.calc_mcd(batch_src_spc__, \
                                        np.array(torch.index_select(batch_trj_src_trg_src_[i][j,:,1:],0,\
                                            spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), \
                                            dtype=np.float64))

                                    # target pair parallel data exists
                                    if file_src_trg_flag_[j]:
                                        # MCD of conversion
                                        _, _, tmp_batch_mcdpow_src_trg, _ = dtw.dtw_org_to_trg(\
                                            np.array(torch.index_select(batch_trj_src_trg_[i][j],0,\
                                                spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), \
                                                dtype=np.float64), \
                                            np.array(torch.index_select(batch_src_trg_[j,:,stdim:],0,\
                                                spcidx_src_trg_[j,:flens_spc_src_trg_[j]]).cpu().data.numpy(),\
                                                dtype=np.float64))
                                        _, _, tmp_batch_mcd_src_trg, _ = dtw.dtw_org_to_trg(\
                                            np.array(torch.index_select(batch_trj_src_trg_[i][j,:,1:],0,\
                                                spcidx_src_[j,:flens_spc_src_[j]]).cpu().data.numpy(), \
                                                dtype=np.float64), \
                                            np.array(torch.index_select(batch_src_trg_[j,:,stdim_:],0,\
                                                spcidx_src_trg_[j,:flens_spc_src_trg_[j]]).cpu().data.numpy(),\
                                                dtype=np.float64))

                                    # record acc.
                                    if file_src_trg_flag_[j]:  # target pair parallel data exists
                                        batch_mcdpow_trg_trg[0].append(tmp_batch_mcdpow_trg_trg)
                                        batch_mcd_trg_trg[0].append(tmp_batch_mcd_trg_trg)
                                        batch_mcdpow_trg_src_trg[0].append(tmp_batch_mcdpow_trg_src_trg)
                                        batch_mcd_trg_src_trg[0].append(tmp_batch_mcd_trg_src_trg)
                                        batch_mcdpow_trg_src[0].append(tmp_batch_mcdpow_trg_src)
                                        batch_mcd_trg_src[0].append(tmp_batch_mcd_trg_src)
                                        batch_mcdpow_src_trg[0].append(tmp_batch_mcdpow_src_trg)
                                        batch_mcd_src_trg[0].append(tmp_batch_mcd_src_trg)
                                        mcdpow_src_src[i].append(tmp_batch_mcdpow_trg_trg)
                                        mcd_src_src[i].append(tmp_batch_mcd_trg_trg)
                                        mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_trg_src_trg)
                                        mcd_src_trg_src[i].append(tmp_batch_mcd_trg_src_trg)
                                        mcdpow_src_trg[i].append(tmp_batch_mcdpow_trg_src)
                                        mcd_src_trg[i].append(tmp_batch_mcd_trg_src)
                                        mcdpow_src_trg[i].append(tmp_batch_mcdpow_src_trg)
                                        mcd_src_trg[i].append(tmp_batch_mcd_src_trg)

                                    # record acc. stats
                                    batch_mcdpow_src_src[0].append(tmp_batch_mcdpow_src_src)
                                    batch_mcd_src_src[0].append(tmp_batch_mcd_src_src)
                                    batch_mcdpow_src_trg_src[0].append(tmp_batch_mcdpow_src_trg_src)
                                    batch_mcd_src_trg_src[0].append(tmp_batch_mcd_src_trg_src)
                                    mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                                    mcd_src_src[i].append(tmp_batch_mcd_src_src)
                                    mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                                    mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)

                                    # logging acc. stats
                                    if file_src_trg_flag_[j]: # at least one pair utterance exists
                                        logging.info("batch trgsrc loss %s %s = %.3f dB %.3f dB , "\
                                            "%.3f dB %.3f dB , %.3f dB %.3f dB , %.3f %.3f" % (\
                                            os.path.join(os.path.basename(os.path.dirname(featfile_trg_[j])),\
                                            os.path.basename(featfile_trg_[j])),os.path.join(os.path.basename(\
                                            os.path.dirname(featfile_trg_src_[j])),\
                                            os.path.basename(featfile_trg_src_[j])),tmp_batch_mcdpow_trg_trg, \
                                            tmp_batch_mcd_trg_trg, tmp_batch_mcdpow_trg_src_trg, \
                                            tmp_batch_mcd_trg_src_trg, tmp_batch_mcdpow_trg_src, \
                                            tmp_batch_mcd_trg_src, tmp_batch_lat_dist_trgsrc1, \
                                            tmp_batch_lat_dist_trgsrc2))
                                        logging.info("batch srctrg loss %s %s = %.3f dB %.3f dB , "\
                                            "%.3f dB %.3f dB , %.3f dB %.3f dB , %.3f %.3f" % (
                                            os.path.join(os.path.basename(os.path.dirname(featfile_src_[j])),\
                                            os.path.basename(featfile_src_[j])),os.path.join(os.path.basename(\
                                            os.path.dirname(featfile_src_trg_[j])),\
                                            os.path.basename(featfile_src_trg_[j])),tmp_batch_mcdpow_src_src, \
                                            tmp_batch_mcd_src_src, tmp_batch_mcdpow_src_trg_src, \
                                            tmp_batch_mcd_src_trg_src, tmp_batch_mcdpow_src_trg, \
                                            tmp_batch_mcd_src_trg, tmp_batch_lat_dist_srctrg1, \
                                            tmp_batch_lat_dist_srctrg2))
                                    else:
                                        logging.info("batch src loss %s %s = %.3f dB %.3f dB , "\
                                            "%.3f dB %.3f dB" % (os.path.join(os.path.basename(\
                                                os.path.dirname(featfile_src_[j])),os.path.basename(\
                                                featfile_src_[j])), spk_trg_, tmp_batch_mcdpow_src_src, \
                                                tmp_batch_mcd_src_src, tmp_batch_mcdpow_src_trg_src, \
                                                tmp_batch_mcd_src_trg_src))

                                # record mean-loss from all utterances in this batch
                                batch_mcdpow_src_src[i] = np.mean(batch_mcdpow_src_src[i])
                                batch_mcd_src_src[i] = np.mean(batch_mcd_src_src[i])
                                batch_mcdpow_src_trg_src[i] = np.mean(batch_mcdpow_src_trg_src[i])
                                batch_mcd_src_trg_src[i] = np.mean(batch_mcd_src_trg_src[i])
                                if len(batch_mcdpow_src_trg[i]) > 0: # at least one target pair exists
                                    batch_mcdpow_src_trg[i] = np.mean(batch_mcdpow_src_trg[i])
                                    batch_mcd_src_trg[i] = np.mean(batch_mcd_src_trg[i])
                                    batch_mcdpow_trg_trg[i] = np.mean(batch_mcdpow_trg_trg[i])
                                    batch_mcd_trg_trg[i] = np.mean(batch_mcd_trg_trg[i])
                                    batch_mcdpow_trg_src_trg[i] = np.mean(batch_mcdpow_trg_src_trg[i])
                                    batch_mcd_trg_src_trg[i] = np.mean(batch_mcd_trg_src_trg[i])
                                    batch_mcdpow_trg_src[i] = np.mean(batch_mcdpow_trg_src[i])
                                    batch_mcd_trg_src[i] = np.mean(batch_mcd_trg_src[i])
                                    batch_lat_dist_srctrg1[i] = np.mean(batch_lat_dist_srctrg1[i])
                                    batch_lat_dist_srctrg2[i] = np.mean(batch_lat_dist_srctrg2[i])
                                    batch_lat_dist_trgsrc1[i] = np.mean(batch_lat_dist_trgsrc1[i])
                                    batch_lat_dist_trgsrc2[i] = np.mean(batch_lat_dist_trgsrc2[i])

                            # loss acc.
                            flag_cat = False
                            for j in range(n_batch_utt_): # iterate over all utterances in this batch
                                if file_src_trg_flag_[j]: # at least one parallel target pair data exists
                                    # valid spectral length
                                    batch_trg_optim = batch_trg_[j,:flens_trg_[j],stdim:]
                                    # mel-cepstral distortion (MCD)-based L1-loss of spectral features
                                    _, tmp_batch_loss_mcd_trg_trg, _ = criterion_mcd(\
                                        batch_trj_trg_trg_[i][j,:flens_trg_[j]], batch_trg_optim)
                                    _, tmp_batch_loss_mcd_trg_src, _ = criterion_mcd(\
                                        batch_trj_trg_src_[i][j,:flens_trg_[j]], batch_trg_optim)
                                    _, tmp_batch_loss_mcd_trg_src_trg, _ = criterion_mcd(\
                                        batch_trj_trg_src_trg_[i][j,:flens_trg_[j]], batch_trg_optim)
                                    # cross-entropy (CE) of speaker-posterior
                                    tmp_batch_loss_scpost_trg = criterion_ce(\
                                        batch_scpost_trg_[i][j,:flens_trg_[j]], \
                                            batch_trg_trg_class_code_[j,:flens_trg_[j]])
                                    tmp_batch_loss_scpost_trg_cv = criterion_ce(\
                                        batch_scpost_trg_src_[i][j,:flens_trg_[j]], \
                                            batch_trg_src_class_code_[j,:flens_trg_[j]])
                                    # KL-divergence of latent-posterior to the standard Laplacian prior
                                    tmp_batch_loss_lat_trg = loss_vae_laplace(\
                                        batch_latpost_trg_[i][j,:flens_trg_[j]], lat_dim=args.lat_dim)
                                    tmp_batch_loss_lat_trg_cv = loss_vae_laplace(\
                                        batch_latpost_trg_src_[i][j,:flens_trg_[j]], lat_dim=args.lat_dim)

                                # valid spectral length
                                batch_src_optim = batch_src_[j,:flens_src_[j],stdim:]
                                # mel-cepstral distortion (MCD)-based L1-loss of spectral features
                                _, tmp_batch_loss_mcd_src_src, _ = criterion_mcd(\
                                    batch_trj_src_src_[i][j,:flens_src_[j]], batch_src_optim)
                                _, tmp_batch_loss_mcd_src_trg, _ = criterion_mcd(\
                                    batch_trj_src_trg_[i][j,:flens_src_[j]], batch_src_optim)
                                _, tmp_batch_loss_mcd_src_trg_src, _ = criterion_mcd(\
                                    batch_trj_src_trg_src_[i][j,:flens_src_[j]], batch_src_optim)
                                # cross-entropy (CE) of speaker-posterior
                                tmp_batch_loss_scpost_src = criterion_ce(\
                                    batch_scpost_src_[i][j,:flens_src_[j]], \
                                        batch_src_src_class_code_[j,:flens_src_[j]])
                                tmp_batch_loss_scpost_src_cv = criterion_ce(\
                                    batch_scpost_src_trg_[i][j,:flens_src_[j]], \
                                        batch_src_trg_class_code_[j,:flens_src_[j]])
                                # KL-divergence of latent-posterior to the standard Laplacian prior
                                tmp_batch_loss_lat_src = loss_vae_laplace(\
                                    batch_latpost_src_[i][j,:flens_src_[j]], lat_dim=args.lat_dim)
                                tmp_batch_loss_lat_src_cv = loss_vae_laplace(\
                                    batch_latpost_src_trg_[i][j,:flens_src_[j]], lat_dim=args.lat_dim)

                                if j > 0:
                                    if file_src_trg_flag_[j]: # at least one parallel utt. exists
                                        if flag_cat: 
                                            batch_loss_mcd_trg_trg[i] = torch.cat((batch_loss_mcd_trg_trg[i], \
                                                tmp_batch_loss_mcd_trg_trg.unsqueeze(0)))
                                            batch_loss_mcd_trg_src[i] = torch.cat((batch_loss_mcd_trg_src[i], \
                                                tmp_batch_loss_mcd_trg_src.unsqueeze(0)))
                                            batch_loss_mcd_trg_src_trg[i] = torch.cat((\
                                                batch_loss_mcd_trg_src_trg[i], \
                                                    tmp_batch_loss_mcd_trg_src_trg.unsqueeze(0)))
                                            batch_loss_lat_trg[i] = torch.cat((batch_loss_lat_trg[i], \
                                                tmp_batch_loss_lat_trg.unsqueeze(0)))
                                            batch_loss_lat_trg_cv[i] = torch.cat((batch_loss_lat_trg_cv[i], \
                                                tmp_batch_loss_lat_trg.unsqueeze(0),\
                                                tmp_batch_loss_lat_trg_cv.unsqueeze(0)))
                                            batch_loss_scpost_trg[i] = torch.cat((batch_loss_scpost_trg[i], \
                                                tmp_batch_loss_scpost_trg.unsqueeze(0)))
                                            batch_loss_scpost_trg_cv[i] = torch.cat((\
                                                batch_loss_scpost_trg_cv[i], \
                                                    tmp_batch_loss_scpost_trg.unsqueeze(0),\
                                                    tmp_batch_loss_scpost_trg_cv.unsqueeze(0)))
                                        else:
                                            batch_loss_mcd_trg_trg[i] = \
                                                tmp_batch_loss_mcd_trg_trg.unsqueeze(0)
                                            batch_loss_mcd_trg_src[i] = \
                                                tmp_batch_loss_mcd_trg_src.unsqueeze(0)
                                            batch_loss_mcd_trg_src_trg[i] = \
                                                tmp_batch_loss_mcd_trg_src_trg.unsqueeze(0)
                                            batch_loss_lat_trg[i] = tmp_batch_loss_lat_trg.unsqueeze(0)
                                            batch_loss_lat_trg_cv[i] = torch.cat((tmp_batch_loss_lat_trg.unsqueeze(0),\
                                                                        tmp_batch_loss_lat_trg_cv.unsqueeze(0)))
                                            batch_loss_scpost_trg[i] = tmp_batch_loss_scpost_trg.unsqueeze(0)
                                            batch_loss_scpost_trg_cv[i] = \
                                                torch.cat((tmp_batch_loss_scpost_trg.unsqueeze(0),\
                                                tmp_batch_loss_scpost_trg_cv.unsqueeze(0)))
                                            flag_cat = True
                                    batch_loss_mcd_src_src[i] = torch.cat((batch_loss_mcd_src_src[i], \
                                        tmp_batch_loss_mcd_src_src.unsqueeze(0)))
                                    batch_loss_mcd_src_trg[i] = torch.cat((batch_loss_mcd_src_trg[i], \
                                        tmp_batch_loss_mcd_src_trg.unsqueeze(0)))
                                    batch_loss_mcd_src_trg_src[i] = torch.cat((batch_loss_mcd_src_trg_src[i], \
                                        tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)))
                                    batch_loss_lat_src[i] = torch.cat((batch_loss_lat_src[i], \
                                        tmp_batch_loss_lat_src.unsqueeze(0)))
                                    batch_loss_lat_src_cv[i] = torch.cat((batch_loss_lat_src_cv[i], \
                                        tmp_batch_loss_lat_src.unsqueeze(0),\
                                        tmp_batch_loss_lat_src_cv.unsqueeze(0)))
                                    batch_loss_scpost_src[i] = torch.cat((batch_loss_scpost_src[i], \
                                        tmp_batch_loss_scpost_src.unsqueeze(0)))
                                    batch_loss_scpost_src_cv[i] = torch.cat((batch_loss_scpost_src_cv[i], \
                                        tmp_batch_loss_scpost_src.unsqueeze(0),\
                                        tmp_batch_loss_scpost_src_cv.unsqueeze(0)))
                                else:
                                    if file_src_trg_flag_[j]: # at least one parallel utt. exists
                                        batch_loss_mcd_trg_trg[i] = tmp_batch_loss_mcd_trg_trg.unsqueeze(0)
                                        batch_loss_mcd_trg_src[i] = tmp_batch_loss_mcd_trg_src.unsqueeze(0)
                                        batch_loss_mcd_trg_src_trg[i] = \
                                            tmp_batch_loss_mcd_trg_src_trg.unsqueeze(0)
                                        batch_loss_lat_trg[i] = tmp_batch_loss_lat_trg.unsqueeze(0)
                                        batch_loss_lat_trg_cv[i] = torch.cat((tmp_batch_loss_lat_trg.unsqueeze(0),\
                                                                    tmp_batch_loss_lat_trg_cv.unsqueeze(0)))
                                        batch_loss_scpost_trg[i] = tmp_batch_loss_scpost_trg.unsqueeze(0)
                                        batch_loss_scpost_trg_cv[i] = \
                                            torch.cat((tmp_batch_loss_scpost_trg.unsqueeze(0),\
                                            tmp_batch_loss_scpost_trg_cv.unsqueeze(0)))
                                        flag_cat = True
                                    batch_loss_mcd_src_src[i] = tmp_batch_loss_mcd_src_src.unsqueeze(0)
                                    batch_loss_mcd_src_trg[i] = tmp_batch_loss_mcd_src_trg.unsqueeze(0)
                                    batch_loss_mcd_src_trg_src[i] = \
                                        tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)
                                    batch_loss_lat_src[i] = tmp_batch_loss_lat_src.unsqueeze(0)
                                    batch_loss_lat_src_cv[i] = torch.cat((tmp_batch_loss_lat_src.unsqueeze(0),\
                                                                    tmp_batch_loss_lat_src_cv.unsqueeze(0)))
                                    batch_loss_scpost_src[i] = tmp_batch_loss_scpost_src.unsqueeze(0)
                                    batch_loss_scpost_src_cv[i] = torch.cat((tmp_batch_loss_scpost_src.unsqueeze(0),\
                                                                    tmp_batch_loss_scpost_src_cv.unsqueeze(0)))

                            # record loss stats.
                            if pair_flag: # at least one target pair parallel utt. exists
                                batch_loss_mcd_trg_trg[i] = torch.mean(batch_loss_mcd_trg_trg[i])
                                batch_loss_mcd_trg_src_trg[i] = torch.mean(batch_loss_mcd_trg_src_trg[i])
                                batch_loss_mcd_trg_src[i] = torch.mean(batch_loss_mcd_trg_src[i])
                                batch_loss_lat_trg[i] = torch.mean(batch_loss_lat_trg[i])
                                batch_loss_lat_trg_cv[i] = torch.mean(batch_loss_lat_trg_cv[i])
                                batch_loss_scpost_trg[i] = torch.mean(batch_loss_scpost_trg[i])
                                batch_loss_scpost_trg_cv[i] = torch.mean(batch_loss_scpost_trg_cv[i])
                                loss_mcd_src_src[i].append(batch_loss_mcd_trg_trg[i].item())
                                loss_mcd_src_trg[i].append(batch_loss_mcd_trg_src[i].item())
                                loss_mcd_src_trg_src[i].append(batch_loss_mcd_trg_src_trg[i].item())
                                loss_scpost_src_cv[i].append(batch_loss_scpost_trg_cv[i].item())
                                loss_scpost_src[i].append(batch_loss_scpost_trg[i].item())
                                loss_lat_src[i].append(batch_loss_lat_trg[i].item())
                                loss_lat_src_cv[i].append(batch_loss_lat_trg_cv[i].item())
                            batch_loss_mcd_src_src[i] = torch.mean(batch_loss_mcd_src_src[i])
                            batch_loss_mcd_src_trg_src[i] = torch.mean(batch_loss_mcd_src_trg_src[i])
                            batch_loss_mcd_src_trg[i] = torch.mean(batch_loss_mcd_src_trg[i])
                            batch_loss_lat_src[i] = torch.mean(batch_loss_lat_src[i])
                            batch_loss_lat_src_cv[i] = torch.mean(batch_loss_lat_src_cv[i])
                            batch_loss_scpost_src[i] = torch.mean(batch_loss_scpost_src[i])
                            batch_loss_scpost_src_cv[i] = torch.mean(batch_loss_scpost_src_cv[i])
                            loss_mcd_src_src[i].append(batch_loss_mcd_src_src[i].item())
                            loss_mcd_src_trg[i].append(batch_loss_mcd_src_trg[i].item())
                            loss_mcd_src_trg_src[i].append(batch_loss_mcd_src_trg_src[i].item())
                            loss_scpost_src_cv[i].append(batch_loss_scpost_src_cv[i].item())
                            loss_scpost_src[i].append(batch_loss_scpost_src[i].item())
                            loss_lat_src[i].append(batch_loss_lat_src[i].item())
                            loss_lat_src_cv[i].append(batch_loss_lat_src_cv[i].item())

                        # logging stats summary for current batch
                        text_log = ""
                        if pair_flag: # at least one utt. of target pair exists
                            for i in range(n_ev_cyc):
                                text_log += "[%d] %.3f %.3f %.3f %.3f %.3f %.3f ; %.3f %.3f %.3f %.3f ; "\
                                    "%.3f %.3f %.3f %.3f ; %.3f dB %.3f dB , %.3f dB %.3f dB , "\
                                    "%.3f dB %.3f dB , %.3f %.3f ; %.3f dB %.3f dB , %.3f dB %.3f dB , "\
                                    "%.3f dB %.3f dB , %.3f %.3f ;; " % (i+1, \
                                    batch_loss_mcd_trg_trg[i].item(), batch_loss_mcd_trg_src_trg[i].item(), \
                                    batch_loss_mcd_trg_src[i].item(), batch_loss_mcd_src_src[i].item(), \
                                    batch_loss_mcd_src_trg_src[i].item(), batch_loss_mcd_src_trg[i].item(), \
                                    batch_loss_scpost_trg[i].item(), batch_loss_scpost_trg_cv[i].item(), \
                                    batch_loss_scpost_src[i].item(), batch_loss_scpost_src_cv[i].item(), \
                                    batch_loss_lat_trg[i].item(), batch_loss_lat_trg_cv[i].item(), \
                                    batch_loss_lat_src[i].item(), batch_loss_lat_src_cv[i].item(), \
                                    batch_mcdpow_trg_trg[i], batch_mcd_trg_trg[i], \
                                    batch_mcdpow_trg_src_trg[i], batch_mcd_trg_src_trg[i], \
                                    batch_mcdpow_trg_src[i], batch_mcd_trg_src[i], batch_lat_dist_trgsrc1[i], \
                                    batch_lat_dist_trgsrc2[i], batch_mcdpow_src_src[i], batch_mcd_src_src[i], \
                                    batch_mcdpow_src_trg_src[i], batch_mcd_src_trg_src[i], \
                                    batch_mcdpow_src_trg[i], batch_mcd_src_trg[i], batch_lat_dist_srctrg1[i], \
                                    batch_lat_dist_srctrg2[i])
                        else: # no parallel data exists
                            for i in range(n_ev_cyc):
                                text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f "\
                                    "; %.3f dB %.3f dB , %.3f dB %.3f dB ;; " % (i+1, \
                                    batch_loss_mcd_src_src[i].item(), \
                                    batch_loss_mcd_src_trg_src[i].item(), batch_loss_mcd_src_trg[i].item(), \
                                    batch_loss_scpost_src[i].item(), batch_loss_scpost_src_cv[i].item(), \
                                    batch_loss_lat_src[i].item(), batch_loss_lat_src_cv[i].item(), \
                                    batch_mcdpow_src_src[i], batch_mcd_src_src[i], \
                                    batch_mcdpow_src_trg_src[i], batch_mcd_src_trg_src[i])
                        logging.info("batch eval loss [%d] = %s  (%.3f sec)" % (\
                            c_idx+1, text_log, time.time() - start))
                        total.append(time.time() - start)
                if len(mcdpow_src_trg[i]) > 0:
                    pair_flag = True
                else:
                    pair_flag = False
                for i in range(n_ev_cyc):
                    eval_loss_mcd_src_src[i] = np.mean(loss_mcd_src_src[i])
                    eval_loss_mcd_src_trg_src[i] = np.mean(loss_mcd_src_trg_src[i])
                    eval_loss_mcd_src_trg[i] = np.mean(loss_mcd_src_trg[i])
                    eval_loss_scpost_src_cv[i] = np.mean(loss_scpost_src_cv[i])
                    eval_loss_scpost_src[i] = np.mean(loss_scpost_src[i])
                    eval_loss_lat_src_cv[i] = np.mean(loss_lat_src_cv[i])
                    eval_loss_lat_src[i] = np.mean(loss_lat_src[i])
                    eval_mcdpow_src_src[i] = np.mean(mcdpow_src_src[i])
                    eval_mcd_src_src[i] = np.mean(mcd_src_src[i])
                    eval_mcdpow_src_trg_src[i] = np.mean(mcdpow_src_trg_src[i])
                    eval_mcd_src_trg_src[i] = np.mean(mcd_src_trg_src[i])
                    if pair_flag:
                        eval_mcdpow_src_trg[i] = np.mean(mcdpow_src_trg[i])
                        eval_mcdpowstd_src_trg[i] = np.std(mcdpow_src_trg[i])
                        eval_mcd_src_trg[i] = np.mean(mcd_src_trg[i])
                        eval_mcdstd_src_trg[i] = np.std(mcd_src_trg[i])
                        eval_lat_dist_srctrg1[i] = np.mean(lat_dist_srctrg1[i])
                        eval_lat_dist_srctrg2[i] = np.mean(lat_dist_srctrg2[i])
                    tmp_gv_1 = []
                    tmp_gv_2 = []
                    tmp_gv_3 = []
                    for j in range(n_spk):
                        if len(gv_src_src[i][j]) > 0:
                            tmp_gv_1.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_src[i][j], \
                                axis=0))-np.log(gv_mean[j])))))
                        if len(gv_src_trg_src[i][j]) > 0:
                            tmp_gv_2.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg_src[i][j], \
                                axis=0))-np.log(gv_mean[j])))))
                        if len(gv_src_trg[i][j]) > 0:
                            tmp_gv_3.append(np.mean(np.sqrt(np.square(np.log(np.mean(gv_src_trg[i][j], \
                                axis=0))-np.log(gv_mean[j])))))
                    eval_gv_src_src[i] = np.mean(tmp_gv_1)
                    eval_gv_src_trg_src[i] = np.mean(tmp_gv_2)
                    eval_gv_src_trg[i] = np.mean(tmp_gv_3)
                text_log = ""
                if pair_flag: # at least one utt. of parallel target pair exists
                    for i in range(n_ev_cyc):
                        text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ; "\
                            "%.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f %.3f dB (+- %.3f) "\
                            "%.6f dB (+- %.3f) , %.6f %.6f ;; " % (i+1, eval_loss_mcd_src_src[i], \
                            eval_loss_mcd_src_trg_src[i], eval_loss_mcd_src_trg[i], eval_loss_scpost_src[i], \
                            eval_loss_scpost_src_cv[i], eval_loss_lat_src[i], eval_loss_lat_src_cv[i], \
                            eval_gv_src_src[i], eval_mcdpow_src_src[i], eval_mcd_src_src[i], \
                            eval_gv_src_trg_src[i], eval_mcdpow_src_trg_src[i], eval_mcd_src_trg_src[i], \
                            eval_gv_src_trg[i], eval_mcdpow_src_trg[i], eval_mcdpowstd_src_trg[i], \
                            eval_mcd_src_trg[i], eval_mcdstd_src_trg[i], eval_lat_dist_srctrg1[i], \
                            eval_lat_dist_srctrg2[i])
                else: # no parallel utt. exists
                    for i in range(n_ev_cyc):
                        text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ; "\
                            "%.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f n/a dB (+- n/a) "\
                            "n/a dB (+- n/a) , n/a n/a ;; " % (i+1, eval_loss_mcd_src_src[i], \
                            eval_loss_mcd_src_trg_src[i], eval_loss_mcd_src_trg[i], eval_loss_scpost_src[i], \
                            eval_loss_scpost_src_cv[i], eval_loss_lat_src[i], eval_loss_lat_src_cv[i], \
                            eval_gv_src_src[i], eval_mcdpow_src_src[i], eval_mcd_src_src[i], \
                            eval_gv_src_trg_src[i], eval_mcdpow_src_trg_src[i], eval_mcd_src_trg_src[i], \
                            eval_gv_src_trg[i])
                logging.info("(EPOCH:%d) average evaluation loss = %s  (%.3f min., %.3f sec / batch)" % (\
                    epoch_idx + 1, text_log, np.sum(total) / 60.0, np.mean(total)))
                if pair_flag: # compare conversion error because parallel data exists
                    if (eval_mcdpow_src_trg[0]+eval_mcdpowstd_src_trg[0]+\
                        eval_mcd_src_trg[0]+eval_mcdstd_src_trg[0]) <= \
                            (min_eval_mcdpow_src_trg+min_eval_mcdpowstd_src_trg+\
                                min_eval_mcd_src_trg+min_eval_mcdstd_src_trg):
                        min_eval_loss_mcd_src_src = eval_loss_mcd_src_src[0]
                        min_eval_loss_mcd_src_trg_src = eval_loss_mcd_src_trg_src[0]
                        min_eval_loss_mcd_src_trg = eval_loss_mcd_src_trg[0]
                        min_eval_loss_scpost_src = eval_loss_scpost_src[0]
                        min_eval_loss_scpost_src_cv = eval_loss_scpost_src_cv[0]
                        min_eval_loss_lat_src = eval_loss_lat_src[0]
                        min_eval_loss_lat_src_cv = eval_loss_lat_src_cv[0]
                        min_eval_gv_src_src = eval_gv_src_src[0]
                        min_eval_mcdpow_src_src = eval_mcdpow_src_src[0]
                        min_eval_mcd_src_src = eval_mcd_src_src[0]
                        min_eval_gv_src_trg_src = eval_gv_src_trg_src[0]
                        min_eval_mcdpow_src_trg_src = eval_mcdpow_src_trg_src[0]
                        min_eval_mcd_src_trg_src = eval_mcd_src_trg_src[0]
                        min_eval_gv_src_trg = eval_gv_src_trg[0]
                        min_eval_mcdpow_src_trg = eval_mcdpow_src_trg[0]
                        min_eval_mcdpowstd_src_trg = eval_mcdpowstd_src_trg[0]
                        min_eval_mcd_src_trg = eval_mcd_src_trg[0]
                        min_eval_mcdstd_src_trg = eval_mcdstd_src_trg[0]
                        min_eval_lat_dist_srctrg1 = eval_lat_dist_srctrg1[0]
                        min_eval_lat_dist_srctrg2 = eval_lat_dist_srctrg2[0]
                        min_idx = epoch_idx
                    text_log = "%.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ; "\
                        "%.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f %.3f dB (+- %.3f) "\
                        "%.6f dB (+- %.3f) , %.6f %.6f ;; " % (\
                            min_eval_loss_mcd_src_src, min_eval_loss_mcd_src_trg_src, \
                            min_eval_loss_mcd_src_trg, min_eval_loss_scpost_src, min_eval_loss_scpost_src_cv, \
                            min_eval_loss_lat_src, min_eval_loss_lat_src_cv, min_eval_gv_src_src, \
                            min_eval_mcdpow_src_src, min_eval_mcd_src_src, min_eval_gv_src_trg_src, \
                            min_eval_mcdpow_src_trg_src, min_eval_mcd_src_trg_src, min_eval_gv_src_trg, \
                            min_eval_mcdpow_src_trg, min_eval_mcdpowstd_src_trg, min_eval_mcd_src_trg, \
                            min_eval_mcdstd_src_trg, min_eval_lat_dist_srctrg1, min_eval_lat_dist_srctrg2)
                else: # compare reconst. error because there is no parallel data
                    if (eval_mcdpow_src_src[0]+eval_mcd_src_src[0]) <= \
                        (min_eval_mcdpow_src_src+min_eval_mcd_src_src):
                        min_eval_loss_mcd_src_src = eval_loss_mcd_src_src[0]
                        min_eval_loss_mcd_src_trg_src = eval_loss_mcd_src_trg_src[0]
                        min_eval_loss_mcd_src_trg = eval_loss_mcd_src_trg[0]
                        min_eval_loss_scpost_src = eval_loss_scpost_src[0]
                        min_eval_loss_scpost_src_cv = eval_loss_scpost_src_cv[0]
                        min_eval_loss_lat_src = eval_loss_lat_src[0]
                        min_eval_loss_lat_src_cv = eval_loss_lat_src_cv[0]
                        min_eval_gv_src_src = eval_gv_src_src[0]
                        min_eval_mcdpow_src_src = eval_mcdpow_src_src[0]
                        min_eval_mcd_src_src = eval_mcd_src_src[0]
                        min_eval_gv_src_trg_src = eval_gv_src_trg_src[0]
                        min_eval_mcdpow_src_trg_src = eval_mcdpow_src_trg_src[0]
                        min_eval_mcd_src_trg_src = eval_mcd_src_trg_src[0]
                        min_eval_gv_src_trg = eval_gv_src_trg[0]
                        min_idx = epoch_idx
                    text_log = "%.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ; "\
                        "%.6f %.3f dB %.6f dB , %.3f %.3f dB %.3f dB , %.6f n/a dB (+- n/a) "\
                        "n/a dB (+- n/a) , n/a n/a ;; " % (\
                            min_eval_loss_mcd_src_src, min_eval_loss_mcd_src_trg_src, \
                            min_eval_loss_mcd_src_trg, min_eval_loss_scpost_src, min_eval_loss_scpost_src_cv, \
                            min_eval_loss_lat_src, min_eval_loss_lat_src_cv, min_eval_gv_src_src, \
                            min_eval_mcdpow_src_src, min_eval_mcd_src_src, min_eval_gv_src_trg_src, \
                            min_eval_mcdpow_src_trg_src, min_eval_mcd_src_trg_src, min_eval_gv_src_trg)
                logging.info("min_eval_acc= %s min_idx=%d" % (text_log, min_idx+1))
                #logging.info(count_spk_pair_cv)
            loss = []
            loss_mcd_src_src = []
            loss_mcd_src_trg_src = []
            loss_mcd_src_trg = []
            loss_lat_src = []
            loss_lat_src_cv = []
            loss_scpost_src = []
            loss_scpost_src_cv = []
            gv_src_src = []
            mcdpow_src_src = []
            mcd_src_src = []
            gv_src_trg_src = []
            mcdpow_src_trg_src = []
            mcd_src_trg_src = []
            gv_src_trg = []
            mcdpow_src_trg = []
            mcd_src_trg = []
            lat_dist_srctrg1 = []
            lat_dist_srctrg2 = []
            for i in range(args.n_cyc):
                loss_mcd_src_src.append([])
                loss_mcd_src_trg_src.append([])
                loss_mcd_src_trg.append([])
                loss_lat_src.append([])
                loss_lat_src_cv.append([])
                loss_scpost_src.append([])
                loss_scpost_src_cv.append([])
                gv_src_src.append([])
                mcdpow_src_src.append([])
                mcd_src_src.append([])
                gv_src_trg_src.append([])
                mcdpow_src_trg_src.append([])
                mcd_src_trg_src.append([])
                gv_src_trg.append([])
                mcdpow_src_trg.append([])
                mcd_src_trg.append([])
                lat_dist_srctrg1.append([])
                lat_dist_srctrg2.append([])
                for j in range(n_spk):
                    gv_src_src[i].append([])
                    gv_src_trg_src[i].append([])
                    gv_src_trg[i].append([])
            total = []
            iter_count = 0 
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model_encoder.train()
            model_decoder.train()
            for param in model_encoder.parameters():
                param.requires_grad = True
            for param in model_decoder.parameters():
                param.requires_grad = True
            for param in model_encoder.scale_in.parameters():
                param.requires_grad = False
            for param in model_decoder.scale_out.parameters():
                param.requires_grad = False
            # start next epoch
            if epoch_idx < args.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))
            
            if args.batch_size > 0: # frame-length mini-batch
                pair_flag = False
                for i in range(n_batch_utt): # n_batch_utt = number of utterances in a batch sequence
                    if file_src_trg_flag[i]: # parallel utterance of target exists
                        logging.info("%s %s %d %d %d %d %d %d %d %d %d %d" % (
                            os.path.join(os.path.basename(os.path.dirname(featfile_src[i])), \
                                os.path.basename(featfile_src[i])),os.path.join(os.path.basename(\
                                    os.path.dirname(featfile_src_trg[i])),os.path.basename(\
                                        featfile_src_trg[i])), flens_src[i], flens_src_trg[i], \
                                            flens_spc_src[i], flens_spc_src_trg[i], src_idx_s, src_idx_e, \
                                                spcidx_src_s_idx[i], spcidx_src_e_idx[i], \
                                                    spcidx_src[i,spcidx_src_s_idx[i]].item(), \
                                                        spcidx_src[i,spcidx_src_e_idx[i]].item()))
                        pair_flag = True
                    else: # no parallel utterance of target
                        logging.info("%s %d %d %d %d %d %d %d %d" % (
                            os.path.join(os.path.basename(os.path.dirname(featfile_src[i])), \
                                os.path.basename(featfile_src[i])), flens_src[i], flens_spc_src[i], \
                                    src_idx_s, src_idx_e, spcidx_src_s_idx[i], spcidx_src_e_idx[i], \
                                        spcidx_src[i,spcidx_src_s_idx[i]].item(), \
                                            spcidx_src[i,spcidx_src_e_idx[i]].item()))
                    text_pair_spk = "" # speaker-pair logging
                    for j in range(args.n_cyc):
                        text_pair_spk += "[%d] %s " % (j+1, pair_spk_list[j][i])
                    logging.info("%s" % text_pair_spk)

                if src_idx_s > 0 and prev_featfile_src == featfile_src and iter_count > 0: # cont. batch seq.
                    for i in range(args.n_cyc):
                        if i > 0: # [2nd, 3rd, ..., Nth] cycle
                            # encoding input features
                            batch_lat_src[i], batch_param, y_in_pp_src[i], h_in_pp_src[i], \
                                batch_mle_lat_src[i] = \
                                model_encoder(torch.cat((batch_src[:,src_idx_s:src_idx_e+1,:stdim], \
                                    batch_trj_src_trg_src[i-1]),2), Variable(y_in_pp_src[i].data).detach(), \
                                        h_in=Variable(h_in_pp_src[i].data).detach(), do=True)
                            batch_scpost_src[i] = batch_param[:,:,:n_spk] #speaker_posterior
                            batch_latpost_src[i] = batch_param[:,:,n_spk:] #latent_posterior

                            # spectral reconstruction
                            batch_trj_src_src[i], y_in_src_src[i], h_in_src_src[i] = \
                                model_decoder(torch.cat((batch_src_src_code, batch_lat_src[i]),2), \
                                    Variable(y_in_src_src[i].data).detach(), \
                                        h_in=Variable(h_in_src_src[i].data).detach(), do=True)

                            # spectral conversion
                            batch_trj_src_trg[i], y_in_src_trg[i], h_in_src_trg[i] = \
                                model_decoder(torch.cat((batch_src_trg_code[i][:,src_idx_s:src_idx_e+1], \
                                    batch_lat_src[i]),2), Variable(y_in_src_trg[i].data).detach(), \
                                        h_in=Variable(h_in_src_trg[i].data).detach(), do=True)

                            # encoding converted features
                            batch_lat_src_trg[i], batch_param, y_in_pp_src_trg[i], \
                                h_in_pp_src_trg[i], batch_mle_lat_src_trg[i] = model_encoder(torch.cat(( \
                                    batch_cv_src[i][:,src_idx_s:src_idx_e+1], batch_trj_src_trg[i]),2), \
                                        Variable(y_in_pp_src_trg[i].data).detach(), \
                                            h_in=Variable(h_in_pp_src_trg[i].data).detach(), do=True)
                            batch_scpost_src_trg[i] = batch_param[:,:,:n_spk] #speaker_posterior
                            batch_latpost_src_trg[i] = batch_param[:,:,n_spk:] #latent_posterior

                            # cyclic spectral reconstruction
                            batch_trj_src_trg_src[i], y_in_src_trg_src[i], h_in_src_trg_src[i] = \
                                model_decoder(torch.cat((batch_src_src_code, \
                                    batch_lat_src_trg[i]),2), Variable(y_in_src_trg_src[i].data).detach(), \
                                        h_in=Variable(h_in_src_trg_src[i].data).detach(), do=True)
                        else: # 1st cycle
                            # encoding input features
                            batch_lat_src[0], batch_param, y_in_pp_src[0], h_in_pp_src[0], \
                                batch_mle_lat_src[0] = \
                                model_encoder(batch_src[:,src_idx_s:src_idx_e+1], \
                                    Variable(y_in_pp_src[0].data).detach(), \
                                        h_in=Variable(h_in_pp_src[0].data).detach(), do=True)
                            batch_scpost_src[0] = batch_param[:,:,:n_spk] #speaker_posterior
                            batch_latpost_src[0] = batch_param[:,:,n_spk:] #latent_posterior

                            # spectral reconstruction
                            batch_trj_src_src[0], y_in_src_src[0], h_in_src_src[0] = \
                                model_decoder(torch.cat((batch_src_src_code, batch_lat_src[0]),2), \
                                    Variable(y_in_src_src[0].data).detach(), \
                                        h_in=Variable(h_in_src_src[0].data).detach(), do=True)

                            # spectral conversion
                            batch_trj_src_trg[0], y_in_src_trg[0], h_in_src_trg[0] = \
                                model_decoder(torch.cat((batch_src_trg_code[i][:,src_idx_s:src_idx_e+1], \
                                    batch_lat_src[0]),2), Variable(y_in_src_trg[0].data).detach(), \
                                        h_in=Variable(h_in_src_trg[0].data).detach(), do=True)

                            # encoding converted features
                            batch_lat_src_trg[0], batch_param, y_in_pp_src_trg[0], h_in_pp_src_trg[0], \
                                batch_mle_lat_src_trg[0] = \
                                    model_encoder(torch.cat((batch_cv_src[i][:,src_idx_s:src_idx_e+1], \
                                    batch_trj_src_trg[0]),2), Variable(y_in_pp_src_trg[0].data).detach(), \
                                        h_in=Variable(h_in_pp_src_trg[0].data).detach(), do=True)
                            batch_scpost_src_trg[0] = batch_param[:,:,:n_spk] #speaker_posterior
                            batch_latpost_src_trg[0] = batch_param[:,:,n_spk:] #latent_posterior

                            # cyclic spectral reconstruction
                            batch_trj_src_trg_src[0], y_in_src_trg_src[0], h_in_src_trg_src[0] = \
                                model_decoder(torch.cat((batch_src_src_code, \
                                    batch_lat_src_trg[0]),2), Variable(y_in_src_trg_src[0].data).detach(), \
                                        h_in=Variable(h_in_src_trg_src[0].data).detach(), do=True)

                            # store est. data from the 1st cycle for accuracy on utterance-level
                            tmp_src_src = torch.cat((tmp_src_src, batch_trj_src_src[0][:,:,1:]), 1)
                            tmp_src_trg = torch.cat((tmp_src_trg, batch_trj_src_trg[0][:,:,1:]), 1)
                            tmp_src_trg_src = torch.cat((tmp_src_trg_src, batch_trj_src_trg_src[0][:,:,1:]), 1)
                            trj_src_trg = torch.cat((trj_src_trg, batch_trj_src_trg[0]), 1)
                            trj_lat_src = torch.cat((trj_lat_src, batch_mle_lat_src[0]), 1)
                else: # new batch sequence, previous output and hidden state are zeros initialized
                    if n_batch_utt == args.batch_size_utt: # batch seq. size is the same as setting
                        y_in_pp_ = y_in_pp
                        y_in_src_ = y_in_src
                        y_in_trg_ = y_in_trg
                    else: # current batch size is reduced due to unrounded values of total_data/batch_size
                        y_in_pp_ = y_in_pp_mod
                        y_in_src_ = y_in_src_mod
                        y_in_trg_ = y_in_trg_mod
                    for i in range(args.n_cyc):
                        if i > 0: # [2nd, 3rd, ..., Nth] cycle
                            # encoding input features
                            batch_lat_src[i], batch_param, y_in_pp_src[i], h_in_pp_src[i], \
                                batch_mle_lat_src[i] = \
                                model_encoder(torch.cat((batch_src[:,src_idx_s:src_idx_e+1,:stdim], \
                                    batch_trj_src_trg_src[i-1]),2), y_in_pp_, do=True)
                            batch_scpost_src[i] = batch_param[:,:,:n_spk] #speaker_posterior
                            batch_latpost_src[i] = batch_param[:,:,n_spk:] #latent_posterior

                            # spectral reconstruction
                            batch_trj_src_src[i], y_in_src_src[i], h_in_src_src[i] = \
                                model_decoder(torch.cat((batch_src_src_code, \
                                    batch_lat_src[i]),2), y_in_src_, do=True)

                            # spectral conversion
                            batch_trj_src_trg[i], y_in_src_trg[i], h_in_src_trg[i] = \
                                model_decoder(torch.cat((batch_src_trg_code[i][:,src_idx_s:src_idx_e+1], \
                                    batch_lat_src[i]),2), y_in_trg_, do=True)

                            # encoding converted features
                            batch_lat_src_trg[i], batch_param, y_in_pp_src_trg[i], h_in_pp_src_trg[i], \
                                batch_mle_lat_src_trg[i] = \
                                    model_encoder(torch.cat((batch_cv_src[i][:,src_idx_s:src_idx_e+1], \
                                    batch_trj_src_trg[i]),2), y_in_pp_, do=True)
                            batch_scpost_src_trg[i] = batch_param[:,:,:n_spk] #speaker_posterior
                            batch_latpost_src_trg[i] = batch_param[:,:,n_spk:] #latent_posterior

                            # cyclic spectral reconstruction
                            batch_trj_src_trg_src[i], y_in_src_trg_src[i], h_in_src_trg_src[i] = \
                                model_decoder(torch.cat((batch_src_src_code, \
                                    batch_lat_src_trg[i]),2), y_in_src_, do=True)
                        else: # 1st cycle
                            # encoding input features
                            batch_lat_src[0], batch_param, y_in_pp_src[0], h_in_pp_src[0], \
                                batch_mle_lat_src[0] = \
                                model_encoder(batch_src[:,src_idx_s:src_idx_e+1], y_in_pp_, do=True)
                            batch_scpost_src[0] = batch_param[:,:,:n_spk] #speaker_posterior
                            batch_latpost_src[0] = batch_param[:,:,n_spk:] #latent_posterior

                            # spectral reconstruction
                            batch_trj_src_src[0], y_in_src_src[0], h_in_src_src[0] = \
                                model_decoder(torch.cat((batch_src_src_code, \
                                    batch_lat_src[0]),2), y_in_src_, do=True)

                            # spectral conversion
                            batch_trj_src_trg[0], y_in_src_trg[0], h_in_src_trg[0] = \
                                model_decoder(torch.cat((batch_src_trg_code[i][:,src_idx_s:src_idx_e+1], \
                                    batch_lat_src[0]),2), y_in_trg_, do=True)

                            # encoding converted features
                            batch_lat_src_trg[0], batch_param, y_in_pp_src_trg[0], h_in_pp_src_trg[0], \
                                batch_mle_lat_src_trg[0] = \
                                    model_encoder(torch.cat((batch_cv_src[i][:,src_idx_s:src_idx_e+1], \
                                    batch_trj_src_trg[0]),2), y_in_pp_, do=True)
                            batch_scpost_src_trg[0] = batch_param[:,:,:n_spk] #speaker_posterior
                            batch_latpost_src_trg[0] = batch_param[:,:,n_spk:] #latent_posterior

                            # cyclic spectral reconstruction
                            batch_trj_src_trg_src[0], y_in_src_trg_src[0], h_in_src_trg_src[0] = \
                                model_decoder(torch.cat((batch_src_src_code, \
                                    batch_lat_src_trg[0]),2), y_in_src_, do=True)

                            # compute utterance-level accuracy if this is not the 1st batch in the epoch
                            if iter_count > 0:
                                for j in range(n_batch_utt):
                                    spk_src_ = os.path.basename(os.path.dirname(prev_featfile_src[j]))
                                    for k in range(n_spk):
                                        if spk_src_ == spk_list[k]:
                                            #GV stat of reconst.
                                            gv_src_src[0][k].append(torch.var(\
                                                tmp_src_src[j,:prev_flens_src[j]], 0).cpu().data.numpy())
                                            #GV stat of cyclic reconst.
                                            gv_src_trg_src[0][k].append(torch.var(\
                                                tmp_src_trg_src[j,:prev_flens_src[j]], 0).cpu().data.numpy())
                                            break
                                    spk_trg_ = prev_pair_spk[i][j] #find the target pair in prev. batch
                                    for k in range(n_spk):
                                        if spk_trg_ == spk_list[k]:
                                            #GV stat of converted
                                            gv_src_trg[0][k].append(torch.var(\
                                                tmp_src_trg[j,:prev_flens_src[j]], 0).cpu().data.numpy())
                                            break

                            # store est. data from the 1st cycle for accuracy on utterance-level
                            tmp_src_src = batch_trj_src_src[0][:,:,1:]
                            tmp_src_trg = batch_trj_src_trg[0][:,:,1:]
                            tmp_src_trg_src = batch_trj_src_trg_src[0][:,:,1:]
                            trj_src_trg = batch_trj_src_trg[0]
                            trj_lat_src = batch_mle_lat_src[0]
                prev_featfile_src = featfile_src #record list of uttterances in current batch seq.
                prev_pair_spk = pair_spk_list #record list of target speakers in current batch seq.

                ## optimization performed only for utterances with valid length,
                ##   i.e., current mini-batch frames are still within the corresponding utterances 
                # check whether current batch has at least 1 utterance with valid length
                # if not, don't optimize at all (this is caused by having only ending silent frames)
                if len(select_utt_idx) > 0:
                    for i in range(args.n_cyc):
                        batch_mcdpow_src_src[i] = []
                        batch_mcd_src_src[i] = []
                        batch_mcdpow_src_trg_src[i] = []
                        batch_mcd_src_trg_src[i] = []

                    for i in range(args.n_cyc): # iterate over all cycles
                        for k, j in enumerate(select_utt_idx): # iterate over all valid utterances
                            src_idx_e_ = src_idx_s + flen_acc[j] # valid length/idcs of current utt. for optim.

                            # valid spectral segment for optim.
                            batch_src_optim = batch_src[j,src_idx_s:src_idx_e_,stdim:]

                            # mel-cepstral distortion (MCD)-based L1-loss of spectral features
                            _, tmp_batch_loss_mcd_src_src, _ = criterion_mcd(\
                                batch_trj_src_src[i][j,:flen_acc[j]], batch_src_optim)
                            _, tmp_batch_loss_mcd_src_trg_src, _ = criterion_mcd(\
                                batch_trj_src_trg_src[i][j,:flen_acc[j]], batch_src_optim)
                            _, tmp_batch_loss_mcd_src_trg, _ = criterion_mcd(\
                                batch_trj_src_trg[i][j,:flen_acc[j]], batch_src_optim)

                            # cross-entropy (CE) of speaker-posterior
                            tmp_batch_loss_scpost_src = criterion_ce(batch_scpost_src[i][j,:flen_acc[j]], \
                                batch_src_src_class_code[j,src_idx_s:src_idx_e_])
                            tmp_batch_loss_scpost_src_cv = criterion_ce(\
                                batch_scpost_src_trg[i][j,:flen_acc[j]], \
                                    batch_src_trg_class_code_list[i][j,src_idx_s:src_idx_e_])

                            # KL-divergence of latent-posterior to the standard Laplacian prior
                            tmp_batch_loss_lat_src = loss_vae_laplace(\
                                batch_latpost_src[i][j,:flen_acc[j]], lat_dim=args.lat_dim, clip=True)
                            tmp_batch_loss_lat_src_cv = loss_vae_laplace(\
                                batch_latpost_src_trg[i][j,:flen_acc[j]], lat_dim=args.lat_dim, clip=True)

                            # record the loss statistics
                            loss_mcd_src_src[i].append(tmp_batch_loss_mcd_src_src.item())
                            loss_mcd_src_trg_src[i].append(tmp_batch_loss_mcd_src_trg_src.item())
                            loss_mcd_src_trg[i].append(tmp_batch_loss_mcd_src_trg.item())
                            loss_scpost_src[i].append(tmp_batch_loss_scpost_src.item())
                            loss_scpost_src_cv[i].append(tmp_batch_loss_scpost_src_cv.item())
                            loss_lat_src_cv[i].append(tmp_batch_loss_lat_src_cv.item())
                            loss_lat_src[i].append(tmp_batch_loss_lat_src.item())
                            if k > 0:
                                batch_loss_mcd_src_src[i] = torch.cat((batch_loss_mcd_src_src[i], \
                                    tmp_batch_loss_mcd_src_src.unsqueeze(0)))
                                batch_loss_mcd_src_trg_src[i] = torch.cat((batch_loss_mcd_src_trg_src[i], \
                                    tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)))
                                batch_loss_mcd_src_trg[i] = torch.cat((batch_loss_mcd_src_trg[i], \
                                    tmp_batch_loss_mcd_src_trg.unsqueeze(0)))
                                batch_loss_scpost_src[i] = torch.cat((batch_loss_scpost_src[i], \
                                    tmp_batch_loss_scpost_src.unsqueeze(0)))
                                # use 1st and 2nd scpost for 2nd scpost loss
                                batch_loss_scpost_src_cv[i] = torch.cat((batch_loss_scpost_src_cv[i], \
                                    tmp_batch_loss_scpost_src.unsqueeze(0), \
                                    tmp_batch_loss_scpost_src_cv.unsqueeze(0)))
                                batch_loss_lat_src[i] = torch.cat((batch_loss_lat_src[i], \
                                    tmp_batch_loss_lat_src.unsqueeze(0)))
                                # use 1st and 2nd latent for 2nd latent loss
                                batch_loss_lat_src_cv[i] = torch.cat((batch_loss_lat_src_cv[i], \
                                                                tmp_batch_loss_lat_src.unsqueeze(0), \
                                                                tmp_batch_loss_lat_src_cv.unsqueeze(0)))
                            else:
                                batch_loss_mcd_src_src[i] = tmp_batch_loss_mcd_src_src.unsqueeze(0)
                                batch_loss_mcd_src_trg_src[i] = tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)
                                batch_loss_mcd_src_trg[i] = tmp_batch_loss_mcd_src_trg.unsqueeze(0)
                                batch_loss_scpost_src[i] = tmp_batch_loss_scpost_src.unsqueeze(0)
                                # use 1st and 2nd scpost for 2nd scpost loss
                                batch_loss_scpost_src_cv[i] = torch.cat((tmp_batch_loss_scpost_src.unsqueeze(0), \
                                                                    tmp_batch_loss_scpost_src_cv.unsqueeze(0)))
                                batch_loss_lat_src[i] = tmp_batch_loss_lat_src.unsqueeze(0)
                                # use 1st and 2nd latent for 2nd latent loss
                                batch_loss_lat_src_cv[i] = torch.cat((tmp_batch_loss_lat_src.unsqueeze(0), \
                                                                tmp_batch_loss_lat_src_cv.unsqueeze(0)))

                        # accumulate loss for optimization
                        if i > 0: # [2nd, 3rd, ..., Nth] cycle
                            batch_loss += \
                                batch_loss_mcd_src_src[i].sum() + batch_loss_mcd_src_trg_src[i].sum() + \
                                batch_loss_lat_src[i].sum() + batch_loss_lat_src_cv[i].sum() + \
                                batch_loss_scpost_src[i].sum() + batch_loss_scpost_src_cv[i].sum()
                        else: # 1st cycle
                            if not half_cyc:
                                batch_loss = \
                                    batch_loss_mcd_src_src[0].sum() + batch_loss_mcd_src_trg_src[0].sum() + \
                                    batch_loss_lat_src[0].sum() + batch_loss_lat_src_cv[0].sum() + \
                                    batch_loss_scpost_src[0].sum() + batch_loss_scpost_src_cv[0].sum()
                            else: # only 1st half of the cycle due to n_cyc = 0 (VAE)
                                batch_loss = \
                                    batch_loss_mcd_src_src[0].sum() + \
                                    batch_loss_lat_src[0].sum() + batch_loss_scpost_src[0].sum()

                        # record mean-loss from all utterances in this batch
                        batch_loss_mcd_src_src[i] = torch.mean(batch_loss_mcd_src_src[i])
                        batch_loss_mcd_src_trg_src[i] = torch.mean(batch_loss_mcd_src_trg_src[i])
                        batch_loss_mcd_src_trg[i] = torch.mean(batch_loss_mcd_src_trg[i])
                        batch_loss_scpost_src[i] = torch.mean(batch_loss_scpost_src[i])
                        batch_loss_scpost_src_cv[i] = torch.mean(batch_loss_scpost_src_cv[i])
                        batch_loss_lat_src[i] = torch.mean(batch_loss_lat_src[i])
                        batch_loss_lat_src_cv[i] = torch.mean(batch_loss_lat_src_cv[i])

                    # compute gradients and backpropagate
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    loss.append(batch_loss.item())

                    # compute MCD-based L2-loss (true MCD values) of (cyclic) reconst. spectra
                    print_mcd_flag = False
                    for i in range(args.n_cyc):
                        batch_mcdpow_src_src[i] = []
                        batch_mcd_src_src[i] = []
                        batch_mcdpow_src_trg_src[i] = []
                        batch_mcd_src_trg_src[i] = []

                    for j in select_utt_idx: # iterate over all valid utterances in optim.
                        if spcidx_src_s_idx[j] >= 0: # calculate MCD only with speech frames
                            print_mcd_flag = True
                            for i in range(args.n_cyc): # iterate over all cycles
                                # reconst. MCD with 0th power
                                tmp_batch_mcdpow_src_src, _ = dtw.calc_mcd(\
                                    np.array(torch.index_select(batch_src[j],0,spcidx_src[j,\
                                        spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1])[:,\
                                        stdim:].cpu().data.numpy(), \
                                        dtype=np.float64), \
                                    np.array(torch.index_select(\
                                        batch_trj_src_src[i][j],0,spcidx_src[j,\
                                        spcidx_src_s_idx[j]:\
                                        spcidx_src_e_idx[j]+1]-src_idx_s).cpu().data.numpy(), \
                                        dtype=np.float64))
                                # reconst. MCD w/o 0th power, i.e., [:,1:]
                                tmp_batch_mcd_src_src, _ = dtw.calc_mcd(\
                                    np.array(torch.index_select(batch_src[j],0,spcidx_src[j,\
                                        spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1])[:,\
                                        stdim_:].cpu().data.numpy(), dtype=np.float64), \
                                    np.array(torch.index_select(\
                                        batch_trj_src_src[i][j],0,spcidx_src[j,\
                                        spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1]-\
                                        src_idx_s)[:,1:].cpu().data.numpy(), dtype=np.float64))
                                # cyclic reconst. MCD with 0th power
                                tmp_batch_mcdpow_src_trg_src, _ = dtw.calc_mcd(\
                                    np.array(torch.index_select(batch_src[j],0,spcidx_src[j,\
                                        spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1])[:,\
                                        stdim:].cpu().data.numpy(), dtype=np.float64), \
                                    np.array(torch.index_select(\
                                        batch_trj_src_trg_src[i][j],0,spcidx_src[j,\
                                        spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1]-\
                                        src_idx_s).cpu().data.numpy(), dtype=np.float64))
                                # cyclic reconst. MCD w/o 0th power, i.e., [:,1:]
                                tmp_batch_mcd_src_trg_src, _ = dtw.calc_mcd(\
                                    np.array(torch.index_select(batch_src[j],0,spcidx_src[j,\
                                        spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1])[:,\
                                        stdim_:].cpu().data.numpy(),\
                                        dtype=np.float64), \
                                    np.array(torch.index_select(\
                                        batch_trj_src_trg_src[i][j],0,spcidx_src[j,\
                                        spcidx_src_s_idx[j]:spcidx_src_e_idx[j]+1]-\
                                        src_idx_s)[:,1:].cpu().data.numpy(), dtype=np.float64))
                              
                                # record loss statistics
                                batch_mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                                batch_mcd_src_src[i].append(tmp_batch_mcd_src_src)
                                batch_mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                                batch_mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)
                                mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                                mcd_src_src[i].append(tmp_batch_mcd_src_src)
                                mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                                mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)
                           
                    # stats logging of current mini-batch
                    text_log = "%.3f ;; " % batch_loss.item()
                    if print_mcd_flag:
                        for i in range(args.n_cyc):
                            text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ; "\
                                "%.3f dB %.3f dB , %.3f dB %.3f dB ;; " % (i+1, \
                                batch_loss_mcd_src_src[i].item(), \
                                batch_loss_mcd_src_trg_src[i].item(), batch_loss_mcd_src_trg[i].item(), \
                                batch_loss_scpost_src[i].item(), batch_loss_scpost_src_cv[i].item(), \
                                batch_loss_lat_src[i].item(), batch_loss_lat_src_cv[i].item(), \
                                np.mean(batch_mcdpow_src_src[i]), np.mean(batch_mcd_src_src[i]), \
                                np.mean(batch_mcdpow_src_trg_src[i]), np.mean(batch_mcd_src_trg_src[i]))
                    else:
                        for i in range(args.n_cyc):
                            text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ;; " % (
                                i+1, batch_loss_mcd_src_src[i].item(), \
                                batch_loss_mcd_src_trg_src[i].item(), batch_loss_mcd_src_trg[i].item(), \
                                batch_loss_scpost_src[i].item(), batch_loss_scpost_src_cv[i].item(), \
                                batch_loss_lat_src[i].item(), batch_loss_lat_src_cv[i].item())
                    logging.info("batch loss [%d] = %s  (%.3f sec)" % (\
                        c_idx_src+1, text_log, time.time() - start))
                    iter_idx += 1
                    iter_count += 1
                total.append(time.time() - start) #time stats
            else: # utterance-length mini-batch
                pair_flag = False
                for i in range(n_batch_utt): # n_batch_utt = number of utterances in a batch sequence
                    if file_src_trg_flag[i]: # parallel utterance of target exists
                        logging.info("%s %s %d %d %d %d" % (
                            os.path.join(os.path.basename(os.path.dirname(featfile_src[i])), \
                                os.path.basename(featfile_src[i])), os.path.join(\
                                    os.path.basename(os.path.dirname(featfile_src_trg[i])), \
                                        os.path.basename(featfile_src_trg[i])), flens_src[i], \
                                            flens_src_trg[i], flens_spc_src[i], flens_spc_src_trg[i]))
                        pair_flag = True
                    else: # no parallel utterance of target
                        logging.info("%s %d %d" % (
                            os.path.join(os.path.basename(os.path.dirname(featfile_src[i])), \
                                os.path.basename(featfile_src[i])), flens_src[i], flens_spc_src[i]))
                    text_pair_spk = "" # speaker-pair logging
                    for j in range(args.n_cyc):
                        text_pair_spk += "[%d] %s " % (j+1, pair_spk_list[j][i])
                    logging.info("%s" % text_pair_spk)

                if n_batch_utt == args.batch_size_utt: # batch seq. size is the same as setting
                    y_in_pp_ = y_in_pp
                    y_in_trg_ = y_in_trg
                    y_in_src_ = y_in_src
                else: # current batch size is reduced due to unrounded values of total_data/batch_size
                    y_in_pp_ = y_in_pp_mod
                    y_in_trg_ = y_in_trg_mod
                    y_in_src_ = y_in_src_mod

                # for utterance-level latent accuracy
                if pair_flag:
                    with torch.no_grad():
                        _, _, _, _, trj_lat_srctrg = model_encoder(batch_src_trg, y_in_pp_)

                # iterate each cycle
                for i in range(args.n_cyc):
                    batch_mcdpow_src_src[i] = []
                    batch_mcd_src_src[i] = []
                    batch_mcdpow_src_trg_src[i] = []
                    batch_mcd_src_trg_src[i] = []

                    if i > 0: # [2nd, 3rd, ..., Nth] cycle
                        # encoding input features
                        batch_lat_src[i], batch_param, _, _, batch_mle_lat_src[i] = \
                            model_encoder(torch.cat((batch_src[:,:,:stdim], \
                                batch_trj_src_trg_src[i-1]), 2), y_in_pp_, do=True)
                        batch_scpost_src[i] = batch_param[:,:,:n_spk] #speaker-posterior
                        batch_latpost_src[i] = batch_param[:,:,n_spk:] #latent-posterior

                        # spectral reconstruction
                        batch_trj_src_src[i], _, _ = \
                            model_decoder(torch.cat((batch_src_src_code, batch_lat_src[i]), 2), \
                                y_in_src_, do=True)

                        # spectral conversion
                        batch_trj_src_trg[i], _, _ = \
                            model_decoder(torch.cat((batch_src_trg_code[i], batch_lat_src[i]), 2), \
                                y_in_trg_, do=True)

                        # encoding converted features
                        batch_lat_src_trg[i], batch_param, _, _, batch_mle_lat_src_trg[i] = \
                            model_encoder(torch.cat((batch_cv_src[i], batch_trj_src_trg[i]), 2), \
                                y_in_pp_, do=True)
                        batch_scpost_src_trg[i] = batch_param[:,:,:n_spk] #speaker-posterior
                        batch_latpost_src_trg[i] = batch_param[:,:,n_spk:] #latent-posterior

                        # cyclic spectral reconstruction
                        batch_trj_src_trg_src[i], _, _ = \
                            model_decoder(torch.cat((batch_src_src_code, batch_lat_src_trg[i]), 2), \
                                y_in_src_, do=True)
                    else: # 1st cycle
                        batch_lat_src[0], batch_param, _, _, batch_mle_lat_src[0] = \
                            model_encoder(batch_src, y_in_pp_, do=True)
                        batch_scpost_src[0] = batch_param[:,:,:n_spk] #speaker-posterior
                        batch_latpost_src[0] = batch_param[:,:,n_spk:] #latent-posterior

                        # spectral reconstruction
                        batch_trj_src_src[0], _, _ = \
                            model_decoder(torch.cat((batch_src_src_code, batch_lat_src[0]), 2), \
                                y_in_src_, do=True)

                        # spectral conversion
                        batch_trj_src_trg[0], _, _ = \
                            model_decoder(torch.cat((batch_src_trg_code[i], batch_lat_src[0]), 2), \
                                y_in_trg_, do=True)

                        # encoding converted features
                        batch_lat_src_trg[0], batch_param, _, _, batch_mle_lat_src_trg[0] = \
                            model_encoder(torch.cat((batch_cv_src[i], batch_trj_src_trg[0]), 2), \
                                y_in_pp_, do=True)
                        batch_scpost_src_trg[0] = batch_param[:,:,:n_spk] #speaker-posterior
                        batch_latpost_src_trg[0] = batch_param[:,:,n_spk:] #latent-posterior

                        # cyclic spectral reconstruction
                        batch_trj_src_trg_src[0], _, _ = \
                            model_decoder(torch.cat((batch_src_src_code, batch_lat_src_trg[0]), 2), \
                                y_in_src_, do=True)

                        # compute utterance-level accuracy
                        batch_mcdpow_src_trg[i] = []
                        batch_mcd_src_trg[i] = []
                        batch_lat_dist_srctrg1[i] = []
                        batch_lat_dist_srctrg2[i] = []
                        for j in range(n_batch_utt):
                            spk_src_ = os.path.basename(os.path.dirname(featfile_src[j]))
                            for k in range(n_spk):
                                if spk_src_ == spk_list[k]:
                                    #GV stat of reconst.
                                    gv_src_src[i][k].append(np.var(\
                                        batch_trj_src_src[i][j,\
                                            :flens_src[j],1:].cpu().data.numpy(), axis=0))
                                    #GV stat of cyclic reconst.
                                    gv_src_trg_src[i][k].append(np.var(batch_trj_src_trg_src[i][j,\
                                        :flens_src[j],1:].cpu().data.numpy(), axis=0))
                                    break
                            spk_trg_ = pair_spk_list[i][j] #find the target pair
                            for k in range(n_spk):
                                if spk_trg_ == spk_list[k]:
                                    #GV stat of converted
                                    gv_src_trg[0][k].append(np.var(\
                                        batch_trj_src_trg[i][j,\
                                            :flens_src[j],1:].cpu().data.numpy(), axis=0))
                                    break
                            if file_src_trg_flag[j]: #target pair exists, calc spectral and latent distortion
                                #latent of converted
                                trj_lat_srctrg_ = np.array(torch.index_select(\
                                    trj_lat_srctrg[j],0,spcidx_src_trg[j,\
                                        :flens_spc_src_trg[j]]).cpu().data.numpy(), dtype=np.float64)
                                #latent of source
                                trj_lat_src_ = np.array(torch.index_select(\
                                    batch_mle_lat_src[0][j],0,spcidx_src[j,\
                                        :flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64)

                                # time-warping of source-to-target for root-mean-square error (RMSE)
                                aligned_lat_srctrg1, _, _, _ = dtw.dtw_org_to_trg(\
                                    trj_lat_src_, trj_lat_srctrg_)
                                tmp_batch_lat_dist_srctrg1 = np.mean(np.sqrt(np.mean((\
                                    aligned_lat_srctrg1-trj_lat_srctrg_)**2, axis=0)))
                                # cosine-similarity (Cos-Sim)
                                _, _, tmp_batch_lat_cdist_srctrg1, _ = dtw.dtw_org_to_trg(\
                                    trj_lat_srctrg_, trj_lat_src_, mcd=0)

                                # time-warping of target-to-source for RMSE
                                aligned_lat_srctrg2, _, _, _ = dtw.dtw_org_to_trg(\
                                    trj_lat_srctrg_, trj_lat_src_)
                                tmp_batch_lat_dist_srctrg2 = np.mean(np.sqrt(np.mean((\
                                    aligned_lat_srctrg2-trj_lat_src_)**2, axis=0)))
                                # Cos-sim
                                _, _, tmp_batch_lat_cdist_srctrg2, _ = dtw.dtw_org_to_trg(\
                                    trj_lat_src_, trj_lat_srctrg_, mcd=0)

                                # RMSE
                                tmp_batch_lat_dist_srctrg1 = (tmp_batch_lat_dist_srctrg1 \
                                                                + tmp_batch_lat_dist_srctrg2)/2
                                # Cos-sim
                                tmp_batch_lat_dist_srctrg2 = (tmp_batch_lat_cdist_srctrg1 \
                                                                + tmp_batch_lat_cdist_srctrg2)/2

                                # record latent stats
                                batch_lat_dist_srctrg1[0].append(tmp_batch_lat_dist_srctrg1)
                                batch_lat_dist_srctrg2[0].append(tmp_batch_lat_dist_srctrg2)

                                # MCD with 0th power
                                _, _, tmp_batch_mcdpow_src_trg, _ = dtw.dtw_org_to_trg(\
                                    np.array(torch.index_select(batch_trj_src_trg[i][j],0,\
                                        spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), \
                                        dtype=np.float64), \
                                    np.array(torch.index_select(batch_src_trg[j,:,stdim:],0,\
                                        spcidx_src_trg[j,:flens_spc_src_trg[j]]).cpu().data.numpy(), \
                                        dtype=np.float64))

                                # MCD w/o 0th power, i.e., [:,1:]
                                _, _, tmp_batch_mcd_src_trg, _ = dtw.dtw_org_to_trg(\
                                    np.array(torch.index_select(batch_trj_src_trg[i][j,:,1:],0,\
                                        spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), \
                                        dtype=np.float64), \
                                    np.array(torch.index_select(batch_src_trg[j,:,stdim_:],0,\
                                        spcidx_src_trg[j,:flens_spc_src_trg[j]]).cpu().data.numpy(), \
                                        dtype=np.float64))

                                # record spectral stats
                                batch_mcdpow_src_trg[0].append(tmp_batch_mcdpow_src_trg)
                                batch_mcd_src_trg[0].append(tmp_batch_mcd_src_trg)
                                mcdpow_src_trg[i].append(tmp_batch_mcdpow_src_trg) #for epoch summary
                                mcd_src_trg[i].append(tmp_batch_mcd_src_trg) #for epoch summary
                                lat_dist_srctrg1[0].append(tmp_batch_lat_dist_srctrg1)
                                lat_dist_srctrg2[0].append(tmp_batch_lat_dist_srctrg2)

                                # accuracy logging
                                text_log = "%s %s = %.3f dB %.3f dB , %.3f %.3f" % (
                                    os.path.join(os.path.basename(os.path.dirname(featfile_src[j])), \
                                    os.path.basename(featfile_src[j])), os.path.join(os.path.basename(\
                                    os.path.dirname(featfile_src_trg[j])),os.path.basename(\
                                    featfile_src_trg[j])), tmp_batch_mcdpow_src_trg, tmp_batch_mcd_src_trg, \
                                    tmp_batch_lat_dist_srctrg1, tmp_batch_lat_dist_srctrg2)
                                logging.info("batch cv loss %s " % (text_log))

                        # at least one speaker-pair exists
                        if len(batch_mcdpow_src_trg[0]) > 0:
                            batch_mcdpow_src_trg[0] = np.mean(batch_mcdpow_src_trg[0])
                            batch_mcd_src_trg[0] = np.mean(batch_mcd_src_trg[0])
                            batch_lat_dist_srctrg1[0] = np.mean(batch_lat_dist_srctrg1[0])
                            batch_lat_dist_srctrg2[0] = np.mean(batch_lat_dist_srctrg2[0])

                    # losses for optimization
                    for j in range(n_batch_utt):
                        # valid spectral length for optim.
                        batch_src_optim = batch_src[j,:flens_src[j],stdim:]

                        # mel-cepstral distortion (MCD)-based L1-loss of spectral features
                        _, tmp_batch_loss_mcd_src_src, _ = criterion_mcd(\
                            batch_trj_src_src[i][j,:flens_src[j]], batch_src_optim)
                        _, tmp_batch_loss_mcd_src_trg, _ = criterion_mcd(\
                            batch_trj_src_trg[i][j,:flens_src[j]], batch_src_optim)
                        _, tmp_batch_loss_mcd_src_trg_src, _ = criterion_mcd(\
                            batch_trj_src_trg_src[i][j,:flens_src[j]], batch_src_optim)

                        # cross-entropy (CE) of speaker-posterior
                        tmp_batch_loss_scpost_src = criterion_ce(batch_scpost_src[i][j,:flens_src[j]], \
                            batch_src_src_class_code[j,:flens_src[j]])
                        tmp_batch_loss_scpost_src_cv = criterion_ce(batch_scpost_src_trg[i][j,:flens_src[j]], \
                            batch_src_trg_class_code_list[i][j,:flens_src[j]])

                        # KL-divergence of latent-posterior to the standard Laplacian prior
                        tmp_batch_loss_lat_src = loss_vae_laplace(batch_latpost_src[i][j,:flens_src[j]], \
                            lat_dim=args.lat_dim, clip=True)
                        tmp_batch_loss_lat_src_cv = \
                            loss_vae_laplace(batch_latpost_src_trg[i][j,:flens_src[j]], \
                                lat_dim=args.lat_dim, clip=True)

                        # time-warping function with speech frames to calc true MCD values
                        batch_src_spc_ = np.array(torch.index_select(\
                            batch_src[j,:,stdim:],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), \
                                dtype=np.float64)
                        batch_src_spc__ = np.array(torch.index_select(\
                            batch_src[j,:,stdim_:],0,spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), \
                                dtype=np.float64)

                        # MCD of reconst.
                        tmp_batch_mcdpow_src_src, _ = dtw.calc_mcd(batch_src_spc_, \
                            np.array(torch.index_select(batch_trj_src_src[i][j],0,\
                                spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64))
                        tmp_batch_mcd_src_src, _ = dtw.calc_mcd(batch_src_spc__, \
                            np.array(torch.index_select(batch_trj_src_src[i][j,:,1:],0,\
                                spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64))

                        # MCD of cyclic reconst.
                        tmp_batch_mcdpow_src_trg_src, _ = dtw.calc_mcd(batch_src_spc_, \
                            np.array(torch.index_select(batch_trj_src_trg_src[i][j],0,\
                                spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64))
                        tmp_batch_mcd_src_trg_src, _ = dtw.calc_mcd(batch_src_spc__, \
                            np.array(torch.index_select(batch_trj_src_trg_src[i][j,:,1:],0,\
                                spcidx_src[j,:flens_spc_src[j]]).cpu().data.numpy(), dtype=np.float64))

                        # record loss statistics
                        if j > 0:
                            batch_loss_mcd_src_src[i] = torch.cat((batch_loss_mcd_src_src[i], \
                                tmp_batch_loss_mcd_src_src.unsqueeze(0)))
                            batch_loss_mcd_src_trg[i] = torch.cat((batch_loss_mcd_src_trg[i], \
                                tmp_batch_loss_mcd_src_trg.unsqueeze(0)))
                            batch_loss_mcd_src_trg_src[i] = torch.cat((batch_loss_mcd_src_trg_src[i], \
                                tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)))
                            batch_loss_scpost_src[i] = torch.cat((batch_loss_scpost_src[i], \
                                tmp_batch_loss_scpost_src.unsqueeze(0)))
                            batch_loss_scpost_src_cv[i] = torch.cat((batch_loss_scpost_src_cv[i], \
                                tmp_batch_loss_scpost_src_cv.unsqueeze(0)))
                            batch_loss_lat_src[i] = torch.cat((batch_loss_lat_src[i], \
                                tmp_batch_loss_lat_src.unsqueeze(0)))
                            batch_loss_lat_src_cv[i] = torch.cat((batch_loss_lat_src_cv[i], \
                                tmp_batch_loss_lat_src_cv.unsqueeze(0)))
                        else:
                            batch_loss_mcd_src_src[i] = tmp_batch_loss_mcd_src_src.unsqueeze(0)
                            batch_loss_mcd_src_trg[i] = tmp_batch_loss_mcd_src_trg.unsqueeze(0)
                            batch_loss_mcd_src_trg_src[i] = tmp_batch_loss_mcd_src_trg_src.unsqueeze(0)
                            batch_loss_scpost_src[i] = tmp_batch_loss_scpost_src.unsqueeze(0)
                            batch_loss_scpost_src_cv[i] = tmp_batch_loss_scpost_src_cv.unsqueeze(0)
                            batch_loss_lat_src[i] = tmp_batch_loss_lat_src.unsqueeze(0)
                            batch_loss_lat_src_cv[i] = tmp_batch_loss_lat_src_cv.unsqueeze(0)

                        # record loss statistics
                        mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src) #for epoch summary
                        mcd_src_src[i].append(tmp_batch_mcd_src_src) #for epoch summary
                        mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src) #for epoch summary
                        mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src) #for epoch summary
                        loss_mcd_src_src[i].append(tmp_batch_loss_mcd_src_src.item())
                        loss_mcd_src_trg[i].append(tmp_batch_loss_mcd_src_trg.item())
                        loss_mcd_src_trg_src[i].append(tmp_batch_loss_mcd_src_trg_src.item())
                        loss_scpost_src[i].append(tmp_batch_loss_scpost_src.item())
                        loss_scpost_src_cv[i].append(tmp_batch_loss_scpost_src_cv.item())
                        loss_lat_src[i].append(tmp_batch_loss_lat_src.item())
                        loss_lat_src_cv[i].append(tmp_batch_loss_lat_src_cv.item())
                        batch_mcdpow_src_src[i].append(tmp_batch_mcdpow_src_src)
                        batch_mcd_src_src[i].append(tmp_batch_mcd_src_src)
                        batch_mcdpow_src_trg_src[i].append(tmp_batch_mcdpow_src_trg_src)
                        batch_mcd_src_trg_src[i].append(tmp_batch_mcd_src_trg_src)

                    # accumulate loss for optimization
                    if i > 0: # [2nd, 3rd, ..., Nth] cycle
                        batch_loss += \
                            batch_loss_mcd_src_src[i].sum() + batch_loss_mcd_src_trg_src[i].sum() + \
                            batch_loss_lat_src[i].sum() + batch_loss_lat_src_cv[i].sum() + \
                            batch_loss_scpost_src[i].sum() + batch_loss_scpost_src_cv[i].sum()
                    else: # 1st cycle
                        if not half_cyc:
                            batch_loss = \
                                batch_loss_mcd_src_src[0].sum() + batch_loss_mcd_src_trg_src[0].sum() + \
                                batch_loss_lat_src[0].sum() + batch_loss_lat_src_cv[0].sum() + \
                                batch_loss_scpost_src[0].sum() + batch_loss_scpost_src_cv[0].sum()
                        else:
                            batch_loss = \
                                batch_loss_mcd_src_src[0].sum() + \
                                batch_loss_lat_src[0].sum() + batch_loss_scpost_src[0].sum()

                    # record mean-loss from all utterances in this batch
                    batch_loss_mcd_src_src[i] = torch.mean(batch_loss_mcd_src_src[i])
                    batch_loss_mcd_src_trg_src[i] = torch.mean(batch_loss_mcd_src_trg_src[i])
                    batch_loss_mcd_src_trg[i] = torch.mean(batch_loss_mcd_src_trg[i])
                    batch_loss_scpost_src[i] = torch.mean(batch_loss_scpost_src[i])
                    batch_loss_scpost_src_cv[i] = torch.mean(batch_loss_scpost_src_cv[i])
                    batch_loss_lat_src[i] = torch.mean(batch_loss_lat_src[i])
                    batch_loss_lat_src_cv[i] = torch.mean(batch_loss_lat_src_cv[i])

                # compute gradients and backpropagate
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                loss.append(batch_loss.item())

                # stats logging of current mini-batch
                text_log = "%.3f ;; " % batch_loss.item()
                for i in range(args.n_cyc):
                    text_log += "[%d] %.3f %.3f %.3f ; %.3f %.3f ; %.3f %.3f ; "\
                        "%.3f dB %.3f dB , %.3f dB %.3f dB ;; " % (i+1, \
                            batch_loss_mcd_src_src[i].item(), batch_loss_mcd_src_trg_src[i].item(), \
                            batch_loss_mcd_src_trg[i].item(), batch_loss_scpost_src[i].item(), \
                            batch_loss_scpost_src_cv[i].item(), batch_loss_lat_src[i].item(), \
                            batch_loss_lat_src_cv[i].item(), np.mean(batch_mcdpow_src_src[i]), \
                            np.mean(batch_mcd_src_src[i]), np.mean(batch_mcdpow_src_trg_src[i]), \
                            np.mean(batch_mcd_src_trg_src[i]))
                logging.info("batch loss [%d] = %s  (%.3f sec)" % (c_idx_src+1, text_log, time.time() - start))
                iter_idx += 1
                iter_count += 1
                total.append(time.time() - start)

    # model at maximum epoch [please monitor the development loss to prevent overfitting]
    model_encoder.cpu()
    model_decoder.cpu()
    torch.save({"model_encoder": model_encoder.state_dict(), "model_decoder": model_decoder.state_dict()}, \
                args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
