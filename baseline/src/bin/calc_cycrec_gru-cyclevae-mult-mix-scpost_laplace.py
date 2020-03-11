#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

from feature_extract import mod_pow, convert_f0, convert_continuos_f0, low_pass_filter
from utils import find_files, read_hdf5, read_txt, write_hdf5, check_hdf5

from dtw_c import dtw_c as dtw
from gru_vae import GRU_RNN, GRU_RNN_STOCHASTIC

MCEP_ALPHA = 0.466
IRLEN = 1024
SHIFTMS = 5.0
LP_CUTOFF = 20


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--spk_list", required=True,
                        type=str, help="hdf5 file including source statistics")
    parser.add_argument("--stats_src", required=True,
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--stats_trg", required=True,
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--stats_jnt",
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--model", required=True,
                        type=str, help="model file")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    # other setting
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--mcep_alpha", default=MCEP_ALPHA,
                        type=float, help="mcep alpha coeff.")
    parser.add_argument("--shiftms", default=SHIFTMS,
                        type=float, help="mcep alpha coeff.")
    parser.add_argument("--GPU_device", default=None,
                        type=int, help="selection of GPU device")
    parser.add_argument("--GPU_device_str", default=None,
                        type=str, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")

    args = parser.parse_args()

    if args.GPU_device is not None or args.GPU_device_str is not None:
        os.environ["CUDA_DEVICE_ORDER"]		= "PCI_BUS_ID"
        if args.GPU_device_str is None:
            os.environ["CUDA_VISIBLE_DEVICES"]	= str(args.GPU_device)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]	= args.GPU_device_str

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load config
    config = torch.load(args.config)

    # get source feat list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]
    spk_src = os.path.basename(args.stats_src).split('.')[0].split('-')[-1]
    spk_trg = os.path.basename(args.stats_trg).split('.')[0].split('-')[-1]
    logging.info(spk_trg)
    spk_list = args.spk_list.split('@')
    n_spk = len(spk_list)
    logging.info(n_spk)
    src_flag = False
    trg_flag = False
    for i in range(n_spk):
        if spk_src == spk_list[i]:
            src_code_idx = i
            src_flag = True
        if spk_trg == spk_list[i]:
            trg_code_idx = i
            trg_flag = True
        if src_flag and trg_flag:
            break

    gv_mean_trg = read_hdf5(args.stats_trg, "/gv_range_mean")[1:]

    f0_range_mean_trg = read_hdf5(args.stats_trg, "/lf0_range_mean")
    f0_range_std_trg = read_hdf5(args.stats_trg, "/lf0_range_std")
    logging.info('{} +- {}'.format(f0_range_mean_trg, f0_range_std_trg))
    f0_range_mean_src = read_hdf5(args.stats_src, "/lf0_range_mean")
    f0_range_std_src = read_hdf5(args.stats_src, "/lf0_range_std")
    logging.info('{} +- {}'.format(f0_range_mean_src, f0_range_std_src))

    model_epoch = os.path.basename(args.model).split('.')[0].split('-')[1]
    logging.info('epoch: '+model_epoch)

    # define gpu decode function
    def gpu_decode(feat_list, gpu, cvlist=None, mcdpow_cvlist=None, mcdpowstd_cvlist=None, mcd_cvlist=None, \
                    mcdstd_cvlist=None, cvlist_dv=None, mcdpow_cvlist_dv=None, mcdpowstd_cvlist_dv=None, \
                    mcd_cvlist_dv=None, mcdstd_cvlist_dv=None):
        with torch.cuda.device(gpu):
            mean_jnt = torch.FloatTensor(read_hdf5(args.stats_jnt, "/mean_feat_org_lf0")[config.stdim:]).cuda()
            std_jnt = torch.FloatTensor(read_hdf5(args.stats_jnt, "/scale_feat_org_lf0")[config.stdim:]).cuda()
            # define model and load parameters
            logging.info("model")
            logging.info(config)
            with torch.no_grad():
                model_encoder = GRU_RNN_STOCHASTIC(
                    in_dim=config.in_dim,
                    out_dim=config.lat_dim,
                    hidden_layers=config.hidden_layers,
                    hidden_units=config.hidden_units,
                    kernel_size=config.kernel_size,
                    dilation_size=config.dilation_size,
                    spk_dim=n_spk,
                    scale_out_flag=False)
                model_decoder = GRU_RNN(
                    in_dim=config.lat_dim+n_spk,
                    out_dim=config.out_dim,
                    hidden_layers=config.hidden_layers,
                    hidden_units=config.hidden_units,
                    kernel_size=config.kernel_size,
                    dilation_size=config.dilation_size,
                    scale_in_flag=False)
                logging.info(model_encoder)
                logging.info(model_decoder)
                model_encoder.load_state_dict(torch.load(args.model)["model_encoder"])
                model_decoder.load_state_dict(torch.load(args.model)["model_decoder"])
                model_encoder.cuda()
                model_decoder.cuda()
                model_encoder.eval()
                model_decoder.eval()
                for param in model_encoder.parameters():
                    param.requires_grad = False
                for param in model_decoder.parameters():
                    param.requires_grad = False
                init_pp = np.zeros((1,1,config.lat_dim*2+n_spk))
                y_in_pp = torch.FloatTensor(init_pp).cuda()
                y_in_src = y_in_trg = torch.unsqueeze(torch.unsqueeze((0-mean_jnt)/std_jnt,0),0)
            for feat_file in feat_list:
                # convert mcep
                logging.info("recmcep " + feat_file)

                feat = read_hdf5(feat_file, "/feat_org_lf0")
                logging.info(feat.shape)
                f0 = read_hdf5(feat_file, "/f0_range")
                cvf0 = convert_f0(f0, f0_range_mean_trg, f0_range_std_trg, f0_range_mean_src, f0_range_std_src)
                cvuv, cont_f0 = convert_continuos_f0(cvf0)
                cvuv = np.expand_dims(cvuv, axis=-1)
                cont_f0_lpf = low_pass_filter(cont_f0, int(1.0 / (args.shiftms * 0.001)), cutoff=LP_CUTOFF)
                if np.min(cont_f0_lpf) <= 0:
                    length = len(cont_f0_lpf)
                    for i in range(length):
                        if cont_f0_lpf[i] <= 0:
                            if i > 0 and i < length-1:
                                for j in range(i-1,-1,-1):
                                    if cont_f0_lpf[j] > 0:
                                        left_val = cont_f0_lpf[j]
                                        break
                                for j in range(i+1,length):
                                    if cont_f0_lpf[j] > 0:
                                        right_val = cont_f0_lpf[j]
                                        break
                                cont_f0_lpf[i] = (left_val+right_val)/2
                            elif i == 0:
                                for j in range(1,length):
                                    if cont_f0_lpf[j] > 0:
                                        right_val = cont_f0_lpf[j]
                                        break
                                cont_f0_lpf[i] = right_val
                            else:
                                for j in range(i-1,-1,-1):
                                    if cont_f0_lpf[j] > 0:
                                        left_val = cont_f0_lpf[j]
                                        break
                                cont_f0_lpf[i] = left_val
                cvlogf0fil = np.expand_dims(np.log(cont_f0_lpf), axis=-1)
                feat_cv = np.c_[cvuv,cvlogf0fil,feat[:,2:config.stdim]]
                with torch.no_grad():
                    lat_feat, _, _, _, _ = model_encoder(torch.FloatTensor(feat).cuda(), \
                                                        y_in_pp, sampling=False)
                    src_code = np.zeros((lat_feat.shape[0],n_spk))
                    src_code[:,src_code_idx] = 1
                    src_code = torch.FloatTensor(src_code).cuda()
                    cvmcep, _, _ = model_decoder(torch.cat((src_code, lat_feat),1), y_in_src)
                    lat_feat, _, _, _, _ = model_encoder(torch.cat((torch.FloatTensor(feat_cv).cuda(), cvmcep),1), \
                                                        y_in_pp, sampling=False)
                    trg_code = np.zeros((lat_feat.shape[0],n_spk))
                    trg_code[:,trg_code_idx] = 1
                    trg_code = torch.FloatTensor(trg_code).cuda()
                    cvmcep, _, _ = model_decoder(torch.cat((trg_code, lat_feat),1), y_in_trg)

                cvmcep = np.array(cvmcep.cpu().data.numpy(), dtype=np.float64)
                logging.info(cvmcep.shape)

                mcep = feat[:,config.stdim:]
                spcidx = read_hdf5(feat_file, "/spcidx_range")[0]
                _, mcdpow_arr = dtw.calc_mcd(np.array(mcep[np.array(spcidx),:], dtype=np.float64), \
                                            np.array(cvmcep[np.array(spcidx),:], dtype=np.float64))
                _, mcd_arr = dtw.calc_mcd(np.array(mcep[np.array(spcidx),1:], dtype=np.float64), \
                                            np.array(cvmcep[np.array(spcidx),1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                dataset = feat_file.split('/')[1].split('_')[0]
                if 'tr' in dataset:
                    logging.info('trn')
                    mcdpow_cvlist.append(mcdpow_mean)
                    mcdpowstd_cvlist.append(mcdpow_std)
                    mcd_cvlist.append(mcd_mean)
                    mcdstd_cvlist.append(mcd_std)
                    cvlist.append(np.var(cvmcep[:,1:], axis=0))
                    logging.info(len(cvlist))
                elif 'dv' in dataset:
                    logging.info('dev')
                    mcdpow_cvlist_dv.append(mcdpow_mean)
                    mcdpowstd_cvlist_dv.append(mcdpow_std)
                    mcd_cvlist_dv.append(mcd_mean)
                    mcdstd_cvlist_dv.append(mcd_std)
                    cvlist_dv.append(np.var(cvmcep[:,1:], axis=0))
                    logging.info(len(cvlist_dv))
                logging.info("mcdpow_rec: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_rec: %.6f dB +- %.6f" % (mcd_mean, mcd_std))

                logging.info("mod_pow")
                cvmcep = mod_pow(cvmcep, mcep, alpha=args.mcep_alpha, irlen=IRLEN)
                logging.info(cvmcep.shape)
                feat_cvmcep = np.c_[feat[:,:config.stdim], cvmcep]
                logging.info(feat_cvmcep.shape)
                write_path = '/feat_recmcep_cycvae-'+model_epoch
                outh5dir = os.path.join(os.path.dirname(os.path.dirname(feat_file)), spk_trg+"-"+spk_src+"-"+spk_trg)
                if not os.path.exists(outh5dir):
                    os.makedirs(outh5dir)
                feat_file_cyc = os.path.join(outh5dir, os.path.basename(feat_file))
                logging.info(feat_file_cyc + ' ' + write_path)
                write_hdf5(feat_file_cyc, write_path, feat_cvmcep)


    # parallel decode training
    with mp.Manager() as manager:
        gpu = 0
        processes = []
        cvlist = manager.list()
        mcd_cvlist = manager.list()
        mcdstd_cvlist = manager.list()
        mcdpow_cvlist = manager.list()
        mcdpowstd_cvlist = manager.list()
        cvlist_dv = manager.list()
        mcd_cvlist_dv = manager.list()
        mcdstd_cvlist_dv = manager.list()
        mcdpow_cvlist_dv = manager.list()
        mcdpowstd_cvlist_dv = manager.list()
        for i, feat_list in enumerate(feat_lists):
            logging.info(i)
            p = mp.Process(target=gpu_decode, args=(feat_list, gpu, cvlist, mcdpow_cvlist, mcdpowstd_cvlist, \
                                                    mcd_cvlist, mcdstd_cvlist, cvlist_dv, mcdpow_cvlist_dv, \
                                                    mcdpowstd_cvlist_dv, mcd_cvlist_dv, mcdstd_cvlist_dv,))
            p.start()
            processes.append(p)
            gpu += 1
            if (i + 1) % args.n_gpus == 0:
                gpu = 0
        # wait for all process
        for p in processes:
            p.join()

        # calculate cv_gv statistics
        logging.info("mcdpow_rec: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist)), \
                    np.std(np.array(mcdpow_cvlist)),np.mean(np.array(mcdpowstd_cvlist)),\
                    np.std(np.array(mcdpowstd_cvlist))))
        logging.info("mcd_rec: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist)), \
                    np.std(np.array(mcd_cvlist)),np.mean(np.array(mcdstd_cvlist)),\
                    np.std(np.array(mcdstd_cvlist))))
        cvgv_mean = np.mean(np.array(cvlist), axis=0)
        cvgv_var = np.var(np.array(cvlist), axis=0)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg)))), \
                                    np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg))))))
        logging.info("mcdpow_rec_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcdpow_cvlist_dv)), \
                    np.std(np.array(mcdpow_cvlist_dv)),np.mean(np.array(mcdpowstd_cvlist_dv)),\
                    np.std(np.array(mcdpowstd_cvlist_dv))))
        logging.info("mcd_rec_dv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_dv)), \
                    np.std(np.array(mcd_cvlist_dv)),np.mean(np.array(mcdstd_cvlist_dv)),\
                    np.std(np.array(mcdstd_cvlist_dv))))
        cvgv_mean = np.mean(np.array(cvlist_dv), axis=0)
        cvgv_var = np.var(np.array(cvlist_dv), axis=0)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg)))), \
                                    np.std(np.sqrt(np.square(np.log(cvgv_mean)-np.log(gv_mean_trg))))))


if __name__ == "__main__":
    main()
