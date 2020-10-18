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
from torch import nn
import torch.multiprocessing as mp

from utils import find_files
from utils import read_hdf5
from utils import read_txt
from utils import check_hdf5
from utils import write_hdf5
from feature_extract import low_cut_filter, convert_f0, read_wav, analyze_range
from feature_extract import convert_continuos_f0, low_pass_filter
from feature_extract import spc2npow, extfrm

import soundfile as sf
import h5py

from dtw_c import dtw_c as dtw
from gru_vae import GRU_RNN_STOCHASTIC, GRU_RNN

import pysptk as ps
import pyworld as pw

FS = 24000
N_GPUS = 1
SHIFT_MS = 5.0
MCEP_ALPHA = 0.466
FFTL = 2048
IRLEN = 1024
SEED = 1
VERBOSE = 1
LP_CUTOFF = 20


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--waveforms", required=True,
                        type=str, help="list or directory of source eval feat files")
    parser.add_argument("--stats_src", required=True,
                        type=str, help="hdf5 file including source statistics")
    parser.add_argument("--stats_trg", required=True,
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--spk_list", required=True,
                        type=str, help="hdf5 file including source statistics")
    parser.add_argument("--stats_jnt",
                        type=str, help="hdf5 file including target statistics")
    parser.add_argument("--model", required=True,
                        type=str, help="GRU_RNN model file")
    parser.add_argument("--config", required=True,
                        type=str, help="GRU_RNN configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--h5outdir", required=True,
                        type=str, help="directory to save converted features")
    parser.add_argument("--fs", default=FS,
                        type=int, help="sampling rate")
    parser.add_argument("--n_gpus", default=N_GPUS,
                        type=int, help="number of gpus")
    # other setting
    parser.add_argument("--shiftms", default=SHIFT_MS,
                        type=float, help="frame shift")
    parser.add_argument("--mcep_alpha", default=MCEP_ALPHA,
                        type=float, help="mcep alpha coeff.")
    parser.add_argument("--fftl", default=FFTL,
                        type=int, help="FFT length")
    parser.add_argument("--seed", default=SEED,
                        type=int, help="seed number")
    parser.add_argument("--minf0", default=40,
                        type=int, help="seed number")
    parser.add_argument("--maxf0", default=700,
                        type=int, help="seed number")
    parser.add_argument("--pow", default=-25.0,
                        type=float, help="seed number")
    parser.add_argument("--GPU_device", default=None,
                        type=int, help="selection of GPU device")
    parser.add_argument("--GPU_device_str", default=None,
                        type=str, help="selection of GPU device")
    parser.add_argument("--verbose", default=VERBOSE,
                        type=int, help="log level")
    args = parser.parse_args()

    if args.GPU_device_str is not None or args.GPU_device is not None:
        os.environ["CUDA_DEVICE_ORDER"]		= "PCI_BUS_ID"
        if args.GPU_device_str is None:
            os.environ["CUDA_VISIBLE_DEVICES"]	= str(args.GPU_device)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]	= args.GPU_device_str

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.exists(args.h5outdir):
        os.makedirs(args.h5outdir)

    # set log level
    log_file = os.path.join(args.outdir, 'decode.log')
    if os.path.exists(os.path.join(args.outdir, 'decode.log')):
        os.remove(log_file)
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=log_file)
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=log_file)
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=log_file)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load config
    config = torch.load(args.config)

    # get file list
    if os.path.isdir(args.waveforms):
        wav_list = sorted(find_files(args.waveforms, "*.wav"))
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)

    spk_src = os.path.basename(args.stats_src).split('.')[0].split('-')[-1]
    spk_trg = os.path.basename(args.stats_trg).split('.')[0].split('-')[-1]
    spk_list = args.spk_list.split('@')
    n_spk = len(spk_list)
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

    # define f0 statistics source
    f0_range_mean_src = read_hdf5(args.stats_src, "/lf0_range_mean")
    f0_range_std_src = read_hdf5(args.stats_src, "/lf0_range_std")
    logging.info(f0_range_mean_src)
    logging.info(f0_range_std_src)

    # define f0 statistics target
    f0_range_mean_trg = read_hdf5(args.stats_trg, "/lf0_range_mean")
    f0_range_std_trg = read_hdf5(args.stats_trg, "/lf0_range_std")
    logging.info(f0_range_mean_trg)
    logging.info(f0_range_std_trg)

    gv_mean_src = read_hdf5(args.stats_src, "/gv_range_mean")[1:]
    gv_mean_trg = read_hdf5(args.stats_trg, "/gv_range_mean")[1:]

    model_epoch = os.path.basename(args.model).split('.')[0].split('-')[1]
    logging.info('epoch: '+model_epoch)

    # prepare the file list for parallel decoding
    wav_lists = np.array_split(wav_list, args.n_gpus)
    wav_lists = [f_list.tolist() for f_list in wav_lists]

    ### GRU-RNN decoding ###
    def decode_RNN(wav_list, gpu, cvlist=None, cvlist_src=None, \
        mcd_cvlist_src=None, mcdstd_cvlist_src=None, mcdpow_cvlist_src=None, mcdpowstd_cvlist_src=None):
        with torch.cuda.device(gpu):
            mean_trg = torch.FloatTensor(read_hdf5(args.stats_jnt, "/mean_feat_org_lf0")[config.stdim:]).cuda()
            std_trg = torch.FloatTensor(read_hdf5(args.stats_jnt, "/scale_feat_org_lf0")[config.stdim:]).cuda()
            # define model and load parameters
            logging.info(config)
            logging.info("model")
            with torch.no_grad():
                model_encoder = GRU_RNN_STOCHASTIC(
                    in_dim=config.in_dim,
                    out_dim=config.lat_dim,
                    hidden_layers=config.hidden_layers,
                    hidden_units=config.hidden_units,
                    kernel_size=config.kernel_size_enc,
                    dilation_size=config.dilation_size_enc,
                    arparam=config.arparam,
                    spk_dim=n_spk,
                    causal_conv=config.causal_conv,
                    scale_out_flag=False)
                model_decoder = GRU_RNN(
                    in_dim=config.lat_dim+n_spk,
                    out_dim=config.out_dim,
                    hidden_layers=config.hidden_layers,
                    hidden_units=config.hidden_units,
                    kernel_size=config.kernel_size_dec,
                    dilation_size=config.dilation_size_dec,
                    causal_conv=config.causal_conv,
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
                if config.arparam:
                    init_pp = np.zeros((1,1,config.lat_dim*2+n_spk))
                else:
                    init_pp = np.zeros((1,1,config.lat_dim+n_spk))
                y_in_pp = torch.FloatTensor(init_pp).cuda()
                y_in_src = y_in_trg = torch.unsqueeze(torch.unsqueeze((0-mean_trg)/std_trg,0),0)
            fs = args.fs
            fft_size = args.fftl
            mcep_dim = model_decoder.out_dim-1
            for wav_file in wav_list:
                # convert mcep
                feat_file = os.path.join(args.h5outdir, os.path.basename(wav_file).replace(".wav", ".h5"))
                logging.info("cvmcep " + feat_file + " " + wav_file)

                fs, x = read_wav(wav_file, cutoff=70)

                time_axis, f0, sp, ap = analyze_range(x, fs=fs, minf0=args.minf0, maxf0=args.maxf0, \
                                                        fperiod=args.shiftms, fftl=args.fftl)
                logging.info(sp.shape)

                mcep = ps.sp2mc(sp, mcep_dim, args.mcep_alpha)
                logging.info(mcep.shape)
                codeap = pw.code_aperiodicity(ap, fs)
                logging.info(codeap.shape)

                npow = spc2npow(sp)
                logging.info(npow.shape)
                _, spcidx = extfrm(mcep, npow, power_threshold=args.pow)
                spcidx = spcidx[0]
                logging.info(spcidx.shape)

                uv, contf0 = convert_continuos_f0(np.array(f0))
                uv = np.expand_dims(uv, axis=-1)
                logging.info(uv.shape)
                cont_f0_lpf = low_pass_filter(contf0, int(1.0 / (args.shiftms * 0.001)), cutoff=LP_CUTOFF)
                logcontf0 = np.expand_dims(np.log(cont_f0_lpf), axis=-1)
                logging.info(logcontf0.shape)
                feat = np.c_[uv,logcontf0,codeap,mcep]
                logging.info(feat.shape)

                logging.info("generate")
                with torch.no_grad():
                    lat_feat_src, _, _, _, _ = \
                        model_encoder(torch.FloatTensor(feat).cuda(), y_in_pp, sampling=False)

                    src_code = np.zeros((lat_feat_src.shape[0],n_spk))
                    src_code[:,src_code_idx] = 1
                    src_code = torch.FloatTensor(src_code).cuda()

                    trg_code = np.zeros((lat_feat_src.shape[0],n_spk))
                    trg_code[:,trg_code_idx] = 1
                    trg_code = torch.FloatTensor(trg_code).cuda()

                    cvmcep_src, _, _ = model_decoder(torch.cat((src_code, lat_feat_src),1), y_in_src)
                    cvmcep_src = np.array(cvmcep_src.cpu().data.numpy(), dtype=np.float64)

                    cvmcep, _, _ = model_decoder(torch.cat((trg_code, lat_feat_src),1), y_in_trg)
                    cvmcep = np.array(cvmcep.cpu().data.numpy(), dtype=np.float64)

                logging.info(lat_feat_src.shape)
                logging.info(cvmcep_src.shape)
                logging.info(cvmcep.shape)

                cvf0 = convert_f0(f0, f0_range_mean_src, f0_range_std_src, f0_range_mean_trg, f0_range_std_trg)
                uv_cv, contf0_cv = convert_continuos_f0(np.array(cvf0))
                uv_cv = np.expand_dims(uv_cv, axis=-1)
                logging.info(uv_cv.shape)
                cont_f0_lpf_cv = low_pass_filter(contf0_cv, int(1.0 / (args.shiftms * 0.001)), cutoff=LP_CUTOFF)
                logcontf0_cv = np.expand_dims(np.log(cont_f0_lpf_cv), axis=-1)
                logging.info(logcontf0_cv.shape)
                feat_cv = np.c_[uv_cv,logcontf0_cv,codeap]
                logging.info(feat_cv.shape)

                feat_cvmcep = np.c_[feat_cv, cvmcep]
                logging.info(feat_cvmcep.shape)
                write_path = '/feat_cvmcep_cycvae-'+model_epoch
                logging.info(feat_file + ' ' + write_path)
                write_hdf5(feat_file, write_path, feat_cvmcep)
                cvlist.append(np.var(cvmcep[:,1:], axis=0))

                _, mcdpow_arr = dtw.calc_mcd(np.array(mcep[np.array(spcidx),:], dtype=np.float64), \
                                                np.array(cvmcep_src[np.array(spcidx),:], dtype=np.float64))
                _, mcd_arr = dtw.calc_mcd(np.array(mcep[np.array(spcidx),1:], dtype=np.float64), \
                                            np.array(cvmcep_src[np.array(spcidx),1:], dtype=np.float64))
                mcdpow_mean = np.mean(mcdpow_arr)
                mcdpow_std = np.std(mcdpow_arr)
                mcd_mean = np.mean(mcd_arr)
                mcd_std = np.std(mcd_arr)
                logging.info("mcdpow_src_cv: %.6f dB +- %.6f" % (mcdpow_mean, mcdpow_std))
                logging.info("mcd_src_cv: %.6f dB +- %.6f" % (mcd_mean, mcd_std))
                mcdpow_cvlist_src.append(mcdpow_mean)
                mcdpowstd_cvlist_src.append(mcdpow_std)
                mcd_cvlist_src.append(mcd_mean)
                mcdstd_cvlist_src.append(mcd_std)
                cvlist_src.append(np.var(cvmcep_src[:,1:], axis=0))

                logging.info("synth voco")
                cvsp = ps.mc2sp(cvmcep, args.mcep_alpha, fft_size)
                logging.info(cvsp.shape)
                wav = np.clip(pw.synthesize(cvf0, cvsp, ap, fs, frame_period=args.shiftms), -1, 1)
                wavpath = os.path.join(args.outdir, os.path.basename(wav_file).replace(".wav", "_cv.wav"))
                sf.write(wavpath, wav, fs, 'PCM_16')
                logging.info(wavpath)

                logging.info("synth anasyn")
                wav = np.clip(pw.synthesize(f0, sp, ap, fs, frame_period=args.shiftms), -1, 1)
                wavpath = os.path.join(args.outdir,os.path.basename(wav_file).replace(".wav","_anasyn.wav"))
                sf.write(wavpath, wav, fs, 'PCM_16')
                logging.info(wavpath)


    with mp.Manager() as manager:
        logging.info("GRU-RNN decoding")
        processes = []
        cvlist = manager.list()
        cvlist_src = manager.list()
        mcd_cvlist_src = manager.list()
        mcdstd_cvlist_src = manager.list()
        mcdpow_cvlist_src = manager.list()
        mcdpowstd_cvlist_src = manager.list()
        gpu = 0
        for i, wav_list in enumerate(wav_lists):
            logging.info(i)
            p = mp.Process(target=decode_RNN, args=(wav_list, gpu, cvlist, \
                    cvlist_src, mcd_cvlist_src, mcdstd_cvlist_src, mcdpow_cvlist_src, mcdpowstd_cvlist_src,))
            p.start()
            processes.append(p)
            gpu += 1
            if (i + 1) % args.n_gpus == 0:
                gpu = 0

        # wait for all process
        for p in processes:
            p.join()

        # summarize acc. stats.
        cvgv_ev_mean = np.mean(np.array(cvlist), axis=0)
        cvgv_ev_var = np.var(np.array(cvlist), axis=0)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg)))), \
                    np.std(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_trg))))))
        logging.info("-------")
        logging.info("mcdpow_src_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (\
                    np.mean(np.array(mcdpow_cvlist_src)),np.std(np.array(mcdpow_cvlist_src)),\
                        np.mean(np.array(mcdpowstd_cvlist_src)),np.std(np.array(mcdpowstd_cvlist_src))))
        logging.info("mcd_src_cv: %.6f dB (+- %.6f) +- %.6f (+- %.6f)" % (np.mean(np.array(mcd_cvlist_src)),\
                    np.std(np.array(mcd_cvlist_src)),np.mean(np.array(mcdstd_cvlist_src)),\
                        np.std(np.array(mcdstd_cvlist_src))))
        cvgv_ev_mean = np.mean(np.array(cvlist_src), axis=0)
        cvgv_ev_var = np.var(np.array(cvlist_src), axis=0)
        logging.info("%lf +- %lf" % (np.mean(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_src)))), \
                    np.std(np.sqrt(np.square(np.log(cvgv_ev_mean)-np.log(gv_mean_src))))))


if __name__ == "__main__":
    main()
