#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import sys
from distutils.util import strtobool

import logging
import numpy as np
from numpy.matlib import repmat
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter

from utils import find_files
from utils import read_txt
from utils import write_hdf5, read_hdf5

from multiprocessing import Array

import pysptk as ps
import pyworld as pw

#np.set_printoptions(threshold=np.inf)

FS = 24000
SHIFTMS = 5
MINF0 = 40
MAXF0 = 700
MCEP_DIM = 49
MCEP_ALPHA = 0.466 #24k
FFTL = 2048
IRLEN = 1024
LOWPASS_CUTOFF = 20
HIGHPASS_CUTOFF = 70
OVERWRITE = True


def low_cut_filter(x, fs, cutoff=HIGHPASS_CUTOFF):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def analyze(wav, fs=FS, minf0=MINF0, maxf0=MAXF0, fperiod=SHIFTMS, fftl=FFTL, f0=None, time_axis=None):
    if f0 is None or time_axis is None:
        _f0, time_axis = pw.harvest(wav, fs, f0_floor=60.0, frame_period=fperiod)
        f0 = pw.stonemask(wav, _f0, time_axis, fs) 
    sp = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(wav, f0, time_axis, fs, fft_size=fftl)

    return time_axis, f0, sp, ap


def analyze_range(wav, fs=FS, minf0=MINF0, maxf0=MAXF0, fperiod=SHIFTMS, fftl=FFTL, f0=None, time_axis=None):
    if f0 is None or time_axis is None:
        _f0, time_axis = pw.harvest(wav, fs, f0_floor=minf0, f0_ceil=maxf0, frame_period=fperiod)
        f0 = pw.stonemask(wav, _f0, time_axis, fs) 
    sp = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(wav, f0, time_axis, fs, fft_size=fftl)

    return time_axis, f0, sp, ap


def read_wav(wav_file, cutoff=HIGHPASS_CUTOFF):
    fs, x = wavfile.read(wav_file)
    x = np.array(x, dtype=np.float64)
    if cutoff != 0:
        x = low_cut_filter(x, fs, cutoff)

    return fs, x


def convert_f0(f0, f0_mean_src, f0_std_src, f0_mean_trg, f0_std_trg):
    nonzero_indices = f0 > 0
    cvf0 = np.zeros(len(f0))
    cvf0[nonzero_indices] = \
            np.exp((f0_std_trg/f0_std_src)*(np.log(f0[nonzero_indices])-f0_mean_src)+f0_mean_trg)

    return cvf0


def mod_pow(cvmcep, mcep, alpha=MCEP_ALPHA, irlen=IRLEN):
    cv_e = ps.mc2e(cvmcep, alpha=alpha, irlen=irlen)
    r_e = ps.mc2e(mcep, alpha=alpha, irlen=irlen)
    dpow = np.log(r_e/cv_e) / 2
    mod_cvmcep = np.copy(cvmcep)
    mod_cvmcep[:,0] += dpow

    return mod_cvmcep


def extfrm(data, npow, power_threshold=-20):
    T = data.shape[0]
    if T != len(npow):
        raise("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata, valid_index


def spc2npow(spectrogram):
    npow = np.apply_along_axis(spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow/meanpow)

    return npow


def spvec2pow(specvec):
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power


def low_pass_filter(x, fs, cutoff=LOWPASS_CUTOFF, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument("--expdir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument(
        "--waveforms", default=None,
        help="directory or list of filename of input wavfile")
    parser.add_argument(
        "--hdf5dir", default=None,
        help="directory to save hdf5")
    parser.add_argument(
        "--wavdir", default=None,
        help="directory to save of preprocessed wav file")
    parser.add_argument(
        "--wavfiltdir", default=None,
        help="directory to save of preprocessed wav file")
    parser.add_argument(
        "--fs", default=FS,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--shiftms", default=SHIFTMS,
        type=int, help="Frame shift in msec")
    parser.add_argument(
        "--minf0", default=MINF0,
        type=int, help="minimum f0")
    parser.add_argument(
        "--maxf0", default=MAXF0,
        type=int, help="maximum f0")
    parser.add_argument(
        "--mcep_dim", default=MCEP_DIM,
        type=int, help="Dimension of mel cepstrum")
    parser.add_argument(
        "--mcep_alpha", default=MCEP_ALPHA,
        type=float, help="Alpha of mel cepstrum")
    parser.add_argument(
        "--pow", default=-20,
        type=float, help="Power threshold")
    parser.add_argument(
        "--fftl", default=FFTL,
        type=int, help="FFT length")
    parser.add_argument("--init", default=False,
        type=strtobool, help="flag for computing stats of dtw-ed feature")
    parser.add_argument(
        "--highpass_cutoff", default=HIGHPASS_CUTOFF,
        type=int, help="Cut off frequency in lowpass filter")
    parser.add_argument(
        "--n_jobs", default=10,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/feature_extract.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/feature_extract.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/feature_extract.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)

    # check directory existence
    if (args.wavdir is not None) and (not os.path.exists(args.wavdir)):
        os.makedirs(args.wavdir)
    if (args.wavfiltdir is not None) and (not os.path.exists(args.wavfiltdir)):
        os.makedirs(args.wavfiltdir)
    if not os.path.exists(args.hdf5dir):
        os.makedirs(args.hdf5dir)

    def feature_extract(wav_list, arr, max_frame_list):
        n_wav = len(wav_list)
        n_sample = 0
        n_frame = 0
        max_frame = 0
        count = 1
        for wav_name in wav_list:
            # load wavfile and apply low cut filter
            fs, x = read_wav(wav_name, cutoff=args.highpass_cutoff)
            n_sample += x.shape[0]
            logging.info(wav_name+" "+str(x.shape[0])+" "+str(n_sample)+" "+str(count))

            # check sampling frequency
            if not fs == args.fs:
                logging.info("ERROR: sampling frequency is not matched.")
                sys.exit(1)

            hdf5name = args.hdf5dir + "/" + os.path.basename(wav_name).replace(".wav", ".h5")

            if not args.init:
                time_axis_range, f0_range, spc_range, ap_range = analyze_range(x, fs=fs, minf0=args.minf0, \
                                                    maxf0=args.maxf0, fperiod=args.shiftms, fftl=args.fftl)
                write_hdf5(hdf5name, "/f0_range", f0_range)

                uv_range, cont_f0_range = convert_continuos_f0(np.array(f0_range))
                cont_f0_lpf_range = \
                    low_pass_filter(cont_f0_range, int(1.0 / (args.shiftms * 0.001)), cutoff=20)

                codeap_range = pw.code_aperiodicity(ap_range, fs)
                mcep_range = ps.sp2mc(spc_range, args.mcep_dim, args.mcep_alpha)

                npow_range = spc2npow(spc_range)

                mcepspc_range, spcidx_range = extfrm(mcep_range, npow_range, power_threshold=args.pow)
                logging.info(wav_name+" "+str(mcepspc_range.shape[0])+" "+str(mcepspc_range.shape[1])+\
                                " "+str(count))

                cont_f0_lpf_range = np.expand_dims(cont_f0_lpf_range, axis=-1)
                uv_range = np.expand_dims(uv_range, axis=-1)
                if codeap_range.ndim == 1:
                    codeap_range = np.expand_dims(codeap_range, axis=-1)

                feat_org_lf0 = np.c_[uv_range,np.log(cont_f0_lpf_range),codeap_range,mcep_range]
                write_hdf5(hdf5name, "/feat_org_lf0", feat_org_lf0)

                write_hdf5(hdf5name, "/npow_range", npow_range)
                write_hdf5(hdf5name, "/spcidx_range", spcidx_range)

                n_frame += feat_org_lf0.shape[0]
                if max_frame < feat_org_lf0.shape[0]:
                    max_frame = feat_org_lf0.shape[0]
                if args.highpass_cutoff != 0 and args.wavfiltdir is not None:
                    wavfile.write(args.wavfiltdir + "/" + os.path.basename(wav_name), fs, np.int16(x))
                wavpath = args.wavdir + "/" + os.path.basename(wav_name)
                logging.info(wavpath)
                sp_rec = ps.mc2sp(mcep_range, args.mcep_alpha, args.fftl)
                wav = np.clip(pw.synthesize(f0_range, sp_rec, ap_range, fs, frame_period=args.shiftms), \
                               -32768, 32767)
                wavfile.write(wavpath, fs, np.int16(wav))
            else:
                time_axis, f0, spc, ap = analyze(x, fs=fs, fperiod=args.shiftms, fftl=args.fftl)
                write_hdf5(hdf5name, "/f0", f0)
                npow = spc2npow(spc)
                write_hdf5(hdf5name, "/npow", npow)
                n_frame += f0.shape[0]
                if max_frame < f0.shape[0]:
                    max_frame = f0.shape[0]

            count += 1
        arr[0] += n_wav
        arr[1] += n_sample
        arr[2] += n_frame
        max_frame_list.append(max_frame)
        if (n_wav > 0):
            logging.info(str(arr[0])+" "+str(n_wav)+" "+str(arr[1])+" "+str(n_sample/n_wav)+" "+\
                    str(arr[2])+" "+str(n_frame/n_wav)+" max_frame = "+str(max_frame))

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        processes = []
        arr = mp.Array('d', 3)
        max_frame_list = manager.list()
        for f in file_lists:
            p = mp.Process(target=feature_extract, args=(f,arr,max_frame_list))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        logging.info(str(arr[0])+" "+str(arr[1])+" "+str(arr[1]/arr[0])+" "+str(arr[2])+" "+str(arr[2]/arr[0]))
        logging.info('max_frame: %ld' % (np.max(max_frame_list)))


if __name__ == "__main__":
    main()
