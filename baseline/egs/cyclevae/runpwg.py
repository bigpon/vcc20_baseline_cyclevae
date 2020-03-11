#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Wu Yi-Chiao (Nagoya University)
# based on a ParallelWaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Pytorch PWG script

Usage: runpwg.py [-h] 
                [-g GPUID]  [-c CONFIG]
                [-M MODE] [-R RESUME] [-I ITER]
                [-s SRCSPK] [-t TARSPK]
                [-1] [-2] [-3]
              
Options:
    -h, --help       Show the help
    -g GPUID         GPU device ID
    -c CONFIG        Name of config version
    -M MODE          Training mode, choices=["natural", "mixed"]
    -R RESUME        Number of iteration to resume model
    -I ITER          Number of iteration of testing model 
    -s SRCSPK        Source speaker
    -t TARSPK        Target speaker
    -1, --step1      Execute step1 (PWG training)
    -2, --step2      Execute step2 (PWG decodeing w/ acoustic features)
    -3, --step3      Execute step3 (PWG decodeing w/ speaker voice converted acoustic features)
    
"""
import os
import sys
from docopt import docopt

# PATH INITIALIZATION
def _path_initial(pathlist):
    for pathdir in pathlist:
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)

# PATH CHECK
def _path_check(pathlist):
    for pathdir in pathlist:
        if not os.path.exists(pathdir):
            raise FileNotFoundError("%s doesn't exist!!" % pathdir)

# PATH & PARAMETER SETTINGS
LIBRARY_DIR = "/usr/local/cuda-10.0/lib64"
CUDA_DIR    = "/usr/local/cuda-10.0"
PRJ_ROOT    = "../.."
SEED        = 1
DECODE_SEED = 100

# MAIN
if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    # step control
    execute_steps = [False] \
        + [args["--step{}".format(step_index)] for step_index in range(1, 4)]
    if not any(execute_steps):
        raise("Please specify steps with options")
    # environment setting
    os.environ['LD_LIBRARY_PATH'] += ":" + LIBRARY_DIR
    os.environ['CUDA_HOME'] = CUDA_DIR
    os.environ["CUDA_DEVICE_ORDER"]	= "PCI_BUS_ID"
    if args['-g'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['-g']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # path setting
    network         = "parallel_wavegan"
    source_root     = "%s/src/%s" % (PRJ_ROOT, network)
    source_train    = "%s/bin/train.py"    % (source_root)
    source_decode   = "%s/bin/decode.py"   % (source_root)
    tr_version      = "tr50_vcc2020_24kHz" # training
    dv_version      = "dv50_vcc2020_24kHz" # development
    model_version   = "%s_%s" % (network, tr_version) # model name    
    config_version  = "default" # config name
    model_iters     = "400000" # iteration of testing model
    train_mode      = "mixed"
    if args['-c'] is not None:
        config_version = args['-c']
    if args['-I'] is not None:
        model_iters = args['-I']
    if args['-M'] is not None:
        train_mode = args['-M']    
    stats           = "data/%s/stats_jnt.h5" % (tr_version)
    config          = "conf_pwg/vcc2020.%s.yaml" % (config_version)
    outdir          = "exp/%s_%s_%s" % (model_version, config_version, train_mode) 
    trainaux_path   = "data/%s" % (tr_version)
    trainwav_path   = "data/%s" % (tr_version)
    validaux_path   = "data/%s" % (dv_version)
    validwav_path   = "data/%s" % (dv_version)    
    _path_check([stats, config])
    # hdf5 path(channel) setting
    natural_path    = "feat_org_lf0"
    artificial_path = "feat_recmcep_cycvae-78"
    converted_path  = "feat_cvmcep_cycvae-78"

    # NETWORK TRAINING
    if execute_steps[1]:
        # training & validation paired list
        if train_mode == "natural":
            trainaux_feats = "%s/feats.scp" % trainaux_path
            trainwaveforms = "%s/wav_filtered.scp" % trainwav_path
            validaux_feats = "%s/feats.scp" % validaux_path
            validwaveforms = "%s/wav_filtered.scp" % validwav_path
        elif train_mode == "mixed":
            trainaux_feats = "%s/feats_ftcyc.scp" % trainaux_path
            trainwaveforms = "%s/wav_filtered_ftcyc.scp" % trainwav_path
            validaux_feats = "%s/feats_ftcyc.scp" % validaux_path
            validwaveforms = "%s/wav_filtered_ftcyc.scp" % validwav_path
        else:
            print("Training mode %s is not supported!!" % train_mode)
            sys.exit(0)
        # resume setting
        if args['-R'] is not None:
            resume = outdir + "/checkpoint-%ssteps.pkl" % (args['-R'])
            _path_check([resume])
        else:
            resume = "None"
        # training
        cmd = "python "                + source_train + \
            " --audio_train "          + trainwaveforms + \
            " --world_train "          + trainaux_feats + \
            " --audio_dev "            + validwaveforms + \
            " --world_dev "            + validaux_feats + \
            " --stats "                + stats + \
            " --outdir "               + outdir + \
            " --config "               + config + \
            " --feat_path1 "           + natural_path + \
            " --feat_path2 "           + artificial_path + \
            " --resume "               + resume + \
            " --seed "                 + str(SEED) + \
            " --verbose 1 "
        #print(cmd)
        os.system(cmd)
    
    # EVALUATION (ANALYSIS-SYNTHESIS)
    if execute_steps[2]:
        # check trained model
        checkpoint = "%s/checkpoint-%ssteps.pkl" % (outdir, model_iters)
        _path_check([checkpoint])
        # path setting
        ts_version  = "dv50_vcc2020_24kHz"
        indir = "hdf5/%s/" % (ts_version) # feature input paths
        outdir_eval = "%s/wav/%s/%s/" % (outdir, model_iters, ts_version) # wav output path
        # load testing list
        testaux_feats = "data/%s/feats_ftcyc.scp" % (ts_version)
        # speech decoding
        cmd = "python "          + source_decode + \
            " --world_test "     + testaux_feats + \
            " --stats "          + stats + \
            " --config "         + config + \
            " --indir "          + indir + \
            " --outdir "         + outdir_eval + \
            " --checkpoint "     + checkpoint + \
            " --feat_path1 "     + natural_path + \
            " --feat_path2 "     + artificial_path + \
            " --seed "           + str(DECODE_SEED) + \
            " --verbose 1 "
        #print(cmd)
        os.system(cmd)

    # EVALUATION (VOICE CONVERSION)
    if execute_steps[3]:
        # speaker and testing set setting
        if args['-s'] is None:
            print("Please assign source speaker by -s SRCSPK!")
            sys.exit(0)
        else:
            srcspk = args['-s']
        if args['-t'] is None:
            print("Please assign source speaker by -t TARSPK!")
            sys.exit(0)
        else:
            tarspk = args['-t']
        # check trained model
        checkpoint = "%s/checkpoint-%ssteps.pkl" % (outdir, model_iters)
        _path_check([checkpoint])
        # path setting
        ts_version  = "dv50_vcc2020_24kHz"
        indir = "hdf5/%s/" % (ts_version) # feature input paths
        outdir_eval = "%s/wav/%s/%s/" % (outdir, model_iters, ts_version) # wav output path
        # load testing list
        testaux_feats = "data/%s/feats_cv_%s-%s.scp" % (ts_version, srcspk, tarspk)
        # speech decoding
        cmd = "python "          + source_decode + \
            " --world_test "     + testaux_feats + \
            " --stats "          + stats + \
            " --config "         + config + \
            " --indir "          + indir + \
            " --outdir "         + outdir_eval + \
            " --checkpoint "     + checkpoint + \
            " --feat_path1 "     + converted_path + \
            " --seed "           + str(DECODE_SEED) + \
            " --verbose 1 "
        #print(cmd)
        os.system(cmd)