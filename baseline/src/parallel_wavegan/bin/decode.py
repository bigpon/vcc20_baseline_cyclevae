#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Wu Yi-Chiao (Nagoya University)
# Modified from a ParallelWaveGAN repo by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

import parallel_wavegan.models

from parallel_wavegan.datasets import WorldDataset
from parallel_wavegan.utils import read_hdf5


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained Parallel WaveGAN Generator "
                    "(See detail in parallel_wavegan/bin/decode.py).")
    parser.add_argument("--world_test", required=True, type=str, 
                        help="list or directory of testing aux feat files")
    parser.add_argument("--stats", required=True, type=str, 
                        help="hdf5 file including statistics")
    parser.add_argument("--indir", required=True, type=str, 
                        help="directory of input feature files")
    parser.add_argument("--outdir", required=True, type=str, 
                        help="directory to save generated samples")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", default=None, type=str, 
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
    parser.add_argument("--feat_path1", default="world", type=str,
                        help="default feature path(channel) of hdf5 files.")
    parser.add_argument("--feat_path2", default=None, type=str,
                        help="second feature path(channel) of hdf5 files.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--seed", default=100, type=int, 
                        help="seed number")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # check directory existence
    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    world_load_fn = lambda x: read_hdf5(x, 
                                        hdf5_path1=args.feat_path1, 
                                        hdf5_path2=args.feat_path2)

    dataset = WorldDataset(
            stats=args.stats,
            world_list=args.world_test,
            world_load_fn=world_load_fn,            
            return_filename=True,
            mean_path=config.get("mean_path", "/world/mean"),
            scale_path=config.get("scale_path", "/world/scale"),
    )  #
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"))
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["model"]["generator"])
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)
    use_noise_input = not isinstance(
        model, parallel_wavegan.models.MelGANGenerator)
    pad_fn = torch.nn.ReplicationPad1d(
        config["generator_params"].get("aux_context_window", 0))

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, (feat_path, c) in enumerate(pbar, 1):
            # setup input
            c = pad_fn(torch.FloatTensor(c).unsqueeze(0).transpose(2, 1)).to(device)
            x = (c,)
            if use_noise_input:
                z_size = (1, 1, (c.size(2) - sum(pad_fn.padding)) * config["hop_size"])
                z = torch.randn(z_size).to(device)
                x = (z,) + x

            # generate
            start = time.time()
            y = model(*x).view(-1).cpu().numpy()
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            feat_path = os.path.splitext(feat_path)[0]
            feat_path = feat_path.replace(args.indir, args.outdir)
            wav_filename = "%s.wav" % feat_path
            if not os.path.exists(os.path.dirname(wav_filename)):
                os.makedirs(os.path.dirname(wav_filename))
            sf.write(wav_filename, y, config["sampling_rate"], "PCM_16")

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
