#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import argparse
import logging
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import check_hdf5
from utils import read_hdf5
from utils import read_txt
from utils import write_hdf5


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--stats", default=None, required=True,
        help="filename of hdf5 format")
    parser.add_argument("--expdir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument("--stdim", default=5,
        type=int, help="directory to save the log")
    parser.add_argument("--spkr", default=None,
        type=str, help="directory to save the log")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/calc_stats.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # read list and define scaler
    filenames = read_txt(args.feats)
    scaler_feat_org_lf0 = StandardScaler()
    logging.info("number of training utterances = "+str(len(filenames)))

    #var = []
    var_range = []
    f0s_range = np.empty((0))
    # process over all of data
    for filename in filenames:
        logging.info(filename)
        feat_org_lf0 = read_hdf5(filename, "/feat_org_lf0")
        scaler_feat_org_lf0.partial_fit(feat_org_lf0)
        mcep_range = feat_org_lf0[:,args.stdim:]
        var_range.append(np.var(mcep_range, axis=0))
        logging.info(mcep_range.shape)
        if check_hdf5(filename, "/f0_range"):
            f0_range = read_hdf5(filename, "/f0_range")
        else:
            f0_range = read_hdf5(filename, "/f0")
        nonzero_indices = np.nonzero(f0_range)
        logging.info(f0_range[nonzero_indices].shape)
        logging.info(f0s_range.shape)
        f0s_range = np.concatenate([f0s_range,f0_range[nonzero_indices]])
        logging.info(f0s_range.shape)

    mean_feat_org_lf0 = scaler_feat_org_lf0.mean_
    scale_feat_org_lf0 = scaler_feat_org_lf0.scale_
    gv_range_mean = np.mean(np.array(var_range), axis=0)
    gv_range_var = np.var(np.array(var_range), axis=0)
    logging.info(gv_range_mean)
    logging.info(gv_range_var)
    f0_range_mean = np.mean(f0s_range)
    f0_range_std = np.std(f0s_range)
    logging.info(f0_range_mean)
    logging.info(f0_range_std)
    lf0_range_mean = np.mean(np.log(f0s_range))
    lf0_range_std = np.std(np.log(f0s_range))
    logging.info(lf0_range_mean)
    logging.info(lf0_range_std)
    logging.info(np.array_equal(f0_range_mean,np.exp(lf0_range_mean)))
    logging.info(np.array_equal(f0_range_std,np.exp(lf0_range_std)))

    logging.info(mean_feat_org_lf0)
    logging.info(scale_feat_org_lf0)
    write_hdf5(args.stats, "/mean_feat_org_lf0", mean_feat_org_lf0)
    write_hdf5(args.stats, "/scale_feat_org_lf0", scale_feat_org_lf0)
    write_hdf5(args.stats, "/gv_range_mean", gv_range_mean)
    write_hdf5(args.stats, "/gv_range_var", gv_range_var)
    write_hdf5(args.stats, "/f0_range_mean", f0_range_mean)
    write_hdf5(args.stats, "/f0_range_std", f0_range_std)
    write_hdf5(args.stats, "/lf0_range_mean", lf0_range_mean)
    write_hdf5(args.stats, "/lf0_range_std", lf0_range_std)


if __name__ == "__main__":
    main()
