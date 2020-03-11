#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for VC by Kazuhiro Kobayashi (Nagoya University)
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import torch
import os
import logging
import soundfile as sf
from utils import read_hdf5, check_hdf5
from torch.utils.data import Dataset



def padding(x, flen, value=0):
    """Pad values to end by flen"""
    diff = flen - x.shape[0]
    if diff > 0:
        if len(x.shape) > 1:
            x = np.concatenate([x, np.ones((diff, x.shape[1])) * value])
        else:
            x = np.concatenate([x, np.ones(diff) * value])
    return x


def proc_multspk_data_mix_random_cls_statcvexcit(featfile, spk_list, n_cyc, src_code, n_spk, \
        spk_idx_dict, stat_spk_list):
    featfile_spk = os.path.basename(os.path.dirname(featfile))
    spk_idx = spk_idx_dict[featfile_spk]
    src_code[:,spk_idx] = 1
    src_class_code = np.ones(src_code.shape[0],dtype=np.int64)*spk_idx
    mean_trg_list = [None]*n_cyc
    std_trg_list = [None]*n_cyc
    trg_code_list = [None]*n_cyc
    pair_spk_list = [None]*n_cyc
    trg_class_code_list = [None]*n_cyc
    for i in range(n_cyc):
        trg_code_list[i] = np.zeros((src_code.shape[0],src_code.shape[1]))
        pair_idx = np.random.randint(0,n_spk)
        while spk_list[pair_idx] == featfile_spk:
            pair_idx = np.random.randint(0,n_spk)
        trg_code_list[i][:,pair_idx] = 1
        trg_class_code_list[i] = np.ones(src_code.shape[0],dtype=np.int64)*pair_idx
        mean_trg_list[i] = read_hdf5(stat_spk_list[pair_idx], "/mean_feat_org_lf0")[1:2]
        std_trg_list[i] = read_hdf5(stat_spk_list[pair_idx], "/scale_feat_org_lf0")[1:2]
        pair_spk_list[i] = spk_list[pair_idx]

    return spk_idx, mean_trg_list, std_trg_list, trg_code_list, featfile_spk, pair_spk_list, \
            src_class_code, trg_class_code_list


class FeatureDatasetMultMixTrainVAEClsStatCvExcit(Dataset):
    """Dataset for training mix many-to-many with speaker-posterior
    """

    def __init__(self, file_list, pad_transform, spk_list, n_cyc, stat_spk_list, pair_utt_flag=True, stdim=5):
        self.file_list = file_list
        self.pad_transform = pad_transform
        self.spk_list = spk_list
        self.pair_utt_flag = pair_utt_flag
        self.stat_spk_list = stat_spk_list
        self.n_spk = len(self.spk_list)
        self.n_cyc = n_cyc
        self.stdim = stdim
        self.spk_idx_dict = {}
        for i in range(self.n_spk):
            self.spk_idx_dict[self.spk_list[i]] = i

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        featfile_src = self.file_list[idx]

        h_src = read_hdf5(featfile_src, "/feat_org_lf0")
        flen_src = h_src.shape[0]
        src_code = np.zeros((flen_src,self.n_spk))

        spk_idx, mean_trg_list, std_trg_list, src_trg_code_list, featfile_spk, pair_spk_list, src_class_code, \
            trg_class_code_list = proc_multspk_data_mix_random_cls_statcvexcit(featfile_src, self.spk_list, \
                self.n_cyc, src_code, self.n_spk, self.spk_idx_dict, self.stat_spk_list)

        mean_src = read_hdf5(self.stat_spk_list[spk_idx], "/mean_feat_org_lf0")[1:2]
        std_src = read_hdf5(self.stat_spk_list[spk_idx], "/scale_feat_org_lf0")[1:2]

        if check_hdf5(featfile_src, "/spcidx_range"):
            spcidx_src = read_hdf5(featfile_src, "/spcidx_range")[0]
        else:
            spk_f0rate = os.path.basename(os.path.dirname(featfile_src))
            spk_ = spk_f0rate.split('_')[0]
            spcidx_src = read_hdf5(os.path.join(os.path.dirname(os.path.dirname(featfile_src)), spk_, \
                            os.path.basename(featfile_src)), "/spcidx_range")[0]
        flen_spc_src = spcidx_src.shape[0]

        src_code = torch.FloatTensor(self.pad_transform(src_code))
        src_class_code = torch.LongTensor(self.pad_transform(src_class_code))

        cv_src_list = [None]*self.n_cyc
        for i in range(self.n_cyc):
            cv_src_list[i] = torch.FloatTensor(self.pad_transform(np.c_[h_src[:,:1], \
                                (std_trg_list[i]/std_src)*(h_src[:,1:2]-mean_src)+mean_trg_list[i], \
                                    h_src[:,2:self.stdim]]))
            src_trg_code_list[i] = torch.FloatTensor(self.pad_transform(src_trg_code_list[i]))
            trg_class_code_list[i] = torch.LongTensor(self.pad_transform(trg_class_code_list[i]))

        h_src = torch.FloatTensor(self.pad_transform(h_src))
        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))

        file_src_trg_flag = False
        if self.pair_utt_flag:
            featfile_src_trg = os.path.dirname(os.path.dirname(featfile_src))+"/"+pair_spk_list[0]+\
                                                "/"+os.path.basename(featfile_src)
            if os.path.exists(featfile_src_trg):
                file_src_trg_flag = True
                h_src_trg = read_hdf5(featfile_src_trg, "/feat_org_lf0")
                flen_src_trg = h_src_trg.shape[0]
                if check_hdf5(featfile_src_trg, "/spcidx_range"):
                    spcidx_src_trg = read_hdf5(featfile_src_trg, "/spcidx_range")[0]
                else:
                    spk_f0rate = os.path.basename(os.path.dirname(featfile_src_trg))
                    spk_ = spk_f0rate.split('_')[0]
                    spcidx_src_trg = read_hdf5(os.path.join(os.path.dirname(\
                            os.path.dirname(featfile_src_trg)), spk_,\
                                os.path.basename(featfile_src_trg)), "/spcidx_range")[0]
                flen_spc_src_trg = spcidx_src_trg.shape[0]
                h_src_trg = torch.FloatTensor(self.pad_transform(h_src_trg))
                spcidx_src_trg = torch.LongTensor(self.pad_transform(spcidx_src_trg))
            else:
                h_src_trg = h_src
                flen_src_trg = flen_src
                spcidx_src_trg = spcidx_src
                flen_spc_src_trg = flen_spc_src
            return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, \
                    'src_trg_code_list': src_trg_code_list, 'cv_src_list': cv_src_list, \
                    'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, 'h_src_trg': h_src_trg, \
                    'flen_src_trg': flen_src_trg, 'spcidx_src_trg': spcidx_src_trg, \
                    'flen_spc_src_trg': flen_spc_src_trg, 'featfile_src': featfile_src, \
                    'featfile_src_trg': featfile_src_trg, 'featfile_spk': featfile_spk, \
                    'pair_spk_list': pair_spk_list, 'src_class_code': src_class_code, \
                    'trg_class_code_list': trg_class_code_list, 'file_src_trg_flag': file_src_trg_flag}
        else:
            return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, \
                    'src_trg_code_list': src_trg_code_list, 'cv_src_list': cv_src_list, \
                    'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, 'featfile_src': featfile_src, \
                    'featfile_spk': featfile_spk, 'pair_spk_list': pair_spk_list, \
                    'src_class_code': src_class_code, 'trg_class_code_list': trg_class_code_list, \
                    'file_src_trg_flag': file_src_trg_flag}


class FeatureDatasetMultMixEvalVAEClsStatCvExcit(Dataset):
    """Dataset for evaluation mix many-to-many with speaker-posterior and/or classifier
    """

    def __init__(self, file_list, pad_transform, spk_list, stat_spk_list, stdim=5):
        self.file_list = file_list
        self.pad_transform = pad_transform
        self.spk_list = spk_list
        self.stat_spk_list = stat_spk_list
        self.n_spk = len(self.spk_list)
        self.file_list_src = []
        self.file_list_src_trg = []
        self.list_src_trg_flag = []
        self.stdim = stdim
        self.spk_idx_dict = {}
        for i in range(self.n_spk):
            self.spk_idx_dict[self.spk_list[i]] = i
        self.count_spk_pair_cv = {}
        for i in range(self.n_spk):
            ## omitting WSOLA-based F0-transformed waveforms ('spk_f0rate' has '.' in 'f0rate'),
            ## and omitting VCTK speakers (starting with 'p') in validation
            ## [please edit accordingly to include]
            if '.' not in spk_list[i] and spk_list[i].find('p') != 0:
                self.count_spk_pair_cv[self.spk_list[i]] = {}
                for j in range(self.n_spk):
                    if i != j and '.' not in spk_list[i] and spk_list[i].find('p') != 0:
                        self.count_spk_pair_cv[self.spk_list[i]][self.spk_list[j]] = 0
        eval_exist = False
        for i in range(self.n_spk):
            if '.' not in spk_list[i] and spk_list[i].find('p') != 0 and len(self.file_list[i]) > 0:
                eval_exist = True
                break
        if eval_exist:
            # deterministically select a conv. pair for each validation utterance,
            # and deal with existence of pair data
            n_pair = self.n_spk // 2 
            n_src = n_pair + self.n_spk % 2
            for spk_src_idx in range(n_src):
                if '.' not in spk_list[spk_src_idx] and spk_list[spk_src_idx].find('p') != 0: 
                    spk_src = self.spk_list[spk_src_idx]
                    spk_src_n_utt = len(self.file_list[spk_src_idx])
                    spk_trg_idx_start = spk_src_idx + n_pair
                    while '.' in spk_list[spk_trg_idx_start] or spk_list[spk_trg_idx_start].find('p') == 0:
                        spk_trg_idx_start += 1
                        if spk_trg_idx_start >= self.n_spk:
                            spk_trg_idx_start = 0
                    flag = False
                    for spk_trg_idx in range(spk_trg_idx_start,self.n_spk):
                        if '.' not in spk_list[spk_trg_idx] and spk_list[spk_trg_idx].find('p') != 0:
                            spk_trg = self.spk_list[spk_trg_idx]
                            for i in range(spk_src_n_utt):
                                file_src = self.file_list[spk_src_idx][i]
                                file_trg = os.path.dirname(os.path.dirname(file_src))+"/"+spk_trg+"/"+\
                                                os.path.basename(file_src)
                                if (file_trg in self.file_list) or os.path.exists(file_trg):
                                    self.file_list_src.append(file_src)
                                    self.file_list_src_trg.append(file_trg)
                                    self.count_spk_pair_cv[spk_src][spk_trg] += 1
                                    flag = True
                                    self.list_src_trg_flag.append(flag)
                                elif flag:
                                    self.file_list_src.append(file_src)
                                    self.file_list_src_trg.append(file_trg)
                                    self.count_spk_pair_cv[spk_src][spk_trg] += 1
                                    self.list_src_trg_flag.append(False)
                            if flag:
                                break
                    if not flag:
                        for spk_trg_idx in range(spk_trg_idx_start):
                            if '.' not in spk_list[spk_trg_idx] and spk_list[spk_trg_idx].find('p') != 0:
                                if spk_trg_idx != spk_src_idx:
                                    spk_trg = self.spk_list[spk_trg_idx]
                                    for i in range(spk_src_n_utt):
                                        file_src = self.file_list[spk_src_idx][i]
                                        file_trg = os.path.dirname(os.path.dirname(file_src))+"/"+spk_trg+\
                                                                    "/"+os.path.basename(file_src)
                                        if (file_trg in self.file_list) or os.path.exists(file_trg):
                                            self.file_list_src.append(file_src)
                                            self.file_list_src_trg.append(file_trg)
                                            self.count_spk_pair_cv[spk_src][spk_trg] += 1
                                            flag = True
                                            self.list_src_trg_flag.append(flag)
                                        elif flag:
                                            self.file_list_src.append(file_src)
                                            self.file_list_src_trg.append(file_trg)
                                            self.count_spk_pair_cv[spk_src][spk_trg] += 1
                                            self.list_src_trg_flag.append(False)
                                    if flag:
                                        break
                        if not flag:
                            spk_trg = self.spk_list[spk_trg_idx_start]
                            for i in range(spk_src_n_utt):
                                file_src = self.file_list[spk_src_idx][i]
                                file_trg = os.path.dirname(os.path.dirname(file_src))+"/"+spk_trg+\
                                                            "/"+os.path.basename(file_src)
                                self.file_list_src.append(file_src)
                                self.file_list_src_trg.append(file_trg)
                                self.count_spk_pair_cv[spk_src][spk_trg] += 1
                                self.list_src_trg_flag.append(False)

    def __len__(self):
        return len(self.file_list_src)

    def __getitem__(self, idx):
        featfile_src = self.file_list_src[idx]
        featfile_src_trg = self.file_list_src_trg[idx]
        file_src_trg_flag = self.list_src_trg_flag[idx]

        spk_src = os.path.basename(os.path.dirname(featfile_src))
        spk_trg = os.path.basename(os.path.dirname(featfile_src_trg))
        idx_src = self.spk_idx_dict[spk_src]
        idx_trg = self.spk_idx_dict[spk_trg]

        mean_src = read_hdf5(self.stat_spk_list[idx_src], "/mean_feat_org_lf0")[1:2]
        std_src = read_hdf5(self.stat_spk_list[idx_src], "/scale_feat_org_lf0")[1:2]
        mean_trg = read_hdf5(self.stat_spk_list[idx_trg], "/mean_feat_org_lf0")[1:2]
        std_trg = read_hdf5(self.stat_spk_list[idx_trg], "/scale_feat_org_lf0")[1:2]

        h_src = read_hdf5(featfile_src, "/feat_org_lf0")
        flen_src = h_src.shape[0]
        src_code = np.zeros((flen_src,self.n_spk))
        src_trg_code = np.zeros((flen_src,self.n_spk))
        src_code[:,idx_src] = 1
        src_trg_code[:,idx_trg] = 1
        cv_src = np.c_[h_src[:,:1], (std_trg/std_src)*(h_src[:,1:2]-mean_src)+mean_trg, h_src[:,2:self.stdim]]
        if check_hdf5(featfile_src, "/spcidx_range"):
            spcidx_src = read_hdf5(featfile_src, "/spcidx_range")[0]
        else:
            spk_f0rate = os.path.basename(os.path.dirname(featfile_src))
            spk_ = spk_f0rate.split('_')[0]
            spcidx_src = read_hdf5(os.path.join(os.path.dirname(os.path.dirname(featfile_src)), spk_, \
                            os.path.basename(featfile_src)), "/spcidx_range")[0]
        src_class_code = np.ones(h_src.shape[0],dtype=np.int64)*idx_src
        src_trg_class_code = np.ones(h_src.shape[0],dtype=np.int64)*idx_trg
        flen_spc_src = spcidx_src.shape[0]

        if file_src_trg_flag:
            h_src_trg = read_hdf5(featfile_src_trg, "/feat_org_lf0")
            flen_src_trg = h_src_trg.shape[0]
            trg_code = np.zeros((flen_src_trg,self.n_spk))
            trg_src_code = np.zeros((flen_src_trg,self.n_spk))
            trg_code[:,idx_trg] = 1
            trg_src_code[:,idx_src] = 1
            cv_trg = np.c_[h_src_trg[:,:1], (std_src/std_trg)*(h_src_trg[:,1:2]-mean_trg)+mean_src, \
                            h_src_trg[:,2:self.stdim]]
            if check_hdf5(featfile_src_trg, "/spcidx_range"):
                spcidx_src_trg = read_hdf5(featfile_src_trg, "/spcidx_range")[0]
            else:
                spk_f0rate = os.path.basename(os.path.dirname(featfile_src_trg))
                spk_ = spk_f0rate.split('_')[0]
                spcidx_src_trg = read_hdf5(os.path.join(os.path.dirname(os.path.dirname(featfile_src_trg)), \
                                        spk_, os.path.basename(featfile_src_trg)), "/spcidx_range")[0]
            trg_class_code = np.ones(h_src_trg.shape[0],dtype=np.int64)*idx_trg
            trg_src_class_code = np.ones(h_src_trg.shape[0],dtype=np.int64)*idx_src
            flen_spc_src_trg = spcidx_src_trg.shape[0]

        h_src = torch.FloatTensor(self.pad_transform(h_src))
        src_code = torch.FloatTensor(self.pad_transform(src_code))
        src_trg_code = torch.FloatTensor(self.pad_transform(src_trg_code))
        cv_src = torch.FloatTensor(self.pad_transform(cv_src))
        spcidx_src = torch.LongTensor(self.pad_transform(spcidx_src))
        src_class_code = torch.LongTensor(self.pad_transform(src_class_code))
        src_trg_class_code = torch.LongTensor(self.pad_transform(src_trg_class_code))

        if file_src_trg_flag:
            h_src_trg = torch.FloatTensor(self.pad_transform(h_src_trg))
            trg_code = torch.FloatTensor(self.pad_transform(trg_code))
            trg_src_code = torch.FloatTensor(self.pad_transform(trg_src_code))
            cv_trg = torch.FloatTensor(self.pad_transform(cv_trg))
            spcidx_src_trg = torch.LongTensor(self.pad_transform(spcidx_src_trg))
            trg_class_code = torch.LongTensor(self.pad_transform(trg_class_code))
            trg_src_class_code = torch.LongTensor(self.pad_transform(trg_src_class_code))
        else:
            flen_src_trg = flen_src
            h_src_trg = h_src
            trg_code = src_code
            trg_src_code = src_trg_code
            cv_trg = cv_src
            spcidx_src_trg = spcidx_src
            trg_class_code = src_class_code
            trg_src_class_code = src_trg_class_code
            flen_spc_src_trg = flen_spc_src

        return {'h_src': h_src, 'flen_src': flen_src, 'src_code': src_code, 'src_trg_code': src_trg_code, \
                'cv_src': cv_src, 'spcidx_src': spcidx_src, 'flen_spc_src': flen_spc_src, \
                'h_src_trg': h_src_trg, 'flen_src_trg': flen_src_trg, 'trg_code': trg_code, \
                'trg_src_code': trg_src_code, 'cv_trg': cv_trg, 'spcidx_src_trg': spcidx_src_trg, \
                'flen_spc_src_trg': flen_spc_src_trg, 'featfile_src': featfile_src, \
                'featfile_src_trg': featfile_src_trg, 'src_class_code': src_class_code, \
                'src_trg_class_code': src_trg_class_code, 'trg_class_code': trg_class_code, \
                'trg_src_class_code': trg_src_class_code, 'file_src_trg_flag': file_src_trg_flag}
