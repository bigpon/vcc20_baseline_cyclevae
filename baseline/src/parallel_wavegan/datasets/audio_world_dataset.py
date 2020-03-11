# -*- coding: utf-8 -*-

# Copyright 2020 Wu Yi-Chiao (Nagoya University)
# Modified from a ParallelWaveGAN repo by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging

from multiprocessing import Manager

import numpy as np

from torch.utils.data import Dataset

from parallel_wavegan.utils import read_txt, check_filename
from parallel_wavegan.utils import read_hdf5
from sklearn.preprocessing import StandardScaler
import soundfile as sf

def _validate_length(x, y, hop_size=None):
    """VALIDATE LENGTH
    Args:
        x (ndarray): numpy.ndarray with x.shape[0] = len_x
        y (ndarray): numpy.ndarray with y.shape[0] = len_y
        hop_size (int): upsampling factor
    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x
    """
    if hop_size is None:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        if x.shape[0] > y.shape[0] * hop_size:
            x = x[:y.shape[0] * hop_size]
        if x.shape[0] < y.shape[0] * hop_size:
            mod_y = y.shape[0] * hop_size - x.shape[0]
            mod_y_frame = mod_y // hop_size + 1
            y = y[:-mod_y_frame]
            x = x[:y.shape[0] * hop_size]
        assert len(x) == len(y) * hop_size

    return x, y

class AudioWorldDataset(Dataset):
    """PyTorch compatible audio and world dataset."""

    def __init__(self,
                 stats,
                 audio_list,
                 world_list,
                 audio_load_fn=sf.read,
                 world_load_fn=lambda x: read_hdf5(x, "world"),
                 hop_size=110,
                 audio_length_threshold=None,
                 world_length_threshold=None,
                 return_filename=False,
                 allow_cache=False,                 
                 mean_path="/world/mean",
                 scale_path="/world/scale",
                 ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            audio_list (str): Filename of the list of audio files.
            world_list (str): Filename of the list of world feature files.
            audio_load_fn (func): Function to load audio file.
            world_load_fn (func): Function to load world feature file.
            hop_size (int): Hope size of world feature
            audio_length_threshold (int): Threshold to remove short audio files.
            world_length_threshold (int): Threshold to remove short world feature files.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            mean_path (str): The data path (channel) of the mean in the statistic hdf5 file.
            scale_path (str): The data path (channel) of the scale in the statistic hdf5 file.

        """
        # load audio and world file list
        audio_files = read_txt(audio_list)
        world_files = read_txt(world_list)
        # check filename
        assert check_filename(audio_files, world_files)

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [idx for idx in range(len(audio_files)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_files) != len(idxs):
                logging.warning(f"Some files are filtered by audio length threshold "
                                f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            world_files = [world_files[idx] for idx in idxs]
        if world_length_threshold is not None:
            world_lengths = [world_load_fn(f).shape[0] for f in world_files]
            idxs = [idx for idx in range(len(world_files)) if world_lengths[idx] > world_length_threshold]
            if len(world_files) != len(idxs):
                logging.warning(f"Some files are filtered by world length threshold "
                                f"({len(world_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            world_files = [world_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${audio_list}."
        assert len(audio_files) == len(world_files), \
            f"Number of audio and world files are different ({len(audio_files)} vs {len(world_files)})."

        self.audio_files = audio_files
        self.world_files = world_files
        self.audio_load_fn = audio_load_fn
        self.world_load_fn = world_load_fn
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.hop_size = hop_size
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]
        # define feature pre-processing funtion
        scaler        = StandardScaler()
        scaler.mean_  = read_hdf5(stats, mean_path)
        scaler.scale_ = read_hdf5(stats, scale_path)
        self.feat_transform = lambda x: scaler.transform(x)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Audio filename (only in return_filename = True).
            str: Feature filename (only in return_filename = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        audio, fs = self.audio_load_fn(self.audio_files[idx])
        world = self.world_load_fn(self.world_files[idx])
        # make the audio length and feature length are matched
        audio, world = _validate_length(audio, world, self.hop_size)
        # audio & feature pre-processing
        audio = audio.astype(np.float32)
        world = self.feat_transform(world)

        if self.return_filename:
            if self.allow_cache:
                self.caches[idx] = self.audio_files[idx], self.world_files[idx], audio, world
            return self.audio_files[idx], self.world_files[idx], audio, world
        else:
            if self.allow_cache:
                self.caches[idx] = audio, world
            return audio, world

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class AudioDataset(Dataset):
    """PyTorch compatible audio dataset."""

    def __init__(self,
                 audio_list,
                 audio_length_threshold=None,
                 audio_load_fn=sf.read,
                 return_filename=False,
                 allow_cache=False,
                 ):
        """Initialize dataset.

        Args:
            audio_list (str): Filename of the list of audio files.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load audio and world file list
        audio_files = read_txt(audio_list)

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [idx for idx in range(len(audio_files)) if audio_lengths[idx] > audio_length_threshold]
            if len(audio_files) != len(idxs):
                logging.warning(f"some files are filtered by audio length threshold "
                               f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${audio_list}."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Filename (only in return_filename = True).
            ndarray: Audio (T,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        audio, fs = self.audio_load_fn(self.audio_files[idx])
        # audio pre-processing
        audio = audio.astype(np.float32)

        if self.return_filename:
            if self.allow_cache:
                self.caches[idx] = self.audio_files[idx], audio
            return self.audio_files[idx], audio
        else:
            if self.allow_cache:
                self.caches[idx] = audio
            return audio

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class WorldDataset(Dataset):
    """PyTorch compatible world dataset."""

    def __init__(self,
                 stats,
                 world_list,
                 world_length_threshold=None,
                 world_load_fn=lambda x: read_hdf5(x, "world"),
                 return_filename=False,
                 allow_cache=False,
                 hop_size=120,
                 fs=24000,
                 mean_path="/world/mean",
                 scale_path="/world/scale",
                 ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            world_list (str): Filename of the list of feature files.
            world_load_fn (func): Function to load world feature file.
            world_length_threshold (int): Threshold to remove short world feature files.
            return_filename (bool): Whether to return the filename with arrays.
            return_dfactors (bool): Whether to return the dilated factors with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            hop_size (int): Hope size of world feature
            fs (int): Sampling rate
            mean_path (str): The data path (channel) of the mean in the statistic hdf5 file.
            scale_path (str): The data path (channel) of the scale in the statistic hdf5 file.

        """
        # load audio and world file list
        world_files = read_txt(world_list)

        # filter by threshold
        if world_length_threshold is not None:
            world_lengths = [world_load_fn(f).shape[0] for f in world_files]
            idxs = [idx for idx in range(len(world_files)) if world_lengths[idx] > world_length_threshold]
            if len(world_files) != len(idxs):
                logging.warning(f"Some files are filtered by world length threshold "
                                f"({len(world_files)} -> {len(idxs)}).")
            world_files = [world_files[idx] for idx in idxs]

        # assert the number of files
        assert len(world_files) != 0, f"Not found any world files in ${world_list}."

        self.world_files = world_files
        self.world_load_fn = world_load_fn
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.fs = fs
        self.hop_size = hop_size
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(world_files))]
        # define feature pre-processing funtion
        scaler        = StandardScaler()
        scaler.mean_  = read_hdf5(stats, mean_path)
        scaler.scale_ = read_hdf5(stats, scale_path)
        self.feat_transform = lambda x: scaler.transform(x)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Filename (only in return_filename = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        world = self.world_load_fn(self.world_files[idx])
             
        # feature pre-processing
        world = self.feat_transform(world)

        if self.return_filename:
            if self.allow_cache:
                self.caches[idx] = self.world_files[idx], world
            return self.world_files[idx], world
        else:
            if self.allow_cache:
                self.caches[idx] = world
            return world

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.world_files)
