### MUST CHANGE - COPIED FROM KYLE 

import copy
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (SequenceSampler, downsample_mask,
                                             get_val_mask)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


# dataset for shepherding
class FrankaPushTDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None
                 ):

        super().__init__()
        # TODO: add observations must add zarr key
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['scene_img', 'ee_img', 'agent_pos', 'action'])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # TODO: If add more obs and img, must normalize them
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['agent_pos']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['scene_img'] = get_image_range_normalizer()
        normalizer['ee_img'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # TODO: If add obs, grab from sample dictionary
        # scene_image = np.moveaxis(sample['scene_img'], -1, 0)/255
        # ee_image = np.moveaxis(sample['ee_img'], -1, 0)/255
        scene_image = sample['scene_img']/255
        ee_image = sample['ee_img']/255
        agent_pos = sample['agent_pos'].astype(np.float)

        # TODO: Add or remove observations
        data = {
            'obs': {
                'scene_img': scene_image,  # T, 3, 230, 230
                'ee_img': ee_image,
                'agent_pos': agent_pos
            },
            'action': sample['action'].astype(np.float32)  # T, 2
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data