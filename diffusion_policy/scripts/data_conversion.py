# TODO: 
### MUST CHANGE - COPIED FROM KYLE
### CHECK COMMENTS

import os
from typing import List, Tuple

import click
import cv2
import numpy as np
from omegaconf import OmegaConf

from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer


@click.command()
@click.option('-c', '--config', required=True)
def main(config):
    """
    Convert the csv and bmp files from the shepherding game to zarr,
    the data format used for training models in this repo.

    Args:
        config (str): data_conversion config that defines how to convert the csv files
            into the input/output formats for the diffusion policy model.
    """
    cfg = OmegaConf.load(config)

    # Build list of dirs to include
    dir_list = []
    for section in cfg.data:
        dir_list.extend(list(range(section['start'], section['end'] + 1)))

    # Init replay buffer for handling output
    zarr_path = str(cfg.output_file + '.zarr')
    replay_buffer = ReplayBuffer.create_from_path(
        zarr_path=zarr_path, mode='w')

    # Add data to zarr, each directory contains one episode
    for dir in dir_list:
        path = cfg.data_dir + str(dir)

        # Read all images into an array
        img_files = sorted(os.listdir(path+"/img/"),
                           key=lambda x: int(os.path.splitext(x)[0]))
        img_list = []
        for img in img_files:
            # cv2.imread("data/1/img/"+img)
            img_list.append(cv2.imread(path+"/img/"+img))

        img_list = np.array(img_list)
        pos_list = np.loadtxt(path + "/pos.csv", delimiter=',', skiprows=1)

        episode = {
            "img": img_list,
            "action": pos_list
        }
        replay_buffer.add_episode(episode, compressors='disk')


    print(
        f"Converted {replay_buffer.n_episodes} episodes to zarr format successfully and saved to {zarr_path}")



if __name__ == '__main__':
    main()