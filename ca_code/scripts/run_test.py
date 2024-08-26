# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import logging
import random
import os
import sys
from typing import List

import torch as th

from addict import Dict as AttrDict

from ca_code.utils.dataloader import BodyDataset, collate_fn

from ca_code.utils.train import load_checkpoint, load_from_config
from ca_code.utils.test import test


from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def main(config: DictConfig):
    device = th.device(f"cuda:0")

    train_dataset = BodyDataset(**config.data)
    
    batch_filter_fn = train_dataset.batch_filter

    static_assets = AttrDict(train_dataset.static_assets)

    model = (
        load_from_config(
            config.model,
            assets=static_assets)
            .to(device)
            .eval()
        )
    
    # Remove losses that are only computed during training
    config.loss.losses.pop("backlit_reg", None)
    config.loss.losses.pop("learn_blur", None)
    config.loss.losses["psnr"] = {
        "src_key": "rgb",
        "tgt_key": "image",
        "mask_key": "image_weight",
        "weight": 1.,
        "data_range": 255. if ("hand" in config.data.root_path.lower() or "body" in config.data.root_path.lower()) else 1.
    }
    
    loss_fn = load_from_config(config.loss, assets=static_assets).to(device)

    if "ckpt" in config.test:
        logger.info(f"loading checkpoint: {config.test.ckpt}")
        load_checkpoint(**config.test.ckpt, modules={"model": model})
    else:
        raise ValueError("No checkpoint provided")

    logger.info("starting test with the config:")
    logger.info(OmegaConf.to_yaml(config))

    test_dataset = BodyDataset(**config.test.data)
    
    # NOTE(julieta) fix seed so frames are consistent
    frames = test_dataset.get_frame_list()
    test_dataset.frames_subset = sorted(random.sample(frames, 12))

    config.dataloader.shuffle = False
    config.dataloader.batch_size = 1
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        **config.dataloader,
    )

    # Disable learn-only stuff
    model.learn_blur_enabled = False
    
    # TODO(julieta) disable for head and hands, enable for bodies
    # model.cal_enabled = False

    summary_fn = load_from_config(config.summary)

    vis_path = None
    if "vis_path" in config.test:
        vis_path = Path(config.test.vis_path)
        os.makedirs(vis_path, exist_ok=True)

    with th.no_grad():
        test(
            model,
            loss_fn,
            test_loader,
            config,
            vis_path,
            summary_fn=summary_fn,
            batch_filter_fn=batch_filter_fn,
            test_writer=None,
            logging_enabled=True,
            summary_enabled=True,
        )


if __name__ == "__main__":

    config_path: str = sys.argv[1]
    console_commands: List[str] = sys.argv[2:]

    config = OmegaConf.load(config_path)
    config_cli = OmegaConf.from_cli(args_list=console_commands)
    if config_cli:
        logger.info("Overriding with the following args values:")
        logger.info(f"{OmegaConf.to_yaml(config_cli)}")
        config = OmegaConf.merge(config, config_cli)

    main(config)
