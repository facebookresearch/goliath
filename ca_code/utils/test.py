# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch as th
import os
import re
import glob
import copy
import typing
import inspect
from typing import Callable, Dict, Any, Iterator, Mapping, Optional, Union, Tuple, List
import torch.nn as nn
import shutil

from pathlib import Path
from collections import OrderedDict, deque
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig

from torch.optim.lr_scheduler import LRScheduler
from ca_code.utils.image import linear2srgb

from ca_code.utils.train import (
    process_losses,
    filter_inputs,
    load_checkpoint,

)
from ca_code.utils.torchutils import to_device
from ca_code.utils.module_loader import load_class, build_optimizer

from torchvision.utils import make_grid, save_image

import logging

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@th.no_grad
def test(
    model: nn.Module,
    loss_fn: nn.Module,
    test_data: Iterator,
    config: Mapping[str, Any],
    vis_path: Optional[Path] = None,
    test_writer: Optional[SummaryWriter] = None,
    summary_fn: Optional[Callable] = None,
    batch_filter_fn: Optional[Callable] = None,
    logging_enabled: bool = True,
    summary_enabled: bool = True,
    iteration: int = 0,
    device: Optional[Union[th.device, str]] = "cuda:0",
) -> None:

    for i, batch in enumerate(test_data):

        if batch is None:
            logger.info("skipping empty batch")
            continue
        batch = to_device(batch, device)
        batch["iteration"] = iteration
                
        if batch_filter_fn is not None:
            batch_filter_fn(batch)

        # leaving only inputs acutally used by the model
        preds = model(**filter_inputs(batch, model, required_only=False))

        # TODO: switch to the old-school loss computation
        loss, loss_dict = loss_fn(preds, batch, iteration=iteration)

        if logging_enabled and iteration % config.test.log_every_n_steps == 0:
            _loss_dict = process_losses(loss_dict)
            loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
            logger.info(f"iter={i+1}/{len(test_data)}: {loss_str}")

            # vis
            if vis_path:

                if "hand" in str(vis_path):
                    preds["rgb"] = preds["rgb"] / 255.0
                    batch["image"] = batch["image"] / 255.0
                
                pred = linear2srgb(preds["rgb"]).squeeze()
                gt = linear2srgb(batch["image"]).squeeze()

                l2 = (pred - gt) ** 2
                out = make_grid([gt, pred, l2 * 20], nrow=4)
                save_image(out, vis_path / f"{i:04d}.png")

        if (
            logging_enabled
            and test_writer is not None
            and iteration % config.test.log_every_n_steps == 0
        ):
            for name, value in _loss_dict.items():
                test_writer.add_scalar(f"Losses/{name}", value, global_step=iteration)
            test_writer.flush()

        if (
            summary_enabled
            and summary_fn is not None
            and test_writer is not None
            and iteration % config.test.summary_every_n_steps == 0
        ):
            summaries = summary_fn(preds, batch)
            for name, value in summaries.items():
                test_writer.add_image(f"Images/{name}", value, global_step=iteration)
