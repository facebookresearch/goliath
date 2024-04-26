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

from collections import OrderedDict, deque
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig

from torch.optim.lr_scheduler import LRScheduler

from ca_body.utils.torchutils import to_device
from ca_body.utils.module_loader import load_class, build_optimizer

from torchvision.utils import make_grid

import logging

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def process_losses(
    loss_dict: Dict[str, Any], reduce: bool = True, detach: bool = True
) -> Dict[str, th.Tensor]:
    """Preprocess the dict of losses outputs."""
    result = {
        k.replace("loss_", ""): v for k, v in loss_dict.items() if k.startswith("loss_")
    }
    if detach:
        result = {k: v.detach() for k, v in result.items()}
    if reduce:
        result = {k: float(v.mean().item()) for k, v in result.items()}
    return result


def load_from_config(config: Mapping[str, Any], **kwargs):
    """Instantiate an object given a config and arguments."""
    assert "class_name" in config and "module_name" not in config
    config = copy.deepcopy(config)
    ckpt = None if "ckpt" not in config else config.pop("ckpt")
    class_name = config.pop("class_name")
    object_class = load_class(class_name)
    instance = object_class(**config, **kwargs)
    if ckpt is not None:
        load_checkpoint(
            ckpt_path=ckpt.path,
            modules={ckpt.get("module_name", "model"): instance},
            ignore_names=ckpt.get("ignore_names", []),
            strict=ckpt.get("strict", False),
        )
    return instance


def save_checkpoint(
    ckpt_path, modules: Dict[str, Any], iteration=None, keep_last_k=None
):
    if keep_last_k is not None:
        raise NotImplementedError()
    ckpt_dict = {}
    if os.path.isdir(ckpt_path):
        assert iteration is not None
        ckpt_path = os.path.join(ckpt_path, f"{iteration:06d}.pt")
        ckpt_dict["iteration"] = iteration
    for name, mod in modules.items():
        if hasattr(mod, "module"):
            mod = mod.module
        ckpt_dict[name] = mod.state_dict()
    th.save(ckpt_dict, ckpt_path)


def filter_params(params, ignore_names):
    return OrderedDict(
        [
            (k, v)
            for k, v in params.items()
            if not any([re.match(n, k) is not None for n in ignore_names])
        ]
    )


def get_inputs(model: nn.Module, required_only: bool = True) -> List[str]:
    """Returns names of model inputs."""
    return [
        name
        for name, param in inspect.signature(model.forward).parameters.items()
        if not required_only or type(None) not in typing.get_args(param.annotation)
    ]


def filter_inputs(
    inputs: Dict[str, th.Tensor], model: nn.Module, required_only: bool = True
) -> Dict[str, th.Tensor]:
    """Returns a subset of inputs for the model."""
    return {
        name: inputs[name]
        for name in get_inputs(model, required_only)
        if name in inputs
    }


def load_checkpoint(
    ckpt_path: str,
    modules: Dict[str, Any],
    iteration: int = None,
    strict: bool = False,
    map_location: Optional[str] = None,
    ignore_names: Optional[Dict[str, List[str]]] = None,
):
    """Load a checkpoint.
    Args:
        ckpt_path: directory or the full path to the checkpoint
    """
    if map_location is None:
        map_location = "cpu"
    # adding
    if os.path.isdir(ckpt_path):
        if iteration is None:
            ckpt_path = os.path.join(ckpt_path, "latest.pt")
        else:
            ckpt_path = os.path.join(ckpt_path, f"{iteration:06d}.pt")
    logger.info(f"loading checkpoint {ckpt_path}")
    ckpt_dict = th.load(ckpt_path, map_location=map_location)
    for name, mod in modules.items():
        params = ckpt_dict[name]
        if ignore_names is not None and name in ignore_names:
            logger.info(f"skipping: {ignore_names[name]}")
            params = filter_params(params, ignore_names[name])
        mod.load_state_dict(params)


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: th.optim.Optimizer,
    train_data: Iterator,
    config: Mapping[str, Any],
    lr_scheduler: Optional[LRScheduler] = None,
    train_writer: Optional[SummaryWriter] = None,
    summary_fn: Optional[Callable] = None,
    batch_filter_fn: Optional[Callable] = None,
    saving_enabled: bool = True,
    logging_enabled: bool = True,
    summary_enabled: bool = True,
    iteration: int = 0,
    device: Optional[Union[th.device, str]] = "cuda:0",
) -> None:

    # Loss history for explosion checking.
    loss_history = deque(maxlen=32)
    loss_history.append(np.inf)
    for batch in train_data:
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

        prev_loss = sum(loss_history) / len(loss_history)
        exploded = (
            loss.item() > 20 * prev_loss or th.isnan(loss) or th.isinf(loss)
        )
        if exploded:
            logger.info(f"explosion detected: iter={iteration}: frame_id=`{batch['frame_id']}`, camera_id=`{batch['camera_id']}`")
        else:
            loss_history.append(loss.item())

        if exploded:
            load_checkpoint(
                config.train.ckpt_dir, modules={"model": model, "optimizer": optimizer}
            )
            loss_history.clear()
            loss_history.append(np.inf)
            continue

        optimizer.zero_grad()
        loss.backward()
        
        optim_params = [p for pg in optimizer.param_groups for p in pg["params"]]
        for p in optim_params:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.data[th.isnan(p.grad.data)] = 0
                p.grad.data[th.isinf(p.grad.data)] = 0
        th.nn.utils.clip_grad_norm_(optim_params, 1.0)
        optimizer.step()

        if logging_enabled and iteration % config.train.log_every_n_steps == 0:
            _loss_dict = process_losses(loss_dict)
            loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
            logger.info(f"iter={iteration}: {loss_str}")

        if (
            logging_enabled
            and train_writer is not None
            and iteration % config.train.log_every_n_steps == 0
        ):
            for name, value in _loss_dict.items():
                train_writer.add_scalar(f"Losses/{name}", value, global_step=iteration)
            train_writer.flush()

        if (
            summary_enabled
            and summary_fn is not None
            and train_writer is not None
            and iteration % config.train.summary_every_n_steps == 0
        ):
            summaries = summary_fn(preds, batch)
            for name, value in summaries.items():
                train_writer.add_image(f"Images/{name}", value, global_step=iteration)

        if (
            saving_enabled
            and iteration is not None
            and iteration % config.train.ckpt_every_n_steps == 0
        ):
            logger.info(
                f"iter={iteration}: saving checkpoint to `{config.train.ckpt_dir}`"
            )
            save_checkpoint(
                f"{config.train.ckpt_dir}/latest.pt",
                {"model": model, "optimizer": optimizer},
                iteration=iteration,
            )

        if (
            lr_scheduler is not None
            and iteration
            and iteration % config.train.update_lr_every == 0
        ):
            lr_scheduler.step()

        iteration += 1
        if iteration >= config.train.n_max_iters:
            logger.info(f"reached max number of iters ({config.train.n_max_iters})")
            break

    if saving_enabled:
        logger.info(
            f"saving the final checkpoint to `{config.train.ckpt_dir}/model.pt`"
        )
        save_checkpoint(f"{config.train.ckpt_dir}/model.pt", {"model": model})
