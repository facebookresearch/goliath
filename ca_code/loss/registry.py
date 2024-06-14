# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# This is very challenging to typehint

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch as th
import torch.nn as nn
from addict import Dict as AttrDict

logger: logging.Logger = logging.getLogger(__name__)

loss_registry: Dict[str, nn.Module] = {}


def register_loss(loss_name: str, *outputs):
    """Registering default functions ."""
    global loss_registry

    # pyre-ignore
    def _register(fn):
        if loss_name in loss_registry:
            existing = loss_registry[loss_name]
            logger.warning(
                "Tried to register function for %s which has already been registered as %s"
                % (loss_name, existing)
            )
        loss_registry[loss_name] = fn
        return fn

    return _register


class FnLoss(nn.Module):
    """A wrapper around a standalone loss function."""

    def __init__(self, fn: nn.Module, function_args: Dict[str, Any]) -> None:
        super().__init__()
        self.fn = fn

        # We need to do this to establish partiy between losses implemented as torch modules
        # and losses implemented as functions with additional args.
        # The base requirement is that the function takes a prediction and a larget.
        self.extra_args: Dict[str, Any] = function_args

    def forward(
        self, preds: Dict[str, Any], targets: Dict[str, Any]
    ) -> Union[th.Tensor, float]:
        return self.fn(preds, targets, **self.extra_args)


def register_loss_by_fn(loss_name=None, *outputs):
    """Registering default functions ."""
    global loss_registry

    # pyre-ignore
    def _register(fn):
        nonlocal loss_name
        if loss_name is None:
            loss_name = fn.__name__
        if loss_name in loss_registry:
            existing = loss_registry[loss_name]
            logger.warning(
                f"Tried to register function for {loss_name} which has already been registered as {existing}"
            )
        # pyre-ignore
        loss_registry[loss_name] = lambda assets, **function_args: FnLoss(
            fn, function_args
        )
        return fn

    return _register


def get_loss(
    name: str,
    assets: Optional[AttrDict] = None,
    init_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Loss constructor."""
    global loss_registry
    if name not in loss_registry:
        raise ValueError(f"loss {name} not registered!")

    if init_kwargs is None:
        init_kwargs = {}

    LossClass = loss_registry[name]
    return LossClass(assets, **init_kwargs)
