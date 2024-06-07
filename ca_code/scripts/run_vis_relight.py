import os
import sys

import torch as th
from addict import Dict as AttrDict

from ca_code.utils.dataloader import BodyDataset, collate_fn
from ca_code.utils.envmap import envmap_to_image, envmap_to_mirrorball
from ca_code.utils.image import linear2srgb
from ca_code.utils.lbs import LBSModule

from ca_code.utils.light_decorator import EnvSpinDecorator, SingleLightCycleDecorator
from ca_code.utils.module_loader import load_from_config
from ca_code.utils.train import load_checkpoint, to_device

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from tqdm import tqdm


def main(config: DictConfig):
    device = th.device("cuda:0")

    model_dir = config.train.run_dir
    os.makedirs("tmp", exist_ok=True)

    ckpt_path = f"runs/{model_dir}/checkpoints/model.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"runs/{model_dir}/checkpoints/latest.pt"
    config_path = f"runs/{model_dir}/config.yml"

    # config
    config = OmegaConf.load(config_path)

    config.data.shuffle = False
    config.data.split = "test"
    config.data.fully_lit_only = True
    config.data.partially_lit_only = False

    dataset = BodyDataset(**config.data)
    batch_filter_fn = dataset.batch_filter

    static_assets = AttrDict(dataset.static_assets)

    config.dataloader.batch_size = 1
    config.dataloader.num_workers = 0

    dataset.cameras = ["401892"]

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        **config.dataloader,
    )

    # building the model
    model = (
        load_from_config(
            config.model,
            assets=static_assets,
        )
        .to(device)
        .eval()
    )

    # loading model checkpoint
    load_checkpoint(
        ckpt_path,
        modules={"model": model},
    )

    # disabling training-only stuff
    model.learn_blur_enabled = False
    model.cal_enabled = False

    model_p = SingleLightCycleDecorator(model, light_rotate_axis=1).to(device)

    # forward
    for i, batch in tqdm(enumerate(loader)):
        batch = to_device(batch, device)
        batch_filter_fn(batch)
        with th.no_grad():
            preds = model_p(**batch, index=[180 + i])

        # visualizing
        rgb_preds_grid = make_grid(linear2srgb(preds["rgb"]), nrow=4)
        save_image(rgb_preds_grid, f"tmp/{i}.png")

        if i > 256:
            break

    os.system(
        f"ffmpeg -y -framerate 30 -i 'tmp/%d.png' -c:v libx264 -g 10 -pix_fmt yuv420p {model_dir}_elem.mp4 -y"
    )

    # download 1k hdr from https://polyhaven.com/a/symmetrical_garden_02
    model_e = EnvSpinDecorator(
        model,
        envmap_path="./symmetrical_garden_02_1k.hdr",
        ydown=True,
        env_scale=18.0,
    ).to(device)

    # forward
    for i, batch in tqdm(enumerate(loader)):
        batch = to_device(batch, device)
        batch_filter_fn(batch)
        with th.no_grad():
            preds = model_e(**batch, index=[i])

        # visualizing
        rgb_preds_grid = make_grid(linear2srgb(preds["rgb"]), nrow=4)
        save_image(rgb_preds_grid, f"tmp/{i}.png")

        if i > 256:
            break

    os.system(
        f"ffmpeg -y -framerate 30 -i 'tmp/%d.png' -c:v libx264 -g 10 -pix_fmt yuv420p {model_dir}_env.mp4 -y"
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
