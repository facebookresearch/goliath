import sys
import os

# set the right device
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# NOTE: assuming we are in `ca_body/scripts`
sys.path.insert(0, '../')
from attrdict import AttrDict

from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

import torch as th
from ca_body.utils.module_loader import load_from_config
from ca_body.utils.lbs import LBSModule
from ca_body.utils.train import load_checkpoint, to_device
from ca_body.utils.image import linear2srgb

from ca_body.utils.dataloader import BodyDataset, collate_fn
from torch.utils.data import DataLoader

from ca_body.utils.light_decorator import EnvSpinDecorator, SingleLightCycleDecorator
from ca_body.utils.envmap import envmap_to_image, envmap_to_mirrorball

from tqdm import tqdm

device = th.device('cuda:0')

# NOTE: assuming we are in `ca_body/scripts`
model_dirs = os.listdir("../runs")
os.makedirs("tmp", exist_ok=True)

for model_dir in model_dirs:
    ckpt_path = f'../runs/{model_dir}/checkpoints/model.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = f'../runs/{model_dir}/checkpoints/latest.pt'
    config_path = f'../runs/{model_dir}/config.yml'

    # config
    config = OmegaConf.load(config_path)

    config.data.shuffle = False
    config.data.split = "test"
    config.data.fully_lit_only = True

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
    model = load_from_config(
        config.model, 
        assets=static_assets,
    ).to(device).eval()

    # loading model checkpoint
    load_checkpoint(
        ckpt_path, 
        modules={'model': model},
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
            preds = model_p(**batch, index=[i])    
                
        # visualizing
        rgb_preds_grid = make_grid(linear2srgb(preds['rgb']), nrow=4)
        save_image(rgb_preds_grid, f"tmp/{i}.png")
        
        if i > 256:
            break
        
    os.system(f"ffmpeg -y -framerate 30 -i 'tmp/%d.png' -c:v libx264 -g 10 -pix_fmt yuv420p {model_dir}_point.mp4 -y")
    
    model_e = EnvSpinDecorator(model, envmap_path="/mnt/captures/saibi/data/envmaps/hdrs/0.hdr", ydown=True).to(device)

    # forward
    for i, batch in tqdm(enumerate(loader)):
        batch = to_device(batch, device)
        batch_filter_fn(batch)
        with th.no_grad():
            preds = model_e(**batch, index=[i])    
                
        # visualizing
        rgb_preds_grid = make_grid(linear2srgb(preds['rgb']), nrow=4)
        save_image(rgb_preds_grid, f"tmp/{i}.png")
        
        if i > 256:
            break
        
    os.system(f"ffmpeg -y -framerate 30 -i 'tmp/%d.png' -c:v libx264 -g 10 -pix_fmt yuv420p {model_dir}_env.mp4 -y")
