from typing import List, Optional, Tuple, Union

import torch as th
import torch.nn.functional as thf


def mipmap_grid_sample(
    input: Union[List[th.Tensor], th.Tensor],
    grid: th.Tensor,
    mipmap_level: th.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: Optional[bool] = False,
) -> th.Tensor:
    if isinstance(input, th.Tensor):
        input = [input]
    q = len(input)
    
    with th.no_grad():
        # Given the max and min gradients, select mipmap levels (assumes linear interpolation
        # between mipmaps)
        lambda_ = mipmap_level
        lambda_ = th.clamp(lambda_, min=0, max=q - 1 - 1e-6)
        d1 = th.floor(lambda_).long()
        a = lambda_ - d1.float()

    result = []
    for level in input:
        r = thf.grid_sample(
            level,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        result.append(r)

    return combine_sampled_mipmaps(result, d1, a)


def combine_sampled_mipmaps(
    sampled_mipmaps: Union[List[th.Tensor], th.Tensor], d1: th.Tensor, a: th.Tensor
) -> th.Tensor:
    if isinstance(sampled_mipmaps, th.Tensor):
        return sampled_mipmaps

    if len(sampled_mipmaps) == 1:
        return sampled_mipmaps[0]
    sampled_mipmaps = th.stack(sampled_mipmaps, dim=0)
    indices = th.cat([d1[None, :, None], d1[None, :, None] + 1], dim=0)
    samples = th.gather(
        sampled_mipmaps,
        dim=0,
        index=indices.expand(
            -1,
            sampled_mipmaps.shape[1],
            sampled_mipmaps.shape[2],
            -1,
            -1
        ),
    )
    # Interpolate two nearest mipmaps. See p.266
    return th.lerp(samples[0], samples[1], a[:, None])
