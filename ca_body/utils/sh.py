import math
from typing import Tuple

import numpy as np
import torch as th

def factratio(N: int, D: int) -> float:
    if N >= D:
        prod = 1.0
        for i in range(D+1, N+1):
            prod *= i
        return prod
    else:
        prod = 1.0
        for i in range(N+1, D+1):
            prod *= i
        return 1.0 / prod

def KVal(M: int, L: int) -> float:
    return math.sqrt(((2 * L + 1) / (4 * math.pi)) * (factratio(L - M, L + M)))

def AssociatedLegendre(M: int, L: int, x: np.ndarray) -> np.ndarray:
    if M < 0 or M > L or np.max(np.abs(x)) > 1.0:
        return np.zeros_like(x)
    
    pmm = np.ones_like(x)
    if M > 0:
        somx2 = np.sqrt((1.0 + x) * (1.0 - x))
        fact = 1.0
        for i in range(1, M+1):
            pmm = -pmm * fact * somx2
            fact = fact + 2
    
    if L == M:
        return pmm
    else:
        pmmp1 = x * (2 * M + 1) * pmm
        if L == M+1:
            return pmmp1
        else:
            pll = np.zeros_like(x)
            for i in range(M+2, L+1):
                pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                pmm = pmmp1
                pmmp1 = pll
            return pll

def AssociatedLegendreTorch(M: int, L: int, x: th.Tensor) -> th.Tensor:
    if M < 0 or M > L or th.max(th.abs(x)) > 1.0:
        return th.zeros_like(x)
    
    pmm = th.ones_like(x)
    if M > 0:
        somx2 = th.sqrt(((1.0 + x) * (1.0 - x)).clamp(min=1e-8))
        fact = 1.0
        for i in range(1, M+1):
            pmm = -pmm * fact * somx2
            fact = fact + 2
    
    if L == M:
        return pmm
    else:
        pmmp1 = x * (2 * M + 1) * pmm
        if L == M+1:
            return pmmp1
        else:
            pll = th.zeros_like(x)
            for i in range(M+2, L+1):
                pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
                pmm = pmmp1
                pmmp1 = pll
            return pll

def SphericalHarmonicTorch(M: int, L: int, theta: th.Tensor, phi: th.Tensor) -> th.Tensor:
    if M > 0:
        return math.sqrt(2.0) * KVal(M, L) * th.cos(M * phi) * AssociatedLegendreTorch(M, L, th.cos(theta))
    elif M < 0:
        return math.sqrt(2.0) * KVal(-M, L) * th.sin(-M * phi) * AssociatedLegendreTorch(-M, L, th.cos(theta))
    else:
        return KVal(0, L) * AssociatedLegendreTorch(0, L, th.cos(theta))


def SphericalHarmonic(M: int, L: int, theta: th.Tensor, phi: th.Tensor) -> th.Tensor:
    if M > 0:
        return math.sqrt(2.0) * KVal(M, L) * np.cos(M * phi) * AssociatedLegendre(M, L, np.cos(theta))
    elif M < 0:
        return math.sqrt(2.0) * KVal(-M, L) * np.sin(-M * phi) * AssociatedLegendre(-M, L, np.cos(theta))
    else:
        return KVal(0, L) * AssociatedLegendre(0, L, np.cos(theta))
    
def dir2angle(dir: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    # dir: [..., 3]
    theta = th.acos(dir[..., 2].clamp(-1.0, 1.0)) # prevent nan
    idx = theta.reshape(-1, 1).max(0)[1]
    phi = th.atan2(dir[..., 1], dir[..., 0])
    
    return theta, phi

def dir2sh(dir: th.Tensor, deg: int) -> th.Tensor:
    theta, phi = dir2angle(dir)
    theta = theta.numpy()
    phi = phi.numpy()
    
    shs = []
    for n in range(0, deg+1):
        for m in range(-n,n+1):
            s = SphericalHarmonic(m, n, theta, phi)
            shs.append(s)
    
    return th.as_tensor(np.stack(shs, -1))

def dir2sh_torch(deg: int, dirs: th.Tensor) -> th.Tensor:
    theta, phi = dir2angle(dirs)
    
    shs = []
    for n in range(0, deg+1):
        for m in range(-n,n+1):
            s = SphericalHarmonicTorch(m, n, theta, phi)
            shs.append(s)
    
    return th.stack(shs, dim=-1)
    
def eval_sh(deg: int, sh: th.Tensor, dirs: th.Tensor) -> th.Tensor:
    theta, phi = dir2angle(dirs)
    
    val = None    
    index = 0
    for n in range(0, deg+1):
        for m in range(-n,n+1):
            s = SphericalHarmonicTorch(m, n, theta, phi)
            if val is None:
                val = sh[..., index] * s[..., None]
            else:
                val += sh[..., index] * s[..., None]
            index += 1
    
    return val
