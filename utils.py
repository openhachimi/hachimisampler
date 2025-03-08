import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

def get_mel_fn(
        sr     : float, 
        n_fft  : int, 
        n_mels : int, 
        fmin   : float, 
        fmax   : float, 
        htk    : bool, 
        device : str = 'cpu'
) -> torch.Tensor:
    '''
    Args:
        htk: bool
            Whether to use HTK formula or Slaney formula for mel calculation'
    Returns:
        weights: Tensor [shape = (n_mels, n_fft // 2 + 1)]
    '''
    fmin = torch.tensor(fmin, device=device)
    fmax = torch.tensor(fmax, device=device)
    
    if htk:
        min_mel = 2595.0 * torch.log10(1.0 + fmin / 700.0)
        max_mel = 2595.0 * torch.log10(1.0 + fmax / 700.0)
        mels = torch.linspace(min_mel, max_mel, n_mels + 2, device=device)
        mel_f = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    else:
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz) / f_sp
        logstep = torch.log(torch.tensor(6.4, device=device)) / 27.0

        if fmin >= min_log_hz:
            min_mel = min_log_mel + torch.log(fmin / min_log_hz) / logstep
        else:
            min_mel = (fmin) / f_sp

        if fmax >= min_log_hz:
            max_mel = min_log_mel + torch.log(fmax / min_log_hz) / logstep
        else:
            max_mel = (fmax) / f_sp

        mels = torch.linspace(min_mel, max_mel, n_mels + 2, device=device)
        mel_f = torch.zeros_like(mels)

        log_t = mels >= min_log_mel
        mel_f[~log_t] =f_sp * mels[~log_t]
        mel_f[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    n_mels = int(n_mels)
    N = 1 + n_fft // 2
    weights = torch.zeros((n_mels, N), device=device)
    
    fftfreqs = (sr / n_fft) * torch.arange(0, N, device=device)
    
    fdiff = torch.diff(mel_f)
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)
    
    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    upper = ramps[2:] / fdiff[1:].unsqueeze(1)
    weights = torch.max(torch.tensor(0.0), torch.min(lower, upper))
    
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm.unsqueeze(1)
    
    return weights

def expand_uv(uv):
    uv = uv.astype('float')
    uv = np.min(np.array([uv[:-2],uv[1:-1],uv[2:]]),axis=0)
    uv = np.pad(uv, (1, 1), constant_values=(uv[0], uv[-1]))

    return uv


def norm_f0(f0: np.ndarray, uv=None):
    if uv is None:
        uv = f0 == 0

    f0 = np.log2(f0 + uv)  # avoid arithmetic error
    f0[uv] = -np.inf

    return f0

def denorm_f0(f0: np.ndarray, uv, pitch_padding=None):
    f0 = 2 ** f0

    if uv is not None:
        f0[uv > 0] = 0
        
    if pitch_padding is not None:
        f0[pitch_padding] = 0

    return f0


def interp_f0_spline(f0: np.ndarray, uv=None):
    if uv is None:
        uv = f0 == 0
    f0max = np.max(f0)
    f0 = norm_f0(f0, uv)

    if uv.any() and not uv.all():
        spline = CubicSpline(np.where(~uv)[0], f0[~uv])
        f0[uv] = spline(np.where(uv)[0])

    return np.clip(denorm_f0(f0, uv=None),0,f0max), uv

def interp_f0(f0: np.ndarray, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)

    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

    return denorm_f0(f0, uv=None), uv



if __name__ == '__main__':
    # test
    pass