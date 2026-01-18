"""
Software Name : QASTAnet
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
Authors: Adrien Llave, Grégory Pallone
Software description: An objective metric assessing the global audio quality of 3D audio signals (binaural or (higher-order) ambisonics)
"""

import torch
from torch.nn import Module
import torchaudio.transforms as T
import numpy as np
from utils_psychoac import ERB, get_freq_centers_erb


def sin_window_func(win_len):
    """
    Sinus window a.k.a square-root of hann window
    """
    return torch.sqrt(torch.hann_window(win_len))


class Directiveness(Module):
    def __init__(self, fs=48000, win_len=1920, n_fft=None, hop=None,
                 erb_args={'fmin':315, 'fmax':12500, 'hitme':500},
                 eps=0,
                 ):
        super().__init__()
        self.eps = eps
        self.fs = fs

        if n_fft is None:
            n_fft = win_len
        if hop is None:
            hop = win_len  # for no overlap (otherwise nfft/2)
        self.win_len = win_len
        self.hop = hop
        self.n_fft = n_fft
        # Layers
        self.stft = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_len,
            window_fn=sin_window_func,
            power=None,  # None for complex
            center=False,
            )

        diffsmooth_cycles = 20
        diffsmooth_limf = 3000
        alpha_dif = self._get_nrj_iir_coef(diffsmooth_cycles, diffsmooth_limf)
        self.register_buffer("alpha_dif", alpha_dif)
        #
        freq_2_erb_m = self.get_freq_erb_grouping_matrix(**erb_args)
        self.register_buffer("freq_2_erb_m", freq_2_erb_m)

    @property
    def n_freq(self):
        return self.n_fft // 2 + 1

    def get_frequency_limits(self, freq_center, fmin, fmax):

        freq_lim = torch.zeros(len(freq_center) + 1)
        freq_lim[0] = fmin
        freq_lim[-1] = fmax

        for i in range(1, len(freq_center)):
            freq_lim[i] = int(
                freq_lim[i - 1]
                + ERB(freq_center[i - 1]) / 2
                + ERB(freq_center[i]) / 2
            )

        return freq_lim

    def get_freq_erb_grouping_matrix(self, fmin, fmax, hitme):
        freq_v = torch.linspace(0, self.fs / 2, steps=self.n_freq)
        self.freq_centers = get_freq_centers_erb(fmin, fmax, hitme)
        f_ERB = self.get_frequency_limits(self.freq_centers, fmin, fmax)

        freq_2_erb_m = torch.zeros((len(self.freq_centers), self.n_freq))
        for i_band in range(len(self.freq_centers)):
            frqs_in_band = (freq_v > f_ERB[i_band]) * (freq_v < f_ERB[i_band+1])
            freq_2_erb_m[i_band, :] = frqs_in_band / torch.sum(frqs_in_band)

        return freq_2_erb_m

    def freq_to_erb(self, x):
        x_erb = torch.einsum(
            "kf,bdfl->bdkl", self.freq_2_erb_m, x
        )  # (n_band, n_freq) and (bs, n_ls, n_freq, n_frm)
        return x_erb

    def _get_nrj_iir_coef(self, diffsmooth_cycles, diffsmooth_limf):
        freq = np.linspace(0, self.fs / 2, self.n_freq)
        freq[0] = freq[1]  # omit infinity value for DC
        period = 1 / freq
        # diffuseness smoothing time constant in sec
        tau_dif = period * diffsmooth_cycles
        # diffuseness smoothing recursive coefficient
        alpha_dif = np.exp(-1 / (tau_dif * (self.fs / self.hop)))
        # limit recursive coefficient
        alpha_dif[freq > diffsmooth_limf] = np.min(alpha_dif[freq <= diffsmooth_limf])
        return torch.tensor(alpha_dif)

    def forward(self, pred, targ):
        pred_directiveness = self.get_directiveness(pred)
        targ_directiveness = self.get_directiveness(targ)

        res_erb = self.freq_to_erb(
            torch.stack((pred_directiveness, targ_directiveness), dim=1)
        )

        return res_erb

    def get_directiveness(self, inp):
        inp_tf = self.stft(inp) / self.win_len
        w_tf = inp_tf[:, 0, :, :]
        X = inp_tf[:, 1:4, :, :] * torch.sqrt(torch.tensor(3 / 2))
        # -- get DoA as cartesian coordinates
        # get intensity (x,y,z coords) dipoles /cancel B-format dipole convention
        intensity = torch.real(w_tf.unsqueeze(1).conj() * X)
        # -- get directiveness
        Intensity_norm = torch.sqrt(torch.sum(intensity**2, axis=1) + self.eps)
        energy = (torch.abs(w_tf) ** 2 + torch.sum(torch.abs(X) ** 2, axis=1)) / 2
        # energy smoothing
        res_smooth = self.smooth_dif(torch.stack((Intensity_norm, energy), dim=1))
        intensity_dif_smooth, energy_dif_smooth = (
            res_smooth[:, 0, ...],
            res_smooth[:, 1, ...],
        )
        # compute directiveness
        directiveness = intensity_dif_smooth / (energy_dif_smooth + self.eps)
        directiveness = torch.clamp(directiveness, 1e-3, 1 - 1e-3)
        return directiveness

    def smooth_dif(self, x):
        """
        x [B, C, F, T]
        """
        x_filt = torch.zeros_like(x)
        alpha_dif = self.alpha_dif.unsqueeze(0).unsqueeze(0)
        for i_frm in range(1, x_filt.shape[-1]):
            x_filt[..., i_frm] = (
                alpha_dif * x_filt[..., i_frm - 1] + (1 - alpha_dif) * x[..., i_frm]
            )
        return x_filt
