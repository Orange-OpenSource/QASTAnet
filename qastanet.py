"""
Software Name : QASTAnet
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
Authors: Adrien Llave, Grégory Pallone
Software description: An objective metric assessing the global audio quality of 3D audio signals (binaural or (higher-order) ambisonics)
"""

import os
import time
import importlib.util
import numpy as np
from scipy.signal import butter, lfilter
import pyfar as pf
import sofar
import torch
import torchaudio
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from layers import Max, SWAP, PNorm
from utils_psychoac import iso389_7, hilbert_transform, get_freq_centers_erb
from hoa_features import Directiveness


class HOAToBinaural(Module):
    """
    Decode ambisonic signals to binaural.
    Ambisonic signals are assumed to be in ACN SN3D convention
    """

    def __init__(self, hoa2bin_fn, fs=48000):
        super().__init__()
        self.fs = fs
        sofa = sofar.read_sofa(hoa2bin_fn)
        hoa2bin_t = sofa.Data_IR  # (n_mes, l/r, n_tap, n_ch)
        hoa2bin_t = np.moveaxis(hoa2bin_t[0, ...], 1, 2)  # (l/r, n_ch, n_tap)
        assert self.fs == sofa.Data_SamplingRate, "HOAToBinaural: sample rate mismatch."
        self.register_buffer("hoa2bin_t", torch.from_numpy(hoa2bin_t), persistent=False)

    def forward(self, hoa):
        batchsize, n_ch, n_smp = hoa.shape
        _, n_ch_filt, n_tap = self.hoa2bin_t.shape
        assert n_ch_filt == n_ch, f"Number of channels mistmatch, {n_ch_filt}, {n_ch}"
        bino = torch.sum(
            torchaudio.functional.fftconvolve(
                hoa.unsqueeze(1), self.hoa2bin_t.unsqueeze(0)
            )[..., : -n_tap + 1],
            dim=2,
        )

        return bino  # [batchsize, left/right, timesteps]


class QASTAnet(Module):
    def __init__(
        self,
        fs=48000,
        frame_len=40,  # ms
        pooling_layer="SWAP",
        feature_names=["ild", "gam", "mon", "dir"],
        ambi2bin_paths="resources/ambi2bin/HRIR_128_Meth5_OLPS_2001_48000_HOA3.sofa",  # str or list of str
        gammatone_fir_fn=os.path.join("resources", "filters", "filter_coeffs_gammatone_48000_2048.py"),
        lowpass_fir_fn=os.path.join("resources", "filters", "filter_coeffs_butter_48000_512.py"),
        checkpoint_fn=None,
        VERBOSE=False,
        filter_mode="IIR",  # 'IIR' or 'FIR_AUTOGRAD'
    ):
        """ """
        super(QASTAnet, self).__init__()
        self.fs = fs
        self.checkpoint_fn = checkpoint_fn
        self.VERBOSE = VERBOSE
        self.filter_mode = filter_mode
        # Ambisonic to binaural filters
        self._load_ambi2bin(ambi2bin_paths)
        # Original parameters from mpar
        self.filters_per_ERBaud = 1  # filters per ERB
        self.bwfactor = 1
        self.flow = 315  # Hz
        self.fbase = 500  # one filter will be centered here --> fc
        self.fhigh = 12500  # just central channel for now
        self.gtorder = 4
        self.env_lowpass_fc = 150  # Hz
        self.env_lowpass_n = 1
        self.mso_rolloff = 1300
        self.frame_len = frame_len

        # Load limit_threshold
        center_frequencies_hz = get_freq_centers_erb(
            self.flow, self.fhigh, self.fbase, self.bwfactor
        )
        hearing_threshold = iso389_7(center_frequencies_hz)
        limit_threshold = 1e-10 * 10 ** (hearing_threshold / 10)
        self.register_buffer(
            "center_frequencies_hz", torch.from_numpy(center_frequencies_hz)
        )
        self.register_buffer("limit_threshold", torch.from_numpy(limit_threshold))

        # for filter_mode 'IIR' # GP: on fait pas un if filter_mode?
        # Load lowpass filter
        self.lowpass_coefs = butter(
            self.env_lowpass_n, 2 * self.env_lowpass_fc / self.fs
        )
        # Gammatone filterbank
        self.GFB = pf.dsp.filter.GammatoneBands(
            [self.flow, self.fhigh],
            reference_frequency=self.fbase,
            sampling_rate=fs,
        )

        # for filter_mode 'FIR' # GP: on fait pas un if filter_mode?
        # Load gammatone filters
        spec = importlib.util.spec_from_file_location("gammatone", gammatone_fir_fn)
        gammatone = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gammatone)
        assert gammatone.ir_fs == fs  # Only 48kHz supported
        gammatone_real = np.array(gammatone.ir_real_list, dtype=np.float32)
        gammatone_imag = np.array(gammatone.ir_imag_list, dtype=np.float32)
        #  Load lowpass filter
        spec = importlib.util.spec_from_file_location("lowpass", lowpass_fir_fn)
        lowpass = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lowpass)
        assert lowpass.ir_fs == fs  # Only 48kHz supported
        lowpass = np.float32(np.array(lowpass.ir_list))
        self.register_buffer("gammatone_real", torch.from_numpy(gammatone_real))
        self.register_buffer("gammatone_imag", torch.from_numpy(gammatone_imag))
        self.register_buffer("lowpass", torch.from_numpy(lowpass))

        self.fb = self.gammatone_real.shape[0]  # Number of frequency bands (29)
        self.gt = self.gammatone_real.shape[1]  # Length of GammaTone filter
        self.lp = self.lowpass.shape[0]  # Length of LowPass filter
        self.lr = 2  # Left/right
        self.rd = 2  # Ref/Degraded

        self.feature_names = feature_names
        self.n_feat = len(self.feature_names)

        # Score estimation DNN
        self.nan_num_replace = 0
        self.freq_weighting = nn.Parameter(torch.ones(1, self.n_feat, 1, self.fb, 1))
        self.conv1 = nn.Conv3d(self.n_feat, 16, kernel_size=1)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=1)
        self.conv3 = nn.Conv3d(16, 6, kernel_size=1)
        self.conv_avg_freq = nn.Conv3d(self.fb, 1, kernel_size=1)
        if pooling_layer == "Max":
            self.temp_pooling = Max()
        elif pooling_layer == "SWAP":
            self.temp_pooling = SWAP()
        elif pooling_layer == "PNorm":
            self.temp_pooling = PNorm()
        else:
            print("Argument given does not correspond to a pooling layer.")
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 1)

        #
        if checkpoint_fn is not None:
            checkpoint = torch.load(checkpoint_fn, map_location="cpu")
            self.load_state_dict(checkpoint["state_dict"])

    def _load_ambi2bin(self, ambi2bin_paths):
        """
        Define ambisonic to binaural processors instances
        """
        if isinstance(ambi2bin_paths, str):
            ambi2bin_paths = [
                ambi2bin_paths,
            ]
        self.n_heads = len(ambi2bin_paths)
        self.ambi2bin_procs = []
        for ambi2bin_path in ambi2bin_paths:
            self.ambi2bin_procs.append(HOAToBinaural(ambi2bin_path, fs=self.fs))
            assert self.ambi2bin_procs[-1].fs == self.fs, (
                f"Sampling frequency mismatch between {self.fs} target sampling rate and from the file {ambi2bin_path}, {self.ambi2bin_procs[-1].fs}"
            )

        return

    def forward(self, deg, ref):
        """
        Parameters:
            deg [batch, n_channels, timesteps]
            ref [batch, n_channels, timesteps]
        """
        assert len(deg.shape) == len(ref.shape) == 3, (
            "Input tensors are assumed to be 3D."
        )
        assert deg.shape == ref.shape, (
            "Input tensors are assumed to be of the same shape."
        )

        mode = "HOA" if deg.shape[1] != 2 else "binaural"
        with torch.no_grad():
            # Low-level features extraction stage
            if mode == "HOA":
                t = time.time()
                print("Compute HOA2Bin...") if self.VERBOSE else None
                deg_bin = self.ambi2bin(deg)
                ref_bin = self.ambi2bin(ref)
                print(f"   HOA2Bin: {time.time() - t:.2f}s") if self.VERBOSE else None

            # Compute binaural features
            print("Compute binaural features...") if self.VERBOSE else None
            t = time.time()
            bs, hd, lr, sp = ref_bin.shape
            all_features_dict = self.binaural_features(deg_bin, ref_bin)
            print(
                f"   Total binaural features: {time.time() - t:.2f}s"
            ) if self.VERBOSE else None

            if mode == "HOA":
                print("Compute HOA features...") if self.VERBOSE else None
                t = time.time()
                hoa_features = self.hoa_features(deg, ref)
                all_features_dict.update(hoa_features)
                print(
                    f"   HOA features: {time.time() - t:.2f}s"
                ) if self.VERBOSE else None
            print("Comparison of low-level features...") if self.VERBOSE else None
            t = time.time()
            comp_features_dict = self.features_comparison(all_features_dict)
            print(
                f"   Comparison features: {time.time() - t:.2f}s"
            ) if self.VERBOSE else None

        # Unflatten batch_size/num_heads dimensions for binaural features
        comp_features = []
        for feature_name, feature_tensor in comp_features_dict.items():
            if feature_name in ["ild", "gam", "mon"]:
                feature_tensor = feature_tensor.unflatten(0, (bs, hd))
                comp_features_dict[feature_name] = feature_tensor
            else:
                bs, fb, fm = feature_tensor.shape
                comp_features_dict[feature_name] = feature_tensor.unsqueeze(1).expand(
                    bs, self.n_heads, fb, fm
                )
            comp_features.append(comp_features_dict[feature_name])
        comp_features = torch.stack(comp_features, dim=1)

        # Pre-processing low-level features before feeding the neural network
        comp_features = comp_features.nan_to_num(
            self.nan_num_replace
        )  # replace nan by 0
        comp_features = comp_features.to(torch.float32)
        # High-level features extraction + time pooling + decision stages
        print(
            "Compute score estimation with neural network..."
        ) if self.VERBOSE else None
        t = time.time()
        score = self.score_estimation(comp_features)
        print(
            f"   score estimation (DNN): {time.time() - t:.2f}s"
        ) if self.VERBOSE else None

        return score

    def ambi2bin(self, x_hoa):
        """
        Render (higher-order) ambisonic signals in binaural with possibly various filters from different heads.

        Parameters
        ----
            x_hoa: [batchsize, n_ch, timesteps]
        """
        x_bins = []
        for ambi2bin_proc in self.ambi2bin_procs:
            x_bin = ambi2bin_proc(x_hoa)
            x_bins.append(x_bin)
        x_bins = torch.stack(x_bins, dim=1)
        return x_bins  # [batchsize, n_heads, 2, timesteps]

    def binaural_features(self, deg_bin, ref_bin):
        """
        Binaural low-level features extraction (ILD, Gamma, and monaural power envelop).
        For more detailed information see Eurich et al. (2024).

        Parameters:
        deg_bin :
            Degraded signals
        ref_bin :
            Reference signals

        Returns:
        gamma : torch.Tensor
            Complex interaural correlation coefficient gamma (binaural)
        ild : torch.Tensor
            Interaural level differences (binaural)
        snr : torch.Tensor
            Spectral coloration (monaural)
        """
        # Stack reference and degraded
        deg_ref = torch.stack((ref_bin, deg_bin), dim=2)  # [bs, n_heads, rd, lr, sp]
        # Flatten batch_size/num_heads dimensions
        deg_ref = deg_ref.flatten(0, 1)

        # tensor # [batch, ref/deg, left/right, samples]  # [bs rd lr sp]
        sp = deg_ref.shape[-1]  # Num of samples in signals
        assert deg_ref.shape[-2] == self.lr  # Signals shall be two-channels
        assert deg_ref.shape[-3] == self.rd  # ref/deg shall be present
        bs = deg_ref.shape[-4]  # batch size

        fl = int(self.frame_len / 1000 * self.fs)  # frame length in samples
        nf = int(sp / fl)  # num frames

        # ==== GAMMATONE FILTERBANK ====
        t = time.time()
        if self.filter_mode == "FIR_AUTOGRAD":
            # FIR implementation, differentiable
            x = deg_ref.unsqueeze(-2)  # [bs rd lr 1 sp]
            kernel_real = (
                self.gammatone_real.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )  # [1 1 1 fb sp]
            kernel_imag = (
                self.gammatone_imag.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )  # [1 1 1 fb sp]
            kl = kernel_real.shape[-1]
            mPeriphFiltSig = torch.complex(
                torchaudio.functional.fftconvolve(x, kernel_real)[
                    ..., : -kl + 1
                ],  # [bs, rd, lr, fb, sp]
                torchaudio.functional.fftconvolve(x, kernel_imag)[
                    ..., : -kl + 1
                ],  # [bs, rd, lr, fb, sp]
            ).to(torch.complex64)
            del kernel_real, kernel_imag
            torch.cuda.empty_cache()
        elif self.filter_mode == "IIR":
            # IIR implementation, not differentiable
            x = pf.Signal(deg_ref, self.fs)  # [bs rd lr sp]
            real, imag = self.GFB.process(x)
            mPeriphFiltSig = torch.complex(
                torch.tensor(np.moveaxis(real.time, 0, -2), dtype=torch.float32),
                torch.tensor(np.moveaxis(imag.time, 0, -2), dtype=torch.float32),
            )
            del real, imag
        else:
            print(
                f"QASTAnet::binaural_features: ERROR, unknown filter_mode: {self.filter_mode}"
            )
        print(f"   GAMMATONE: {time.time() - t:.2f}s") if self.VERBOSE else None

        # ENVELOPE extraction
        t = time.time()
        complex_envelope = hilbert_transform(
            torch.abs(mPeriphFiltSig)
        )  # [bs rd lr fb sp]
        torch.cuda.empty_cache()
        print(f"   ENVELOPE: {time.time() - t:.2f}s") if self.VERBOSE else None

        # LOWPASS filtering
        t = time.time()
        if self.filter_mode == "FIR_AUTOGRAD":
            kernel = (
                self.lowpass.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )  # [1 1 1 1 sp]
            kl = kernel.shape[-1]
            filtered_complex_envelope = torch.zeros_like(complex_envelope)
            filtered_complex_envelope.real = torchaudio.functional.fftconvolve(
                torch.real(complex_envelope), kernel
            )[..., : -kl + 1].to(torch.float32)  # [bs rd lr fb sp]
            filtered_complex_envelope.imag = torchaudio.functional.fftconvolve(
                torch.imag(complex_envelope), kernel
            )[..., : -kl + 1].to(torch.float32)  # [bs rd lr fb sp]
            del kernel, complex_envelope
            torch.cuda.empty_cache()
        elif self.filter_mode == "IIR":
            filtered_complex_envelope = lfilter(
                *self.lowpass_coefs, complex_envelope, axis=-1
            )
            filtered_complex_envelope = torch.tensor(
                filtered_complex_envelope, dtype=torch.complex64
            )
        else:
            print(
                f"QASTAnet::binaural_features: ERROR, unknown filter_mode: {self.filter_mode}"
            )
        print(f"   LOWPASS: {time.time() - t:.2f}s") if self.VERBOSE else None

        t = time.time()
        # Frame vectorization
        filtered_complex_envelope = filtered_complex_envelope[
            ..., : nf * fl
        ]  # truncation to integer num of frames
        filtered_complex_envelope = torch.reshape(
            filtered_complex_envelope, (bs, self.rd, self.lr, self.fb, nf, fl)
        )  # [bs rd lr fb nf fl]

        he = filtered_complex_envelope.real / torch.sqrt(
            torch.tensor(2)
        )  # [bs rd lr fb nf fl]

        he_cat = torch.reshape(
            he.permute(0, 1, 3, 4, 5, 2), ((bs, self.rd, self.fb, nf, self.lr * fl))
        )  # [bs rd fb nf lr*fl]
        Power = torch.mean(he_cat, axis=-1) ** 2 / 2  # [bs rd fb nf]
        del he_cat
        limit_expanded = self.limit_threshold.view(1, 1, -1, 1)
        Power = torch.where(
            Power <= limit_expanded, limit_expanded, Power
        )  # [bs rd fb nf]
        print(f"   POWER: {time.time() - t:.2f}s") if self.VERBOSE else None

        # -- Feature d_ild
        Power_ild = torch.mean(he, axis=-1) ** 2  # [bs rd lr fb nf]
        del he
        mild = 20 * torch.log10(
            Power_ild[:, :, 0, :, :] / Power_ild[:, :, 1, :, :]
        )  # [bs rd fb nf]

        # Feature d_gamma
        mPeriphFiltSig = mPeriphFiltSig[
            ..., : nf * fl
        ]  # truncation to integer num of frames
        mPeriphFiltSig = torch.reshape(
            mPeriphFiltSig, (bs, self.rd, self.lr, self.fb, nf, fl)
        )  # [bs rd lr fb nf fl]

        t = time.time()
        gamma_mso = torch.mean(
            mPeriphFiltSig[:, :, 0, :, :, :]
            * torch.conj(mPeriphFiltSig[:, :, 1, :, :, :]),
            axis=-1,
        ) / torch.sqrt(
            torch.mean(torch.abs(mPeriphFiltSig[:, :, 0, :, :, :]) ** 2, axis=-1)
            * torch.mean(torch.abs(mPeriphFiltSig[:, :, 1, :, :, :]) ** 2, axis=-1)
        )
        del mPeriphFiltSig
        print(f"   gamma_mso: {time.time() - t:.2f}s") if self.VERBOSE else None

        t = time.time()
        gamma_lso = torch.mean(
            filtered_complex_envelope[:, :, 0, :, :, :]
            * torch.conj(filtered_complex_envelope[:, :, 1, :, :, :]),
            axis=-1,
        ) / torch.sqrt(
            torch.mean(
                torch.abs(filtered_complex_envelope[:, :, 0, :, :, :]) ** 2, axis=-1
            )
            * torch.mean(
                torch.abs(filtered_complex_envelope[:, :, 1, :, :, :]) ** 2, axis=-1
            )
        )
        del filtered_complex_envelope
        print(f"   gamma_lso: {time.time() - t:.2f}s") if self.VERBOSE else None

        # Combined gamma feature: TFS for <1300 Hz, Envelope for >1300Hz
        channels_below_rolloff = self.center_frequencies_hz < self.mso_rolloff
        gamma = torch.zeros_like(gamma_mso, dtype=torch.complex64)  # [bs rd fb nf]
        mask = torch.zeros_like(gamma, dtype=torch.bool)
        mask[:, :, channels_below_rolloff, :] = True
        gamma[mask] = gamma_mso[mask]
        mask = torch.zeros_like(gamma, dtype=torch.bool)
        mask[:, :, ~channels_below_rolloff, :] = True
        gamma[mask] = gamma_lso[mask]

        mBP = torch.abs(gamma) * torch.exp(1j * torch.angle(gamma))  # [bs rd fb nf]
        mBP = torch.where(Power <= limit_expanded, float("nan"), mBP)  # [bs rd fb nf]

        output = {
            "ild": mild,
            "gamma": mBP,
            "Power": Power,
        }
        return output

    def hoa_features(self, deg, ref):
        features = {}
        if "dir" in self.feature_names:
            win_len = int(self.frame_len / 1000 * self.fs)
            directiveness = Directiveness(fs=self.fs, win_len=win_len)
            dir_diff_t = directiveness(deg, ref)
            features["dir"] = dir_diff_t

        return features

    def features_comparison(self, all_features):
        comp_features = {}
        for feature_name in self.feature_names:
            if feature_name == "ild":
                mild = all_features["ild"]
                Power = all_features["Power"]
                # -- ILD Difference ref/deg:
                d_ild = torch.abs(mild[:, 1, :, :] - mild[:, 0, :, :])  # [bs fb nf]
                # apply hearing threshold masking
                limit_expanded = self.limit_threshold.view(1, 1, -1, 1)
                d_ild = torch.where(
                    Power[:, 0, :, :] <= limit_expanded[:, 0, ...], float("nan"), d_ild
                )
                comp_features["d_ild"] = d_ild

            if feature_name == "gam":
                # -- Gamma Difference ref/deg:
                mBP = all_features["gamma"]
                d_gamma = torch.abs(mBP[:, 1, :, :] - mBP[:, 0, :, :])  # [bs fb nf]
                comp_features["d_gamma"] = d_gamma

            if feature_name == "mon":
                # -- monaural enveloppe difference, rewriting of eMoBi-Q
                # 'Normalized' difference ref/deg:
                d_snr = (
                    torch.abs(Power[:, 1, ...] - Power[:, 0, ...])
                    / torch.min(Power, dim=1)[0]
                )  # [bs fb nf]
                d_snr = torch.clamp(d_snr, None, 20) / 2
                comp_features["d_snr"] = d_snr

            if feature_name == "pml":
                pml = all_features["pml"]  # [bs, rd, ls, fb, fm]
                d_pm = torch.abs(pml[:, 0, ...] - pml[:, 1, ...]) ** 2
                d_pm = torch.max(d_pm, dim=1)[0]  # reduction in loudspeaker dim
                comp_features["d_pm"] = d_pm

            if feature_name == "dir":
                dir_erb = all_features["dir"]  # [bs, rd, fb, fm]
                d_dir = torch.abs(dir_erb[:, 0, ...] - dir_erb[:, 1, ...]) ** 2
                # apply hearing threshold masking
                limit_expanded = self.limit_threshold.view(1, 1, -1, 1)
                d_dir = torch.where(
                    Power[:, 0, :, :] <= limit_expanded[:, 0, ...], float("nan"), d_dir
                )
                comp_features["d_dir"] = d_dir

        return comp_features

    def score_estimation(self, x):
        """
        Parameters:
        ----
            x: [batchsize, n_features, n_heads, n_fbands, n_frames]
        """
        x = x * self.freq_weighting
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))  # b x 16 x n_heads x n_fbands x n_frames
        x = F.leaky_relu(self.conv3(x))  # b x  6 x n_heads x n_fbands x n_frames
        x = x.permute(0, 3, 1, 2, 4)  # b x n_fbands x 6 x n_heads x n_frames
        x = F.leaky_relu(self.conv_avg_freq(x))
        x = torch.squeeze(x, dim=1)  # b x 6 x n_heads x n_frames
        x = self.temp_pooling(x)  # b x 6 x n_heads
        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.fc1(x))  # b x n_heads x 16
        x = torch.sigmoid(self.fc2(x))  # b x n_heads x 1
        x = torch.squeeze(x, dim=-1)  # b x n_heads
        x = torch.mean(x, dim=-1)  # b
        x = torch.clamp(x, min=0.001, max=0.999)
        return x
