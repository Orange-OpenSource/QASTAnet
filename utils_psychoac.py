"""
Software Name : QASTAnet
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
Authors: Adrien Llave, Grégory Pallone
Software description: An objective metric assessing the global audio quality of 3D audio signals (binaural or (higher-order) ambisonics)
"""

import numpy as np
import torch


def hilbert_transform(x):
    out = torch.zeros_like(x, dtype=torch.complex64)
    out.real = x
    transforms = -1j * torch.fft.rfft(x, axis=-1)
    transforms[..., 0] = 0
    out.imag = torch.fft.irfft(transforms, axis=-1)
    return out


def ERB(f):
    """
    compute the ERB bandwidth for a center frequency (Glasberg & Moore (1990))
    """
    return 24.7 * (4.37 * f / 1000 + 1)


def get_freq_centers_erb(fmin = 315,
                        fmax = 12500,
                        hitme = 500,
                        bw = 1,
                        ):

    freq = [fmin, fmax, hitme]

    audlimits = 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)

    audrangelow = audlimits[2] - audlimits[0]
    audrangehigh = audlimits[1] - audlimits[2]

    # Calculate number of points, excluding final point.
    nlow = int(audrangelow // bw)
    nhigh = int(audrangehigh // bw)

    audpoints = (np.arange(-nlow, nhigh + 1)) * bw + audlimits[2]

    freq = (1 / 0.00437) * np.sign(audpoints) * (np.exp(np.abs(audpoints) / 9.2645) - 1)
    return freq


def iso389_7(center_frequencies_hz):
    isoThr = np.array(
        [
            78.5,
            78.5,
            68.7,
            59.5,
            51.1,
            44.0,
            37.5,
            31.5,
            26.5,
            22.1,
            17.9,
            14.4,
            11.4,
            8.6,
            6.2,
            4.4,
            3.0,
            2.4,
            2.2,
            2.4,
            3.5,
            2.4,
            1.7,
            -1.3,
            -4.2,
            -5.8,
            -6.0,
            -5.4,
            -1.5,
            4.3,
            6.0,
            12.6,
            13.9,
            13.9,
            13.0,
            12.3,
            18.4,
            40.2,
            73.2,
            73.2,
        ]
    )

    isoFreq = np.array(
        [
            0.0,
            20.0,
            25.0,
            31.5,
            40.0,
            50.0,
            63.0,
            80.0,
            100.0,
            125.0,
            160.0,
            200.0,
            250.0,
            315.0,
            400.0,
            500.0,
            630.0,
            750.0,
            800.0,
            1000.0,
            1250.0,
            1500.0,
            1600.0,
            2000.0,
            2500.0,
            3000.0,
            3150.0,
            4000.0,
            5000.0,
            6000.0,
            6300.0,
            8000.0,
            9000.0,
            10000.0,
            11200.0,
            12500.0,
            14000.0,
            16000.0,
            18000.0,
            20000.0,
        ]
    )

    # Interpolate to find the ISO threshold
    hearing_threshold = np.interp(
        center_frequencies_hz, isoFreq, isoThr, left=None, right=None
    )

    return hearing_threshold
