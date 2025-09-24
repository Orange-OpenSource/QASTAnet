"""
Software Name : QASTAnet
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
Authors: Adrien Llave, Grégory Pallone
Software description: An objective metric assessing the global audio quality of 3D audio signals (binaural or (higher-order) ambisonics)


Description:    Example script of the usage of QASTAnet metric.
                This script achieves the following steps:
                - loading of the reference and degraded demo audio samples (speech located at 75° azimuth and 46° elevation, ideal plane wave spatialization, degraded with IVAS @32kbps)
                - defining an instance of the QASTAnet metric class
                - recalling the metric weights from a file
                - running the prediction of the model over the reference and degraded signals

"""

import torch
import soundfile as sf
from qastanet import QASTAnet
import time
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ref",
    type=str,
    default="resources/audio/ref.wav",
    help="Reference audio file path",
)
parser.add_argument(
    "--deg",
    type=str,
    default="resources/audio/deg.wav",
    help="Degraded audio file path",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="resources/metric_config/model_weights.pth.tar",
    help="Model weights file path",
)
parser.add_argument(
    "--config",
    type=str,
    default="resources/metric_config/model_args.yml",
    help="Model configuration file path",
)

parser.add_argument(
    "--verbose",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Verbose mode add some logs about processing time.",
)

args = parser.parse_args()

# -- Paths
checkpoint_fn = args.checkpoint
ref_fn = args.ref
deg_fn = args.deg

# -- Load audio samples
ref, fs = sf.read(ref_fn)
deg, fs = sf.read(deg_fn)
ref = torch.tensor(ref.T, dtype=torch.float32).unsqueeze(0)
deg = torch.tensor(deg.T, dtype=torch.float32).unsqueeze(0)

# -- Define metric and load model weights
with open(args.config, "r") as stream:
    config_model_d = yaml.safe_load(stream)
metric = QASTAnet(**config_model_d, checkpoint_fn=checkpoint_fn, VERBOSE=args.verbose)

# -- Run percetual quality score estimation
st = time.time()
score = metric(deg, ref)
print(
    f"Score estimated in {(time.time() - st):.2f} seconds for an audio example of {(ref.shape[-1] / fs):.2f} seconds."
)
print(
    f"QASTAnet predicts a quality of {score.detach()[0]:.2f} (scaled between 0 and 1)."
)
