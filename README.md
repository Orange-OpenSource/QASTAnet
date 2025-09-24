# QASTAnet

Python implementation of [QASTAnet](https://arxiv.org/abs/2509.16715), an intrusive objective metric for predicting global audio quality of 3D audio signals (binaural, (higher-order) ambisonics).

## Setup
Install the required packages by running:
```
pip install -r requirements.txt
```
* Tested on:
    - Windows 10 with python 3.11.5
    - Debian 5.10 with python 3.10.13
    - MacOS Silicon 15.6.1 with python 3.10.15

## Usage
The program can be used using the command line tool:

```
python run_qastanet.py --ref /path/to/dir/reference_signal --deg /path/to/dir/degraded_signal [--checkpoint /path/to/dir/model_weights]
```

### Options
* `ref` - Reference audio file.
* `deg` - Degraded audio file.
* `checkpoint` - Model weights file path.

### Example
In [resources/audio](resources/audio), we provide a 3<sup>rd</sup> order ambisonics (ACN/SN3D [convention](https://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats)) audio example `ref.wav` of male speech located at 75° azimuth and 46° elevation with ideal plane wave spatialization. We provide the same signal degraded by the [IVAS](https://www.3gpp.org/ftp/Specs/archive/26\_series/26.258/26258-i20.zip) codec @32kbps in SBA mode, `deg.wav`.
```
python run_qastanet.py --ref resources/audio/ref.wav --deg resources/audio/deg.wav
```
For this sample, QASTAnet should predict a quality of 0.78.

We provide a second sample, `ref_rev.wav`, identical to the first one but spatialized by convoluting it with an SRIR from an Eigenmike 32 in a reverberant room.
```
python run_qastanet.py --ref resources/audio/ref_rev.wav --deg resources/audio/deg_rev.wav
```
For this sample, QASTAnet should predict a quality of 0.19.

## Citation
If you use this code, please cite both the repository and the associated paper:

Llave Adrien, Granier Emma, and Pallone Grégory, "*QASTAnet: A DNN-based Quality Metric for Spatial Audio*", 2025, [https://arxiv.org/abs/2509.16715](https://arxiv.org/abs/2509.16715)

## License
This project is licensed under the [MIT License](LICENSE.txt).
