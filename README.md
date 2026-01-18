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

We provide a second sample, `ref_rev.wav`, identical to the first one but spatialized by convolving it with an SRIR from an Eigenmike 32 in a reverberant room.
```
python run_qastanet.py --ref resources/audio/ref_rev.wav --deg resources/audio/deg_rev.wav
```
For this sample, QASTAnet should predict a quality of 0.19.

## Citation
If you use this code, please cite both the repository and the associated paper:

Llave Adrien, Granier Emma, and Pallone Grégory (2026). "[*QASTAnet: A DNN-based Quality Metric for Spatial Audio*](https://arxiv.org/abs/2509.16715)". In : ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE.

## License
This project is licensed under the [MIT License](LICENSE.txt).

## Description of the stimuli used for training, validation and test

The test dataset is composed of the audio samples with a name ending by "_1".
The validation dataset is composed of the audio samples with a name ending by "_2".
The training dataset is composed of the audio samples with a name ending by "_3" to "_6".

| Name | Content | Spatialization | Source |
|-----|----------|----------------|--------|
| AMB_1 | Party | EM64 recording | Clarity Challenge |
| AMB_2 | Park  | EM64 recording | Clarity Challenge |
| AMB_3 | Park  | ZM-1 recording | Internal |
| AMB_4 | Stadium | EM32 recording | Internal |
| AMB_5 | Stadium | EM32 recording | Internal |
| AMB_6 | Outdoor crowd | EM32 recording | Internal |
| APP_1 | Applause in concert hall 1 | EM32 recording | Internal |
| APP_2 | Applause in concert hall 2 | EM32 recording | Internal |
| APP_3 | Applause in church concert | EM32 recording | Internal |
| APP_4 | Applause in concert hall 3 | EM32 recording | Internal |
| APP_5 | Applause in concert hall 4 | EM32 recording | Internal |
| APP_6 | Applause in large room | EM32 recording | Internal |
| ORC_1 | Symphonic orchestra in concert hall 1 | EM32 recording | Internal |
| ORC_2 | Symphonic orchestra in concert hall 1 | EM32 recording | Internal |
| ORC_3 | Symphonic orchestra in concert hall 2 | EM32 recording | Internal |
| ORC_4 | Symphonic orchestra in concert hall 3 | EM32 recording | Internal |
| ORC_5 | Symphonic orchestra in concert hall 2 | EM32 recording | Internal |
| ORC_6 | Symphonic orchestra in concert hall 2 | EM32 recording | Internal |
| BND_1 | Big band in concert hall | EM32 recording | Internal |
| BND_2 | Vocal quatuor | EM32 recording | Internal |
| BND_3 | Jazz band | EM32 recording | Internal |
| BND_4 | Pop band | EM32 recording | Internal |
| BND_5 | Funk band | EM32 recording | Internal |
| BND_6 | Funk band | EM32 recording | Internal |
| SPK1_ANE_1 | 1 male speaker | Plane wave encoding at az=75° and el=26° | Speech from DNS2022 |
| SPK1_ANE_2 | 1 female speaker | Plane wave encoding at az=-93° and el=19° | Speech from DNS2022 |
| SPK1_ANE_3 | 1 female speaker | Plane wave encoding at az=-127° and el=-2° | Speech from DNS2022 |
| SPK1_ANE_4 | 1 male speaker | Plane wave encoding at az=-93° and el=24° | Speech from DNS2022 |
| SPK1_ANE_5 | 1 male speaker | Plane wave encoding at az=88° and el=-11° | Speech from DNS2022 |
| SPK1_ANE_6 | 1 female speaker | Plane wave encoding at az=24° and el=65° | Speech from DNS2022 |
| SPK1_REV_1 | 1 male speaker | EM32 SRIR convolution at az=75° and el=26° | Speech from DNS2022, internal SRIR |
| SPK1_REV_2 | 1 female speaker | EM32 SRIR convolution at az=-93° and el=19° | Speech from DNS2022, internal SRIR |
| SPK1_REV_3 | 1 female speaker | EM32 SRIR convolution at az=-127° and el=-2° | Speech from DNS2022, internal SRIR |
| SPK1_REV_4 | 1 male speaker | EM32 SRIR convolution at az=-93° and el=24° | Speech from DNS2022, internal SRIR |
| SPK1_REV_5 | 1 male speaker | EM32 SRIR convolution at az=88° and el=-11° | Speech from DNS2022, internal SRIR |
| SPK1_REV_6 | 1 female speaker | EM32 SRIR convolution at az=24° and el=65° | Speech from DNS2022, internal SRIR |
| SPK3_ANE_1 | 3 M/F speakers | Plane wave encoding in the upper hemisphere | Speech from DNS2022 |
| SPK3_ANE_2 | 3 M/F speakers | Plane wave encoding in the upper hemisphere | Speech from DNS2022 |
| SPK3_ANE_3 | 3 M/F speakers | Plane wave encoding in the upper hemisphere | Speech from DNS2022 |
| SPK3_ANE_4 | 3 M/F speakers | Plane wave encoding in the upper hemisphere | Speech from DNS2022 |
| SPK3_ANE_5 | 3 M/F speakers | Plane wave encoding in the upper hemisphere | Speech from DNS2022 |
| SPK3_ANE_6 | 3 M/F speakers | Plane wave encoding in the upper hemisphere | Speech from DNS2022 |
| SPK3_REV_1 | 3 M/F speakers | EM32 SRIR convolution in the upper hemisphere | Speech from DNS2022, internal SRIR |
| SPK3_REV_2 | 3 M/F speakers | EM32 SRIR convolution in the upper hemisphere | Speech from DNS2022, internal SRIR |
| SPK3_REV_3 | 3 M/F speakers | EM32 SRIR convolution in the upper hemisphere | Speech from DNS2022, internal SRIR |
| SPK3_REV_4 | 3 M/F speakers | EM32 SRIR convolution in the upper hemisphere | Speech from DNS2022, internal SRIR |
| SPK3_REV_5 | 3 M/F speakers | EM32 SRIR convolution in the upper hemisphere | Speech from DNS2022, internal SRIR |
| SPK3_REV_6 | 3 M/F speakers | EM32 SRIR convolution in the upper hemisphere | Speech from DNS2022, internal SRIR |
| THE_1 | Drama in large room 1 | EM32 recording | Internal |
| THE_2 | Drama in large room 1 | EM32 recording | Internal |
| THE_3 | Drama in large room 1 | EM32 recording | Internal |
| THE_4 | Drama in large room 1 | EM32 recording | Internal |
| THE_5 | Drama in large room 2 | EM32 recording | Internal |
| THE_6 | Drama in large room 2 | EM32 recording | Internal |
| MTG_1 | Meeting in office | EM32 recording | Internal |
| MTG_2 | Meeting in office | EM32 recording | Internal |
| MTG_3 | Meeting in office | EM32 recording | Internal |
| MTG_4 | Meeting in office | EM32 recording | Internal |
| MTG_5 | Meeting in office | EM32 recording | Internal |
| MTG_6 | Meeting in office | EM32 recording | Internal |
| POP_1 | Vocal + cello | Synthetic mixing | Internal |
| POP_2 | Vocal + cello | Synthetic mixing | Internal |
| POP_3 | Vocal + cello | Synthetic mixing | Internal |
| POP_4 | Vocal + cello | Synthetic mixing | Internal |
| POP_5 | Vocal + cello | Synthetic mixing | Internal |
| POP_6 | Vocal + cello | Synthetic mixing | Internal |
| FLK_ANE_1 | Violin + banjo | Synthetic mixing in the horizontal plane at az=+/-45° | Internal |
| FLK_ANE_2 | Violin + banjo | Synthetic mixing in the horizontal plane at az=+/-45° | Internal |
| FLK_ANE_3 | Violin + banjo | Synthetic mixing in the horizontal plane at az=+/-45° | Internal |
| FLK_ANE_4 | Violin + banjo | Synthetic mixing in the horizontal plane at az=+/-45° | Internal |
| FLK_ANE_5 | Violin + banjo | Synthetic mixing in the horizontal plane at az=+/-45° | Internal |
| FLK_ANE_6 | Violin + banjo | Synthetic mixing in the horizontal plane at az=+/-45° | Internal |
| FLK_REV_1 | Violin + banjo | ZM-1 recording | Internal |
| FLK_REV_2 | Violin + banjo | ZM-1 recording | Internal |
| FLK_REV_3 | Violin + banjo | ZM-1 recording | Internal |
| FLK_REV_4 | Violin + banjo | ZM-1 recording | Internal |
| FLK_REV_5 | Violin + banjo | ZM-1 recording | Internal |
| FLK_REV_6 | Violin + banjo | ZM-1 recording | Internal |




