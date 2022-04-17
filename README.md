# Band Selection for an Online Sorting System for Tobacco

## Data preprocessing

We conduct a series of pre-processing steps including the black-white calibration(done by C program when data was collected), bandwised signal-noize rate thresholding, data smoothing, Multiplicative Scattering Correction(MSC) and Standard Normal correction (SNC).

Detailed preprocessing steps can be found in this [file](./01_data_preprocess.ipynb).

The data preprocessed and raw data were saved under the folder dataset. The whole folder can be downloaded from both BaiduNetDisk or GoogleDrive:

[百度网盘链接 ☞ ](https://pan.baidu.com/s/1l8d3qqpf7pV1mAkyff5HlA) 提取码: fphm, [GoogleDriveLink](https://drive.google.com/drive/folders/1iU4NhLJjOcQHfNosY72MnKl7GyueRt1g?usp=sharing)

## Band Selection

In practical industry environment, the data transferring speed cannot maintain an extremely high standard for long time when it comes to the hyperspectral domain.
Unlike the commonly used line scanned RGB camera whose image frame is a line contain 3 line of value corresponding to the red, blue and green reflection of object, the line scanned hyperspectral camera generate is able to generate hundreds lines of value in each image frame which reval the chemical characteristic of the object, however, bring burden to not only the Ether Net but also the complaint devices to cope with.
In order to avoid this situation, the band selection based on the requirement of the hyperspectral camera is essential.

The detailed information about our band selection method can be found in this [file](./02_band_selection.ipynb).

### Encoding Method

Conventional encoding cannot suit the newly designed low cost hyperspectral camera, which only can capture spectral information with serval bands window.
So we redesigned the encoding method.

### Objective Function

variable weighted

## Introducing Prior Knowledge with Magnet

## Comparing Experiments
