# Tumor deconvolution with machine learning
### McGill ECSE design project 1 and 2: ECSE 456, ECSE 457
This project's goal is to analyze the bulk gene expression profile of tumor data using machine learning algorithms.

## Getting Started
### Prerequisites

* python 3
* pip

Then install the required packages from the requirements file
```
pip3 install -r requirements.txt
```
or
```
python3 -m pip install -r requirements.txt
```

## Running
The current running configuration is to generate data using randomized multivariate normal distributions with multiple with covariance matrix scaling factors from 0 to 1 in increments of 0.1 and run NMF and CLS for deconvolution on each generated instance. To do this, use:
```
python3 scripts/data-generation.py \
[patient data file] \
[cell signature file] \
[gene mapping file] \
[output path] \
```
make sure the output path is a directory that currently exists and also contains a subfolder called `plots`

The output will be the estimated cell fractions for each scaling factor and each method (total of 20 files) as a matrix in .csv format. Additionally, the true weights will also be provided.

## Authors
Patrick Lai, Yu-Yueh Liu, Keven Liu, Ben Landry

See also the list of [contributors](https://github.com/patlai/ML-tumor-deconvolution/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

