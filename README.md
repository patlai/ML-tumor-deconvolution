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
To generate new data:
```
python3 scripts/data-generation.py [patient data file] [cell signature file] [gene mapping file] [output path]
```

## Authors
Patrick Lai, Yu-Yueh Liu, Keven Liu, Ben Landry

See also the list of [contributors](https://github.com/patlai/ML-tumor-deconvolution/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

