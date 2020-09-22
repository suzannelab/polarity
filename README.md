# Polarity

![Sample output from the model](data/header.svg)

[![DOI](https://zenodo.org/badge/185421680.svg)](https://zenodo.org/badge/latestdoi/185421680) [**Need to be updated**]


This is a leg fold formation simulation package for the article:

### Mechanical control of morphogenesis robustness in an inherently challenging environment

Emmanuel Martin, Sophie Theis, Guillaume Gay, Bruno Monier, and Magali Suzanne https://www.nature.com/articles/s41467-019-10720-0 [**Need to be updated**]



## Try it with my binder by clicking the badge bellow:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/suzannelab/invagination/master?filepath=notebooks%2FIndex.ipynb) [**Need to be updated**]


## Dependencies

- python > 3.6
- tyssue >= 0.7.0


## Installation

This package is based on the [`tyssue`](https://tyssue.readthedocs.org) library and its dependencies.

The recommanded installation route is to use the `conda` package manager. You can get a `conda` distribution for your OS at https://www.anaconda.com/download . Make sure to choose a python 3.6 version. Once you have installed conda, you can install tyssue with:

```bash
$ conda install -c conda-forge tyssue
```

You can then download and install polarity from github:

- with git:

```bash
$ git clone https://github.com/suzannelab/polarity.git
$ cd polarity
$ python setup.py install
```

- or by downloading https://github.com/suzannelab/polarity/archive/master.zip ,  uncompressing the archive and running `python setup.py install` in the root directory.

## Licence

This work is free software, published under the MPLv2 licence, see LICENCE for details.


&copy; The article authors -- all rights reserved