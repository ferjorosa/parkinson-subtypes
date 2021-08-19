# parkinson-subtypes
[![Build Status](https://travis-ci.com/ferjorosa/incremental-latent-forests.png?branch=master)](https://travis-ci.com/ferjorosa/parkinson-multidimensional-clustering) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the code repository of the paper **Identifying Parkinson's disease subtypes with motor and non-motor symptoms via model-based multi-partition clustering**. It includes the scripts, data and instructions necessary to reproduce the results included in the article. 

## Project organization
This project is organized in several folders:

* **documentation**. It contains information about the MDS-NMS and the MDS-UPDRS, as well as a Codebook with the meaning of the variables considered in the study.
* **data**. It contains the original data and the transformed data necessary for learning the clustering models.
* **src**. Main repository of source code. It contains the Java implementations as well as the code necessary for learning the clustering models.
* **python-project**. Secondary repository of source code. It contains the Python source code necessary for transforming the original data as well as for executing the article's comparative cluster analysis.
* **results**. It contains the results of the learning scripts (i.e., models in AMIDST *.bn format, models in GENIE *.xdsl format, JSON files with scores and times, and completed data for those algorithms that could work with missing data).
* **best-model-genie**. For ease of use, it contains the best model in GENIE format. This model is necessary for executing the article's probabilistic inference analysis.

## Usage
* [Project installation](https://github.com/ferjorosa/parkinson-subtypes/wiki)
* [Learning clustering models](https://github.com/ferjorosa/parkinson-subtypes/wiki/Learn-clustering-models)
* [Analysis of the results using the tool Genie](https://github.com/ferjorosa/parkinson-subtypes/wiki/Analysis-with-Genie)

## Disclaimer
The data provided by this repository should not be used for independent publications without the approval of the Movement disorders society. 

## Contact
* For any enquiries about the project, please email [ferjorosa@gmail.com](mailto:ferjorosa@gmail.com).
* For any enquieres about the data, please email [crodb@isciii.es](mailto:crodb@isciii.es).
