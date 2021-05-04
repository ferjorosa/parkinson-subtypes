# parkinson-multidimensional-clustering
[![Build Status](https://travis-ci.com/ferjorosa/incremental-latent-forests.png?branch=master)](https://travis-ci.com/ferjorosa/parkinson-multidimensional-clustering) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the code repository of the paper **Identifying Parkinson's disease subtypes via multidimensional clustering** (Movement Disorders) [1]. It includes the following: 

* A Java implementation of the *greedy latent structure learner* (GLSL) algorithm [1] (see Appendix B).
* A copy of the Java implementation of the *incremental learner* (IL) and the *constrained incremental learner* (CIL) algorithms [2] used as initialization for GLSL. [Original Github repository](https://github.com/ferjorosa/incremental-latent-forests).
* A copy of the Java implementation of the *Gaussian expansion, adjustment, and simplification until termination* (GEAST) algorithm [3]. [Original Github repository](https://github.com/kmpoon/pltm-east).
* A copy of the Java implementation of the *Gaussian mixture model* [4]. [Original Github repository](https://github.com/kmpoon/pltm-east).

In addition, this repository provides the scripts, data and instructions necessary to reproduce the results included in the article. 

## Project organization
This project is organized in several folders:

* **documentation**. It contains information about the MDS-NMS and the MDS-UPDRS, as well as a Codebook with the meaning of the variables considered in the study.
* **data**. It contains the original data and the transformed data necessary for executing the experiments.
* **src**. Main repository of source code. It contains the Java implementations as well as the code necessary for executing the experiments.
* **python-project**. Secondary repository of source code. It contains the Python source code necessary for transforming the original data as well as for executing the article's comparative cluster analysis.
* **results**. It contains the results of the experiments (i.e., models in AMIDST *.bn format, models in GENIE *.xdsl format, JSON files with scores and times, and completed data for those algorithms that could work with missing data).
* **best-model-genie**. For ease of use, it contains the best model in GENIE format. This model is necessary for executing the article's probabilistic inference analysis.

## Usage
* [Project installation](https://github.com/ferjorosa/parkinson-multidimensional-clustering/wiki)
* [Learning clustering models](https://github.com/ferjorosa/parkinson-multidimensional-clustering/wiki/Learn-clustering-models)
* [Analysis of the results using the tool Genie](https://github.com/ferjorosa/parkinson-multidimensional-clustering/wiki/Analysis-with-Genie)

## Disclaimer
The data provided by this repository should not be used for independent publications without the approval of the Movement disorders society. 

## Contact
* For any enquiries about the project, please email [ferjorosa@gmail.com](mailto:ferjorosa@gmail.com).
* For any enquieres about the data, please email [crodb@isciii.es](mailto:crodb@isciii.es).

## References

* [1] [Rodriguez-Sanchez F., Rodriguez-Blazquez C., Bielza C., Larrañaga P., D. Weintraub, A. Schrag, A. Rizos, P. Martinez-Martin and K. Ray Chaudhuri. Identifying Parkinson's disease subtypes via multidimensional clustering. Movement Disorders. In review.]()
* [2] [Rodriguez-Sanchez F., Larrañaga P., Bielza C. Incremental learning of latent forests. IEEE Access. 2020;8:224420–224432.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9207730)
* [3] [Poon, L. K., Zhang, N. L., Liu, T., & Liu, A. H. (2013). Model-based clustering of high-dimensional data: Variable selection versus facet determination. International Journal of Approximate Reasoning, 54(1), 196-215.](https://www.sciencedirect.com/science/article/pii/S0888613X12001429/pdf?md5=6dd8ae15f83255027dddb8d4e160f5c7&pid=1-s2.0-S0888613X12001429-main.pdf).
