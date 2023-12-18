[![DOI](https://zenodo.org/badge/345181329.svg)](https://zenodo.org/badge/latestdoi/345181329)

# PhageHostLearn

This is the repository related to our manuscript in submission at Nature Microbiology:
"PhageHostLearn: actionable prediction of Klebsiella phage-host specificity at the subspecies level", authored by Boeckaerts D, Stock M, Ferriol-González C, Jesús O-I, Sanjuan R, Domingo-Calap P, De Baets B and Briers Y.

## Overview of the repository
- `code`: all the final code for the PhageHostLearn system, allowing researchers to train models, reproduce our analyses or make new predictions (see Quick start)
- `notebooks_exploration`: various exploratory analyses, for informative purposes
- `notebooks_models`: previous iterations of the PhageHostLearn system, for informative purposes.
- `notebooks_processing`: separate old notebooks for processing genome data, for informative purposes.

## Quick start
1. All the code to run the PhageHostLearn system is available in the `code` folder and presented as easy-to-follow IPython notebooks.
2. If you want to train a PhageHostLearn model from scratch using your own phage genome and bacterial genome data, go to the `phagehostlearn_training` notebook in the `code` folder and follow each steps from the beginning.
3. If you want to reproduce our analyses, first download the processed data from [this Zenodo repository](https://doi.org/10.5281/zenodo.8095914) and then follow the steps in the `phagehostlearn_training` notebook in the `code` folder (you will be able to skip step 2).
4. If you want to run the PhageHostLearn system we trained on our _Klebsiella_ interaction data to predict new interactions for _Klebsiella_ phage-host pairs, go to the `phagehostlearn_inference` notebook in the `code` folder and follow the steps. If you want to demo inference on the Klebsiella and phage genomes in our study, [the Zenodo repository](https://doi.org/10.5281/zenodo.8095914) also provides these as .zip files.

For typical datasets (up to hundreds of phages and/or hundreds of bacteria), no specialized GPU hardware is strictly needed (although it can speed things up). For comparison, our dataset of around 100 phages and 200 bacteria took 5-6 hours to process and make predictions for on an 8-core Apple M1.

## Installation requirements
This software has been developed in Python v3.9.7 on an Apple M1 Macbook Air. It requires the following software dependencies: PHANOTATE v1.5.0 (https://github.com/deprekate/PHANOTATE), PhageRBPdetection v2.1.3 (https://github.com/dimiboeckaerts/PhageRBPdetection), Kaptive v2.0.0 (https://github.com/klebgenomics/Kaptive), ESM-2 v1.0.3 (https://github.com/facebookresearch/esm), XGBoost v1.5.0 (https://github.com/dmlc/xgboost), Scikit-learn v0.24.2 (https://scikit-learn.org/stable/), biopython v1.79, joblib v1.1.0, json v4.2.1, matplotlib v3.4.3, numpy v1.20.3, pandas v1.3.4, pickle 0.7.5 and seaborn v0.11.2. Alls of these dependencies can be conveniently installed with `pip` (should only take minutes), apart from Kaptive and PhageRBPdetection, which can be downloaded from their GitHub repositories.
