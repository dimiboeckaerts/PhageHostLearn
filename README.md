[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11074747.svg)](https://doi.org/10.5281/zenodo.11074747)

# PhageHostLearn

This is the repository related to our manuscript published in [Nature Communications](https://www.nature.com/articles/s41467-024-48675-6):
"Prediction of _Klebsiella_ phage-host specificity at the strain level", authored by Boeckaerts D, Stock M, Ferriol-González C, Jesús O-I, Sanjuan R, Domingo-Calap P, De Baets B and Briers Y.

## Overview of the repository
- `code`: all the final code for the PhageHostLearn system, allowing researchers to train models, reproduce our analyses or make new predictions (see below)
- `analysis_notebooks`: folder including various subfolders related to certain analyses of the work, for informative purposes; _notebooks_exploration_ (various exploratory analyses), _notebooks_models_ (previous iterations of the PhageHostLearn system) and _notebooks_processing_ (separate old notebooks for processing genome data)

## Making predictions for your data
1. Clone or download this repository on your local computer.
2. Install all the necessary software to run PhageHostLearn, see below.
3. Navigate to the `phagehostlearn_inference.ipynb` notebook in the `code` folder.
4. Follow the steps outlined in the notebook, in particular: (1) make folders for phage genomes and bacterial genomes and store each genome separately as a FASTA file; (2) download our training phage/bacterial genomes depending on what predictions you want to make; (3) run the code cells in the notebook to process the genomes into phage RBPs and bacterial K-loci protein, which are then transformed into numerical representations as inputs to make predictions.

## Training a model from scratch or reproducing our analyses
1. Clone or download this repository on your local computer.
2. Install all the necessary software to run PhageHostLearn, see below.
3. If you want to reproduce our analyses, download the processed data from [this Zenodo repository](https://doi.org/10.5281/zenodo.11061100).
3. Navigate to the `phagehostlearn_training.ipynb` notebook in the `code` folder.
4. Follow the steps in the notebook to process your own data and train a model or reproduce our analyses.

For typical datasets (up to hundreds of phages and/or hundreds of bacteria), no specialized GPU hardware is strictly needed (although it can speed things up). For comparison, our dataset of around 100 phages and 200 bacteria took 5-6 hours to process and train a model for on an 8-core Apple M1.

## Installation requirements
This software has been developed in Python v3.9.7 on an Apple M1 Macbook Air. It requires the following software dependencies: PHANOTATE v1.5.0 (https://github.com/deprekate/PHANOTATE), PhageRBPdetection v2.1.3 (https://github.com/dimiboeckaerts/PhageRBPdetection), Kaptive v2.0.0 (https://github.com/klebgenomics/Kaptive), ESM-2 v1.0.3 (https://github.com/facebookresearch/esm), XGBoost v1.5.0 (https://github.com/dmlc/xgboost), Scikit-learn v0.24.2 (https://scikit-learn.org/stable/), biopython v1.79, joblib v1.1.0, json v4.2.1, matplotlib v3.4.3, numpy v1.20.3, pandas v1.3.4, pickle 0.7.5 and seaborn v0.11.2. Alls of these dependencies can be conveniently installed with `pip` (should only take minutes), apart from Kaptive and PhageRBPdetection, which were downloaded from their GitHub repositories and are incorporated in this repository. Kaptive requires BLAST+ to be installed on the command line, see the [NCBI website](http://www.ncbi.nlm.nih.gov/books/NBK279690/) for installation.
