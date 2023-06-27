[![DOI](https://zenodo.org/badge/345181329.svg)](https://zenodo.org/badge/latestdoi/345181329)

# PhageHostLearn
Pairwise machine learning models for phage-host interaction prediction

## General information

This is the repository related to our manuscript in submission at Nature Microbiology:
"PhageHostLearn: actionable prediction of Klebsiella phage-host specificity at the subspecies level", authored by Boeckaerts D, Stock M, Ferriol-González C, Jesús O-I, Sanjuan R, Domingo-Calap P, De Baets B and Briers Y.

## Quick start
1. All the code to run the PhageHostLearn system is available in the `code` folder and presented as easy-to-follow IPython notebooks.
2. If you want to train a PhageHostLearn model from scratch using your own phage genome and bacterial genome data, go to the `phagehostlearn_training` notebook in the `code` folder and follow each steps from the beginning.
3. If you want to reproduce our analyses, first download the processed data from [this Zenodo repository](https://doi.org/10.5281/zenodo.8052911) and then follow the steps in the `phagehostlearn_training` notebook in the `code` folder (you will be able to skip step 2).
4. If you want to run the PhageHostLearn system we trained on our _Klebsiella_ interaction data to predict new interactions for _Klebsiella_ phage-host pairs, go to the `phagehostlearn_inference` notebook in the `code` folder and follow the steps.

## Overview of the repository
- `code`: all the final code for the PhageHostLearn system, allowing researchers to train models, reproduce our analyses or make new predictions (see Quick start)
- `notebooks_exploration`: various exploratory analyses, for informative purposes
- `notebooks_models`: previous iterations of the PhageHostLearn system, for informative purposes.
- `notebooks_processing`: separate old notebooks for processing genome data, for informative purposes.
