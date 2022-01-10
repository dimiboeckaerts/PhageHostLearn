"""
KLEBSIELLA Jumbo phage analysis

Created on 10/01/22

@author: dimiboeckaerts
"""
# %% 0 - LIBRARIES & DIRECTORIES
# --------------------------------------------------
import os
import processing_utils as pu
import pandas as pd
import numpy as np
from Bio import SeqIO

cdpath = '/Users/dimi/cd-hit-v4.8.1-2019-0228'
klebsiella_dir = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/klebsiella_RBP_data'


# %%
# first cluster to 60%
input_file = klebsiella_dir+'/jumbo_sequences_full.fasta'
output_file = klebsiella_dir+'/jumbo_clusters_30%'
psipath = cdpath+'/psi-cd-hit'

# now cluster at 30%
cdout, cderr = pu.psi_cdhit_python(psipath, input_file, output_file, c=0.30)
print('done!')
# %%
