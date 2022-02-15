"""
PhageHostPredict (klebsiella) DATA PROCESSING

Created on 25/11/21

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
data_dir = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA'
klebsiella_dir = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/klebsiella_RBP_data'
valencia_dir = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/Valencia_data'
project_dir = '/Users/dimi/Documents/GitHub_Local/PhageHostLearning'

# %% 1 - PROCESSING RBP DATA
# --------------------------------------------------
# Klebsiella RBP subset
pu.RBPbase_species_filter('RBPbase_011221.csv', data_dir, klebsiella_dir, 'klebsiella_pneumoniae')

# %% Adding known Klebsiella RBPs
klebsiella_RBPs = klebsiella_dir+'/sequences/klebsiella_RBPs.fasta'
RBPbase_klebsiella = pd.read_csv(klebsiella_dir+'/RBPbase_011221_klebsiella_pneumoniae.csv')
combined_fasta = open(klebsiella_dir+'/all_klebsiella_RBPs_for_clustering.fasta', 'w')

for record in SeqIO.parse(klebsiella_RBPs, 'fasta'):
    sequence = str(record.seq)
    rec_id = record.id
    combined_fasta.write('>identified_'+rec_id+'\n'+sequence+'\n')
for i, protein in enumerate(RBPbase_klebsiella['protein_seq']):
    unique_id = RBPbase_klebsiella['unique_ID'][i]
    combined_fasta.write('>'+unique_id+'\n'+protein+'\n')
combined_fasta.close()

# %% cluster with CD-HIT
input_file = klebsiella_dir+'/all_klebsiella_RBPs_for_clustering.fasta'
output_file = klebsiella_dir+'/all_klebsiella_RBPS_clusters'
cdout, cderr = pu.cdhit_python(cdpath, input_file, output_file)

# %% 1) loop over clusters, identify representatives without identified in cluster
output_file = klebsiella_dir+'/all_klebsiella_RBPS_clusters'
clusters = open(output_file+'.clstr.txt')
cluster_iter = 0
white_check = 0
cluster_accessions = []
white_list = []
representatives_to_check = []
for line in clusters.readlines():
    # new cluster
    if line[0] == '>':
        # finish old cluster if not first one
        if (cluster_iter > 0) and (len(cluster_accessions) >= 1):
            if white_check == 1: # an identified RBP in cluster
                white_list = white_list+cluster_accessions
            else: # if not, save representative to predict structure
                representatives_to_check.append(representative)
        # initiate new cluster
        cluster_accessions = []
        cluster_iter += 1
        white_check = 0
        
    # in a cluster
    else:
        acc = line.split('>')[1].split('...')[0]
        cluster_accessions.append(acc)
        if 'identified' in acc:
            white_check = 1
        if '*' in line:
            representative = acc
        
# finish last cluster
if len(cluster_accessions) >= 1:
    if white_check == 1: # an identified RBP in cluster
        white_list = white_list+cluster_accessions
    else: # if not, save representative to predict structure
        representatives_to_check.append(representative)

# %% make separate fasta files of those
os.mkdir(klebsiella_dir+'/representatives_fasta_files')
for record in SeqIO.parse(output_file+'.fasta', 'fasta'):
    identifier = record.id
    sequence = str(record.seq)
    if identifier in representatives_to_check:
        # make new fasta file in directory for ColabFold
        temp_fasta = open(klebsiella_dir+'/representatives_fasta_files/'+identifier+'.fasta', 'w')
        temp_fasta.write('>'+identifier+'\n'+sequence+'\n')
        temp_fasta.close()

# 2) make 3D structure predictions with ColabFold & manual check
# done in the cloud.

# %% 3) make list RBPs_with_Bhelix -> all identified + manuals checked
RBPs_with_Bhelix = [x for x in white_list if 'identified' in x]
manual_checks = ['7klebsiella_pneumoniae_RBP0', '9klebsiella_pneumoniae_RBP1', '19klebsiella_pneumoniae_RBP0', '170klebsiella_pneumoniae_RBP0', 
                    '213klebsiella_pneumoniae_RBP0', '253klebsiella_pneumoniae_RBP1', '339klebsiella_pneumoniae_RBP0', '807klebsiella_pneumoniae_RBP0']
RBPs_with_Bhelix = RBPs_with_Bhelix + manual_checks

# 4) loop over clusters: if one in RBPs_with_Bhelix -> white list entire cluster
clusters = open(output_file+'.clstr.txt')
cluster_iter = 0
white_check = 0
cluster_accessions = []
white_list = []
for line in clusters.readlines():
    # new cluster
    if line[0] == '>':
        # finish old cluster if not first one
        if (cluster_iter > 0) and (len(cluster_accessions) >= 1):
            if white_check == 1: # an identified RBP in cluster
                white_list = white_list+cluster_accessions
        # initiate new cluster
        cluster_accessions = []
        cluster_iter += 1
        white_check = 0
    # in a cluster
    else:
        acc = line.split('>')[1].split('...')[0]
        cluster_accessions.append(acc)
        if acc in RBPs_with_Bhelix:
            white_check = 1
        
# finish last cluster
if len(cluster_accessions) >= 1:
    if white_check == 1: # an identified RBP in cluster
        white_list = white_list+cluster_accessions

# %% Load files for interaction matrix
RBPbase_klebsiella = pd.read_csv(klebsiella_dir+'/RBPbase_031221_klebsiella_pneumoniae.csv')
genomes_klebsiella = pd.read_csv(klebsiella_dir+'/klebsiella_genomes_031221.csv')
valencia_interactions = pd.read_csv(valencia_dir+'/interactionsValencia.csv', index_col=0)

# %% Construct interaction matrix
row_names = genomes_klebsiella['accession']
column_names = RBPbase_klebsiella['unique_ID']
klebsiella_interactions = np.empty((genomes_klebsiella.shape[0], RBPbase_klebsiella.shape[0]))
klebsiella_interactions[:] = np.nan

for i, row in enumerate(row_names):
    for j, column in enumerate(column_names):
        # known prophage interactions
        if RBPbase_klebsiella['host_accession'][j] == row:
            klebsiella_interactions[i,j] = 1

        # known Valencia interactions
        if (row in valencia_interactions.index) and (column in valencia_interactions.columns):
            klebsiella_interactions[i,j] = int(valencia_interactions[column][row])

# %% adjust the interaction matrix & save
# RUN AFTER COMPLETING THE WHITE LIST (steps 3 & 4)!
for j, column in enumerate(column_names):
    if column not in white_list:
        klebsiella_interactions[:,j] = np.nan

IM_klebsiella = pd.DataFrame(klebsiella_interactions, index=row_names, columns=column_names)
IM_klebsiella.to_csv(klebsiella_dir+'/interactions_klebsiella.csv')


# %% 2 - PROCESSING BACTERIAL DATA
# --------------------------------------------------
# compute Kaptive and get proteins?! + serotype
# TAKES ABOUT 12 HOURS
database = 'Klebsiella_k_locus_primary_reference.gbk'
genomes = pd.read_csv(klebsiella_dir+'/klebsiella_genomes_031221.csv')
names, serotypes = pu.compute_loci(genomes, project_dir, klebsiella_dir, database) # results in the same data_dir
pd.DataFrame(serotypes, columns=['sero']).to_csv(klebsiella_dir+'/kaptive_serotypes_klebsiella_genomes_031221.csv', index=False)


# %% 3 - Compute similarity matrices and delete identical interactions
# --------------------------------------------------
# similarity matrix RBP sequences
"""
cmd line: julia pairwise_alignment.jl "/Users/Dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/Klebsiella_RBP_data/RBPbase_250621_klebsiella_pneumoniae.fasta" "protein"
takes about 30 mins
"""
rbp_names = pu.RBPbase_fasta_processing('RBPbase_031221_klebsiella_pneumoniae.csv', klebsiella_dir)
pu.pairwise_alignment_julia(klebsiella_dir+'/RBPbase_031221_klebsiella_pneumoniae.fasta', 'protein', project_dir)

# %% similarity matrix kaptive loci
genomes = pd.read_csv(klebsiella_dir+'/klebsiella_genomes_031221.csv')
kaptive_file_names = pu.get_kaptive_file_names(genomes)
kaptive_fasta_all = pu.kaptive_fasta_processing(kaptive_file_names, klebsiella_dir)
kaptive_out = klebsiella_dir+'/kaptive_loci'
pu.cdhit_est_python(cdpath, klebsiella_dir, kaptive_file_names, kaptive_fasta_all, kaptive_out)

# %% adjust interaction matrix
loci_sim = np.loadtxt(klebsiella_dir+'/kaptive_loci_score_matrix.txt')
rbp_sim = np.loadtxt(klebsiella_dir+'/RBPbase_031221_klebsiella_pneumoniae.fasta_score_matrix.txt')
IM_klebsiella = pd.read_csv(klebsiella_dir+'/interactions_klebsiella.csv', index_col=0)
klebsiella_interactions = np.asarray(IM_klebsiella)

# delete identical interactions
for i in range(rbp_sim.shape[0]-1):
    for j in range(i, rbp_sim.shape[0]):
        if (rbp_sim[i,j] == 1) and (i != j):
            # get the corresponding loci
            locus_i = list(np.where(klebsiella_interactions[:, i] == 1)[0])
            locus_j = list(np.where(klebsiella_interactions[:, j] == 1)[0])
            
            for li in locus_i:
                for lj in locus_j:
                    if loci_sim[li, lj] == 1: # if identical delete one of both
                        klebsiella_interactions[lj,j] = np.nan
print(sum(sum(klebsiella_interactions == 1)))
IM_klebsiella = pd.DataFrame(klebsiella_interactions, index=row_names, columns=column_names)
IM_klebsiella.to_csv(klebsiella_dir+'/interactions_klebsiella.csv')


# %% 4 - MONO RBP FILTERING
# --------------------------------------------------
"""
Here, we adjust the interaction matrix again to only contain phages that have a single RBP 
(if any multi-RBP phages left after previous filtering). Again, this only changes the interaction matrix,
it's this one that will be used to construct a frame for training later.
"""
IM_klebsiella = pd.read_csv(klebsiella_dir+'/interactions_klebsiella.csv', index_col=0)
klebsiella_interactions = np.asarray(IM_klebsiella)
multi_RBP_phages = [name[:-5] for name in IM_klebsiella.columns if 'RBP1' in name]
row_names = IM_klebsiella.index
column_names = IM_klebsiella.columns
for j, column in enumerate(IM_klebsiella.columns):
    if column[:-5] in multi_RBP_phages:
        for i in range(IM_klebsiella.shape[0]):
            klebsiella_interactions[i,j] = np.nan
IM_klebsiella = pd.DataFrame(klebsiella_interactions, index=row_names, columns=column_names)
IM_klebsiella.to_csv(klebsiella_dir+'/interactions_klebsiella_mono.csv')

# %%
